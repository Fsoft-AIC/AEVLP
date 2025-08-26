from typing_extensions import final
import torch
from torch._C import ThroughputBenchmark
import torch.nn.functional as F
import math 
from utils import temperature_scaled_softmax


def GPR_loss(logits, pseudo_labels, label_vec,K,V,Q,epoch,P=None):
    preds = torch.sigmoid(logits)
    q2,q3=Q
    loss_mtx = torch.zeros_like(preds)
    final_labels = torch.where(label_vec == 0, pseudo_labels, label_vec)
    # preds_detached = preds.clone().detach()
    # V[label_vec == 1] = 0.1 * (preds_detached[label_vec == 1] ** 2) - 2 * 0.1 *(preds_detached[label_vec == 1]) + 0.1 + 1
    
    loss_mtx[label_vec == 1 ] = neg_log(preds[label_vec == 1])
    # loss_mtx[label_vec == 1] = ((1 - preds[label_vec == 1]) ** 2) * neg_log(preds[label_vec == 1])
    loss_mtx[final_labels == 0] = V[final_labels == 0]*(K[final_labels == 0]*loss1(preds[final_labels == 0],q2)+(1-K[final_labels == 0])*loss2(preds[final_labels == 0],q3)) 

    if P['use_pl']:
        # Positive pseudo-label 
        mask_pos = (label_vec  == 0) & (pseudo_labels == 1)

        # Narrow the range of weights of positive pseudo-label into range lambda 1 and lambda 2
        V[mask_pos] = torch.where(V[mask_pos] > 1 - P['lam_1'], V[mask_pos], 1 - P['lam_1'])
        V[mask_pos] = torch.where(V[mask_pos] < 1 - P['lam_2'], V[mask_pos], 1 - P['lam_2'])
        
        loss_mtx[mask_pos] = (1 - V[mask_pos]) * (P['rho']* neg_log(preds[mask_pos]) + (1 - P['rho'])* neg_log(1-preds[mask_pos]))
        # loss_mtx[mask_pos] = 0.7 * (P['rho']* neg_log(preds[mask_pos]) + (1 - P['rho'])* neg_log(1-preds[mask_pos]))
        # Negative pseudo-label 
        loss_mtx[final_labels == -1] = V[final_labels == -1] * neg_log(1 - preds[final_labels == -1])
    
    main_loss = loss_mtx.mean() + P['reg'] * expected_positive_regularizer(preds, P)
    # main_loss = loss_mtx.mean()
    return main_loss

def expected_positive_regularizer(preds, P, norm='2'):
    # Assumes predictions in [0,1].
    if norm == '1':
        reg = torch.abs(preds.sum(1).mean(0) - P['expected_num_pos'])
    elif norm == '2':
        reg = (preds.sum(1).mean(0) - P['expected_num_pos'])**2
    else:
        raise NotImplementedError
    return reg / (P['num_classes'] ** 2)

def AN_PL_loss(logits, pseudo_labels, label_vec,K,V,Q,epoch,P=None):
    preds = torch.sigmoid(logits) 
    final_labels = torch.where(label_vec == 0, pseudo_labels, label_vec)
    mask = (label_vec==0) & (pseudo_labels<1)
    loss_mtx = torch.zeros_like(preds) 
    loss_mtx[final_labels==1] = neg_log(preds[final_labels==1]) 
    loss_mtx[mask] = neg_log(1 - preds[mask]) 
    main_loss = loss_mtx.mean()
    return main_loss 
def AN_loss(logits, pseudo_labels, label_vec,K,V,Q,epoch,P=None):
    preds = torch.sigmoid(logits) 
    loss_mtx = torch.zeros_like(preds) 
    loss_mtx[label_vec==1] = neg_log(preds[label_vec==1])
    loss_mtx[label_vec==0] = neg_log(1 - preds[label_vec==0])
    main_loss = loss_mtx.mean() 
    return main_loss

def smooth_ce_loss(global_logits, label_vec_obs, P):
    # Soft ground truth labels
    pl_gt = label_vec_obs * 0.9 + (1 - 0.9) / P['num_classes']
    pl_gt = label_vec_obs
    cross_entropy_loss = F.cross_entropy(global_logits, pl_gt, reduction='mean')
    return cross_entropy_loss

def hill_loss(preds,label_vec,K,epoch,P=None):
    loss_mtx = torch.zeros_like(preds)
    loss_mtx[label_vec == 1]=neg_log(preds[label_vec == 1])
    loss_mtx[label_vec == 0]=(1.5-preds[label_vec == 0])*preds[label_vec == 0]*preds[label_vec == 0]
    main_loss=loss_mtx.mean()
    return main_loss

def EPR_loss(preds,label_vec,K,epoch,P=None):
    loss_mtx = torch.zeros_like(preds)
    loss_mtx[label_vec == 1]=neg_log(preds[label_vec == 1])
    loss_mtx[label_vec == 0]=0*preds[label_vec == 0]
    main_loss=loss_mtx.mean()
    return main_loss

def weight_loss(preds,label_vec,K,epoch,P=None):
    loss_mtx = torch.zeros_like(preds)
    loss_mtx[label_vec == 1]=8*neg_log(preds[label_vec == 1])
    loss_mtx[label_vec == 0]=0.9*loss2(preds[label_vec == 0],0.99)
    main_loss=loss_mtx.mean()
    return main_loss

LOG_EPSILON = 1e-7

def neg_log(x):
    return - torch.log(x + LOG_EPSILON)

def loss1(x,q):
    return (1 - torch.pow(x, q)) / q

def loss2(x,q):
    return (1 - torch.pow(1-x, q)) / q