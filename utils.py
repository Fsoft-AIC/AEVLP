import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
import os 

def temperature_scaled_softmax(logits, temperature):
        return F.softmax(logits / temperature, dim=-1)

def get_pseudo_labels(logits, preds, pseudo_label_vec, label_vec, P):
    """
    Function to generate pseudo-label for multi-label learning:
        logits: (batch_size * (grid_size* grid_size + 1), num_classes) contains logits from CLIP of global and local images with their labels
        preds: (batch_size, num_classes) current predictions of model
        label_vec: (batch_size, num_classes), contain only a single positive label = 1, other = 0
    """
    batch_size = int(logits.shape[0]) // (P['grid_size'] * P['grid_size'] + 1)

    similarity = temperature_scaled_softmax(logits, P['temp'])
    sim_global = similarity[0:batch_size] # score of global region
    sim_locals = torch.chunk(similarity[batch_size:], P['grid_size'] * P['grid_size'], dim=0) 
    sim_locals = torch.stack(sim_locals, dim=0) # score of local regions
    sim_locals = sim_locals.permute(1, 0, 2) # shape: (batch_size, num_locals, num_classes)
    alpha = torch.max(sim_locals, dim=1)[0]
    beta = torch.min(sim_locals, dim=1)[0]
    avg = torch.mean(sim_locals, dim=1)
    single_pos_pro = torch.max(sim_global * label_vec, dim=1)[0]
    # eta = P['eta'] 
    eta = torch.minimum(single_pos_pro, torch.tensor(P['eta']))
    eta = eta.unsqueeze(1).repeat(1, P['num_classes'])

    # gamma_i = 1 if alpha_i > eta else 0
    gamma = torch.where(alpha > eta, torch.tensor(1).to(P['device']), torch.tensor(0).to(P['device'])) 
    sim_ag = alpha * gamma + beta * (1 - gamma)
    sim_final = 1/ 2 * (sim_global + sim_ag)

    sim = 1 / 2 * (sim_global + avg)
    
    # define pseudo labels
    pseudo_labels = torch.zeros_like(sim_final)
    #positive pseudo label
    scalar_pos = torch.tensor(1, dtype=torch.float32).to(P['device'])
    # Choose labels with top-k highest scores and greater than threshold
    # Get the top-k indices and values for each sample in the batch
    values, indices = torch.topk(sim_final, k=P['top_k'], dim=1) # change
    
    # Create a mask for values greater than the threshold
    mask = values >= P['threshold']
    
    # Initialize the pseudo labels tensor with zeros
    pseudo_labels = torch.zeros_like(sim_final, dtype=torch.int)
    
    # Use the mask to set the appropriate positions to 1
    pseudo_labels.scatter_(1, indices, mask.int()) 

    acc_pseudo_labels = pseudo_labels.clone()
    acc_pseudo_labels = torch.where(pseudo_label_vec==0, acc_pseudo_labels, pseudo_label_vec)

    # Use 10% of low-confidence of labels to be negative peseudo-labels 
    num_negatives = int(P['negative_ratio'] * P['num_classes'])
    if num_negatives > 0:
        _, neg_indices = torch.topk(sim, k=num_negatives, largest=False, dim=1)
        # Set negative pseudo-labels for these indices
        pseudo_labels.scatter_(-1, neg_indices, torch.zeros_like(neg_indices, dtype=torch.int))

    return acc_pseudo_labels, pseudo_labels

