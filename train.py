import numpy as np
from sklearn import metrics
import torch
import datasets
import models
from instrumentation import compute_metrics,compute_precision_recall_pl
import losses
import datetime
import os
from tqdm import tqdm
from utils import get_pseudo_labels

def run_train(P):
   
   # Get data
    dataset = datasets.get_data(P)
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size = P['bsize'],
            shuffle = phase == 'train',
            sampler = None,
            num_workers = P['num_workers'],
            drop_last = False,
            pin_memory = True
        )
    # Define model
    model = models.ImageClassifier(P)
    feature_extractor_params = [param for param in list(model.feature_extractor.parameters()) if param.requires_grad]
    opt_params = [
        {'params': feature_extractor_params, 'lr' : P['lr']}
        ]
    if P['arch'] == "resnet50":
        linear_classifier_params = [param for param in list(model.linear_classifier.parameters()) if param.requires_grad]
        opt_params.append({'params': linear_classifier_params, 'lr' : P['lr_mult'] * P['lr']})

    # If use Pseudo-label Generator
    if P['use_pl'] == True:
        if P['source'] == 'openai':
            pseudo_label_generator = models.PseudoLabelGeneratorOpenAI(P)
        
        else:
            pseudo_label_generator = models.PseudoLabelGeneratorOpenCLIP(P)

        for name, params in pseudo_label_generator.named_parameters():
            if "gc" in name:
                params.requires_grad = False # Freeze GCN
        # If train Pseudo-label Generator
        pl_opt_params = [
            {'params': [param for param in list(pseudo_label_generator.parameters()) if param.requires_grad]}
        ]    

    # Set device 
    if torch.cuda.is_available():
        P['device'] = f"cuda:{P['gpu_num']}"
    else:
        print("CUDA is not available. Using CPU.")
        P['device'] = 'cpu'

    # Set optimizer
    if P['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=P['lr'])
        if P['use_pl'] == True:
            pl_optimizer = torch.optim.Adam(pl_opt_params, lr=P['lr'])
    elif P['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(opt_params, lr=P['lr'], momentum=0.9, weight_decay=0.001)
        if P['use_pl'] == True:
            pl_optimizer = torch.optim.SGD(pl_opt_params, lr=P['lr']/10, momentum=0.9, weight_decay=0.001)

    # Load module to device
    model.to(P['device'])
    if P['use_pl'] == True:
        pseudo_label_generator.to(P['device'])
    
    # Initialize metrics
    bestmap_val = 0
    bestmap_test = 0
    val_loss_lst=[]
    dis_lst=[]
    early_stop = 0 

    print("Start training")

    # Training loop
    for epoch in range(1, P['num_epochs']+1):
        P['epoch'] = epoch
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                if P['use_pl'] == True:
                    pseudo_label_generator.eval()

                    # Compute Pseudo-label Generator Performance
                    y_pseudo = np.zeros((len(dataset[phase]), P['num_classes']))
                    y_train_true = np.zeros((len(dataset[phase]), P['num_classes'])) 
                    y_train_obs = np.zeros((len(dataset[phase]), P['num_classes'])) 
                    batch_stack_train = 0
                    
                n = P['linear_init']
                if epoch <= n:
                    for param in model.feature_extractor.parameters():
                        param.requires_grad = False
                else:
                    for param in model.feature_extractor.parameters():
                        param.requires_grad = True
                # Beta
                w_0, w_max, b_0, b_max = P['beta']
                w = w_0 + (w_max - w_0) * max((epoch - n - 1),0)/(P['num_epochs'] - n)
                b = b_0 +(b_max - b_0) * max((epoch - n - 1),0)/(P['num_epochs'] - n)
                # Alpha
                mu_0, sigma_0, mu_max, sigma_max = P['alpha']
                mu = mu_0 + (mu_max - mu_0)*max((epoch - n), 0)/(P['num_epochs'] - n)
                sigma = sigma_0 + (sigma_max - sigma_0)*max(( epoch - n), 0)/(P['num_epochs'] - n)  
                # q2q3
                Q = P['q2q3']             
                k_function = models.KFunction(w, b)
                
                print(f'w:{w},b:{b}')
                print(f'mu:{mu};sigma:{sigma}')
                print(f'q2:{Q[0]};q3:{Q[1]}')


            else:
                model.eval()
                y_pred_val = np.zeros((len(dataset[phase]), P['num_classes']))
                y_val_true = np.zeros((len(dataset[phase]), P['num_classes']))
                y_val_obs = np.zeros((len(dataset[phase]), P['num_classes']))
                batch_stack_val = 0

            with torch.set_grad_enabled(phase == 'train'):
                for batch in tqdm(dataloader[phase]):
                    # Move data to GPU
                    image = batch['image'].to(P['device'], non_blocking=True)
                    label_vec_obs = batch['label_vec_obs'].to(P['device'], non_blocking=True)
                    label_vec_true = batch['label_vec_true'].clone().numpy()
                    idx = batch['idx']
                    if phase == 'train' and P['use_pl'] == True:
                        sub_images = batch['sub_images'].to(P['device'], non_blocking=True)
                        pseudo_label_vec = batch['pseudo_label_vec'].to(P['device'], non_blocking=True)

                    # Forward pass
                    optimizer.zero_grad() 
                    # pl_optimizer.zero_grad()
                        
                    logits = model(image)
                  
                    if torch.isnan(logits).any():
                        print(logits) 
                    if logits.dim() == 1:
                        logits = torch.unsqueeze(logits, 0)
                    preds = torch.sigmoid(logits)
                    detached_preds = preds.detach()
                    
                    K = k_function(detached_preds) 
                    V = models.VFunction(detached_preds, mu, sigma)
                    
                    if phase == 'train':
                        if P['use_pl'] == True:
                            clip_images = torch.chunk(sub_images, P['grid_size'] * P['grid_size'] + 1, dim=1)
                           
                            clip_input = torch.cat([x.squeeze(1) for x in clip_images], dim=0)
                            if P['source'] == 'openclip':
                                clip_input = clip_input.half()
                            pl_logits = pseudo_label_generator.generate(clip_input)   
                                        
                            detached_pl_logits = pl_logits.detach()
                            _, pseudo_labels = get_pseudo_labels(detached_pl_logits, detached_preds, pseudo_label_vec, label_vec_obs, P)
                            dataset[phase].pseudo_label_matrix[idx] = pseudo_labels.clone().cpu().numpy()

                            # Used for determining performance of pseudo-labeling 
                            this_batch_size = pseudo_labels.shape[0]
                            y_pseudo[batch_stack_train : batch_stack_train+this_batch_size] = pseudo_labels.clone().cpu().numpy()
                            y_train_true[batch_stack_train : batch_stack_train+this_batch_size] = label_vec_true
                            y_train_obs[batch_stack_train: batch_stack_train+this_batch_size] = label_vec_obs.clone().cpu().numpy()
                            batch_stack_train += this_batch_size
                        else:
                            pseudo_labels = torch.zeros_like(label_vec_obs)

                        if P['loss'] == 'gpr_loss':
                            loss = losses.GPR_loss(logits, pseudo_labels, label_vec_obs,K ,V ,Q ,epoch, P)
                        elif P['loss'] == 'an_pl_loss':
                            loss = losses.AN_PL_loss(logits, pseudo_labels, label_vec_obs,K ,V ,Q ,epoch, P)
                        elif P['loss'] == 'an_loss':
                            loss = losses.AN_loss(logits, pseudo_labels, label_vec_obs,K ,V ,Q ,epoch, P) 
                        else: 
                            raise "Loss function is undefined!!!"
                        loss.backward()
                        optimizer.step()
                    else:
                        preds_np = preds.cpu().numpy()
                        label_vec_obs_np=label_vec_obs.cpu().numpy()
                        this_batch_size = preds_np.shape[0]
                        y_pred_val[batch_stack_val : batch_stack_val+this_batch_size] = preds_np
                        y_val_true[batch_stack_val : batch_stack_val+this_batch_size] = label_vec_true
                        y_val_obs[batch_stack_val : batch_stack_val+this_batch_size] = label_vec_obs_np
                        

                        batch_stack_val += this_batch_size
        metrics = compute_metrics(y_pred_val, y_val_true)

        map_val = metrics['map']

        if P['use_pl']:
            pl_precision, pl_recall, pl_f1 = compute_precision_recall_pl(y_pseudo, y_train_obs, y_train_true)
            print(f"Epoch {epoch} : val mAP {map_val:.3f}")
            print(f"Pseudo-labeling Performance: precision {pl_precision:.3f}, recall {pl_recall:.3f}, f1 {pl_f1:.3f}")
            del y_pseudo
            del y_train_true
            del y_train_obs
        else:
            print(f"Epoch {epoch} : val mAP {map_val:.3f}")
        val_loss_lst.append(map_val)
        
        del y_pred_val
        del y_val_true
        del y_val_obs
        if bestmap_val < map_val:
            early_stop = 0
            bestmap_val = map_val
            bestmap_epoch = epoch
            
            print(f'Saving model weight for best val mAP {bestmap_val:.3f}')
            path = os.path.join(P['save_path'], 'bestmodel.pt')
            torch.save((model.state_dict(), P), path)
        else:
            early_stop += 1
            if bestmap_val - map_val > 2 or early_stop == 2:
                print('Early stopped.')
                break
    # Test phase

    model_state, _ = torch.load(path)
    model.load_state_dict(model_state)

    phase = 'test'
    model.eval()

    y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
    y_true = np.zeros((len(dataset[phase]), P['num_classes']))
    batch_stack = 0
    with torch.set_grad_enabled(phase == 'train'):
        for batch in tqdm(dataloader[phase]):
            # Move data to GPU
            image = batch['image'].to(P['device'], non_blocking=True)
            label_vec_obs = batch['label_vec_obs'].to(P['device'], non_blocking=True)
            label_vec_true = batch['label_vec_true'].clone().numpy()
            idx = batch['idx']

            # Forward pass
            optimizer.zero_grad()
            logits = model(image)
                   
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            preds = torch.sigmoid(logits)
               
            preds_np = preds.cpu().numpy()
            this_batch_size = preds_np.shape[0]
            y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
            y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
            batch_stack += this_batch_size

    metrics = compute_metrics(y_pred, y_true)
    map_test = metrics['map']
    print('val_map :',[round(x,2) for x in val_loss_lst])
    print(f'Test mAP : {map_test:.3f} when trained until epoch {bestmap_epoch}')
    # print([round(x,3) for x in dis_lst])
    
    with open(f"{P['save_path']}/result.txt", "w") as f:
        f.write("Config : \n")
        f.write(f"- lr : {P['lr']}\n")
        f.write(f"- bsize : {P['bsize']}\n")
        f.write(f"- num_epochs : {P['num_epochs']}\n")
        f.write(f"- grid_size : {P['grid_size']}\n")
        f.write(f"- temp : {P['temp']}\n")
        f.write(f"- eta : {P['eta']}\n")
        f.write(f"- threshold : {P['threshold']}\n")
        f.write(f"Best epoch : {bestmap_epoch}\n")
        f.write(f"Val map :{[round(x,2) for x in val_loss_lst]}\n")
        f.write(f"Test mAP : {map_test:.3f} when trained until epoch {bestmap_epoch}\n")
    