import os
import json
import numpy as np
from PIL import Image
import torch
import copy
from torch.utils.data import Dataset
from torchvision import transforms
import open_clip

def get_metadata(dataset_name):
    if dataset_name == 'pascal':
        meta = {
            'num_classes': 20,
            'path_to_dataset': '/cm/archive/luongtk/spml/data/pascal',
            'path_to_images': '/cm/archive/luongtk/spml/data/pascal/VOCdevkit/VOC2012/JPEGImages',
            'path_to_classnames': '/cm/archive/luongtk/AEVLP/data/classnames/voc_labels.txt'
        }
    elif dataset_name == 'coco':
        meta = {
            'num_classes': 80,
            'path_to_dataset': '/cm/archive/luongtk/spml/data/coco',
            'path_to_images': '/cm/archive/luongtk/spml/data/coco',
            'path_to_classnames': '/cm/archive/luongtk/AEVLP/data/classnames/coco_labels.txt'            
        }
    elif dataset_name == 'nuswide':
        meta = {
            'num_classes': 81,
            'path_to_dataset': '/cm/archive/luongtk/spml/data/nuswide',
            'path_to_images': '/cm/archive/luongtk/spml/data/nuswide/Flickr',
            'path_to_classnames': '/cm/archive/luongtk/AEVLP/data/classnames/nuswide_labels.txt'
        }
    elif dataset_name == 'cub':
        meta = {
            'num_classes': 312,
            'path_to_dataset': '/cm/archive/luongtk/spml/data/cub',
            'path_to_images': '/cm/archive/luongtk/spml/data/cub/CUB_200_2011/images',
            'path_to_classnames': '/cm/archive/luongtk/AEVLP/data/classnames/cub_labels.txt'
        }
    else:
        raise NotImplementedError('Metadata dictionary not implemented.')
    return meta

def get_imagenet_stats():
    '''
    Returns standard ImageNet statistics. 
    '''
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    return (imagenet_mean, imagenet_std)

def get_transforms(P):
    '''
    Returns image transforms.
    '''
    if P['arch'] == "vit-l":
        size =  336 
    elif P['arch'] == 'resnet50_clip':
        size = 224
    else: 
        size = 448
    (imagenet_mean, imagenet_std) = get_imagenet_stats()
    tx = {}
    if P['source'] == 'open_clip':
        model, _, tx['pl'] = open_clip.create_model_and_transforms(P['clip_model'], pretrained=P['clip_weight'])
        # release memory by model 
        del model
        del _
    else:
        tx['pl'] = transforms.Compose([ # transformation for clip input (including augmentation)
            transforms.Resize((224, 224)), # clip size input
            transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomApply([
            #     transforms.RandomChoice([
            #         transforms.RandomRotation(degrees=15),
            #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            #         transforms.RandomAdjustSharpness(sharpness_factor=1.5),
            #         transforms.RandomAutocontrast()
            #     ])
            # ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
   
    tx['train'] = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['val'] = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['test'] = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return tx

def generate_split(num_ex, frac, rng):
    '''
    Computes indices for a randomized split of num_ex objects into two parts,
    so we return two index vectors: idx_1 and idx_2. Note that idx_1 has length
    (1.0 - frac)*num_ex and idx_2 has length frac*num_ex. Sorted index sets are 
    returned because this function is for splitting, not shuffling. 
    '''
    
    # compute size of each split:
    n_2 = int(np.round(frac * num_ex))
    n_1 = num_ex - n_2
    
    # assign indices to splits:
    idx_rand = rng.permutation(num_ex)
    idx_1 = np.sort(idx_rand[:n_1])
    idx_2 = np.sort(idx_rand[-n_2:])
    
    return (idx_1, idx_2)

def get_data(P):
    '''
    Given a parameter dictionary P, initialize and return the specified dataset. 
    '''
    
    # define transforms:
    tx = get_transforms(P)
    
    # select and return the right dataset:
    if P['dataset'] == 'coco':
        ds = multilabel(P, tx).get_datasets()
    elif P['dataset'] == 'pascal':
        ds = multilabel(P, tx).get_datasets()
    elif P['dataset'] == 'nuswide':
        ds = multilabel(P, tx).get_datasets()
    elif P['dataset'] == 'cub':
        ds = multilabel(P, tx).get_datasets()
    else:
        raise ValueError('Unknown dataset.')
    
    # Optionally overwrite the observed training labels with clean labels:
    # assert P['train_set_variant'] in ['clean', 'observed']
    if P['train_set_variant'] == 'clean':
        print('Using clean labels for training.')
        ds['train'].label_matrix_obs = copy.deepcopy(ds['train'].label_matrix)
    else:
        print('Using single positive labels for training.')
    
    # Optionally overwrite the observed val labels with clean labels:
    assert P['val_set_variant'] in ['clean', 'observed']
    if P['val_set_variant'] == 'clean':
        print('Using clean labels for validation.')
        ds['val'].label_matrix_obs = copy.deepcopy(ds['val'].label_matrix)
    else:
        print('Using single positive labels for validation.')
    
    # We always use a clean test set:
    ds['test'].label_matrix_obs = copy.deepcopy(ds['test'].label_matrix)
            
    return ds

def load_data(base_path, P):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        data[phase]['labels'] = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
        data[phase]['labels_obs'] = np.load(os.path.join(base_path, 'formatted_{}_labels_obs.npy'.format(phase)))
        data[phase]['images'] = np.load(os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)))
        data[phase]['feats'] = np.load(P['{}_feats_file'.format(phase)]) if P['use_feats'] else []
    return data

class multilabel:

    def __init__(self, P, tx):
        
        # get dataset metadata:
        meta = get_metadata(P['dataset'])
        self.base_path = meta['path_to_dataset']
        
        # load data:
        source_data = load_data(self.base_path, P)
        
        # generate indices to split official train set into train and val:
        split_idx = {}
        (split_idx['train'], split_idx['val']) = generate_split(
            len(source_data['train']['images']),
            P['val_frac'],
            np.random.RandomState(P['split_seed'])
            )
        # subsample split indices: # commenting this out makes the val set map be low?
        ss_rng = np.random.RandomState(P['ss_seed'])
        temp_train_idx = copy.deepcopy(split_idx['train'])
        for phase in ['train', 'val']:
            num_initial = len(split_idx[phase])
            num_final = int(np.round(P['ss_frac_{}'.format(phase)] * num_initial))
            split_idx[phase] = split_idx[phase][np.sort(ss_rng.permutation(num_initial)[:num_final])]
        
        # define train set:
        self.train = ds_multilabel(
            P['dataset'],
            source_data['train']['images'][split_idx['train']],
            source_data['train']['labels'][split_idx['train'], :],
            source_data['train']['labels_obs'][split_idx['train'], :],
            source_data['train']['feats'][split_idx['train'], :] if P['use_feats'] else [],
            tx['train'],
            P['use_feats'],
            P['grid_size'],
            tx['pl'],
            True
        )
            
        # define val set:
        self.val = ds_multilabel(
            P['dataset'],
            source_data['train']['images'][split_idx['val']],
            source_data['train']['labels'][split_idx['val'], :],
            source_data['train']['labels_obs'][split_idx['val'], :],
            source_data['train']['feats'][split_idx['val'], :] if P['use_feats'] else [],
            tx['val'],
            P['use_feats']
        )
        
        # define test set:
        self.test = ds_multilabel(
            P['dataset'],
            source_data['val']['images'],
            source_data['val']['labels'],
            source_data['val']['labels_obs'],
            source_data['val']['feats'],
            tx['test'],
            P['use_feats']
        )
        
        # define dict of dataset lengths: 
        self.lengths = {'train': len(self.train), 'val': len(self.val), 'test': len(self.test)}
    
    def get_datasets(self):
        return {'train': self.train, 'val': self.val, 'test': self.test}

class ds_multilabel(Dataset):

    def __init__(self, dataset_name, image_ids, label_matrix, label_matrix_obs, feats, tx, use_feats, grid_size=None, tx_pl=None, is_train_dataset=False):
        meta = get_metadata(dataset_name)
        self.dataset_name = dataset_name
        self.num_classes = meta['num_classes']
        self.path_to_images = meta['path_to_images']
        
        self.image_ids = image_ids
        self.label_matrix = label_matrix
        self.label_matrix_obs = label_matrix_obs
        if is_train_dataset:
            self.pseudo_label_matrix = np.zeros_like(label_matrix)
        self.feats = feats
        self.tx = tx
        self.use_feats = use_feats
        self.tx_pl = tx_pl
        self.is_train_dataset = is_train_dataset
        self.grid_size = grid_size # define the grid size for splitting image into sub-images 

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,idx):
        if self.use_feats:
            # Set I to be a feature vector:
            I = torch.FloatTensor(np.copy(self.feats[idx, :]))
        else:
            # Set I to be an image: 
            image_path = os.path.join(self.path_to_images, self.image_ids[idx])
            with Image.open(image_path) as I_raw:
                I = self.tx(I_raw.convert('RGB'))
            if self.tx_pl is not None and self.is_train_dataset == True and self.grid_size is not None: 
                # Split image into a nxn grid
                with Image.open(image_path) as I_raw:
                    height = I_raw.size[1]
                    width = I_raw.size[0]
                    
                    # Convert image to RGB if not already
                    I_raw = I_raw.convert('RGB')
                    I_clip = self.tx_pl(I_raw)
                    # Calculate the size of each tile
                    tile_width = width // self.grid_size
                    tile_height = height // self.grid_size
                    
                    # Initialize a list to store the cropped tiles
                    sub_images = []
                    sub_images.append(I_clip.unsqueeze(0))
                    # Iterate through each row and column to split the image
                    for y in range(self.grid_size):
                        for x in range(self.grid_size):
                            # Define the overlap ratio for cropping based on standard size
                            tile_width_offset = int(tile_width * np.random.uniform(0, 0.2))
                            tile_height_offset = int(tile_height * np.random.uniform(0, 0.2))

                            tile_width_offset = 0 
                            tile_height_offset = 0
                            
                            # Computing the coordinates for cropping
                            left = x * tile_width - tile_width_offset
                            left = max(0, left)
                            upper = y * tile_height - tile_height_offset
                            upper = max(0, upper)
                            right = (x + 1) * tile_width + tile_width_offset
                            right = min(width, right)
                            lower = (y + 1) * tile_height + tile_height_offset
                            lower = min(height, lower)
                            
                            # Crop the tile from the original image
                            tile = I_raw.crop((left, upper, right, lower))
                            
                            # Apply transformations
                            tile = self.tx_pl(tile).unsqueeze(0)
                            
                            # Append the transformed tile to the list
                            sub_images.append(tile)
        
        out = {
            'image': I,
            'label_vec_obs': torch.FloatTensor(np.copy(self.label_matrix_obs[idx, :])),
            'label_vec_true': torch.FloatTensor(np.copy(self.label_matrix[idx, :])),
            'idx': idx,
            'image_path': image_path # added for CAM visualization purpose
        }
        
        if self.is_train_dataset:
            # out['image_clip'] = I_clip
            out['sub_images'] = torch.cat(sub_images) # Contains the original image and the sub-images (n * n + 1 images in total)
            out['pseudo_label_vec'] = torch.FloatTensor(np.copy(self.pseudo_label_matrix[idx, :]))

        return out

def parse_categories(categories):
    category_list = []
    id_to_index = {}
    for i in range(len(categories)):
        category_list.append(categories[i]['name'])
        id_to_index[categories[i]['id']] = i
    return (category_list, id_to_index)

def get_category_list(P):
    if P['dataset'] == 'pascal':
        catName_to_catID = {
            'aeroplane': 0,
            'bicycle': 1,
            'bird': 2,
            'boat': 3,
            'bottle': 4,
            'bus': 5,
            'car': 6,
            'cat': 7,
            'chair': 8,
            'cow': 9,
            'diningtable': 10,
            'dog': 11,
            'horse': 12,
            'motorbike': 13,
            'person': 14,
            'pottedplant': 15,
            'sheep': 16,
            'sofa': 17,
            'train': 18,
            'tvmonitor': 19
        }
        return {catName_to_catID[k]: k for k in catName_to_catID}
    
    elif P['dataset'] == 'coco':
        load_path = '/cm/archive/luongtk/spml/data/coco'
        meta = {}
        meta['category_id_to_index'] = {}
        meta['category_list'] = []

        with open(os.path.join(load_path, 'annotations', 'instances_train2014.json'), 'r') as f:
            D = json.load(f)

        (meta['category_list'], meta['category_id_to_index']) = parse_categories(D['categories'])
        return meta['category_list']

    elif P['dataset'] == 'nuswide':
        pass # TODO
    
    elif P['dataset'] == 'cub':
        pass # TODO