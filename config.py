import os
import sys
import argparse
from munch import Munch as mch
from os.path import join as ospj
from datetime import datetime
import json


_DATASET = ('pascal', 'coco', 'nuswide', 'cub')
_TRAIN_SET_VARIANT = ('observed', 'clean')
_OPTIMIZER = ('adam', 'sgd')
_SCHEMES = ('LL-R', 'LL-Ct', 'LL-Cp')
_LOOKUP = {
    'feat_dim': {
        'resnet50': 2048,
        'resnet50_clip': 1024,
        'resnet101': 2048,
        'convnext_large_1k': 1536,
        'convnext_large_22k': 1536,
        'convnext_xlarge_1k': 2048,
        'convnext_xlarge_22k': 2048,
        'vit-l': 768
    },
    'num_classes': {
        'pascal': 20,
        'coco': 80,
        'nuswide': 81,
        'cub': 312
    },
    'classnames': {
        'pascal': './data/classnames/voc_labels.txt',
        'coco': './data/classnames/coco_labels.txt',
        'nuswide': './data/classnames/nuswide_labels.txt',
        'cub': './data/classnames/cub_labels.txt'
    },
    'relation':{ # matrix similarity between labels
        'pascal':'./data/relation/relation+voc.npy',
        'coco':'./data/relation/relation+coco.npy',
        'nuswide':'./data/relation/relation+nuswide.npy',
        'cub':'./data/relation/relation+cub.npy'
    
    },
    'sparse_topk': { # label-to-label correspondance GCN
        'pascal': 20,
        'coco': 62,
        'nuswide': 50,
        'cub': 312
    },
    'reweight_p':{ # label-to-label correspondance GCN
        'pascal': 0.2,
        'coco': 0.2,
        'nuswide': 0.2,
        'cub': 0.2
    },
    'T':{ # label-to-label correspondance GCN
        'pascal': 0.3,
        'coco': 0.2,
        'nuswide': 0.2,
        'cub': 0.3
    },
    'top_k': { # top k highest score of pseudo labels
        'pascal': 3,
        'coco': 3,
        'nuswide': 3,
        'cub': 32
    },
    'expected_num_pos': {
            'pascal': 1.5,
            'coco': 2.9,
            'nuswide': 1.9,
            'cub': 31.4
    },
    
}
_CLIP_BACKBONES = ('ViT-B-16-SigLIP', 'ViT-B/32', 'ViT-B/16', 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN101x4', 'RN50x4', 'RN50x16')
_SOURCE = ('openai', 'open_clip')



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_dir(runs_dir, exp_name):
    runs_dir = ospj(runs_dir, exp_name)
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    return runs_dir



def set_follow_up_configs(args):
    args.feat_dim = _LOOKUP['feat_dim'][args.arch]
    args.num_classes = _LOOKUP['num_classes'][args.dataset]
    now = datetime.now()
    args.experiment_name = str(now).split(".")[0].replace('-','').replace(" ","_").replace(":","")
    args.save_path = set_dir(args.save_path, args.experiment_name)
    # clip
    args.classnames = _LOOKUP['classnames'][args.dataset]
    args.relation = _LOOKUP['relation'][args.dataset]
    args.sparse_topk = _LOOKUP['sparse_topk'][args.dataset]
    args.reweight_p = _LOOKUP['reweight_p'][args.dataset]
    args.T = _LOOKUP['T'][args.dataset]
    args.top_k = _LOOKUP['top_k'][args.dataset] 
    args.expected_num_pos = _LOOKUP['expected_num_pos'][args.dataset]
    args.lam_1 = args.lam[0] 
    args.lam_2 = args.lam[1]

    if args.loss == 'an_loss':
        args.use_pl = False

    # if args.delta_rel != 0:
    #     args.delta_rel /= 100
    # args.delta_rel = 0.001
    # args.clean_rate = 1
    
    return args


def get_configs():
    parser = argparse.ArgumentParser()

    # Default settings
    parser.add_argument('--seed', type=int, default=1200, 
                        help='overall numpy seed')
    parser.add_argument('--ss_seed', type=int, default=999,
                        help='seed fo subsampling')
    parser.add_argument('--ss_frac_train', type=float, default=1.0,
                        help='fraction of training set to subsample')
    parser.add_argument('--ss_frac_val', type=float, default=1.0,
                        help='fraction of val set to subsample')
    parser.add_argument('--use_feats', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='False if end-to-end training, True if linear training')
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--split_seed', type=int, default=1200)
    parser.add_argument('--train_set_variant', type=str, default='observed',
                        choices=_TRAIN_SET_VARIANT)
    parser.add_argument('--val_set_variant', type=str, default='clean')
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--freeze_feature_extractor', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--use_pretrained', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--num_epochs', type=int, default=8)
    
    # Util
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--exp_name', type=str, default='exp_default')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_num', type=str, default='7')
    
    # Data
    parser.add_argument('--dataset', type=str, default='coco',
                        choices=_DATASET)

    # Hyperparameters
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=_OPTIMIZER)
    parser.add_argument('--bsize', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_mult', type=float, default=10)
    parser.add_argument('--is_train', type=bool,default=False)
    parser.add_argument('--linear_init', type=int,default=0)

    # Loss function 
    parser.add_argument('--loss', type=str, default='gr_loss')
    
    #KVQ
    parser.add_argument('--beta', type=list, default=[0, 2,-2,-2])
    parser.add_argument('--alpha', type=list, default=[0.5,2,0.8,0.5])
    parser.add_argument('--q2q3', type=list, default=[0.01,1])
    parser.add_argument('--lam', type=list, default=[0.8, 0.3])
    parser.add_argument('--rho', type=float, default=0.9) 
    parser.add_argument('--reg', type=float, default=0.001)
    
    # Pseudo Label Generator configs
    parser.add_argument('--use_pl', type=bool, default=True)
    parser.add_argument('--clip_model', type=str, default='ViT-B/16', choices=_CLIP_BACKBONES)
    parser.add_argument('--clip_weight', type=str, default='openai')
    parser.add_argument('--source', type=str, default='openai', choices=_SOURCE)
    parser.add_argument('--n_ctx', type=int, default=4)
    parser.add_argument('--ctx_init', type=str, default='a photo of a')
    parser.add_argument('--temp', type=float, default=0.01)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--grid_size', type=int, default=4) 
    parser.add_argument('--negative_ratio', type=float, default=0.1)
    
    # Config file path 
    parser.add_argument('--use_config_file', type=bool, default=False)    
    
    args = parser.parse_args() 
    if args.use_config_file:
        # load json file
        with open(f"./configs/{args.dataset}.json", 'r') as f:
            config = json.load(f)
            print("Use pre-defined config file!")
            args = mch(config)
            args['save_path'] = 'results'
    args = set_follow_up_configs(args)
    args = mch(**vars(args))
    

    return args


