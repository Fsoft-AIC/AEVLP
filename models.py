import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import timm
import open_clip
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPProcessor, CLIPVisionModel, CLIPVisionModelWithProjection, AutoTokenizer, CLIPTextModelWithProjection


_tokenizer = _Tokenizer()

def load_clip_to_cpu(backbone_name='RN50'):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(  # type: ignore
            model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())  # type: ignore

    return model

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    
    def forward(self, feature_map):
        return F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)


### Define the main backbone for multi-label classification
class ImageClassifier(torch.nn.Module):
    def __init__(self, P):
        super(ImageClassifier, self).__init__()
        
        self.arch = P['arch']
        if self.arch == 'resnet50':
            if P['use_pretrained']:
                weights='IMAGENET1K_V2'
                # feature_extractor = torchvision.models.resnet50(weights=weights) 
                feature_extractor = torchvision.models.resnet50(pretrained=P['use_pretrained'])
            else:
                feature_extractor = torchvision.models.resnet50(pretrained=P['use_pretrained'])
            feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])
            
            if P['freeze_feature_extractor']:
                for param in feature_extractor.parameters():
                    param.requires_grad = False
            else:
                for param in feature_extractor.parameters():
                    param.requires_grad = True
            self.feature_extractor = feature_extractor
                
            self.avgpool = GlobalAvgPool2d()

            linear_classifier = torch.nn.Linear(P['feat_dim'], P['num_classes'], bias=True)
            self.linear_classifier = linear_classifier
        elif self.arch == 'resnet101':
            feature_extractor = torchvision.models.resnet101(pretrained=P['use_pretrained'])
            self.feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            self.avgpool = GlobalAvgPool2d()
            linear_classifier = torch.nn.Linear(P['feat_dim'], P['num_classes'], bias=True)
            self.linear_classifier = linear_classifier
        elif self.arch == 'convnext_large_1k':
            self.feature_extractor = timm.create_model('convnext_large.fb_in1k', pretrained=True)
            self.feature_extractor.head.fc=nn.Linear(in_features=P['feat_dim'], out_features=P['num_classes'], bias=True)
            for param in self.feature_extractor.parameters():
                    param.requires_grad = True
        elif self.arch == 'convnext_large_22k': 
            self.feature_extractor = timm.create_model('convnext_large.fb_in22k', pretrained=True)
            self.feature_extractor.head.fc=nn.Linear(in_features=P['feat_dim'], out_features=P['num_classes'], bias=True)
            for param in self.feature_extractor.parameters():
                    param.requires_grad = True 
        elif self.arch == 'convnext_xlarge_22k':
            self.feature_extractor = timm.create_model('convnext_xlarge.fb_in22k', pretrained=True)
            self.feature_extractor.head.fc=nn.Linear(in_features=P['feat_dim'], out_features=P['num_classes'], bias=True)
            for param in self.feature_extractor.parameters():
                    param.requires_grad = True
        elif self.arch == 'convnext_xlarge_1k':
            self.feature_extractor = timm.create_model('convnext_xlarge.fb_in22k_ft_in1k', pretrained=True)
            self.feature_extractor.head.fc=nn.Linear(in_features=P['feat_dim'], out_features=P['num_classes'], bias=True)
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
        elif self.arch == 'vit-l':
            self.feature_extractor = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336")
            for p in self.feature_extractor.parameters():
                p.requires_grad = True 
            self.linear_classifier = nn.Linear(P['feat_dim'], P['num_classes'])
        elif self.arch == 'resnet50_clip':
            self.feature_extractor = load_clip_to_cpu("RN50").visual.float() 
            for p in self.feature_extractor.parameters():
                p.requires_grad = True 
            self.linear_classifier = nn.Linear(1024, P['num_classes'])       
        else: 
            raise "Model is not implemented!!!"

    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def get_cam(self, x):
        feats = self.feature_extractor(x)
        CAM = F.conv2d(feats, self.linear_classifier.weight.unsqueeze(-1).unsqueeze(-1))
        return CAM

    def foward_linearinit(self, x):
        x = self.linear_classifier(x)
        return x
        
    def forward(self, x):
        if self.arch == 'resnet50' or self.arch == 'resnet101':
            feats = self.feature_extractor(x)
            pooled_feats = self.avgpool(feats)
            logits = self.linear_classifier(pooled_feats)
        elif self.arch == 'convnext_xlarge_1k':
            logits = self.feature_extractor(x) 
        elif self.arch == 'vit-l':
            feats = self.feature_extractor(x).image_embeds
            logits = self.linear_classifier(feats)
        elif self.arch == "resnet50_clip": 
            feats = self.feature_extractor(x)
            logits = self.linear_classifier(feats)
        else:
            logits = self.feature_extractor(x)
        return logits
    
class KFunction(nn.Module):
    def __init__(self, w, b):
        super(KFunction, self).__init__()
        self.w = w
        self.b = b

    def forward(self, p):
        return 1 / (1 + torch.exp(-(self.w * p + self.b)))

def VFunction(p, mu, sigma):
    return torch.exp(-0.5 * ((p - mu) / sigma) ** 2)

def PFunction(p, gamma):
    return gamma * torch.exp(-0.5 * ((p - 0.5) / 0.5) ** 2)

### Define Graph Convolutional Network

import math
import numpy as np
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features).to(torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features).to(torch.float16))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        adj = adj.to(torch.float16)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):

    def __init__(self, classnames, clip_model, P):
        super().__init__()
        with open(classnames, "r") as f:
            classnames = f.read().strip().split("\n")
        n_cls = len(classnames)
        n_ctx = P['n_ctx']
        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        # print(clip_imsize)
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # use given words to initialize context vectors
        ctx_init = P['ctx_init'].replace("_", " ")
        assert (n_ctx == len(ctx_init.split(" ")))
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1:1 + n_ctx, :]
        prompt_prefix = ctx_init

        self.ctx = nn.Parameter(ctx_vectors)  # type: ignore
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix",
                             embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("token_middle", embedding[:, 1:(1 + n_ctx), :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],  # type: ignore
            dim=1,
        )
        return prompts

def get_relation(P):
    relation = torch.Tensor(np.load(P['relation']))
        
    _ ,max_idx = torch.topk(relation, P['sparse_topk'])
    mask = torch.ones_like(relation).type(torch.bool)
    for i, idx in enumerate(max_idx):
        mask[i][idx] = 0
    relation[mask] = 0
    sparse_mask = mask
    dialog = torch.eye(P['num_classes']).type(torch.bool)
    relation[dialog] = 0
    relation = relation / torch.sum(relation, dim=1).reshape(-1, 1) * P['reweight_p']
    relation[dialog] = 1 - P['reweight_p']

    gcn_relation = relation.clone()
    assert(gcn_relation.requires_grad == False)
    relation = torch.exp(relation/P['T']) / torch.sum(torch.exp(relation/P['T']), dim=1).reshape(-1,1)
    relation[sparse_mask] = 0
    relation = relation / torch.sum(relation, dim=1).reshape(-1, 1) 
    
    return relation, gcn_relation
    

               
### Pseudo Label Generator using OpenAI's CLIP model, including the text encoder and prompt learner (can use learnable tokens)
class PseudoLabelGeneratorOpenAI(torch.nn.Module):
    def __init__(self, P):
        super(PseudoLabelGeneratorOpenAI, self).__init__()
        self.P = P
        self.clip = load_clip_to_cpu(P['clip_model']) 
        self.pseudo_image_encoder = self.clip.visual 
        self.pseudo_text_encoder = TextEncoder(self.clip)
        self.prompt_learner = PromptLearner(P['classnames'], self.clip, P) # For learnable prompts, not supported in this version
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts 
        self.logit_scale = self.clip.logit_scale
        self.dtype = self.clip.dtype 
        # print("CLIP dtype: ", self.dtype)
        
        # frozen the clip model
        for param in self.pseudo_image_encoder.parameters():
            param.requires_grad = False
        for param in self.pseudo_text_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_learner.parameters():
            param.requires_grad = False
        
        relation, gcn_relation = get_relation(P)
        self.relation = relation 
        self.gcn_relation = gcn_relation
        
        self.gc1 = GraphConvolution(512, 1024)
        self.gc2 = GraphConvolution(1024, 1024)
        self.gc3 = GraphConvolution(1024, 512)
        self.relu = nn.LeakyReLU(0.2)
        self.relu2 = nn.LeakyReLU(0.2) 
    
    def gcn_forward(self, text_features):
        text_features = self.gc1(text_features, self.gcn_relation.to(self.P['device']))
        text_features = self.relu(text_features)
        text_features = self.gc2(text_features, self.gcn_relation.to(self.P['device']))
        text_features = self.relu2(text_features)
        text_features = self.gc3(text_features, self.gcn_relation.to(self.P['device']))
        return text_features
    def generate(self, x: torch.Tensor):
        # CLIP forward
        image_features = self.pseudo_image_encoder(x.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
           
        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()
        text_features = self.pseudo_text_encoder(prompts, tokenized_prompts)
        
        # Add noise based on label-to-label relation to text features using Graph Convolution layers
        identity = text_features
        text_features = self.gcn_forward(text_features)
        text_features += identity

        text_features = text_features / text_features.norm(dim=-1,keepdim=True)
        pseudo_label_logits = image_features @ text_features.t()
        
        return pseudo_label_logits  
    def forward(self, x): 
        # CLIP forward
        image_features = self.pseudo_image_encoder(x.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)

        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()
        text_features = self.pseudo_text_encoder(prompts, tokenized_prompts)
    
        # Add noise based on label-to-label relation to text features using Graph Convolution layers
        identity = text_features
        text_features = self.gcn_forward(text_features)
        text_features += identity

        text_features = text_features / text_features.norm(dim=-1,keepdim=True)

        logit_scale = self.logit_scale.exp()
        pseudo_label_logits = logit_scale * image_features @ text_features.t()
        
        return pseudo_label_logits  
    
## Pseudo Label Generator using Open CLIP's CLIP model (numerous pretrained weights)
class PseudoLabelGeneratorOpenCLIP(torch.nn.Module):
    def __init__(self, P):
        super(PseudoLabelGeneratorOpenCLIP, self).__init__()
        self.P = P 
        self.model = open_clip.create_model(P['clip_model'], pretrained=P['clip_weight'], precision="fp16") 
        for param in self.model.parameters():
            param.requires_grad = False
        with open(P['classnames'], "r") as f:
            classnames = f.read().strip().split("\n")
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [P['ctx_init'] + " " + name + "." for name in classnames]
        self.tokenizer = open_clip.get_tokenizer(P['clip_model'].replace('/', '-'))
        self.text = self.tokenizer(prompts)
        self.register_buffer("t", self.text)
        relation, gcn_relation = get_relation(P)
        self.relation = relation
        self.gcn_relation = gcn_relation
        
        self.gc1 = GraphConvolution(768, 1024)
        self.gc2 = GraphConvolution(1024, 1024)
        self.gc3 = GraphConvolution(1024, 768)

        self.relu = nn.LeakyReLU(0.2)
        self.relu2 = nn.LeakyReLU(0.2) 
    
    def gcn_forward(self, text_features):
        text_features = self.gc1(text_features, self.gcn_relation.to(self.P['device']))
        text_features = self.relu(text_features)
        text_features = self.gc2(text_features, self.gcn_relation.to(self.P['device']))
        text_features = self.relu2(text_features)
        text_features = self.gc3(text_features, self.gcn_relation.to(self.P['device']))
        return text_features 
    def generate(self, x):
        image_features = self.model.encode_image(x)
        text_features = self.model.encode_text(self.text.to(self.P['device']))
                
        identity = text_features
        text_features = self.gcn_forward(text_features)
        text_features += identity
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        pseudo_label_logits = image_features @ text_features.t()
        return pseudo_label_logits
    def forward(self, x): 
        # CLIP forward
        image_features = self.pseudo_image_encoder(x.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)

        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()
        text_features = self.pseudo_text_encoder(prompts, tokenized_prompts)
    
        # Add noise based on label-to-label relation to text features using Graph Convolution layers
        identity = text_features
        text_features = self.gcn_forward(text_features)
        text_features += identity

        text_features = text_features / text_features.norm(dim=-1,keepdim=True)

        logit_scale = self.logit_scale.exp()
        pseudo_label_logits = logit_scale * image_features @ text_features.t()
        
        return pseudo_label_logits 
        
            
        
        