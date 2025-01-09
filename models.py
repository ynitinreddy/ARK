import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torchvision.models as models

import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, PatchEmbed
from functools import partial
import simmim

def build_classification_model(args):
    model = None
    print("Creating model...")
    if args.pretrained_weights is None or args.pretrained_weights =='':
        print('Loading pretrained {} weights for {} from timm.'.format(args.init, args.model_name))
        if args.model_name.lower() == "vit_base":
            if args.init.lower() =="random":
                model = VisionTransformer(num_classes=args.num_class,
                        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                # model = timm.create_model('vit_base_patch16_224', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('vit_base_patch16_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                model = timm.create_model('vit_base_patch16_224_in21k', num_classes=args.num_class, pretrained=True)  
            elif args.init.lower() =="sam":
                model = timm.create_model('vit_base_patch16_224_sam', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="dino":
                model = VisionTransformer(num_classes=args.num_class,
                        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                #model = timm.create_model('vit_base_patch16_224_dino', num_classes=args.num_class, pretrained=True) #not available in current timm version
                url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
                state_dict = torch.hub.load_state_dict_from_url(url=url)
                model.load_state_dict(state_dict, strict=False)
            elif args.init.lower() =="deit":
                model = timm.create_model('deit_base_patch16_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="beit":
                model = timm.create_model('beit_base_patch16_224', num_classes=args.num_class, pretrained=True)

        elif args.model_name.lower() == "vit_small":
            if args.init.lower() =="random":
                model = timm.create_model('vit_small_patch16_224', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('vit_small_patch16_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                model = timm.create_model('vit_small_patch16_224_in21k', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="dino":
                #model = timm.create_model('vit_small_patch16_224_dino', num_classes=args.num_class, pretrained=True)
                model = VisionTransformer(num_classes=args.num_class,
                    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
                state_dict = torch.hub.load_state_dict_from_url(url=url)
                model.load_state_dict(state_dict, strict=False)
            elif args.init.lower() =="deit":
                model = timm.create_model('deit_small_patch16_224', num_classes=args.num_class, pretrained=True)           

        elif args.model_name.lower() == "swin_base": 
            if args.init.lower() =="random":
                model = timm.create_model('swin_base_patch4_window7_224_in22k', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_21kto1k":
                model = timm.create_model('swin_base_patch4_window7_224', num_classes=args.num_class, pretrained=True)
            elif args.init.lower() =="imagenet_21k":
                model = timm.create_model('swin_base_patch4_window7_224_in22k', num_classes=args.num_class, pretrained=True)
            
        elif args.model_name.lower() == "swin_tiny": 
            if args.init.lower() =="random":
                model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class, pretrained=False)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class, pretrained=True)
        
    elif os.path.isfile(args.pretrained_weights):
        print("Creating model from pretrained weights: "+ args.pretrained_weights)
        if args.model_name.lower() == "vit_base":
            if args.init.lower() == "simmim":
                model = simmim.create_model(args)
            else:
                model = VisionTransformer(num_classes=args.num_class,
                        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
                model.default_cfg = _cfg()
                load_pretrained_weights(model, args.init.lower(), args.pretrained_weights)
            
        elif args.model_name.lower() == "vit_small":
            model = VisionTransformer(num_classes=args.num_class,
                    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6))
            model.default_cfg = _cfg()
            load_pretrained_weights(model, args.init.lower(), args.pretrained_weights)  
            
        elif args.model_name.lower() == "swin_base":
            if args.init.lower() == "simmim":
                model = simmim.create_model(args)
            elif args.init.lower() =="imagenet_1k":
                model = timm.create_model('swin_base_patch4_window7_224', num_classes=args.num_class, checkpoint_path=args.pretrained_weights)
            elif args.init.lower() == "ark":
                model = timm.create_model('swin_base_patch4_window7_224', num_classes=args.num_class, pretrained=False)
                load_pretrained_weights(model, args.init.lower(), args.pretrained_weights)
                
        elif args.model_name.lower() == "swin_tiny": 
            model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class)
            load_pretrained_weights(model, args.init.lower(), args.pretrained_weights)
          
    if model is None:
        print("Not provide {} pretrained weights for {}.".format(args.init, args.model_name))
        raise Exception("Please provide correct parameters to load the model!")
    return model  
    
def ClassificationNet(args):
    if args.model_name.lower() == "vit_base":
        model = timm.create_model('vit_base_patch16_224', num_classes=args.num_class)
    elif args.model_name.lower() == "vit_small":
        model = timm.create_model('vit_small_patch16_224', num_classes=args.num_class)  
    elif args.model_name.lower() == "swin_base": 
        model = timm.create_model('swin_base_patch4_window7_224', num_classes=args.num_class)
    elif args.model_name.lower() == "swin_tiny": 
        model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=args.num_class)
    return model

def load_pretrained_weights(model, init, pretrained_weights):
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    if init =="dino":
        checkpoint_key = "teacher"
        if checkpoint_key is not None and checkpoint_key in checkpoint:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = checkpoint[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    elif init =="moco_v3":
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    elif init == "moby":
        state_dict = checkpoint['model']
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'encoder.' in k}
    elif init == "mae":
        state_dict = checkpoint['model']   
    elif init == "ark":
        state_dict = checkpoint
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
          if k in state_dict:
              print(f"Removing key {k} from pretrained checkpoint")
              del state_dict[k]   
    else:
        print("Trying to load the checkpoint for {} at {}, but we cannot guarantee the success.".format(init, pretrained_weights))
        
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded with msg: {}'.format(msg))
    return model

def save_checkpoint(state,filename='model'):

    torch.save( state,filename + '.pth.tar')


