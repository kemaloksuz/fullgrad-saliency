#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Compute saliency maps of images from dataset folder 
    and dump them in a results folder """

import torch
import subprocess
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils, models
import numpy as np
import os

from fullgrad import FullGrad
from simple_fullgrad import SimpleFullGrad
from vgg_imagenet import *
from misc_functions import *
import pdb
from mmcv.parallel import MMDataParallel
import os.path as osp
from collections import OrderedDict
def load_checkpoint(fpath):
    r"""Loads checkpoint.
    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.
    Args:
        fpath (str): path to checkpoint.
    Returns:
        dict
    Examples::  
        >>> from torchreid.utils import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint

def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.
    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".
    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.
    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.
        
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    
    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path))
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(weight_path))
        if len(discarded_layers) > 0:
            print('** The following layers are discarded '
                  'due to unmatched keys or layer size: {}'.format(discarded_layers))
    return model        

def init_model(path, num_classes, num_gpus):
    model = models.vgg16_bn(pretrained = False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    #model = MMDataParallel(model, device_ids=range(num_gpus)).cuda()
    model=load_pretrained_weights(model, path)
    #model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), strict=False))
    return model
# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'dataset/'

batch_size = 1
num_classes = 80
num_gpus = 1
model_path = "/Users/Kemal/Desktop/vgg16bn_fromscratch_90_best.pth"

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Dataset loader for sample images
sample_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(dataset, transform=transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
                   ])),
    batch_size= batch_size, shuffle=False)

unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225])


#1. Buraya bizim modelin yuklenmesi lazim
model = init_model(model_path, num_classes, num_gpus)
#model2 = vgg16_bn()
#print(model)
# for batch_idx, (data, target) in enumerate(sample_loader):
#         data, target = data.to(device).requires_grad_(), target.to(device)
#         input = data.to(device)
#         model.eval()
#         raw_output = model(input)
#         print(raw_output)


# Get raw outputs


for child in model.children():
    if not isinstance(child,torch.nn.modules.pooling.AdaptiveAvgPool2d):
        for i in range(len(child)):
            if isinstance(child[i],torch.nn.modules.activation.ReLU):
                child[i]=torch.nn.ReLU(inplace=False)
# for batch_idx, (data, target) in enumerate(sample_loader):
#         data, target = data.to(device).requires_grad_(), target.to(device)
#         input = data.to(device)
#         model.eval()
#         raw_output = model(input)
#         print(raw_output)
# sys.exit()                
#model = models.resnext50_32x4d(pretrained=True)
# Initialize FullGrad object
fullgrad = FullGrad(model, device)
#simple_fullgrad = SimpleFullGrad(model)

#2. Buraya imagein gt classinin etiketi verilmeli
target_class=torch.tensor([[0]]).to(device)

save_path = PATH + 'results/'

def compute_saliency_and_save():
    for batch_idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        # Compute saliency maps for the input data
        cam = fullgrad.saliency(data, target_class = target_class)
        #cam_simple = simple_fullgrad.saliency(data, target_class = target_class)

        # Save saliency maps
        for i in range(data.size(0)):
            filename = save_path + str( (batch_idx+1) * (i+1)) + str( target_class.numpy())
            #filename_simple = filename + '_simple'

            image = unnormalize(data[i,:,:,:].cpu())
            saliency_map=save_saliency_map(image, cam[i,:,:,:], filename + '.jpg')
            pdb.set_trace()
            #save_saliency_map(image, cam_simple[i,:,:,:], filename_simple + '.jpg')


#---------------------------------------------------------------------------------#

# Create folder to saliency maps
create_folder(save_path)

compute_saliency_and_save()

print('Saliency maps saved.')

#---------------------------------------------------------------------------------#
        
        




