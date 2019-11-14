#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Misc helper functions """

import cv2
import numpy as np
import subprocess

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import os.path as osp
from collections import OrderedDict
import pdb

class NormalizeInverse(transforms.Normalize):
    # Undo normalization on images

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())

def create_folder(folder_name):
    try:
        subprocess.call(['mkdir','-p',folder_name])
    except OSError:
        None

def save_saliency_map(image, saliency_map, filename):
    """ 
    Save saliency map on image.
    
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W) 
        filename: string with complete path and file extension

    """

    image = image.data.cpu().numpy()
    saliency_map = saliency_map.data.cpu().numpy()

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map_return = saliency_map.clip(0,1)

    saliency_map = np.uint8(saliency_map_return * 255).transpose(1, 2, 0)
    saliency_map = cv2.resize(saliency_map, (224,224))

    image = np.uint8(image * 255).transpose(1,2,0)
    image = cv2.resize(image, (224, 224))

    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    
    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)

    cv2.imwrite(filename, np.uint8(255 * img_with_heatmap))
    return saliency_map_return

def integral_image_compute(masks,gt_number,h,w, device):
    integral_images= [None] * gt_number
    pad_row=torch.zeros([gt_number,1,w]).type(torch.DoubleTensor).to(device)
    pad_col=torch.zeros([gt_number,h+1,1]).type(torch.DoubleTensor).to(device)
    integral_images=torch.cumsum(torch.cumsum(torch.cat([pad_col,torch.cat([pad_row,masks],dim=1)], dim=2),dim=1), dim=2)
    return integral_images

def integral_image_fetch(mask,bboxes):
    #import pdb
    #pdb.set_trace()
    bboxes[:,[2,3]]+=1
    #Create indices
    TLx=bboxes[:,0].tolist()
    TLy=bboxes[:,1].tolist()
    BRx=bboxes[:,2].tolist()
    BRy=bboxes[:,3].tolist()
    area=mask[BRx,BRy]+mask[TLx,TLy]-mask[TLx,BRy]-mask[BRx,TLy]    
    #area=mask[BRy,BRx]+mask[TLy,TLx]-mask[TLy,BRx]-mask[BRy,TLx]
    return area

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

def init_model(path):
    num_classes = 80
    model = models.vgg16_bn(pretrained = False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    model=load_pretrained_weights(model, path)
    
    for child in model.children():
        if not isinstance(child, torch.nn.modules.pooling.AdaptiveAvgPool2d):
            for i in range(len(child)):
                if isinstance(child[i], torch.nn.modules.activation.ReLU):
                    child[i] = torch.nn.ReLU(inplace=False)

    return model

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

def get_transforms_inverse(img):
    inverter = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    return inverter(img)

def get_transforms(part):
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])

def get_indices(labels, CLASSES, CLASSES_sorted):
    labels = labels - 1 
    names = np.take(CLASSES, labels)
    indices_ = []
    for name in names:
        indices_.append(np.where(CLASSES_sorted == name)[0][0])
    return np.array(indices_)
