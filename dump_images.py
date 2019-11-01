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

def init_model(path, num_classes, num_gpus):
    model = models.vgg16_bn(pretrained = False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    model = MMDataParallel(model, device_ids=range(num_gpus)).cuda()
    model.load_state_dict(torch.load(path))
    return model
# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'dataset/'

batch_size = 1
num_classes = 80
num_gpus = 1
model_path = "models/vgg16bn_fromscratch_90_best.pth"

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
#model = models.resnext50_32x4d(pretrained=True)
pdb.set_trace()
# Initialize FullGrad object
fullgrad = FullGrad(model, device)
simple_fullgrad = SimpleFullGrad(model)

#2. Buraya imagein gt classinin etiketi verilmeli
target_class=torch.tensor([[852]]).to(device)

save_path = PATH + 'results/'

def compute_saliency_and_save():
    for batch_idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        # Compute saliency maps for the input data
        cam = fullgrad.saliency(data, target_class = target_class)
        
        cam_simple = simple_fullgrad.saliency(data, target_class = target_class)

        # Save saliency maps
        for i in range(data.size(0)):
            filename = save_path + str( (batch_idx+1) * (i+1)) + str( target_class.numpy())
            filename_simple = filename + '_simple'

            image = unnormalize(data[i,:,:,:].cpu())
            save_saliency_map(image, cam[i,:,:,:], filename + '.jpg')
            save_saliency_map(image, cam_simple[i,:,:,:], filename_simple + '.jpg')


#---------------------------------------------------------------------------------#

# Create folder to saliency maps
create_folder(save_path)

compute_saliency_and_save()

print('Saliency maps saved.')

#---------------------------------------------------------------------------------#
        
        




