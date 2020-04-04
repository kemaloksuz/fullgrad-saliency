#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" 
    Implement a simpler FullGrad-like saliency algorithm.

    Instead of exactly computing bias-gradients, we only
    extract gradients w.r.t. biases, which are simply
    gradients of intermediate spatial features *before* ReLU.
    The rest of the algorithm including post-processing
    and the aggregation is the same.

    Note: this algorithm is only provided for convenience and
    performance may not be match that of FullGrad for different
    post-processing functions.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose


class SimpleFullGrad():
    """
    Compute simple FullGrad saliency map 
    """

    def __init__(self, model):
        self.model = model

    def _getGradients(self, image, target_class=None):
        """
        Compute intermediate gradients for an image
        """

        image = image.requires_grad_()
        out, features = self.model.getFeatures(image)

        if target_class is None:
            target_class = out.data.max(1, keepdim=True)[1]

        agg = 0
        for i in range(image.size(0)):
            agg += out[i,target_class[i]]

        self.model.zero_grad()
        # Gradients w.r.t. input and features
        gradients = torch.autograd.grad(outputs = agg, inputs = features, only_inputs=True)

        # First element in the feature list is the image
        input_gradient = gradients[0]

        # Loop through remaining gradients
        intermediate_gradient = []
        for i in range(1, len(gradients)):
            intermediate_gradient.append(gradients[i]) 
        
        return input_gradient, intermediate_gradient

    def _postProcess(self, input, eps=1e-6):
        # Absolute value
        input = abs(input)

        # Rescale operations to ensure gradients lie between 0 and 1
        input = input - input.min()
        input = input / (input.max() + eps)
        return input

    def saliency(self, image, target_class=None):
        #Simple FullGrad saliency
        
        self.model.eval()
        input_grad, intermed_grad = self._getGradients(image, target_class=target_class)
        
        im_size = image.size()

        # Input-gradient * image
        grd = input_grad * image
        gradient = self._postProcess(grd).sum(1, keepdim=True)
        cam = gradient

        # Intermediate-gradients
        for i in range(len(intermed_grad)):
            if len(intermed_grad[i].size()) == len(im_size):
                temp = self._postProcess(intermed_grad[i])
                if len(im_size) == 3:
                    gradient = F.interpolate(temp, size=im_size[2], mode = 'bilinear', align_corners=False) 
                elif len(im_size) == 4:
                    gradient = F.interpolate(temp, size=(im_size[2], im_size[3]), mode = 'bilinear', align_corners=False) 
                cam += gradient.sum(1, keepdim=True)

        return cam
        
