import random
import numpy as np
import torch
import torch.nn as nn

from torchvision.transforms.v2 import *
from torchvision.tv_tensors import Mask, Image

from data.augmentations import *


class ShuffleBeforeImages(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, sample):
        if not isinstance(sample, dict):
            return sample
        
        if 'before' not in sample:
            return sample
        
        random.shuffle(sample['before'])
        
        assert not isinstance(sample['label'], list), "ShuffleBeforeImages not supported for continuous labels"

        return sample

class CustomJitter(Transform):

    def __init__(self, brightness=0.2, contrast=0.2):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
    
    def transform(self, inpt, params):
        if isinstance(inpt, Mask):
            return inpt

        if self.brightness > 0:
            factor = (1-self.brightness/2) + random.random() * self.brightness
            inpt = inpt * factor
        
        if self.contrast > 0:
            mean = inpt.mean(dim=(1, 2), keepdim=True)
            factor = (1-self.contrast/2) + random.random() * self.contrast
            inpt = (inpt - mean) * factor + mean

        return inpt

def augmentation_from_dict(conf):
    return Compose([eval(t['name'])(**t['args']) for t in conf])
