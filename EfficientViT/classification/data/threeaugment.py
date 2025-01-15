"""
3Augment implementation from (https://github.com/facebookresearch/deit/blob/main/augment.py)
Data-augmentation (DA) based on dino DA (https://github.com/facebookresearch/dino)
and timm DA(https://github.com/rwightman/pytorch-image-models)
Can be called by adding "--ThreeAugment" to the command line


"""
import torch
from torchvision import transforms


from timm.data.transforms import str_to_pil_interp, RandomResizedCropAndInterpolation, ToNumpy, ToTensor


import numpy as np
from torchvision import datasets, transforms
import random






from PIL import ImageFilter, ImageOps
import torchvision.transforms.functional as TF

from torchvision.transforms import v2



class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.3, radius_max=.5):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max


    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img


        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.3):
        self.p = p


    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img,5)
        else:
            return img


class ElasticTransform(object):
    def __init__(self, p=.99):
        self.p = p
        self.transf = v2.ElasticTransform(100.0)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
       
class ColorJitter(object):
    def __init__(self, p=1):
        self.p = p
        self.transf = v2.ColorJitter(0.4, 0.4, 0.4, 0.1)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
        
class RandomPerspective(object):
    def __init__(self, p=.5):
        self.p = p
        self.transf = v2.RandomPerspective()
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img

class RandomRotation(object):
    def __init__(self, p=.8):
        self.p = p
        self.transf = v2.RandomRotation((-5,5))
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img      

class JPEG(object):
    def __init__(self, p=.5):
        self.p = p
        self.transf = v2.JPEG(random.randint(1, 100))
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img  
            
class VerticalFlip(object):
    def __init__(self, p=.3):
        self.p = p
        self.transf = v2.RandomVerticalFlip(p=1)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img      
                
class HorizontalFlip(object):
    def __init__(self, p=.3):
        self.p = p
        self.transf = v2.RandomHorizontalFlip(p=1)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
        
def new_data_aug_generator(is_train, args = None):
    img_size = 224
    mean, std = [0.5031732320785522,0.5031732320785522 , 0.5031732320785522], [ 0.28851956129074097,0.28851956129074097,0.28851956129074097]
    primary_tfl = []
    scale=(0.08, 1.0)
    interpolation='bicubic'


    if is_train:
        primary_tfl = [
            transforms.Resize(img_size, interpolation=3),
        ]


        secondary_tfl = [ElasticTransform(p=.99),
                        Solarization(p=.3),
                        GaussianBlur(p=.3),
                        ColorJitter(p=1),
                        RandomPerspective(p=.5),
                        RandomRotation(p=.8),
                        JPEG(p=.5),
                        VerticalFlip(p=.3),
                        HorizontalFlip(p=.3)
                        ]


        final_tfl = [
                transforms.Grayscale(3),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ]
        return transforms.Compose(primary_tfl+secondary_tfl+final_tfl)
    else:
        tf1 = [  
            transforms.Resize(img_size, interpolation=3),
            transforms.Grayscale(3),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
        return transforms.Compose(tf1)



