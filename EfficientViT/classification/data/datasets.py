'''
Build trainining/testing datasets
'''
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import numpy as np

import random 
from PIL import ImageFilter, ImageOps

try:
    from timm.data import TimmDatasetTar
except ImportError:
    # for higher version of timm
    from timm.data import ImageDataset as TimmDatasetTar
    
class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(
            root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        """
        Args:
            root (str): Root directory of the CheXpert dataset.
            train (bool): Whether to use the training or validation split.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root = root
        self.transform = transform
        self.train = train
        self.samples = []

        csv_file = 'train.csv' if train else 'valid.csv'
        csv_path = os.path.join(root, csv_file)

        with open(csv_path, 'r') as file:
            lines = file.readlines()[1:]
            for line in lines:
                parts = line.strip().split(',')
                
                image_path = os.path.join(root, parts[0])
                
  
                label = []
                for x in parts[5:19]:
                    if x == '-1.0': 

                        label.append(1.0)
                    elif x == '':  
                        label.append(0.0)
                    else:
                        label.append(float(x))

                self.samples.append((image_path, torch.tensor(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = default_loader(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        prefix = 'train' if is_train else 'val'
        data_dir = os.path.join(args.data_path, f'{prefix}.tar')
        if os.path.exists(data_dir):
            dataset = TimmDatasetTar(data_dir, transform=transform)
        else:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNETEE':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 10
    elif args.data_set == 'FLOWERS':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        if is_train:
            dataset = torch.utils.data.ConcatDataset(
                [dataset for _ in range(100)])
        nb_classes = 102
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'CHEXPERT':
        dataset = CheXpertDataset(root=args.data_path,
                                  train=is_train,transform=transform)
    nb_classes = 14
    return dataset, nb_classes


def normalize_hu(image, min_hu=-1024, max_hu=1024):
    """Normalize Hounsfield Units (HU) to [0,1] range."""
    image = (image - min_hu) / (max_hu - min_hu)
    return torch.clamp(image, 0, 1)
    
    return transforms.Compose(transform_list)

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=.5):
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

class ElasticTransform(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.ElasticTransform(100.0)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
            
def build_transform(is_train, args):
    if is_train:
        transform_list = [
            transforms.Resize((args.input_size, args.input_size), interpolation=3),
            ElasticTransform(p=0.6),
            GaussianBlur(p=0.4),     
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0)),
            transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).float()),
            transforms.Lambda(lambda x: x.mean(dim=-1) if x.ndim == 3 and x.shape[-1] == 3 else x),
            transforms.Lambda(lambda x: x * (2048.0 / 255.0) - 1024.0),
            transforms.Lambda(lambda x: x.unsqueeze(0).repeat(3, 1, 1) if x.ndim == 2 else x),
            transforms.Grayscale(1)

        ]
    else:
        transform_list = [
            transforms.Resize((args.input_size, args.input_size), interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).float()),
            transforms.Lambda(lambda x: x.mean(dim=-1) if x.ndim == 3 and x.shape[-1] == 3 else x),
            transforms.Lambda(lambda x: x * (2048.0 / 255.0) - 1024.0),
            transforms.Lambda(lambda x: x.unsqueeze(0).repeat(3, 1, 1) if x.ndim == 2 else x),
            transforms.Grayscale(1)
        ]
    
    return transforms.Compose(transform_list)