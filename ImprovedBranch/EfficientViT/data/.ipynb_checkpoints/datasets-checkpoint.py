'''
Build trainining/testing datasets
'''
import csv
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
    CLASS_NAMES = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

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
        self.class_names = self.CLASS_NAMES
        self.nb_classes = len(self.class_names)

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

                self.samples.append((image_path, torch.tensor(label, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = default_loader(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class NIHChestXrayDataset(torch.utils.data.Dataset):
    LABELS = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia',
    ]

    def __init__(self, root, train=True, transform=None,
                 csv_filename='Data_Entry_2017.csv'):
        self.root = root
        self.transform = transform
        self.train = train
        self.csv_path = os.path.join(root, csv_filename)
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f'NIH metadata not found: {self.csv_path}')

        self.label_to_index = {label: idx for idx, label in enumerate(self.LABELS)}
        self.image_map = self._index_images(root)
        self.samples = self._load_samples()
        self.nb_classes = len(self.LABELS)

    def _index_images(self, root):
        image_map = {}
        image_dirs = []
        for entry in sorted(os.listdir(root)):
            candidate = os.path.join(root, entry)
            if entry.startswith('images_') and os.path.isdir(candidate):
                nested = os.path.join(candidate, 'images')
                image_dirs.append(nested if os.path.isdir(nested) else candidate)
        if not image_dirs:
            fallback = os.path.join(root, 'images')
            if os.path.isdir(fallback):
                image_dirs.append(fallback)
        if not image_dirs:
            raise FileNotFoundError(f'No NIH image folders found under: {root}')

        for directory in image_dirs:
            for filename in os.listdir(directory):
                lower = filename.lower()
                if lower.endswith('.png') or lower.endswith('.jpg') or lower.endswith('.jpeg'):
                    image_map[filename] = os.path.join(directory, filename)

        if not image_map:
            raise FileNotFoundError(f'No NIH images found after scanning: {image_dirs}')

        return image_map

    def _load_samples(self):
        samples = []
        with open(self.csv_path, 'r') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_name = row['Image Index']
                patient_id = row.get('Patient ID')
                if not patient_id:
                    continue
                try:
                    patient_id = int(patient_id)
                except ValueError:
                    continue

                is_train_sample = (patient_id % 5) != 0
                if self.train != is_train_sample:
                    continue

                image_path = self.image_map.get(image_name)
                if image_path is None:
                    raise FileNotFoundError(f'NIH image missing for index: {image_name}')

                labels = torch.zeros(len(self.LABELS), dtype=torch.float32)
                for label in row['Finding Labels'].split('|'):
                    label = label.strip()
                    if not label or label == 'No Finding':
                        continue
                    idx = self.label_to_index.get(label)
                    if idx is not None:
                        labels[idx] = 1.0
                samples.append((image_path, labels))

        if not samples:
            raise ValueError('NIH dataset split produced no samples. Check data path and split logic.')

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, labels = self.samples[idx]
        image = default_loader(image_path)
        if self.transform:
            image = self.transform(image)
        return image, labels


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
                                  train=is_train, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'NIH':
        dataset = NIHChestXrayDataset(root=args.data_path,
                                      train=is_train, transform=transform)
        nb_classes = dataset.nb_classes
    else:
        raise ValueError(f'Unsupported dataset: {args.data_set}')
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
