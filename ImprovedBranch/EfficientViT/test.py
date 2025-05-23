"""
CheXpert Evaluation Script
"""
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import re

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from data.samplers import RASampler
from data.datasets import build_dataset
from data.threeaugment import new_data_aug_generator
from engine import train_one_epoch, evaluate, load_custom_teacher_model
from losses import DistillationLoss

from models import build
import utils

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import random
from PIL import ImageFilter

from timm.models import create_model
from engine import evaluate
import utils 

from timm.models.registry import register_model


"""
Build training/testing datasets for CheXpert
"""

class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.samples = []

        # Use 'test.csv' for test eval
        csv_file = 'test.csv'
        csv_path = os.path.join(root, csv_file)
    
        with open(csv_path, 'r') as file:
            lines = file.readlines()[1:] 
            for line in lines:
                parts = line.strip().split(',')
                image_path = os.path.join(root, parts[0])
                label = []
                for x in parts[1:15]:
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
        
def normalize_hu(image, min_hu=-1024, max_hu=1024):
    """Normalize Hounsfield Units (HU) to [0,1] range."""
    image = (image - min_hu) / (max_hu - min_hu)
    return torch.clamp(image, 0, 1)
    
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
    Apply Elastic Transform to the PIL image.
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

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    dataset = CheXpertDataset(root=args.data_path, train=is_train, transform=transform)
    nb_classes = 14
    return dataset, nb_classes

def get_args_parser():
    parser = argparse.ArgumentParser('CheXpert Evaluation and Logging Script', add_help=False)
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the CheXpert dataset root")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Directory to output log files")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for evaluation")
    parser.add_argument("--model", type=str, required=True,
                        help="Name of model architecture")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path or URL to a single checkpoint for evaluation")
    parser.add_argument("--checkpoint-dir", type=str, default="",
                        help="Directory containing .pth checkpoint files to evaluate")
    parser.add_argument("--input_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=10,
                        help="Number of DataLoader workers")
    parser.add_argument("--pin-mem", action='store_true',
                        help="Pin CPU memory in DataLoader")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--dist-eval", action='store_true', default=False,
                        help="Enable distributed evaluation")
    parser.add_argument("--world_size", default=1, type=int,
                        help="Number of distributed processes")
    parser.add_argument("--dist_url", default='env://',
                        help="URL used to set up distributed evaluation")
    return parser

def natural_sort_key(s):
    """
    Sort keys naturally, so that 'checkpoint_2.pth' comes before 'checkpoint_10.pth'.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

def evaluate_checkpoint(checkpoint_path, model, data_loader_val, device):
    print("Evaluating checkpoint:", checkpoint_path)
    if str(checkpoint_path).startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print("Checkpoint load message:", msg)
    model.to(device)
    
    test_stats = evaluate(data_loader_val, model, device)
    print(f"Accuracy on {len(data_loader_val.dataset)} test images: {test_stats['accuracy']:.1f}%")
    print(f"Micro F1 on {len(data_loader_val.dataset)} test images: {test_stats['f1_micro']:.3f}")
    print(f"Micro AUC on {len(data_loader_val.dataset)} test images: {test_stats['auc_micro']:.3f}")
    return test_stats

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    cudnn.benchmark = True

    dataset_val, _ = build_dataset(is_train=False, args=args)
    
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = create_model(
        args.model,
        num_classes=14,
        pretrained=False
    )

    checkpoint_paths = []
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        checkpoint_paths = sorted(ckpt_dir.glob("*.pth"), key=natural_sort_key)
        if not checkpoint_paths:
            print(f"No .pth files found in directory {ckpt_dir}")
            return
    elif args.checkpoint:
        checkpoint_paths = [args.checkpoint]
    else:
        print("No checkpoint or checkpoint directory provided. Exiting.")
        return

    results = {}
    for ckpt in checkpoint_paths:
        test_stats = evaluate_checkpoint(ckpt, model, data_loader_val, device)
        results[str(ckpt)] = {
            'accuracy': test_stats['accuracy'],
            'f1_micro': test_stats['f1_micro'],
            'auc_micro': test_stats['auc_micro'],
            'auc_per_label': test_stats['auc_per_label'],
            'n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'evaluated_at': datetime.datetime.now().isoformat()
        }
        # Log the evaluation result.
        if args.output_dir and utils.is_main_process():
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file = output_dir / "log.txt"
            with log_file.open("a") as f:
                f.write(json.dumps({str(ckpt): results[str(ckpt)]}) + "\n")
    print("Evaluation Results Summary:")
    for ckpt_path, stats in results.items():
        print(f"{ckpt_path}: Accuracy = {stats['accuracy']:.1f}%, "
              f"Micro F1 = {stats['f1_micro']:.3f}, "
              f"Micro AUC = {stats['auc_micro']:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CheXpert Evaluation and Logging Script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
