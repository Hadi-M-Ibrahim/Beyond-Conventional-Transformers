# --------------------------------------------------------
# CheXpert Evaluation and Logging Script for Multiple Checkpoints
# --------------------------------------------------------
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

from timm.models import create_model

from data.datasets import build_dataset
from engine import evaluate

from model import build
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

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


'''
Build training/testing datasets for CheXpert
'''
class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        """
        Args:
            root (str): Root directory of the CheXpert dataset.
            train (bool): If True, use 'train.csv'; otherwise, use 'test.csv'.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root = root
        self.transform = transform
        self.train = train
        self.samples = []

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
            
def build_transform(is_train=False, args=None):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    size = int((256 / 224) * args.input_size)
    t.append(
    # to maintain same ratio w.r.t. 224 images
    transforms.Resize(size, interpolation=3),
    )
    t.append(transforms.CenterCrop(args.input_size))
    
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

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
            'n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'evaluated_at': datetime.datetime.now().isoformat()
        }
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
