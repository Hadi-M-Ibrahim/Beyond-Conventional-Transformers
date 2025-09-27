"""
Train, eval, teacher-load functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import ModelEma

from losses import DistillationLoss
import utils

import torchvision.models as models

from sklearn.metrics import f1_score, roc_auc_score

import torchxrayvision as xrv

import torchvision.transforms.functional as TF

import numpy as np

from data.datasets import NIHChestXrayDataset

class TeacherOutputAdapter(torch.nn.Module):
    def __init__(self, teacher_indices, num_classes, compute_no_finding=False, zero_indices=None):
        super().__init__()
        self.teacher_indices = teacher_indices
        self.num_classes = num_classes
        self.compute_no_finding = compute_no_finding
        self.zero_indices = set(zero_indices or [])

    def forward(self, teacher_logits):
        batch_size = teacher_logits.shape[0]
        mapped = torch.zeros((batch_size, self.num_classes), device=teacher_logits.device)

        if self.compute_no_finding:
            mapped[:, 0] = torch.prod(1 - torch.sigmoid(teacher_logits), dim=1)

        for target_idx, teacher_idx in enumerate(self.teacher_indices):
            if teacher_idx is None:
                continue
            mapped[:, target_idx] = teacher_logits[:, teacher_idx]

        for idx in self.zero_indices:
            mapped[:, idx] = 0.0

        return mapped


def _build_teacher_indices_for_labels(labels):
    default_pathologies = xrv.datasets.default_pathologies
    teacher_indices = []
    for label in labels:
        name_candidates = [label, label.replace('_', ' ')]
        teacher_idx = None
        for candidate in name_candidates:
            if candidate in default_pathologies:
                teacher_idx = default_pathologies.index(candidate)
                break
        if teacher_idx is None:
            raise ValueError(f"Teacher model does not provide logits for label '{label}'.")
        teacher_indices.append(teacher_idx)
    return teacher_indices


def load_custom_teacher_model(teacher_path, data_set):
    teacher_model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    teacher_model.eval()

    if data_set == 'CHEXPERT':
        chex_teacher_indices = [
            None,  # No Finding handled separately
            17,    # Enlarged Cardiomediastinum
            10,    # Cardiomegaly
            16,    # Lung Opacity
            14,    # Lung Lesion
            4,     # Edema
            1,     # Consolidation
            8,     # Pneumonia
            0,     # Atelectasis
            3,     # Pneumothorax
            7,     # Pleural Effusion (mapped from Effusion)
            9,     # Pleural Other (approx. Pleural Thickening)
            15,    # Fracture
            None,  # Support Devices (not provided by teacher)
        ]
        adapter = TeacherOutputAdapter(
            teacher_indices=chex_teacher_indices,
            num_classes=len(chex_teacher_indices),
            compute_no_finding=True,
            zero_indices={13},
        )
    elif data_set == 'NIH':
        nih_labels = NIHChestXrayDataset.LABELS
        nih_teacher_indices = _build_teacher_indices_for_labels(nih_labels)
        adapter = TeacherOutputAdapter(
            teacher_indices=nih_teacher_indices,
            num_classes=len(nih_teacher_indices),
            compute_no_finding=False,
        )
    else:
        raise ValueError(f"Unsupported dataset for teacher adapter: {data_set}")

    teacher_model = torch.nn.Sequential(teacher_model, adapter)
    return teacher_model


def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    set_bn_eval=False,):
    model.train(set_training_mode)
    if set_bn_eval:
        set_bn_state(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if True:  # with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        is_second_order = getattr(optimizer, 'is_second_order', False)

        loss_scaler(loss, optimizer, clip_grad=clip_grad, 
                    clip_mode=clip_mode,parameters=model.parameters(),
                    create_graph=is_second_order)



        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()

def evaluate(data_loader, model, device):
    criterion = torch.nn.BCEWithLogitsLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for images, target in metric_logger.log_every(data_loader, 10, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)

            probs = torch.sigmoid(output)
            preds = probs > 0.5
            correct = preds.eq(target).sum().item()
            total = target.numel()
            accuracy = correct / total

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['accuracy'].update(accuracy, n=batch_size)
            
            all_preds.append(probs.detach().cpu())
            all_targets.append(target.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    f1_micro = f1_score(all_targets, (all_preds > 0.5).astype(int), average='micro')
    auc_micro = roc_auc_score(all_targets, all_preds, average='micro')
    
    dataset = data_loader.dataset
    if hasattr(dataset, 'class_names'):
        pathology_names = list(dataset.class_names)
    elif hasattr(dataset, 'LABELS'):
        pathology_names = list(dataset.LABELS)
    else:
        pathology_names = [
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

    num_labels = all_targets.shape[1]
    auc_per_label = {}
    for i in range(num_labels):
        try:
            auc_value = roc_auc_score(all_targets[:, i], all_preds[:, i])
        except ValueError:
            auc_value = float('nan')
        auc_per_label[pathology_names[i]] = auc_value

    metric_logger.synchronize_between_processes()
    print('* Accuracy: {acc:.3f} loss: {loss:.3f} f1_micro: {f1:.3f} auc_micro: {auc:.3f}'
          .format(acc=metric_logger.accuracy.global_avg,
                  loss=metric_logger.loss.global_avg,
                  f1=f1_micro,
                  auc=auc_micro))
    
    print("Per-label AUC:")
    for pathology, auc in auc_per_label.items():
        print("{}: {:.3f}".format(pathology, auc))
        
    metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    metrics['f1_micro'] = f1_micro
    metrics['auc_micro'] = auc_micro
    metrics['auc_per_label'] = auc_per_label
    return metrics
    
