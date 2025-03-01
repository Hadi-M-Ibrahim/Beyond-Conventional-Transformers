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

class TeacherOutputAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # (student index : teacher index)
        self.mapping = {
            1: 17,  # Enlarged Cardiomediastinum
            2: 10,  # Cardiomegaly
            3: 16,  # Lung Opacity
            4: 14,  # Lung Lesion
            5: 4,   # Edema
            6: 1,   # Consolidation
            7: 8,   # Pneumonia
            8: 0,   # Atelectasis
            9: 3,   # Pneumothorax
            10: 7,  # Pleural Effusion
            11: 9,  # Pleural Other
            12: 15, # Fracture
        }
    
    def forward(self, teacher_logits):
        """
        teacher_logits: Tensor of shape (batch_size, 18)
        Returns:
            mapped_logits: Tensor of shape (batch_size, 14) with desired ordering:
            [No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion,
             Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion,
             Pleural Other, Fracture, Support Devices]
        """
        batch_size = teacher_logits.shape[0]
        mapped = torch.zeros((batch_size, 14), device=teacher_logits.device)
        
        # Compute "No Finding"
        no_finding = torch.prod(1 - torch.sigmoid(teacher_logits), dim=1)
        mapped[:, 0] = no_finding
        
        for student_idx, teacher_idx in self. mapping.items():
            mapped[:, student_idx] = teacher_logits[:, teacher_idx]
        
        # "Support Devices" set to 0
        mapped[:, 13] = 0.0
        
        return mapped
        
def load_custom_teacher_model(teacher_path):
    teacher_model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    teacher_model.eval()  # Ensure the teacher is in eval mode.
    
    # Wrap the teacher model with the adapter that maps its output to 14 classes.
    adapter = TeacherOutputAdapter()
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

    metric_logger.synchronize_between_processes()
    print('* Accuracy: {acc:.3f} loss: {loss:.3f} f1_micro: {f1:.3f} auc_micro: {auc:.3f}'
          .format(acc=metric_logger.accuracy.global_avg,
                  loss=metric_logger.loss.global_avg,
                  f1=f1_micro,
                  auc=auc_micro))

    metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    metrics['f1_micro'] = f1_micro
    metrics['auc_micro'] = auc_micro
    return metrics
