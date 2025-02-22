import math
import sys
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.utils import ModelEma
from losses import DistillationLoss
import utils
from sklearn.metrics import precision_recall_curve
import torchvision.models as models

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import accuracy_score


def load_custom_teacher_model(teacher_path):
    teacher_model = models.densenet121(pretrained=False, num_classes=14)
    
    checkpoint = torch.load(teacher_path, map_location="cpu")
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    
    state_dict = {k: v for k, v in checkpoint.items() if k in teacher_model.state_dict()}
    missing_keys, unexpected_keys = teacher_model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    return teacher_model

def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
                    clip_grad=0, clip_mode='norm', model_ema=None, mixup_fn=None, 
                    set_training_mode=True, set_bn_eval=False):
    model.train(set_training_mode)
    if set_bn_eval:
        set_bn_state(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples, targets = samples.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)
        
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode, parameters=model.parameters())
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, val_loader):
    criterion = torch.nn.BCEWithLogitsLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    
    # Step 1: Collect predictions on the validation set (on CPU)
    all_targets, all_outputs = [], []
    for images, targets in val_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        all_targets.append(targets.cpu())
        all_outputs.append(outputs.cpu())
    
    all_targets = torch.cat(all_targets)
    all_outputs = torch.cat(all_outputs)
    all_probs = torch.sigmoid(all_outputs)  # Remains on CPU
    
    # Step 2: Compute best thresholds per class (on CPU)
    thresholds_list = []
    for i in range(all_probs.shape[1]):
        precision, recall, thres = precision_recall_curve(all_targets[:, i], all_probs[:, i])
        best_thresh = 0.5 
        thresholds_list.append(best_thresh)
    thresholds_cpu = torch.tensor(thresholds_list)
    
    predicted = (all_probs > thresholds_cpu.unsqueeze(0)).float()
    y_true = all_targets.numpy()
    y_pred = predicted.numpy()
    y_scores = all_probs.numpy()
    
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    try:
        auc_micro = roc_auc_score(y_true, y_scores, average='micro', multi_class='ovr')
    except ValueError:
        auc_micro = float('nan')
        thresholds = thresholds_cpu.to(device)
    
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        
        probs = torch.sigmoid(output)
        preds = (probs > thresholds.unsqueeze(0)).float()  # thresholds now on same device
        
        correct = preds.eq(target).sum().item()
        total = target.numel()
        accuracy = correct / total
        
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['accuracy'].update(accuracy, n=batch_size)
    
    metric_logger.synchronize_between_processes()
    print('* Accuracy: {accuracy.global_avg:.3f} loss: {losses.global_avg:.3f} F1 (micro): {f1:.3f} AUC (micro): {auc:.3f}'
          .format(accuracy=metric_logger.accuracy,
                  losses=metric_logger.loss,
                  f1=f1_micro,
                  auc=auc_micro))
    
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['f1_micro'] = f1_micro
    results['auc_micro'] = auc_micro
    return results

