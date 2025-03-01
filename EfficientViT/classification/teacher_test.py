import os
import argparse
import torch
import math
import sys
import random
import json
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
import torchxrayvision as xrv
import utils
import numpy as np

# -----------------------------------------------------------------------------
# Use the evaluate function
# -----------------------------------------------------------------------------
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
    print('* Accuracy: {acc:.3f} loss: {loss:.3f} f1_micro: {f1:.3f} auc_micro: {auc_micro:.3f}'
      .format(acc=metric_logger.accuracy.global_avg,
              loss=metric_logger.loss.global_avg,
              f1=f1_micro,
              auc_micro=auc_micro))

    metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    metrics['f1_micro'] = f1_micro
    metrics['auc_micro'] = auc_micro
    return metrics

# -----------------------------------------------------------------------------
# CheXpert Dataset Definition
# -----------------------------------------------------------------------------
class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
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
                self.samples.append((image_path, torch.tensor(label, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = default_loader(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# -----------------------------------------------------------------------------
# TeacherOutputAdapter & Teacher Model Loader
# -----------------------------------------------------------------------------
class TeacherOutputAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Mapping: (student index : teacher index)
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
        Args:
            teacher_logits: Tensor of shape (batch_size, 18)
        Returns:
            mapped_logits: Tensor of shape (batch_size, 14)
        """
        batch_size = teacher_logits.shape[0]
        mapped = torch.zeros((batch_size, 14), device=teacher_logits.device)
        # Compute "No Finding"
        no_finding = torch.prod(1 - torch.sigmoid(teacher_logits), dim=1)
        mapped[:, 0] = no_finding
        
        for student_idx, teacher_idx in self.mapping.items():
            mapped[:, student_idx] = teacher_logits[:, teacher_idx]
        
        # "Support Devices" set to 0.
        mapped[:, 13] = 0.0
        return mapped

def load_custom_teacher_model(teacher_path):
    teacher_model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    teacher_model.eval()  # Ensure the teacher is in eval mode.
    
    adapter = TeacherOutputAdapter()
    teacher_model = torch.nn.Sequential(teacher_model, adapter)
    return teacher_model

# -----------------------------------------------------------------------------
# Main function to test the teacher model on CheXpert test dataset.
# -----------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    teacher_model = load_custom_teacher_model(None)
    teacher_model.to(device)
    
    test_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).float()),
            transforms.Lambda(lambda x: x.mean(dim=-1) if x.ndim == 3 and x.shape[-1] == 3 else x),
            transforms.Lambda(lambda x: x * (2048.0 / 255.0) - 1024.0),
            transforms.Lambda(lambda x: x.unsqueeze(0).repeat(3, 1, 1) if x.ndim == 2 else x),
            transforms.Grayscale(1)
    ])
    
    test_dataset = CheXpertDataset(root=args.data_path, train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    metrics = evaluate(test_loader, teacher_model, device)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Teacher Model on CheXpert Dataset and Print AUC")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the root directory of the CheXpert dataset (should contain valid.csv and image folders)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for DataLoader")
    parser.add_argument("--num-workers", type=int, default=26, help="Number of DataLoader workers")
    args = parser.parse_args()
    main(args)
