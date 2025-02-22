#!/usr/bin/env python
import argparse
import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            # If outputs are logits, convert them to probabilities
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute AUC for each class
    auc_scores = []
    for i in range(all_targets.shape[1]):
        try:
            auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
        except ValueError:
            auc = float('nan')
        auc_scores.append(auc)
    print("AUC per class:")
    for idx, auc in enumerate(auc_scores):
        if np.isnan(auc):
            print(f"  Class {idx}: NA")
        else:
            print(f"  Class {idx}: {auc:.4f}")
    print("Mean AUC:", np.nanmean(auc_scores))

def main():
    parser = argparse.ArgumentParser(description="Evaluate DenseNet121 on CheXpert validation set")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the CheXpert dataset directory")
    parser.add_argument("--checkpoint", type=str, default="densenet121_chexpert.pth",
                        help="Path to the DenseNet121 CheXpert checkpoint")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load DenseNet121 model from TorchXrayVision without preloaded weights.
    model = xrv.models.DenseNet(weights=None)
    # Load your checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    
    # Define a transform for CheXpert images.
    # CheXpert images are often grayscale; here we convert them to 3 channels.
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Create the CheXpert validation dataset.
    test_dataset = xrv.datasets.CheXpertDataset(
        imgpath=args.data_path,
        views=["AP"],         # Change if you want to include other view types
        transform=transform,
        full_path=True,
        split="valid"         # Use "valid" for validation split (or "test" if available)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
