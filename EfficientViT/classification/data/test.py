from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),  
    transforms.Grayscale(3),  # Ensure images are single-channel
    transforms.ToTensor()     # Convert to tensor
])

dataset = datasets.DatasetFolder(
    root=r'D:\hadis stuff\Dataset MXAH project\CheXpert\CheXpert-v1.0\valid',
    loader=lambda x: Image.open(x),
    extensions=('jpg', 'jpeg', 'png'),
    transform=transform
)

loader = DataLoader(dataset, batch_size=64, shuffle=False)

def get_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        images = images.view(images.size(0), -1)  
        mean += images.mean(1).sum()
        std += images.std(1).sum()
        total_images += images.size(0)

    mean /= total_images
    std /= total_images
    return mean.item(), std.item()

mean, std = get_mean_std(loader)
print(f'Mean: {mean}, Std: {std}')