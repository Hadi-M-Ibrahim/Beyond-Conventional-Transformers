import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from threeaugment import new_data_aug_generator  # Import your ThreeAugment

# Mock 'Args' class to simulate your argument parser
class Args:
    input_size = 224
    color_jitter = 0
    aa = 'rand-m9-mstd0.5-inc1'
    reprob = 0.25
    remode = 'pixel'
    recount = 1
    train_interpolation = 'bicubic'
    finetune = False

args = Args()

# Build the ThreeAugment transform
transform = new_data_aug_generator(args)

# Function to track applied augmentations
class WrappedTransform:
    def __init__(self, transform):
        self.transform = transform
        self.applied_transforms = []

    def __call__(self, img):
        self.applied_transforms = []  # Reset list for each image
        for t in self.transform.transforms:
            img = t(img)
            if isinstance(t, transforms.RandomChoice):
                self.applied_transforms.append(t.transforms[0].__class__.__name__)
            else:
                self.applied_transforms.append(t.__class__.__name__)
        return img

# Wrap the transform to track augmentations
wrapped_transform = WrappedTransform(transform)

# Load a sample image
original_img = Image.open(r"D:\hadis stuff\Dataset MXAH project\CheXpert\CheXpert-v1.0\train\patient00001\study1\view1_frontal.jpg").convert('RGB')

# Apply the transform multiple times and visualize
augmented_imgs = []
augmentation_labels = []

for _ in range(4):
    aug_img = wrapped_transform(original_img)
    augmented_imgs.append(aug_img)
    augmentation_labels.append(", ".join(wrapped_transform.applied_transforms))

# Plot the results
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
axs[0].imshow(original_img)
axs[0].set_title("Original")
axs[0].axis('off')

for i, (aug_img, labels) in enumerate(zip(augmented_imgs, augmentation_labels)):
    img_to_show = aug_img.permute(1, 2, 0).numpy()
    axs[i+1].imshow(img_to_show)
    axs[i+1].set_title(labels, fontsize=6)  # Smaller font size
    axs[i+1].axis('off')

plt.tight_layout()
plt.show()
