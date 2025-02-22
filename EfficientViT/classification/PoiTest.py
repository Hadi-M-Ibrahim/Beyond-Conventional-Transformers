import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision import transforms

class MedicalXRayAttention(nn.Module):
    """
    Medical X-Ray Attention (MXA) Module with dynamic ROI selection and CBAM-like attention.
    """
    def __init__(self, in_channels, reduction=16):
        super(MedicalXRayAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.roi_predictor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 4, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass for MXA with dynamic ROI selection.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Attention-weighted output.
        """
        B, C, H, W = x.shape

        roi_coords = self.roi_predictor(x)  
        roi_coords = roi_coords.mean(dim=(2, 3))
        
        x_pooled = []
        for i in range(B):
            x1, y1, x2, y2 = roi_coords[i]
            x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)  

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)


            if x2 > x1 and y2 > y1:
                x_cropped = x[i, :, y1:y2, x1:x2]
            else:
                x_cropped = x[i, :, :, :]

            x_cropped = x_cropped.unsqueeze(0)
            x_resized = F.interpolate(x_cropped, size=(H, W), mode='bilinear', align_corners=False)
            x_pooled.append(x_resized)
        
        x = torch.cat(x_pooled, dim=0)

        ca_weights = self.channel_attention(x)
        x = x * ca_weights

        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa_input = torch.cat([max_pool, avg_pool], dim=1)
        sa_weights = self.spatial_attention(sa_input)
        x = x * sa_weights

        return x

def visualize_roi(image_path, model, device='cpu', output_path="roi_visualization.png"):
    """
    Loads an image, computes the ROI using the model's ROI predictor, visualizes the ROI
    by drawing a rectangle, and saves the result as a PNG.
    """
    # Load the image and record its original size.
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size

    # Preprocess the image: resize to 224x224 and convert to a tensor.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)

    # Get ROI parameters from the model's ROI predictor.
    model.eval()
    with torch.no_grad():
        roi_params = model.roi_predictor(img_tensor)
    # Average over spatial dimensions to get one ROI per image: (B, 4)
    roi_params = roi_params.mean(dim=(2, 3))[0].cpu().numpy()  # (x1, y1, x2, y2)

    # Ensure proper ordering of coordinates.
    x1_norm = min(roi_params[0], roi_params[2])
    x2_norm = max(roi_params[0], roi_params[2])
    y1_norm = min(roi_params[1], roi_params[3])
    y2_norm = max(roi_params[1], roi_params[3])

    # Map normalized coordinates back to original image dimensions.
    x1_pixel = int(x1_norm * orig_w)
    x2_pixel = int(x2_norm * orig_w)
    y1_pixel = int(y1_norm * orig_h)
    y2_pixel = int(y2_norm * orig_h)

    print(f"Predicted ROI (normalized): x1={x1_norm:.2f}, y1={y1_norm:.2f}, x2={x2_norm:.2f}, y2={y2_norm:.2f}")
    print(f"Predicted ROI (pixels): x1={x1_pixel}, y1={y1_pixel}, x2={x2_pixel}, y2={y2_pixel}")

    # Create the visualization.
    fig, ax = plt.subplots(1)
    ax.imshow(np.array(image))
    width = x2_pixel - x1_pixel
    height = y2_pixel - y1_pixel
    rect = patches.Rectangle((x1_pixel, y1_pixel), width, height,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.set_title("ROI Visualization")
    plt.axis('off')
    
    # Save the visualization as a PNG file.
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved visualization to {output_path}")
    plt.close(fig)

def train_model_unsupervised(image_path, model, device, epochs=50):
    """
    A simple unsupervised training loop that uses a reconstruction loss.
    The model's attention output is used to reconstruct the original image.
    Over time, the ROI predictor is expected to focus on regions that minimize the reconstruction loss.
    Every few epochs the current ROI is visualized.
    """
    # Load and preprocess the image.
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    recon_loss_fn = nn.L1Loss()  # or use nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        # Forward pass: obtain the attention-weighted output.
        x_att = model(img_tensor)
        # Unsupervised reconstruction loss: compare the attention output to the input.
        loss = recon_loss_fn(x_att, img_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}, Reconstruction Loss: {loss.item():.4f}")
            output_file = f"roi_epoch_{epoch}.png"
            visualize_roi(image_path, model, device, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize the ROI from a Medical X-Ray Attention Model and track its evolution during unsupervised training."
    )
    parser.add_argument("--image_path", type=str, help="Path to the input image file")
    parser.add_argument("--output", type=str, default="roi_visualization.png",
                        help="Path to the output PNG file (for single visualization)")
    parser.add_argument("--train_epochs", type=int, default=0,
                        help="Number of training epochs to run for unsupervised ROI learning (set >0 to train and visualize over time)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the model with 3 input channels (RGB).
    model = MedicalXRayAttention(in_channels=3).to(device)
    
    # (Optional: load pre-trained weights here.)
    
    if args.train_epochs > 0:
        # Train the model in an unsupervised manner and visualize ROI updates over time.
        train_model_unsupervised(args.image_path, model, device, epochs=args.train_epochs)
    else:
        # Simply visualize the ROI once.
        visualize_roi(args.image_path, model, device, args.output)
