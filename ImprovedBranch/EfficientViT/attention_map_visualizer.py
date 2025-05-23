import os, sys
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from torchvision import transforms
from models.efficientvit import EfficientViT, LocalWindowAttention, MedicalXRayAttention
import argparse

parser = argparse.ArgumentParser(
    description="Compare EfficientViT attention maps"
)
parser.add_argument(
    "--naive_checkpoint",
    type=str,
    help="path to the baseline (naive) .pth file"
)
parser.add_argument(
    "--improved_checkpoint",
    type=str,
    help="path to the improved .pth file"
)
parser.add_argument(
    "--image_dir",
    type=str,
    help="directory containing your .jpg CXR images"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="attention_maps",
    help="where to save the output figures"
)

args = parser.parse_args()

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

def framed_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

PWD = Path.cwd()
MODELS_DIR = PWD / "EfficientViT" / "classification" / "models"
sys.path.insert(0, str(MODELS_DIR))

def make_model(use_mxa):
    model = EfficientViT(
        img_size=224, patch_size=16, in_chans=3,
        num_classes=14,
        stages=['s','s','s'], embed_dim=[192,288,384],
        key_dim=[16,16,16], depth=[1,3,4],
        num_heads=[3,3,4], window_size=[7,7,7],
        kernels=[5,5,5,5],
        distillation=False,
        multi_label=True
    )
    if not use_mxa:
        for m in model.modules():
            if isinstance(m, MedicalXRayAttention):
                m.forward = lambda x: x
    return model

BASE_CKPT = args.naive_checkpoint
IMP_CKPT  = args.improved_checkpoint
base_ckpt = torch.load(BASE_CKPT, map_location="cpu")
imp_ckpt  = torch.load(IMP_CKPT, map_location="cpu")
baseline_model = make_model(False)
improved_model = make_model(True)
baseline_model.load_state_dict(base_ckpt["model"], strict=False)
improved_model.load_state_dict(imp_ckpt["model"], strict=False)
baseline_model.eval()
improved_model.eval()

attention_maps = {"baseline": [], "improved": []}
def hook_gen(key):
    def hook(module, inp, out):
        attention_maps[key].append(out.detach().cpu())
    return hook

for m in baseline_model.modules():
    if isinstance(m, LocalWindowAttention):
        m.attn.register_forward_hook(hook_gen("baseline"))
for m in improved_model.modules():
    if isinstance(m, LocalWindowAttention):
        m.attn.register_forward_hook(hook_gen("improved"))
    if isinstance(m, MedicalXRayAttention):
        m.register_forward_hook(hook_gen("improved"))

# --------------- IMAGE TRANSFORMS ----------------
to_tensor = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -------------- UTILS ----------------
def normalize(x: np.ndarray):
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x

def upsample_to(a, size):
    t = torch.from_numpy(a[None,None]).float()
    t2 = F.interpolate(t, size=size, mode='bilinear', align_corners=False)
    return t2[0,0].numpy()

# ------------ COMPUTE GLOBAL RANGES ------------
pics_dir = Path(args.image_dir)
all_naive, all_improved, all_delta = [], [], []

for img_path in sorted(pics_dir.glob("*.jpg")):
    attention_maps = {"baseline": [], "improved": []}
    pil = Image.open(img_path).convert("RGB")
    inp = to_tensor(pil).unsqueeze(0)
    _ = baseline_model(inp)
    _ = improved_model(inp)
    b_map = attention_maps["baseline"][-1]
    i_map = attention_maps["improved"][-1]
    def to_hw(x):
        if x.ndim==2: return x.numpy()
        if x.ndim==3: return x[0].numpy()
        if x.ndim==4: return x.mean(1)[0].numpy()
        raise ValueError
    b2d = to_hw(b_map)
    i2d = to_hw(i_map)
    d2d = i2d - b2d
    all_naive.append(b2d)
    all_improved.append(i2d)
    all_delta.append(d2d)

vmin_vi = min(np.min(all_naive), np.min(all_improved))
vmax_vi = max(np.max(all_naive), np.max(all_improved))
delta_abs_max = max(np.max(np.abs(all_delta)), 1e-6)

# ---------------- PLOTTING ----------------
def plot_comparison(img, mhsa, mxa, out_path):
    mhsa = normalize(mhsa)
    mxa  = normalize(mxa)
    if mhsa.shape != mxa.shape:
        Ht, Wt = max(mhsa.shape[0], mxa.shape[0]), max(mhsa.shape[1], mxa.shape[1])
        mhsa = upsample_to(mhsa, (Ht, Wt))
        mxa  = upsample_to(mxa, (Ht, Wt))
    diff = mxa - mhsa

    target_size = (32,32)
    mhsa = upsample_to(mhsa, target_size)
    mxa  = upsample_to(mxa,  target_size)
    diff = upsample_to(diff, target_size)

    fig = plt.figure(figsize=(7.0,2.2), dpi=300)
    gs = fig.add_gridspec(1, 6, width_ratios=[1,1,1,1,0.05,0.05], wspace=0.1)
    axs = [fig.add_subplot(gs[0,i]) for i in range(4)]
    cax_vi  = fig.add_subplot(gs[0,4])
    cax_d   = fig.add_subplot(gs[0,5])

    framed_ax(axs[0]); axs[0].imshow(img, cmap="gray"); axs[0].set_title("CXR", pad=2)
    framed_ax(axs[1]); axs[1].imshow(mhsa, cmap="viridis",
                                     vmin=vmin_vi, vmax=vmax_vi, interpolation="nearest")
    axs[1].set_title("Naive", pad=2)
    framed_ax(axs[2]); axs[2].imshow(mxa, cmap="viridis",
                                     vmin=vmin_vi, vmax=vmax_vi, interpolation="nearest")
    axs[2].set_title("Improved", pad=2)
    framed_ax(axs[3]); axs[3].imshow(diff, cmap="RdBu_r",
                                     vmin=-delta_abs_max, vmax=delta_abs_max, interpolation="nearest")
    axs[3].set_title("Delta", pad=2)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin_vi, vmax_vi))
    cb1 = fig.colorbar(sm, cax=cax_vi, ticks=[vmin_vi, (vmin_vi+vmax_vi)/2, vmax_vi])
    cb1.set_label("Attention", rotation=90, labelpad=4)

    sm2 = plt.cm.ScalarMappable(cmap="RdBu_r",
                                norm=plt.Normalize(-delta_abs_max, delta_abs_max))
    cb2 = fig.colorbar(sm2, cax=cax_d, ticks=[-delta_abs_max, 0, delta_abs_max])
    cb2.set_label("Δ Attention", rotation=90, labelpad=4)

    plt.tight_layout(pad=0.1)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ---------------- LOOP ----------------
out_dir = Path(args.output_dir)
out_dir.mkdir(exist_ok=True)
for img_path in sorted(pics_dir.glob("*.jpg")):
    attention_maps = {"baseline": [], "improved": []}
    pil = Image.open(img_path).convert("RGB")
    inp = to_tensor(pil).unsqueeze(0)
    _ = baseline_model(inp)
    _ = improved_model(inp)
    b_map = attention_maps["baseline"][-1]
    i_map = attention_maps["improved"][-1]
    b2d = to_hw(b_map)
    i2d = to_hw(i_map)
    full_gray = np.array(pil.convert("L"))
    plot_comparison(full_gray, b2d, i2d, out_dir/f"{img_path.stem}_attn.png")

print("Done — check attention_maps/")