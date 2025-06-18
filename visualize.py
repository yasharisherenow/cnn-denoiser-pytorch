import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Import local modules
import sys
PROJECT_ROOT = Path().resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Configuration
sigma = 25
checkpoint = Path("weights/model_best.pth")

# Load a single random patch
ds = PatchDataset(Path("unseen_data/"), sigma=sigma, crops_per_img=2000)
idx = 7  # example index, you can change this
noisy, clean = ds[idx]

# Load model
model = BottleneckDenoiser()
model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
model.eval()

# Denoise
with torch.no_grad():
    output = model(noisy.unsqueeze(0)).squeeze(0)

# Helper to convert tensors to NumPy arrays
def to_np(img_tensor):
    # Move channel to last dimension and convert to numpy
    return img_tensor.permute(1, 2, 0).cpu().numpy()

noisy_np = to_np(noisy)
clean_np = to_np(clean)
output_np = to_np(output)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img, title in zip(axes, [clean_np, noisy_np, output_np], ["Clean", "Noisy", "Denoised"]):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()