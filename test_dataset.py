from pathlib import Path
from torch.utils.data import DataLoader

# ---- CHANGE ME if your repo uses a different data folder ---------------
data_root = Path("data/")          # expects 5 images saved here
test_sigma = 25                       # pixel-space σ
# ------------------------------------------------------------------------

ds = PatchDataset(data_root, sigma=test_sigma)
dl = DataLoader(ds, batch_size= 4, shuffle=True)

noisy, clean = next(iter(dl))
assert noisy.shape == (4, 3, 48, 48), "Bad tensor shape"
assert clean.shape == (4, 3, 48, 48), "Bad target shape"
assert 0.0 <= noisy.min() and noisy.max() <= 1.0, "Noisy out of [0,1] range"

print(
    "Dataset sanity check passed —",
    f"len={len(ds)},",
    f"batch shape={noisy.shape},",
    f"val range=({noisy.min().item():.3f}, {noisy.max().item():.3f})",
)
