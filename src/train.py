import argparse, math, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path.cwd()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))



# --------------------------------------------------------------------------- #
def validate(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            total += psnr(model(noisy), clean)
    return total / len(loader)


def train_one_epoch(model, loader, opt, scaler, device, use_amp):
    model.train()
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = F.mse_loss(model(noisy), clean)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()


# --------------------------------------------------------------------------- #
def main(cfg):
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- dataset & split --------------------------------------------
    full_ds = PatchDataset(cfg.data_root, cfg.sigma)
    val_len = math.ceil(len(full_ds) * cfg.val_split)
    train_len = len(full_ds) - val_len
    g = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=g)

    dl_kw = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=(device.type == "cuda"),
    )
    train_dl = DataLoader(train_ds, shuffle=True, **dl_kw)
    val_dl   = DataLoader(val_ds,   shuffle=False, **dl_kw)

    # ---------- model, optim, scaler ---------------------------------------
    model = BottleneckDenoiser().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_psnr, patience_left = -1.0, cfg.patience

    # -------------------------- loop ---------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_one_epoch(model, train_dl, opt, scaler, device, cfg.amp)
        val_psnr = validate(model, val_dl, device)

        print(f"Epoch {epoch:03d}/{cfg.epochs} | val PSNR {val_psnr:5.2f} dB | {time.time()-t0:4.1f}s")

        # --------- checkpoint / early stop ---------------------------------
        if val_psnr > best_psnr + 0.05:
            best_psnr = val_psnr
            patience_left = cfg.patience
            torch.save(model.state_dict(), cfg.out_dir / "model_best.pth")
            print("  ↳ new best, checkpoint saved")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping triggered.")
                break


# ---- configuration & run ----
from argparse import Namespace
from pathlib import Path

cfg = Namespace(
    data_root = Path("data"),
    sigma      = 25,          # Gaussian σ (pixels)
    batch_size = 128,
    epochs     = 10,
    lr         = 1e-3,
    val_split  = 0.2,
    workers    = 16,
    out_dir    = Path("weights"),
    seed       = 42,
    patience   = 7,           # early-stop patience
    amp        = True         # mixed precision
)

# call your training / evaluation entry point
main(cfg)