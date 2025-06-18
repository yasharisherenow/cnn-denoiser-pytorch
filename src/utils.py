import random
import numpy as np

import torch
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image


def seed_everything(seed: int = 42) -> None:
    """
    Make *every* source of randomness use the same seed so that you
    and your grader get identical results (data split, noise, weights).

    Parameters
    ----------
    seed : int
        Any 32-bit integer; use the same value in training and evaluation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio in *decibels* assuming inputs are
    normalised to the [0, 1] range.

    A perfect reconstruction returns ∞ dB; higher is better.
    """
    mse = torch.mean((pred - target) ** 2)
    return 10.0 * torch.log10(1.0 / mse).item()

