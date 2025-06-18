class PatchDataset(Dataset):
    """
    Parameters
    ----------
    img_dir : Path
        Folder that contains the five raw images.
    sigma : int
        Gaussian noise σ in pixel units (e.g. 25 for moderate noise).
    crops_per_img : int, default=2000
        How many random crops we will *pretend* each image yields.
        The dataset length is len(img_dir) × crops_per_img.
    """

    def __init__(self, img_dir: Path, sigma: int, crops_per_img: int = 2000):
        self.paths: List[Path] = [p for p in Path(img_dir).glob("*") if p.is_file()] # Exclude directories
        if not self.paths:
            raise ValueError(f"No images found in {img_dir}")
        self.sigma = sigma / 255.0
        self.crops_per_img = crops_per_img
        self.to_tensor = ToTensor()

    # ------------------------------------------------------------------ #
    #  Methods required by torch.utils.data.Dataset
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        """Total number of samples in an *epoch*."""
        return len(self.paths) * self.crops_per_img

    def __getitem__(self, idx: int):
        """
        Return one (noisy, clean) pair.
        `idx // crops_per_img` maps idx range to an image file.
        """
        img_path = self.paths[idx // self.crops_per_img]
        img = self.to_tensor(Image.open(img_path).convert("RGB"))

        # ---- random 48×48 crop ------------------------------------------------
        h, w = img.shape[1:]
        top  = random.randint(0, h - 48)
        left = random.randint(0, w - 48)
        clean = img[:, top : top + 48, left : left + 48]

        # ---- additive Gaussian noise -----------------------------------------
        noise = torch.randn_like(clean) * self.sigma
        noisy = (clean + noise).clamp(0.0, 1.0)

        return noisy, clean
