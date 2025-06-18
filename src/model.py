import torch.nn as nn

class BottleneckDenoiser(nn.Module):
    """
    A 10-layer convolutional denoiser that predicts the *noise* and
    adds it back to the input (residual learning = faster training).

    Layer count:
        • 1   stem   Conv(3→mid)
        • 2×d bottleneck blocks (each has two Conv layers)
        • 1   head   Conv(mid→3)
      For depth=4  → 1 + 8 + 1 = 10 learnable Conv layers.
    """

    def __init__(self, mid: int = 64, bottleneck: int = 32, depth: int = 4):
        super().__init__()

        layers = [
            nn.Conv2d(3, mid, kernel_size=3, padding=1),  # stem (learnable-1)
            nn.ReLU(inplace=True),
        ]

        # ---------------- bottleneck blocks ----------------
        for _ in range(depth):
            layers += [
                nn.Conv2d(mid, bottleneck, 3, padding=1),  # narrow (learnable)
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck, mid, 3, padding=1),  # back to wide
                nn.ReLU(inplace=True),
            ]

        # ---------------- projection head ------------------
        layers.append(nn.Conv2d(mid, 3, kernel_size=3, padding=1))  # learnable-N

        self.body = nn.Sequential(*layers)

        # safety check at construction time
        conv_layers = sum(isinstance(m, nn.Conv2d) for m in self.body)
        assert conv_layers <= 10, f"Too many layers ({conv_layers})!"

    # ------------------------------------------------------
    # forward
    # ------------------------------------------------------
    def forward(self, x):
        """Predict noise residual and add it back to the noisy input."""
        return x + self.body(x)
