import torch, sys
net = BottleneckDenoiser()
x   = torch.randn(2, 3, 48, 48)
y   = net(x)
params = sum(p.numel() for p in net.parameters()) / 1e3
print(f"✅ forward OK  | out={y.shape} | params≈{params:.0f}k")
