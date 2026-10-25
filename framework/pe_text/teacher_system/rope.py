"""Torch-only PE axial RoPE; exact integer coordinates and interleaved pairs.

The CLS token is (0,0); patch coordinates start at (1,1), matching Meta.
Temporary frequencies are rebuilt for grid/device changes, never checkpointed.
"""
import torch

class Rope2D:
    def __init__(self, dim, use_cls_token=False):
        self.dim = dim
        self.use_cls_token = use_cls_token
        self.grid_size = None
        self.freq = None

    def init_tensors(self):
        axis_dim = self.dim // 2
        self.base = 1.0 / (10000 ** (torch.arange(0, axis_dim, 2).float() / axis_dim))

    def update_grid(self, device, grid_h, grid_w):
        if self.grid_size != (grid_h, grid_w) or self.freq is None or self.freq.device != device:
            base = self.base.to(device)
            offset = int(self.use_cls_token)
            y = torch.arange(grid_h, device=device) + offset
            x = torch.arange(grid_w, device=device) + offset
            fy = (y[:, None] * base).repeat_interleave(2, -1)[:, None].expand(grid_h, grid_w, -1)
            fx = (x[:, None] * base).repeat_interleave(2, -1)[None].expand(grid_h, grid_w, -1)
            self.freq = torch.cat((fx, fy), -1).reshape(grid_h * grid_w, -1)
            if self.use_cls_token:
                self.freq = torch.cat((torch.zeros(1, self.dim, device=device), self.freq))
            self.freq = self.freq[None, None]
            self.grid_size = (grid_h, grid_w)

    def __call__(self, q, k):
        def rotate(t):
            dtype = t.dtype
            pairs = t.reshape(*t.shape[:-1], -1, 2)
            half = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)
            return (t * self.freq.cos() + half * self.freq.sin()).to(dtype)
        return rotate(q), rotate(k)
