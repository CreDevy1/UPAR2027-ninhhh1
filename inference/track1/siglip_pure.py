"""Pure-PyTorch SigLIP vision tower (no `transformers`), name-compatible with
HF `SiglipVisionModel(...).vision_model` so checkpoints load with strict=True.
Positional embedding follows HF interpolate_pos_encoding=True exactly
(bicubic, align_corners=False) for non-square inputs."""
import torch
import torch.nn.functional as F
from torch import nn

DEFAULTS = dict(hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                num_attention_heads=12, num_channels=3, image_size=224, patch_size=16,
                layer_norm_eps=1e-6)


class _Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.k_proj = nn.Linear(dim, dim); self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim); self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape; h = self.heads
        sp = lambda t: t.view(b, n, h, d // h).transpose(1, 2)
        out = F.scaled_dot_product_attention(sp(self.q_proj(x)), sp(self.k_proj(x)), sp(self.v_proj(x)))
        return self.out_proj(out.transpose(1, 2).reshape(b, n, d))


class _MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden); self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate='tanh'))


class _Layer(nn.Module):
    def __init__(self, dim, heads, hidden, eps):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=eps); self.self_attn = _Attention(dim, heads)
        self.layer_norm2 = nn.LayerNorm(dim, eps=eps); self.mlp = _MLP(dim, hidden)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class _Embeddings(nn.Module):
    def __init__(self, dim, image_size, patch_size, channels):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        self.num_positions = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_positions, dim)

    def forward(self, pixel_values):
        _, _, height, width = pixel_values.shape
        x = self.patch_embedding(pixel_values.to(dtype=self.patch_embedding.weight.dtype))
        x = x.flatten(2).transpose(1, 2)
        weight = self.position_embedding.weight
        if x.shape[1] == self.num_positions and height == width:
            return x + weight.unsqueeze(0)
        dim = x.shape[-1]; g = int(self.num_positions ** 0.5)
        pos = weight.unsqueeze(0).reshape(1, g, g, dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(height // self.patch_size, width // self.patch_size),
                            mode='bicubic', align_corners=False)
        return x + pos.permute(0, 2, 3, 1).reshape(1, -1, dim)


class _Encoder(nn.Module):
    def __init__(self, n, dim, heads, hidden, eps):
        super().__init__()
        self.layers = nn.ModuleList(_Layer(dim, heads, hidden, eps) for _ in range(n))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Head(nn.Module):
    def __init__(self, dim, heads, hidden, eps):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, dim))
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.layernorm = nn.LayerNorm(dim, eps=eps)
        self.mlp = _MLP(dim, hidden)

    def forward(self, x):
        probe = self.probe.expand(x.shape[0], -1, -1)
        h = self.attention(probe, x, x, need_weights=False)[0]
        h = h + self.mlp(self.layernorm(h))
        return h[:, 0]


class SiglipVision(nn.Module):
    """pixel_values (B,3,H,W) -> pooler_output (B,hidden)."""
    def __init__(self, **cfg):
        super().__init__()
        c = dict(DEFAULTS); c.update({k: v for k, v in cfg.items() if k in DEFAULTS})
        d, eps = c['hidden_size'], c['layer_norm_eps']
        self.embeddings = _Embeddings(d, c['image_size'], c['patch_size'], c['num_channels'])
        self.encoder = _Encoder(c['num_hidden_layers'], d, c['num_attention_heads'], c['intermediate_size'], eps)
        self.post_layernorm = nn.LayerNorm(d, eps=eps)
        self.head = _Head(d, c['num_attention_heads'], c['intermediate_size'], eps)

    def forward(self, pixel_values):
        return self.head(self.post_layernorm(self.encoder(self.embeddings(pixel_values))))
