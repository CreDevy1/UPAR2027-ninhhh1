"""Pure-PyTorch SigLIP vision tower for UF-v1 sl2l (no transformers at inference).
Parameter names match HF SiglipVisionModel.vision_model. The 16x8 position embedding
(`embeddings.fixed_pos`) was precomputed at build time with HF interpolate_pos_encoding."""
import torch
import torch.nn.functional as F
from torch import nn


class _Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.k_proj = nn.Linear(dim, dim); self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim); self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads
        split = lambda t: t.view(b, n, h, d // h).transpose(1, 2)
        out = F.scaled_dot_product_attention(split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x)))
        return self.out_proj(out.transpose(1, 2).reshape(b, n, d))


class _MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden); self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class _Layer(nn.Module):
    def __init__(self, dim, heads, hidden, eps):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=eps); self.self_attn = _Attention(dim, heads)
        self.layer_norm2 = nn.LayerNorm(dim, eps=eps); self.mlp = _MLP(dim, hidden)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class _Embeddings(nn.Module):
    def __init__(self, dim, image_size, patch_size, grid_hw):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Embedding((image_size // patch_size) ** 2, dim)
        self.grid_hw = tuple(grid_hw)
        self.register_buffer("fixed_pos", torch.zeros(1, grid_hw[0] * grid_hw[1], dim))

    def forward(self, pixel_values):
        x = self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype))
        if tuple(x.shape[-2:]) != self.grid_hw:
            raise ValueError(f"patch grid {tuple(x.shape[-2:])} != {self.grid_hw}")
        return x.flatten(2).transpose(1, 2) + self.fixed_pos


class _Encoder(nn.Module):
    def __init__(self, n, dim, heads, hidden, eps):
        super().__init__()
        self.layers = nn.ModuleList(_Layer(dim, heads, hidden, eps) for _ in range(n))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _PoolingHead(nn.Module):
    def __init__(self, dim, heads, hidden, eps):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, dim))
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.layernorm = nn.LayerNorm(dim, eps=eps)
        self.mlp = _MLP(dim, hidden)

    def forward(self, x):
        h = self.attention(self.probe.repeat(x.shape[0], 1, 1), x, x)[0]
        h = h + self.mlp(self.layernorm(h))
        return h[:, 0]


class SiglipVisionTower(nn.Module):
    def __init__(self, image_size, patch_size, hidden_size, num_layers, num_heads, intermediate_size,
                 layer_norm_eps=1e-6, grid_hw=(16, 8)):
        super().__init__()
        self.embeddings = _Embeddings(hidden_size, image_size, patch_size, grid_hw)
        self.encoder = _Encoder(num_layers, hidden_size, num_heads, intermediate_size, layer_norm_eps)
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.head = _PoolingHead(hidden_size, num_heads, intermediate_size, layer_norm_eps)

    def forward(self, pixel_values):
        return self.head(self.post_layernorm(self.encoder(self.embeddings(pixel_values))))
