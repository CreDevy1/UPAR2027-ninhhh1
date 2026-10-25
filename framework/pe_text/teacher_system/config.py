"""Local PE configuration; the pilot selects the official L14-336 entry."""
from dataclasses import dataclass

@dataclass
class PEConfig:
    patch_size: int = 16
    width: int = 768
    layers: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    output_dim: int = 1024
    image_size: int = 224
    use_cls_token: bool = True
    pool_type: str = 'attn'

@dataclass
class PETextConfig:
    context_length: int = 32
    width: int = 1024
    heads: int = 16
    layers: int = 24
    output_dim: int = 1024

PE_VISION_CONFIG = {'PE-Core-L14-336': PEConfig(patch_size=14,width=1024,layers=24,heads=16,image_size=336)}
PE_TEXT_CONFIG = {'PE-Core-L14-336': PETextConfig()}

def fetch_pe_checkpoint(name, path=None):
    if path is None:
        raise ValueError('Offline runtime requires a local checkpoint')
    return path
