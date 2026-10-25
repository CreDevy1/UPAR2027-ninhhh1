"""Only checkpoint/position/normalization/tokenization differ between adapters."""
import gc
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import common as C
from teacher_system.text import build_directions, validate_prompts


class SiglipTower(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision

    def forward(self, x):
        return self.vision(pixel_values=x, interpolate_pos_encoding=True).pooler_output

    def set_grad_checkpointing(self, enable=True):
        self.vision.encoder.gradient_checkpointing = enable
        if enable and not hasattr(self.vision.encoder, '_gradient_checkpointing_func'):
            from torch.utils.checkpoint import checkpoint
            from functools import partial
            self.vision.encoder._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)


class Adapter:
    key = None

    def __init__(self):
        self.spec = dict(C.config()['backbones'][self.key], mean=[.5]*3, std=[.5]*3)

    def load(self, names, device='cpu', text=True):
        if self.key.startswith('sl2'):
            return self._siglip(names, device, text)
        return self._pe(names, device, text)

    def _directions(self, names, embed):
        prompts = C.read(C.WORK / 'pe_text/teacher_system/text_prompts.json')
        validate_prompts(prompts, names)
        dirs, cosine, used, _, _ = build_directions(prompts, embed)
        assert dirs.shape[0] == 40 and torch.isfinite(dirs).all()
        return dirs, dict(prompts_sha256=C.sha(C.WORK / 'pe_text/teacher_system/text_prompts.json'),
                          strings_used=used, pos_neg_cosines=cosine)

    def _siglip(self, names, device, text):
        from transformers import AutoModel, AutoTokenizer, SiglipImageProcessor
        from huggingface_hub import snapshot_download
        # Local snapshots only; never download or mutate an older experiment.
        path = Path(snapshot_download(self.spec['model'], local_files_only=True))
        # The local weight cache omits preprocessor_config.json. Use the native
        # installed SigLIP processor's .5/.5 normalization, already used in upar_system.
        processor = SiglipImageProcessor()
        self.spec.update(mean=processor.image_mean, std=processor.image_std)
        model = AutoModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32).eval()
        provenance = dict(model=self.spec['model'], snapshot=path.name,
                          checkpoint_files={p.name: C.sha(p) for p in sorted(path.glob('*.safetensors'))},
                          position_policy='native SigLIP bicubic align_corners=False to 16x8')
        dirs = None
        if text:
            token_cache = C.WORK.parent / 'weights/hf_cache'
            token_path = Path(snapshot_download(self.spec['model'], revision=path.name,
                              cache_dir=token_cache, local_files_only=True))
            tokenizer = AutoTokenizer.from_pretrained(token_path, local_files_only=True)
            provenance['tokenizer_files'] = {p.name: C.sha(p) for p in token_path.glob('*')
                                            if p.is_file() and p.name != 'model.safetensors'}
            provenance['normalization_source'] = 'installed transformers.SiglipImageProcessor defaults: mean/std .5'
            model.text_model.to(device)
            @torch.inference_mode()
            def embed(strings):
                out = []
                for i in range(0, len(strings), 16):
                    tokens = tokenizer(strings[i:i+16], padding='max_length', max_length=64,
                                       truncation=True, return_tensors='pt').to(device)
                    out.append(model.get_text_features(**tokens).float().cpu())
                return torch.cat(out)
            dirs, record = self._directions(names, embed)
            provenance.update(record)
        tower = SiglipTower(model.vision_model)
        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        return tower.to(device), dirs, provenance

    def _pe(self, names, device, text):
        from teacher_system.pe import VisionTransformer, TextTransformer
        from teacher_system.config import PETextConfig
        from teacher_system.tokenizer import SimpleTokenizer
        if self.key == 'peb':
            path = C.WORK.parent / 'weights/pretrained/PE-Core-B16-224.pt'
            width, layers, heads = 768, 12, 12
        else:
            path = C.WORK.parent / 'weights/pretrained/PE-Core-L14-336.pt'
            width, layers, heads = 1024, 24, 16
        state = torch.load(path, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state.get('weights', state))
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        tower = VisionTransformer(patch_size=self.spec['patch'], width=width, layers=layers,
                                  heads=heads, mlp_ratio=4, output_dim=1024,
                                  image_size=self.spec['native'], use_cls_token=True)
        tower.load_state_dict({k.removeprefix('visual.'): v for k, v in state.items()
                               if k.startswith('visual.')}, strict=True)
        provenance = dict(model=self.spec['model'], checkpoint_file=str(path), checkpoint_sha256=C.sha(path),
                          position_policy='native PE bilinear align_corners=False abs + integer axial RoPE to 16x8',
)
        dirs = None
        if text:
            encoder = TextTransformer(**asdict(PETextConfig())).eval()
            encoder.load_state_dict({k: v for k, v in state.items()
                                     if not k.startswith('visual.') and k != 'logit_scale'}, strict=True)
            encoder.to(device)
            tokenizer = SimpleTokenizer(context_length=32)
            @torch.inference_mode()
            def embed(strings):
                return torch.cat([encoder(tokenizer(strings[i:i+16]).to(device)).float().cpu()
                                  for i in range(0, len(strings), 16)])
            dirs, record = self._directions(names, embed)
            provenance.update(record)
            del encoder
        del state
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        return tower.to(device), dirs, provenance

    @torch.inference_mode()
    def inspect_positions(self, tower, x):
        h, w = x.shape[-2:]
        assert (h // self.spec['patch'], w // self.spec['patch']) == (16, 8)
        if self.key.startswith('sl2'):
            tokens = tower.vision.embeddings(x, interpolate_pos_encoding=True)
            assert tokens.shape[1] == 128
            pos = tower.vision.embeddings.interpolate_pos_encoding(tokens, h, w)
        else:
            tokens = tower.conv1(x).flatten(2).transpose(1, 2)
            assert tokens.shape[1] == 128
            pos = tower._sample_abs_posemb(16, 8)
            tower.rope.update_grid(x.device, 16, 8)
        assert torch.isfinite(pos).all()
        return dict(patch_tokens=128, grid_hw=[16, 8], input_hw=[h, w],
                    position_shape=list(pos.shape), cls_excluded_from_patch_count=not self.key.startswith('sl2'))


class SL2B(Adapter):
    key = 'sl2b'


class SL2L(Adapter):
    key = 'sl2l'


class PEB(Adapter):
    key = 'peb'


class PEL(Adapter):
    key = 'pel'


ADAPTERS = {a.key: a for a in (PEB, SL2B, SL2L, PEL)}


def head_from_dirs(dirs):
    head = nn.Linear(dirs.shape[1], 40)
    with torch.no_grad():
        head.weight.copy_(dirs)
        head.bias.zero_()
    return head
