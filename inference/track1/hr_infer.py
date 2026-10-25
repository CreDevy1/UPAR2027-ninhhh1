"""Offline PE-L high-resolution inference, with bounded decoding and OOM retry.

No time guard, skipped images, prior filling, or model dropping. Batch reduction
is performed only after a real CUDA OOM on the grading machine.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'assets/config.json').read_text(encoding='utf-8'))
STATE = None


def column_index(names):
    own = CONFIG['attribute_names']
    if len(names) != 40 or set(names) != set(own):
        raise ValueError('Exactly the 40 organizer attributes required')
    return [own.index(n) for n in names]


def load():
    global STATE
    if STATE is not None:
        return STATE
    from teacher_system.pe import VisionTransformer
    torch.set_num_threads(int(os.environ.get('UPAR_THREADS', '4')))
    device = 'cuda' if os.environ.get('UPAR_FORCE_CPU') != '1' and torch.cuda.is_available() else 'cpu'
    member = CONFIG['members'][0]
    if len(CONFIG['members']) != 1 or member['name'] != 'pel_hr':
        raise ValueError('This package contains exactly one pel_hr member')
    spec = member['spec']
    tower = VisionTransformer(patch_size=spec['patch'], width=1024, layers=24,
                              heads=16, mlp_ratio=4, output_dim=1024,
                              image_size=spec['native'], use_cls_token=True)
    ck = torch.load(ROOT / member['checkpoint'], map_location='cpu', weights_only=True)
    tower.load_state_dict(ck['tower_state_dict'], strict=True)
    hs = ck['head_state_dict']
    head = nn.Linear(hs['weight'].shape[1], hs['weight'].shape[0], bias='bias' in hs)
    head.load_state_dict(hs, strict=True)
    del ck, hs
    STATE = dict(tower=tower.to(device).eval(), head=head.to(device).eval(),
                 device=device, batch=int(CONFIG['batch']) if device == 'cuda' else 8)
    print(f'pel_hr loaded on {device}; batch={STATE["batch"]}; input_hw={member["input_hw"]}', flush=True)
    return STATE


def decode(path):
    h, w = CONFIG['members'][0]['input_hw']
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB').resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8).copy()


def _forward(pixels, state, need_features):
    device = state['device']
    t = torch.from_numpy(np.ascontiguousarray(pixels))
    if device == 'cuda':
        t = t.pin_memory()
    x = ((t.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.) - .5) / .5
    with torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
        f = state['tower'](x).float()
        z = state['head'](f).float()
    # Match the existing T2 package: normalize FP32 features before FP16 storage.
    feature = F.normalize(f, dim=1).half().cpu() if need_features else None
    return z.cpu(), feature


@torch.inference_mode()
def infer(paths, need_features=False):
    state = load()
    total = len(paths)
    logits = torch.empty((total, 40), dtype=torch.float32)
    feats = torch.empty((total, 1024), dtype=torch.float16) if need_features else None
    workers = max(1, min(int(CONFIG['decode_threads']), os.cpu_count() or 4))
    started = time.monotonic()
    # One active batch and one prefetched batch; a reduced CUDA batch takes
    # effect immediately, including inside already-decoded larger batches.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit(start):
            return [pool.submit(decode, p) for p in paths[start:start + state['batch']]]
        start = 0
        pending = submit(start)
        while pending:
            pixels = np.stack([f.result() for f in pending])
            next_start = start + len(pixels)
            pending = submit(next_start)
            offset = 0
            while offset < len(pixels):
                n = min(state['batch'], len(pixels) - offset)
                retry = False
                try:
                    z, f = _forward(pixels[offset:offset + n], state, need_features)
                except torch.cuda.OutOfMemoryError:
                    if state['device'] != 'cuda' or n <= 1:
                        raise
                    state['batch'] = max(1, n // 2)
                    retry = True
                if retry:
                    # Outside the handler, its exception traceback and failed
                    # _forward tensors have been released before empty_cache.
                    torch.cuda.empty_cache()
                    print(f'CUDA OOM: retry same images with batch={state["batch"]}', flush=True)
                    continue
                logits[start + offset:start + offset + n] = z
                if need_features:
                    feats[start + offset:start + offset + n] = f
                offset += n
            start = next_start
            print(f'pel_hr images {start}/{total}; batch={state["batch"]}; elapsed={time.monotonic()-started:.1f}s', flush=True)
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Non-finite logits')
    return logits, feats
