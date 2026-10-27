"""Fast batched inference for UPAR T2 (torch / numpy / Pillow only).

Images are decoded by a thread pool (Pillow releases the GIL), stacked as uint8 and normalised on the GPU.
Every member returns raw logits (40) and its pooled visual feature. Features of the members listed in
config 'feat_members' are L2-normalised and concatenated (used for gallery neighbour aggregation).
A time guard drops trailing non-feature members for the remaining images if the run would exceed the budget.
"""
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'assets' / 'config.json').read_text(encoding='utf-8'))
T0 = time.monotonic()
STATE = None


def log(msg):
    print(f'T2 [{time.monotonic() - T0:6.1f}s] {msg}', flush=True)


def _build(member, device):
    key, spec = member['name'], member['spec']
    if key.startswith('sl2'):
        from siglip_vision import SiglipVisionTower
        vc = member['vision_config']
        tower = SiglipVisionTower(image_size=vc['image_size'], patch_size=spec['patch'], hidden_size=vc['hidden_size'],
                                  num_layers=vc['num_hidden_layers'], num_heads=vc['num_attention_heads'],
                                  intermediate_size=vc['intermediate_size'],
                                  layer_norm_eps=vc.get('layer_norm_eps', 1e-6), grid_hw=tuple(member['grid_hw']))
    else:
        from teacher_system.pe import VisionTransformer
        width, layers, heads = (768, 12, 12) if key == 'peb' else (1024, 24, 16)
        tower = VisionTransformer(patch_size=spec['patch'], width=width, layers=layers, heads=heads, mlp_ratio=4,
                                  output_dim=1024, image_size=spec['native'], use_cls_token=True)
    ck = torch.load(ROOT / member['checkpoint'], map_location='cpu', weights_only=True)
    tower.load_state_dict({k: v.float() for k, v in ck['tower_state_dict'].items()}, strict=True)
    hs = {k: v.float() for k, v in ck['head_state_dict'].items()}
    head = nn.Linear(hs['weight'].shape[1], hs['weight'].shape[0], bias='bias' in hs)
    head.load_state_dict(hs, strict=True)
    del ck
    return tower.to(device).eval(), head.to(device).eval()


def load():
    global STATE
    if STATE is not None:
        return STATE
    device = 'cuda' if torch.cuda.is_available() and os.environ.get('UPAR_FORCE_CPU') != '1' else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        log(f'gpu {torch.cuda.get_device_name(0)}')
    models = []
    for m in CONFIG['members']:
        models.append(_build(m, device))
        log(f'loaded {m["name"]} on {device}')
    STATE = dict(device=device, models=models)
    return STATE


def _load_one(path, sizes):
    with Image.open(path) as im:
        im = im.convert('RGB')
        return [np.asarray(im.resize(s, Image.Resampling.BILINEAR), dtype=np.uint8) for s in sizes]


def _producer(paths, sizes, batch, out_q, workers, pin):
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(paths), batch):
                chunk = paths[start:start + batch]
                arrays = list(pool.map(lambda p: _load_one(p, sizes), chunk))
                stacked = [torch.from_numpy(np.stack([a[j] for a in arrays])) for j in range(len(sizes))]
                if pin:
                    stacked = [t.pin_memory() for t in stacked]
                out_q.put((start, stacked))
    except BaseException as exc:  # surface decode errors in the main thread
        out_q.put(exc)
        return
    out_q.put(None)


@torch.inference_mode()
def infer(paths, budget_s):
    """-> logits: list (per member) of G x 40 float32 CPU tensors, NaN rows where a member was dropped;
          feats : G x D float16 tensor on the device (concatenated L2-normalised features of feat_members)."""
    st = load()
    device, members, models = st['device'], CONFIG['members'], st['models']
    batch = int(CONFIG.get('batch', 128))
    workers = max(2, min(int(CONFIG.get('decode_threads', 16)), (os.cpu_count() or 4)))
    flip = bool(CONFIG.get('flip_tta', False))
    feat_names = list(CONFIG.get('feat_members', [members[0]['name']]))
    is_feat = [m['name'] in feat_names for m in members]
    def _pil_size(m):
        return (int(m['input_hw'][1]), int(m['input_hw'][0])) if 'input_hw' in m else (8 * m['spec']['patch'], 16 * m['spec']['patch'])
    sizes = sorted({_pil_size(m) for m in members})  # PIL (W,H)
    size_ix = {s: i for i, s in enumerate(sizes)}
    G = len(paths)
    logits = [torch.full((G, 40), float('nan'), dtype=torch.float32) for _ in members]
    feats = None
    alive = [True] * len(members)
    q = queue.Queue(maxsize=6)
    th = threading.Thread(target=_producer, args=(paths, sizes, batch, q, workers, device == 'cuda'), daemon=True)
    th.start()
    log(f'inference start: G={G} members={[m["name"] for m in members]} feat={feat_names} flip={flip} '
        f'batch={batch} threads={workers}')
    t_start = time.monotonic()
    done, next_log = 0, t_start
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, BaseException):
            raise item
        start, stacked = item
        xs = [((t.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.) - .5) / .5 for t in stacked]
        n = xs[0].shape[0]
        fparts = []
        for mi, m in enumerate(members):
            if not alive[mi]:
                continue
            tower, head = models[mi]
            x = xs[size_ix[_pil_size(m)]]
            with torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
                f = tower(x).float()
                z = head(f).float()
                if flip:
                    ff = tower(x.flip(-1)).float()
                    z = 0.5 * (z + head(ff).float())
                    f = 0.5 * (f + ff)
            logits[mi][start:start + n] = z.cpu()
            if is_feat[mi]:
                fparts.append(F.normalize(f, dim=1))
        fb = torch.cat(fparts, 1)
        if feats is None:
            feats = torch.zeros((G, fb.shape[1]), dtype=torch.float16, device=device)
        feats[start:start + n] = fb.to(torch.float16)
        done += n
        now = time.monotonic()
        rate = done / max(now - t_start, 1e-6)
        projected = (now - T0) + (G - done) / max(rate, 1e-6)
        if not CONFIG.get('disable_guard', False) and done >= 4 * batch and done < G and projected > budget_s:
            drop = [i for i in range(len(members) - 1, -1, -1) if alive[i] and not is_feat[i]]
            if drop:
                alive[drop[0]] = False
                log(f'time guard: projected {projected:.0f}s > budget {budget_s}s -> drop '
                    f'{members[drop[0]]["name"]} for the remaining {G - done} images')
        if now >= next_log or done == G:
            log(f'images {done}/{G} rate {rate:.1f}/s projected_end {projected:.0f}s alive {sum(alive)}')
            next_log = now + 15
    th.join()
    if device == 'cuda':
        torch.cuda.synchronize()
    return logits, feats, dict(inference_s=round(time.monotonic() - t_start, 1), images=G, alive=alive)
