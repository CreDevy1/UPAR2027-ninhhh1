"""UPAR 2027 Track 2 entry points (code submission).

rank_gallery: per-image attribute logits (ensemble) -> automatic gallery-neighbour aggregation on visual
features -> Bernoulli log-likelihood (temperature T) -> query-axis normalisation -> similarities.
Every step is the same for every image and query; nothing depends on file names or directories.
"""
import time
IMPORT_T = time.monotonic()
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import t2fast
import t2post

t2fast.T0 = min(t2fast.T0, IMPORT_T)
CFG = t2fast.CONFIG


def load_model():
    t2fast.load()


def column_index(model_names, names):
    if len(names) != 40 or set(names) != set(model_names):
        raise ValueError('sample attribute_names must be the 40 UPAR attributes')
    pos = {n: i for i, n in enumerate(model_names)}
    return torch.tensor([pos[n] for n in names], dtype=torch.long)


def rank_gallery(sample):
    names = list(sample['attribute_names'])
    cols = column_index(CFG['attribute_names'], names)
    paths = [g['image_path'] for g in sample['gallery']]
    logits, feats, info = t2fast.infer(paths, budget_s=float(os.environ.get('UPAR_T2_BUDGET', CFG.get('budget_s', 230))))
    device = feats.device
    # NaN-aware weighted mean over members (members dropped by the time guard are skipped per image)
    w = [float(m.get('weight', 1.0)) for m in CFG['members']]
    num = torch.zeros_like(logits[0], dtype=torch.float64)
    den = torch.zeros((logits[0].shape[0], 1), dtype=torch.float64)
    for wi, z in zip(w, logits):
        ok = torch.isfinite(z).all(1, keepdim=True)
        num += torch.where(ok, z.double(), torch.zeros_like(z, dtype=torch.float64)) * wi
        den += ok.double() * wi
    if (den == 0).any():
        raise RuntimeError('an image has no member logits')
    z = (num / den)[:, cols].to(device)
    queries = torch.as_tensor(np.asarray(sample['queries'], dtype=np.float64), device=device)
    t = time.monotonic()
    post = dict(CFG['post'])
    if post.get('prior_logit') is not None:  # model order -> sample order
        post['prior_logit'] = [float(post['prior_logit'][i]) for i in cols.tolist()]
    if post.get('graph', 'visual') == 'logits':  # neighbour graph on centred ensemble logits instead of features
        feats = (z - z.mean(0, keepdim=True)).float()
    S = t2post.score_full(queries, z, feats, post)
    out = S.double().cpu().numpy()
    if not np.isfinite(out).all():
        raise FloatingPointError('non-finite similarities')
    t2fast.log(f'post-processing {time.monotonic() - t:.1f}s; total since import {time.monotonic() - t2fast.T0:.1f}s; '
               f'{info}')
    return {'similarities': out}
