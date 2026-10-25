"""Run a submission package the way the Codabench ingestion program calls it.

Track 1: imports run.py, calls load_model() and predict_batch() on the images of a gt.csv,
         thresholds at 0.5 and reports mA, instance F1 and their harmonic mean.
Track 2: imports run.py, calls rank_gallery() once with all queries and gallery images,
         and reports mAP and Rank-1 with exact attribute matches as relevant items.

    python tools/run_package.py --track 1 --package weights/submissions/t1_pel_hr_tv_A1_20261021.zip \\
        --data /path/to/UPAR-Challenge-2027/data --split val --limit 2000
    python tools/run_package.py --track 2 --package weights/submissions/t2_v4b.zip \\
        --data /path/to/UPAR-Challenge-2027/data --split val

--package accepts the zip or an unpacked folder. --grader-imports blocks the packages that are
missing on the grader (transformers, timm, open_clip, huggingface_hub, safetensors, einops) and
network sockets, so an import that would fail on Codabench fails here too.
Use the official image (docker/Dockerfile) to match the grader exactly.
"""
import argparse
import builtins
import csv
import importlib
import socket
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

BLOCKED = ('transformers', 'timm', 'open_clip', 'huggingface_hub', 'safetensors', 'einops')


def block_grader_imports():
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named '{name}' (not available on the grader)")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guarded

    def no_network(*args, **kwargs):
        raise OSError('network access is disabled on the grader')

    socket.socket.connect = no_network


def read_gt(path):
    with open(path, encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    names = [c.strip() for c in rows[0][1:]] if rows[0][0].lstrip('# ').strip() == 'image' else None
    if names is None:
        raise ValueError(f'unexpected header in {path}')
    images = [r[0] for r in rows[1:]]
    labels = np.asarray([[int(v) for v in r[1:]] for r in rows[1:]], dtype=np.int64)
    return names, images, labels


def read_queries(path):
    with open(path, encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    names = [c.lstrip('# ').strip() for c in rows[0]]
    return names, np.asarray([[int(v) for v in r] for r in rows[1:]], dtype=np.int64)


def load_package(package, workdir):
    package = Path(package)
    if package.is_dir():
        root = package
    else:
        root = Path(workdir) / package.stem
        with zipfile.ZipFile(package) as z:
            z.extractall(root)
    sys.path.insert(0, str(root))
    return importlib.import_module('run')


def track1(run, data, split, limit, batch):
    names, images, labels = read_gt(data / 'annotations' / 'task1' / split / 'gt.csv')
    if limit:
        images, labels = images[:limit], labels[:limit]
    run.load_model()
    probs = []
    t0 = time.monotonic()
    for i in range(0, len(images), batch):
        samples = [dict(image_path=str(data / p), attribute_names=names) for p in images[i:i + batch]]
        probs += run.predict_batch(samples)
    seconds = time.monotonic() - t0
    pred = np.asarray(probs) > 0.5
    y = labels.astype(bool)
    pos, neg = y.sum(0), (~y).sum(0)
    tpr = np.where(pos > 0, (pred & y).sum(0) / np.maximum(pos, 1), np.nan)
    tnr = np.where(neg > 0, (~pred & ~y).sum(0) / np.maximum(neg, 1), np.nan)
    ma = float(np.nanmean((tpr + tnr) / 2))
    tp = (pred & y).sum(1)
    prec = np.where(pred.sum(1) > 0, tp / np.maximum(pred.sum(1), 1), 0.)
    rec = np.where(y.sum(1) > 0, tp / np.maximum(y.sum(1), 1), 0.)
    f1 = float(np.mean(np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.)))
    print(f'images {len(images)}  time {seconds:.1f}s  mA {ma:.4f}  instance F1 {f1:.4f}  '
          f'average {2 * ma * f1 / (ma + f1):.4f}')
    print('note: the official score averages MEVID and Synth; numbers on the source VAL split are not comparable.')


def track2(run, data, split, limit):
    names, images, labels = read_gt(data / 'annotations' / 'task2' / split / 'gt.csv')
    qnames, queries = read_queries(data / 'annotations' / 'task2' / split / 'queries.csv')
    if qnames != names:
        raise ValueError('attribute order differs between gt.csv and queries.csv')
    if limit:
        images, labels = images[:limit], labels[:limit]
        keep = [i for i, q in enumerate(queries) if (labels == q).all(1).any()]
        queries = queries[keep]
    sample = dict(attribute_names=names, queries=queries.tolist(),
                  gallery=[dict(image_path=str(data / p)) for p in images])
    run.load_model()
    t0 = time.monotonic()
    out = run.rank_gallery(sample)
    seconds = time.monotonic() - t0
    if 'similarities' in out:
        order = np.argsort(-np.asarray(out['similarities']), axis=1, kind='stable')
    elif 'distances' in out:
        order = np.argsort(np.asarray(out['distances']), axis=1, kind='stable')
    else:
        order = np.asarray(out['ranking'])
    aps, r1 = [], []
    for qi, q in enumerate(queries):
        rel = (labels[order[qi]] == q).all(1)
        if not rel.any():
            continue
        hits = np.cumsum(rel)
        aps.append(float((hits[rel] / (np.flatnonzero(rel) + 1)).mean()))
        r1.append(float(rel[0]))
    print(f'gallery {len(images)}  queries {len(queries)}  time {seconds:.1f}s  '
          f'mAP {np.mean(aps):.4f}  Rank-1 {np.mean(r1):.4f}')
    print('note: the challenge ranks by mADM on MEVID and Synth; this is only a functional check.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', type=int, choices=(1, 2), required=True)
    ap.add_argument('--package', required=True)
    ap.add_argument('--data', required=True, help='organizer data folder (contains annotations/)')
    ap.add_argument('--split', default='val')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--grader-imports', action='store_true')
    a = ap.parse_args()
    if a.grader_imports:
        block_grader_imports()
    with tempfile.TemporaryDirectory() as tmp:
        run = load_package(a.package, tmp)
        if a.track == 1:
            track1(run, Path(a.data), a.split, a.limit, a.batch)
        else:
            track2(run, Path(a.data), a.split, a.limit)


if __name__ == '__main__':
    main()
