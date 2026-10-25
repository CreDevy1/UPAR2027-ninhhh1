"""Check that a package rebuilt from the released checkpoints matches the submitted one.

    python tools/verify_rebuild.py framework/recipes/submissions/t1_pel_hr_tv_A1_20261021.zip \\
        weights/submissions/t1_pel_hr_tv_A1_20261021.zip

Compares every tensor of every assets/*.pt file and the numeric parts of assets/config.json.
Paths recorded in the config (where a checkpoint was read from) are allowed to differ.
"""
import io
import json
import sys
import zipfile

import torch

SKIP = {'source_checkpoint', 'asset_sha256'}


def load(path):
    z = zipfile.ZipFile(path)
    cfg = json.loads(z.read('assets/config.json'))
    tensors = {}
    for name in z.namelist():
        if name.startswith('assets/') and name.endswith('.pt'):
            tensors[name] = torch.load(io.BytesIO(z.read(name)), map_location='cpu', weights_only=True)
    return cfg, tensors


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in SKIP}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def main(new, ref):
    cfg_a, ta = load(new)
    cfg_b, tb = load(ref)
    ok = True
    if strip({k: v for k, v in cfg_a.items() if k != 'name'}) != strip({k: v for k, v in cfg_b.items() if k != 'name'}):
        print('config differs')
        ok = False
    if set(ta) != set(tb):
        print('asset files differ:', sorted(set(ta) ^ set(tb)))
        ok = False
    for name in sorted(set(ta) & set(tb)):
        for part in ('tower_state_dict', 'head_state_dict'):
            a, b = ta[name][part], tb[name][part]
            if set(a) != set(b):
                print(name, part, 'keys differ')
                ok = False
                continue
            worst = max(float((a[k].float() - b[k].float()).abs().max()) if a[k].numel() else 0. for k in a)
            same_dtype = all(a[k].dtype == b[k].dtype for k in a)
            print(f'{name} {part}: {len(a)} tensors, max |diff| {worst:.3g}, dtypes match {same_dtype}')
            ok &= worst == 0 and same_dtype
    print('MATCH' if ok else 'DIFFERENT')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:3]))
