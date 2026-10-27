"""Check that a package rebuilt from the released checkpoints matches the submitted one.

    python tools/verify_rebuild.py framework/recipes/submissions/t1_ens2_hr_tv_pmcd_20261021.zip \\
        weights/submissions/t1_ens2_hr_tv_pmcd_20261021.zip

Compares every tensor of every assets/*.pt file and assets/config.json. Paths recorded in the config (where a
checkpoint was read from) and the asset hashes are allowed to differ. Assets are loaded one at a time, so the
two-encoder packages need about 2.5 GB of free RAM per encoder.
"""
import io
import json
import sys
import zipfile

import torch

SKIP = {'source_checkpoint', 'asset_sha256'}


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in SKIP}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def assets(z):
    return sorted(n for n in z.namelist() if n.startswith('assets/') and n.endswith('.pt'))


def main(new, ref):
    za, zb = zipfile.ZipFile(new), zipfile.ZipFile(ref)
    ok = True
    cfg_a, cfg_b = (json.loads(z.read('assets/config.json')) for z in (za, zb))
    if strip({k: v for k, v in cfg_a.items() if k != 'name'}) != strip({k: v for k, v in cfg_b.items() if k != 'name'}):
        print('config differs')
        ok = False
    if assets(za) != assets(zb):
        print('asset files differ:', sorted(set(assets(za)) ^ set(assets(zb))))
        ok = False
    for name in sorted(set(assets(za)) & set(assets(zb))):
        ta = torch.load(io.BytesIO(za.read(name)), map_location='cpu', weights_only=True)
        tb = torch.load(io.BytesIO(zb.read(name)), map_location='cpu', weights_only=True)
        for part in ('tower_state_dict', 'head_state_dict'):
            a, b = ta[part], tb[part]
            if set(a) != set(b):
                print(name, part, 'keys differ')
                ok = False
                continue
            worst = max(float((a[k].double() - b[k].double()).abs().max()) if a[k].numel() else 0. for k in a)
            same = all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape for k in a)
            print(f'{name} {part}: {len(a)} tensors, max |diff| {worst:.3g}, dtypes and shapes match {same}')
            ok &= worst == 0 and same
        del ta, tb
    print('MATCH' if ok else 'DIFFERENT')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:3]))
