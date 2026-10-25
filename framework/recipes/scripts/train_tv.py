"""Train one member of the Track 2 ensemble on TRAIN + VAL (16x8 patch grid).

    python -u framework/recipes/scripts/train_tv.py pel     # PE-Core L/14 at 224x112
    python -u framework/recipes/scripts/train_tv.py sl2l    # SigLIP 2 L/16 at 256x128

The recipe is configs/uf_v1.json of the training core with the label mask (arm lc_m1).
TRAIN and VAL are merged into recipes/data/train_val_gt.csv, which the 2027 terms allow.
The checkpoint is written to recipes/runs_tv/allsource_<backbone>_lc_m1_s42/epoch6.pt.
The held-out evaluation at the end of the run uses VAL, which is part of training here, so it is not meaningful.
"""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parent
UF = ROOT / 'core'
sys.path.insert(0, str(UF))
from uf import common as C  # noqa: E402


def merged_csv():
    out = WORK / 'data' / 'train_val_gt.csv'
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tr = C.TRAIN.read_text(encoding='utf-8-sig').splitlines()
    va = C.VAL.read_text(encoding='utf-8-sig').splitlines()
    assert tr[0] == va[0], 'header mismatch'
    a = {r.split(',')[0] for r in tr[1:]}
    b = {r.split(',')[0] for r in va[1:]}
    assert not (a & b), 'train/val overlap'
    tmp = out.with_suffix('.tmp')
    tmp.write_text('\n'.join(tr + va[1:]) + '\n', encoding='utf-8')
    tmp.replace(out)
    return out


def main():
    backbone = sys.argv[1]
    ident, preflight = C.validate_milestones()
    merged = merged_csv()
    C.TRAIN = merged
    C.validate_milestones = lambda: (ident, preflight)
    rel = Path('..') / '..' / WORK.name / 'runs_tv'
    orig = C.run_id
    C.run_id = lambda b, p, a, aug='none': str(rel / orig(b, p, a, aug))
    (WORK / 'runs_tv').mkdir(exist_ok=True)
    note = dict(backbone=backbone, train_csv=str(merged),
                train_csv_sha256=hashlib.sha256(merged.read_bytes()).hexdigest(),
                rows=len(merged.read_text(encoding='utf-8').splitlines()) - 1,
                recipe='UF-v1 allsource lc_m1, unchanged config', framework_identity_code_sha=ident['code_sha'])
    (WORK / 'runs_tv' / f'NOTE_{backbone}.json').write_text(json.dumps(note, indent=2), encoding='utf-8')
    print(json.dumps(note), flush=True)
    import torch
    torch.set_num_threads(4)
    from uf import engine as E
    E.run(backbone, 'allsource', 'lc_m1', None, 'none')
    print('TRAIN_TV DONE', backbone, flush=True)


if __name__ == '__main__':
    main()
