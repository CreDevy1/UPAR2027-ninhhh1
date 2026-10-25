"""Build the Track 2 code-submission package from trained checkpoints.

Scored entry (submission 1004665):
    python -u framework/recipes/scripts/build_pkg.py --name t2_v4b --members pel,sl2l --feat pel --budget 270 \
        --post @configs/post_v4b.json \
        --ckpt pel=framework/recipes/runs_tv/allsource_pel_lc_m1_s42/epoch6.pt \
        --ckpt sl2l=framework/recipes/runs_tv/allsource_sl2l_lc_m1_s42/epoch6.pt

Writes recipes/build/<name>/ and recipes/submissions/<name>.zip (+ .sha256, build_<name>.json).
Encoder weights are stored in FP16, heads in FP32; inference runs under FP16 autocast.
The SigLIP 2 tower is rewritten in plain PyTorch because the grader has no transformers. Its 16x8 position
embedding is computed here once with HF interpolate_pos_encoding (FP32) and stored as embeddings.fixed_pos.
Needs transformers at build time only.
"""
import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import torch

WORK = Path(__file__).resolve().parents[1]          # framework/recipes
ROOT = WORK.parent                                   # framework
REPO = ROOT.parent
TEMPLATE = REPO / 'inference' / 'track2'             # runtime files of the scored package
OLD = TEMPLATE
OLD4 = TEMPLATE                                      # member specs and attribute order
EXAMPLE = TEMPLATE                                   # metadata.yaml
PKG_SRC = TEMPLATE
GRID = (16, 8)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def siglip_pure_state(tower_sd, vc_partial, patch):
    from transformers import SiglipVisionConfig, SiglipVisionModel
    sys.path.insert(0, str(PKG_SRC))
    from siglip_vision import SiglipVisionTower
    cfg = SiglipVisionConfig(**vc_partial)
    vc = dict(hidden_size=cfg.hidden_size, image_size=cfg.image_size, intermediate_size=cfg.intermediate_size,
              num_attention_heads=cfg.num_attention_heads, num_hidden_layers=cfg.num_hidden_layers,
              layer_norm_eps=cfg.layer_norm_eps, model_type='siglip_vision_model')
    hf = SiglipVisionModel(cfg).vision_model
    sd = {k[len('vision.'):]: v.float() for k, v in tower_sd.items()}
    sd.pop('embeddings.position_ids', None)
    hf.load_state_dict(sd, strict=True)
    hf.eval()
    with torch.no_grad():
        pos = hf.embeddings.interpolate_pos_encoding(torch.zeros(1, GRID[0] * GRID[1], vc['hidden_size']),
                                                     GRID[0] * patch, GRID[1] * patch)
    pure = SiglipVisionTower(vc['image_size'], patch, vc['hidden_size'], vc['num_hidden_layers'],
                             vc['num_attention_heads'], vc['intermediate_size'], vc['layer_norm_eps'], GRID)
    sd['embeddings.fixed_pos'] = pos.clone().contiguous()
    pure.load_state_dict(sd, strict=True)
    pure.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, GRID[0] * patch, GRID[1] * patch, generator=torch.Generator().manual_seed(0))
        delta = float((hf(pixel_values=x, interpolate_pos_encoding=True).pooler_output - pure(x)).abs().max())
    assert delta <= 1e-4, f'pure SigLIP tower mismatch {delta}'
    return {k: v.contiguous() for k, v in pure.state_dict().items()}, vc, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--members', required=True)
    ap.add_argument('--feat', required=True)
    ap.add_argument('--post', required=True)
    ap.add_argument('--ckpt', action='append', default=[])
    ap.add_argument('--weights', default='')
    ap.add_argument('--flip', action='store_true')
    ap.add_argument('--budget', type=float, default=230.0)
    ap.add_argument('--batch', type=int, default=128)
    a = ap.parse_args()
    members = a.members.split(',')
    feat = a.feat.split(',')
    assert set(feat) <= set(members)
    post = json.loads((WORK / a.post[1:]).read_text(encoding='utf-8') if a.post.startswith('@') else a.post)
    ck_paths = {m: WORK / 'runs_tv' / f'allsource_{m}_lc_m1_s42' / 'epoch6.pt' for m in members}
    for item in a.ckpt:
        k, v = item.split('=', 1)
        ck_paths[k] = Path(v)
    weights = {m: 1.0 for m in members}
    if a.weights:
        for item in a.weights.split(','):
            k, v = item.split('=')
            weights[k] = float(v)
    old4 = {m['name']: m for m in json.loads((OLD4 / 'assets' / 'config.json').read_text(encoding='utf-8'))['members']}
    attr = json.loads((OLD4 / 'assets' / 'config.json').read_text(encoding='utf-8'))['attribute_names']

    dst = WORK / 'build' / a.name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / 'assets').mkdir(parents=True)
    shutil.copytree(OLD / 'teacher_system', dst / 'teacher_system', ignore=shutil.ignore_patterns('__pycache__'))
    for f in ('LICENSE.PE', 'NOTICE.PE.md'):
        shutil.copy2(OLD / f, dst / f)
    shutil.copy2(EXAMPLE / 'metadata.yaml', dst / 'metadata.yaml')
    for f in ('run.py', 't2fast.py', 't2post.py', 'siglip_vision.py'):
        shutil.copy2(PKG_SRC / f, dst / f)

    cfg_members, report = [], dict(name=a.name, members={})
    prior_logits = []
    for m in members:
        src = ck_paths[m]
        ck = torch.load(src, map_location='cpu', weights_only=False)
        pr = torch.as_tensor(ck['prior'], dtype=torch.float64).clamp(1e-4, 1 - 1e-4)
        prior_logits.append(torch.log(pr) - torch.log1p(-pr))
        spec = dict(old4[m]['spec'])
        entry = dict(name=m, weight=weights[m], checkpoint=f'assets/{m}.pt', spec=spec,
                     source_checkpoint=str(src), source_checkpoint_sha256=sha(src))
        if m.startswith('sl2'):
            tsd, vc, delta = siglip_pure_state(ck['tower_state_dict'], old4[m]['vision_config'], spec['patch'])
            entry.update(vision_config=vc, grid_hw=list(GRID))
            report['members'][m] = dict(pure_vs_hf_max_abs=delta)
        else:
            tsd = {k: v for k, v in ck['tower_state_dict'].items()}
        out = dict(tower_state_dict={k: v.half() if v.is_floating_point() else v for k, v in tsd.items()},
                   head_state_dict={k: v.float() for k, v in ck['head_state_dict'].items()})
        torch.save(out, dst / 'assets' / f'{m}.pt')
        entry['asset_sha256'] = sha(dst / 'assets' / f'{m}.pt')
        cfg_members.append(entry)
        print('member', m, 'from', src, flush=True)
        del ck, out, tsd
    post = dict(post, prior_logit=[float(v) for v in torch.stack(prior_logits).mean(0)])
    cfg = dict(name=a.name, attribute_names=attr, members=cfg_members, feat_members=feat, flip_tta=a.flip,
               batch=a.batch, decode_threads=16, budget_s=a.budget, post=post)
    (dst / 'assets' / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    # no forbidden imports in runtime files
    for p in dst.rglob('*.py'):
        txt = p.read_text(encoding='utf-8')
        for mod in ('transformers', 'huggingface_hub', 'safetensors', 'timm', 'einops', 'open_clip'):
            assert f'import {mod}' not in txt and f'from {mod}' not in txt, f'{p} imports {mod}'
    subs = WORK / 'submissions'
    subs.mkdir(exist_ok=True)
    zpath = subs / f'{a.name}.zip'
    tmp = zpath.with_suffix('.zip.partial')
    files = sorted(p for p in dst.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as z:
        for p in files:
            z.write(p, p.relative_to(dst).as_posix())
    tmp.replace(zpath)
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert 'run.py' in names and 'metadata.yaml' in names
    digest = sha(zpath)
    (subs / f'{a.name}.zip.sha256').write_text(f'{digest}  {zpath.name}\n', encoding='utf-8')
    report.update(zip=str(zpath), bytes=zpath.stat().st_size, sha256=digest, files=names, config=cfg)
    (subs / f'build_{a.name}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('BUILD OK', zpath, zpath.stat().st_size, digest, flush=True)


if __name__ == '__main__':
    main()
