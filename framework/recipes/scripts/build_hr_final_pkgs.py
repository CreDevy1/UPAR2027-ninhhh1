"""Build the two final code-submission packages from the two high-resolution checkpoints.

    python -u framework/recipes/scripts/build_hr_final_pkgs.py            # both packages
    python -u framework/recipes/scripts/build_hr_final_pkgs.py t1         # Track 1 only
    python -u framework/recipes/scripts/build_hr_final_pkgs.py t2         # Track 2 only

  Track 1  t1_ens2_hr_tv_pmcd_20261021.zip  (Codabench submission 1011642)
  Track 2  t2_v5a_hr_fp32.zip               (Codabench submission 1011654)

Both packages use the same two encoders, fine-tuned on TRAIN+VAL:
  pel   PE-Core L/14 at 336x336   framework/recipes/runs_hr_tv/allsource_pel_hr_lc_m1_s42/epoch6.pt
  sl2l  SigLIP 2 L/16 at 384x384  framework/recipes/runs_hr_tv/allsource_sl2l_hr_lc_m1_s42/epoch6.pt
Weights are stored in FP32; inference runs under FP16 autocast on CUDA.

Runtime code and the configuration template come from inference/track1 and inference/track2. The script
recomputes everything that depends on the checkpoints (weights, training priors, the Track 2 prior logit), so
tools/verify_rebuild.py can compare the result with the submitted zips. CPU only, no model forward, no
transformers. Existing outputs are never overwritten.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1')
import ast
import gc
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import torch

sys.dont_write_bytecode = True
WORK = Path(__file__).resolve().parents[1]          # framework/recipes
REPO = WORK.parents[1]
INF = REPO / 'inference'
RUNS = WORK / 'runs_hr_tv'
CKPT = {'pel': RUNS / 'allsource_pel_hr_lc_m1_s42' / 'epoch6.pt',
        'sl2l': RUNS / 'allsource_sl2l_hr_lc_m1_s42' / 'epoch6.pt'}
CKPT_SHA = {'pel': '1734cca370d80e26dd2b912006e69bfa68ce5c400f8a195a5645b3fcb6899d24',
            'sl2l': 'e12b900310476d333e78e6284398ae992e46e955fe9127f0a01c2ca9e352fc9e'}
FORBIDDEN = {'transformers', 'timm', 'open_clip', 'sentencepiece', 'huggingface_hub', 'safetensors', 'einops'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_ckpt(m):
    digest = sha(CKPT[m])
    assert digest == CKPT_SHA[m], f'{CKPT[m]}: unexpected SHA-256 {digest}'
    ck = torch.load(CKPT[m], map_location='cpu', weights_only=False)
    assert ck['n_update'] == 3378, ck['n_update']
    return ck, digest


def fp32(sd):
    return {k: (v.float() if v.is_floating_point() else v).contiguous() for k, v in sd.items()}


def siglip_pure_state(tower_sd, template_dir, vc, grid):
    """HF SigLIP vision weights -> plain PyTorch tower of the Track 2 package (24x24 grid, native position
    embedding, no interpolation)."""
    sys.path.insert(0, str(template_dir))
    from siglip_vision import SiglipVisionTower
    sys.path.pop(0)
    sd = {k[len('vision.'):]: v.float() for k, v in tower_sd.items()}
    sd.pop('embeddings.position_ids', None)
    sd['embeddings.fixed_pos'] = sd['embeddings.position_embedding.weight'].unsqueeze(0).clone()
    tower = SiglipVisionTower(vc['image_size'], 16, vc['hidden_size'], vc['num_hidden_layers'],
                              vc['num_attention_heads'], vc['intermediate_size'], vc['layer_norm_eps'], tuple(grid))
    tower.load_state_dict(sd, strict=True)
    return {k: v.contiguous() for k, v in tower.state_dict().items()}


def stage_code(template, dst):
    (dst / 'assets').mkdir(parents=True)
    for p in sorted(template.rglob('*')):
        rel = p.relative_to(template)
        if p.is_dir() or '__pycache__' in rel.parts or rel.as_posix() == 'assets/config.json':
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / rel)


def finish(name, dst, cfg):
    (dst / 'assets' / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    for p in dst.rglob('*.py'):
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
            mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module or ''] if isinstance(node, ast.ImportFrom) and node.level == 0 else [])
            bad = {m.split('.')[0] for m in mods} & FORBIDDEN
            assert not bad, f'{p} imports {bad}, not available on the grader'
    subs = WORK / 'submissions'
    subs.mkdir(exist_ok=True)
    archive = subs / f'{name}.zip'
    partial = archive.with_suffix('.zip.partial')
    files = sorted(p for p in dst.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    with zipfile.ZipFile(partial, 'x', zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in files:
            z.write(p, p.relative_to(dst).as_posix())
    with zipfile.ZipFile(partial) as z:
        assert z.testzip() is None
        assert {'run.py', 'metadata.yaml', 'assets/config.json'} <= set(z.namelist())
    partial.replace(archive)
    digest = sha(archive)
    archive.with_suffix('.zip.sha256').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    print('BUILD OK', archive, archive.stat().st_size, digest, flush=True)


def prepare(name):
    dst = WORK / 'build' / name
    archive = WORK / 'submissions' / f'{name}.zip'
    if dst.exists() or archive.exists():
        raise FileExistsError(f'Will not overwrite {dst} / {archive}')
    return dst


def build_t1():
    name = 't1_ens2_hr_tv_pmcd_20261021'
    template = INF / 'track1'
    dst = prepare(name)
    cfg = json.loads((template / 'assets' / 'config.json').read_text(encoding='utf-8'))
    stage_code(template, dst)
    prior = None
    for entry in cfg['members']:
        m = entry['name']
        ck, digest = load_ckpt(m)
        p = [float(x) for x in ck['prior']]
        if prior is None:
            prior = p
        else:   # both encoders see the same TRAIN+VAL labels
            assert max(abs(a - b) for a, b in zip(prior, p)) < 1e-7
        tower = fp32(ck['tower_state_dict'])
        tower.pop('vision.embeddings.position_ids', None)
        out = dict(tower_state_dict=tower, head_state_dict=fp32(ck['head_state_dict']))
        target = dst / entry['checkpoint']
        torch.save(out, target)
        entry.update(train_prior=prior, source_checkpoint=CKPT[m].relative_to(REPO).as_posix(),
                     source_checkpoint_sha256=digest, asset_sha256=sha(target), dtype='float32')
        print('T1 member', m, 'asset', entry['asset_sha256'], flush=True)
        del ck, tower, out
        gc.collect()
    cfg['track1']['pmcd_reference_prior'] = prior
    finish(name, dst, cfg)


def build_t2():
    name = 't2_v5a_hr_fp32'
    template = INF / 'track2'
    dst = prepare(name)
    cfg = json.loads((template / 'assets' / 'config.json').read_text(encoding='utf-8'))
    stage_code(template, dst)
    prior_logits = []
    for entry in cfg['members']:
        m = entry['name']
        ck, digest = load_ckpt(m)
        pr = torch.as_tensor(ck['prior'], dtype=torch.float64).clamp(1e-4, 1 - 1e-4)
        prior_logits.append(torch.log(pr) - torch.log1p(-pr))
        if m == 'sl2l':
            tower = siglip_pure_state(ck['tower_state_dict'], template, entry['vision_config'], entry['grid_hw'])
        else:
            tower = dict(ck['tower_state_dict'])
        out = dict(tower_state_dict=fp32(tower), head_state_dict=fp32(ck['head_state_dict']))
        target = dst / entry['checkpoint']
        torch.save(out, target)
        entry.update(source_checkpoint=CKPT[m].relative_to(REPO).as_posix(),
                     source_checkpoint_sha256=digest, asset_sha256=sha(target))
        print('T2 member', m, 'asset', entry['asset_sha256'], flush=True)
        del ck, tower, out
        gc.collect()
    cfg['post']['prior_logit'] = [float(v) for v in torch.stack(prior_logits).mean(0)]
    finish(name, dst, cfg)


if __name__ == '__main__':
    which = sys.argv[1:] or ['t1', 't2']
    assert set(which) <= {'t1', 't2'}, 'usage: build_hr_final_pkgs.py [t1] [t2]'
    if 't1' in which:
        build_t1()
    if 't2' in which:
        build_t2()
