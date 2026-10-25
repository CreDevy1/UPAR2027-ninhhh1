"""Build the Track 1 code-submission package (submission 1007246) from the pel_hr checkpoint.

    python -u framework/recipes/scripts/build_hr_pel_pkg.py

CPU only, one thread, no model forward. Reads recipes/runs_hr_tv/allsource_pel_hr_lc_m1_s42/epoch6.pt
and writes recipes/build/t1_pel_hr_tv_A1_20261021/ and recipes/submissions/t1_pel_hr_tv_A1_20261021.zip.
Existing outputs are never overwritten.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1', UPAR_FORCE_CPU='1',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
import ast
import gc
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
sys.dont_write_bytecode = True
WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parent
RUN = WORK / 'runs_hr_tv/allsource_pel_hr_lc_m1_s42'
TEMPLATE = ROOT.parent / 'inference' / 'track1'      # runtime files of the scored package
RUNTIME = TEMPLATE
BASE = TEMPLATE
T1BASE = TEMPLATE


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, obj):
    p.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    assert not torch.cuda.is_available(), 'Build must not see CUDA'
    assert read(RUN / 'STATUS.json')['status'] == 'DONE'
    note = read(WORK / 'runs_hr_tv/NOTE_pel_hr.json')
    assert note['input_hw'] == [336, 336] and note['grid_hw'] == [24, 24]
    source = RUN / 'epoch6.pt'
    source_sha = sha(source)
    assert source_sha == read(RUN / 'result.json')['checkpoint_sha256']
    ck = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    assert ck['selected_epoch'] == 6 and ck['n_update'] == 3378
    assert all(t.device.type == 'cpu' for t in ck['tower_state_dict'].values())
    assert all(t.dtype == torch.float32 for t in ck['tower_state_dict'].values() if t.is_floating_point())
    prior = torch.as_tensor(ck['prior'], dtype=torch.float64).clamp(1e-4, 1-1e-4)
    assert prior.shape == (40,)
    ref = read(T1BASE / 'assets/config.json')
    member = dict(name='pel_hr', weight=1.0, checkpoint='assets/pel_hr.pt',
                  spec=dict(model=note['model_id'], patch=14, native=336),
                  input_hw=[336, 336], grid_hw=[24, 24], patch_tokens=576,
                  train_prior=prior.tolist(), dtype='float32',
                  source_checkpoint=str(source), source_checkpoint_sha256=source_sha)
    exported = dict(tower_state_dict=ck['tower_state_dict'], head_state_dict=ck['head_state_dict'])
    asset = None
    (WORK / 'submissions').mkdir(exist_ok=True)
    for track in (1,):
        name = f't{track}_pel_hr_tv_A1_20261021'
        dst = WORK / 'build' / name
        archive = WORK / 'submissions' / (name + '.zip')
        if dst.exists() or archive.exists():
            raise FileExistsError(f'Will not overwrite {dst} / {archive}')
        (dst / 'assets').mkdir(parents=True)
        shutil.copytree(BASE / 'teacher_system', dst / 'teacher_system',
                        ignore=shutil.ignore_patterns('__pycache__'))
        for f in ('LICENSE.PE', 'NOTICE.PE.md'):
            shutil.copy2(BASE / f, dst / f)
        shutil.copy2(TEMPLATE / 'metadata.yaml', dst / 'metadata.yaml')
        shutil.copy2(RUNTIME / 'hr_infer.py', dst / 'hr_infer.py')
        shutil.copy2(RUNTIME / 'run.py', dst / 'run.py')
        target = dst / member['checkpoint']
        if asset is None:
            torch.save(exported, target)
            asset = target
            member['asset_sha256'] = sha(asset)
        else:
            shutil.copy2(asset, target)
            assert sha(target) == member['asset_sha256']
        cfg = dict(name=name, attribute_names=ref['attribute_names'], members=[member],
                   batch=1024, decode_threads=16, flip_tta=False,
                   runtime=dict(time_guard=False, prior_fill=False, drop_members=False,
                                oom_retry='halve batch and retry same images',
                                weight_dtype='float32', amp='cuda fp16',
                                features='pel_hr only', validated_gpu=False))
        if track == 1:
            shutil.copy2(T1BASE / 'pmcd.py', dst / 'pmcd.py')
            cfg['track1'] = dict(decoder='pmcd', alpha=1.0,
                                pmcd_reference_prior=ref['track1']['pmcd_reference_prior'],
                                reference_prior_source=ref['track1']['reference_prior_source'])
        write(dst / 'assets/config.json', cfg)
        write(dst / 'assets/training_note.json', note)
        manifest = {}
        for p in sorted(dst.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:
                continue
            rel = p.relative_to(dst).as_posix()
            if p.suffix == '.py':
                tree = ast.parse(p.read_text(encoding='utf-8'))
                for node in ast.walk(tree):
                    mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                            else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                    assert not any(m.split('.')[0] in {'transformers', 'timm', 'open_clip',
                                                      'huggingface_hub', 'safetensors', 'einops'} for m in mods)
            manifest[rel] = sha(p)
        partial = archive.with_suffix('.zip.partial')
        with zipfile.ZipFile(partial, 'x', zipfile.ZIP_STORED, allowZip64=True) as z:
            for rel in manifest:
                z.write(dst / rel, rel)
        with zipfile.ZipFile(partial) as z:
            assert z.testzip() is None
            assert set(z.namelist()) == set(manifest)
            assert 'run.py' in manifest and 'metadata.yaml' in manifest
        partial.replace(archive)
        digest = sha(archive)
        archive.with_suffix('.zip.sha256').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
        write(WORK / 'submissions' / f'build_{name}.json',
              dict(zip=str(archive), bytes=archive.stat().st_size, sha256=digest,
                   source_checkpoint_sha256=source_sha, manifest=manifest, config=cfg,
                   build_cuda_visible=False, build_torch_threads=1,
                   forward_executed=False, runtime_benchmark_executed=False,
                   image_parity_executed=False, checks='checkpoint hash, completion/epoch/update count, Python AST/imports, archive CRC'))
        print('BUILD OK', archive, archive.stat().st_size, digest, flush=True)
        gc.collect()
    print('Package complete; no GPU or model forward used.', flush=True)


if __name__ == '__main__':
    main()
