"""Train a high-resolution encoder on TRAIN + VAL (24x24 patch grid, 576 tokens).

    python -u framework/recipes/scripts/train_hr_tv.py pel           # PE-Core L/14 at 336x336
    python -u framework/recipes/scripts/train_hr_tv.py sl2l          # SigLIP 2 L/16 at 384x384
  The two final packages (Track 1 and Track 2) both use these two encoders.
    python -u framework/recipes/scripts/train_hr_tv.py selftest      # optional: short GPU check, rewrites configs/hr_resources.json
    python -u framework/recipes/scripts/train_hr_tv.py calib_check   # optional: FP32 vs FP16 head-calibration check

Same recipe as train_tv.py. Only the input size changes, plus the sl2l checkpoint (google/siglip2-large-patch16-384).
The changes are applied in memory to the training core:
  - C.config(): sl2l uses the 384 checkpoint.
  - P.Transform.__init__: square input at the native size of the checkpoint.
  - A.Adapter.inspect_positions: expects a 24x24 grid.
  - E.calibrate: frozen features for the head calibration are computed under FP16 autocast (batch 64) when the
    FP16/FP32 check in configs/hr_calibration_check.json passes.
Micro-batch 16 with 4 accumulation steps (configs/hr_resources.json) keeps the effective batch at 64 on an 8 GB GPU.
Output: recipes/runs_hr_tv/allsource_<backbone>_hr_lc_m1_s42/epoch6.pt
"""
import copy
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parent
UF = ROOT / 'core'
sys.path.insert(0, str(UF))

from uf import common as C  # noqa: E402
from uf import protocol as P  # noqa: E402
from uf import adapters as A  # noqa: E402
from uf import engine as E  # noqa: E402
from teacher_system.text import fit_calibration  # noqa: E402

OUT_DIR = WORK / 'runs_hr_tv'
PREFLIGHT_HR = WORK / 'configs' / 'hr_resources.json'
CALIB_PARITY_HR = WORK / 'configs' / 'hr_calibration_check.json'


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


def verify_base_identity():
    ident, preflight = C.validate_milestones()
    return ident, preflight


def apply_hr_patches(use_fp16_calib=False, calib_batch=64):
    orig_config = C.config

    def hr_config():
        c = copy.deepcopy(orig_config())
        c['backbones']['sl2l'] = {
            'model': 'google/siglip2-large-patch16-384',
            'patch': 16,
            'native': 384,
        }
        return c

    C.config = hr_config

    def hr_transform_init(self, spec, train, aug='none'):
        if aug not in ('none', 'illum'):
            raise ValueError('Unknown augmentation: ' + aug)
        self.aug = aug
        native = int(spec['native'])
        if native not in (336, 384):
            raise ValueError(f'Expected native 336 or 384 for HR A1, got {native}')
        self.hw = (native, native)
        self.train = train
        self.degrader = P.SurveillanceTransform(self.hw)
        self.mean, self.std = spec['mean'], spec['std']

    P.Transform.__init__ = hr_transform_init

    import torch
    from torch import nn

    @torch.inference_mode()
    def hr_inspect_positions(self, tower, x):
        h, w = x.shape[-2:]
        gh, gw = h // self.spec['patch'], w // self.spec['patch']
        assert (gh, gw) == (24, 24), f'Expected 24x24 grid, got {(gh, gw)} for input {(h, w)}'
        if self.key.startswith('sl2'):
            tokens = tower.vision.embeddings(x, interpolate_pos_encoding=True)
            assert tokens.shape[1] == 576, f'Expected 576 tokens, got {tokens.shape[1]}'
            pos = tower.vision.embeddings.interpolate_pos_encoding(tokens, h, w)
        else:
            tokens = tower.conv1(x).flatten(2).transpose(1, 2)
            assert tokens.shape[1] == 576, f'Expected 576 tokens, got {tokens.shape[1]}'
            pos = tower._sample_abs_posemb(24, 24)
            tower.rope.update_grid(x.device, 24, 24)
        assert torch.isfinite(pos).all()
        return dict(
            patch_tokens=576,
            grid_hw=[24, 24],
            input_hw=[h, w],
            position_shape=list(pos.shape),
            cls_excluded_from_patch_count=not self.key.startswith('sl2'),
        )

    A.Adapter.inspect_positions = hr_inspect_positions

    def hr_calibrate(tower, dirs, table, spec, arm, micro, workers):
        """Identical to uf.engine.calibrate except amp/batch follow the pre-train parity gate."""
        train_paths = set(P.tables('allsource').image_paths)
        if not set(table.image_paths) <= train_paths:
            raise ValueError('Calibration must receive TRAIN rows only')
        head = nn.Linear(dirs.shape[1], 40, bias=False).cuda()
        with torch.no_grad():
            head.weight.copy_(dirs.cuda())
        tower.requires_grad_(False)
        eff_batch = calib_batch if use_fp16_calib else micro
        eff_amp = bool(use_fp16_calib)
        scores = E.infer(tower, head, P.loader(table, spec, batch=eff_batch, workers=workers), amp=eff_amp)
        a, b, convergence = fit_calibration(
            torch.from_numpy(scores),
            torch.from_numpy(table.labels),
            torch.from_numpy(P.masks(table, arm)),
        )
        del head
        head = A.head_from_dirs(dirs).cuda()
        with torch.no_grad():
            head.weight.copy_((a[:, None] * dirs).cuda())
            head.bias.copy_(b.cuda())
        tower.requires_grad_(True)
        return head, dict(
            convergence=convergence,
            rows=len(table.image_paths),
            source_train_sha256=C.sha(C.TRAIN),
            fitting_split='TRAIN',
            amp=eff_amp,
            batch=eff_batch,
        )

    E.calibrate = hr_calibrate


def verify_geometry_and_parity(backbone):
    import numpy as np
    import torch
    from PIL import Image

    E.seed()
    adapter = A.ADAPTERS[backbone]()
    table = P.tables('lodo_market')
    tower, dirs, source = adapter.load(table.attribute_names, 'cuda')
    head = A.head_from_dirs(dirs).cuda()
    tower.eval()
    with Image.open(C.DATA / table.image_paths[0]) as image:
        image = image.convert('RGB')
        transformed = P.Transform(adapter.spec, False)(image).unsqueeze(0).cuda()
        native_im = image.resize((adapter.spec['native'], adapter.spec['native']), Image.Resampling.BILINEAR)
        native = torch.from_numpy(np.asarray(native_im, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        native = ((native - torch.tensor(adapter.spec['mean'])[:, None, None]) /
                  torch.tensor(adapter.spec['std'])[:, None, None]).unsqueeze(0).cuda()
    assert tuple(transformed.shape[-2:]) == (adapter.spec['native'], adapter.spec['native'])
    assert torch.equal(transformed, native), 'Transform(False) must match native bilinear resize at A1 resolution'
    with torch.inference_mode():
        geometry = adapter.inspect_positions(tower, transformed)
        assert geometry['patch_tokens'] == 576 and geometry['grid_hw'] == [24, 24]
        out_a = tower(transformed)
        assert dirs.shape == (40, out_a.shape[1]) and torch.isfinite(dirs).all()
        geometry['head_shape'] = list(head.weight.shape)
        if backbone.startswith('sl2'):
            original = tower.vision(native, interpolate_pos_encoding=False).pooler_output
            torch.testing.assert_close(out_a, original, atol=1e-6, rtol=1e-6)
            geometry['native_forward_parity'] = float((out_a - original).abs().max())
        else:
            geometry['native_forward_parity'] = 0.0
    del transformed, native, out_a, tower, head, dirs
    gc.collect()
    torch.cuda.empty_cache()
    return geometry, source


def check_calib_parity_one(backbone, micro=16, workers=4, tv_rows=131076):
    import numpy as np
    import torch
    from torch import nn

    E.seed()
    adapter = A.ADAPTERS[backbone]()
    table = P.tables('lodo_market')
    ix = np.linspace(0, len(table.labels) - 1, 2000, dtype=int)
    sample = P.AnnotationTable(table.attribute_names, tuple(table.image_paths[i] for i in ix), table.labels[ix])
    mask = torch.from_numpy(P.masks(sample, 'lc_m1'))
    labels = torch.from_numpy(sample.labels)

    tower, dirs, _ = adapter.load(table.attribute_names, 'cuda')
    tower.requires_grad_(False)
    head = nn.Linear(dirs.shape[1], 40, bias=False).cuda()
    with torch.no_grad():
        head.weight.copy_(dirs.cuda())

    # Warmup 1 batch so CUDA kernels are initialized before timing
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=True):
        _ = head(tower(torch.zeros(4, 3, adapter.spec['native'], adapter.spec['native'], device='cuda')))
    torch.cuda.synchronize()

    # 1. FP32 extraction (batch=micro, amp=False)
    t0 = time.perf_counter()
    z32 = E.infer(tower, head, P.loader(sample, adapter.spec, batch=micro, workers=workers), amp=False)
    torch.cuda.synchronize()
    fp32_sec = time.perf_counter() - t0
    a32, b32, conv32 = fit_calibration(torch.from_numpy(z32), labels, mask)

    # 2. FP16 extraction (batch=64, amp=True)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    z16 = E.infer(tower, head, P.loader(sample, adapter.spec, batch=64, workers=workers), amp=True)
    torch.cuda.synchronize()
    fp16_sec = time.perf_counter() - t1
    fp16_reserved_gb = torch.cuda.max_memory_reserved() / 2**30
    a16, b16, conv16 = fit_calibration(torch.from_numpy(z16), labels, mask)

    del tower, head, dirs
    gc.collect()
    torch.cuda.empty_cache()

    da = (a16 - a32).abs()
    rel_da = da / a32.abs().clamp_min(1e-12)
    db = (b16 - b32).abs()
    z_diff = np.abs(z16 - z32)

    max_rel_da = float(rel_da.max().item())
    med_rel_da = float(rel_da.median().item())
    max_abs_db = float(db.max().item())
    med_abs_db = float(db.median().item())
    gate_pass = bool(max_rel_da <= 1e-2 and max_abs_db <= 1e-2)

    fp32_ips = 2000.0 / fp32_sec
    fp16_ips = 2000.0 / fp16_sec
    speedup = fp16_ips / fp32_ips

    # Full 131,076-row calibration ETA + 33,407-row heldout eval ETA (both run with amp=True in E.run)
    fit_sec_est = conv16.get('iterations', 100) * 0.02 * (tv_rows / 2000.0)
    eta_calib_fp16_h = (tv_rows / fp16_ips + fit_sec_est) / 3600.0
    eta_calib_fp32_h = (tv_rows / fp32_ips + fit_sec_est) / 3600.0
    eta_heldout_fp16_h = (33407.0 / fp16_ips + 30.0) / 3600.0

    res = dict(
        backbone=backbone,
        rows=2000,
        arm='lc_m1',
        gate_thresholds=dict(max_rel_da=1e-2, max_abs_db=1e-2),
        pass_=gate_pass,
        selected_amp='fp16' if gate_pass else 'fp32',
        selected_batch=64 if gate_pass else micro,
        max_rel_da=max_rel_da,
        median_rel_da=med_rel_da,
        max_abs_da=float(da.max().item()),
        median_abs_da=float(da.median().item()),
        max_abs_db=max_abs_db,
        median_abs_db=med_abs_db,
        score_max_abs_diff=float(z_diff.max()),
        score_median_abs_diff=float(np.median(z_diff)),
        fp32_seconds_2000=fp32_sec,
        fp32_img_per_s=fp32_ips,
        fp16_seconds_2000=fp16_sec,
        fp16_img_per_s=fp16_ips,
        fp16_speedup_x=speedup,
        fp16_reserved_peak_gb=fp16_reserved_gb,
        convergence_fp32=conv32,
        convergence_fp16=conv16,
        eta_calib_fp32_h=eta_calib_fp32_h,
        eta_calib_fp16_h=eta_calib_fp16_h,
        eta_heldout_fp16_h=eta_heldout_fp16_h,
    )
    print(
        f'=== CALIB PARITY {backbone}: PASS={gate_pass} | '
        f'max|da|/|a|={max_rel_da:.6f} (median={med_rel_da:.6f}) | '
        f'max|db|={max_abs_db:.6f} (median={med_abs_db:.6f}) | '
        f'FP32={fp32_ips:.2f} img/s ({eta_calib_fp32_h:.2f}h) -> '
        f'FP16(b=64)={fp16_ips:.2f} img/s ({eta_calib_fp16_h:.2f}h, {speedup:.2f}x faster) ===',
        flush=True,
    )
    return res


def run_calib_check(ident):
    import torch

    torch.set_num_threads(4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = merged_csv()
    tv_rows = len(merged.read_text(encoding='utf-8').splitlines()) - 1
    hr_pf = C.read(PREFLIGHT_HR) if PREFLIGHT_HR.exists() else None

    out = dict(
        status='RUNNING',
        framework_identity_code_sha=ident['code_sha'],
        train_val_rows=tv_rows,
        backbones={},
    )
    total_eta_h = 0.0
    for b in ('pel', 'sl2l'):
        micro = hr_pf['backbones'][b]['microbatch'] if hr_pf and b in hr_pf.get('backbones', {}) else 16
        res = check_calib_parity_one(b, micro=micro, workers=4, tv_rows=tv_rows)
        if hr_pf and b in hr_pf.get('backbones', {}):
            eta_train_h = hr_pf['backbones'][b]['eta_train_h']
            eta_cal_h = res['eta_calib_fp16_h'] if res['pass_'] else res['eta_calib_fp32_h']
            eta_total_h = eta_train_h + eta_cal_h + res['eta_heldout_fp16_h']
            res['eta_train_h'] = eta_train_h
            res['eta_total_h'] = eta_total_h
            total_eta_h += eta_total_h
            hr_pf['backbones'][b]['calib_parity'] = res
            hr_pf['backbones'][b]['eta_calib_extract_h'] = eta_cal_h
            hr_pf['backbones'][b]['eta_heldout_eval_h'] = res['eta_heldout_fp16_h']
            hr_pf['backbones'][b]['eta_total_h'] = eta_total_h
            print(
                f'=== {b} UPDATED ETA: train={eta_train_h:.2f}h + calib({res["selected_amp"]})={eta_cal_h:.2f}h '
                f'+ heldout(fp16)={res["eta_heldout_fp16_h"]:.2f}h => TOTAL={eta_total_h:.2f}h ===',
                flush=True,
            )
        out['backbones'][b] = res
    out['status'] = 'PASS'
    out['total_eta_h'] = total_eta_h
    C.write(CALIB_PARITY_HR, out)
    if hr_pf:
        hr_pf['total_eta_h'] = total_eta_h
        hr_pf['calib_parity_sha256'] = C.sha(CALIB_PARITY_HR)
        C.write(PREFLIGHT_HR, hr_pf)
    print(f'=== CALIB_CHECK DONE | UPDATED COMBINED ETA (pel_hr + sl2l_hr): {total_eta_h:.2f} hours ===', flush=True)


def run_selftest(ident, base_preflight):
    import torch

    torch.set_num_threads(4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = merged_csv()
    tv_rows = len(merged.read_text(encoding='utf-8').splitlines()) - 1
    results = dict(
        status='RUNNING',
        framework_identity_code_sha=ident['code_sha'],
        train_val_rows=tv_rows,
        backbones={},
    )
    for b in ('pel', 'sl2l'):
        print(f'=== SELFTEST & PROBE: {b} ===', flush=True)
        geometry, source = verify_geometry_and_parity(b)
        print(f'{b} geometry PASS:', json.dumps(geometry), flush=True)
        probe_res = E.probe(b, updates=20, selftest=False)
        probe_res['geometry'] = geometry
        probe_res['img_per_s_train'] = 1.0 / probe_res['s_per_sample']
        results['backbones'][b] = probe_res
    results['status'] = 'PASS'
    C.write(PREFLIGHT_HR, results)
    run_calib_check(ident)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('selftest', 'calib_check', 'pel', 'sl2l'):
        raise SystemExit('Usage: python -u scripts/train_hr_tv.py <selftest|calib_check|pel|sl2l>')
    arg = sys.argv[1]
    ident, base_preflight = verify_base_identity()
    apply_hr_patches(use_fp16_calib=False)

    if arg == 'selftest':
        run_selftest(ident, base_preflight)
        return
    if arg == 'calib_check':
        run_calib_check(ident)
        return

    backbone = arg
    if not PREFLIGHT_HR.exists() or not CALIB_PARITY_HR.exists():
        raise FileNotFoundError(f'Run selftest / calib_check first to create {PREFLIGHT_HR} and {CALIB_PARITY_HR}')
    hr_pf = C.read(PREFLIGHT_HR)
    cal_pf = C.read(CALIB_PARITY_HR)
    if hr_pf.get('status') != 'PASS' or backbone not in hr_pf.get('backbones', {}):
        raise ValueError(f'Invalid or incomplete {PREFLIGHT_HR}')
    if cal_pf.get('status') != 'PASS' or backbone not in cal_pf.get('backbones', {}):
        raise ValueError(f'Invalid or incomplete {CALIB_PARITY_HR}')

    b_cal = cal_pf['backbones'][backbone]
    use_fp16 = bool(b_cal['pass_'])
    apply_hr_patches(use_fp16_calib=use_fp16, calib_batch=64)

    merged = merged_csv()
    C.TRAIN = merged

    merged_pf = copy.deepcopy(base_preflight)
    merged_pf['backbones'][backbone] = hr_pf['backbones'][backbone]
    C.validate_milestones = lambda: (ident, merged_pf)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rel = Path('..') / '..' / WORK.name / 'runs_hr_tv'

    def hr_run_id(b, p, a, aug='none'):
        suffix = '' if aug == 'none' else '_aug_' + aug
        return str(rel / f'{p}_{b}_hr_{a}{suffix}_s42')

    C.run_id = hr_run_id

    spec = C.config()['backbones'][backbone]
    b_pf = hr_pf['backbones'][backbone]
    note = dict(
        backbone=backbone,
        run_dir=f'allsource_{backbone}_hr_lc_m1_s42',
        model_id=spec['model'],
        patch=spec['patch'],
        input_hw=[spec['native'], spec['native']],
        grid_hw=[24, 24],
        patch_tokens=576,
        microbatch=b_pf['microbatch'],
        accumulation=b_pf['accumulation'],
        calibration_amp='fp16' if use_fp16 else 'fp32',
        calibration_batch=64 if use_fp16 else b_pf['microbatch'],
        calibration_parity_gate=dict(
            pass_=use_fp16,
            max_rel_da=b_cal['max_rel_da'],
            median_rel_da=b_cal['median_rel_da'],
            max_abs_db=b_cal['max_abs_db'],
            median_abs_db=b_cal['median_abs_db'],
            fp16_img_per_s=b_cal['fp16_img_per_s'],
            fp32_img_per_s=b_cal['fp32_img_per_s'],
        ),
        train_csv=str(merged),
        train_csv_sha256=hashlib.sha256(merged.read_bytes()).hexdigest(),
        rows=len(merged.read_text(encoding='utf-8').splitlines()) - 1,
        recipe='UF-v1 allsource lc_m1 A1 High-Res (576 tokens), unchanged hyperparameters',
        framework_identity_code_sha=ident['code_sha'],
    )
    C.write(OUT_DIR / f'NOTE_{backbone}_hr.json', note)
    print(json.dumps(note, indent=2), flush=True)

    import torch

    torch.set_num_threads(4)
    E.run(backbone, 'allsource', 'lc_m1', None, 'none')
    print('TRAIN_HR_TV DONE', backbone, flush=True)


if __name__ == '__main__':
    main()
