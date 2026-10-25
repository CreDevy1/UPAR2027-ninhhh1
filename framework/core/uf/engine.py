"""Shared calibration, optimization, and final-epoch evaluation for all adapters."""
import gc
import math
import random
import time
import numpy as np
import torch
from torch import nn
from . import common as C, protocol as P
from .adapters import ADAPTERS, head_from_dirs
from .scoring import score
from teacher_system.text import fit_calibration


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def optimizer(tower, head):
    c = C.config()
    groups = []
    for model, lr in ((tower, c['backbone_lr']), (head, c['head_lr'])):
        for decay in (False, True):
            params = [p for p in model.parameters() if p.requires_grad and (p.ndim > 1) == decay]
            if params:
                groups.append(dict(params=params, lr=lr, weight_decay=c['weight_decay'] if decay else 0.))
    opt = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8, foreach=False)
    total = c['n_update']
    warmup = max(1, round(total*c['warmup_ratio']))
    def schedule(s):
        if s < warmup:
            return (s+1)/warmup
        progress = min(1., (s-warmup)/max(1, total-warmup))
        return c['cosine_min_ratio'] + (1-c['cosine_min_ratio'])*.5*(1+math.cos(math.pi*progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    scaler = torch.amp.GradScaler('cuda', init_scale=c['amp_initial_scale'])
    return opt, sched, scaler


def window_bce(logits, labels, valid, denominator):
    return (nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction='none')
            * valid).sum()/denominator


def update(tower, head, batch, micro, opt, sched, scaler, valid_all, capture=False):
    pixels, labels, _, rows = batch
    valid = torch.from_numpy(valid_all[rows.numpy()])
    denominator = int(valid.sum())
    assert denominator > 0
    opt.zero_grad(set_to_none=True)
    value = 0.
    for start in range(0, len(pixels), micro):
        x, y, v = (pixels[start:start+micro].cuda(), labels[start:start+micro].cuda(), valid[start:start+micro].cuda())
        with torch.autocast('cuda', dtype=torch.float16):
            logits = head(tower(x))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite AMP logits')
        loss = window_bce(logits, y, v, denominator)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite BCE')
        scaler.scale(loss).backward()
        value += float(loss.detach())
    scaler.unscale_(opt)
    params = [p for g in opt.param_groups for p in g['params']]
    if any(p.grad is None for p in head.parameters()):
        raise RuntimeError('Head missing gradient')
    norm = nn.utils.clip_grad_norm_(params, C.config()['clip_grad_norm'])
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite gradients')
    signature = None
    if capture:
        import hashlib
        hashes = {}
        for prefix, model in [('tower', tower), ('head', head)]:
            for name, p in model.named_parameters():
                hashes[prefix+'.'+name] = (hashlib.sha256(p.grad.detach().float().cpu().numpy().tobytes()).hexdigest()
                                          if p.grad is not None else None)
        signature = hashes
    before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    if scaler.get_scale() < before:
        raise FloatingPointError('AMP skipped update')
    sched.step()
    opt.zero_grad(set_to_none=True)
    result = dict(loss=value, grad_norm=float(norm), amp_scale=float(scaler.get_scale()))
    if capture:
        result['grad_sha256'] = signature
    return result


def update_pass(tower, head, batch, micro, opt, sched, scaler, valid_all, candidates_all, capture=False):
    pixels, labels, _, rows = batch
    valid = torch.from_numpy(valid_all[rows.numpy()])
    from .pass_loss import pass_loss, effective_valid
    candidates = torch.from_numpy(candidates_all[rows.numpy()])
    denominator = int(effective_valid(valid, candidates).sum())
    assert denominator > 0
    opt.zero_grad(set_to_none=True)
    value = 0.
    for start in range(0, len(pixels), micro):
        x, y, v = (pixels[start:start+micro].cuda(), labels[start:start+micro].cuda(), valid[start:start+micro].cuda())
        with torch.autocast('cuda', dtype=torch.float16):
            logits = head(tower(x))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite AMP logits')
        loss = pass_loss(logits.float(), y.float(), v, candidates[start:start+micro].to(logits.device), denominator)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite BCE')
        scaler.scale(loss).backward()
        value += float(loss.detach())
    scaler.unscale_(opt)
    params = [p for g in opt.param_groups for p in g['params']]
    if any(p.grad is None for p in head.parameters()):
        raise RuntimeError('Head missing gradient')
    norm = nn.utils.clip_grad_norm_(params, C.config()['clip_grad_norm'])
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite gradients')
    signature = None
    if capture:
        import hashlib
        hashes = {}
        for prefix, model in [('tower', tower), ('head', head)]:
            for name, p in model.named_parameters():
                hashes[prefix+'.'+name] = (hashlib.sha256(p.grad.detach().float().cpu().numpy().tobytes()).hexdigest()
                                          if p.grad is not None else None)
        signature = hashes
    before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    if scaler.get_scale() < before:
        raise FloatingPointError('AMP skipped update')
    sched.step()
    opt.zero_grad(set_to_none=True)
    result = dict(loss=value, grad_norm=float(norm), amp_scale=float(scaler.get_scale()))
    if capture:
        result['grad_sha256'] = signature
    return result


@torch.inference_mode()
def infer(tower, head, batches, amp=True):
    tower.eval()
    head.eval()
    out, paths = [], []
    for x, _, pp, _ in batches:
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            z = head(tower(x.cuda())).float().cpu().numpy()
        if not np.isfinite(z).all():
            raise FloatingPointError('Nonfinite inference')
        out.append(z)
        paths.extend(pp)
    if paths != list(batches.dataset.table.image_paths):
        raise ValueError('Inference order changed')
    return np.concatenate(out).astype(np.float32)


def calibrate(tower, dirs, table, spec, arm, micro, workers):
    """Fitting API takes source TRAIN only; no held-out/VAL argument exists."""
    train_paths = set(P.tables('allsource').image_paths)
    if not set(table.image_paths) <= train_paths:
        raise ValueError('Calibration must receive TRAIN rows only')
    head = nn.Linear(dirs.shape[1], 40, bias=False).cuda()
    with torch.no_grad():
        head.weight.copy_(dirs.cuda())
    tower.requires_grad_(False)
    scores = infer(tower, head, P.loader(table, spec, batch=micro, workers=workers), amp=False)
    a, b, convergence = fit_calibration(torch.from_numpy(scores), torch.from_numpy(table.labels),
                                        torch.from_numpy(P.masks(table, arm)))
    del head
    head = head_from_dirs(dirs).cuda()
    with torch.no_grad():
        head.weight.copy_((a[:, None]*dirs).cuda())
        head.bias.copy_(b.cuda())
    tower.requires_grad_(True)
    return head, dict(convergence=convergence, rows=len(table.image_paths),
                      source_train_sha256=C.sha(C.TRAIN), fitting_split='TRAIN', amp=False)


def calibrate_pass(tower, dirs, table, spec, arm, micro, workers):
    """Fitting API takes source TRAIN only; no held-out/VAL argument exists."""
    train_paths = set(P.tables('allsource').image_paths)
    if not set(table.image_paths) <= train_paths:
        raise ValueError('Calibration must receive TRAIN rows only')
    head = nn.Linear(dirs.shape[1], 40, bias=False).cuda()
    with torch.no_grad():
        head.weight.copy_(dirs.cuda())
    tower.requires_grad_(False)
    scores = infer(tower, head, P.loader(table, spec, batch=micro, workers=workers), amp=False)
    from .pass_loss import fit_calibration as fit_pass, candidate_sets
    from teacher_system.calibration import domain_from_path
    candidates = candidate_sets(table.labels, [domain_from_path(p) for p in table.image_paths])
    a, b, convergence = fit_pass(torch.from_numpy(scores), torch.from_numpy(table.labels),
                                        torch.from_numpy(P.masks(table, arm)), torch.from_numpy(candidates))
    del head
    head = head_from_dirs(dirs).cuda()
    with torch.no_grad():
        head.weight.copy_((a[:, None]*dirs).cuda())
        head.bias.copy_(b.cuda())
    tower.requires_grad_(True)
    return head, dict(convergence=convergence, rows=len(table.image_paths),
                      source_train_sha256=C.sha(C.TRAIN), fitting_split='TRAIN', amp=False)


class ReservedLimitExceeded(RuntimeError):
    def __init__(self, reserved):
        self.reserved = reserved
        super().__init__(f'Reserved peak {reserved:.3f} GiB exceeds 7.0 GiB')


def probe(backbone, updates, workers=None, micro=None, selftest=False):
    """Largest fitting microbatch; two smoke updates then twenty timed updates."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for real update/resource measurements')
    workers = C.loader_workers(workers)
    candidates = [micro] if micro else [64, 32, 16, 8, 4, 2, 1]
    attempts = []
    for mb in candidates:
        try:
            result = _probe(backbone, updates, workers, mb, [], selftest)
        except (torch.cuda.OutOfMemoryError, ReservedLimitExceeded) as exc:
            attempts.append(dict(microbatch=mb, status='REJECTED', reason=str(exc),
                                 reserved_peak_gb=getattr(exc, 'reserved', torch.cuda.max_memory_reserved()/2**30)))
            print(backbone, 'candidate rejected', attempts[-1], flush=True)
        else:
            attempts.append(dict(microbatch=mb, status='PASS', reserved_peak_gb=result['reserved_peak_gb']))
            result.update(attempts=attempts, workers=workers, reserved_limit_gb=7.0,
                          oom_retries=[r for r in attempts if r['status']=='REJECTED'])
            return result
        # Exception traceback and all failed model references are released first.
        gc.collect()
        torch.cuda.empty_cache()
    raise RuntimeError(f'No candidate fits resource limit: {attempts}')


def _probe(backbone, updates, workers, micro, failures, selftest=False):
    seed()
    load_begin = time.perf_counter()
    adapter = ADAPTERS[backbone]()
    table = P.tables('lodo_market')
    tower, dirs, source = adapter.load(table.attribute_names, 'cuda')
    head = head_from_dirs(dirs).cuda()
    load_seconds = time.perf_counter()-load_begin
    geometry = None
    if selftest:
        import json
        from PIL import Image
        tower.eval()
        with Image.open(C.DATA / table.image_paths[0]) as image:
            image = image.convert('RGB')
            small = P.Transform(adapter.spec, False)(image).unsqueeze(0).cuda()
            native = image.resize((adapter.spec['native'], adapter.spec['native']), Image.Resampling.BILINEAR)
            native = torch.from_numpy(np.asarray(native, dtype=np.float32).copy()).permute(2,0,1)/255.
            native = ((native-torch.tensor(adapter.spec['mean'])[:,None,None])/
                      torch.tensor(adapter.spec['std'])[:,None,None]).unsqueeze(0).cuda()
        with torch.inference_mode():
            geometry = adapter.inspect_positions(tower, small)
            a, b = tower(small), tower(native)
            geometry['frozen_native_cosine_no_gate'] = float(nn.functional.cosine_similarity(a, b).item())
            assert dirs.shape == (40, a.shape[1]) and torch.isfinite(dirs).all()
            geometry['head_shape'] = list(head.weight.shape)
            if backbone.startswith('sl2'):
                original = tower.vision(native, interpolate_pos_encoding=False).pooler_output
                torch.testing.assert_close(b, original, atol=1e-6, rtol=1e-6)
                geometry['native_forward_parity'] = float((b-original).abs().max())
            else:
                # Adapter wraps the unchanged native PE implementation itself.
                geometry['native_forward_parity'] = 0.
        del small, native, a, b
    tower.requires_grad_(True)
    tower.set_grad_checkpointing(True)
    tower.train()
    head.train()
    opt, sched, scaler = optimizer(tower, head)
    masks = P.masks(table)
    torch.cuda.reset_peak_memory_stats()
    records, smoke, samples = [], [], 0
    extra = 0 if selftest else 2
    begin = time.perf_counter()
    loop_begin = begin
    for i, batch in enumerate(P.loader(table, adapter.spec, True, batch=64, workers=workers)):
        if i == updates + extra:
            break
        torch.cuda.synchronize()
        start = time.perf_counter()
        r = update(tower, head, batch, micro, opt, sched, scaler, masks, capture=selftest and i == 0)
        if selftest and i == 0:
            folder = C.ROOT / 'reports/regression_parts'
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f'{backbone}.npz', pixels=batch[0].numpy(), labels=batch[1].numpy(),
                                paths=np.asarray(batch[2]), valid_mask=masks[batch[3].numpy()],
                                loss=np.asarray(r['loss']), grad_norm=np.asarray(r['grad_norm']),
                                grad_sha256_json=np.asarray(json.dumps(r.pop('grad_sha256'), sort_keys=True)),
                                microbatch=np.asarray(micro), seed=np.asarray(42))
        torch.cuda.synchronize()
        r.update(update=i+1, seconds=time.perf_counter()-start, samples=len(batch[0]))
        reserved = torch.cuda.max_memory_reserved()/2**30
        if reserved > 7.0:
            raise ReservedLimitExceeded(reserved)
        if i < extra:
            smoke.append(r)
            begin = time.perf_counter()
        else:
            records.append(r)
            samples += len(batch[0])
        print(backbone, 'smoke' if i < extra else 'probe', i+1, '/', updates+extra, r, flush=True)
    assert len(records) == updates and sched.last_epoch == updates+extra
    elapsed = time.perf_counter()-begin
    # Include loading/augmentation overhead within the measured loop; initialization separate.
    s = elapsed/samples
    result = dict(status='PASS', updates=records, smoke_updates=smoke, microbatch=micro, accumulation=64//micro,
                  load_seconds=load_seconds, warmup_and_loader_start_seconds=begin-loop_begin,
                  s_per_sample=s, vram_peak_gb=torch.cuda.max_memory_allocated()/2**30,
                  reserved_peak_gb=torch.cuda.max_memory_reserved()/2**30,
                  total_vram_gb=torch.cuda.get_device_properties(0).total_memory/2**30,
                  eta_train_h=s*216000/3600, eta_excludes='checkpoint load, full TRAIN calibration, final export/bootstrap',
                  head='own text directions, uncalibrated probe; full TRAIN calibration is part of run',
                  oom_retries=failures, source=source, geometry=geometry, trained_weights_discarded=True)
    del opt, sched, scaler, batch
    gc.collect()
    torch.cuda.empty_cache()
    if not selftest:
        result['overhead'] = measure_overhead(tower, dirs, table, adapter.spec, micro, workers)
    del tower, head
    gc.collect()
    torch.cuda.empty_cache()
    return result


def measure_overhead(tower, dirs, table, spec, micro, workers):
    """Disposable TRAIN-only timing; no held-out images or labels are loaded."""
    ix = np.linspace(0, len(table.labels)-1, 2000, dtype=int)
    sample = P.AnnotationTable(table.attribute_names, tuple(table.image_paths[i] for i in ix), table.labels[ix])
    tower.requires_grad_(False)
    head = nn.Linear(dirs.shape[1], 40, bias=False).cuda()
    with torch.no_grad():
        head.weight.copy_(dirs.cuda())
    torch.cuda.synchronize()
    start = time.perf_counter()
    z = infer(tower, head, P.loader(sample, spec, batch=micro, workers=workers), amp=False)
    torch.cuda.synchronize()
    feature_seconds = time.perf_counter()-start
    fitting = {}
    for arm in ('base', 'lc_m1'):
        start = time.perf_counter()
        _, _, convergence = fit_calibration(torch.from_numpy(z), torch.from_numpy(sample.labels),
                                             torch.from_numpy(P.masks(sample, arm)))
        fitting[arm] = dict(seconds=time.perf_counter()-start, convergence=convergence)
    start = time.perf_counter()
    score(z, sample.labels, P.masks(sample), P.prior(table, 'base'), sample.attribute_names)
    score_seconds = time.perf_counter()-start
    return dict(rows=2000, split='TRAIN', sampling='evenly spaced source lodo_market TRAIN rows',
                image_paths_sha256=__import__('hashlib').sha256('\n'.join(sample.image_paths).encode()).hexdigest(),
                feature_seconds=feature_seconds, feature_s_per_sample=feature_seconds/2000,
                feature_amp=False, includes_loader_start_and_shutdown=True, fitting=fitting,
                scoring_seconds=score_seconds, bootstrap_samples=C.config()['bootstrap_samples'],
                estimator='linear row extrapolation; FP32 features conservatively also price AMP held-out export',
                residual_unmeasured='checkpoint/cache disk writes and full-size LBFGS convergence variation')


def run(backbone, protocol, arm, workers=None, aug='none'):
    if arm == 'pass' and aug != 'none':
        raise ValueError('PASS and TCA are separate arms')
    if arm == 'pass' and not (C.ROOT/'runs'/C.run_id(backbone,protocol,'base')/'heldout.npz').is_file():
        raise FileNotFoundError('Run the same-fold base before PASS')
    ident, preflight = C.validate_milestones()
    workers = C.loader_workers(workers)
    resource = preflight['backbones'][backbone]
    micro = resource['microbatch']
    out = C.ROOT / 'runs' / C.run_id(backbone, protocol, arm, aug)
    if out.exists():
        raise FileExistsError('Preserve existing complete/partial run: ' + str(out))
    out.mkdir(parents=True)
    seed()
    c = C.config()
    begin = time.perf_counter()
    try:
        adapter = ADAPTERS[backbone]()
        table = P.tables(protocol)
        tower, dirs, source = adapter.load(table.attribute_names, 'cuda')
        calibrator = calibrate_pass if arm == 'pass' else calibrate
        head, calibration = calibrator(tower, dirs, table, adapter.spec, arm, micro, workers)
        tower.set_grad_checkpointing(True)
        opt, sched, scaler = optimizer(tower, head)
        valid = P.masks(table, arm)
        label_prior = P.prior(table, 'base' if arm == 'pass' else arm)
        prior = label_prior.copy()
        candidates = None
        if arm == 'pass':
            from .pass_loss import candidate_sets
            from teacher_system.calibration import domain_from_path
            candidates = candidate_sets(table.labels, [domain_from_path(p) for p in table.image_paths])
        provenance = dict(identity=ident, source=source, calibration=calibration, config=c,
                          microbatch=micro, accumulation=64//micro, loader_workers=workers, arm=arm, aug=aug)
        C.write(out / 'provenance.json', provenance)
        torch.cuda.reset_peak_memory_stats()
        records, sampled_seconds = [], 0.
        for epoch in range(1, c['epochs']+1):
            tower.train()
            head.train()
            t0 = time.perf_counter()
            count = 0
            for i, batch in enumerate(P.loader(table, adapter.spec, True, epoch, workers=workers, aug=aug), 1):
                if arm == 'pass':
                    loss = update_pass(tower, head, batch, micro, opt, sched, scaler, valid, candidates)
                else:
                    loss = update(tower, head, batch, micro, opt, sched, scaler, valid)
                count += len(batch[0])
                if i % 20 == 0:
                    print(out.name, 'epoch', epoch, 'step', i, loss, flush=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-t0
            sampled_seconds += seconds
            assert count == c['draws_per_epoch']
            records.append(dict(epoch=epoch, updates=sched.last_epoch, draws=count, seconds=seconds))
            C.write(out / 'STATUS.json', dict(status='RUNNING', history=records))
        assert sched.last_epoch == c['n_update']
        if arm == 'pass':
            source_logits = infer(tower, head, P.loader(table, adapter.spec, batch=micro, workers=workers))
            colour = list(range(8, 20)) + list(range(21, 33))
            prior[colour] = (1/(1+np.exp(-np.clip(source_logits.astype(np.float64), -60, 60)))).mean(0)[colour].clip(1e-3, 1-1e-3)
            C.write(out/'pass_priors.json', dict(label_prior=label_prior.tolist(), model_prior=prior.tolist(),
                    split='source TRAIN', augmentation='none', mode='eval', rows=len(table.labels), train_sha256=C.sha(C.TRAIN)))
        checkpoint = out / 'epoch6.pt'
        torch.save(dict(tower_state_dict={k:v.detach().cpu() for k,v in tower.state_dict().items()},
                        head_state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()},
                        prior=prior, label_prior=label_prior, provenance=provenance, n_update=sched.last_epoch, selected_epoch=6), checkpoint)
        torch.save(dict(directions=dirs, source=source), out / 'text_dirs.pt')
        target = P.heldout(protocol)
        z = infer(tower, head, P.loader(target, adapter.spec, batch=micro, workers=workers))
        mask = P.masks(target)
        conv = P.masks(target, convention=True)
        np.savez_compressed(out / 'heldout.npz', logits=z, labels=target.labels, valid_mask=mask,
                            valid_convention=conv, image_paths=np.asarray(target.image_paths),
                            attribute_names=np.asarray(target.attribute_names), source_train_prior=prior, source_label_prior=label_prior,
                            checkpoint_sha256=np.asarray(C.sha(checkpoint)))
        evaluators = dict(legacy=score(z, target.labels, mask, prior, target.attribute_names))
        if protocol == 'lodo_pa':
            evaluators['convention'] = score(z, target.labels, conv, prior, target.attribute_names)
        cross = None
        cross_reason = None
        base_heldout = C.ROOT / 'runs' / C.run_id(backbone, protocol, 'base') / 'heldout.npz'
        if arm == 'lc_m1' and not base_heldout.is_file():
            # No base run for this protocol: skip the 2x2 diagnostic.
            cross_reason = 'base heldout absent for this protocol; 2x2 skipped'
        elif arm == 'lc_m1':
            with np.load(base_heldout, allow_pickle=False) as b:
                assert np.array_equal(b['image_paths'], target.image_paths)
                assert np.array_equal(b['labels'], target.labels)
                cross = {}
                for logit_name, zz in [('base', b['logits']), ('lc_m1', z)]:
                    for prior_name, pp in [('base', b['source_train_prior']), ('lc_m1', prior)]:
                        cross[logit_name+'/'+prior_name] = score(zz, target.labels, mask, pp,
                                                               target.attribute_names, bootstrap=0)
        from .posthoc_pmcd import evaluate_cache
        posthoc = evaluate_cache(out/'heldout.npz', protocol=protocol, bootstrap=c['bootstrap_samples'], alpha_scores=evaluators)
        if arm == 'pass':
            cross = {}
            base_cache = C.ROOT/'runs'/C.run_id(backbone, protocol, 'base')/'heldout.npz'
            if not base_cache.is_file():
                raise FileNotFoundError('PASS 2x2 requires the same-fold base cache: '+str(base_cache))
            with np.load(base_cache, allow_pickle=False) as b:
                assert np.array_equal(b['image_paths'], target.image_paths) and np.array_equal(b['labels'], target.labels)
                for evaluator, vv in [('legacy', mask), ('convention', conv)]:
                    cross[evaluator] = {}
                    for zn, zz in [('base', b['logits']), ('pass', z)]:
                        for pn, pp in [('label', label_prior), ('model', prior)]:
                            cross[evaluator][zn+'/'+pn] = {decoder: score(zz, target.labels, vv, pp, target.attribute_names,
                                            bootstrap=0, decoder=decoder) for decoder in ('alpha', 'pmcd')}
        amodal = None
        if protocol == 'lodo_market':
            from .amodal_eval import from_cache
            amodal = from_cache(out/'heldout.npz')
        result = dict(run_id=out.name, backbone=backbone, protocol=protocol, arm=arm, aug=aug, seed=42,
                      n_update=sched.last_epoch, config_sha=ident['config_sha'], code_sha=ident['code_sha'],
                      evaluators=evaluators, pmcd=posthoc['pmcd'], lc_2x2=cross if arm == 'lc_m1' else None,
                      lc_2x2_reason=cross_reason,
                      pass_2x2=cross if arm == 'pass' else None, amodal=amodal,
                      timing=dict(s_per_sample=sampled_seconds/216000, total_h=(time.perf_counter()-begin)/3600),
                      vram_peak_gb=torch.cuda.max_memory_allocated()/2**30,
                      checkpoint_sha256=C.sha(checkpoint), heldout_sha256=C.sha(out/'heldout.npz'))
        C.write(out / 'result.json', result)
        lines = [f'# {out.name}', '', 'Final epoch 6; TRAIN-only fitting; alpha .75 + force-one.', '',
                 '| Evaluator | CA | mA | F1 | AUC | BCE |', '|---|---:|---:|---:|---:|---:|']
        lines += [f'| {k} | '+ ' | '.join(f'{v[x]:.6f}' for x in ('CA','mA','F1','macro_auc','bce'))+' |'
                  for k,v in evaluators.items()]
        lines += ['', 'PMCD (same evaluator and mask):']
        lines += [f'| {k} | '+ ' | '.join(f'{v[x]:.6f}' for x in ('CA','mA','F1','macro_auc','bce'))+' |'
                  for k,v in posthoc['pmcd'].items()]
        if amodal is not None:
            C.write(out/'amodal.json',amodal)
            lines += ['', 'Amodal frame/identity-OR diagnostic: amodal.json (also embedded in result.json).']
        (out / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        C.write(out / 'STATUS.json', dict(status='DONE', history=records))
    except BaseException as exc:
        C.write(out / 'STATUS.json', dict(status='FAIL:'+str(exc), error=repr(exc)))
        raise
