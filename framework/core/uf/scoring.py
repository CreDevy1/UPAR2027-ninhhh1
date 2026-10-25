"""One scorer for runs, posthoc consumers, and paper tables."""
import numpy as np
from . import common as C
from lc import score as old
from lc.convention import M1_TARGET

DETAIL = ['Accessory-Glasses', 'Accessory-Sunglasses', 'Accessory-Bag',
          'Accessory-Hat', 'Accessory-Backpack', 'Hair-Bald']


def scalar(x):
    return float(x) if np.isfinite(x) else None


def score(z, y, valid, prior, names, bootstrap=None, decoder='alpha'):
    z, y, valid = np.asarray(z, np.float64), np.asarray(y), np.asarray(valid, bool)
    if z.shape != y.shape or valid.shape != y.shape or z.shape[1] != 40 or not np.isfinite(z).all():
        raise ValueError('Invalid score arrays')
    if decoder == 'alpha':
        pred = old.decide(z, prior)
    elif decoder == 'pmcd':
        from .pmcd import pmcd_decisions
        pred = pmcd_decisions(z, prior, valid)
    else:
        raise ValueError('Unknown decoder: ' + decoder)
    m, per = old.metrics(pred, y, valid)
    auc = old.auc(z, y, valid)
    out = dict(m, macro_auc=scalar(np.nanmean(auc)), bce=old.bce(z, y, valid),
               ci95={}, per_attr={}, group_auc={})
    for j, name in enumerate(names):
        mask = valid[:, j]
        out['per_attr'][name] = dict(mA=scalar(per[j]), auc=scalar(auc[j]),
                                   support=dict(positive=int((y[:, j]*mask).sum()), negative=int(((1-y[:, j])*mask).sum())))
    groups = dict(detail=[i for i, n in enumerate(names) if n in DETAIL or
                          any(s in n.lower() for s in ('glasses', 'sunglasses', 'backpack', 'bald')) or n.endswith(('-Bag', '-Hat'))],
                  ub_color=list(range(8, 20)), lb_color=list(range(21, 33)), global_=list(range(40)))
    for key, ix in groups.items():
        out['group_auc']['global' if key == 'global_' else key] = scalar(np.nanmean(auc[ix])) if ix else None
    target = list(names).index(M1_TARGET)
    non = [j for j in range(40) if j != target]
    out['nontarget_mA'] = scalar(np.nanmean(per[non]))
    skirt = list(names).index('LowerBody-Type-Skirt&Dress')
    short = valid[:, target] & (y[:, skirt] == 1) & (y[:, target] == 1)
    long = valid[:, target] & (y[:, skirt] == 1) & (y[:, target] == 0)
    out['skirt'] = dict(short_support=int(short.sum()), long_support=int(long.sum()),
                        short_recall=float(pred[short, target].mean()) if short.any() else None,
                        long_specificity=float((~pred[long, target]).mean()) if long.any() else None)
    count = C.config()['bootstrap_samples'] if bootstrap is None else bootstrap
    draws = {k: [] for k in ('CA', 'mA', 'F1', 'macro_auc', 'bce')}
    rng = np.random.default_rng(42)
    for _ in range(count):
        ix = rng.integers(0, len(y), len(y))
        bm, _ = old.metrics(pred[ix], y[ix], valid[ix])
        ba = old.auc(z[ix], y[ix], valid[ix])
        bm.update(macro_auc=float(np.nanmean(ba)), bce=old.bce(z[ix], y[ix], valid[ix]))
        for k in draws:
            draws[k].append(bm[k])
    out['ci95'] = {k: [float(x) for x in np.quantile(v, [.025, .975])] for k, v in draws.items() if v}
    out['bootstrap'] = dict(unit='image', samples=count, seed=42)
    return out
