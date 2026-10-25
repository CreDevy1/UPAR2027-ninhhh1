"""Market frame versus identity-OR labels; diagnostic only, never a gate."""
import argparse
from pathlib import Path
import numpy as np
from . import common as C
from lc.score import auc, decide

ATTRIBUTES = ('Accessory-Glasses-Normal', 'Accessory-Glasses-Sun', 'Hair-Length-Long',
              'Hair-Length-Short', 'Accessory-Hat', 'Accessory-Backpack', 'Accessory-Bag')


def evaluate(z, y, paths, names, prior, valid=None):
    z, y = np.asarray(z, np.float64), np.asarray(y)
    paths = [str(p).replace('\\', '/') for p in paths]
    if not all(p.startswith('Market1501/') for p in paths):
        raise ValueError('Amodal evaluation is restricted to Market VAL')
    if z.shape != y.shape or z.shape != (len(paths), 40) or len(set(paths)) != len(paths):
        raise ValueError('Unaligned or duplicate Market rows')
    valid = np.ones(y.shape, bool) if valid is None else np.asarray(valid, bool)
    if valid.shape != y.shape:
        raise ValueError('Mask shape mismatch')
    ids = np.asarray([p.rsplit('/', 1)[-1][:4] for p in paths])
    if not all(len(i) == 4 and i.isdigit() for i in ids):
        raise ValueError('Expected four-digit Market identity prefixes')
    _, group = np.unique(ids, return_inverse=True)
    pred = decide(z, prior)
    out = dict(unit='identity OR (first four filename characters)', rows=len(y), identities=int(group.max()+1),
               diagnostic_only=True, attributes={})
    for name in ATTRIBUTES:
        k = list(names).index(name)
        v = valid[:, k]
        yy = y[:, k].astype(bool)
        identity = np.zeros(int(group.max()+1), bool)
        np.logical_or.at(identity, group[v], yy[v])
        yo = identity[group]
        res = {}
        for tag, labels in [('image', yy), ('identityOR', yo)]:
            p, n = v & labels, v & ~labels
            tpr = float(pred[p, k].mean()) if p.any() else None
            tnr = float((~pred[n, k]).mean()) if n.any() else None
            a = auc(z[:, k:k+1], labels[:, None], v[:, None])[0]
            res[tag] = dict(pos_rate=float(labels[v].mean()) if v.any() else None,
                           TPR=tpr, TNR=tnr, bal_acc=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,
                           AUC=float(a) if np.isfinite(a) else None, positive=int(p.sum()), negative=int(n.sum()))
        hidden, negative = v & ~yy & yo, v & ~yo
        q = 1/(1+np.exp(-np.clip(z[:, k], -60, 60)))
        res['hidden_pos'] = dict(n=int(hidden.sum()), pred_pos_rate=float(pred[hidden, k].mean()) if hidden.any() else None,
                                mean_sigmoid=float(q[hidden].mean()) if hidden.any() else None,
                                mean_sigmoid_true_neg=float(q[negative].mean()) if negative.any() else None)
        out['attributes'][name] = res
    return out


def from_cache(path):
    with np.load(path, allow_pickle=False) as d:
        out = evaluate(d['logits'], d['labels'], d['image_paths'], d['attribute_names'],
                       d['source_train_prior'], d['valid_mask'] if 'valid_mask' in d else None)
    out['source'] = dict(file=str(Path(path).resolve()), sha256=C.sha(path))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cache', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    C.write(a.out, from_cache(a.cache))
    print('PASS', a.out)


if __name__ == '__main__':
    main()
