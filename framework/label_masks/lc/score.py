"""Masked numpy scoring (alpha/force-one decoder of upar_system.score_fold, masked mA/F1/CA, AUC, BCE)."""
import numpy as np

GROUPS = {'age': [0, 1, 2], 'hair': [4, 5, 6], 'upper_color': list(range(8, 20)),
          'lower_color': list(range(21, 33)), 'lower_type': [33, 34]}


def plogit(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log1p(-p)


def decide(z, prior, alpha=.75, force_one=True):
    s = np.asarray(z, dtype=np.float64) - alpha * plogit(prior)[None, :]
    dec = s >= 0
    if force_one:
        for ix in GROUPS.values():
            rows = np.where(dec[:, ix].sum(1) == 0)[0]
            if len(rows):
                dec[np.ix_(rows, ix)] = False
                dec[rows, np.asarray(ix)[s[rows][:, ix].argmax(1)]] = True
    return dec


def metrics(dec, Y, V):
    y = np.asarray(Y).astype(bool); V = np.asarray(V, dtype=bool)
    p = (y & V).sum(0); q = (~y & V).sum(0); ok = (p > 0) & (q > 0)
    per = np.full(y.shape[1], np.nan)
    per[ok] = ((dec & y & V).sum(0)[ok] / p[ok] + (~dec & ~y & V).sum(0)[ok] / q[ok]) / 2
    ma = float(np.nanmean(per))
    tp = (dec & y & V).sum(1); pr = (dec & V).sum(1); po = (y & V).sum(1)
    prec = float(np.divide(tp, pr, out=np.zeros(len(y)), where=pr > 0).mean())
    rec = float(np.divide(tp, po, out=np.zeros(len(y)), where=po > 0).mean())
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.
    ca = 2 * ma * f1 / (ma + f1) if ma + f1 else 0.
    return dict(mA=ma, F1=f1, CA=ca), per


def auc(z, Y, V):
    out = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        m = np.asarray(V[:, j], dtype=bool); y = Y[m, j] == 1; s = z[m, j]
        n1 = int(y.sum()); n0 = len(y) - n1
        if n1 == 0 or n0 == 0:
            continue
        order = np.argsort(s, kind='mergesort'); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
        # average ranks for ties
        ss = s[order]; i = 0
        while i < len(ss):
            k = i
            while k + 1 < len(ss) and ss[k + 1] == ss[i]:
                k += 1
            if k > i:
                r[order[i:k + 1]] = (i + k + 2) / 2.
            i = k + 1
        out[j] = (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def bce(z, Y, V):
    z = np.asarray(z, dtype=np.float64)
    cell = np.logaddexp(0, z) - Y * z
    return float(cell[np.asarray(V, dtype=bool)].mean())


def full(z, Y, V, prior, names, targets):
    names = list(names)
    dec = decide(z, prior, .75, True)
    m, per = metrics(dec, Y, V)
    raw, _ = metrics(decide(z, prior, 0., False), Y, V)
    a = auc(z, Y, V)
    tix = [names.index(t) for t in targets]
    non = [j for j in range(len(names)) if j not in tix]
    return dict(primary=m, raw05=raw, macro_AUC=float(np.nanmean(a)), raw_BCE=bce(z, Y, V),
                target_mA=float(np.nanmean(per[tix])) if not np.all(np.isnan(per[tix])) else None,
                nontarget_mA=float(np.nanmean(per[non])),
                per_attribute_mA={names[j]: (None if np.isnan(per[j]) else float(per[j])) for j in range(len(names))},
                per_attribute_AUC={names[j]: (None if np.isnan(a[j]) else float(a[j])) for j in range(len(names))})
