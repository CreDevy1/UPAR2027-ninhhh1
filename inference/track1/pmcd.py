"""Fixed protocol/metric decoder. NumPy only; no cross-image state or fitting."""
import numpy as np

GROUPS = ((0, 1, 2), (4, 5, 6), tuple(range(8, 20)), tuple(range(21, 33)), (33, 34))


def plogit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def _inputs(z, prior, valid):
    z, prior = np.asarray(z, np.float64), np.asarray(prior, np.float64)
    if z.ndim != 2 or z.shape[1] != 40 or prior.shape != (40,):
        raise ValueError('Expected logits [N,40] and prior [40]')
    if not np.isfinite(z).all() or not np.isfinite(prior).all() or ((prior <= 0) | (prior >= 1)).any():
        raise ValueError('Finite logits and TRAIN priors strictly in (0,1) required')
    v = np.ones(z.shape, bool) if valid is None else np.asarray(valid, bool)
    if v.shape != z.shape:
        raise ValueError('valid must match logits')
    return z, prior, v


def _decode(z, prior, valid=None, projection=True, force_one=True, wf=.5):
    """Ablation switches are private and used only by the degeneration selftest."""
    z, prior, v = _inputs(z, prior, valid)
    if len(z) == 0:
        return np.zeros(z.shape, bool)
    if wf == 0 and not projection and not force_one:
        return (z - plogit(prior) > 0) & v
    q = 1 / (1 + np.exp(-np.clip(z, -60, 60)))
    if projection:
        for ix in GROUPS:
            ix = list(ix)
            q[:, ix] /= q[:, ix].sum(1, keepdims=True)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    pi = np.clip(prior, 1e-4, 1 - 1e-4)[None, :]
    gain_ma = (.5 / 80) * (q / pi - (1 - q) / (1 - pi))
    expected = np.maximum((q * v).sum(1), 1.)
    n = len(z)
    on = np.zeros(z.shape, bool)
    size, hits = np.zeros(n), np.zeros(n)
    active = np.ones(n, bool)
    arange = np.arange(n)
    for _ in range(40):
        precision = np.where(size > 0, hits / np.maximum(size, 1), 0.)
        gain = gain_ma + wf * .5 * (q - precision[:, None]) / (size[:, None] + 1) + wf * .5 * q / expected[:, None]
        gain = np.where(v & ~on, gain, -np.inf)
        best = gain.argmax(1)
        add = active & (gain[arange, best] > 0)
        if not add.any():
            break
        rows = np.flatnonzero(add)
        on[rows, best[rows]] = True
        size[rows] += 1
        hits[rows] += q[rows, best[rows]]
        active = add
    if force_one:
        s = plogit(q) - plogit(pi)
        for ix in GROUPS:
            ix = np.asarray(ix)
            rows = np.flatnonzero(~on[:, ix].any(1))
            on[rows, ix[s[rows][:, ix].argmax(1)]] = True
    return on


def pmcd_decisions(z, prior, valid=None):
    return _decode(z, prior, valid)


def pmcd_probabilities(z, prior, valid=None):
    on = pmcd_decisions(z, prior, valid)
    q = (1 / (1 + np.exp(-np.clip(np.asarray(z, np.float64), -60, 60)))).astype(np.float32)
    out = np.minimum(q, np.nextafter(np.float32(.5), np.float32(0)))
    out[on] = np.float32(1)
    return out
