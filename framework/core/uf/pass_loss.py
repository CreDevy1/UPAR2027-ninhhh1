"""PASS-v1: only multi-positive PETA colours gain Other as a candidate."""
import numpy as np
import torch
from torch.nn import functional as F

COLOURS = (tuple(range(8, 20)), tuple(range(21, 33)))


def candidate_sets(labels, domains):
    y = np.asarray(labels)
    if y.ndim != 2 or y.shape[1] != 40 or not np.isin(y, (0, 1)).all() or len(domains) != len(y):
        raise ValueError('Binary [N,40] labels and aligned source domains required')
    c = np.zeros(y.shape, bool)
    peta = np.asarray(domains) == 'PETA'
    for group in COLOURS:
        ix = list(group)
        g = y[:, ix] > 0
        count = g.sum(1)
        g[peta & (count >= 2), -1] = True
        g[count == 0] = True  # full candidate set carries no information
        c[:, ix] = g
    return c


def effective_valid(valid, candidates):
    """16 non-colour cells plus 12 cells for each informative colour group."""
    if torch.is_tensor(valid):
        v = valid.clone().bool()
        for group in COLOURS:
            ix = list(group)
            v[:, ix] = (candidates[:, ix].sum(1) < 12)[:, None]
    else:
        v = np.asarray(valid, bool).copy()
        for group in COLOURS:
            ix = list(group)
            v[:, ix] = (candidates[:, ix].sum(1) < 12)[:, None]
    return v


def log1mexp(s):
    # Evaluate only the appropriate branch: torch.where evaluates both and can
    # backpropagate NaN from an unused log1p(-1) at large negative logits.
    result = torch.empty_like(s)
    near = s > -0.6931471805599453
    result[near] = torch.log(-torch.expm1(s[near]))
    result[~near] = torch.log1p(-torch.exp(s[~near]))
    return result


def pass_loss(logits, labels, valid, candidates, denominator=None):
    """Global cell normalization, also used for accumulation and FP64 calibration.

    Singleton groups retain the exact BCE cells and reduction order, including
    at extreme logits. Only genuinely ambiguous groups use the stable OR term.
    """
    v = effective_valid(valid, candidates)
    cells = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype), reduction='none') * v
    for group in COLOURS:
        ix = list(group)
        cg = candidates[:, ix].bool()
        multi = (cg.sum(1) > 1) & (cg.sum(1) < 12)
        rows = torch.where(multi)[0]
        if len(rows):
            zg = logits[rows][:, ix]
            cc = cg[rows]
            logneg = F.logsigmoid(-zg)
            s = (logneg * cc).sum(1).clamp(max=-1e-6)
            group_loss = -log1mexp(s) - (logneg * ~cc).sum(1)
            cells = cells.clone()
            cells[rows[:, None], torch.tensor(ix, device=logits.device)[None, :]] = 0
            cells[rows, ix[0]] = group_loss
    denom = v.sum() if denominator is None else denominator
    if int(denom) <= 0:
        raise ValueError('PASS window has no valid supervision')
    return cells.sum() / denom


def fit_calibration(scores, labels, valid, candidates):
    """Full TRAIN-only FP64 LBFGS with the same PASS objective as fine-tuning."""
    scores, labels = scores.double().cpu(), labels.double().cpu()
    valid, candidates = valid.bool().cpu(), candidates.bool().cpu()
    v = effective_valid(valid, candidates)
    prior = ((labels*v).sum(0) + .5) / (v.sum(0) + 1)
    a = torch.nn.Parameter(torch.zeros(40, dtype=torch.float64))
    b = torch.nn.Parameter(torch.logit(prior))
    opt = torch.optim.LBFGS([a, b], lr=1, max_iter=1000, max_eval=1250,
                           tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(set_to_none=True)
        loss = pass_loss(scores*a+b, labels, valid, candidates)
        loss.backward()
        return loss
    opt.step(closure)
    loss = closure()
    grad = max(float(a.grad.abs().max()), float(b.grad.abs().max()))
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or grad > 1e-5:
        raise RuntimeError(f'PASS calibration did not converge: {grad}')
    return a.detach().float(), b.detach().float(), dict(max_gradient=grad,
                iterations=int(opt.state[a]['n_iter']), loss=float(loss.detach()), dtype='float64', objective='PASS-v1')
