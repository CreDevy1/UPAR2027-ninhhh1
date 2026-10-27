"""T2 post-processing core (torch, works on CPU or CUDA; torch>=2.4 API only).

All steps are automatic and identical for every gallery image / query:
  1. ensemble raw logits                      z = sum_m w_m z_m / sum_m w_m
  2. gallery neighbour aggregation (optional) P~_g = (s0 P_g + sum_n a_gn P_n) / (s0 + sum_n a_gn)
     neighbours = cosine kNN on visual features (self excluded), a_gn = exp((sim-1)/tau_k) * [sim >= thr]
  3. Bernoulli log-likelihood at temperature T  L[q,g] = sum_k q_k logsig(z~_gk/T) + (1-q_k) logsig(-z~_gk/T)
  4. query-axis normalisation (optional)      S = L - beta * tau_q * logsumexp_q(L / tau_q)
Returns similarities (bigger = better), Q x G.
v4, both off by default so v3 configs behave exactly as before:
  qprior_gamma > 0 : query-set prior shift  z += gamma * (logit mean_q(Q) - logit pi_train)   (before step 3)
  qd_mode='gated'  : similarity-gated query diffusion (weights NOT row-normalised, self weight qd_s0):
                     X_g <- (s0*X0_g + sum_n w_gn X_n) / (s0 + sum_n w_gn),  w = exp((sim-1)/qd_tau_k)
                     -> an image without close neighbours keeps its own score (sparse galleries are not smeared).
v4b, off by default:
  sk_gamma > 0     : Sinkhorn-balanced per-image normaliser. Query weights r_q (sk_iters steps) give every query the same
                     total mass G/Q, so queries that already claim many images are discounted in each image's normaliser:
                     c_g = LSE_q(A_qg - gamma r_q),  A = (L - qprior)/tau_q,  S = L - beta*tau_q*c_g.  gamma=0 == v3/v4a.
"""
import math

import torch
import torch.nn.functional as F


def ensemble(logits_list, weights=None):
    w = weights or [1.0] * len(logits_list)
    z = sum(wi * li.double() for wi, li in zip(w, logits_list)) / float(sum(w))
    return z


@torch.no_grad()
def knn_graph(feats, k, chunk=4096):
    """feats: (G,D) any float; cosine kNN excluding self -> idx (G,k) long, sim (G,k) float32."""
    f = F.normalize(feats.float(), dim=1)
    G = f.shape[0]
    k = min(k, G - 1)
    idx = torch.empty(G, k, dtype=torch.long, device=f.device)
    sim = torch.empty(G, k, dtype=torch.float32, device=f.device)
    for s in range(0, G, chunk):
        e = min(G, s + chunk)
        S = f[s:e] @ f.T
        r = torch.arange(e - s, device=f.device)
        S[r, s + r] = -2.0
        v, i = S.topk(k, dim=1)
        idx[s:e] = i
        sim[s:e] = v
    return idx, sim


def neighbour_weights(sim, tau_k=0.05, thr=-1.0, mutual_idx=None):
    a = torch.exp((sim - 1.0) / tau_k) * (sim >= thr)
    return a


@torch.no_grad()
def aggregate_prob(P, idx, a, self_w=1.0, iters=1, alpha=None):
    """P: (G,40) probabilities. one step = weighted mean of self and neighbours.
    iters>1 -> diffusion  P_{t+1} = (s0 P_0 + sum a P_t[nbr]) / (s0 + sum a)."""
    den = self_w + a.sum(1, keepdim=True)
    P0 = P
    Pt = P
    for _ in range(iters):
        nb = (a.unsqueeze(-1) * Pt[idx]).sum(1)
        Pt = (self_w * P0 + nb) / den
    return Pt


def prob_to_logit(P, eps=1e-6):
    P = P.clamp(eps, 1 - eps)
    return torch.log(P) - torch.log1p(-P)


@torch.no_grad()
def loglik(Q, z, T=1.0, block=8192):
    """Q (nq,40) {0,1}, z (G,40) logits -> (nq,G) float64 log-likelihood."""
    Q = Q.double()
    out = torch.empty(Q.shape[0], z.shape[0], dtype=torch.float64, device=z.device)
    for s in range(0, z.shape[0], block):
        zs = z[s:s + block].double() / T
        out[:, s:s + block] = Q @ F.logsigmoid(zs).T + (1 - Q) @ F.logsigmoid(-zs).T
    return out


@torch.no_grad()
def query_norm(L, tau_q=1.0, beta=0.0, qprior=None):
    """S = L - beta*tau_q*logsumexp_q((L - qprior)/tau_q). qprior (nq,1) = c * q.logit(pi_train) removes the
    TRAIN prior of each query vector inside the per-image normaliser (uniform prior over the given queries)."""
    if beta == 0:
        return L
    Lc = L if qprior is None else L - qprior
    return L - beta * tau_q * torch.logsumexp(Lc / tau_q, dim=0, keepdim=True)


@torch.no_grad()
def query_norm_sk(L, tau_q=1.0, beta=0.0, qprior=None, gamma=1.0, iters=3):
    """v4b: per-image normaliser with Sinkhorn query weights (uniform target mass G/Q per query)."""
    if beta == 0:
        return L
    A = (L if qprior is None else L - qprior) / tau_q
    nq, G = A.shape
    r = torch.zeros(nq, 1, dtype=A.dtype, device=A.device)
    for _ in range(iters):
        c = torch.logsumexp(A - gamma * r, dim=0, keepdim=True)
        r = torch.logsumexp(A - c, dim=1, keepdim=True) - math.log(G / nq)
    c = torch.logsumexp(A - gamma * r, dim=0, keepdim=True)
    return L - beta * tau_q * c


@torch.no_grad()
def score_z(queries, z, feats=None, cfg=None):
    """queries (nq,40) {0,1}; z (G,40) ensembled raw logits; feats (G,D) or None -> similarities (nq,G)."""
    cfg = dict(cfg or {})
    z = z.double()
    if feats is not None and cfg.get('knn_k', 0) > 0:
        idx, sim = knn_graph(feats, cfg['knn_k'])
        a = neighbour_weights(sim, cfg.get('tau_k', 0.05), cfg.get('thr', -1.0)).double()
        P = torch.sigmoid(z)
        P = aggregate_prob(P, idx, a, cfg.get('self_w', 1.0), cfg.get('iters', 1))
        z = prob_to_logit(P, cfg.get('eps', 1e-6))
    L = loglik(torch.as_tensor(queries, device=z.device), z, cfg.get('T', 1.0))
    return query_norm(L, cfg.get('tau_q', 1.0), cfg.get('beta', 0.0))


@torch.no_grad()
def score(queries, logits_list, feats=None, cfg=None):
    cfg = dict(cfg or {})
    return score_z(queries, ensemble(logits_list, cfg.get('weights')), feats, cfg)

@torch.no_grad()
def diffuse_query_log(S, idx, w, alpha=0.8, iters=3, tau_s=1.0, chunk=32):
    """Query-specific manifold diffusion over the gallery kNN graph, fully in log space.
    S (nq,G) scores; idx (G,k) neighbours; w (G,k) row-normalised non-negative weights.
    X0 = softmax_g(S/tau_s);  X <- (1-alpha) X0 + alpha * W X ;  returns log X (nq,G)."""
    if alpha <= 0 or iters <= 0:
        return S
    logw = torch.log(w.clamp_min(1e-300)).to(S.dtype)
    la, lb = torch.log(torch.tensor(1.0 - alpha, dtype=S.dtype)), torch.log(torch.tensor(alpha, dtype=S.dtype))
    out = torch.empty_like(S)
    for s in range(0, S.shape[0], chunk):
        L0 = torch.log_softmax(S[s:s + chunk] / tau_s, dim=1)
        L = L0
        for _ in range(iters):
            nb = torch.logsumexp(L[:, idx] + logw.unsqueeze(0), dim=-1)
            L = torch.logaddexp(la + L0, lb + nb)
        out[s:s + chunk] = L
    return out


@torch.no_grad()
def gated_query_log(S, idx, sim, tau_k=0.05, s0=0.1, iters=3, tau_s=2.0, chunk=64):
    """Similarity-gated query diffusion in log space (v4). S (nq,G); idx/sim (G,k) from knn_graph.
    X0 = softmax_g(S/tau_s);  X_g <- (s0 X0_g + sum_n w_gn X_n) / (s0 + sum_n w_gn),  w = exp((sim-1)/tau_k)."""
    w = torch.exp((sim.double() - 1.0) / tau_k)
    logw = torch.log(w.clamp_min(1e-300)).to(S.dtype)
    lden = torch.log(s0 + w.sum(1)).to(S.dtype)
    ls0 = torch.log(torch.tensor(s0, dtype=S.dtype, device=S.device))
    out = torch.empty_like(S)
    for s in range(0, S.shape[0], chunk):
        L0 = torch.log_softmax(S[s:s + chunk] / tau_s, dim=1)
        L = L0
        for _ in range(iters):
            nb = torch.logsumexp(L[:, idx] + logw.unsqueeze(0), dim=-1)
            L = torch.logaddexp(ls0 + L0, nb) - lden.unsqueeze(0)
        out[s:s + chunk] = L
    return out


@torch.no_grad()
def score_full(queries, z, feats=None, cfg=None):
    """z-aggregation (knn_k>0) -> loglik(T) -> query-axis norm (beta) -> query diffusion (qd_alpha>0)."""
    cfg = dict(cfg or {})
    z = z.double()
    if cfg.get('qprior_gamma', 0) and cfg.get('prior_logit') is not None:   # v4 query-set prior shift
        Qs = torch.as_tensor(queries, device=z.device).double()
        pq = Qs.mean(0).clamp(1e-3, 1 - 1e-3)
        ps = torch.sigmoid(torch.as_tensor(cfg['prior_logit'], device=z.device).double()).clamp(1e-3, 1 - 1e-3)
        z = z + cfg['qprior_gamma'] * (torch.logit(pq) - torch.logit(ps))
    idx = sim = None
    if feats is not None and (cfg.get('knn_k', 0) > 0 or cfg.get('qd_alpha', 0) > 0):
        k = max(cfg.get('knn_k', 0), cfg.get('qd_k', 0))
        idx, sim = knn_graph(feats, k)
    if cfg.get('knn_k', 0) > 0:
        kk = cfg['knn_k']
        a = neighbour_weights(sim[:, :kk], cfg.get('tau_k', 0.05), cfg.get('thr', -1.0)).double()
        P = aggregate_prob(torch.sigmoid(z), idx[:, :kk], a, cfg.get('self_w', 1.0), cfg.get('iters', 1))
        z = prob_to_logit(P, cfg.get('eps', 1e-6))
    Qt = torch.as_tensor(queries, device=z.device).double()
    L = loglik(Qt, z, cfg.get('T', 1.0))
    qprior = None
    if cfg.get('prior_c', 0) and cfg.get('prior_logit') is not None:
        qprior = cfg['prior_c'] * (Qt @ torch.as_tensor(cfg['prior_logit'], device=z.device).double())[:, None]
    if cfg.get('sk_gamma', 0) > 0:   # v4b
        S = query_norm_sk(L, cfg.get('tau_q', 1.0), cfg.get('beta', 0.0), qprior, cfg['sk_gamma'], cfg.get('sk_iters', 3))
    else:
        S = query_norm(L, cfg.get('tau_q', 1.0), cfg.get('beta', 0.0), qprior)
    if cfg.get('qd_alpha', 0) > 0 and cfg.get('qd_mode', 'row') == 'gated':   # v4 (qd_alpha only enables)
        kq = cfg.get('qd_k', cfg.get('knn_k', 10))
        return gated_query_log(S, idx[:, :kq], sim[:, :kq], cfg.get('qd_tau_k', 0.05), cfg.get('qd_s0', 0.1),
                               cfg.get('qd_iters', 3), cfg.get('qd_tau_s', 1.0))
    if cfg.get('qd_alpha', 0) > 0:
        kq = cfg.get('qd_k', cfg.get('knn_k', 10))
        w = neighbour_weights(sim[:, :kq], cfg.get('qd_tau_k', 0.05), cfg.get('qd_thr', -1.0)).double()
        w = w / w.sum(1, keepdim=True).clamp_min(1e-300)
        S = diffuse_query_log(S, idx[:, :kq], w, cfg['qd_alpha'], cfg.get('qd_iters', 3), cfg.get('qd_tau_s', 1.0))
    return S
