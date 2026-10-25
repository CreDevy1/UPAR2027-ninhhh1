"""Track 1 prior-based logit correction, shared by validation and packaging."""
from __future__ import annotations

import numpy as np


DOMAINS = ("Market1501", "PA100k", "PETA")


def domain_from_path(path: str) -> str:
    domain = path.replace("\\", "/").split("/", 1)[0]
    if domain not in DOMAINS:
        raise ValueError(f"Unknown dataset domain in image path: {path}")
    return domain


def smoothed_prior(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 2 or len(labels) == 0 or not np.isin(labels, (0, 1)).all():
        raise ValueError("Expected a nonempty binary label matrix")
    return (labels.sum(axis=0, dtype=np.float64) + 0.5) / (len(labels) + 1.0)


def prior_logit(prior: np.ndarray) -> np.ndarray:
    prior = np.asarray(prior, dtype=np.float64)
    if prior.ndim != 1 or not np.isfinite(prior).all() or np.any((prior <= 0) | (prior >= 1)):
        raise ValueError("Prior must be a finite vector strictly between 0 and 1")
    return np.log(prior) - np.log1p(-prior)


def corrected_probabilities(logits: np.ndarray, prior: np.ndarray,
                            alpha: float | np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    alpha_array = np.asarray(alpha, dtype=np.float64)
    if alpha_array.ndim == 0:
        alpha_array = np.full(len(prior), float(alpha_array))
    if alpha_array.shape != (len(prior),) or not np.isfinite(alpha_array).all():
        raise ValueError("alpha must be a finite scalar or a vector matching the prior")
    shift = alpha_array * prior_logit(prior)
    if logits.ndim != 2 or logits.shape[1] != len(shift) or not np.isfinite(logits).all():
        raise ValueError("Logits must be a finite matrix matching the prior")
    z = np.clip(logits - shift[None, :], -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-z))
