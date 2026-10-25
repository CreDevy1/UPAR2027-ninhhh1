"""Explicit domain/attribute exclusions for hard and soft training targets."""
from __future__ import annotations

import numpy as np
import torch

from .calibration import DOMAINS, domain_from_path


def parse_mask(spec: str | dict | None, names: list | tuple) -> dict[str, list[str]]:
    if not spec:
        return {}
    if isinstance(spec, str):
        entries = {}
        for part in spec.split(";"):
            domain, sep, attributes = part.strip().partition(":")
            if not sep or domain in entries:
                raise ValueError("mask-domain-attrs must be DOMAIN:attr,attr;DOMAIN:attr")
            entries[domain] = attributes.split(",")
    else:
        entries = spec
    result = {}
    for domain, attributes in sorted(entries.items()):
        attributes = sorted(set(a.strip() for a in attributes))
        if domain not in DOMAINS or not attributes or any(a not in names for a in attributes):
            raise ValueError("Unknown domain or attribute in mask-domain-attrs")
        result[domain] = attributes
    return result


def valid_cells(domains: list | np.ndarray, names: list | tuple, spec: str | dict | None) -> np.ndarray:
    mask = np.ones((len(domains), len(names)), dtype=bool)
    domains = np.asarray(domains)
    for domain, attributes in parse_mask(spec, names).items():
        mask[np.ix_(np.flatnonzero(domains == domain), [names.index(a) for a in attributes])] = False
    return mask


def path_mask(paths, names, spec, device=None):
    return torch.as_tensor(valid_cells([domain_from_path(p) for p in paths], names, spec), device=device)


def masked_prior(labels: np.ndarray, domains, names, spec) -> np.ndarray:
    valid = valid_cells(domains, names, spec)
    counts = valid.sum(0)
    if np.any(counts == 0):
        raise ValueError("No unmasked source rows for an attribute")
    # Match calibration.smoothed_prior's Laplace smoothing.
    return ((np.asarray(labels) * valid).sum(0) + 0.5) / (counts + 1.0)


def masked_bce(logits, targets, valid=None):
    if valid is None:
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    if valid.shape != logits.shape:
        raise ValueError("Loss mask shape differs from logits")
    cells = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (cells * valid).sum() / valid.sum().clamp_min(1)
