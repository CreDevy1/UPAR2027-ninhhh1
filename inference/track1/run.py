"""UPAR Track 1: pel_hr, fixed PMCD decoder, no time fallback."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import hr_infer
from pmcd import pmcd_probabilities, plogit

BATCH_SIZE = 2147483647  # Ingestion passes paths; internal GPU batch is bounded.


def load_model():
    hr_infer.load()


def predict_batch(samples):
    if not samples:
        return []
    names = samples[0]['attribute_names']
    if any(s['attribute_names'] != names for s in samples):
        raise ValueError('Attribute order changed within a batch')
    cols = hr_infer.column_index(names)
    logits, _ = hr_infer.infer([s['image_path'] for s in samples])
    cfg = hr_infer.CONFIG
    # Same frozen PMCD transformation as the existing UF package, with the
    # member's actual TRAIN+VAL prior and the fixed TRAIN reference prior.
    z = logits.numpy().astype(np.float64) - plogit(cfg['members'][0]['train_prior'])
    p = np.asarray(cfg['track1']['pmcd_reference_prior'], dtype=np.float64)
    return pmcd_probabilities(z + plogit(p), p)[:, cols].tolist()


def predict_image(sample):
    return predict_batch([sample])[0]
