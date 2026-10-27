"""Codabench entry points for the Track 1 package (load_model, predict_batch)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import upar_infer
BATCH_SIZE=2147483647

def load_model(flip_tta=None,rate_preserving=None):
    upar_infer.load(flip_tta=flip_tta,rate_preserving=rate_preserving)

def predict_batch(samples):
    if not samples:
        return []
    names=samples[0]['attribute_names']
    if any(s['attribute_names']!=names for s in samples):
        raise ValueError('Attribute order changed within a batch')
    columns=upar_infer.column_index(names)
    return upar_infer.probabilities([s['image_path'] for s in samples])[:,columns].tolist()
