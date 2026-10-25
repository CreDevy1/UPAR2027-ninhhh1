"""Score a saved held-out cache with both decoders."""
from pathlib import Path
import numpy as np
from . import common as C
from .scoring import score


def evaluate_cache(path, protocol=None, bootstrap=0, alpha_scores=None):
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        names = list(d['attribute_names'])
        z, y, prior = d['logits'], d['labels'], d['source_train_prior']
        masks = dict(legacy=d['valid_mask'])
        if protocol == 'lodo_pa' or (protocol is None and all(str(p).replace('\\', '/').startswith('PA100k/') for p in d['image_paths'])):
            masks['convention'] = d['valid_convention']
        out = {decoder: {ev: (alpha_scores[ev] if decoder=='alpha' and alpha_scores is not None else
                             score(z, y, v, prior, names, bootstrap=bootstrap, decoder=decoder))
                         for ev, v in masks.items()} for decoder in ('alpha', 'pmcd')}
    out['source'] = dict(path=str(path.resolve()), sha256=C.sha(path))
    return out
