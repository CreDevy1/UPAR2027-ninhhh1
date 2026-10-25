import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
DATA = Path(os.environ.get('UPAR_DATA', WORK.parent / 'data'))
TRAIN = DATA / 'annotations/task1/train/gt.csv'
VAL = DATA / 'annotations/task1/val/gt.csv'
for name in ('pe_text', 'label_masks'):
    sys.path.insert(0, str(WORK / name))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def config():
    c = read(ROOT / 'configs/uf_v1.json')
    required = dict(seed=42, epochs=6, grid_hw=[16, 8], draws_per_epoch=36000,
                    effective_batch=64, n_update=3378, checkpoint_selection='last',
                    backbone_lr=1e-5, head_lr=3e-4, weight_decay=.01,
                    warmup_ratio=.05, cosine_min_ratio=.1, clip_grad_norm=1.,
                    amp='fp16', amp_initial_scale=1024, gradient_checkpointing=True,
                    alpha=.75, force_one=True)
    for k, v in required.items():
        if c.get(k) != v:
            raise ValueError('Config value changed: ' + k)
    return c


def identity():
    import platform
    import torch
    import transformers
    own = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'uf').glob('*.py'))}
    reused = []
    for folder, names in [
        ('pe_text/teacher_system',
         ['data.py', 'text.py', 'text_prompts.json', 'pe.py', 'rope.py', 'config.py',
          'tokenizer.py', 'bpe_simple_vocab_16e6.txt.gz', 'domain_mask.py', 'calibration.py']),
        ('label_masks/lc', ['convention.py', 'score.py'])]:
        reused += [WORK / folder / n for n in names]
    source = {str(p.relative_to(WORK)): sha(p) for p in reused}
    code_sha = hashlib.sha256(json.dumps([own, source], sort_keys=True).encode()).hexdigest()
    return dict(config_sha=sha(ROOT / 'configs/uf_v1.json'), code_sha=code_sha,
                code_files=own, reused_files=source, train_sha=sha(TRAIN), val_sha=sha(VAL),
                runtime=dict(python=platform.python_version(), torch=torch.__version__,
                             transformers=transformers.__version__, cuda=torch.version.cuda))


def run_id(backbone, protocol, arm, aug='none'):
    suffix = '' if aug == 'none' else '_aug_' + aug
    return f'{protocol}_{backbone}_{arm}{suffix}_s42'


def loader_workers(override=None):
    workers = config()['loader_workers'] if override is None else override
    if not isinstance(workers, int) or workers < 0:
        raise ValueError('loader_workers must be a nonnegative integer')
    return workers


def validate_milestones():
    """Code/data identity and per-backbone micro-batch sizes used by engine.run."""
    return identity(), read(ROOT / 'configs/resources.json')
