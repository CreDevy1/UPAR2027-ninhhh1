from collections import Counter
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from . import common as C
from teacher_system.data import read_annotation_csv, AnnotationTable, SurveillanceTransform
from teacher_system.calibration import domain_from_path
from teacher_system.domain_mask import valid_cells
from lc.convention import lc_valid, convention_eval_valid, masked_prior_from_valid

DOMAINS = {'lodo_market': ('PA100k', 'PETA'), 'lodo_pa': ('Market1501', 'PETA'),
           'allsource': ('Market1501', 'PA100k', 'PETA')}
TARGET = {'lodo_market': ('Market1501',), 'lodo_pa': ('PA100k',),
          'allsource': ('Market1501', 'PA100k', 'PETA')}


def subset(table, domains):
    ix = [i for i, p in enumerate(table.image_paths) if domain_from_path(p) in domains]
    return AnnotationTable(table.attribute_names, tuple(table.image_paths[i] for i in ix), table.labels[ix])


def tables(protocol):
    # Fitting only receives this TRAIN subset. Held-out is a separate API.
    return subset(read_annotation_csv(C.TRAIN), DOMAINS[protocol])


def heldout(protocol):
    return subset(read_annotation_csv(C.VAL), TARGET[protocol])


def masks(table, arm='base', convention=False):
    d = [domain_from_path(p) for p in table.image_paths]
    v = valid_cells(d, table.attribute_names, C.config()['domain_mask'])
    if arm == 'lc_m1':
        v = lc_valid(table.attribute_names, d, table.labels, v, True, False)
    elif arm == 'pass':
        # PASS colour candidates replace the PETA-Other mask.
        v[:, list(range(8, 20)) + list(range(21, 33))] = True
    elif arm != 'base':
        raise ValueError('Unknown arm: ' + arm)
    if convention:
        v = convention_eval_valid(table.attribute_names, d, table.labels, v)
    return v


def prior(table, arm):
    return masked_prior_from_valid(table.labels, masks(table, arm))


def audit():
    tr, va = read_annotation_csv(C.TRAIN), read_annotation_csv(C.VAL)
    assert tr.attribute_names == va.attribute_names and len(set(tr.attribute_names)) == 40
    canonical = lambda p: p.replace('\\', '/').casefold()
    a, b = [canonical(p) for p in tr.image_paths], [canonical(p) for p in va.image_paths]
    assert len(a) == len(set(a)) and len(b) == len(set(b)) and not set(a) & set(b)
    assert np.isin(tr.labels, (0, 1)).all() and np.isin(va.labels, (0, 1)).all()
    t = tables('lodo_market')
    masked = int((masks(t) & ~masks(t, 'lc_m1')).sum())
    assert masked == 9277, masked
    return dict(train_counts=dict(Counter(map(domain_from_path, tr.image_paths))),
                val_counts=dict(Counter(map(domain_from_path, va.image_paths))),
                lc_m1_extra_masked=masked, base_extra_masked=0, path_overlap=0,
                identity_overlap_not_excluded=True, heldout_used_for_fitting=False)


class Transform:
    def __init__(self, spec, train, aug='none'):
        if aug not in ('none', 'illum'):
            raise ValueError('Unknown augmentation: ' + aug)
        self.aug = aug
        self.hw = (16 * spec['patch'], 8 * spec['patch'])
        self.train = train
        self.degrader = SurveillanceTransform(self.hw)
        # Both pinned families use mean/std .5. Adapter verifies processor metadata.
        self.mean, self.std = spec['mean'], spec['std']

    def __call__(self, image):
        image = image.convert('RGB')
        if self.train:
            if self.aug == 'illum' and random.random() < .5:
                from .illum import tca
                image = Image.fromarray(tca(np.asarray(image)))
            image = self.degrader.augment_crop(image)
            if random.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = image.resize((self.hw[1], self.hw[0]), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.
        return (x - torch.tensor(self.mean)[:, None, None]) / torch.tensor(self.std)[:, None, None]


class Images(Dataset):
    def __init__(self, table, spec, train=False, aug='none'):
        self.table, self.transform = table, Transform(spec, train, aug)

    def __len__(self):
        return len(self.table.image_paths)

    def __getitem__(self, i):
        path = self.table.image_paths[i]
        with Image.open(C.DATA / path) as im:
            x = self.transform(im)
        return x, torch.from_numpy(self.table.labels[i].copy()), path, i


def worker_seed(_):
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def sampler(table, epoch):
    d = [domain_from_path(p) for p in table.image_paths]
    counts = Counter(d)
    return WeightedRandomSampler([counts[x]**-.5 for x in d], C.config()['draws_per_epoch'], True,
                                 generator=torch.Generator().manual_seed(42000 + epoch))


def loader(table, spec, train=False, epoch=0, batch=64, workers=None, aug='none'):
    return DataLoader(Images(table, spec, train, aug), batch_size=batch,
                      sampler=sampler(table, epoch) if train else None,
                      num_workers=C.loader_workers(workers), worker_init_fn=worker_seed,
                      generator=torch.Generator().manual_seed(42500 + epoch), persistent_workers=False)
