"""M1/M2 label-convention masks. Labels are never edited; masked cells become UNKNOWN (valid=False)."""
import numpy as np

M1_DOMAIN, M1_CONDITION, M1_TARGET = 'PA100k', 'LowerBody-Type-Skirt&Dress', 'LowerBody-Length-Short'
M2_DOMAIN = 'Market1501'
M2_TARGETS = ['UpperBody-Color-Brown', 'UpperBody-Color-Orange', 'UpperBody-Color-Pink',
              'LowerBody-Color-Orange', 'LowerBody-Color-Red']
PROVENANCE = {
    'M1': ('PA100k LowerBody-Length-Short is derived from the original PA100k "Shorts" attribute: '
           '0 of 9,277 TRAIN Skirt&Dress rows are short (Market 1,276/1,433, PETA 98/547). '
           'UPAR (Specker et al., WACV 2023) manually added lower-body length only for PETA/RAPv2.'),
    'M2': ('Original Market-1501 attributes have 8 upper / 9 lower colours; Upper Brown/Orange/Pink and '
           'Lower Orange/Red have exactly 0 positives in all 26,458 Market TRAIN+VAL rows (structural zeros).'),
}


def lc_valid(names, domains, labels, base_valid, use_m1=True, use_m2=True):
    names = list(names); domains = np.asarray(domains)
    v = np.asarray(base_valid, dtype=bool).copy()
    if use_m1:
        rows = (domains == M1_DOMAIN) & (np.asarray(labels)[:, names.index(M1_CONDITION)] == 1)
        v[rows, names.index(M1_TARGET)] = False
    if use_m2:
        v[np.ix_(domains == M2_DOMAIN, [names.index(a) for a in M2_TARGETS])] = False
    return v


def convention_eval_valid(names, domains, labels, base_valid):
    """Evaluation-side convention: PA100k skirt rows have unknown LowerBody-Length-Short."""
    return lc_valid(names, domains, labels, base_valid, use_m1=True, use_m2=False)


def counts(names, domains, labels, valid_before, valid_after):
    names = list(names); domains = np.asarray(domains); labels = np.asarray(labels)
    out = {}
    for a in [M1_TARGET] + M2_TARGETS:
        j = names.index(a); out[a] = {}
        for d in sorted(set(domains.tolist())):
            m = domains == d
            out[a][d] = dict(
                valid_before=int(valid_before[m, j].sum()), valid_after=int(valid_after[m, j].sum()),
                positive_before=int((labels[m, j] == 1)[valid_before[m, j]].sum()),
                positive_after=int((labels[m, j] == 1)[valid_after[m, j]].sum()),
                negative_before=int((labels[m, j] == 0)[valid_before[m, j]].sum()),
                negative_after=int((labels[m, j] == 0)[valid_after[m, j]].sum()))
    return out


def masked_prior_from_valid(labels, valid):
    """Same Laplace smoothing as upar_system.domain_mask.masked_prior: (sum+0.5)/(count+1)."""
    valid = np.asarray(valid, dtype=bool); labels = np.asarray(labels, dtype=np.float64)
    c = valid.sum(0)
    if np.any(c == 0):
        raise ValueError('An attribute has no valid source rows after LC masking')
    return ((labels * valid).sum(0) + .5) / (c + 1.)
