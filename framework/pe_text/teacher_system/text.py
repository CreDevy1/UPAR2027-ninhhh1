import torch
from torch import nn

def normalize(x):
    norms = x.norm(dim=-1, keepdim=True)
    if not torch.isfinite(x).all() or (norms < 1e-10).any():
        raise ValueError("Non-finite or degenerate text embedding/direction")
    return x / norms


def unit_mean(embeddings):
    return normalize(normalize(embeddings.float()).mean(0))


def validate_prompts(prompts, names):
    attrs = prompts["attributes"]
    if len(attrs) != 40 or [a["name"] for a in attrs] != list(names):
        raise ValueError("Prompts must contain 40 attributes in assets order")
    strings = list(prompts["templates"]) + list(prompts["generic"])
    for a in attrs:
        if not a["pos"] or set(a) != {"name", "pos", "neg", "group"}:
            raise ValueError("Invalid prompt attribute")
        strings += a["pos"] + (a["neg"] or [])
    if any(not isinstance(s, str) or not s or s != s.lower() for s in strings):
        raise ValueError("Every prompt string must be nonempty and lowercase")


def build_directions(prompts, embed):
    """embed receives expanded lowercase strings and returns raw FP32 features."""
    used = {}

    def u(phrases, key):
        texts = [t.format(p).lower() for t in prompts["templates"] for p in phrases]
        used[key] = texts
        return unit_mean(embed(texts))

    attrs = prompts["attributes"]
    pos = {a["name"]: u(a["pos"], a["name"] + "/pos") for a in attrs}
    generic = u(prompts["generic"], "generic")
    directions, cosines, negatives = [], [], {}
    for a in attrs:
        name = a["name"]
        if a["neg"]:
            neg = u(a["neg"], name + "/neg")
        elif a["group"]:
            peers = [pos[b["name"]] for b in attrs
                     if b["group"] == a["group"] and b["name"] != name]
            if not peers:
                raise ValueError(f"No negative group peers for {name}")
            neg = normalize(torch.stack(peers).mean(0))
        else:
            neg = generic
        negatives[name] = neg
        cosines.append(float(pos[name] @ neg))
        directions.append(normalize(pos[name] - neg))
    return torch.stack(directions).float(), cosines, used, pos, negatives


def fit_calibration(scores, y, valid):
    """Unregularized full-batch separable logistic regression, FP64 LBFGS."""
    scores, y, valid = scores.double(), y.double(), valid.double()
    if (valid.sum(0) == 0).any():
        raise ValueError("No unmasked train rows for an attribute")
    a = nn.Parameter(torch.zeros(scores.shape[1], dtype=torch.float64))
    prior = ((y*valid).sum(0)+.5)/(valid.sum(0)+1)
    c = nn.Parameter(torch.logit(prior))
    opt = torch.optim.LBFGS([a, c], lr=1, max_iter=1000, max_eval=1250,
                            tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        # Sum attribute mean losses: independent optimum, robust gradient tolerance.
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(scores*a+c, y, reduction="none")
                * valid).sum(0).div(valid.sum(0)).sum()
        loss.backward()
        return loss

    opt.step(closure)
    closure()
    grad = max(float(a.grad.abs().max()), float(c.grad.abs().max()))
    if not torch.isfinite(a).all() or not torch.isfinite(c).all() or grad > 1e-5:
        raise RuntimeError(f"Text calibration did not converge: max_gradient={grad}")
    return a.detach().float(), c.detach().float(), dict(max_gradient=grad,
            iterations=int(opt.state[a]["n_iter"]), loss=float(closure().detach()))


