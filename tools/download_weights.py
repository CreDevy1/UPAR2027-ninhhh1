"""Download the released files from Hugging Face and check their SHA-256.

    python tools/download_weights.py                 # submitted packages + fine-tuned checkpoints
    python tools/download_weights.py --packages      # only the two submitted packages
    python tools/download_weights.py --pretrained    # PE-Core L/14-336 and SigLIP 2 L/16-256, needed for training

Packages and pretrained weights go to weights/; fine-tuned checkpoints go to framework/recipes/runs_*/,
where the packaging scripts expect them. Only the standard library is used for the first two modes;
--pretrained also needs huggingface_hub (installed with transformers).
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'weights'
RECIPES = REPO / 'framework' / 'recipes'
HF = 'https://huggingface.co/Carridevy/UPAR2027-ninhhh1/resolve/main/'

PACKAGES = {
    'submissions/t1_pel_hr_tv_A1_20261021.zip': '73133aae50c1e5f32620cf8026c950e0b0f48c6f49d9681e64c90cc71ebb191c',
    'submissions/t2_v4b.zip': 'c21427daf40e470c8e895cc6bbf85dec4ab6c8af60ea80a0f6015fcfa4efd650',
}
# Fine-tuned checkpoints and the run files the packaging scripts read; they go to framework/recipes/.
CHECKPOINTS = {
    'runs_hr_tv/allsource_pel_hr_lc_m1_s42/epoch6.pt': '1734cca370d80e26dd2b912006e69bfa68ce5c400f8a195a5645b3fcb6899d24',
    'runs_hr_tv/allsource_pel_hr_lc_m1_s42/STATUS.json': None,
    'runs_hr_tv/allsource_pel_hr_lc_m1_s42/result.json': None,
    'runs_hr_tv/NOTE_pel_hr.json': None,
    'runs_tv/allsource_pel_lc_m1_s42/epoch6.pt': '21364f3afbc533b50c6d78bd98c85b76718d9150a121568fd6935335d1ad378d',
    'runs_tv/allsource_sl2l_lc_m1_s42/epoch6.pt': 'a8623705c2702f3575d43621c56bd99a6c9aee0f05538bc5639dc1f560030b7b',
}
PE_L = ('facebook/PE-Core-L14-336', 'ee1fdb467ab42497fa2e60e94a79e9d7383c1cbd', 'PE-Core-L14-336.pt',
        '0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8')
SIGLIP = ('google/siglip2-large-patch16-256', '787800c8990e6f058423089178e718139608408c')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest, digest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (digest is None or sha256(dest) == digest):
        print('ok (cached)', dest.relative_to(REPO))
        return
    part = dest.with_suffix(dest.suffix + '.part')
    print('downloading', url)
    with urllib.request.urlopen(url) as r, open(part, 'wb') as f:
        while True:
            block = r.read(8 << 20)
            if not block:
                break
            f.write(block)
    got = sha256(part)
    if digest is not None and got != digest:
        part.unlink()
        sys.exit(f'SHA-256 mismatch for {dest.name}: {got}')
    part.replace(dest)
    print('ok', dest.relative_to(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--packages', action='store_true')
    ap.add_argument('--checkpoints', action='store_true')
    ap.add_argument('--pretrained', action='store_true')
    a = ap.parse_args()
    if not (a.packages or a.checkpoints or a.pretrained):
        a.packages = a.checkpoints = True
    if a.packages:
        for name, digest in PACKAGES.items():
            fetch(HF + name, OUT / name, digest)
    if a.checkpoints:
        for name, digest in CHECKPOINTS.items():
            fetch(HF + name, RECIPES / name, digest)
    if a.pretrained:
        repo, rev, name, digest = PE_L
        fetch(f'https://huggingface.co/{repo}/resolve/{rev}/{name}', OUT / 'pretrained' / name, digest)
        from huggingface_hub import snapshot_download
        repo, rev = SIGLIP
        snapshot_download(repo, revision=rev)                                   # model weights, default HF cache
        snapshot_download(repo, revision=rev, cache_dir=str(OUT / 'hf_cache'))  # tokenizer files read at training time
        print('ok', repo, rev)


if __name__ == '__main__':
    main()
