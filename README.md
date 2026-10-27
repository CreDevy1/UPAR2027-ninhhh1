# UPAR Challenge 2027: Track 1 and Track 2 (team ninhhh1)

Training and inference code for our entries in the [UPAR Challenge 2027](https://vap.aau.dk/rws/challenge/)
(RWS workshop at WACV 2027): Track 1, pedestrian attribute recognition, and Track 2, attribute-based person retrieval.
The method is described in our workshop paper, *Convention-Aware Fine-Tuning and Closed-Set Re-Ranking for Pedestrian
Attribute Recognition and Retrieval: UPAR 2027 System Report* ([OpenReview, submission 65](https://openreview.net/forum?id=TvG118JOn1)).

Weights: [huggingface.co/Carridevy/UPAR2027-ninhhh1](https://huggingface.co/Carridevy/UPAR2027-ninhhh1)

## Results

Final submissions (Codabench user `ninhhh1`, test set, NVIDIA A10).

| Track | Submission | Package | Score | Inference time |
|---|---|---|---|---|
| 1 | 1011642 | `t1_ens2_hr_tv_pmcd_20261021.zip` | Challenge average 0.7902 (mA 0.7923, instance F1 0.7881) | 1030 s |
| 2 | 1011654 | `t2_v5a_hr_fp32.zip` | mADM 0.7036 (mAP 0.4257, Rank-1 0.4391) | 685 s |

Both entries use the same two encoders, each fine-tuned end to end on TRAIN+VAL with a head initialised from its
text encoder and a label mask for source labels that do not follow the UPAR guideline:
PE-Core L/14 at 336x336 and SigLIP 2 L/16 at 384x384 (both a 24x24 patch grid, 576 tokens).
Weights are stored in FP32 and inference runs under FP16 autocast.

- **Track 1.** The logits of the two encoders are averaged. Prior-shifted logits are decoded per image by a greedy
  decoder with at least one label per exclusive group.
- **Track 2.** Averaged logits give a Bernoulli likelihood for every query and gallery image, followed by query-set
  centring, a Sinkhorn normalisation over queries and a similarity-gated diffusion on a kNN graph of the gallery
  built from the PE-Core features.

Earlier entries, kept for reference (code in `inference/earlier/`):

| Track | Submission | Package | Score |
|---|---|---|---|
| 1 | 1007246 | `t1_pel_hr_tv_A1_20261021.zip` | Challenge average 0.7906 (PE-Core L/14 336x336 only) |
| 2 | 1004665 | `t2_v4b.zip` | mADM 0.6979 (PE-Core L/14 224x112 + SigLIP 2 L/16 256x128) |

## Layout

```
inference/track1, inference/track2   code of the two final packages (weights are on Hugging Face)
inference/earlier                    code of the two earlier packages
framework/core/uf                    training loop, data pipeline, augmentation, label masks, head calibration
framework/pe_text/teacher_system     PE-Core vision and text towers, tokenizer, prompts
framework/label_masks/lc             masks for source labels that do not follow the UPAR guideline
framework/recipes/scripts            training and packaging scripts
framework/recipes/configs            Track 2 re-ranking parameters, high-resolution run settings
tools/                               download, local run of a package, rebuild check
docker/                              grader image (inference) and training image
```

## 1. Run the submitted packages

The zips on Hugging Face are byte-for-byte the files we submitted. They need only torch, torchvision, numpy and
Pillow, as on the grader.

```bash
python tools/download_weights.py --packages          # -> weights/submissions/*.zip, SHA-256 checked
```

`tools/run_package.py` loads a package and calls it the way the Codabench ingestion program does
(`load_model` + `predict_batch` for Track 1, one `rank_gallery` call for Track 2). Point `--data` to the organizers'
`data/` folder (created by `download_datasets.py` of the
[challenge repository](https://github.com/speckean/UPAR-Challenge-2027)).

```bash
docker build -f docker/Dockerfile -t upar2027-infer .
docker run --gpus all --rm -v "$PWD":/repo -v /path/to/UPAR-Challenge-2027/data:/data upar2027-infer \
    --track 1 --package /repo/weights/submissions/t1_ens2_hr_tv_pmcd_20261021.zip --data /data --split val --grader-imports
docker run --gpus all --rm -v "$PWD":/repo -v /path/to/UPAR-Challenge-2027/data:/data upar2027-infer \
    --track 2 --package /repo/weights/submissions/t2_v5a_hr_fp32.zip --data /data --split val --grader-imports
```

Without Docker, the same commands work with `python tools/run_package.py ...` in any environment with
torch 2.4.1. The script prints a functional score on the chosen split. The models were trained on TRAIN+VAL, so
VAL scores are not a measure of accuracy.

## 2. Train

### Environment and data

```bash
conda env create -f environment.yml       # Python 3.11, torch 2.4.1 + CUDA 12.1, transformers 4.49.0
conda activate upar2027
export UPAR_DATA=/path/to/UPAR-Challenge-2027/data   # default: ./data
python tools/download_weights.py --pretrained        # PE-Core-L14-336.pt and the SigLIP 2 L/16-256 and L/16-384 snapshots
```

Training reads `annotations/task1/train/gt.csv` and `annotations/task1/val/gt.csv`, merges them into
`framework/recipes/data/train_val_gt.csv` (131,076 images) and uses the image paths in those files relative to
`UPAR_DATA`. Using the validation split for the final models is allowed by the 2027 terms. No other data is used.

### Final entries (Track 1 and Track 2)

```bash
python -u framework/recipes/scripts/train_hr_tv.py pel        # PE-Core L/14, 336x336
python -u framework/recipes/scripts/train_hr_tv.py sl2l       # SigLIP 2 L/16, 384x384
python -u framework/recipes/scripts/build_hr_final_pkgs.py    # -> framework/recipes/submissions/t1_ens2_hr_tv_pmcd_20261021.zip
                                                              #    framework/recipes/submissions/t2_v5a_hr_fp32.zip
```

Each model runs 6 epochs of 36,000 sampled images (3,378 updates, effective batch 64, seed 42) and the last checkpoint
is kept. All hyper-parameters are in `framework/core/configs/uf_v1.json`; the Track 2 re-ranking parameters are in
`framework/recipes/configs/post_v4b.json` and are copied into the package configuration. On one RTX 4060 laptop GPU
(8 GB) training took 6.5 h for the PE-Core model and 5.7 h for the SigLIP 2 model (plus head calibration). GPU training is not bit-exact
between runs, so a retrained model will differ slightly from the released one.

### Earlier entries

```bash
python -u framework/recipes/scripts/build_hr_pel_pkg.py       # Track 1, 1007246, from the pel_hr checkpoint above
python -u framework/recipes/scripts/train_tv.py pel           # Track 2, 1004665: PE-Core L/14, 224x112
python -u framework/recipes/scripts/train_tv.py sl2l          #                    SigLIP 2 L/16, 256x128
python -u framework/recipes/scripts/build_pkg.py --name t2_v4b --members pel,sl2l --feat pel --budget 270 \
    --post @configs/post_v4b.json \
    --ckpt pel=framework/recipes/runs_tv/allsource_pel_lc_m1_s42/epoch6.pt \
    --ckpt sl2l=framework/recipes/runs_tv/allsource_sl2l_lc_m1_s42/epoch6.pt
```

### Rebuild the packages from the released checkpoints

```bash
python tools/download_weights.py                 # packages + fine-tuned checkpoints into framework/recipes/runs_*
python -u framework/recipes/scripts/build_hr_final_pkgs.py
python tools/verify_rebuild.py framework/recipes/submissions/t1_ens2_hr_tv_pmcd_20261021.zip weights/submissions/t1_ens2_hr_tv_pmcd_20261021.zip
python tools/verify_rebuild.py framework/recipes/submissions/t2_v5a_hr_fp32.zip weights/submissions/t2_v5a_hr_fp32.zip
```

`verify_rebuild.py` compares every weight tensor and the configuration of the two packages. The packaging scripts
run on CPU and need no `transformers`.

## License

Code: Apache License 2.0. The PE-Core files are adapted from Meta's Perception Models (Apache 2.0, see
`framework/pe_text/NOTICE.PE.md`); the pretrained PE-Core and SigLIP 2 weights keep their original licenses.
