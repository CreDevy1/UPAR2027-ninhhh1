pe.py, rope.py and config.py are adapted from Meta's Perception Models
(https://github.com/facebookresearch/perception_models, commit 3e352cca660658d4b5c90f42a7808b11469e4c66),
released under the Apache License 2.0 (LICENSE.PE).

Changes: einops replaced with plain torch, DropPath replaced with Identity (drop rate 0),
axial RoPE written in torch, and the non-reentrant gradient checkpointing call fixed.
tokenizer.py keeps the OpenAI CLIP copyright header.

Pretrained weights: facebook/PE-Core-L14-336, commit ee1fdb467ab42497fa2e60e94a79e9d7383c1cbd.
