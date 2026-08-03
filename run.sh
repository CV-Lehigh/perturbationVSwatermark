#!/usr/bin/env bash
# Example: absolute CLIP/PAC ITA scores on folder-paired image sets
# (e.g. Mist / Glaze transformation outputs). Edit paths before running.

python PerChang1.py \
    --device cuda:0 \
    --PACcheckpoint /path/to/PAC++_clip_ViT-L-14.pth \
    --SEED 9222 \
    --ori_dataset_path /path/to/reference_images \
    --adv_dataset_path /path/to/mist_or_glaze_images
