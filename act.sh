#!/usr/bin/env bash
# Example: CLIP/PAC percentage-change between clean vs protected generations.
# Edit paths before running.

python PerChang.py --device cuda:0 \
    --caption_path /path/to/captions.json \
    --PACcheckpoint /path/to/PAC++_clip_ViT-L-14.pth \
    --ori_dataset_path /path/to/clean_generations \
    --adv_dataset_path /path/to/protected_generations

# Optional perceptual metrics (protected input vs original input, or gen vs gen):
# python lpips_sim.py --device cuda:0 \
#     --data1_path /path/to/images_a \
#     --data2_path /path/to/images_b
#
# python psnr_sim.py \
#     --data1_path /path/to/images_a \
#     --data2_path /path/to/images_b
