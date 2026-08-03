import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import clip
import torch

from CP_scores import get_cp_score

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Getting ITA scores.')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--PACcheckpoint', type=str, default="PAC++_clip_ViT-L-14.pth", help='PAC-S++ checkpoint path')
    parser.add_argument('--SEED', type=int, nargs='+', default=[9222], help='seeds (kept for CLI compat)')
    parser.add_argument('--ori_dataset_path', type=str, required=True, help='reference image folder')
    parser.add_argument('--adv_dataset_path', type=str, required=True, help='protected / transformed image folder')
    return parser.parse_args()

def main(args):

    clip_model, _ = clip.load("ViT-B/32", device=args.device, jit=False)
    clip_model.eval()

    pac_model, pac_preprocess = clip.load("ViT-L/14", device=args.device)
    pac_model = pac_model.float()
    checkpoint = torch.load(args.PACcheckpoint, map_location=args.device, weights_only=True)
    pac_model.load_state_dict(checkpoint['state_dict'], strict=False)
    pac_model.eval()
    adv_dataset_path = args.adv_dataset_path
    if any(os.path.isdir(os.path.join(adv_dataset_path, f)) for f in os.listdir(adv_dataset_path)):
        image_files = []
        for subfolder in sorted(os.listdir(adv_dataset_path)):
            subfolder_path = os.path.join(adv_dataset_path, subfolder)
            if os.path.isdir(subfolder_path):
                image_files.extend([os.path.join(subfolder, f) for f in sorted(os.listdir(subfolder_path))
                                if f.lower().endswith(('.jpg', '.png'))])
    else:
        image_files = [f for f in sorted(os.listdir(adv_dataset_path)) 
                    if f.lower().endswith(('.jpg', '.png'))]
    print(image_files)
    c_score, p_score = [], []
    for img in tqdm(image_files, colour="blue", desc="images", leave=False):
        prompt = "change to style to Picasso"             
        ori_image = Image.open(os.path.join(args.ori_dataset_path, img))
        adv_image = Image.open(os.path.join(args.adv_dataset_path, img))
        input_images = [ori_image, adv_image]
    
        # Compute scores
        clip_score = get_cp_score(clip_model, None, input_images, [prompt] * len(input_images), args.device, "clip")

        ori_clip_scores = clip_score[0]
        adv_clip_scores = clip_score[1]

        pac_score = get_cp_score(pac_model, pac_preprocess, input_images, [prompt] * len(input_images), args.device, "pac")
        ori_pac_scores = pac_score[0]
        adv_pac_scores = pac_score[1]

        c_score.append(adv_clip_scores)
        p_score.append(adv_pac_scores)

    print("Mean CLIP: ", sum(c_score)/len(c_score))
    print("Mean PAC: ", sum(p_score)/len(p_score))

if __name__ == '__main__':
    args = get_args()
    main(args)