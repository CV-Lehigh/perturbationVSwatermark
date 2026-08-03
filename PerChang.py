import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from matplotlib import pyplot as plt
import clip
import torch

from CP_scores import get_cp_score

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Getting ITA Percentage Change scores.')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--PACcheckpoint', type=str, default="PAC++_clip_ViT-L-14.pth", help='PAC-S++ checkpoint path')
    parser.add_argument('--SEED', type=int, nargs='+', default=[9222, 42, 66, 123, 999], help='seeds')
    parser.add_argument('--caption_path', type=str, required=True, help='JSON captions with Img / transfer_style')
    parser.add_argument('--ori_dataset_path', type=str, required=True, help='clean generation folder')
    parser.add_argument('--adv_dataset_path', type=str, required=True, help='protected generation folder')
    return parser.parse_args()

def plot_score_histograms(c_score, p_score, save_path='score_histograms.png'):
    # Convert to numpy arrays
    c_score = np.array(c_score)
    p_score = np.array(p_score)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot CLIP histogram
    ax1.hist(c_score, bins=20, color='blue', alpha=0.7)
    ax1.set_title('CLIP Score Changes')
    ax1.set_xlabel('Relative Score Change')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Add CLIP statistics
    clip_stats = f'Mean: {np.mean(c_score):.3f}\nStd: {np.std(c_score):.3f}'
    ax1.text(0.05, 0.95, clip_stats, 
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot PAC histogram
    ax2.hist(p_score, bins=20, color='red', alpha=0.7)
    ax2.set_title('PAC Score Changes')
    ax2.set_xlabel('Relative Score Change')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Add PAC statistics
    pac_stats = f'Mean: {np.mean(p_score):.3f}\nStd: {np.std(p_score):.3f}'
    ax2.text(0.05, 0.95, pac_stats,
             transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_histogram(c_score, p_score, save_path='combined_histogram.png'):
    plt.figure(figsize=(10, 6))
    
    # Plot both histograms
    plt.hist(c_score, bins=20, alpha=0.5, label='CLIP', color='blue')
    plt.hist(p_score, bins=20, alpha=0.5, label='PAC', color='red')
    
    # Add labels and title
    plt.xlabel('Relative Score Change')
    plt.ylabel('Frequency')
    plt.title('Distribution of Score Changes')
    plt.legend()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add statistics text box
    stats_text = f'CLIP: mean={np.mean(c_score):.3f}, std={np.std(c_score):.3f}\n'
    stats_text += f'PAC: mean={np.mean(p_score):.3f}, std={np.std(p_score):.3f}'
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    with open(args.caption_path, "r") as f:
        captions = json.load(f)

    clip_model, _ = clip.load("ViT-B/32", device=args.device, jit=False)
    clip_model.eval()

    pac_model, pac_preprocess = clip.load("ViT-L/14", device=args.device)
    pac_model = pac_model.float()
    checkpoint = torch.load(args.PACcheckpoint, map_location=args.device, weights_only=True)
    pac_model.load_state_dict(checkpoint['state_dict'], strict=False)
    pac_model.eval()
    
    c_score, p_score = [], []

    for caption in tqdm(captions, colour="blue", desc="images", leave=False):
        img_name = caption['Img']
        prompt = "change to style to Picasso"     #"change to style to "+caption['transfer_style']

        if os.path.exists(os.path.join(args.ori_dataset_path, str(args.SEED[0]) +'_'+caption['transfer_style']+'_'+img_name)):
            for seed in tqdm(args.SEED, colour="green", desc="Seeds", leave=True):
                ori_image = Image.open(os.path.join(args.ori_dataset_path, str(seed) +'_'+ caption['transfer_style']+'_'+img_name))
                try:
                    adv_image = Image.open(os.path.join(args.adv_dataset_path, str(seed) +'_'+ caption['transfer_style']+'_'+img_name.replace('.jpg','.png')))     #img_name.replace('.jpg','.png')
                except:
                    adv_image = Image.open(os.path.join(args.adv_dataset_path, str(seed) +'_'+ caption['transfer_style']+'_'+img_name))
                input_images = [ori_image, adv_image]
            
            # Compute scores
            clip_score = get_cp_score(clip_model, None, input_images, [prompt] * len(input_images), args.device, "clip")

            ori_clip_scores = clip_score[0]
            adv_clip_scores = clip_score[1]

            pac_score = get_cp_score(pac_model, pac_preprocess, input_images, [prompt] * len(input_images), args.device, "pac")
            ori_pac_scores = pac_score[0]
            adv_pac_scores = pac_score[1]
            
            c_score.append((adv_clip_scores - ori_clip_scores)/ori_clip_scores)
            p_score.append((adv_pac_scores - ori_pac_scores)/ori_pac_scores)
    # plot_score_histograms(c_score, p_score)  # Separate histograms
    # plot_combined_histogram(c_score, p_score)  # Combined histogram
    print("Mean CLIP Percentage Change: ", sum(c_score)/len(c_score))
    print("Mean PAC Percentage Change: ", sum(p_score)/len(p_score))

if __name__ == '__main__':
    args = get_args()
    main(args)