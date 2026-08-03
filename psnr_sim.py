import numpy as np
import os
import argparse
from PIL import Image
from tqdm import tqdm

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description="The usage of PSNR between 2 images. ")
    args.add_argument('--data1_path', type=str, required=True, help='path of image set 1')
    args.add_argument('--data2_path', type=str, required=True, help='path of image set 2')
    
    parsed_args = args.parse_args()

    scores = []
    for img_name in tqdm(os.listdir(parsed_args.data1_path)):
        img1 = Image.open(os.path.join(parsed_args.data1_path, img_name))
        for img_name2 in os.listdir(parsed_args.data2_path):
            if img_name2.split('.')[0] in img_name:
                img2 = Image.open(os.path.join(parsed_args.data2_path, img_name2))
                if img2.size != img1.size and img2.size != (512,512):
                    img1 = img1.resize((512,512), Image.Resampling.BICUBIC)
                    assert img1.size == (512,512)
                    img2 = img2.resize((512,512), Image.Resampling.BICUBIC)
        scores.append(calculate_psnr(img1, img2))
    print("Mean PSNR: ", sum(scores)/len(scores))