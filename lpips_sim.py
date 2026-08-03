import os
import lpips
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to a fixed size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    image = transform(image)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    return image

if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description="The usage of LPIPS perceptual similarity between 2 images. ")
    args.add_argument('--device', type=str, default="cuda:1", help='device')
    args.add_argument('--data1_path', type=str, required=True, help='path of image set 1')
    args.add_argument('--data2_path', type=str, required=True, help='path of image set 2')
    
    parsed_args = args.parse_args()
    device = parsed_args.device
    
    loss_fn = lpips.LPIPS(net='alex').to(device)  
    # net='vgg' closer to "traditional" perceptual loss, when used for optimization
    # net='alex' best forward scores
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
        # images should be RGB, normalized to [-1,1] range
        distance = loss_fn.forward(transform_image(img1).to(device), transform_image(img2).to(device)).item()
        scores.append(distance)
    print("Mean LPIPS: ", sum(scores)/len(scores))