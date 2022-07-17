import torch
import argparse
from utils import load_checkpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
import torch.optim as optim
import config
from generator import Generator
from PIL import Image
import numpy as np


parser = argparse.ArgumentParser(description="pix2pix ariel to map")
parser.add_argument("-i", "--image", type=str, required=True, help="origin image")
args = parser.parse_args()

gen = Generator(in_channels=3, features=64).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
load_checkpoint(
    config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
)
input_image = np.array(Image.open(args.image))
transform = A.Compose(
    [A.Resize(width=256, height=256),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
    ToTensorV2(),]
)
gen.eval()
with torch.no_grad():
    augmentations = transform(image=input_image)
    input_image = augmentations["image"]
    input_image = input_image.unsqueeze(0)
    input_image = input_image.to(config.DEVICE)
    y_fake = gen(input_image)
    y_fake = y_fake * 0.5 + 0.5  # remove normalization#
    save_image(y_fake, "y_result.png")
    save_image(input_image * 0.5 + 0.5, "input.png")
    print(input_image.shape)
