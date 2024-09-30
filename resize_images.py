import cv2
import os
import yaml
from tqdm import tqdm

# Directory Configs
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


VERBOSE = config['resize_settings']['verbose']
INPUT_DIR = os.path.join(os.getcwd(), config['paths']['real_dataset_root_dir'], config['paths']['full_images_dir'])
OUTPUT_DIR = os.path.join(os.getcwd(), config['paths']['real_dataset_root_dir'], config['resize_settings']['output_directory'])
IMAGE_SIZE = config['resize_settings']['image_size']

# Create output directory if it doesn't exist
if VERBOSE:
    print(f"█═══[ Searching for output directory... ]")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    if VERBOSE:
        print(f"  ╚═══[ Created output directory at: {OUTPUT_DIR} ]")
elif VERBOSE:
    print(f"  ╚═══[ Output directory found at: {OUTPUT_DIR} ]")

# Resize all images in the input directory
for _, filename in enumerate(
        tqdm(os.listdir(INPUT_DIR), desc=f"█═══[ Resizing images") if VERBOSE else ''):
    if filename.endswith(".jpg"):
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)

        # Resize The iamge
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # Save resized image to output directory
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, img_resized)

if VERBOSE:
    print(f"  ╚═══[ Images have been resized to: {IMAGE_SIZE}x{IMAGE_SIZE} ]")