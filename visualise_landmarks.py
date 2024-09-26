import os
import os.path as osp
import torch
import cv2
from PIL import Image
import numpy as np


images_dir = "datasets/real" 
landmarks_path = 'landmarks/landmarks.pt'
names_path = 'landmarks/image_filenames.txt'

def draw_landmarks(image, landmarks):
    """Overlay the landmarks on the image using OpenCV"""
    for (x, y, z) in landmarks:
        # Draw circles for each landmark
        cv2.circle(image, (int(x), int(y)), 2, (0, 256, 0), -1)
    return image  
    
def show_real(verbose, image_dir, filenames_path, landmarks_path, amount_to_visualise):
    
    if verbose:
        print(f"█═╦═[ Generating landmark visualisation for real images ]")
        print(f"  ╠═══[ Loading filenames... ]")
    # Get Valid image names
    with open(osp.join(images_dir, filenames_path), 'r') as f:
        tmp_names = f.readlines()
    names = [x.strip() for x in tmp_names]
    
    # store all landmarks in an array
    if verbose:
        print(f"  ╠═══[ Retrieving landmarks... ]")
    all_landmarks = torch.load(osp.join(image_dir, landmarks_path))

    # Create the output directory if it doesn't exist
    if verbose:
        print(f"  ╠═══[ Creating output directory at: {osp.join(image_dir, landmarks_path)} ]")
    output_dir = osp.join(images_dir, 'landmark_layered')
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"  ╠═══[ Drawing Landamrks... ]")
    counter = 0
    # Iterate over the images and overlay landmarks
    for index, landmarks in enumerate(all_landmarks):
        # Load the image using PIL or OpenCV
        img_path = osp.join(images_dir, 'data_resized', names[index])
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"  ╠═══[! Could not load image: {names[index]}. Skipping... !]")
            continue

        # Convert the landmarks from tensor to numpy (if needed) and draw them on the image
        landmarks_np = landmarks.numpy() if torch.is_tensor(landmarks) else np.array(landmarks)
        
        # Overlay the landmarks on the image
        image_with_landmarks = draw_landmarks(image.copy(), landmarks_np)

        # Save the output image with landmarks
        cv2.imwrite(osp.join(output_dir, f"{names[index]}_landmarked.jpg"), image_with_landmarks)

        counter += 1
        if counter >= amount_to_visualise:
            break
        
    if verbose:
        print(f"  ╚═══[ Completed ]")
    
def show_fake(verbose, fake_root_dir, hashes_path, landmarks_path, amount_to_visualise):
    if verbose:
        print(f"█═╦═[ Generating landmark visualisation for fake images ]")
        print(f"  ╠═══[ Loading hashes... ]")
    # Get Valid image names
    with open(osp.join(fake_root_dir, hashes_path), 'r') as f:
        tmp_hashes = f.readlines()
    hashes = [x.strip() for x in tmp_hashes]
    
    # store all landmarks in an array
    if verbose:
        print(f"  ╠═══[ Retrieving landmarks... ]")
    all_landmarks = torch.load(osp.join(fake_root_dir, landmarks_path))

    # Create the output directory if it doesn't exist
    if verbose:
        print(f"  ╠═══[ Creating output directory at: {fake_root_dir} ]")
    output_dir = osp.join(fake_root_dir, 'landmark_layered')
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"  ╠═══[ Drawing Landamrks... ]")
    counter = 0
    # Iterate over the images and overlay landmarks
    for index, landmarks in enumerate(all_landmarks):
        # Load the image using PIL or OpenCV
        img_path = osp.join(fake_root_dir, hashes[index], 'resized_image.jpg')
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"  ╠═══[! Could not load image: {hashes[index]}. Skipping... !]")
            continue

        # Convert the landmarks from tensor to numpy (if needed) and draw them on the image
        landmarks_np = landmarks.numpy() if torch.is_tensor(landmarks) else np.array(landmarks)
        
        # Overlay the landmarks on the image
        image_with_landmarks = draw_landmarks(image.copy(), landmarks_np)

        # Save the output image with landmarks
        cv2.imwrite(osp.join(output_dir, f"{hashes[index]}_landmarked.jpg"), image_with_landmarks)

        counter += 1
        if counter >= amount_to_visualise:
            break
        
    if verbose:
        print(f"  ╚═══[ Completed ]")
    

#show_real(True, "datasets/real", "landmarks/image_filenames.txt", "landmarks/landmarks.pt", 5)
show_fake(True, "datasets/fake", "valid_latent_code_hashes.txt", "landmarks.pt", 5)
