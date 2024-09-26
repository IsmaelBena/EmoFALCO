import os
import os.path as osp
import torch
import json
import yaml

from torch.utils import data
from torchvision import transforms

from lib import DATASETS, CelebAHQ, ArcFace
from tqdm import tqdm

import face_alignment
from skimage import io

import multiprocessing as mp

def main():
    # Load Config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    VERBOSE = config['feature_extraction_settings']['verbose']
    USE_CUDA = config['feature_extraction_settings']['use_cuda']
    BATCH_SIZE = config['feature_extraction_settings']['batch_size']

    REAL_ROOT_DIR = config['paths']['real_dataset_root_dir']
    LANDMARKS_INPUT = config['feature_extraction_settings']['landmarks_input']
    LANDMARKS_DIR = osp.join(REAL_ROOT_DIR, config['paths']['facial_landmarks_dir'])
    ARCFACE_DIR = osp.join(REAL_ROOT_DIR, config['paths']['arcface_dir'])

    #############################################################
    #                   Extract Facial Landmarks                #
    #############################################################


    # Set device
    device = 'cpu'

    # Check if the landmarks target directory exists, if not then create it.
    if VERBOSE:
        print(f"█═╦═[ Searching for landmarks output directory... ]")
    if not os.path.exists(LANDMARKS_DIR):
        os.makedirs(LANDMARKS_DIR)
        if VERBOSE:
            print(f"  ╚═══[ Created landmarks output directory at: {LANDMARKS_DIR} ]")
    elif VERBOSE:
        print(f"  ╚═══[ Output directory found at: {LANDMARKS_DIR} ]")

    # Buliding Face Alignment model used to extract facial landmarks
    if VERBOSE:
        print(f"█═══[ Building Face Alignment Model... ]")

    fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='cpu')

    # Generating landmarks from the targetted images directory
    if VERBOSE:
        print(f"█═╦═[ Generating landmarks from images found at: {osp.join(REAL_ROOT_DIR, LANDMARKS_INPUT)} ]")
    landmarks_dict = fa_model.get_landmarks_from_directory(osp.join(REAL_ROOT_DIR, LANDMARKS_INPUT))
    if VERBOSE:
        print(f"  ╚═══[ Landmarks dictionary generated ]")

    """ 
    Looping through the output of the model, which is in the format of:
    -  {'filename': [landmark-data]} 
    and converting the keys to:
    - A txt file with 1 filename on each line
    - A Tensor representing all the landmarks to be saved at a pt file
    Tensor index should match with the line of the txt file in order to represent which landmark belongs to which file.
    """
    img_filenames = []
    landmarks = []
    
    for _, (filepath, landmark) in enumerate(
            tqdm(landmarks_dict.items(), desc="█═╦═[ Formatting Face Alignment outputs... ]")):

        if landmark is not None:
            if len(landmark) == 1:
                try:
                    landmarks.append(torch.FloatTensor(landmark).cpu())
                    _, filename = osp.split(filepath)
                    img_filenames.append(filename)
                except:
                    print(f"  ╠═══[! There was no face detected in {filename}, appending to the tensor threw and error. !]")
            else:
                print(f"  ╠═══[! Multiple Faces were detected in {filename}, the image was skipped. !]")
        else:
            print(f"  ╠═══[! No Faces were detected in {filename}, the image was skipped. !]")
                
    
    print("  ╚═══[ Formatting complete ]")

    # Create target files
    img_filenames_file = osp.join(LANDMARKS_DIR, 'image_filenames.txt')
    landmarks_file = osp.join(LANDMARKS_DIR, 'landmarks.pt')

    if VERBOSE:
        print(f"█═╦═ [ Saving filenames to: {img_filenames_file} ]")
    with open(img_filenames_file, 'w') as f:
        for h in img_filenames:
            f.write(f"{h}\n")
    if VERBOSE:
        print(f"  ╚═══[ Filenames saved ]")
    
    # Save FA features
    if VERBOSE:
        print(f"#. Saving landmarks to: {landmarks_file}")
    landmarks = torch.cat(landmarks)
    torch.save(landmarks, landmarks_file)
    if VERBOSE:
        print(f"  ╚═══[ Landmarks saved ]")

    #############################################################
    #                        CUDA                               #
    #############################################################

    use_cuda = False
    if torch.cuda.is_available():
        if USE_CUDA:
            use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print("█═══[! Cuda is avaialable but has not been selected in the configs file !]")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Set device
    device = 'cuda' if use_cuda else 'cpu'

    if VERBOSE:
        print(f"█═══[ Device is set to: {device} ]")

    #############################################################
    #                   Extract Arcface Features                #
    #############################################################

    # Check if the landmarks target directory exists, if not then create it.
    if VERBOSE:
        print(f"█═╦═[ Searching for Arcface output directory... ]")
    if not os.path.exists(ARCFACE_DIR):
        os.makedirs(ARCFACE_DIR)
        if VERBOSE:
            print(f"  ╚═══[ Created Arcface output directory at: {ARCFACE_DIR} ]")
    elif VERBOSE:
        print(f"  ╚═══[ Output directory found at: {ARCFACE_DIR} ]")

    # load_dataset
    if VERBOSE:
        print(f"█═╦═[ Loading dataset... ]")
    dataset = CelebAHQ(root_dir=REAL_ROOT_DIR, subset='train+val+test')
    dataloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    if VERBOSE:
        print(f"  ╚═══[ Dataset loaded ]")

    # Buliding Pretrained Arcface model used to extract identity related features
    if VERBOSE:
        print(f"█═╦═[ Building Pretrained Arcface Model... ]")
    arcface_model = ArcFace()
    arcface_model.eval()
    arcface_model.float()

    arcface_img_transform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(256)])

    # Extract Arcface Features
    img_filenames = []
    arcface_features = []
    for i_batch, data_batch in enumerate(
            tqdm(dataloader, desc="  ╚═╦═[ Extracting Arcface Features... ]" if VERBOSE else '')):
        
        # Keep batch images' names
        img_orig_id = []
        for f in data_batch[1]:
            img_orig_id.append(osp.basename(f))
        img_filenames.extend(list(img_orig_id))
        
        with torch.no_grad():
            img_feat = arcface_model(arcface_img_transform(data_batch[0]).to(device))
        arcface_features.append(img_feat.cpu())
        
    print("    ╚═══[ Feature extraction complete ]")
    
    
    # Save dataset images' filenames
    img_filenames_file = osp.join(ARCFACE_DIR, 'image_filenames.txt')
    if VERBOSE:
        print(f"█═╦═[ Saving filenames to: {img_filenames_file} ]")
    with open(img_filenames_file, 'w') as f:
        for h in img_filenames:
            f.write(f"{h}\n")
    if VERBOSE:
        print(f"  ╚═══[ Filenames saved ]")
        
        
    # Create target file
    arcface_features_file = osp.join(ARCFACE_DIR, 'arcface_features.pt')
    # Save ArcFace features
    if VERBOSE:
        print(f"█═╦═[ Saving Features to: arcface_features.pt ]")
    arcface_features = torch.cat(arcface_features)
    torch.save(arcface_features, arcface_features_file)
    if VERBOSE:
        print(f"  ╠═══[ ArcFace features shape: {arcface_features.shape} ]")
        print(f"  ╚═══[ ArcFace features saved to: {arcface_features_file} ]")
    
if __name__ == '__main__':
    #mp.set_start_method('spawn')
    main()