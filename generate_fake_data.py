# imports
import os
import os.path as osp
import argparse
import json
import yaml
import torch
from hashlib import sha1
from lib import GENFORCE_MODELS, ArcFace, tensor2image
from torchvision import transforms
import shutil
from tqdm import tqdm
from models.load_generator import load_generator
import face_alignment
import cv2

def main():
    
    # Load Config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)


    VERBOSE = config['fake_data_generator_settings']['verbose']
    USE_CUDA = config['fake_data_generator_settings']['use_cuda']
    
    GAN_GENERATOR = config['fake_data_generator_settings']['gan_generator']
    TRUNCATION = config['fake_data_generator_settings']['truncation']
    NUM_SAMPLES = config['fake_data_generator_settings']['num_samples']
    
    FAKE_ROOT_DIR = config['paths']['fake_dataset_root_dir']
    DATA_DIR = osp.join(FAKE_ROOT_DIR, config['fake_data_generator_settings']['data_output_dir'])
    
    LANDMARKS_DIR = osp.join(FAKE_ROOT_DIR, config['paths']['facial_landmarks_dir'])
    ARCFACE_DIR = osp.join(FAKE_ROOT_DIR, config['paths']['arcface_dir'])
    
    # if osp.exists(FAKE_ROOT_DIR):
    #     shutil.rmtree(FAKE_ROOT_DIR)
    # os.makedirs(FAKE_ROOT_DIR, exist_ok=True)
    
    # # -----------------------------------------  CUDA
    
    # use_cuda = False
    # if torch.cuda.is_available():
    #     if USE_CUDA:
    #         use_cuda = True
    #         torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #     else:
    #         print("#. !!! Cuda is avaialable but has not been selected in the configs file !!!")
    #         torch.set_default_tensor_type('torch.FloatTensor')
    # else:
    #     torch.set_default_tensor_type('torch.FloatTensor')


    # #  ------------------------------------------ Pretrained GAN


    # # Build GAN generator model and load with pre-trained weights
    # if VERBOSE:
    #     print("#. Build GAN generator model G and load with pre-trained weights...")
    #     print("  \\__GAN generator : {} (res: {})".format(GAN_GENERATOR, GENFORCE_MODELS[GAN_GENERATOR][1]))
    #     print("  \\__Pre-trained weights: {}".format(GENFORCE_MODELS[GAN_GENERATOR][0]))

    # G = load_generator(model_name=GAN_GENERATOR, latent_is_s='stylegan' in GAN_GENERATOR, verbose=VERBOSE).eval()

    # # Upload GAN generator model to GPU
    # if use_cuda:
    #     G = G.cuda()
        
    # # --------------------------------------------- Feature Models
    
    fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='cpu')
    
    # # === ArcFace ===

    # if VERBOSE:
    #     print("#. Build pre-trained ArcFace model...")

    # arcface_model = ArcFace()
    # arcface_model.eval()
    # arcface_model.float()

    # arcface_img_transform = transforms.Compose([transforms.Resize(256),
    #                                             transforms.CenterCrop(256)])

    # # ----------------------------------- Latent Code Sampling / Generation / Feature Extraction
    
    # # Latent codes sampling
    # if VERBOSE:
    #     print("#. Sample {} {}-dimensional latent codes (Z space)...".format(NUM_SAMPLES, G.dim_z))
    # zs = torch.randn(NUM_SAMPLES, G.dim_z)

    # if use_cuda:
    #     zs = zs.cuda()

    # if VERBOSE:
    #     print("#. Generate images...")
    #     print("  \\__{}".format(osp.join(FAKE_ROOT_DIR, DATA_DIR)))

    # # Iterate over given latent codes
    # latent_code_hashes = []
    # arcface_features = []
    # for i in tqdm(range(NUM_SAMPLES)):
    #     # Un-squeeze current latent code in shape [1, dim] and create hash code for it
    #     z = zs[i, :].unsqueeze(0)
    #     latent_code_hash = sha1(z.cpu().numpy()).hexdigest()
    #     latent_code_hashes.append(latent_code_hash)

    #     # Create directory for current latent code
    #     latent_code_dir = osp.join(FAKE_ROOT_DIR, '{}'.format(latent_code_hash))
    #     os.makedirs(latent_code_dir, exist_ok=True)

    #     # Get W+ latent codes from z code
    #     wp = G.get_w(z, truncation=TRUNCATION)

    #     # Get S latent codes from wp codes
    #     styles_dict = G.get_s(wp)

    #     # Generate image
    #     with torch.no_grad():
    #         img = G(styles_dict)

    #     # Calculate ArcFace features
    #     with torch.no_grad():
    #         img_feat = arcface_model(arcface_img_transform(img))
    #     torch.save(img_feat.cpu(), osp.join(latent_code_dir, 'arcface_features.pt'))
    #     arcface_features.append(img_feat.cpu())

    #     # Save image
    #     tensor2image(img.cpu(), adaptive=True).save(osp.join(latent_code_dir, 'image.jpg'),
    #                                                 "JPEG", quality=95, subsampling=0, progressive=True)
        
    #     # Resize The iamge
    #     img2resize = cv2.imread(osp.join(latent_code_dir, 'image.jpg'))
    #     img_resized = cv2.resize(img2resize, (256, 256))
    #     cv2.imwrite(osp.join(latent_code_dir, 'resized_image.jpg'), img_resized)
        
    #     # Save latent codes in W and S spaces
    #     torch.save(wp.cpu(), osp.join(latent_code_dir, 'latent_code_w+.pt'))
    #     torch.save(styles_dict, osp.join(latent_code_dir, 'latent_code_s.pt'))

    # # Write latent codes hashes to file
    # with open(osp.join(FAKE_ROOT_DIR, 'latent_code_hashes.txt'), 'w') as f:
    #     for h in latent_code_hashes:
    #         f.write(f"{h}\n")

    with open(osp.join(FAKE_ROOT_DIR, 'latent_code_hashes.txt'), 'r') as f:
        hashes = f.readlines()
    latent_code_hashes = [x.strip() for x in hashes]
    
    # # Save features
    # if VERBOSE:
    #     print("#. Save features....")

    # # Save ArcFace features
    # arcface_features = torch.cat(arcface_features)
    # arcface_features_file = osp.join(FAKE_ROOT_DIR, 'arcface_features.pt')
    # if VERBOSE:
    #     print("  \\__ArcFace features : {}".format(arcface_features.shape))
    #     print("  \\__Save @ {}".format(arcface_features_file))
    # torch.save(arcface_features, arcface_features_file)

    torch.set_default_tensor_type('torch.FloatTensor')

    all_landmarks = []
    valid_hashes = []
    for hash_dir in tqdm(latent_code_hashes):
        
        unfiltered_landmark = fa_model.get_landmarks_from_image(osp.join(FAKE_ROOT_DIR, hash_dir, 'resized_image.jpg'))
        if unfiltered_landmark is not None:
            if len(unfiltered_landmark) == 1:
                landmarks = torch.Tensor(unfiltered_landmark)
                torch.save(landmarks.cpu(), osp.join(FAKE_ROOT_DIR, hash_dir, 'landmarks.pt'))
                all_landmarks.append(landmarks.cpu())
                valid_hashes.append(hash_dir)
            else:
                print("Multiface")
        else:
            print("No Face")

    all_landmarks = torch.cat(all_landmarks)
    landmarks_file = osp.join(FAKE_ROOT_DIR, 'landmarks.pt')
    if VERBOSE:
        print("  \\__Landmarks saved")
    torch.save(all_landmarks, landmarks_file)
    
    # Write latent codes hashes to file
    with open(osp.join(FAKE_ROOT_DIR, 'valid_latent_code_hashes.txt'), 'w') as f:
        for h in valid_hashes:
            f.write(f"{h}\n")
    
    
if __name__ == '__main__':
    main()