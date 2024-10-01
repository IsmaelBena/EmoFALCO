import os.path as osp
import argparse
from sklearn.neighbors import NearestNeighbors
import torch
import json
import yaml
from lib import DATASETS

"""
Original file from: https://github.com/chi0tzp/FALCO,
Major modifications made for the EmoFalco Dissertation project.
"""

def main():
    # Load Config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    VERBOSE = config['feature_extraction_settings']['verbose']
    USE_CUDA = config['feature_extraction_settings']['use_cuda']

    REAL_ROOT_DIR = config['paths']['real_dataset_root_dir']
    FAKE_ROOT_DIR = config['paths']['fake_dataset_root_dir']
    LANDMARKS_FOLDER = config['paths']['facial_landmarks_dir']
    
    NN_ALGORITHMS = [config['pair_nn_settings']['algorithm']]
    if NN_ALGORITHMS[0] == 'all':
        NN_ALGORITHMS = ['auto', 'ball_tree', 'kd_tree', 'brute']
    
    NN_METRICS = [config['pair_nn_settings']['metrics']]
    if NN_METRICS[0] == 'all':
        NN_METRICS = ['euclidean', 'cosine']

    #############################################################
    #                        Load real dataset                  #
    #############################################################
    
    real_dataset_landmarks_dir = osp.join(REAL_ROOT_DIR, LANDMARKS_FOLDER)
    if not osp.isdir(real_dataset_landmarks_dir):
        raise NotADirectoryError(
            "Directory of real dataset features ({}) not found -- use `extract_features.py` to create it.".format(
                real_dataset_landmarks_dir))

    if VERBOSE:
        print(f"█═╦═[ Real dataset features root directory: {REAL_ROOT_DIR} ]")

    # Get real dataset image filenames
    with open(osp.join(real_dataset_landmarks_dir, 'image_filenames.txt')) as f:
        content_list = f.readlines()
    real_dataset_image_filenames = [x.strip() for x in content_list]

    if VERBOSE:
        print(f"  ╠═══[ Found {len(real_dataset_image_filenames)} real image filenames ]")

    # Load real landmarks
    real_landmarks_file = osp.join(real_dataset_landmarks_dir, 'landmarks.pt')
    real_landmarks = torch.load(real_landmarks_file)
    # Flatten landmarks to work with NN model
    flattened_real_landmarks = torch.flatten(real_landmarks, start_dim=1).numpy()

    if VERBOSE:
        print(f"  ╚═══[ Real landmarks shape: {flattened_real_landmarks.shape}")

    if VERBOSE:
        print(f"█═╦═[  Finding NNs for the following algorithms and metrics: ]")
        print(f"  ╠═══[ NN algorithms : {NN_ALGORITHMS} ]")
        print(f"  ╠═╦═[ NN metrics    : {NN_METRICS} ]")

    for nn_metric in NN_METRICS:
        for nn_algorithm in NN_ALGORITHMS:

            print(f"  ║ ╠═══[ metric: {nn_metric} - algorithm: {nn_algorithm} ]")

            if ((nn_metric == 'cosine') and (nn_algorithm == 'ball_tree')) or \
                    ((nn_metric == 'cosine') and (nn_algorithm == 'kd_tree')):
                print(f"  ║ ╠═══[! Invalid combination - Moving to next combination !]")
                continue


            #############################################################
            #                        Load fake dataset                  #
            #############################################################

            if not osp.isdir(FAKE_ROOT_DIR):
                raise NotADirectoryError

            if VERBOSE:
                print(f"  ║ ╠═══[ Fake dataset root directory: {FAKE_ROOT_DIR} ]")

            # # Get fake dataset image filenames
            with open(osp.join(FAKE_ROOT_DIR, 'valid_latent_code_hashes.txt')) as f:
                content_list = f.readlines()
            fake_dataset_image_filenames = [x.strip() for x in content_list]

            if VERBOSE:
                print(f"  ║ ╠═══[ Found {len(fake_dataset_image_filenames)} fake image filenames ]")

            # Fit NN models on fake data samples
            # Load fake landmarks
            fake_landmarks_file = osp.join(FAKE_ROOT_DIR, 'landmarks.pt')

            landmarks_nn_map_file = osp.join(FAKE_ROOT_DIR, f"landmarks_{nn_algorithm}_{nn_metric}_nn_map.json")

            fake_landmarks = torch.load(fake_landmarks_file)
            # Flatten to fit with NN model
            flattened_fake_landmarks = torch.flatten(fake_landmarks, start_dim=1).numpy()

            if VERBOSE:
                print(f"  ║ ╠═══[ Fake Landmarks Shape: {flattened_fake_landmarks.shape} ]")
                print(f"  ║ ╠═══[ Fiting NN model... ]")

            # Fit fake Data
            nn_model = NearestNeighbors(n_neighbors=1,
                                                algorithm=nn_algorithm,
                                                metric=nn_metric
                                                ).fit(flattened_fake_landmarks)
            
            if VERBOSE:
                print(f"  ║ ╠═══[ Finished fitting ]")
                print(f"  ║ ╠═══[ Pairing images with Nearest Neighbours ]")
            # Find NNs
            _, indices = nn_model.kneighbors(flattened_real_landmarks)
            if VERBOSE:
                print(f"  ║ ╠═══[ Found NNs ]")
                print(f"  ║ ╠═══[ Saving NNs to a mapping file... ]")

            # Build NN map dictionary
            nn_map = dict()
            for i in range(len(real_dataset_image_filenames)):
                nn_map.update({real_dataset_image_filenames[i]: fake_dataset_image_filenames[int(indices[i])]})

            # Save NN map
            with open(landmarks_nn_map_file, "w") as f:
                json.dump(nn_map, f)
                
            if VERBOSE:
                print(f"  ║ ╚═══[ NN mapping saved ]")


    if VERBOSE:
        print(f"  ╚═══[ All NN Mappings Completed ]")

if __name__ == '__main__':
    main()
