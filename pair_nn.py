import os.path as osp
import argparse
from sklearn.neighbors import NearestNeighbors
import torch
import json
import yaml
from lib import DATASETS


def check_nn_map_file(nn_map_file, real_dataset_image_filenames):
    """Check if existing nn map file contains the same real images' filenames.

    Args:
        nn_map_file (str): nn map file
        real_dataset_image_filenames (list): list of real images' filenames

    Returns:
        exists (bool): whether the nn map file exists and contains the correct real images' filenames
    """
    exists = osp.exists(nn_map_file)
    if not exists:
        return exists

    with open(nn_map_file) as f:
        nn_map = json.load(f)
    exists = set(nn_map.keys()) == set(real_dataset_image_filenames)

    return exists


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



    ####################################################################################################################
    ##                                                                                                                ##
    ##                                            [ Real Dataset Features ]                                           ##
    ##                                                                                                                ##
    ####################################################################################################################
    real_dataset_landmarks_dir = osp.join(REAL_ROOT_DIR, LANDMARKS_FOLDER)
    if not osp.isdir(real_dataset_landmarks_dir):
        raise NotADirectoryError(
            "Directory of real dataset features ({}) not found -- use `extract_features.py` to create it.".format(
                real_dataset_landmarks_dir))

    if VERBOSE:
        print(f"#. Real dataset features root directory: {REAL_ROOT_DIR}")

    # Get real dataset image filenames
    with open(osp.join(real_dataset_landmarks_dir, 'image_filenames.txt')) as f:
        content_list = f.readlines()
    real_dataset_image_filenames = [x.strip() for x in content_list]

    if VERBOSE:
        print("  \\__real_dataset_image_filenames: {}".format(len(real_dataset_image_filenames)))

    # === Landmarks ===
    real_landmarks_file = osp.join(real_dataset_landmarks_dir, 'landmarks.pt')
    real_landmarks = torch.load(real_landmarks_file)
    flattened_real_landmarks = torch.flatten(real_landmarks, start_dim=1).numpy()

    if VERBOSE:
        print("  \\__Real Landmarks Shape: {}".format(flattened_real_landmarks.shape))

    if VERBOSE:
        print("#. Finding NNs for the following algorithms and metrics:")
        print("  \\__NN algorithms : {}".format(NN_ALGORITHMS))
        print("  \\__NN metrics    : {}".format(NN_METRICS))
        print("#. Process...")

    for nn_metric in NN_METRICS:
        for nn_algorithm in NN_ALGORITHMS:

            print("  \\__.(metric, algorithm) = ({}, {})".format(nn_metric, nn_algorithm))

            if ((nn_metric == 'cosine') and (nn_algorithm == 'ball_tree')) or \
                    ((nn_metric == 'cosine') and (nn_algorithm == 'kd_tree')):
                print("      \\__.Invalid combination -- Abort!")
                continue

            ############################################################################################################
            ##                                                                                                        ##
            ##                                       [ Fake Dataset Features ]                                        ##
            ##                                                                                                        ##
            ############################################################################################################
            if not osp.isdir(FAKE_ROOT_DIR):
                raise NotADirectoryError

            if VERBOSE:
                print("      \\__.Fake dataset root directory: {}".format(FAKE_ROOT_DIR))

            # # Get fake dataset image filenames
            with open(osp.join(FAKE_ROOT_DIR, 'valid_latent_code_hashes.txt')) as f:
                content_list = f.readlines()
            fake_dataset_image_filenames = [x.strip() for x in content_list]

            if VERBOSE:
                print("          \\__fake_dataset_image_filenames: {}".format(len(fake_dataset_image_filenames)))

            # Fit NN models on fake data samples
            # === Landmarks ===
            fake_landmarks_file = osp.join(FAKE_ROOT_DIR, 'landmarks.pt')

            landmarks_nn_map_file = osp.join(FAKE_ROOT_DIR, f"landmarks_{nn_algorithm}_{nn_metric}_nn_map.json")

            fake_landmarks = torch.load(fake_landmarks_file)
            print(fake_landmarks)
            flattened_fake_landmarks = torch.flatten(fake_landmarks, start_dim=1).numpy()

            if VERBOSE:
                print("          \\__Fake Landmarks Shape: {}".format(flattened_fake_landmarks.shape))
                print("          \\__Fit NN model...", end="")

            nn_model = NearestNeighbors(n_neighbors=1,
                                                algorithm=nn_algorithm,
                                                metric=nn_metric
                                                ).fit(flattened_fake_landmarks)
            if VERBOSE:
                print("Done!")
                print("          \\__.Find NNs...")

            # === Landmarrks ===
            if VERBOSE:
                print("              \\__Landmarks...", end="")
            _, indices = nn_model.kneighbors(flattened_real_landmarks)
            if VERBOSE:
                print("Done!")

            # Build NN map dictionary
            nn_map = dict()
            for i in range(len(real_dataset_image_filenames)):
                nn_map.update({real_dataset_image_filenames[i]: fake_dataset_image_filenames[int(indices[i])]})

            # Save nn map
            with open(landmarks_nn_map_file, "w") as f:
                json.dump(nn_map, f)


if __name__ == '__main__':
    main()
