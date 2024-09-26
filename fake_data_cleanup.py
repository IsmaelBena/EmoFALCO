import os
import os.path as osp
import yaml
import tqdm

def main():
    # Load Config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    FAKE_ROOT_DIR = config['paths']['fake_dataset_root_dir']
    
    # Get fake dataset image filenames
    with open(osp.join(FAKE_ROOT_DIR, 'latent_code_hashes.txt')) as f:
        content_list = f.readlines()
    fake_dataset_image_filenames = [x.strip() for x in content_list]
    
    valid_hashes = []
    for hash_dir in fake_dataset_image_filenames:
        if osp.isfile(osp.join(FAKE_ROOT_DIR, hash_dir, 'landmarks.pt')):
            valid_hashes.append(hash_dir)
        else:
            print(f"{hash_dir} is invalid.")
    
    # Write latent codes hashes to file
    with open(osp.join(FAKE_ROOT_DIR, 'valid_latent_code_hashes.txt'), 'w') as f:
        for h in valid_hashes:
            f.write(f"{h}\n")
    
if __name__ == '__main__':
    main()