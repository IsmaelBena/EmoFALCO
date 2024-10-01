import torch
from torch.utils import data
from torch.optim.lr_scheduler import MultiStepLR
import os
import os.path as osp
from lib import DATASETS, CelebAHQ, DataParallelPassthrough, IDLoss, LandmarkLoss, LatentCode, tensor2image, anon_exp_dir
from models.load_generator import load_generator
from tqdm import tqdm
import json
import yaml
import numpy as np

"""
Original file from: https://github.com/chi0tzp/FALCO,
Major modifications made for the EmoFalco Dissertation project.
"""

def main():
    # Load Config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    #############################################################
    #                   Global Variables                        #
    #############################################################

    VERBOSE = config['anonymize_settings']['verbose']
    USE_CUDA = config['anonymize_settings']['use_cuda']
    
    # Directories
    REAL_ROOT_DIR = config['paths']['real_dataset_root_dir']
    FAKE_ROOT_DIR = config['paths']['fake_dataset_root_dir']
    FAKE_NN_MAP_DIR = osp.join(FAKE_ROOT_DIR, config['paths']['fake_nn_map'])
    OUTPUT_DIR = config['paths']['anonymized_dir']
    
    # Learning Related Settings
    LATENT_SPACE = config['anonymize_settings']['latent_space']
    ID_LOSS_MARGIN = config['anonymize_settings']['id_loss_margin']
    EPOCHS = config['anonymize_settings']['epochs']
    OPTIMIZER = config['anonymize_settings']['optimizer']
    LR = config['anonymize_settings']['lr']
    LR_MILESTONES = config['anonymize_settings']['lr_milestones']
    LR_GAMMA = config['anonymize_settings']['lr_gamma']
    LAMBDA_ID = config['anonymize_settings']['lambda_id']
    LAMBDA_LM = config['anonymize_settings']['lambda_lm']
    LAYER_START = config['anonymize_settings']['layer_start']
    LAYER_END = config['anonymize_settings']['layer_end']
    
    # Stop the model early if anonymizing the full dataset takes too long
    STOP_EARLY = config['anonymize_settings']['stop_early']
    STOP_AT = config['anonymize_settings']['stop_at']
    
    #############################################################   
    #                   Configure Output directory              #
    #############################################################
    
    # set the name of the output directory according to the parameters used
    if VERBOSE:
        print("█═╦═[ Creating directory for storing the anonymized dataset... ]")
    out_dir = anon_exp_dir(LATENT_SPACE, ID_LOSS_MARGIN, LAMBDA_ID, LAMBDA_LM, OPTIMIZER, LR, EPOCHS, FAKE_NN_MAP_DIR, LAYER_START, LAYER_END)
    
    # Create subdirectory to save images
    out_data_dir = osp.join(out_dir, 'data')
    os.makedirs(out_data_dir, exist_ok=True)

    # Create subdirectory to save latent codes
    out_code_dir = osp.join(out_dir, 'latent_codes')
    os.makedirs(out_code_dir, exist_ok=True)
    
    if VERBOSE:
        print(f"  ╚═══[ Directory created at: {out_dir} ]")


    #############################################################
    #                        CUDA                               #
    #############################################################
    
    use_cuda = False
    multi_gpu = False
    if torch.cuda.is_available():
        if USE_CUDA:
            use_cuda = True
            if torch.cuda.device_count() > 1:
                multi_gpu = True
        else:
            print("█═══[! Cuda is avaialable but has not been selected in the configs file !]")
    device = 'cuda' if use_cuda else 'cpu'
    
    if VERBOSE:
        print(f"█═══[ Device is set to: {device} ]")

    
    #############################################################
    #                        GAN Model                          #
    #############################################################
    
    # Build GAN generator model and load with pre-trained weights
    if VERBOSE:
        print("█═╦═[ Building pretrained StyleGAN2 generator model... ]")

    G = load_generator(model_name='stylegan2_ffhq1024', latent_is_w=True, verbose=VERBOSE).eval().to(device)

    if VERBOSE:
        print("  ╚═══[ Model successfully built ]")
        
    #############################################################
    #                        Load Dataset                       #
    #############################################################
    
    if VERBOSE:
        print("█═╦═[ Loading dataset... ]")

    """ 
    Load dataset 
    set "filtered" to true in order to only retrieve valid data 
    i.e the data that made it through all the other seections of the model so that there
    is a valid landmark and nearest neighbour mapping for each entry loaded    
    """
    
    dataset = CelebAHQ(root_dir=REAL_ROOT_DIR,
                        subset='train+val+test',
                        fake_nn_map=FAKE_NN_MAP_DIR,
                        inv=True,
                        filtered=True)
    dataloader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    if VERBOSE:
        print("  ╚═══[ Dataset ready ]")

    #############################################################
    #                        Loss Functions                     #
    #############################################################
    
    # Load the classes that calculation loss values for each loss metric
    id_criterion = IDLoss(id_margin=ID_LOSS_MARGIN).eval().to(device)
    landmark_loss = LandmarkLoss().eval().to(device)

    # Parallelize GAN's generator G
    if multi_gpu:
        G = DataParallelPassthrough(G)

    #############################################################
    #                        Anonymization                      #
    #############################################################
    
    # Create the dictionary to store relevant information for late graphing and comparisons
    anon_eval = {
        'latent_sppace': LATENT_SPACE,
        'id_loss_margin': ID_LOSS_MARGIN,
        'lr': LR,
        'lambda_id': LAMBDA_ID,
        'lambda_lm': LAMBDA_LM,
        'optimizer': OPTIMIZER,
        'image_eval': []
    }
    
    for data_idx, data_ in enumerate(
            tqdm(dataloader, desc="█═══[ Anonymizing images ]" if VERBOSE else '')):
        
        # Get required data
        img_orig = data_[0]
        img_orig_id = int(osp.basename(data_[1][0]).split('.')[0])
        img_nn_code = data_[3]
        img_recon_code = data_[5]

        # Build anonymization latent code
        latent_code = LatentCode(latent_code_real=img_recon_code, latent_code_fake_nn=img_nn_code, img_id=img_orig_id, 
                                 out_code_dir=out_code_dir, layer_start=LAYER_START, layer_end=LAYER_END, latent_space='W+')
        latent_code.to(device)

        # Build optimizer
        optimizer = None
        if OPTIMIZER == 'sgd':
            optimizer = torch.optim.SGD(params=latent_code.parameters(), lr=LR)
        elif OPTIMIZER == 'adam':
            optimizer = torch.optim.Adam(params=latent_code.parameters(), lr=LR)

        # Set learning rate scheduler
        lr_scheduler = MultiStepLR(optimizer=optimizer,
                                   milestones=[int(m * EPOCHS) for m in LR_MILESTONES],
                                   gamma=LR_GAMMA)

        # Zero out gradients
        G.zero_grad()
        id_criterion.zero_grad()
        landmark_loss.zero_grad()

        # Create dictionary to store loss values for current image
        image_eval = {
            'image_name': img_orig_id,
            'combined_losses': [],
            'id_loss': [],
            'lm_loss': []
        }

        # Optimize the current image (Training loop)
        for epoch in range(EPOCHS):
            # Clear gradients wrt parameters
            optimizer.zero_grad()

            # Generate anonymized image
            img_anon = G(latent_code())

            # Calculate identity and landmarks preservation losses
            id_loss = id_criterion(img_anon.to(device), img_orig.to(device))
            lm_loss = landmark_loss(np.array(tensor2image(img_orig.cpu(), img_size=256)), np.array(tensor2image(img_anon.cpu(), img_size=256)))
            # Incase of a disfigured image wherre the face cannot be detected, set high loss in an attempt to "fix" training direction
            if lm_loss == None:
                print('Model could not detect a face in the generated image.')
                lm_loss = 100

                
            # Calculate total loss
            loss = LAMBDA_ID * id_loss + LAMBDA_LM * lm_loss

            # Back-propagation
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # If the image is at one of the checkpoints, store the current loss values for evaluation
            if epoch in [0, 4, 24, 49]:
                image_eval['combined_losses'].append({'epoch': epoch+1, 'loss': loss.item()})
                image_eval['id_loss'].append({'epoch': epoch+1, 'loss': id_loss.item()})
                image_eval['lm_loss'].append({'epoch': epoch+1, 'loss': lm_loss.item()})

        # Append image losses
        anon_eval['image_eval'].append(image_eval)

        # Store optimized anonymization latent codes
        latent_code.save()

        # Generate and save anonymized image
        with torch.no_grad():
            anonymized_image = G(latent_code())
        tensor2image(anonymized_image.cpu(), adaptive=True).save(osp.join(out_data_dir, '{}.jpg'.format(img_orig_id)),
                                                                 "JPEG", quality=75, subsampling=0, progressive=True)
        
        # Stop early if configured
        if STOP_EARLY and data_idx >= STOP_AT:
            break

    # Store the evaluation metrics to be visualised later
    with open(osp.join(out_dir, 'eval.json'), 'w') as eval_file:
        json.dump(anon_eval, eval_file)

if __name__ == '__main__':
    main()
