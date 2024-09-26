import argparse
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

def main():
    # Load Config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    

    VERBOSE = config['anonymize_settings']['verbose']
    USE_CUDA = config['anonymize_settings']['use_cuda']
    
    REAL_ROOT_DIR = config['paths']['real_dataset_root_dir']
    FAKE_ROOT_DIR = config['paths']['fake_dataset_root_dir']
    FAKE_NN_MAP_DIR = osp.join(FAKE_ROOT_DIR, config['paths']['fake_nn_map'])
    OUTPUT_DIR = config['paths']['anonymized_dir']
    
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
    
    STOP_EARLY = config['anonymize_settings']['stop_early']
    STOP_AT = config['anonymize_settings']['stop_at']
    
    ####################################################################################################################
    ##                                                                                                                ##
    ##                                      [ Anonymized Dataset Directory  ]                                         ##
    ##                                                                                                                ##
    ####################################################################################################################
    out_dir = anon_exp_dir(LATENT_SPACE, ID_LOSS_MARGIN, LAMBDA_ID, LAMBDA_LM, OPTIMIZER, LR, EPOCHS, FAKE_NN_MAP_DIR, LAYER_START, LAYER_END)
    if VERBOSE:
        print("#. Create dir for storing the anonymized dataset...")
        print("  \\__{}".format(out_dir))


    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                    [ CUDA ]                                                    ##
    ##                                                                                                                ##
    ####################################################################################################################
    use_cuda = False
    multi_gpu = False
    if torch.cuda.is_available():
        if USE_CUDA:
            use_cuda = True
            if torch.cuda.device_count() > 1:
                multi_gpu = True
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
    device = 'cuda' if use_cuda else 'cpu'

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                         [ Pre-trained GAN Generator ]                                          ##
    ##                                                                                                                ##
    ####################################################################################################################

    # Build GAN generator model and load with pre-trained weights
    if VERBOSE:
        print("#. Build StyleGAN2 generator model G and load with pre-trained weights...")

    G = load_generator(model_name='stylegan2_ffhq1024', latent_is_w=True, verbose=VERBOSE).eval().to(device)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                [ Data Loader ]                                                 ##
    ##                                                                                                                ##
    ####################################################################################################################
    if VERBOSE:
        print("#. Loading dataset...")

    ####################################################################################################################
    ##                                                 [ CelebA-HQ ]                                                  ##
    ####################################################################################################################
    
    dataset = CelebAHQ(root_dir=REAL_ROOT_DIR,
                        subset='train+val+test',
                        fake_nn_map=FAKE_NN_MAP_DIR,
                        inv=True,
                        filtered=True)
    dataloader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    # Create output directory to save images
    out_data_dir = osp.join(out_dir, 'data')
    os.makedirs(out_data_dir, exist_ok=True)

    # Create output directory to save latent codes
    out_code_dir = osp.join(out_dir, 'latent_codes')
    os.makedirs(out_code_dir, exist_ok=True)


    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                   [ Losses ]                                                   ##
    ##                                                                                                                ##
    ####################################################################################################################
    id_criterion = IDLoss(id_margin=ID_LOSS_MARGIN).eval().to(device)
    landmark_loss = LandmarkLoss().eval().to(device)

    # Parallelize GAN's generator G
    if multi_gpu:
        G = DataParallelPassthrough(G)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                               [ Anonymization ]                                                ##
    ##                                                                                                                ##
    ####################################################################################################################
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
            tqdm(dataloader, desc="#. Anonymize images" if VERBOSE else '')):
        
        # Get data
        img_orig = data_[0]
        img_orig_id = int(osp.basename(data_[1][0]).split('.')[0])
        img_nn_code = data_[3]
        img_recon_code = data_[5]

        # Build anonymization latent code
        latent_code = LatentCode(latent_code_real=img_recon_code, latent_code_fake_nn=img_nn_code, img_id=img_orig_id, 
                                 out_code_dir=out_code_dir, layer_start=LAYER_START, layer_end=LAYER_END, latent_space='W+')
        latent_code.to(device)

        # Count trainable parameters
        # latent_code_trainable_parameters = sum(p.numel() for p in latent_code.parameters() if p.requires_grad)
        # print("latent_code_trainable_parameters: {}".format(latent_code_trainable_parameters))

        # Check whether anonymization latent code has already been optimized -- if so, continue with the next one
        if not latent_code.do_optim():
            continue

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

        image_eval = {
            'image_name': img_orig_id,
            'epoch_losses': []
        }

        # Training (anonymization) loop for the current batch of images / latent codes
        for epoch in range(EPOCHS):
            # Clear gradients wrt parameters
            optimizer.zero_grad()

            # Generate anonymized image
            img_anon = G(latent_code())

            # Calculate identity and attribute preservation losses
            print(f"epoch: {epoch}")
            id_loss = id_criterion(img_anon.to(device), img_orig.to(device))
            lm_loss = landmark_loss(np.array(tensor2image(img_orig.cpu(), img_size=256)), np.array(tensor2image(img_anon.cpu(), img_size=256)))
            if lm_loss == None:
                print('Model could not detect a face in the generated image.')
                lm_loss = 100
            else:
                print('example loss:',lm_loss)
                
            # Calculate total loss
            loss = LAMBDA_ID * id_loss + LAMBDA_LM * lm_loss

            # Back-propagation
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if epoch in [0, 4, 24, 49]:
                image_eval['epoch_losses'].append({'epoch': epoch+1, 'loss': loss.item()})


        anon_eval['image_eval'].append(image_eval)

        # Store optimized anonymization latent codes
        latent_code.save()

        # Generate and save anonymized image
        with torch.no_grad():
            anonymized_image = G(latent_code())
        tensor2image(anonymized_image.cpu(), adaptive=True).save(osp.join(out_data_dir, '{}.jpg'.format(img_orig_id)),
                                                                 "JPEG", quality=75, subsampling=0, progressive=True)
        
        if STOP_EARLY and data_idx >= STOP_AT:
            break

    with open(osp.join(out_dir, 'eval.json'), 'w') as eval_file:
        json.dump(anon_eval, eval_file)

if __name__ == '__main__':
    main()
