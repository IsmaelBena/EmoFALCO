import os
import os.path as osp
import argparse
import torch
from torch.utils import data
from torchvision import transforms
from models.psp import pSp
from lib import DATASETS, CelebAHQ, FaceAligner
from tqdm import tqdm
import yaml
import cv2
from PIL import Image
from models.load_generator import load_generator
from lib import tensor2image


def get_img_id(img_file):
    return osp.basename(img_file).split('.')[0]


def save_img(img_file, img):
    cv2.imwrite(img_file, cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))


def get_latents(net, x):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)

    return codes


def main():
    # Load Config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)


    VERBOSE = config['invert_settings']['verbose']
    USE_CUDA = config['invert_settings']['use_cuda']
    BATCH_SIZE = config['invert_settings']['batch_size']
    SAVE_ALIGNED = config['invert_settings']['save_aligned']
    SAVE_RECON = config['invert_settings']['save_reconstructed']
    
    REAL_ROOT_DIR = config['paths']['real_dataset_root_dir']
    INV_DIR = osp.join(REAL_ROOT_DIR, config['paths']['inverted_dir'])
    
    

    # LANDMARKS_INPUT = config['feature_extraction_settings']['landmarks_input']
    # LANDMARKS_DIR = osp.join(REAL_ROOT_DIR, config['paths']['facial_landmarks_dir'])
    # ARCFACE_DIR = osp.join(REAL_ROOT_DIR, config['paths']['arcface_dir'])

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                    [ CUDA ]                                                    ##
    ##                                                                                                                ##
    ####################################################################################################################
    use_cuda = False
    if torch.cuda.is_available():
        if USE_CUDA:
            use_cuda = True
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
    device = 'cuda' if use_cuda else 'cpu'

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                       [ Inverted Dataset Directory  ]                                          ##
    ##                                                                                                                ##
    ####################################################################################################################
    if VERBOSE:
        print(F"#. Create dir for storing the inverted {INV_DIR} dataset...")
        print("  \\__{}".format(INV_DIR))
    os.makedirs(INV_DIR, exist_ok=True)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                              [ Face Alignment ]                                                ##
    ##                                                                                                                ##
    ####################################################################################################################
    # Build landmark-based face aligner (required by e4e inversion)
    face_aligner = FaceAligner(device=device)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                          [ Pre-trained e4e Encoder ]                                           ##
    ##                                                                                                                ##
    ####################################################################################################################
    if VERBOSE:
        print("#. Build pre-trained e4e model...")

    e4e_checkpoint_path = osp.join('models', 'pretrained', 'e4e', 'e4e_ffhq_encode.pt')
    e4e_checkpoint = torch.load(e4e_checkpoint_path, map_location='cpu')
    e4e_opts = e4e_checkpoint['opts']
    e4e_opts['checkpoint_path'] = e4e_checkpoint_path
    e4e_opts['device'] = 'cuda' if use_cuda else 'cpu'
    e4e_opts = argparse.Namespace(**e4e_opts)
    e4e = pSp(e4e_opts)
    e4e.eval()
    e4e = e4e.to(device)

    e4e_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                         [ Pre-trained GAN Generator ]                                          ##
    ##                                                                                                                ##
    ####################################################################################################################

    # Build GAN generator model and load with pre-trained weights
    if VERBOSE:
        print("#. Build StyleGAN2 generator model G and load with pre-trained weights...")

    G = load_generator(model_name='stylegan2_ffhq1024', latent_is_w=True, verbose=VERBOSE).eval()

    # Upload GAN generator model to GPU
    if use_cuda:
        G = G.cuda()

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                [ Data Loader ]                                                 ##
    ##                                                                                                                ##
    ####################################################################################################################
    if VERBOSE:
        print("#. Loading dataset...")

    dataset = CelebAHQ(root_dir=REAL_ROOT_DIR, subset='train+val+test')
    dataloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create output directory to save images
    out_data_dir = osp.join(INV_DIR, 'data')
    os.makedirs(out_data_dir, exist_ok=True)

    # Create output directory to save the e4e latent codes
    out_codes_dir = osp.join(INV_DIR, 'latent_codes')
    os.makedirs(out_codes_dir, exist_ok=True)

    # Create files to store errors on alignment and face detection
    alignment_errors_file = osp.join(INV_DIR, 'alignment_errors.txt')
    with open(alignment_errors_file, 'w'):
        pass
    face_detection_errors_file = osp.join(INV_DIR, 'face_detection_errors.txt')
    with open(face_detection_errors_file, 'w'):
        pass

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                 [ Inversion ]                                                  ##
    ##                                                                                                                ##
    ####################################################################################################################
    for i_batch, data_batch in enumerate(
            tqdm(dataloader, desc="#. Processing images" if VERBOSE else '')):

        aligned_images = []
        for i in range(BATCH_SIZE):

            ############################################################################################################
            ##                                    [ Face alignment and crop ]                                         ##
            ############################################################################################################
            with torch.no_grad():
                img_aligned = face_aligner.align_face(image_file=data_batch[1][i],
                                                      alignment_errors_file=alignment_errors_file,
                                                      face_detection_errors_file=face_detection_errors_file)

            img_aligned_file = osp.join(out_data_dir, '{}_aligned.jpg'.format(get_img_id(data_batch[1][i])))
            save_img(img_file=img_aligned_file, img=img_aligned)

            # TODO: do not re-read the image from disk -- use `img_aligned` from above
            aligned_images.append(e4e_transforms(Image.open(img_aligned_file).convert('RGB')).unsqueeze(0))

        aligned_images = torch.cat(aligned_images).cuda()

        ################################################################################################################
        ##                                       [ Image Inversion (e4e) ]                                            ##
        ################################################################################################################
        image_latent_codes = get_latents(e4e, aligned_images)

        # Save latent codes
        for i in range(BATCH_SIZE):
            img_latent_code_file = osp.join(out_codes_dir, '{}.pt'.format(get_img_id(data_batch[1][i])))
            torch.save(image_latent_codes[i], img_latent_code_file)

        ################################################################################################################
        ##                                     [ Generate reconstructed images ]                                      ##
        ################################################################################################################
        with torch.no_grad():
            img_recon = G(image_latent_codes)

        ################################################################################################################
        ##                        [ Save reconstructed images (using inverted latent codes) ]                         ##
        ################################################################################################################
        if SAVE_RECON:
            for i in range(BATCH_SIZE):
                img_recon_file = osp.join(out_data_dir, '{}_recon.jpg'.format(get_img_id(data_batch[1][i])))
                tensor2image(img_recon[i].cpu(), adaptive=True).save(
                    img_recon_file, "JPEG", quality=90, subsampling=0, progressive=True)


if __name__ == '__main__':
    main()
