import os
import torch
DEVICE = torch.device("cuda:0")
BATCH_SIZE = 32
LEARNING_RATE = 6e-5
DROPOUT = 0.5
MODEL_NAME = 'google/electra-base-discriminator'
EPOCHS = 150
BEST_EPOCH = 10
ACOUSTIC_DIM = 74
TEXT_DIM = 768
VISUAL_DIM = 0
ALIGNED = True
DATASETS = 'mosi'

if ALIGNED:
    if DATASETS == 'mosi':
        VISUAL_DIM = 47
    if DATASETS == 'mosei':
        VISUAL_DIM = 35
else:
    if DATASETS == 'mosi_new_noalign':
        VISUAL_DIM = 20
        ACOUSTIC_DIM = 5
    if DATASETS == 'mosei_new_noalign':
        VISUAL_DIM = 35

ALPHA = 10
BETA = -0.01
GAMA = -0.01
BN = 50
# case study setting
SAVE_EPOCH = 22
BL=11
alpha = 0.2
XLNET_INJECTION_INDEX = 1
alpha_noise = 0.001
alpha_adv = 0.01
alpha_js = 0.01

PRETRAIN_PATH = './data/CMU/pretrain/'
DATASETS_PATH = './data/CMU/datasets/'
PROJ = 'DARSM'

alpha_mi = 0.1
alpha_recon = 1
alpha_grl = 0.01

LATENT_DIM = 768
HIDDEN_STATE_DIM = 768


alpha_kld_s = 0.01
alpha_kld_pt = 0.01
alpha_kld_pv = 0.01
alpha_kld_pa = 0.01

alpha_t_recon = 0.1
alpha_v_recon = 0.1
alpha_a_recon = 0.1

alpha_space = 0.2
