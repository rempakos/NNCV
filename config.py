#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# CONFIG — DINOv2 + Linear Decoder for Robust Cityscapes Segmentation
BACKBONE = "vit_base_patch14_dinov2.lvd142m"
PATCH_SIZE = 14
EMBED_DIM = 768
N_CLASSES = 19

# Input must be divisible by patch_size=14
#Dinov2 needs both dimensions to be able to be divided by the patch size (14)
#We could go for higher dimension than 518x1036 but that would require much greater compute for our poor A100
# 518 = 37*14, 1036 = 74*14
INPUT_H = 518
INPUT_W = 1036

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#training hyperparameters
BATCH_SIZE = 4
EPOCHS = 80
LEARNING_RATE = 1e-4         # for the segmentation head
BACKBONE_LR_FACTOR = 0.1    # backbone LR = LEARNING_RATE * this factor
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 10
SEED = 42

# EMA
USE_EMA = True
EMA_DECAY = 0.999

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#Augmentation parameters
# Standard augmentations
APPLY_FOURIER = True
FOURIER_ALPHA = 0.3
FOURIER_PROBABILITY = 0.3
APPLY_COPYPASTE = True
COPYPASTE_PROBABILITY = 0.4

# Randomly zeroes annular rings in the Fourier spectrum to prevent
# the model from over-relying on any single spatial frequency range.
APPLY_FREQ_BAND_DROPOUT = True
FREQ_BAND_DROPOUT_PROBABILITY = 0.3
FREQ_BAND_DROPOUT_WIDTH = 0.15
FREQ_BAND_DROPOUT_MAX_BANDS = 2

# Per-class histogram transfer from random donor images, creating
# spatially-varying style inconsistency that global augmentations miss.
APPLY_SEMANTIC_STYLE_SWAP = True
SEMANTIC_STYLE_SWAP_PROBABILITY = 0.2
SEMANTIC_STYLE_SWAP_BETA = 0.25

#Mixed loss weight for cross-entropy and dice loss 
LOSS_WEIGHTS = {"cross_entropy": 0.5, "dice": 0.5}

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#Paths
DATA_DIR = "./data/cityscapes"
CHECKPOINT_DIR = "./checkpoints"
EXPERIMENT_ID = "dinov2-linear-robust"
WANDB_PROJECT = "5lsm0-cityscapes-segmentation"

# Cityscape class groups for supercategory-aware augmentations
SUPERCATEGORY_MAP = {
    "flat":         [0, 1],
    "construction": [2, 3, 4],
    "object":       [5, 6, 7],
    "nature":       [8, 9],
    "sky":          [10],
    "human":        [11, 12],
    "vehicle":      [13, 14, 15, 16, 17, 18],
}
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@