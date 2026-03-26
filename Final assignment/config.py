# =============================================================================
# CONFIG — DINOv2 + Linear Decoder for Robust Cityscapes Segmentation
# =============================================================================

# ── Model ─────────────────────────────────────────────────────────────
# DINOv2 ViT-B/14 backbone with a simple linear segmentation head.
# The BRAVO Challenge 2024 winner showed this outperforms complex
# decoders by a large margin on robustness benchmarks.
BACKBONE = "vit_base_patch14_dinov2.lvd142m"   # timm model name
PATCH_SIZE = 14
EMBED_DIM = 768              # ViT-B embedding dimension
N_CLASSES = 19               # Cityscapes train-id classes
IN_CHANNELS = 3

# Input must be divisible by patch_size=14
# 518 = 37*14, 1036 = 74*14  (closest to 512x1024)
INPUT_H = 518
INPUT_W = 1036

# ── Training ──────────────────────────────────────────────────────────
BATCH_SIZE = 4               # ViT-B is heavier than ResNet; 4 is safe on A100
EPOCHS = 80
LEARNING_RATE = 1e-4         # for the segmentation head
BACKBONE_LR_FACTOR = 0.1    # backbone LR = LEARNING_RATE * this factor
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 10
SEED = 42

# EMA
USE_EMA = True
EMA_DECAY = 0.999

# ── Data Augmentation ─────────────────────────────────────────────────
# Standard augmentations
APPLY_FOURIER = True
FOURIER_ALPHA = 0.3
FOURIER_PROBABILITY = 0.3
APPLY_COPYPASTE = True
COPYPASTE_PROBABILITY = 0.4

# Novel augmentation 1: Frequency Band Dropout
# Randomly zeroes annular rings in the Fourier spectrum to prevent
# the model from over-relying on any single spatial frequency range.
APPLY_FREQ_BAND_DROPOUT = True
FREQ_BAND_DROPOUT_PROBABILITY = 0.3
FREQ_BAND_DROPOUT_WIDTH = 0.15
FREQ_BAND_DROPOUT_MAX_BANDS = 2

# Novel augmentation 2: Semantic Region Style Swap
# Per-class histogram transfer from random donor images, creating
# spatially-varying style inconsistency that global augmentations miss.
APPLY_SEMANTIC_STYLE_SWAP = True
SEMANTIC_STYLE_SWAP_PROBABILITY = 0.2
SEMANTIC_STYLE_SWAP_BETA = 0.25

# ── Loss ──────────────────────────────────────────────────────────────
LOSS_WEIGHTS = {"cross_entropy": 0.5, "dice": 0.5}

# ── Paths & Logging ──────────────────────────────────────────────────
DATA_DIR = "./data/cityscapes"
CHECKPOINT_DIR = "./checkpoints"
EXPERIMENT_ID = "dinov2-linear-robust"
WANDB_PROJECT = "5lsm0-cityscapes-segmentation"

# ── Cityscapes supercategory groupings (train_ids) ────────────────────
SUPERCATEGORY_MAP = {
    "flat":         [0, 1],
    "construction": [2, 3, 4],
    "object":       [5, 6, 7],
    "nature":       [8, 9],
    "sky":          [10],
    "human":        [11, 12],
    "vehicle":      [13, 14, 15, 16, 17, 18],
}
