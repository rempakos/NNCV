# =============================================================================
# CONFIG — Tuned for robustness benchmark performance
# =============================================================================

# DATA AUGMENTATION SETTINGS
APPLY_FOURIER = True
FOURIER_ALPHA = 0.3
FOURIER_PROBABILITY = 0.3
APPLY_COPYPASTE = True
COPYPASTE_PROBABILITY = 0.4

# ── Novel augmentations ──────────────────────────────────────────────
# Frequency Band Dropout: randomly zero out annular rings in the
# Fourier spectrum so the model cannot over-rely on any single
# frequency range (texture, edges, broad color).
APPLY_FREQ_BAND_DROPOUT = True
FREQ_BAND_DROPOUT_PROBABILITY = 0.3
FREQ_BAND_DROPOUT_WIDTH = 0.15      # fraction of spectrum radius to zero
FREQ_BAND_DROPOUT_MAX_BANDS = 2     # how many rings to drop per image

# Semantic Region Style Swap: for each semantic class region in the
# image, independently replace its low-frequency Fourier "style" with
# that of the same class from a different training image.  Creates
# per-region style inconsistency that forces locally robust features.
APPLY_SEMANTIC_STYLE_SWAP = True
SEMANTIC_STYLE_SWAP_PROBABILITY = 0.2
SEMANTIC_STYLE_SWAP_BETA = 0.25     # how much donor style to blend in

# MODEL ARCHITECTURE SETTINGS
ENCODER_NAME = "resnet101"          # Deeper backbone → better features
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 3
N_CLASSES = 19                       # Cityscapes train-id classes
MODEL_ARCH = "DeepLabV3Plus"         # DeepLabV3+ > UNet for segmentation

# TRAINING SETTINGS
BATCH_SIZE = 8
EPOCHS = 150
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 10
SEED = 42

# EMA (Exponential Moving Average) — smooths weights for robustness
USE_EMA = True
EMA_DECAY = 0.999

# PATHS & LOGGING
DATA_DIR = "./data/cityscapes"
CHECKPOINT_DIR = "./checkpoints"
EXPERIMENT_ID = "deeplabv3plus-robust"
WANDB_PROJECT = "5lsm0-cityscapes-segmentation"

# LOSS & OPTIMIZATION
LOSS_WEIGHTS = {"cross_entropy": 0.5, "dice": 0.5}
LR_SCHEDULER = "onecycle"
COSINE_ANNEALING_ETA_MIN = 1e-6

# CITYSCAPES SUPER-CATEGORY GROUPINGS (train_ids)
# Used for per-category metric reporting
SUPERCATEGORY_MAP = {
    "flat":         [0, 1],          # road, sidewalk
    "construction": [2, 3, 4],       # building, wall, fence
    "object":       [5, 6, 7],       # pole, traffic light, traffic sign
    "nature":       [8, 9],          # vegetation, terrain
    "sky":          [10],            # sky
    "human":        [11, 12],        # person, rider
    "vehicle":      [13, 14, 15, 16, 17, 18],  # car, truck, bus, train, motorcycle, bicycle
}
