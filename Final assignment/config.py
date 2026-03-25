# DATA AUGMENTATION SETTINGS
APPLY_FOURIER = True  # Fast fourier  transform for data augmentation
FOURIER_ALPHA = 0.3    # blending factor
FOURIER_PROBABILITY = 0.3  # Probability of applying fft (having this less than 1 allows for more diverse training data. If its equal to 1 we focus on robustness completely)
APPLY_COPYPASTE = True  # Copy Paste augmentation set to true to apply or false to not apply
COPYPASTE_PROBABILITY = 0.5  # Probability of applying copy-paste (having this less than 1 allows for more diverse training data. If its equal to 1 we focus on robustness completely)

# MODEL ARCHITECTURE SETTINGS
ENCODER_NAME = "resnet50"  # alternatives could be: resnet50, resnet152, etc.
ENCODER_WEIGHTS = "imagenet"  # pretrained imagenet or none
IN_CHANNELS = 3  # RGB images
N_CLASSES = 19  # Cityscapes dataset classes
DECODER_CHANNELS = (256, 128, 64, 32, 16)  # U-Net decoder channels
DECODER_USE_BATCHNORM = True  # Use batch normalization in decoder
DECODER_ATTENTION_TYPE = None  # Attention mechanism could be: None, scse, or spatial

# TRAINING SETTINGS
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 10
SEED = 42

# PATHS & LOGGING
DATA_DIR = "./data/cityscapes"
CHECKPOINT_DIR = "./checkpoints"
EXPERIMENT_ID = "unet-training"
WANDB_PROJECT = "5lsm0-cityscapes-segmentation"

# LOSS & OPTIMIZATION
LOSS_WEIGHTS = {"cross_entropy": 0.3, "dice": 0.7}  # Weights for combined loss, dice increased for focus on robustness
LR_SCHEDULER = "cosine"  # Learning rate scheduler type
COSINE_ANNEALING_T_MAX = EPOCHS  # T_max for CosineAnnealingLR
COSINE_ANNEALING_ETA_MIN = 1e-6  # Minimum learning rate
