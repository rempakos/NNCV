import numpy as np
import torch
import albumentations as A
import config
from torchvision.datasets import Cityscapes

_IGNORE_RAW_IDS = {cls.id for cls in Cityscapes.classes if cls.train_id == 255}


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# AUGMENTATIONS

train_transformation = A.Compose([
    # Spatial
    A.RandomScale(scale_limit=(-0.5, 1.0), interpolation=1, p=1.0),
    A.PadIfNeeded(
        min_height=config.INPUT_H, min_width=config.INPUT_W,
        border_mode=0, p=1.0,
    ),
    A.RandomCrop(height=config.INPUT_H, width=config.INPUT_W, p=1.0),
    A.HorizontalFlip(p=0.5),

    # Photometric
    A.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.2, p=0.3
    ),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
    A.RandomGamma(gamma_limit=(80, 120), p=0.2),

    # Corruption-style
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.ISONoise(
            color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0
        ),
    ], p=0.4),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MotionBlur(blur_limit=(3, 7), p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
    ], p=0.3),
    A.ImageCompression(quality_lower=40, quality_upper=95, p=0.2),

    # Weather-like
    A.OneOf([
        A.RandomFog(
            fog_coef_lower=0.1, fog_coef_upper=0.3,
            alpha_coef=0.1, p=1.0,
        ),
        A.RandomRain(
            slant_lower=-10, slant_upper=10,
            drop_length=10, drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=3, brightness_coefficient=0.9, p=1.0,
        ),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5), angle_lower=0.0,
            src_radius=150,
            num_flare_circles_lower=3,
            num_flare_circles_upper=5, p=1.0,
        ),
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1, num_shadows_upper=3,
            shadow_dimension=5, p=1.0,
        ),
    ], p=0.25),

    # Normalize (ImageNet stats, matching DINOv2 pretraining)
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

validation_transformation = A.Compose([
    A.Resize(height=config.INPUT_H, width=config.INPUT_W, interpolation=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


#@@@@@@@@@@@@@@@@@CUSTOM AUGMENTATIONS@@@@@@@@@@@@@@@@@@@@@

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#AUGMENTATION : COPY-PASTE OCCLUSION
def occlusion_copy_paste(image, mask, dataset, probability=0.5):
    """
    Paste a random crop from another training image on top of the
    current image+mask to simulate occlusions.
    """
    if np.random.rand() >= probability:
        return image, mask

    idx = np.random.randint(0, len(dataset))
    img2, mask2 = dataset[idx]
    img2, mask2 = np.array(img2), np.array(mask2)

    h1, w1 = image.shape[:2]
    h2, w2 = img2.shape[:2]
    crop_h = np.random.randint(64, min(h2, 256) + 1)
    crop_w = np.random.randint(64, min(w2, 256) + 1)

    y2 = np.random.randint(0, max(1, h2 - crop_h))
    x2 = np.random.randint(0, max(1, w2 - crop_w))
    y1 = np.random.randint(0, max(1, h1 - crop_h))
    x1 = np.random.randint(0, max(1, w1 - crop_w))

    image[y1:y1+crop_h, x1:x1+crop_w] = img2[y2:y2+crop_h, x2:x2+crop_w]
    mask[y1:y1+crop_h, x1:x1+crop_w] = mask2[y2:y2+crop_h, x2:x2+crop_w]
    return image, mask


def fast_fourier_transform(img1, img2, alpha=config.FOURIER_ALPHA):
    """
    Fourier-based style transfer: blend magnitude spectra of two images
    while keeping the phase (structure) of the source image intact.
    Inspired by "FDA: Fourier Domain Adaptation for Semantic Segmentation".
    """
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    fft1 = np.fft.fftn(img1, axes=(0, 1))
    fft2 = np.fft.fftn(img2, axes=(0, 1))

    blended_mag = alpha * np.abs(fft1) + (1 - alpha) * np.abs(fft2)
    fft_blended = blended_mag * np.exp(1j * np.angle(fft1))

    result = np.fft.ifftn(fft_blended, axes=(0, 1)).real
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#AUGMENTATION : FREQUENCY BAND DROPOUT

def frequency_band_dropout(image, band_width=0.15, max_bands=2):
    """
    Randomly zero out 1..max_bands annular rings in the 2D Fourier
    magnitude spectrum, then reconstruct the image. Phase is preserved,
    so spatial structure stays intact — only texture at that spatial
    scale disappears.

    Args:
        image:      uint8 (H, W, 3)
        band_width: width of each ring as fraction of spectrum radius
        max_bands:  max number of rings to drop per image

    Returns:
        uint8 (H, W, 3)
    """
    h, w = image.shape[:2]
    img_f = image.astype(np.float32) / 255.0

    # Radial distance map in normalised frequency space [0, 1]
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    max_r = np.sqrt(cy**2 + cx**2)
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2) / max_r

    n_bands = np.random.randint(1, max_bands + 1)
    dropout_mask = np.ones((h, w), dtype=np.float32)
    for _ in range(n_bands):
        centre = np.random.uniform(0.05, 0.95)
        half = band_width / 2.0
        ring = (dist >= centre - half) & (dist <= centre + half)
        dropout_mask[ring] = 0.0

    # Apply per channel in shifted Fourier space
    channels = []
    for c in range(3):
        fft = np.fft.fftshift(np.fft.fft2(img_f[:, :, c]))
        fft *= dropout_mask
        channels.append(np.fft.ifft2(np.fft.ifftshift(fft)).real)

    result = np.stack(channels, axis=-1)
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#AUGMENTATION : SEMANTIC REGION STYLE SWAP

def semantic_region_style_swap(image, mask, dataset, beta=0.25):
    """
    For each semantic class present in the mask, sample a donor image,
    extract the colour statistics (mean, std) of that class's region
    in the donor, and blend them into the source via histogram matching.

    Args:
        image:   uint8 (H, W, 3)
        mask:    int (H, W) — semantic label IDs
        dataset: raw Cityscapes dataset to sample donors from
        beta:    blending strength (0 = no effect, 1 = full transfer)

    Returns:
        uint8 (H, W, 3)
    """
    result = image.copy().astype(np.float32)
    unique_classes = np.unique(mask)

    for cls_id in unique_classes:
        if cls_id in _IGNORE_RAW_IDS:
            continue

        region = mask == cls_id
        if region.sum() < 500:
            continue

        # Sample a donor
        donor_idx = np.random.randint(0, len(dataset))
        donor_img, donor_target = dataset[donor_idx]
        donor_img = np.array(donor_img, dtype=np.float32)
        donor_mask = np.array(donor_target)

        donor_region = donor_mask == cls_id
        if donor_region.sum() < 100:
            continue

        src_pixels = image[region].astype(np.float32)
        don_pixels = donor_img[donor_region]

        src_mean, src_std = src_pixels.mean(0), src_pixels.std(0) + 1e-6
        don_mean, don_std = don_pixels.mean(0), don_pixels.std(0) + 1e-6

        normalised = (src_pixels - src_mean) / src_std
        transferred = normalised * don_std + don_mean
        blended = (1 - beta) * src_pixels + beta * transferred

        result[region] = blended

    return np.clip(result, 0, 255).astype(np.uint8)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#DATASET WRAPPER

class CityscapeAlbumentations(torch.utils.data.Dataset):
    """
    Wraps a Cityscapes dataset and applies the full augmentation pipeline:
    copy-paste to Fourier blending to Freq Band Dropout to Semantic Style
    Swap to albumentations (spatial + photometric + corruption).
    """

    def __init__(
        self, dataset, transform=None,
        apply_fourier=False, apply_copypaste=False,
        apply_freq_band_dropout=False,
        apply_semantic_style_swap=False,
    ):
        self.dataset = dataset
        self.transform = transform
        self.apply_fourier = apply_fourier
        self.apply_copypaste = apply_copypaste
        self.apply_freq_band_dropout = apply_freq_band_dropout
        self.apply_semantic_style_swap = apply_semantic_style_swap

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        image = np.array(image)
        mask = np.array(target)

        # Occlusion copy-paste
        if self.apply_copypaste:
            image, mask = occlusion_copy_paste(
                image, mask, self.dataset,
                probability=config.COPYPASTE_PROBABILITY,
            )

        # Fourier magnitude blending (global style transfer)
        if self.apply_fourier and np.random.rand() < config.FOURIER_PROBABILITY:
            j = np.random.randint(0, len(self.dataset))
            img2, _ = self.dataset[j]
            image = fast_fourier_transform(
                image, np.array(img2), alpha=config.FOURIER_ALPHA
            )

        # Frequency Band Dropout
        if (self.apply_freq_band_dropout
                and np.random.rand() < config.FREQ_BAND_DROPOUT_PROBABILITY):
            image = frequency_band_dropout(
                image,
                band_width=config.FREQ_BAND_DROPOUT_WIDTH,
                max_bands=config.FREQ_BAND_DROPOUT_MAX_BANDS,
            )

        # Semantic Region Style Swap
        if (self.apply_semantic_style_swap
                and np.random.rand() < config.SEMANTIC_STYLE_SWAP_PROBABILITY):
            image = semantic_region_style_swap(
                image, mask, self.dataset,
                beta=config.SEMANTIC_STYLE_SWAP_BETA,
            )

        # Albumentations pipeline
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0).long()

        return image, mask
