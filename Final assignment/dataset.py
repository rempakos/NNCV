"""
Dataset wrapper with robustness-oriented augmentations.
Includes Fourier blending, occlusion copy-paste, and corruption-simulating transforms.
"""

import numpy as np
import torch
import albumentations as A
import config


# ═══════════════════════════════════════════════════════════════════
# AUGMENTATION PIPELINES
# ═══════════════════════════════════════════════════════════════════

train_transformation = A.Compose([
    # ── Spatial ────────────────────────────────────────────────
    A.RandomScale(scale_limit=(-0.5, 1.0), interpolation=1, p=1.0),
    A.PadIfNeeded(min_height=512, min_width=1024, border_mode=0, p=1.0),
    A.RandomCrop(height=512, width=1024, p=1.0),
    A.HorizontalFlip(p=0.5),

    # ── Photometric (simulate lighting / camera changes) ──────
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
    A.RandomGamma(gamma_limit=(80, 120), p=0.2),

    # ── Corruption-style (noise, blur, compression) ───────────
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
    ], p=0.4),

    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MotionBlur(blur_limit=(3, 7), p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
    ], p=0.3),

    A.ImageCompression(quality_lower=40, quality_upper=95, p=0.2),

    # ── Weather-like effects ──────────────────────────────────
    A.OneOf([
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=1.0),
        A.RandomRain(slant_lower=-10, slant_upper=10,
                     drop_length=10, drop_width=1,
                     drop_color=(200, 200, 200),
                     blur_value=3, brightness_coefficient=0.9, p=1.0),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5),
                         angle_lower=0.0,
                         src_radius=150,
                         num_flare_circles_lower=3,
                         num_flare_circles_upper=5, p=1.0),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1),
                       num_shadows_lower=1, num_shadows_upper=3,
                       shadow_dimension=5, p=1.0),
    ], p=0.25),

    # ── Normalize last ────────────────────────────────────────
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

validation_transformation = A.Compose([
    A.Resize(height=512, width=1024, interpolation=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


# ═══════════════════════════════════════════════════════════════════
# CUSTOM AUGMENTATIONS
# ═══════════════════════════════════════════════════════════════════

def occlusion_copy_paste(image, mask, dataset, probability=0.5):
    """
    Paste a random crop from another training image on top of the
    current image+mask.  Simulates occlusions.
    """
    if np.random.rand() >= probability:
        return image, mask

    idx = np.random.randint(0, len(dataset))
    img2, mask2 = dataset[idx]
    img2 = np.array(img2)
    mask2 = np.array(mask2)

    h1, w1 = image.shape[:2]
    h2, w2 = img2.shape[:2]

    crop_h = np.random.randint(64, min(h2, 256) + 1)
    crop_w = np.random.randint(64, min(w2, 256) + 1)

    y2 = np.random.randint(0, max(1, h2 - crop_h))
    x2 = np.random.randint(0, max(1, w2 - crop_w))
    crop_img = img2[y2:y2 + crop_h, x2:x2 + crop_w]
    crop_mask = mask2[y2:y2 + crop_h, x2:x2 + crop_w]

    y1 = np.random.randint(0, max(1, h1 - crop_h))
    x1 = np.random.randint(0, max(1, w1 - crop_w))
    image[y1:y1 + crop_h, x1:x1 + crop_w] = crop_img
    mask[y1:y1 + crop_h, x1:x1 + crop_w] = crop_mask

    return image, mask


def fast_fourier_transform(img1, img2, alpha=config.FOURIER_ALPHA):
    """
    Fourier-based style transfer: blend the magnitude spectra of two
    images while keeping the phase of the source image intact.
    Encourages robustness to style / domain shifts.
    """
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    fft1 = np.fft.fftn(img1, axes=(0, 1))
    fft2 = np.fft.fftn(img2, axes=(0, 1))

    magnitude1 = np.abs(fft1)
    phase1 = np.angle(fft1)
    magnitude2 = np.abs(fft2)

    blended = alpha * magnitude1 + (1 - alpha) * magnitude2
    fft_blended = blended * np.exp(1j * phase1)

    result = np.fft.ifftn(fft_blended, axes=(0, 1)).real
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return result


# ═══════════════════════════════════════════════════════════════════
# NOVEL AUGMENTATION 1: FREQUENCY BAND DROPOUT
# ═══════════════════════════════════════════════════════════════════
#
# Motivation:
#   CNNs tend to latch onto specific frequency ranges — e.g. relying
#   heavily on texture (mid-high frequency) or on broad color blobs
#   (low frequency).  Corruption benchmarks test exactly this: blur
#   kills high frequencies, noise kills low ones, compression kills
#   mid-range detail.
#
#   Existing augmentations (blur, noise, etc.) each target ONE end of
#   the spectrum.  Frequency Band Dropout is *spectrum-agnostic*: it
#   randomly removes arbitrary annular rings from the 2D Fourier
#   magnitude, forcing the network to build features that are
#   redundant across all frequency ranges.
#
#   If it has no effect, the image is just mildly filtered — strictly
#   no worse than a random blur or sharpen.
#
def frequency_band_dropout(image, band_width=0.15, max_bands=2):
    """
    Randomly zero out 1..max_bands annular rings in the Fourier
    magnitude spectrum, then reconstruct the image.  Phase is
    preserved, so spatial structure stays intact — only the
    "texture at that spatial scale" disappears.

    Args:
        image:      uint8 (H, W, 3)
        band_width: width of each annular ring as a fraction of the
                    maximum frequency radius (0.15 ≈ 15 % of the spectrum)
        max_bands:  randomly drop 1..max_bands rings

    Returns:
        uint8 (H, W, 3)
    """
    h, w = image.shape[:2]
    img_f = image.astype(np.float32) / 255.0

    # Build a radial distance map in normalised frequency space [0, 1]
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    max_r = np.sqrt(cy ** 2 + cx ** 2)
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2) / max_r  # (H, W) in [0,1]

    # Decide how many bands to drop
    n_bands = np.random.randint(1, max_bands + 1)

    # Create the dropout mask (1 = keep, 0 = drop)
    mask = np.ones((h, w), dtype=np.float32)
    for _ in range(n_bands):
        # Random centre of the annular ring somewhere in [0, 1]
        centre = np.random.uniform(0.05, 0.95)
        half_w = band_width / 2.0
        ring = (dist >= centre - half_w) & (dist <= centre + half_w)
        mask[ring] = 0.0

    # Apply per channel in shifted Fourier space
    result_channels = []
    for c in range(3):
        fft = np.fft.fftshift(np.fft.fft2(img_f[:, :, c]))
        fft *= mask  # zero out the selected rings
        ch = np.fft.ifft2(np.fft.ifftshift(fft)).real
        result_channels.append(ch)

    result = np.stack(result_channels, axis=-1)
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return result


# ═══════════════════════════════════════════════════════════════════
# NOVEL AUGMENTATION 2: SEMANTIC REGION STYLE SWAP
# ═══════════════════════════════════════════════════════════════════
#
# Motivation:
#   Standard Fourier style transfer (FDA, and our own FFT above)
#   swaps the global style of the entire image.  But in the real
#   world, style inconsistency is *local*: a shadowed road next to a
#   sunlit building, fog on the road but clear sky above.  Robustness
#   benchmarks test exactly these spatially-varying corruptions.
#
#   This augmentation operates *per semantic region*.  For each class
#   present in the image, it independently grabs a donor image,
#   extracts the low-frequency Fourier "style" of that class's pixels,
#   and blends it into the source — but only within that class's mask.
#   The result is an image where different semantic regions have
#   different photometric styles, which never happens with global
#   augmentations.
#
#   If it fails, it's just a mild per-region color shift — harmless.
#
def semantic_region_style_swap(image, mask, dataset, beta=0.25):
    """
    For each semantic class present in `mask`, sample a donor image
    from `dataset`, extract the low-frequency Fourier style of the
    matching class region, and blend it into the source image's
    pixels for that class.

    Args:
        image:   uint8 (H, W, 3)
        mask:    uint8/int (H, W) — semantic label IDs
        dataset: the raw Cityscapes dataset to sample donors from
        beta:    blending strength for the donor style (0 = no effect)

    Returns:
        uint8 (H, W, 3)  — image with per-region style swapped
    """
    result = image.copy().astype(np.float32)
    h, w = image.shape[:2]

    # Find unique classes present (ignore very small regions < 500 px)
    unique_classes = np.unique(mask)

    for cls_id in unique_classes:
        if cls_id == 255:  # ignore unlabelled
            continue

        region_mask = (mask == cls_id)
        if region_mask.sum() < 500:
            continue

        # Pick a random donor image
        donor_idx = np.random.randint(0, len(dataset))
        donor_img, donor_target = dataset[donor_idx]
        donor_img = np.array(donor_img, dtype=np.float32)
        donor_mask = np.array(donor_target)

        # Check if the donor has this class too
        donor_region = (donor_mask == cls_id)
        if donor_region.sum() < 100:
            continue  # donor doesn't have this class, skip

        # Extract the mean colour of this class in both images
        # (this is the "style" — the DC component / low-freq)
        src_pixels = image[region_mask].astype(np.float32)   # (N, 3)
        don_pixels = donor_img[donor_mask == cls_id]          # (M, 3)

        src_mean = src_pixels.mean(axis=0)  # (3,)
        src_std = src_pixels.std(axis=0) + 1e-6
        don_mean = don_pixels.mean(axis=0)
        don_std = don_pixels.std(axis=0) + 1e-6

        # Histogram-style transfer for this region only:
        #   normalise source pixels → re-scale with donor stats
        #   then blend with original using beta
        normalised = (src_pixels - src_mean) / src_std
        transferred = normalised * don_std + don_mean
        blended = (1 - beta) * src_pixels + beta * transferred

        result[region_mask] = blended

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


# ═══════════════════════════════════════════════════════════════════
# DATASET WRAPPER
# ═══════════════════════════════════════════════════════════════════

class CityscapeAlbumentations(torch.utils.data.Dataset):
    """
    Wraps a Cityscapes dataset and applies albumentations transforms
    plus optional Fourier / copy-paste augmentations.
    """

    def __init__(self, dataset, transform=None,
                 apply_fourier=False, apply_copypaste=False,
                 apply_freq_band_dropout=False,
                 apply_semantic_style_swap=False):
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

        # Occlusion copy-paste (before any other augmentation)
        if self.apply_copypaste:
            image, mask = occlusion_copy_paste(
                image, mask, self.dataset,
                probability=config.COPYPASTE_PROBABILITY,
            )

        # Fourier magnitude blending (global style transfer)
        if self.apply_fourier and np.random.rand() < config.FOURIER_PROBABILITY:
            j = np.random.randint(0, len(self.dataset))
            img2, _ = self.dataset[j]
            img2 = np.array(img2)
            image = fast_fourier_transform(image, img2, alpha=config.FOURIER_ALPHA)

        # ── Novel: Frequency Band Dropout ─────────────────────
        if self.apply_freq_band_dropout and np.random.rand() < config.FREQ_BAND_DROPOUT_PROBABILITY:
            image = frequency_band_dropout(
                image,
                band_width=config.FREQ_BAND_DROPOUT_WIDTH,
                max_bands=config.FREQ_BAND_DROPOUT_MAX_BANDS,
            )

        # ── Novel: Semantic Region Style Swap ─────────────────
        if self.apply_semantic_style_swap and np.random.rand() < config.SEMANTIC_STYLE_SWAP_PROBABILITY:
            image = semantic_region_style_swap(
                image, mask, self.dataset,
                beta=config.SEMANTIC_STYLE_SWAP_BETA,
            )

        # Albumentations pipeline (spatial + photometric + corruption)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0).long()

        return image, mask
