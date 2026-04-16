"""
Compare predictions from three different models:
1. DINO v2 without augmentations
2. DINO v2 with augmentations 
3. Baseline UNet

This script loads the same image through all three models and displays
side-by-side predictions like the example image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import argparse
from typing import Tuple
import importlib.util
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    ToDtype,
    Normalize,
    InterpolationMode,
)

# DINO imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import Model as DINOModel
import config as dino_config

# Baseline UNet imports - use importlib to avoid name collisions
# Use relative paths from script location for portability
script_dir = Path(__file__).parent
baseline_src_dir = script_dir / "baseline_src"
models_dir = script_dir / "models"

def load_baseline_module():
    """Dynamically load baseline model module to avoid import collisions."""
    baseline_model_file = baseline_src_dir / "model.py"
    if not baseline_model_file.exists():
        raise FileNotFoundError(
            f"Baseline model code not found at: {baseline_model_file}\n"
            f"Expected structure: {script_dir}/baseline_src/model.py"
        )
    spec = importlib.util.spec_from_file_location(
        "baseline_model",
        baseline_model_file
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

baseline_module = load_baseline_module()
BaselineModel = baseline_module.Model

# Cityscapes colors
CITYSCAPES_COLORS = np.array([
    [128, 64, 128],    # road
    [244, 35, 232],    # sidewalk
    [70, 70, 70],      # building
    [102, 102, 156],   # wall
    [190, 153, 153],   # fence
    [153, 153, 153],   # pole
    [250, 170, 100],   # traffic light
    [220, 220, 0],     # traffic sign
    [107, 142, 35],    # vegetation
    [152, 251, 152],   # terrain
    [70, 130, 180],    # sky
    [220, 20, 60],     # person
    [255, 0, 0],       # rider
    [0, 0, 142],       # car
    [0, 0, 70],        # truck
    [0, 60, 100],      # bus
    [0, 80, 100],      # train
    [0, 0, 230],       # motorcycle
    [119, 11, 32],     # bicycle
])


def load_dino_model(model_path: str, augmented: bool = False) -> DINOModel:
    """Load DINO v2 model from checkpoint."""
    model = DINOModel(
        backbone_name=dino_config.BACKBONE,
        n_classes=dino_config.N_CLASSES,
        pretrained=False,
    )
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_baseline_model(model_path: str) -> BaselineModel:
    """Load baseline UNet model from checkpoint."""
    model = BaselineModel(in_channels=3, n_classes=19)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def preprocess_dino(img: Image.Image) -> torch.Tensor:
    """Preprocess image for DINO v2."""
    transform = Compose([
        ToImage(),
        Resize(
            size=(dino_config.INPUT_H, dino_config.INPUT_W),
            interpolation=InterpolationMode.BILINEAR,
        ),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(img).unsqueeze(0)


def preprocess_baseline(img: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Preprocess image for baseline UNet."""
    original_h, original_w = img.size[::-1]
    
    # Resize to standard dimension (e.g., 512x1024 for Cityscapes)
    img_resized = img.resize((1024, 512), Image.Resampling.BILINEAR)
    
    transform = Compose([
        ToImage(),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    tensor = transform(img_resized).unsqueeze(0)
    return tensor, (original_h, original_w)


def predict_dino(model: DINOModel, img_tensor: torch.Tensor) -> np.ndarray:
    """Get prediction from DINO v2 model."""
    with torch.no_grad():
        pred = model(img_tensor)
    
    pred_soft = nn.Softmax(dim=1)(pred)
    pred_max = torch.argmax(pred_soft, dim=1, keepdim=True)
    
    return pred_max.cpu().numpy().squeeze()


def predict_baseline(model: BaselineModel, img_tensor: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
    """Get prediction from baseline UNet model."""
    with torch.no_grad():
        pred = model(img_tensor)
    
    # Upsample back to original size
    pred_upsampled = F.interpolate(
        pred, 
        size=original_shape, 
        mode='bilinear', 
        align_corners=False
    )
    
    pred_max = torch.argmax(pred_upsampled, dim=1, keepdim=True)
    
    return pred_max.cpu().numpy().squeeze()


def mask_to_rgb(mask: np.ndarray) -> Image.Image:
    """Convert semantic mask to RGB image using Cityscapes colors."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in enumerate(CITYSCAPES_COLORS):
        rgb[mask == class_id] = color
    
    return Image.fromarray(rgb)


def create_comparison_figure(
    original_img: Image.Image,
    baseline_pred: np.ndarray,
    dino_noaug_pred: np.ndarray,
    dino_aug_pred: np.ndarray,
    output_path: str = "comparison.png"
) -> None:
    """Create a 2x2 (or more) comparison figure like the example image."""
    
    # Ensure all predictions are resized to original image size for consistent display
    original_size = original_img.size
    
    # Resize predictions to original image size
    baseline_rgb = mask_to_rgb(baseline_pred)
    baseline_rgb = baseline_rgb.resize(original_size, Image.Resampling.NEAREST)
    
    dino_noaug_rgb = mask_to_rgb(dino_noaug_pred)
    dino_noaug_rgb = dino_noaug_rgb.resize(original_size, Image.Resampling.NEAREST)
    
    dino_aug_rgb = mask_to_rgb(dino_aug_pred)
    dino_aug_rgb = dino_aug_rgb.resize(original_size, Image.Resampling.NEAREST)
    
    # Create figure with 2x2 layout
    img_width, img_height = original_size
    figure_width = img_width * 2 + 40  # 20px margin on each side
    figure_height = img_height * 2 + 100  # Extra space for labels
    
    figure = Image.new("RGB", (figure_width, figure_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(figure)
    
    # Try to use a nice font, fallback to default if not available
    try:
        title_font = ImageFont.truetype("arial.ttf", 48)
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        title_font = ImageFont.load_default()
        font = ImageFont.load_default()
    
    # Layout: top-left=original, top-right=baseline, bottom-left=dino_noaug, bottom-right=dino_aug
    label_color = (255, 255, 255)  # White text
    bg_color = (50, 50, 50)  # Dark gray background
    
    positions = [
        (10, 50, "Original Image"),
        (img_width + 20, 50, "Baseline (UNet)"),
        (10, img_height + 60, "DINO v2 (No Augmentation)"),
        (img_width + 20, img_height + 60, "DINO v2 (With Augmentation)"),
    ]
    
    images = [original_img, baseline_rgb, dino_noaug_rgb, dino_aug_rgb]
    
    for (x, y, label), img in zip(positions, images):
        figure.paste(img, (x, y))
        # Add label with background
        text_y = y - 50
        bbox = draw.textbbox((x, text_y), label, font=title_font)
        # Draw semi-transparent background box
        draw.rectangle(
            [(bbox[0] - 10, bbox[1] - 5), (bbox[2] + 10, bbox[3] + 5)],
            fill=bg_color
        )
        draw.text((x, text_y), label, fill=label_color, font=title_font)
    
    figure.save(output_path)
    print(f"Comparison figure saved to: {output_path}")
    return figure


def main():
    parser = argparse.ArgumentParser(description="Compare model predictions")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image",
        default=None
    )
    parser.add_argument(
        "--dino-noaug",
        type=str,
        default=str(models_dir / "dino_model_without_augmentation" / "model.pt"),
        help="Path to DINO model without augmentations"
    )
    parser.add_argument(
        "--dino-aug",
        type=str,
        default=str(models_dir / "dino_model_with_augmentation" / "model.pt"),
        help="Path to DINO model with augmentations"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=str(models_dir / "baseline" / "model.pt"),
        help="Path to baseline UNet model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison.png",
        help="Output path for comparison image"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # If no image specified, try to find one in the test data directories
    if args.image is None:
        # Check multiple possible locations
        search_dirs = [
            script_dir.parent.parent / "sample_data",  # ../../../sample_data
            script_dir / "sample_data",  # ./sample_data
            script_dir / "local_data",  # ./local_data (backward compatibility)
            script_dir / "img",  # ./img
        ]
        
        for img_dir in search_dirs:
            if img_dir.exists():
                images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
                if images:
                    args.image = str(images[0])  # Always use first image
                    print(f"Using image: {args.image}")
                    break
        
        if args.image is None:
            print("Error: No image specified and no test images found in:")
            for d in search_dirs:
                print(f"  - {d}")
            print("Usage: python compare_models.py --image /path/to/image.png")
            return
    
    if not Path(args.image).exists():
        print(f"Error: Image not found at {args.image}")
        return
    
    print("Loading models...")
    device = torch.device(args.device)
    
    # Load models
    dino_noaug = load_dino_model(args.dino_noaug).to(device)
    baseline = load_baseline_model(args.baseline).to(device)
    
    # Load DINO augmented - check if it exists
    if Path(args.dino_aug).exists():
        dino_aug = load_dino_model(args.dino_aug).to(device)
    else:
        print(f"Warning: Augmented DINO model not found at {args.dino_aug}")
        print("Using non-augmented version for comparison instead")
        dino_aug = dino_noaug
    
    print(f"Loading image: {args.image}")
    original_img = Image.open(args.image).convert("RGB")
    
    print("Running predictions...")
    
    # DINO predictions
    dino_input = preprocess_dino(original_img).to(device)
    dino_noaug_pred = predict_dino(dino_noaug, dino_input)
    dino_aug_pred = predict_dino(dino_aug, dino_input)
    
    # Baseline prediction
    baseline_input, original_shape = preprocess_baseline(original_img)
    baseline_input = baseline_input.to(device)
    baseline_pred = predict_baseline(baseline, baseline_input, original_img.size[::-1])
    
    print("Creating comparison figure...")
    create_comparison_figure(
        original_img,
        baseline_pred,
        dino_noaug_pred,
        dino_aug_pred,
        args.output
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
