"""
Prediction pipeline for the trained DINOv2 segmentation model.
Loads a pre-trained model, processes input images, and saves
predicted segmentation masks.

Compatible with the challenge submission server.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    ToDtype,
    Normalize,
    InterpolationMode,
)

from model import Model
import config

# Fixed paths inside participant container
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"


def preprocess(img: Image.Image) -> torch.Tensor:
    """Resize to DINOv2-compatible dimensions and normalize."""
    transform = Compose([
        ToImage(),
        Resize(
            size=(config.INPUT_H, config.INPUT_W),
            interpolation=InterpolationMode.BILINEAR,
        ),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(img).unsqueeze(0)


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    """Argmax + resize back to original resolution."""
    pred_soft = nn.Softmax(dim=1)(pred)
    pred_max = torch.argmax(pred_soft, dim=1, keepdim=True)
    prediction = Resize(
        size=original_shape, interpolation=InterpolationMode.NEAREST
    )(pred_max)
    return prediction.cpu().detach().numpy().squeeze()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model (pretrained=False because we load our own weights)
    model = Model(pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = np.array(img).shape[:2]

            img_tensor = preprocess(img).to(device)
            pred = model(img_tensor)
            seg_pred = postprocess(pred, original_shape)

            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()
