import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import config


class Model(nn.Module):
    """
    DINOv2 backbone + linear segmentation head.

    The backbone outputs patch-level features of shape (B, N, D) where
    N = (H/P)*(W/P) patches. We reshape to a spatial grid, apply a
    1x1 conv to map D->num_classes, and bilinearly upsample to the
    input resolution.
    """

    def __init__(
        self,
        backbone_name=None,
        n_classes=None,
        pretrained=False,
        input_h=None,
        input_w=None,
    ):
        super().__init__()

        backbone_name = backbone_name or config.BACKBONE
        n_classes = n_classes or config.N_CLASSES
        self.input_h = input_h or config.INPUT_H
        self.input_w = input_w or config.INPUT_W
        self.patch_size = config.PATCH_SIZE
        embed_dim = config.EMBED_DIM

        # Patch grid dimensions
        self.grid_h = self.input_h // self.patch_size
        self.grid_w = self.input_w // self.patch_size

        #DINOv2 backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,          # remove classification head
            img_size=(self.input_h, self.input_w),
        )

        #Linear segmentation head
        # 1×1 conv is equivalent to a linear layer applied per patch
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, n_classes, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images

        Returns:
            (B, n_classes, H, W) logits at input resolution
        """
        B, _, H, W = x.shape

        # Extract patch features: (B, N, D)
        features = self.backbone.forward_features(x)

        # timm DINOv2 returns dict or tensor depending on version
        if isinstance(features, dict):
            features = features.get("x_norm_patchtokens", features.get("x"))
        
        # If there's a CLS token prepended, remove it
        expected_n = self.grid_h * self.grid_w
        if features.shape[1] != expected_n:
            # First token is CLS, rest are patches
            features = features[:, -expected_n:, :]

        # Reshape to spatial grid: (B, D, grid_h, grid_w)
        features = features.reshape(B, self.grid_h, self.grid_w, -1)
        features = features.permute(0, 3, 1, 2)

        # Linear head: (B, n_classes, grid_h, grid_w)
        logits = self.head(features)

        # Upsample to input resolution
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )

        return logits


#Exponential Moving Average
class EMA:
    """
    Maintains a smoothed copy of model weights for evaluation.
    EMA weights generalise better than raw training weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            s_param.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def module(self):
        return self.shadow
