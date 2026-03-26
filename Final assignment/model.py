"""
Model for semantic segmentation using segmentation_models_pytorch (SMP).
Supports UNet and DeepLabV3+ architectures.
"""

import copy
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import config


class Model(nn.Module):
    """
    Segmentation model — defaults to DeepLabV3+ with ResNet-101 backbone.
    """

    def __init__(
        self,
        in_channels=None,
        n_classes=None,
        encoder_name=None,
        encoder_weights=None,
        arch=None,
    ):
        super().__init__()

        in_channels = in_channels or config.IN_CHANNELS
        n_classes = n_classes or config.N_CLASSES
        encoder_name = encoder_name or config.ENCODER_NAME
        encoder_weights = encoder_weights if encoder_weights is not None else config.ENCODER_WEIGHTS
        arch = arch or getattr(config, "MODEL_ARCH", "DeepLabV3Plus")

        builder = getattr(smp, arch)
        self.model = builder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)


# ── Exponential Moving Average wrapper ──────────────────────────────
class EMA:
    """
    Maintains an exponential moving average of model parameters.
    Use the shadow weights at evaluation time for better robustness.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)

    def module(self):
        return self.shadow
