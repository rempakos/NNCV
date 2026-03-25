"""
UNet model for semantic segmentation using segmentation_models_pytorch (SMP).
"""

import torch
import segmentation_models_pytorch as smp
import config


class Model(torch.nn.Module):
    """
    U-Net model for semantic segmentation.
    Uses ResNet50 backbone with ImageNet pre-training.
    """
    
    def __init__(
        self,
        in_channels=None,
        n_classes=None,
        encoder_name=None,
        encoder_weights=None,
    ):
        """
        Args:
            in_channels (int): Number of input channels. Default is 3 for RGB.
            n_classes (int): Number of output classes.
            encoder_name (str): Name of the encoder backbone (e.g., 'resnet50').
            encoder_weights (str): Pre-trained weights ('imagenet' or None).
        """
        super().__init__()
        
        # Use config defaults if not provided
        in_channels = in_channels or config.IN_CHANNELS
        n_classes = n_classes or config.N_CLASSES
        encoder_name = encoder_name or config.ENCODER_NAME
        encoder_weights = encoder_weights if encoder_weights is not None else config.ENCODER_WEIGHTS
        
        # Create UNet model with spatial attention
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
            activation=None,  # No activation, raw logits
            decoder_attention_type="scse",  # Squeeze-and-Excitation attention for robustness
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output logits of shape (B, num_classes, H, W)
        """
        return self.model(x)