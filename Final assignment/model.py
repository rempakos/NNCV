import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import config


class Model(nn.Module):
    """ 
    A simple U-Net architecture for image segmentation.
    Based on the U-Net architecture from the original paper:
    Olaf Ronneberger et al. (2015), "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Adapt this model as needed for your problem-specific requirements. You can make multiple model classes and compare them,
    however, the CodaLab server requires the model class to be named "Model". Also, it will use the default values of the constructor
    to create the model, so make sure to set the default values of the constructor to the ones you want to use for your submission.
    """
    def __init__(
        self, 
        in_channels=None, 
        n_classes=None,
        encoder_name=None,
        encoder_weights=None,
        decoder_channels=None,
        decoder_attention_type=None,
    ):
        """
        Args:
            in_channels (int): Number of input channels. Default is 3 for RGB images.
            n_classes (int): Number of output classes. Default is 19 for the Cityscapes dataset.
            encoder_name (str): Backbone encoder. Default is resnet101.
            encoder_weights (str): Pre-trained weights. Default is imagenet.
            decoder_channels (tuple): Decoder channel dimensions.
            decoder_attention_type (str): Attention mechanism type.
        """
        # Use config defaults if parameters not provided
        in_channels = in_channels or config.IN_CHANNELS
        n_classes = n_classes or config.N_CLASSES
        encoder_name = encoder_name or config.ENCODER_NAME
        encoder_weights = encoder_weights or config.ENCODER_WEIGHTS
        decoder_channels = decoder_channels or config.DECODER_CHANNELS
        decoder_attention_type = decoder_attention_type or config.DECODER_ATTENTION_TYPE
        
        super().__init__()
        # Implement the Unet with configurable encoder as backbone already trained with imagenet
        self.in_channels = in_channels
        self.model = smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights=encoder_weights, 
            in_channels=in_channels, 
            classes=n_classes,
            decoder_use_batchnorm=config.DECODER_USE_BATCHNORM,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        """
        # Check if the input tensor has the expected number of channels
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")
        
        logits = self.model(x)

        return logits