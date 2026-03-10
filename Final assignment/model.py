import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


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
        in_channels=3, 
        n_classes=19
    ):
        """
        Args:
            in_channels (int): Number of input channels. Default is 3 for RGB images.
            n_classes (int): Number of output classes. Default is 19 for the Cityscapes dataset.
        """
        
        super().__init__()
        # Implement the Unet with ResNet50 encoder as backbone encoder already trained with imagenet
        self.model = smp.Unet(
            encoder_name="resnet50", 
            encoder_weights="imagenet", 
            in_channels=in_channels, 
            classes=n_classes,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type=None,
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        """
        # Check if the input tensor has the expected number of channels
        if x.shape[1] != self.model.encoder.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")
        
        logits = self.model(x)

        return logits