import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import config


class Model(nn.Module):
    """
    OCRNet (Object-Contextual Representations) for Semantic Segmentation.
    Specifically designed for handling occlusions, making it ideal for robustness benchmarking.
    
    The model uses a ResNet50 backbone with Object Contextual Representation heads
    to aggregate pixel-level and object-level context for improved segmentation.
    """
    def __init__(
        self, 
        in_channels=None, 
        n_classes=None,
        encoder_name=None,
        encoder_weights=None,
        ocr_mid_channels=512,
        ocr_key_channels=256,
        align_corners=False,
    ):
        """
        Args:
            in_channels (int): Number of input channels. Default is 3 for RGB images.
            n_classes (int): Number of output classes. Default is 19 for Cityscapes.
            encoder_name (str): Backbone encoder. Only resnet50 supported.
            encoder_weights (str): Pre-trained weights. Default is imagenet.
            ocr_mid_channels (int): Hidden channels in OCR head.
            ocr_key_channels (int): Key channels for attention in OCR.
            align_corners (bool): Interpolation mode for upsampling.
        """
        super().__init__()
        
        # Use config defaults
        in_channels = in_channels or config.IN_CHANNELS
        n_classes = n_classes or config.N_CLASSES
        encoder_name = encoder_name or config.ENCODER_NAME
        encoder_weights = encoder_weights if encoder_weights is not None else config.ENCODER_WEIGHTS
        
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.align_corners = align_corners
        
        # Load ResNet50 backbone (ignoring encoder_name for now, always use resnet50)
        weights = "IMAGENET1K_V1" if encoder_weights == "imagenet" else None
        resnet = models.resnet50(weights=weights)
        
        # Extract backbone layers (remove classification head)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,      # 256 channels, 1/4
            resnet.layer2,      # 512 channels, 1/8
            resnet.layer3,      # 1024 channels, 1/16
            resnet.layer4,      # 2048 channels, 1/32
        )
        
        # Backbone channel dimensions
        self.backbone_channels = [256, 512, 1024, 2048]
        
        # OCR Head
        self.ocr_head = OCRHead(
            num_classes=n_classes,
            in_channels=self.backbone_channels,
            ocr_mid_channels=ocr_mid_channels,
            ocr_key_channels=ocr_key_channels,
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            logits: Segmentation logits of shape (batch_size, n_classes, height, width)
        """
        # Check if the input tensor has the expected number of channels
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")
        
        # Extract features from backbone at different scales
        feats = []
        feat = x
        for i, layer in enumerate(self.backbone):
            if i < 4:  # conv1, bn1, relu, maxpool
                feat = layer(feat)
            elif i < 7:  # layer1, layer2, layer3
                feat = layer(feat)
                if i >= 5:  # Collect from layer2 and layer3
                    feats.append(feat)
            else:  # layer4
                feat = layer(feat)
                feats.append(feat)
        
        # Get backbone outputs at different scales
        feat_shallow = feats[0]  # 1/8 scale, 512 channels
        feat_deep = feats[-1]    # 1/32 scale, 2048 channels
        
        # OCR Head forward
        logit, _ = self.ocr_head([feat_shallow, feat_deep])
        
        # Interpolate to input size
        logit = F.interpolate(
            logit, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=self.align_corners
        )
        
        return logit


class OCRHead(nn.Module):
    """Object Contextual Representation Head"""
    
    def __init__(self, num_classes, in_channels, ocr_mid_channels=512, ocr_key_channels=256):
        super().__init__()
        
        self.num_classes = num_classes
        self.ocr_mid_channels = ocr_mid_channels
        
        # Spatial gather module
        self.spatial_gather = SpatialGatherBlock(ocr_mid_channels, num_classes)
        
        # Spatial OCR module
        self.spatial_ocr = SpatialOCRModule(ocr_mid_channels, ocr_key_channels, ocr_mid_channels)
        
        # Convolution for feature extraction
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(in_channels[-1], ocr_mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
        )
        
        # Main classification head
        self.cls_head = nn.Conv2d(ocr_mid_channels, num_classes, kernel_size=1)
        
        # Auxiliary head for deep supervision
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], kernel_size=1),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[0], num_classes, kernel_size=1),
        )
    
    def forward(self, feat_list):
        """
        Args:
            feat_list: List of features from backbone at different scales
        Returns:
            Main logits and auxiliary logits
        """
        feat_shallow = feat_list[0]  # 512 channels
        feat_deep = feat_list[-1]    # 2048 channels
        
        # Auxiliary head prediction
        soft_regions = self.aux_head(feat_shallow)
        
        # Process deep features through OCR
        pixels = self.conv3x3_ocr(feat_deep)
        
        # Gather object regions
        object_regions = self.spatial_gather(pixels, soft_regions)
        
        # Apply spatial OCR
        ocr_feat = self.spatial_ocr(pixels, object_regions)
        
        # Final classification
        logit = self.cls_head(ocr_feat)
        
        return logit, soft_regions


class SpatialGatherBlock(nn.Module):
    """Aggregates pixel features into object/region representations"""
    
    def __init__(self, pixels_channels, regions_channels):
        super().__init__()
        self.pixels_channels = pixels_channels
        self.regions_channels = regions_channels
    
    def forward(self, pixels, regions):
        """
        Args:
            pixels: (N, C, H, W) - pixel-level features
            regions: (N, K, H, W) - region predictions
        Returns:
            feats: (N, C, K, 1) - aggregated region features
        """
        # Reshape pixels: (N, C, H, W) -> (N, H*W, C)
        N, C, H, W = pixels.shape
        pixels_flat = pixels.view(N, C, -1).transpose(1, 2)  # (N, H*W, C)
        
        # Reshape regions: (N, K, H, W) -> (N, K, H*W)
        regions_flat = regions.view(N, self.regions_channels, -1)
        regions_flat = F.softmax(regions_flat, dim=2)
        
        # Aggregate: (N, K, H*W) @ (N, H*W, C) -> (N, K, C)
        feats = torch.bmm(regions_flat, pixels_flat)
        
        # Reshape to (N, C, K, 1)
        feats = feats.transpose(1, 2).unsqueeze(-1)
        
        return feats


class SpatialOCRModule(nn.Module):
    """Aggregates global object context to each pixel"""
    
    def __init__(self, in_channels, key_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        
        self.attention_block = ObjectAttentionBlock(in_channels, key_channels)
        
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )
    
    def forward(self, pixels, regions):
        """
        Args:
            pixels: (N, C, H, W) - pixel features
            regions: (N, C, K, 1) - region features
        Returns:
            Enhanced pixel features
        """
        context = self.attention_block(pixels, regions)
        
        # Concatenate context and original pixels
        feats = torch.cat([context, pixels], dim=1)
        feats = self.conv1x1(feats)
        
        return feats


class ObjectAttentionBlock(nn.Module):
    """Self-attention module for object-pixel relationships"""
    
    def __init__(self, in_channels, key_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.key_channels = key_channels
        
        # Query projection: pixels
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        
        # Key projection: regions
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        
        # Value projection
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        
        # Output projection
        self.f_up = nn.Sequential(
            nn.Conv2d(key_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, proxy):
        """
        Args:
            x: (N, C, H, W) - pixel features
            proxy: (N, C, K, 1) - region/proxy features
        Returns:
            Context-aggregated features (N, C, H, W)
        """
        x_shape = x.shape
        
        # Query from pixels: (N, C, H, W) -> (N, H*W, key_channels)
        query = self.f_pixel(x)
        query = query.view(query.shape[0], self.key_channels, -1).transpose(1, 2)
        
        # Key from regions: (N, C, K, 1) -> (N, key_channels, K)
        key = self.f_object(proxy)
        key = key.view(key.shape[0], self.key_channels, -1)
        
        # Value from regions: (N, C, K, 1) -> (N, K, key_channels)
        value = self.f_down(proxy)
        value = value.view(value.shape[0], self.key_channels, -1).transpose(1, 2)
        
        # Attention: (N, H*W, key_channels) @ (N, key_channels, K) -> (N, H*W, K)
        sim_map = torch.bmm(query, key)
        sim_map = sim_map * (self.key_channels ** -0.5)
        sim_map = F.softmax(sim_map, dim=-1)
        
        # Apply attention: (N, H*W, K) @ (N, K, key_channels) -> (N, H*W, key_channels)
        context = torch.bmm(sim_map, value)
        
        # Reshape back: (N, H*W, key_channels) -> (N, key_channels, H, W)
        context = context.transpose(1, 2)
        context = context.view(x_shape[0], self.key_channels, x_shape[2], x_shape[3])
        context = self.f_up(context)
        
        return context