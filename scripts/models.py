"""
Model Architectures for Defect Classification
Includes EfficientNet-Lite with Attention and MobileNetV3 for Stage 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import MODEL_CONFIG, STAGE1_CONFIG, DATASET_CONFIG


class AttentionBlock(nn.Module):
    """Spatial Attention Module"""
    
    def __init__(self, channels: int):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate attention map
        attention_map = self.sigmoid(self.conv(x))
        # Apply attention
        return x * attention_map


class EfficientNetWithAttention(nn.Module):
    """
    EfficientNet-Lite3 with Spatial Attention for defect classification
    Modified for grayscale input (1 channel)
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        attention: bool = True,
        dropout: float = 0.3
    ):
        super(EfficientNetWithAttention, self).__init__()
        
        self.num_classes = num_classes
        self.attention_enabled = attention
        
        # Load pretrained EfficientNet-B0 (closest to Lite3)
        efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # Modify first conv layer for grayscale (1 channel input)
        original_first_conv = efficientnet.features[0][0]
        self.first_conv = nn.Conv2d(
            1,  # Change from 3 to 1 channel
            original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=False
        )
        
        # Initialize with averaged RGB weights
        if pretrained:
            with torch.no_grad():
                self.first_conv.weight = nn.Parameter(
                    original_first_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Use rest of the features (excluding first conv)
        self.features = nn.Sequential(
            self.first_conv,
            *list(efficientnet.features)[1:]
        )
        
        # Add attention modules at different scales
        if self.attention_enabled:
            self.attention1 = AttentionBlock(112)  # After block 4
            self.attention2 = AttentionBlock(320)  # After block 6
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1280, num_classes)  # EfficientNet-B0 has 1280 features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through feature extractor
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Apply attention at specific layers
            if self.attention_enabled:
                if i == 5 and x.shape[1] == 112:  # After block 4
                    x = self.attention1(x)
                elif i == 7 and x.shape[1] == 320:  # After block 6
                    x = self.attention2(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_attention_maps(self, x: torch.Tensor) -> list:
        """Get attention maps for visualization"""
        attention_maps = []
        
        if not self.attention_enabled:
            return attention_maps
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            if i == 5 and x.shape[1] == 112:
                attention_map = self.attention1.sigmoid(self.attention1.conv(x))
                attention_maps.append(attention_map)
            elif i == 7 and x.shape[1] == 320:
                attention_map = self.attention2.sigmoid(self.attention2.conv(x))
                attention_maps.append(attention_map)
        
        return attention_maps


class MobileNetV3Binary(nn.Module):
    """
    MobileNetV3-Small for Stage 1 binary classification (Clean vs Defect)
    Modified for grayscale input
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super(MobileNetV3Binary, self).__init__()
        
        # Load pretrained MobileNetV3-Small
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Modify first conv layer for grayscale
        original_first_conv = mobilenet.features[0][0]
        self.first_conv = nn.Conv2d(
            1,
            original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=False
        )
        
        # Initialize with averaged RGB weights
        if pretrained:
            with torch.no_grad():
                self.first_conv.weight = nn.Parameter(
                    original_first_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Use rest of the features
        self.features = nn.Sequential(
            self.first_conv,
            *list(mobilenet.features)[1:]
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Binary classifier
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(576, 2)  # MobileNetV3-Small has 576 features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CascadeDefectClassifier(nn.Module):
    """
    Two-stage cascade classifier:
    - Stage 1: Fast binary screener (Clean vs Defect)
    - Stage 2: Detailed 8-class classifier
    """
    
    def __init__(
        self,
        stage1_model: Optional[nn.Module] = None,
        stage2_model: Optional[nn.Module] = None,
        confidence_threshold: float = 0.95
    ):
        super(CascadeDefectClassifier, self).__init__()
        
        self.stage1 = stage1_model if stage1_model is not None else MobileNetV3Binary()
        self.stage2 = stage2_model if stage2_model is not None else EfficientNetWithAttention()
        self.confidence_threshold = confidence_threshold
    
    def forward(self, x: torch.Tensor, use_cascade: bool = True) -> torch.Tensor:
        """
        Forward pass with optional cascade logic
        
        Args:
            x: Input tensor
            use_cascade: If True, use cascade logic. If False, use only stage 2.
        """
        if not use_cascade:
            return self.stage2(x)
        
        # Stage 1: Binary classification
        with torch.no_grad():
            stage1_output = self.stage1(x)
            stage1_probs = F.softmax(stage1_output, dim=1)
            clean_confidence = stage1_probs[:, 0]  # Confidence for "clean" class
        
        # If confidence for "clean" is high, return "clean" classification
        # Otherwise, run detailed classifier
        final_output = torch.zeros(x.size(0), 8, device=x.device)
        
        high_confidence_mask = clean_confidence > self.confidence_threshold
        
        # For high confidence clean samples, set clean class prob to 1
        final_output[high_confidence_mask, 0] = clean_confidence[high_confidence_mask]
        
        # For low confidence samples, run stage 2
        if (~high_confidence_mask).any():
            low_conf_indices = torch.where(~high_confidence_mask)[0]
            stage2_input = x[low_conf_indices]
            stage2_output = self.stage2(stage2_input)
            final_output[low_conf_indices] = stage2_output
        
        return final_output


def create_model(
    model_type: str = 'efficientnet',
    num_classes: int = 8,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: One of 'efficientnet', 'mobilenet', 'cascade'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    """
    if model_type == 'efficientnet':
        return EfficientNetWithAttention(
            num_classes=num_classes,
            pretrained=pretrained,
            attention=MODEL_CONFIG.get('attention', True),
            dropout=MODEL_CONFIG.get('dropout', 0.3)
        )
    
    elif model_type == 'mobilenet':
        return MobileNetV3Binary(
            pretrained=pretrained,
            dropout=STAGE1_CONFIG.get('dropout', 0.2)
        )
    
    elif model_type == 'cascade':
        stage1 = MobileNetV3Binary(pretrained=pretrained)
        stage2 = EfficientNetWithAttention(
            num_classes=num_classes,
            pretrained=pretrained,
            attention=MODEL_CONFIG.get('attention', True),
            dropout=MODEL_CONFIG.get('dropout', 0.3)
        )
        return CascadeDefectClassifier(stage1, stage2)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


if __name__ == "__main__":
    """Test model creation"""
    print("Testing Model Creation...")
    
    # Test EfficientNet with Attention
    print("\n1. EfficientNet with Attention:")
    model_eff = create_model('efficientnet', num_classes=8, pretrained=False)
    print(f"  Parameters: {count_parameters(model_eff):,}")
    print(f"  Estimated size: {get_model_size_mb(model_eff):.2f} MB")
    
    # Test input
    dummy_input = torch.randn(1, 1, 224, 224)
    output = model_eff(dummy_input)
    print(f"  Output shape: {output.shape}")
    
    # Test MobileNetV3 Binary
    print("\n2. MobileNetV3 Binary Classifier:")
    model_mob = create_model('mobilenet', pretrained=False)
    print(f"  Parameters: {count_parameters(model_mob):,}")
    print(f"  Estimated size: {get_model_size_mb(model_mob):.2f} MB")
    
    dummy_input_small = torch.randn(1, 1, 128, 128)
    output = model_mob(dummy_input_small)
    print(f"  Output shape: {output.shape}")
    
    # Test Cascade
    print("\n3. Cascade Classifier:")
    model_cascade = create_model('cascade', num_classes=8, pretrained=False)
    print(f"  Stage 1 parameters: {count_parameters(model_cascade.stage1):,}")
    print(f"  Stage 2 parameters: {count_parameters(model_cascade.stage2):,}")
    print(f"  Total parameters: {count_parameters(model_cascade):,}")
    
    output = model_cascade(dummy_input, use_cascade=False)
    print(f"  Output shape: {output.shape}")
    
    print("\n✅ Model creation test passed!")
