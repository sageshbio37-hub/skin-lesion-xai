import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 8

# ============================================
# Model 1: EfficientNet B0
# ============================================
def get_efficientnet():
    model = models.efficientnet_b0(weights=None)
    
    # Unfreeze last 3 blocks
    for name, param in model.named_parameters():
        if 'features.7' in name or 'features.8' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )
    return model

# ============================================
# Model 2: Vision Transformer (ViT)
# ============================================
def get_vit():
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, NUM_CLASSES)
    return model

# ============================================
# Test both models
# ============================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test EfficientNet
    print("\n=== EfficientNet B0 ===")
    eff_model = get_efficientnet().to(device)
    dummy = torch.randn(2, 3, 224, 224).to(device)
    out = eff_model(dummy)
    print(f"Output shape: {out.shape}")
    total = sum(p.numel() for p in eff_model.parameters())
    trainable = sum(p.numel() for p in eff_model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    
    # Test ViT
    print("\n=== Vision Transformer ===")
    vit_model = get_vit().to(device)
    out2 = vit_model(dummy)
    print(f"Output shape: {out2.shape}")
    total2 = sum(p.numel() for p in vit_model.parameters())
    trainable2 = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    print(f"Total params: {total2:,}")
    print(f"Trainable params: {trainable2:,}")
    
    print("\nBoth models ready!")