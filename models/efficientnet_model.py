"""
=============================================================
STEP 5 — Model 2: EfficientNet-B4
=============================================================
Project: Image Forgery Detection in Social Media Images
         Using Deep Learning Techniques

PURPOSE OF THIS MODEL:
-----------------------
This is the IMPROVED model — better than ResNet50 but
still not our best.

Unlike ResNet50 (Model 1):
  ✓ Uses EfficientNet-B4 — more powerful backbone
  ✓ Trained WITH social media compressed images
  ✗ No frequency domain analysis yet (that's Model 3)

In our comparison table this model shows:
  - Better than ResNet50 on all platforms
  - But still weaker than our Hybrid Model (Model 3)
  - Proves that compression-aware training alone helps

ARCHITECTURE:
-------------
  EfficientNet-B4 (pretrained on ImageNet)
      ↓
  Custom classifier head:
      FC(1792 → 512) → ReLU → Dropout(0.4)
      FC(512  → 2)   → output: [authentic, forged]

WHY EfficientNet-B4 over B0/B7:
  B0 = too small, misses subtle forgery patterns
  B4 = best balance of accuracy vs speed for our task
  B7 = too large, slow to train, overkill for prototype
=============================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────

class EfficientNetModel(nn.Module):
    """
    EfficientNet-B4 model for image forgery detection.

    Trained WITH social media compressed images — so it
    is compression-aware unlike the ResNet50 baseline.

    Still uses only spatial (pixel-level) features.
    Model 3 (Hybrid) adds frequency domain analysis on top.
    """

    def __init__(self, num_classes=2, pretrained=True,
                 freeze_backbone=False):
        """
        Args:
            num_classes     : 2 (authentic / forged)
            pretrained      : use ImageNet pretrained weights
            freeze_backbone : only train classifier head
        """
        super(EfficientNetModel, self).__init__()

        # Load EfficientNet-B4
        if pretrained:
            try:
                weights  = models.EfficientNet_B4_Weights.IMAGENET1K_V1
                backbone = models.efficientnet_b4(weights=weights)
                print('  EfficientNet-B4: loaded ImageNet pretrained weights')
            except Exception:
                backbone = models.efficientnet_b4(weights=None)
                print('  EfficientNet-B4: pretrained download failed — '
                      'using random weights (run on your PC for full weights)')
        else:
            backbone = models.efficientnet_b4(weights=None)
            print('  EfficientNet-B4: no pretrained weights')

        # Freeze backbone if requested
        if freeze_backbone:
            for param in backbone.features.parameters():
                param.requires_grad = False
            print('  EfficientNet-B4: backbone frozen')

        # Keep the feature extraction part
        # EfficientNet-B4 output features: 1792
        self.features    = backbone.features
        self.avgpool     = backbone.avgpool

        # Custom classifier head
        # 1792 → 512 → 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1792, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )

    # ----------------------------------------------------------
    def forward(self, x):
        """
        Forward pass.

        Args:
            x : tensor (B x 3 x 224 x 224)

        Returns:
            logits : tensor (B x 2)
        """
        features = self.features(x)       # (B, 1792, 7, 7)
        pooled   = self.avgpool(features)  # (B, 1792, 1, 1)
        logits   = self.classifier(pooled) # (B, 2)
        return logits

    # ----------------------------------------------------------
    def get_features(self, x):
        """
        Extract feature vector before classifier.
        Used by Grad-CAM.

        Returns: tensor (B x 1792)
        """
        features = self.features(x)
        pooled   = self.avgpool(features)
        return pooled.squeeze(-1).squeeze(-1)

    # ----------------------------------------------------------
    def predict(self, x):
        """
        Get prediction probabilities.

        Returns:
            probs  : tensor (B x 2)
            labels : tensor (B,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)
            labels = torch.argmax(probs, dim=1)
        return probs, labels


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────────────────────
# SELF TEST
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('STEP 5 — Model 2: EfficientNet-B4')
    print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    # Build model
    print('\nBuilding EfficientNet-B4...')
    model = EfficientNetModel(num_classes=2, pretrained=True)
    model = model.to(device)

    # Count parameters
    total, trainable = count_parameters(model)
    print(f'\nModel parameters:')
    print(f'  Total     : {total:,}')
    print(f'  Trainable : {trainable:,}')

    # Test forward pass
    print('\nTesting forward pass...')
    dummy = torch.randn(4, 3, 224, 224).to(device)
    out   = model(dummy)
    print(f'  Input  shape : {dummy.shape}')
    print(f'  Output shape : {out.shape}')

    # Test predict
    probs, labels = model.predict(dummy)
    print(f'\n  Probabilities  : {probs[0].detach().cpu().numpy()}')
    print(f'  Predicted class: {labels[0].item()} '
          f'({["authentic","forged"][labels[0].item()]})')

    # Test features
    feats = model.get_features(dummy)
    print(f'\n  Feature vector shape: {feats.shape}')

    # Compare with ResNet50
    from models.resnet_baseline import ResNetBaseline
    resnet   = ResNetBaseline(pretrained=False)
    _, r_tr  = count_parameters(resnet)
    _, e_tr  = count_parameters(model)
    print(f'\nModel comparison:')
    print(f'  ResNet50       : {r_tr:,} params')
    print(f'  EfficientNet-B4: {e_tr:,} params')

    print('\n' + '=' * 60)
    print('Step 5 COMPLETE ✅ — efficientnet_model.py is ready')
    print('Next: python models/hybrid_model.py')
    print('=' * 60)
