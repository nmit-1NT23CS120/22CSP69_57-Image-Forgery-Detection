"""
=============================================================
STEP 4 — Model 1: ResNet50 Baseline
=============================================================
Project: Image Forgery Detection in Social Media Images
         Using Deep Learning Techniques

PURPOSE OF THIS MODEL:
-----------------------
This is the BASELINE model — it represents what existing
forgery detection systems look like.

It is trained ONLY on clean images with NO social media
compression awareness.

In our comparison table, this model will:
  ✓ Perform well on clean test images (~80-85%)
  ✗ Perform BADLY on Facebook/Instagram/WhatsApp images
    (accuracy drops to ~50-60%)

This PROVES that existing models fail on social media images.
Our hybrid model (Step 6) will stay strong across all platforms.

ARCHITECTURE:
-------------
  ResNet50 (pretrained on ImageNet)
      ↓
  Custom classifier head:
      FC(2048 → 512) → ReLU → Dropout(0.5)
      FC(512 → 2)    → output: [authentic, forged]
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

class ResNetBaseline(nn.Module):
    """
    ResNet50 baseline model for image forgery detection.

    Uses ResNet50 pretrained on ImageNet as the backbone.
    Replaces the final classification layer with a custom
    2-class head (authentic vs forged).

    This is the "existing model" we compare against.
    It has NO social media compression awareness.
    """

    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes     : 2 (authentic / forged)
            pretrained      : use ImageNet pretrained weights
            freeze_backbone : if True, only train the classifier head
                              (faster but less accurate)
        """
        super(ResNetBaseline, self).__init__()

        # Load pretrained ResNet50
        if pretrained:
            try:
                weights  = models.ResNet50_Weights.IMAGENET1K_V1
                backbone = models.resnet50(weights=weights)
                print('  ResNet50: loaded ImageNet pretrained weights')
            except Exception:
                # Fallback — no pretrained weights (fine for testing)
                backbone = models.resnet50(weights=None)
                print('  ResNet50: pretrained download failed — '
                      'using random weights (run on your PC for full weights)')
        else:
            backbone = models.resnet50(weights=None)
            print('  ResNet50: no pretrained weights')

        # Optionally freeze backbone — only train classifier
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            print('  ResNet50: backbone frozen — training head only')

        # Keep everything except the final FC layer
        # ResNet50 output features: 2048
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Custom classifier head
        # 2048 → 512 → 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
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
                     [score_authentic, score_forged]
        """
        features = self.backbone(x)    # (B, 2048, 1, 1)
        logits   = self.classifier(features)  # (B, 2)
        return logits

    # ----------------------------------------------------------
    def get_features(self, x):
        """
        Extract feature vector before classifier.
        Used by Grad-CAM for visualization.

        Returns: tensor (B x 2048)
        """
        features = self.backbone(x)
        return features.squeeze(-1).squeeze(-1)

    # ----------------------------------------------------------
    def predict(self, x):
        """
        Get prediction probabilities.

        Returns:
            probs  : tensor (B x 2) — softmax probabilities
            labels : tensor (B,)    — predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)
            labels = torch.argmax(probs, dim=1)
        return probs, labels


# ─────────────────────────────────────────────────────────────
# MODEL INFO HELPER
# ─────────────────────────────────────────────────────────────

def count_parameters(model):
    """Count total and trainable parameters in model."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────────────────────
# SELF TEST
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('STEP 4 — Model 1: ResNet50 Baseline')
    print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    # Build model
    print('\nBuilding ResNet50 baseline...')
    model = ResNetBaseline(num_classes=2, pretrained=True)
    model = model.to(device)

    # Count parameters
    total, trainable = count_parameters(model)
    print(f'\nModel parameters:')
    print(f'  Total     : {total:,}')
    print(f'  Trainable : {trainable:,}')

    # Test forward pass
    print('\nTesting forward pass...')
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output      = model(dummy_input)
    print(f'  Input  shape : {dummy_input.shape}')
    print(f'  Output shape : {output.shape}')
    print(f'  Output (logits): {output[0].detach().cpu().numpy()}')

    # Test predict
    probs, labels = model.predict(dummy_input)
    print(f'\n  Probabilities : {probs[0].detach().cpu().numpy()}')
    print(f'  Predicted class: {labels[0].item()} '
          f'({["authentic","forged"][labels[0].item()]})')

    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f'\n  Feature vector shape: {features.shape}')

    print('\n' + '=' * 60)
    print('Step 4 COMPLETE ✅ — resnet_baseline.py is ready')
    print('Next: python models/efficientnet_model.py')
    print('=' * 60)
