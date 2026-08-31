"""
=============================================================
Hybrid Model V2 — with EXIF Metadata Branch (Phase 2)
=============================================================
Extends the Phase 1 HybridModel (spatial + frequency) with a
third branch that consumes the 16-dim EXIF feature vector from
exif_utils.py.

Fusion: spatial(1792) + freq(256) + metadata(32) = 2080-dim -> classifier

IMPORTANT — designed for fine-tuning, not training from scratch:
  __init__(pretrained_hybrid_path=...) loads your existing Phase 1
  hybrid_model.pth weights into the spatial + frequency branches,
  freezes them initially, and only trains the new metadata branch
  + classifier head. You can unfreeze later for full fine-tuning.
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from exif_utils import EXIF_FEATURE_DIM


class FrequencyBranch(nn.Module):
    """Identical to Phase 1 — copied here so this file is self-contained."""
    def __init__(self):
        super().__init__()
        self.freq_cnn = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(128*4*4, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.3))

    def forward(self, x):
        g = x.mean(dim=1, keepdim=True)
        k = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                          dtype=x.dtype, device=x.device)
        f = F.conv2d(g, k, padding=1)
        f = torch.abs(f) / (torch.abs(f).max() + 1e-8)
        return self.fc(self.freq_cnn(f))


class MetadataBranch(nn.Module):
    """Small MLP encoding the 16-dim EXIF feature vector into 32-dim."""
    def __init__(self, in_dim=EXIF_FEATURE_DIM, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class HybridModelV2(nn.Module):
    """
    Three-branch fusion model: spatial (EfficientNet-B4) + frequency
    (Laplacian CNN) + metadata (EXIF MLP).
    """
    def __init__(self):
        super().__init__()
        b = models.efficientnet_b4(weights=None)
        self.spatial_features = b.features
        self.spatial_pool     = b.avgpool
        self.freq_branch      = FrequencyBranch()
        self.meta_branch      = MetadataBranch()

        fused_dim = 1792 + 256 + 32   # = 2080
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, x, exif_vec):
        sp = self.spatial_pool(self.spatial_features(x)).squeeze(-1).squeeze(-1)
        fr = self.freq_branch(x)
        md = self.meta_branch(exif_vec)
        fused = torch.cat([sp, fr, md], dim=1)
        return self.classifier(fused)

    def load_phase1_weights(self, phase1_state_dict, freeze_pretrained=True):
        """
        Loads spatial_features / spatial_pool / freq_branch weights from
        your Phase 1 hybrid_model.pth checkpoint (keys match exactly,
        since those submodules are unchanged). The metadata branch and
        classifier are left randomly initialized (classifier shape changed
        anyway, since fused_dim grew from 2048 -> 2080).
        """
        own_state = self.state_dict()
        loaded, skipped = [], []
        for name, param in phase1_state_dict.items():
            if name.startswith('classifier'):
                continue  # shape changed (2048 -> 2080 input), skip
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded.append(name)
            else:
                skipped.append(name)
        self.load_state_dict(own_state)
        print(f'Loaded {len(loaded)} tensors from Phase 1 checkpoint, skipped {len(skipped)}')

        if freeze_pretrained:
            for p in self.spatial_features.parameters():
                p.requires_grad = False
            for p in self.freq_branch.parameters():
                p.requires_grad = False
            print('Froze spatial_features + freq_branch. '
                  'Only meta_branch + classifier will train initially.')

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        print('Unfroze all parameters for full fine-tuning.')
