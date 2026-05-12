"""
=============================================================
STEP 3 — Dataset Loader
=============================================================
Project: Image Forgery Detection in Social Media Images
         Using Deep Learning Techniques

This file provides PyTorch Dataset and DataLoader objects
that all 3 models will use for training and evaluation.

What this does:
  - Loads images from dataset/processed/ folder
  - Applies normalization (ImageNet mean/std for pretrained models)
  - Applies data augmentation during training
  - Returns (image_tensor, label) pairs for PyTorch

Used by:
  - training/train_resnet.py      (Model 1)
  - training/train_efficientnet.py (Model 2)
  - training/train_hybrid.py      (Model 3)
  - evaluation/evaluate_all.py
=============================================================
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')


# ─────────────────────────────────────────────────────────────
# TRANSFORMS
# ImageNet mean/std normalization required for pretrained
# EfficientNet and ResNet models
# ─────────────────────────────────────────────────────────────

# Training transforms — includes augmentation to prevent overfitting
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.05),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std =[0.229, 0.224, 0.225]),   # ImageNet std
])

# Validation/Test transforms — no augmentation, just normalize
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────
# MAIN DATASET CLASS
# ─────────────────────────────────────────────────────────────

class ForgeryDataset(Dataset):
    """
    PyTorch Dataset for image forgery detection.

    Loads images from the processed/ folder structure:

      processed/
        {split}/          ← 'train', 'val', 'test'
          {version}/      ← 'clean', 'compressed',
          │                  'facebook', 'instagram',
          │                  'whatsapp', 'twitter'
          ├── authentic/  ← label 0
          └── forged/     ← label 1

    Args:
        split   : 'train', 'val', or 'test'
        version : 'clean', 'compressed', 'facebook',
                  'instagram', 'whatsapp', 'twitter',
                  or 'all' (loads all versions — used for training)
        transform: torchvision transforms to apply

    Example:
        # Training dataset (clean + compressed mixed)
        train_ds = ForgeryDataset('train', 'all', TRAIN_TRANSFORMS)

        # Test on Facebook-compressed images only
        test_fb  = ForgeryDataset('test', 'facebook', VAL_TRANSFORMS)
    """

    CLASS_NAMES = ['authentic', 'forged']

    def __init__(self, split='train', version='all', transform=None):
        self.split     = split
        self.version   = version
        self.transform = transform
        self.samples   = []   # list of (image_path, label)

        self._load_samples()

    # ----------------------------------------------------------
    def _load_samples(self):
        """Scan processed/ folder and collect all image paths."""

        processed_dir = PROCESSED_DIR

        if not os.path.exists(processed_dir):
            raise FileNotFoundError(
                f'Processed dataset not found at: {processed_dir}\n'
                f'Please run: python preprocessing/prepare_dataset.py'
            )

        split_dir = os.path.join(processed_dir, self.split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f'Split folder not found: {split_dir}'
            )

        # Decide which versions to load
        if self.version == 'all':
            # Load ALL available versions for this split
            versions = [
                d for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ]
        else:
            versions = [self.version]

        IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

        for ver in versions:
            ver_dir = os.path.join(split_dir, ver)
            if not os.path.exists(ver_dir):
                continue

            for label_idx, label_name in enumerate(self.CLASS_NAMES):
                label_dir = os.path.join(ver_dir, label_name)
                if not os.path.exists(label_dir):
                    continue

                for fname in os.listdir(label_dir):
                    if Path(fname).suffix.lower() in IMG_EXTS:
                        self.samples.append((
                            os.path.join(label_dir, fname),
                            label_idx   # 0=authentic, 1=forged
                        ))

        if len(self.samples) == 0:
            raise ValueError(
                f'No images found for split={self.split}, '
                f'version={self.version} in {processed_dir}'
            )

    # ----------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ----------------------------------------------------------
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            # Return a black image if file is corrupt
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

    # ----------------------------------------------------------
    def class_counts(self):
        """Returns (num_authentic, num_forged) for this dataset."""
        labels   = [l for _, l in self.samples]
        auth     = labels.count(0)
        forged   = labels.count(1)
        return auth, forged

    # ----------------------------------------------------------
    def __repr__(self):
        auth, forg = self.class_counts()
        return (f'ForgeryDataset('
                f'split={self.split}, '
                f'version={self.version}, '
                f'authentic={auth}, '
                f'forged={forg}, '
                f'total={len(self.samples)})')


# ─────────────────────────────────────────────────────────────
# DATALOADER FACTORY FUNCTIONS
# These are what the training scripts call
# ─────────────────────────────────────────────────────────────

def get_train_loader(batch_size=32, num_workers=4):
    """
    Training DataLoader.
    Loads BOTH clean and compressed images mixed together.
    This is what makes our model robust — it trains on
    both clean and social-media-compressed images.

    Args:
        batch_size  : images per batch (32 recommended)
        num_workers : parallel loading workers
                      (use 0 on Windows if you get errors)
    """
    dataset = ForgeryDataset(
        split='train',
        version='all',          # clean + compressed mixed
        transform=TRAIN_TRANSFORMS
    )
    auth, forg = dataset.class_counts()
    print(f'Train loader: {len(dataset)} images '
          f'(auth={auth}, forged={forg})')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


def get_val_loader(batch_size=32, num_workers=4):
    """
    Validation DataLoader.
    Uses clean images only for validation.
    """
    dataset = ForgeryDataset(
        split='val',
        version='clean',
        transform=VAL_TRANSFORMS
    )
    auth, forg = dataset.class_counts()
    print(f'Val loader:   {len(dataset)} images '
          f'(auth={auth}, forged={forg})')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


def get_test_loader(version='clean', batch_size=32, num_workers=4):
    """
    Test DataLoader for a specific platform version.

    Args:
        version : 'clean', 'facebook', 'instagram',
                  'whatsapp', or 'twitter'

    This is called 5 times in evaluate_all.py — once per
    version — to build the comparison table showing how
    each model performs on each social media platform.
    """
    dataset = ForgeryDataset(
        split='test',
        version=version,
        transform=VAL_TRANSFORMS
    )
    auth, forg = dataset.class_counts()
    print(f'Test loader ({version:<10}): {len(dataset)} images '
          f'(auth={auth}, forged={forg})')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


def get_all_test_loaders(batch_size=32, num_workers=4):
    """
    Returns a dict of test loaders for ALL platforms.
    Used in evaluate_all.py to build the comparison table.

    Returns:
        {
          'clean'    : DataLoader,
          'facebook' : DataLoader,
          'instagram': DataLoader,
          'whatsapp' : DataLoader,
          'twitter'  : DataLoader,
        }
    """
    versions = ['clean', 'facebook', 'instagram', 'whatsapp', 'twitter']
    loaders  = {}
    for v in versions:
        loaders[v] = get_test_loader(v, batch_size, num_workers)
    return loaders


# ─────────────────────────────────────────────────────────────
# SELF TEST — run: python dataset_loader.py
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('STEP 3 — Dataset Loader Self-Test')
    print('=' * 60)

    # Load metadata
    meta_path = os.path.join(PROCESSED_DIR, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f'\nDataset metadata:')
        for k, v in meta.items():
            print(f'  {k:<20}: {v}')

    print('\nTesting dataset loading...\n')

    # Test train dataset
    try:
        train_ds = ForgeryDataset('train', 'all', TRAIN_TRANSFORMS)
        print(f'  {train_ds}')

        # Test one batch
        img, lbl = train_ds[0]
        print(f'  Sample image tensor : shape={img.shape}  '
              f'dtype={img.dtype}')
        print(f'  Sample label        : {lbl.item()} '
              f'({ForgeryDataset.CLASS_NAMES[lbl.item()]})')
    except Exception as e:
        print(f'  Train dataset error: {e}')

    print()

    # Test val dataset
    try:
        val_ds = ForgeryDataset('val', 'clean', VAL_TRANSFORMS)
        print(f'  {val_ds}')
    except Exception as e:
        print(f'  Val dataset error: {e}')

    print()

    # Test all test versions
    for ver in ['clean', 'facebook', 'instagram', 'whatsapp', 'twitter']:
        try:
            ds = ForgeryDataset('test', ver, VAL_TRANSFORMS)
            print(f'  {ds}')
        except Exception as e:
            print(f'  Test [{ver}] error: {e}')

    # Test DataLoader
    print('\nTesting DataLoader (one batch)...')
    try:
        loader = get_train_loader(batch_size=8, num_workers=0)
        images, labels = next(iter(loader))
        print(f'  Batch images shape : {images.shape}')
        print(f'  Batch labels shape : {labels.shape}')
        print(f'  Labels in batch    : {labels.tolist()}')
    except Exception as e:
        print(f'  DataLoader error: {e}')

    print()
    print('=' * 60)
    print('Step 3 COMPLETE ✅ — dataset_loader.py is ready')
    print('Next: python models/resnet_baseline.py')
    print('=' * 60)
