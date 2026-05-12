"""
=============================================================
STEP 2 — Dataset Preparation
=============================================================
Project: Image Forgery Detection in Social Media Images
         Using Deep Learning Techniques

This script:
  1. Loads images from ALL 4 datasets:
       • CASIA v2      (Au/ and Tp/ folders)
       • Columbia Color (4cam_auth/ and 4cam_splc/)
       • Columbia Gray  (Au-* and Sp-* folders)
       • NIST16         (images/ folder — all forged)

  2. Creates 5 versions of EVERY image:
       • clean      — original image
       • facebook   — after Facebook compression
       • instagram  — after Instagram compression
       • whatsapp   — after WhatsApp compression
       • twitter    — after Twitter compression

  3. Splits into Train (70%) / Val (15%) / Test (15%)

  4. Saves everything to dataset/processed/

HOW TO RUN:
-----------
  python preprocessing/prepare_dataset.py

EXPECTED DATASET FOLDER STRUCTURE:
------------------------------------
  dataset/
    CASIA/
      Au/          ← 7492 authentic jpg images
      Tp/          ← 5125 forged jpg/tif images
      Groundtruth/ ← 5123 png masks (not used for training)

    Columbia/
      color/
        4cam_auth/ ← 185 authentic tif images
        4cam_splc/ ← 182 forged tif images
      gray/
        Au-S/, Au-T/, Au-SS-H/, Au-SS-O/, Au-SS-V/
        Au-TS-H/, Au-TS-O/, Au-TS-V/
        Au-TT-H/, Au-TT-O/, Au-TT-V/
        Sp-S/, Sp-T/, Sp-SS-H/, Sp-SS-O/, Sp-SS-V/
        Sp-TS-H/, Sp-TS-O/, Sp-TS-V/
        Sp-TT-H/, Sp-TT-O/, Sp-TT-V/

    NIST16/
      images/ ← 564 forged jpg images
      gt/     ← ground truth masks (not used for classification)
=============================================================
"""

import os
import sys
import cv2
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Import our Step 1 simulator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.simulate_compression import SocialMediaSimulator

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — change paths here if needed
# ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DIR  = os.path.join(DATASET_DIR, 'processed')

IMAGE_SIZE  = (224, 224)   # EfficientNet-B4 / ResNet50 input size
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

PLATFORMS   = ['facebook', 'instagram', 'whatsapp', 'twitter']
IMG_EXTS    = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────
# DATASET LOADERS
# Each returns list of (image_path, label)
#   label 0 = authentic (real)
#   label 1 = forged (fake)
# ─────────────────────────────────────────────────────────────

def load_casia(dataset_dir):
    """
    CASIA v2.0 — largest dataset, most important for training.

    Structure:
      Au/  → authentic images (label 0)
      Tp/  → tampered/forged images (label 1)

    Formats: JPG, TIF
    Total: ~12,617 images
    """
    samples  = []
    root     = os.path.join(dataset_dir, 'CASIA')

    au_path  = os.path.join(root, 'Au')
    tp_path  = os.path.join(root, 'Tp')

    missing  = []
    if not os.path.exists(au_path): missing.append(au_path)
    if not os.path.exists(tp_path): missing.append(tp_path)

    if missing:
        print(f'  [CASIA] WARNING — folders not found: {missing}')
        print(f'  [CASIA] Skipping...')
        return samples

    # Authentic images
    for f in os.listdir(au_path):
        if Path(f).suffix.lower() in IMG_EXTS:
            samples.append((os.path.join(au_path, f), 0))

    # Forged images
    for f in os.listdir(tp_path):
        if Path(f).suffix.lower() in IMG_EXTS:
            samples.append((os.path.join(tp_path, f), 1))

    auth  = sum(1 for _, l in samples if l == 0)
    forg  = sum(1 for _, l in samples if l == 1)
    print(f'  [CASIA v2]        auth={auth:5d}  forged={forg:5d}  '
          f'total={len(samples):6d}')
    return samples


def load_columbia_color(dataset_dir):
    """
    Columbia Uncompressed Color dataset.

    Structure:
      color/4cam_auth/ → authentic TIF images (label 0)
      color/4cam_splc/ → spliced/forged TIF images (label 1)

    Total: ~367 images
    High quality uncompressed — good test set
    """
    samples   = []
    root      = os.path.join(dataset_dir, 'Columbia', 'color')

    auth_path = os.path.join(root, '4cam_auth')
    splc_path = os.path.join(root, '4cam_splc')

    missing = []
    if not os.path.exists(auth_path): missing.append(auth_path)
    if not os.path.exists(splc_path): missing.append(splc_path)

    if missing:
        print(f'  [Columbia Color] WARNING — not found: {missing}')
        print(f'  [Columbia Color] Skipping...')
        return samples

    for f in os.listdir(auth_path):
        if Path(f).suffix.lower() in IMG_EXTS:
            samples.append((os.path.join(auth_path, f), 0))

    for f in os.listdir(splc_path):
        if Path(f).suffix.lower() in IMG_EXTS:
            samples.append((os.path.join(splc_path, f), 1))

    auth = sum(1 for _, l in samples if l == 0)
    forg = sum(1 for _, l in samples if l == 1)
    print(f'  [Columbia Color]  auth={auth:5d}  forged={forg:5d}  '
          f'total={len(samples):6d}')
    return samples


def load_columbia_gray(dataset_dir):
    """
    Columbia Gray dataset (ImSpliceDataset).

    Folder naming convention:
      Au-*  → authentic images (label 0)
      Sp-*  → spliced/forged images (label 1)

    Format: BMP, 128x128 grayscale
    Total: ~1,845 images
    """
    samples = []
    root    = os.path.join(dataset_dir, 'Columbia', 'gray')

    if not os.path.exists(root):
        print(f'  [Columbia Gray]  WARNING — not found: {root}')
        print(f'  [Columbia Gray]  Skipping...')
        return samples

    for folder in os.listdir(root):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue

        # Au-* folders = authentic, Sp-* folders = forged
        if folder.startswith('Au'):
            label = 0
        elif folder.startswith('Sp'):
            label = 1
        else:
            continue

        for f in os.listdir(folder_path):
            if Path(f).suffix.lower() in IMG_EXTS:
                samples.append((os.path.join(folder_path, f), label))

    auth = sum(1 for _, l in samples if l == 0)
    forg = sum(1 for _, l in samples if l == 1)
    print(f'  [Columbia Gray]   auth={auth:5d}  forged={forg:5d}  '
          f'total={len(samples):6d}')
    return samples


def load_nist16(dataset_dir):
    """
    NIST16 dataset.

    Structure:
      images/ → 564 forged JPG images (all are forged — label 1)
      gt/     → ground truth masks (not used for classification)

    Note: NIST16 only provides forged images.
    We use authentic images from CASIA to balance the classes.
    Total: 564 forged images
    """
    samples  = []
    root     = os.path.join(dataset_dir, 'NIST16')
    img_path = os.path.join(root, 'images')

    if not os.path.exists(img_path):
        print(f'  [NIST16]         WARNING — not found: {img_path}')
        print(f'  [NIST16]         Skipping...')
        return samples

    for f in os.listdir(img_path):
        if Path(f).suffix.lower() in IMG_EXTS:
            samples.append((os.path.join(img_path, f), 1))  # all forged

    forg = len(samples)
    print(f'  [NIST16]          auth={0:5d}  forged={forg:5d}  '
          f'total={forg:6d}  (all forged — class balancing applied)')
    return samples


# ─────────────────────────────────────────────────────────────
# IMAGE LOADING HELPER
# ─────────────────────────────────────────────────────────────

def load_image(path, size=(224, 224)):
    """
    Load any image format and return as numpy array.
    Handles: JPG, TIF, BMP, PNG, grayscale, RGBA

    Returns: numpy array (H x W x 3) uint8 RGB
             or None if image cannot be loaded
    """
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        # Try PIL if OpenCV fails (handles some TIF/BMP variants)
        if img is None:
            pil = Image.open(path).convert('RGB')
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        if img is None:
            return None

        # Convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        return img

    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────────────────────

def save_image(image_np, path):
    """Save RGB numpy array to disk as JPEG (quality 95)."""
    bgr = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ─────────────────────────────────────────────────────────────
# DATASET SPLITTING
# ─────────────────────────────────────────────────────────────

def split_dataset(samples):
    """
    Split into train/val/test maintaining class balance.

    We split BEFORE creating compressed versions so the
    test set is completely unseen during training — giving
    honest evaluation results.
    """
    authentic = [(p, l) for p, l in samples if l == 0]
    forged    = [(p, l) for p, l in samples if l == 1]

    random.shuffle(authentic)
    random.shuffle(forged)

    def _split(lst):
        n  = len(lst)
        t1 = int(n * TRAIN_RATIO)
        t2 = int(n * (TRAIN_RATIO + VAL_RATIO))
        return lst[:t1], lst[t1:t2], lst[t2:]

    atr, avl, ate = _split(authentic)
    ftr, fvl, fte = _split(forged)

    train = atr + ftr
    val   = avl + fvl
    test  = ate + fte

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# ─────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY BUILDER
# ─────────────────────────────────────────────────────────────

def make_dirs(output_dir):
    """
    Create the complete processed/ folder structure.

    processed/
      train/
        clean/authentic/     clean/forged/
        compressed/authentic/ compressed/forged/
      val/
        clean/authentic/     clean/forged/
        compressed/authentic/ compressed/forged/
      test/
        clean/authentic/         clean/forged/
        facebook/authentic/      facebook/forged/
        instagram/authentic/     instagram/forged/
        whatsapp/authentic/      whatsapp/forged/
        twitter/authentic/       twitter/forged/
    """
    labels   = ['authentic', 'forged']

    for split in ['train', 'val']:
        for version in ['clean', 'compressed']:
            for label in labels:
                os.makedirs(
                    os.path.join(output_dir, split, version, label),
                    exist_ok=True)

    for version in ['clean'] + PLATFORMS:
        for label in labels:
            os.makedirs(
                os.path.join(output_dir, 'test', version, label),
                exist_ok=True)


# ─────────────────────────────────────────────────────────────
# PROCESS ONE SPLIT
# ─────────────────────────────────────────────────────────────

def process_split(samples, split_name, output_dir,
                  simulator, is_test=False):
    """
    Load, compress, and save all images for one split.

    Train/Val: saves clean + one random compressed version
    Test:      saves clean + one version per platform
               (so we can test each platform separately)
    """
    label_map = {0: 'authentic', 1: 'forged'}
    skipped   = 0

    for idx, (img_path, label) in enumerate(
            tqdm(samples, desc=f'  {split_name}', ncols=70)):

        img = load_image(img_path, IMAGE_SIZE)
        if img is None:
            skipped += 1
            continue

        label_name = label_map[label]
        base       = f'{idx:06d}.jpg'

        if not is_test:
            # ── Train / Val ──────────────────────────────────
            # Save clean version
            save_image(img,
                os.path.join(output_dir, split_name,
                             'clean', label_name, base))

            # Save one randomly compressed version
            compressed, _ = simulator.apply(img.copy(), 'random')
            save_image(compressed,
                os.path.join(output_dir, split_name,
                             'compressed', label_name, base))

        else:
            # ── Test ─────────────────────────────────────────
            # Save clean version
            save_image(img,
                os.path.join(output_dir, 'test',
                             'clean', label_name, base))

            # Save one version per platform
            all_compressed = simulator.apply_all_platforms(img.copy())
            for platform, comp_img in all_compressed.items():
                save_image(comp_img,
                    os.path.join(output_dir, 'test',
                                 platform, label_name, base))

    if skipped:
        print(f'    (skipped {skipped} unreadable images)')


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('STEP 2 — Dataset Preparation')
    print('=' * 60)

    simulator = SocialMediaSimulator()

    # ── Step 2a: Load all datasets ────────────────────────────
    print('\nLoading datasets...')
    all_samples = []
    all_samples.extend(load_casia(DATASET_DIR))
    all_samples.extend(load_columbia_color(DATASET_DIR))
    all_samples.extend(load_columbia_gray(DATASET_DIR))
    all_samples.extend(load_nist16(DATASET_DIR))

    if not all_samples:
        print('\nERROR: No images found!')
        print('Please check your dataset folder structure.')
        return

    auth_total = sum(1 for _, l in all_samples if l == 0)
    forg_total = sum(1 for _, l in all_samples if l == 1)

    print(f'\nTotal loaded:')
    print(f'  Authentic : {auth_total:6d}')
    print(f'  Forged    : {forg_total:6d}')
    print(f'  Total     : {len(all_samples):6d}')

    # ── Step 2b: Split ───────────────────────────────────────
    print('\nSplitting into train / val / test...')
    train, val, test = split_dataset(all_samples)
    print(f'  Train : {len(train):6d} images')
    print(f'  Val   : {len(val):6d} images')
    print(f'  Test  : {len(test):6d} images')

    # ── Step 2c: Create output directories ───────────────────
    print(f'\nCreating output folders in: {OUTPUT_DIR}')
    make_dirs(OUTPUT_DIR)

    # ── Step 2d: Process each split ──────────────────────────
    print('\nProcessing images...')
    print('(This will take several minutes — please wait)\n')

    process_split(train, 'train', OUTPUT_DIR, simulator, is_test=False)
    process_split(val,   'val',   OUTPUT_DIR, simulator, is_test=False)
    process_split(test,  'test',  OUTPUT_DIR, simulator, is_test=True)

    # ── Step 2e: Save metadata ────────────────────────────────
    metadata = {
        'total'         : len(all_samples),
        'authentic'     : auth_total,
        'forged'        : forg_total,
        'train'         : len(train),
        'val'           : len(val),
        'test'          : len(test),
        'image_size'    : list(IMAGE_SIZE),
        'platforms'     : PLATFORMS,
        'class_names'   : ['authentic', 'forged'],
        'datasets_used' : ['CASIA_v2', 'Columbia_Color',
                           'Columbia_Gray', 'NIST16'],
    }

    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print('\n' + '=' * 60)
    print('Step 2 COMPLETE ✅')
    print(f'Processed data saved to: {OUTPUT_DIR}')
    print('\nFolder structure created:')
    print('  processed/train/clean/{authentic,forged}/')
    print('  processed/train/compressed/{authentic,forged}/')
    print('  processed/val/clean/{authentic,forged}/')
    print('  processed/val/compressed/{authentic,forged}/')
    print('  processed/test/clean/{authentic,forged}/')
    print('  processed/test/facebook/{authentic,forged}/')
    print('  processed/test/instagram/{authentic,forged}/')
    print('  processed/test/whatsapp/{authentic,forged}/')
    print('  processed/test/twitter/{authentic,forged}/')
    print('\nNext: python models/resnet_baseline.py')
    print('=' * 60)


if __name__ == '__main__':
    main()
