"""
=============================================================
STEP 1 — Social Media Compression Simulator
=============================================================
Project: Image Forgery Detection in Social Media Images
         Using Deep Learning Techniques

This file simulates what happens to an image when it is
uploaded to Facebook, Instagram, WhatsApp, or Twitter.

WHY THIS IS THE MOST IMPORTANT FILE IN OUR PROJECT:
----------------------------------------------------
Normal forgery detection models are trained on clean images.
When those images get shared on social media, the platform
automatically compresses, resizes, and filters them.
This DESTROYS the forensic traces — and normal models FAIL.

Our model is trained WITH these compression effects applied,
so it learns to detect forgeries even after social media
processing. This is what makes our project different.

Reference: Wu et al., CVPR 2022
=============================================================
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
import random


# ─────────────────────────────────────────────────────────────
# CORE HELPER: JPEG compress an image in memory
# ─────────────────────────────────────────────────────────────
def jpeg_compress(image_np, quality):
    """
    Compress a numpy image using JPEG at a given quality level.

    quality: 0 = worst quality (most compressed)
             100 = best quality (least compressed)

    This is the single biggest thing social media does to images.
    It destroys pixel-level forensic traces left by editing tools.

    Args:
        image_np : numpy array (H x W x 3), uint8, RGB
        quality  : int between 0 and 100

    Returns:
        numpy array (H x W x 3), uint8, RGB — compressed image
    """
    pil_img = Image.fromarray(image_np.astype(np.uint8))
    buffer  = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=int(quality))
    buffer.seek(0)
    return np.array(Image.open(buffer))


# ─────────────────────────────────────────────────────────────
# FACEBOOK SIMULATION
# Based on Wu et al. (CVPR 2022) analysis of Facebook pipeline
# ─────────────────────────────────────────────────────────────
def simulate_facebook(image_np):
    """
    Simulates Facebook's image processing pipeline.

    Facebook does 3 things to every uploaded image:
      1. Resize  — if image is larger than 2048px on any side
      2. Enhancement filtering — secret sharpening filter
         (Wu et al. discovered this through reverse engineering)
      3. JPEG compression — quality factor between 71 and 95
         (adaptive, depends on image content)

    After this pipeline, IoU of existing forgery detectors
    drops by ~10% (Wu et al. Table 1).

    Our model trains WITH this — so it stays robust.
    """
    img = image_np.copy()
    h, w = img.shape[:2]

    # Step 1: Resize if larger than 2048px
    if max(h, w) > 2048:
        scale = 2048 / max(h, w)
        img   = cv2.resize(img,
                           (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    # Step 2: Facebook-style enhancement (sharpening)
    pil_img  = Image.fromarray(img.astype(np.uint8))
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img  = enhancer.enhance(random.uniform(1.0, 1.3))

    # Occasional slight blur before sharpening (Facebook behavior)
    if random.random() < 0.3:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))

    img = np.array(pil_img)

    # Step 3: JPEG compression QF 71-95 (Facebook range per Wu et al.)
    img = jpeg_compress(img, random.randint(71, 95))

    return img


# ─────────────────────────────────────────────────────────────
# INSTAGRAM SIMULATION
# ─────────────────────────────────────────────────────────────
def simulate_instagram(image_np):
    """
    Simulates Instagram's image processing pipeline.

    Instagram does:
      1. Resize to max 1080px width
      2. Slight color enhancement (makes images more vivid)
      3. JPEG compression at quality 70-85

    Instagram is slightly more aggressive than Facebook
    in terms of color processing.
    """
    img = image_np.copy()
    h, w = img.shape[:2]

    # Step 1: Resize to max 1080px
    if max(h, w) > 1080:
        scale = 1080 / max(h, w)
        img   = cv2.resize(img,
                           (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    # Step 2: Color enhancement
    pil_img = Image.fromarray(img.astype(np.uint8))
    if random.random() < 0.5:
        pil_img = ImageEnhance.Color(pil_img).enhance(
                      random.uniform(1.0, 1.2))
    img = np.array(pil_img)

    # Step 3: JPEG compression QF 70-85
    img = jpeg_compress(img, random.randint(70, 85))

    return img


# ─────────────────────────────────────────────────────────────
# WHATSAPP SIMULATION
# ─────────────────────────────────────────────────────────────
def simulate_whatsapp(image_np):
    """
    Simulates WhatsApp's image processing pipeline.

    WhatsApp is the MOST aggressive compressor:
      1. Resize to max 1600px
      2. Heavy JPEG compression at quality 50-80
      3. 40% chance of double compression (images get
         forwarded multiple times — very common on WhatsApp)

    After WhatsApp, forensic traces are almost entirely gone.
    This is the hardest test for our model.
    """
    img = image_np.copy()
    h, w = img.shape[:2]

    # Step 1: Resize to max 1600px
    if max(h, w) > 1600:
        scale = 1600 / max(h, w)
        img   = cv2.resize(img,
                           (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    # Step 2: Aggressive JPEG compression QF 50-80
    img = jpeg_compress(img, random.randint(50, 80))

    # Step 3: Double compression (forwarded images)
    if random.random() < 0.4:
        img = jpeg_compress(img, random.randint(50, 75))

    return img


# ─────────────────────────────────────────────────────────────
# TWITTER SIMULATION
# ─────────────────────────────────────────────────────────────
def simulate_twitter(image_np):
    """
    Simulates Twitter/X's image processing pipeline.

    Twitter does:
      1. Resize to max 1200px
      2. JPEG compression at quality 80-90
         (Twitter uses WebP internally — we approximate with JPEG)

    Twitter is less aggressive than WhatsApp but still
    damages forensic traces significantly.
    """
    img = image_np.copy()
    h, w = img.shape[:2]

    # Step 1: Resize to max 1200px
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img   = cv2.resize(img,
                           (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    # Step 2: JPEG compression QF 80-90
    img = jpeg_compress(img, random.randint(80, 90))

    return img


# ─────────────────────────────────────────────────────────────
# MAIN SIMULATOR CLASS
# This is what all other files import and use
# ─────────────────────────────────────────────────────────────
class SocialMediaSimulator:
    """
    Main class for applying social media compression.

    Used in THREE places:
      1. preprocessing/prepare_dataset.py
         → Creates compressed versions of training data
      2. training/train_hybrid.py
         → Applied on-the-fly during training (data augmentation)
      3. web_app/app.py
         → Tests uploaded images through each platform

    Usage:
        simulator = SocialMediaSimulator()

        # Single platform
        compressed, platform = simulator.apply(image, 'facebook')

        # Random platform (used during training)
        compressed, platform = simulator.apply(image, 'random')

        # All platforms (used for comparison table)
        results = simulator.apply_all_platforms(image)
    """

    PLATFORMS = ['facebook', 'instagram', 'whatsapp', 'twitter']

    def __init__(self):
        self._fns = {
            'facebook' : simulate_facebook,
            'instagram': simulate_instagram,
            'whatsapp' : simulate_whatsapp,
            'twitter'  : simulate_twitter,
        }

    # ----------------------------------------------------------
    def apply(self, image_np, platform='random'):
        """
        Apply one platform's compression to an image.

        Args:
            image_np : numpy array (H x W x 3), uint8, RGB
            platform : 'facebook' | 'instagram' |
                       'whatsapp' | 'twitter' | 'random'
        Returns:
            (compressed_image_np, platform_name_used)
        """
        # Ensure uint8 RGB
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

        # Ensure 3-channel RGB (handle grayscale inputs)
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 1:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

        if platform == 'random':
            platform = random.choice(self.PLATFORMS)

        if platform not in self._fns:
            raise ValueError(
                f"Unknown platform '{platform}'. "
                f"Choose from: {self.PLATFORMS}"
            )

        return self._fns[platform](image_np), platform

    # ----------------------------------------------------------
    def apply_all_platforms(self, image_np):
        """
        Apply ALL platforms to one image.
        Returns dict: { 'facebook': img, 'instagram': img, ... }

        Used for:
          - Building the comparison table (evaluate_all.py)
          - Website results page (shows all platform results)
        """
        # Ensure uint8 RGB
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

        return {
            platform: fn(image_np.copy())
            for platform, fn in self._fns.items()
        }

    # ----------------------------------------------------------
    def batch_apply(self, images, platform='random'):
        """
        Apply compression to a list of images.
        Used during dataset preparation.

        Args:
            images   : list of numpy arrays
            platform : platform name or 'random'
        Returns:
            (list of compressed images, list of platform names)
        """
        compressed, platforms = [], []
        for img in images:
            c, p = self.apply(img, platform)
            compressed.append(c)
            platforms.append(p)
        return compressed, platforms


# ─────────────────────────────────────────────────────────────
# QUICK SELF-TEST — run: python simulate_compression.py
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('STEP 1 — Social Media Compression Simulator')
    print('Self-test running...')
    print('=' * 60)

    # Create a dummy 512x512 colour test image
    test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    sim      = SocialMediaSimulator()

    print(f'\nOriginal image  : shape={test_img.shape}  '
          f'dtype={test_img.dtype}')
    print()

    # Test every platform
    for plat in SocialMediaSimulator.PLATFORMS:
        out, _ = sim.apply(test_img.copy(), plat)
        print(f'  {plat:<12} → shape={out.shape}  '
              f'dtype={out.dtype}')

    # Test random selection
    print()
    print('Random platform selection (5 runs):')
    for i in range(5):
        _, p = sim.apply(test_img.copy(), 'random')
        print(f'  Run {i+1}: {p}')

    # Test apply_all_platforms
    print()
    print('apply_all_platforms:')
    results = sim.apply_all_platforms(test_img.copy())
    for p, img in results.items():
        print(f'  {p:<12} → shape={img.shape}')

    # Test grayscale input handling
    print()
    gray_img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    out, p   = sim.apply(gray_img, 'facebook')
    print(f'Grayscale input (128x128) → output shape={out.shape} '
          f'[grayscale→RGB handled ✓]')

    print()
    print('=' * 60)
    print('Step 1 COMPLETE ✅ — simulate_compression.py is ready')
    print('Next: python preprocessing/prepare_dataset.py')
    print('=' * 60)
