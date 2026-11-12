"""
=============================================================
Phase 2 — Train Hybrid Model V2 (with EXIF branch)
=============================================================
Run this on Colab/Kaggle (same setup as your Phase 1 training).

WHAT THIS DOES:
  1. Loads your existing CASIA/Columbia/NIST16 dataset (same folder
     structure you used in Phase 1 — EDIT DATASET_ROOT below)
  2. For each image: loads pixels (same TRANSFORM as Phase 1) AND
     extracts its EXIF feature vector
  3. Loads your Phase 1 hybrid_model.pth weights into the spatial +
     frequency branches (frozen), trains metadata branch + classifier
  4. After a few epochs, unfreezes everything for full fine-tuning
  5. Saves hybrid_model_v2_exif.pth

NOTE ON YOUR DATA: your original datasets (CASIA v2.0, Columbia,
NIST16) are mostly uncompressed/lightly-compressed benchmark images
that were never posted to social media, so most will have LITTLE OR
NO real EXIF (scanners/tools strip it, or it was never added).
That's fine — it means the model mainly learns from the "no EXIF"
vs "has EXIF" signal itself plus whatever real cases exist, which is
exactly the realistic distribution (since social media strips EXIF
too). Don't try to synthetically inject fake EXIF into clean training
images — that would teach the model a false correlation.
=============================================================
"""

import os, glob, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from exif_utils import extract_exif_features, EXIF_FEATURE_DIM
from models_v2 import HybridModelV2

# ---------------- CONFIG — EDIT THESE ----------------
DATASET_ROOT   = '/kaggle/input/datasets/rubyjane212005/forgery-processed/'  # same as Phase 1
PHASE1_CKPT    = '/kaggle/input/phase1-models/hybrid_model.pth'              # your existing hybrid weights
OUTPUT_CKPT    = '/kaggle/working/hybrid_model_v2_exif.pth'
BATCH_SIZE     = 32
EPOCHS_FROZEN  = 5   # train only meta_branch + classifier
EPOCHS_FULL    = 10  # then unfreeze and fine-tune everything
LR_FROZEN      = 1e-3
LR_FULL        = 1e-5
NUM_WORKERS    = 0   # keep 0 on Windows; can raise on Colab/Kaggle Linux
# -------------------------------------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ForgeryExifDataset(Dataset):
    """
    Expects the same folder layout you used in Phase 1:
        DATASET_ROOT/authentic/*.jpg
        DATASET_ROOT/forged/*.jpg
    Adjust `_load_paths` if your layout differs (e.g. separate per-dataset
    folders for CASIA/Columbia/NIST16 — just extend the glob patterns).
    """
    def __init__(self, root, split='train', transform=None):
        self.transform = transform
        self.samples = self._load_paths(root)
        random.seed(42)
        random.shuffle(self.samples)
        n = len(self.samples)
        if split == 'train':
            self.samples = self.samples[:int(0.7 * n)]
        elif split == 'val':
            self.samples = self.samples[int(0.7 * n):int(0.85 * n)]
        else:
            self.samples = self.samples[int(0.85 * n):]

    def _load_paths(self, root):
        samples = []
        for label, folder in [(0, 'authentic'), (1, 'forged')]:
            pattern = os.path.join(root, folder, '**', '*.*')
            for path in glob.glob(pattern, recursive=True):
                if path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    samples.append((path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            pil_img = Image.open(path).convert('RGB')
        except Exception:
            # Corrupt file fallback — return a black image, label unchanged
            pil_img = Image.new('RGB', (224, 224))
        # Extract from the file path (not the already-opened PIL object) so
        # raw EXIF bytes are read fresh and reliably.
        exif_vec = extract_exif_features(path)
        tensor = self.transform(pil_img) if self.transform else pil_img
        return tensor, torch.from_numpy(exif_vec), label


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, exifs, labels in loader:
            imgs, exifs = imgs.to(DEVICE), exifs.to(DEVICE)
            out = model(imgs, exifs)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return {
        'accuracy' : accuracy_score(all_labels, all_preds),
        'f1'       : f1_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall'   : recall_score(all_labels, all_preds),
    }


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for imgs, exifs, labels in loader:
        imgs, exifs, labels = imgs.to(DEVICE), exifs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs, exifs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def main():
    train_ds = ForgeryExifDataset(DATASET_ROOT, 'train', TRANSFORM)
    val_ds   = ForgeryExifDataset(DATASET_ROOT, 'val', TRANSFORM)
    test_ds  = ForgeryExifDataset(DATASET_ROOT, 'test', TRANSFORM)
    print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = HybridModelV2().to(DEVICE)

    phase1_ckpt = torch.load(PHASE1_CKPT, map_location=DEVICE)
    phase1_state = phase1_ckpt.get('model_state', phase1_ckpt)
    model.load_phase1_weights(phase1_state, freeze_pretrained=True)

    criterion = nn.CrossEntropyLoss()

    # ---- Stage 1: train only meta_branch + classifier ----
    print('\n=== Stage 1: training metadata branch + classifier (frozen backbone) ===')
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR_FROZEN)
    for epoch in range(EPOCHS_FROZEN):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        metrics = evaluate(model, val_loader)
        print(f'[Stage1 Epoch {epoch+1}/{EPOCHS_FROZEN}] loss={loss:.4f} '
              f'val_acc={metrics["accuracy"]:.4f} val_f1={metrics["f1"]:.4f}')

    # ---- Stage 2: unfreeze, full fine-tune at low LR ----
    print('\n=== Stage 2: full fine-tuning ===')
    model.unfreeze_all()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FULL)
    best_val_acc = 0.0
    for epoch in range(EPOCHS_FULL):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        metrics = evaluate(model, val_loader)
        print(f'[Stage2 Epoch {epoch+1}/{EPOCHS_FULL}] loss={loss:.4f} '
              f'val_acc={metrics["accuracy"]:.4f} val_f1={metrics["f1"]:.4f}')
        if metrics['accuracy'] > best_val_acc:
            best_val_acc = metrics['accuracy']
            torch.save({'model_state': model.state_dict()}, OUTPUT_CKPT)
            print(f'  -> New best val_acc {best_val_acc:.4f}, saved to {OUTPUT_CKPT}')

    # ---- Final test evaluation ----
    print('\n=== Final test evaluation (best checkpoint) ===')
    ck = torch.load(OUTPUT_CKPT, map_location=DEVICE)
    model.load_state_dict(ck['model_state'])
    test_metrics = evaluate(model, test_loader)
    print(test_metrics)


if __name__ == '__main__':
    main()
