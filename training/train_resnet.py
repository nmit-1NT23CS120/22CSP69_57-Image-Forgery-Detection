"""
=============================================================
STEP 7a — Train Model 1: ResNet50 Baseline
=============================================================
Run this in Google Colab cell by cell.

This trains the BASELINE model on CLEAN images only.
Purpose: prove existing models fail on social media images.
=============================================================
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# CONFIG — adjust if needed
# ─────────────────────────────────────────────────────────────
PROJECT    = '/content/drive/MyDrive/image-forgery-detection'
DATA_DIR   = os.path.join(PROJECT, 'dataset', 'processed')
SAVE_DIR   = os.path.join(PROJECT, 'saved_models')
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS     = 15
LR         = 0.0001
NUM_WORKERS= 2
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device  : {DEVICE}')
print(f'Data    : {DATA_DIR}')
print(f'Save to : {SAVE_DIR}')

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

class ForgeryDataset(Dataset):
    def __init__(self, split, version, transform):
        self.transform = transform
        self.samples   = []
        split_dir = os.path.join(DATA_DIR, split, version)
        for label_idx, label_name in enumerate(['authentic', 'forged']):
            label_dir = os.path.join(split_dir, label_name)
            if not os.path.exists(label_dir):
                continue
            for f in os.listdir(label_dir):
                if Path(f).suffix.lower() in IMG_EXTS:
                    self.samples.append(
                        (os.path.join(label_dir, f), label_idx))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224))
        return self.transform(img), torch.tensor(label, dtype=torch.long)

TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ResNet baseline trains on CLEAN images only
# (this is why it will fail on social media images)
train_ds = ForgeryDataset('train', 'clean',      TRAIN_TF)
val_ds   = ForgeryDataset('val',   'clean',      VAL_TF)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)

print(f'Train : {len(train_ds)} images')
print(f'Val   : {len(val_ds)}   images')

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
backbone_layers = list(backbone.children())[:-1]

model = nn.Sequential(
    nn.Sequential(*backbone_layers),
    nn.Flatten(),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2)
).to(DEVICE)

print(f'\nModel built — ResNet50 Baseline')
total = sum(p.numel() for p in model.parameters())
print(f'Parameters: {total:,}')

# ─────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def train_one_epoch(model, loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc='  Train', leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), 100 * correct / total

def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)
    return total_loss / len(loader), 100 * correct / total

# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────
print(f'\nStarting training — {EPOCHS} epochs on {DEVICE}')
print('=' * 55)

best_val_acc = 0
history      = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss,   val_acc   = evaluate(model, val_loader)
    scheduler.step()

    elapsed = time.time() - t0
    history.append({
        'epoch': epoch,
        'train_loss': round(train_loss, 4),
        'train_acc' : round(train_acc,  2),
        'val_loss'  : round(val_loss,   4),
        'val_acc'   : round(val_acc,    2),
    })

    print(f'Epoch {epoch:2d}/{EPOCHS} | '
          f'Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}% | '
          f'Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}% | '
          f'{elapsed:.0f}s')

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path    = os.path.join(SAVE_DIR, 'resnet_baseline.pth')
        torch.save({
            'epoch'     : epoch,
            'model_state': model.state_dict(),
            'val_acc'   : val_acc,
            'history'   : history,
        }, save_path)
        print(f'  ✅ Best model saved — val_acc={val_acc:.1f}%')

print('=' * 55)
print(f'Training complete!')
print(f'Best validation accuracy : {best_val_acc:.1f}%')
print(f'Model saved to           : {save_path}')

# Save history
with open(os.path.join(SAVE_DIR, 'resnet_history.json'), 'w') as f:
    json.dump(history, f, indent=2)
print('History saved.')
