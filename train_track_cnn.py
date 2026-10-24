#!/usr/bin/env python3
"""
Fine-tune the lane CNN on Gazebo track images collected by collect_track_data.py.

  python3 ~/ros2_ws/train_track_cnn.py [epochs]

Reads  ~/ros2_ws/track_data/img_<index>_<steering>.png
Starts from ~/ros2_ws/models/lane_cnn_v2.pt (if present)
Writes ~/ros2_ws/models/lane_cnn_v3.pt (the epoch with the lowest validation error)
"""
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path.home() / 'ros2_ws' / 'src' / 'lane_follower_perception'))
from lane_follower_perception.lane_cnn_model import LaneCNN, load_model  # noqa: E402

DATA = Path.home() / 'ros2_ws' / 'track_data'
START = Path.home() / 'ros2_ws' / 'models' / 'lane_cnn_v2.pt'
OUT = Path.home() / 'ros2_ws' / 'models' / 'lane_cnn_v3.pt'
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
BLOCK = 100   # consecutive frames are near-duplicates, so split train/validation by blocks


def label_of(p):
    return float(p.stem.rsplit('_', 1)[1])


def load_image(p):
    """Same preprocessing as lane_cnn_model.preprocess_image at inference time."""
    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    if img.shape[0] > 100:
        img = img[int(img.shape[0] * 0.25):]
    img = cv2.resize(img, (160, 100))
    return img.astype(np.float32) / 255.0


class TrackSet(Dataset):
    def __init__(self, files, augment):
        self.files, self.augment = files, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        img = load_image(p)
        if self.augment:   # lighting changes only; no mirroring (the dashed line is always on the left)
            img = np.clip(img * random.uniform(0.75, 1.25) + random.uniform(-0.05, 0.05), 0.0, 1.0)
        x = torch.from_numpy(np.transpose(img, (2, 0, 1)).copy())
        return x, torch.tensor([label_of(p)], dtype=torch.float32)


def evaluate(model, loader):
    model.eval()
    se = ae = n = 0.0
    with torch.no_grad():
        for x, y in loader:
            out = model(x).clamp(-1.2, 1.2)   # inference clamps the same way
            se += ((out - y) ** 2).sum().item()
            ae += (out - y).abs().sum().item()
            n += len(y)
    return se / n, ae / n


def main():
    torch.manual_seed(0)
    random.seed(0)
    files = sorted(DATA.glob('img_*.png'))
    if not files:
        sys.exit(f'No images found in {DATA}')
    blocks = [files[i:i + BLOCK] for i in range(0, len(files), BLOCK)]
    random.Random(0).shuffle(blocks)
    n_val = max(1, len(blocks) // 7)
    val = [f for b in blocks[:n_val] for f in b]
    train = [f for b in blocks[n_val:] for f in b]
    labels = np.array([label_of(f) for f in files])
    print(f'{len(files)} images ({len(train)} train, {len(val)} validation); '
          f'steering labels {labels.min():+.2f} to {labels.max():+.2f} rad/s')

    train_loader = DataLoader(TrackSet(train, True), batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(TrackSet(val, False), batch_size=64, num_workers=2)

    model = load_model(str(START), device='cpu') if START.exists() else LaneCNN()
    mse0, mae0 = evaluate(model, val_loader)
    straight = float((np.array([label_of(f) for f in val]) ** 2).mean())
    print(f'before training: validation MSE {mse0:.4f}, mean error {mae0:.3f} rad/s '
          f'(always-steer-straight baseline MSE {straight:.4f})', flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()
    best = float('inf')
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0, total, m = time.time(), 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            m += len(y)
        mse, mae = evaluate(model, val_loader)
        note = ''
        if mse < best:
            best = mse
            torch.save(model.state_dict(), str(OUT))
            note = '  <- saved'
        print(f'epoch {ep:2d}/{EPOCHS}: train MSE {total / m:.4f} | validation MSE {mse:.4f}, '
              f'mean error {mae:.3f} rad/s | {time.time() - t0:.0f} s{note}', flush=True)
    print(f'✅ best validation MSE {best:.4f} (was {mse0:.4f} before training); model saved to {OUT}')


if __name__ == '__main__':
    main()
