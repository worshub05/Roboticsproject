#!/usr/bin/env python3
"""
Train the multi-task lane CNN (steering + lane offset + heading error + curvature ahead).

  python3 ~/ros2_ws/train_multitask_cnn.py [epochs]

Reads  ~/ros2_ws/track_data_mt/labels.csv and the images next to it
Starts the backbone from ~/ros2_ws/models/lane_cnn_v3.pt (the curved-track steering model)
Writes ~/ros2_ws/models/lane_mt_v1.pt (the epoch with the lowest validation loss)
Then checks the Monte Carlo dropout uncertainty on the validation images.
"""
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path.home() / 'ros2_ws' / 'src' / 'lane_follower_perception'))
from lane_follower_perception.multitask_cnn import (  # noqa: E402
    AUX_NAMES, MultiTaskLaneCNN, enable_mc_dropout, preprocess)

DATA = Path.home() / 'ros2_ws' / 'track_data_mt'
START = Path.home() / 'ros2_ws' / 'models' / 'lane_cnn_v3.pt'
OUT = Path.home() / 'ros2_ws' / 'models' / 'lane_mt_v1.pt'
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
BLOCK = 100          # consecutive frames are near-duplicates, so split train/validation by blocks
UNITS = {'steer': 'rad/s', 'e_lat': 'm', 'e_head': 'rad', 'kappa': '1/m'}


def read_rows():
    rows = []
    with open(DATA / 'labels.csv') as f:
        for r in csv.DictReader(f):
            if (DATA / r['file']).exists():
                rows.append((r['file'], float(r['steer'])) + tuple(float(r[n]) for n in AUX_NAMES))
    return rows


def load_image(name):
    return preprocess(cv2.cvtColor(cv2.imread(str(DATA / name)), cv2.COLOR_BGR2RGB))


class MTSet(Dataset):
    def __init__(self, rows, augment):
        self.rows, self.augment = rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name, steer, *aux = self.rows[i]
        x = load_image(name)
        if self.augment:     # lighting changes only; mirroring would break the left/right lane lines
            x = torch.clamp(x * random.uniform(0.75, 1.25) + random.uniform(-0.05, 0.05), 0.0, 1.0)
        return x, torch.tensor([steer]), torch.tensor(aux, dtype=torch.float32)


def run_epoch(model, loader, opt=None):
    """One pass over the data. Returns total loss and mean absolute error per output (real units)."""
    model.train(opt is not None)
    tot, n, abs_err = 0.0, 0, torch.zeros(1 + len(AUX_NAMES))
    with torch.set_grad_enabled(opt is not None):
        for x, steer, aux in loader:
            p_steer, p_aux_n = model(x)
            aux_n = (aux - model.aux_mean) / model.aux_std
            loss = F.mse_loss(p_steer, steer) + F.mse_loss(p_aux_n, aux_n)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            pred = torch.cat([p_steer.clamp(-1.2, 1.2), model.denorm(p_aux_n)], 1)
            abs_err += (pred - torch.cat([steer, aux], 1)).abs().sum(0).detach()
            tot += loss.item() * len(x)
            n += len(x)
    return tot / n, (abs_err / n).tolist()


@torch.no_grad()
def uncertainty_check(model, rows, passes=10, limit=600):
    """Do the images the network is unsure about really have bigger errors?"""
    rows = rows[:limit]
    stds, errs, stds_dark = [], [], []
    enable_mc_dropout(model)
    for name, steer, *_ in rows:
        x = load_image(name).unsqueeze(0)
        s, _ = model(x.repeat(passes, 1, 1, 1))
        stds.append(s.std().item())
        errs.append(abs(s.mean().item() - steer))
        # an image the network never saw in training: very dark and noisy (camera fault / night)
        bad = torch.clamp(x * 0.25 + 0.15 * torch.randn_like(x), 0, 1)
        s2, _ = model(bad.repeat(passes, 1, 1, 1))
        stds_dark.append(s2.std().item())
    model.eval()
    stds, errs = np.array(stds), np.array(errs)
    top = stds >= np.quantile(stds, 0.9)
    print('\nUncertainty check (Monte Carlo dropout, %d passes, %d validation images):' % (passes, len(rows)))
    print(f'  correlation between uncertainty and steering error: {np.corrcoef(stds, errs)[0, 1]:+.2f}')
    print(f'  mean error on the 10% most uncertain images: {errs[top].mean():.3f} rad/s '
          f'vs {errs[~top].mean():.3f} on the rest')
    print(f'  mean uncertainty: normal images {stds.mean():.3f}, dark/noisy images {np.mean(stds_dark):.3f} '
          f'(should be clearly higher)')
    print(f'  suggested alarm level (99th percentile on normal images): {np.quantile(stds, 0.99):.3f} rad/s')


def main():
    torch.manual_seed(0)
    random.seed(0)
    if not (DATA / 'labels.csv').exists():
        sys.exit(f'No labels.csv in {DATA}: collect data first')
    rows = read_rows()
    blocks = [rows[i:i + BLOCK] for i in range(0, len(rows), BLOCK)]
    random.Random(0).shuffle(blocks)
    n_val = max(1, len(blocks) // 7)
    val = [r for b in blocks[:n_val] for r in b]
    train = [r for b in blocks[n_val:] for r in b]
    print(f'{len(rows)} images ({len(train)} train, {len(val)} validation)')

    model = MultiTaskLaneCNN()
    if START.exists():
        model.base.load_state_dict(torch.load(str(START), map_location='cpu'))
        print(f'backbone and steering head start from {START.name}')
    aux = np.array([r[2:] for r in train])
    model.aux_mean.copy_(torch.tensor(aux.mean(0), dtype=torch.float32))
    model.aux_std.copy_(torch.tensor(aux.std(0) + 1e-3, dtype=torch.float32))
    for i, n in enumerate(AUX_NAMES):
        print(f'  {n:6s} range {aux[:, i].min():+.3f} to {aux[:, i].max():+.3f} {UNITS[n]}')

    train_loader = DataLoader(MTSet(train, True), batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(MTSet(val, False), batch_size=64, num_workers=2)
    opt = torch.optim.Adam([{'params': model.base.parameters(), 'lr': 2e-4},
                            {'params': model.aux.parameters(), 'lr': 1e-3}])
    names = ('steer',) + AUX_NAMES
    best = float('inf')
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, train_loader, opt)
        va_loss, mae = run_epoch(model, val_loader)
        note = ''
        if va_loss < best:
            best, best_mae = va_loss, mae
            torch.save(model.state_dict(), str(OUT))
            note = '  <- saved'
        errs = ', '.join(f'{n} {e:.3f}' for n, e in zip(names, mae))
        print(f'epoch {ep:2d}/{EPOCHS}: train {tr_loss:.3f} | val {va_loss:.3f} | mean errors: {errs} '
              f'| {time.time() - t0:.0f} s{note}', flush=True)

    print(f'\n✅ model saved to {OUT}')
    print('Validation mean absolute errors of the saved model:')
    for n, e in zip(names, best_mae):
        print(f'  {n:6s} {e:.3f} {UNITS[n]}')
    model.load_state_dict(torch.load(str(OUT), map_location='cpu'))
    uncertainty_check(model, val)


if __name__ == '__main__':
    main()
