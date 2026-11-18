#!/usr/bin/env python3
"""
DAgger training step: fine-tune the multi-task lane CNN on the AGGREGATED data set
(the original expert data plus every DAgger round so far).

  python3 tools/dagger_train.py --start ~/ros2_ws/models/lane_mt_v1.pt --out ~/ros2_ws/models/lane_mt_d1.pt \
      --data ~/ros2_ws/track_data_mt ~/ros2_ws/dagger/round1 [--epochs 6]

Each data folder holds labels.csv (file,steer,e_lat,e_head,kappa) and the images, as written by
collect_track_data.py and dagger.py. The model starts from --start (all weights, and the label
normalisation stays the same), and the epoch with the lowest validation loss is saved to --out.
Validation uses whole blocks of 100 consecutive frames from every folder (near-duplicate frames
must not be split between training and validation).
"""
import argparse
import csv
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lane_follower_perception.multitask_cnn import AUX_NAMES, load_multitask, preprocess  # noqa: E402

BLOCK = 100


def read_rows(folder):
    rows = []
    path = os.path.join(folder, 'labels.csv')
    if not os.path.exists(path):
        sys.exit(f'no labels.csv in {folder}')
    with open(path) as f:
        for r in csv.DictReader(f):
            img = os.path.join(folder, r['file'])
            if os.path.exists(img):
                rows.append((img, float(r['steer'])) + tuple(float(r[n]) for n in AUX_NAMES))
    return rows


class MTSet(Dataset):
    def __init__(self, rows, augment):
        self.rows, self.augment = rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, steer, *aux = self.rows[i]
        x = preprocess(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
        if self.augment:
            x = torch.clamp(x * random.uniform(0.75, 1.25) + random.uniform(-0.05, 0.05), 0.0, 1.0)
        return x, torch.tensor([steer], dtype=torch.float32), torch.tensor(aux, dtype=torch.float32)


def run_epoch(model, loader, opt=None):
    model.train(opt is not None)
    tot, n, err = 0.0, 0, 0.0
    with torch.set_grad_enabled(opt is not None):
        for x, steer, aux in loader:
            p_steer, p_aux_n = model(x)
            loss = F.mse_loss(p_steer, steer) + F.mse_loss(p_aux_n, (aux - model.aux_mean) / model.aux_std)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * len(x)
            err += (p_steer.clamp(-1.2, 1.2) - steer).abs().sum().item()
            n += len(x)
    return tot / n, err / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    train, val = [], []
    for folder in args.data:
        rows = read_rows(os.path.expanduser(folder))
        blocks = [rows[i:i + BLOCK] for i in range(0, len(rows), BLOCK)]
        random.Random(0).shuffle(blocks)
        n_val = max(1, len(blocks) // 7)
        val += [r for b in blocks[:n_val] for r in b]
        train += [r for b in blocks[n_val:] for r in b]
        print(f'  {folder}: {len(rows)} images')
    print(f'{len(train) + len(val)} images in all ({len(train)} train, {len(val)} validation)')

    model = load_multitask(os.path.expanduser(args.start))
    train_loader = DataLoader(MTSet(train, True), batch_size=32, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(MTSet(val, False), batch_size=64, num_workers=args.workers)
    opt = torch.optim.Adam([{'params': model.base.parameters(), 'lr': args.lr},
                            {'params': model.aux.parameters(), 'lr': 5 * args.lr}])
    va0, err0 = run_epoch(model, val_loader)
    print(f'start model: val loss {va0:.4f}, steering error {err0:.4f} rad/s')
    best = va0
    out = os.path.expanduser(args.out)
    torch.save(model.state_dict(), out)                    # never worse than the start model
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr, _ = run_epoch(model, train_loader, opt)
        va, err = run_epoch(model, val_loader)
        note = ''
        if va < best:
            best = va
            torch.save(model.state_dict(), out)
            note = '  <- saved'
        print(f'epoch {ep}/{args.epochs}: train {tr:.4f} | val {va:.4f} | steering error {err:.4f} rad/s | '
              f'{time.time() - t0:.0f} s{note}', flush=True)
    print(f'✅ saved {out} (val loss {va0:.4f} -> {best:.4f})')


if __name__ == '__main__':
    main()
