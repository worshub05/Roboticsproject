"""
Multi-task lane CNN with uncertainty.

The existing LaneCNN (steering) is kept unchanged as the backbone. A small extra head reads the
same features that feed the steering output and predicts three more things about the road:

  e_lat   offset from the lane centre (m, + = left)
  e_head  heading error to the lane (rad, + = pointing left)
  kappa   curvature of the road 0.5 to 2.5 m ahead (1/m, + = left bend)

Uncertainty comes from Monte Carlo dropout: the image is run N times with dropout switched on,
and the spread of the N answers says how unsure the network is (Gal and Ghahramani, 2016).
"""
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from .lane_cnn_model import LaneCNN

AUX_NAMES = ('e_lat', 'e_head', 'kappa')


class MultiTaskLaneCNN(nn.Module):
    def __init__(self, p_conv=0.1, p_fc=0.2):
        super().__init__()
        self.base = LaneCNN()
        # the steering output is the last Linear layer, whatever it is called (fc, fc2, ...)
        last = [m for m in self.base.modules() if isinstance(m, nn.Linear)][-1]
        self.aux = nn.Sequential(nn.Linear(last.in_features, 64), nn.ReLU(), nn.Linear(64, len(AUX_NAMES)))
        # Target scaling, so all four outputs matter equally in the loss. Saved with the model.
        self.register_buffer('aux_mean', torch.zeros(len(AUX_NAMES)))
        self.register_buffer('aux_std', torch.ones(len(AUX_NAMES)))
        # LaneCNN has no dropout, and Monte Carlo dropout needs some. Add it with hooks, so the
        # LaneCNN code and its saved weights stay exactly as they are (dropout has no weights):
        #   channel dropout after the last two BatchNorm layers, plain dropout before the output layer.
        self.drop_conv = nn.Dropout2d(p_conv)
        self.drop_fc = nn.Dropout(p_fc)
        bns = [m for m in self.base.modules() if isinstance(m, nn.BatchNorm2d)]
        for bn in bns[-2:]:
            bn.register_forward_hook(lambda mod, inp, out: self.drop_conv(out))
        last.register_forward_pre_hook(self._drop_and_grab)
        self._feat = None

    def _drop_and_grab(self, module, inputs):
        self._feat = self.drop_fc(inputs[0])   # shared features (after dropout) feed both heads
        return (self._feat,)

    def forward(self, x):
        """Returns steering (N,1) and the three extra outputs in normalised units (N,3)."""
        steer = self.base(x)
        return steer, self.aux(self._feat)

    def denorm(self, aux_n):
        return aux_n * self.aux_std + self.aux_mean


def preprocess(img_rgb):
    """Same preprocessing as training: crop the top quarter if taller than 100, resize, scale, CHW."""
    if img_rgb.shape[0] > 100:
        img_rgb = img_rgb[int(img_rgb.shape[0] * 0.25):]
    img = cv2.resize(img_rgb, (160, 100)).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(img, (2, 0, 1)).copy())


def load_multitask(path, device='cpu'):
    model = MultiTaskLaneCNN().to(device)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def enable_mc_dropout(model):
    """Eval mode everywhere (BatchNorm uses its stored statistics) but dropout layers stay on."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def predict(model, img_rgb, passes=10, device='cpu'):
    """
    One camera image -> dict with the mean and standard deviation of every output.
    passes=1 gives a plain deterministic prediction (std = 0).
    """
    x = preprocess(img_rgb).unsqueeze(0).to(device)
    if passes > 1:
        enable_mc_dropout(model)
        x = x.repeat(passes, 1, 1, 1)
    else:
        model.eval()
    steer, aux_n = model(x)
    aux = model.denorm(aux_n)
    out = torch.cat([steer, aux], dim=1).cpu().numpy()
    model.eval()
    mean, std = out.mean(0), out.std(0)
    names = ('steer',) + AUX_NAMES
    res = {n: float(mean[i]) for i, n in enumerate(names)}
    res.update({n + '_std': float(std[i]) for i, n in enumerate(names)})
    return res
