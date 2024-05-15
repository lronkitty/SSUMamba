import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-12


def abs(x):
    return torch.sqrt(x[:, :, :, :, 0] ** 2 + x[:, :, :, :, 1] ** 2 + EPS)


def real(x):
    return x[:, :, :, :, 0]


def imag(x):
    return x[:, :, :, :, 1]


def downsample(img1, img2, maxSize=256):
    _, channels, H, W = img1.shape
    f = int(max(1, np.round(min(H, W) / maxSize)))
    if f > 1:
        aveKernel = (torch.ones(channels, 1, f, f) / f ** 2).to(img1.device)
        img1 = F.conv2d(img1, aveKernel, stride=f, padding=0, groups=channels)
        img2 = F.conv2d(img2, aveKernel, stride=f, padding=0, groups=channels)
    return img1, img2
