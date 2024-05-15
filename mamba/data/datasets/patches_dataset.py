import itertools
import logging

import numpy as np
import torch
from torch.utils.data import Dataset


def get_patch_coords(h, w, patch_size, stride):
    img_size = np.array([h, w])
    n_patches = 1 + np.floor(
        np.maximum(img_size - patch_size, 0) / stride
    ).astype(int)
    h_coords = [i * stride for i in range(n_patches[0])]
    w_coords = [i * stride for i in range(n_patches[1])]
    coords = list(itertools.product(h_coords, w_coords))
    coords = torch.tensor(coords)
    return coords


class PatchesDataset(Dataset):
    def __init__(self, img, patch_size, stride):
        """
        img: (C, H, W)
        """
        assert len(img.shape) == 3
        self.img = img.to(torch.float32)
        self.c, self.h, self.w = self.img.shape

        self.patch_size = patch_size
        self.stride = stride
        self.compute_coords()

    def compute_coords(self):
        self.coords = get_patch_coords(
            self.h, self.w, self.patch_size, self.stride
        )

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        patch = self.img[
            :, y : y + self.patch_size, x : x + self.patch_size
        ].clone()
        return patch

    def __len__(self):
        return len(self.coords)
