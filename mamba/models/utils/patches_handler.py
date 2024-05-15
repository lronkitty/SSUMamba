import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PatchesHandler(nn.Module):
    def __init__(self, size, channels, stride, padding="constant"):
        super().__init__()
        if isinstance(size, int):
            self.size = np.array([size, size])
        else:
            self.size = np.array(size)
        self.channels = channels
        self.n_elements = self.channels * self.size[0] * self.size[1]
        self.stride = stride

        self.padding = padding
        self.fold = None
        self.normalizer = None
        self.img_size = None

    def forward(self, x, mode="extract"):
        if mode == "extract":
            x = self.pad(x)
            x = self.extract(x)
            return x
        elif mode == "aggregate":
            x = self.aggregate(x)
            x = self.unpad(x)
            return x
        else:
            raise ValueError(f"Mode {mode!r} not recognized")

    def set_img_size(self, img_size):
        if np.any(self.img_size != np.array(img_size)):
            self.img_size = np.array(img_size)

            self.n_patches = 1 + np.ceil(
                np.maximum(self.img_size - self.size, 0) / self.stride
            ).astype(int)
            pads = (
                self.size + (self.n_patches - 1) * self.stride - self.img_size
            )
            self.padded_size = tuple(self.img_size + pads)
            _pads = []
            for i in reversed(range(2)):
                _pads += [0, pads[i]]
            self.pads = _pads

    def pad(self, x):
        self.set_img_size(x.shape[2:])
        x = F.pad(x, pad=self.pads, mode=self.padding)
        return x

    def unpad(self, x):
        x = x[:, :, : self.img_size[0], : self.img_size[1]]
        return x

    def extract(self, x):

        x = x.unfold(dimension=2, size=self.size[0], step=self.stride)
        x = x.unfold(dimension=3, size=self.size[1], step=self.stride)
        x = x.permute(0, 1, 4, 5, 2, 3)

        x = x.contiguous()

        return x

    def aggregate(self, x):
        x = x.view(-1, self.n_elements, self.n_patches[0] * self.n_patches[1])

        if self.fold is None or self.padded_size != self.fold.output_size:
            self.init_fold()

        out = self.fold(x)
        if self.normalizer is None or self.normalizer.shape != x.shape:
            ones_input = torch.ones_like(x)
            self.normalizer = self.fold(ones_input)
        return (out / self.normalizer).squeeze(1)

    def init_fold(self):
        logger.debug(f"Initializing fold, padded shape: {self.padded_size}")
        self.fold = nn.Fold(
            output_size=self.padded_size,
            kernel_size=tuple(self.size),
            stride=self.stride,
        )
