import logging

import torch
from torch.utils.data import Dataset

from mamba.data import noise_models
from torchvision import transforms as tfs
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NoisyTransformDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        noise,
        bands: int,
        seed: int,
        mode: str,
        transforms=None,
        compute_noise=False,
        estimate_noise=False,
    ):
        super().__init__()
        assert mode in [None, "train", "test", "val", "ssl"]
        self.dataset = dataset
        self.seed = seed
        self.mode = mode
        self.transforms = transforms
        self.noise = noise
        self.bands = bands
        if noise is not None:
            noise_cls = noise_models.__dict__[self.noise.name]
            self.noise_model = noise_cls(bands=self.bands, **self.noise.params)
        assert not (compute_noise and estimate_noise)
        self.compute_noise = compute_noise
        self.estimate_noise = estimate_noise

    def __getitem__(self, idx):
        item = self.dataset[idx]
        clean = item["y"]

        if self.transforms is not None:
            clean = self.transforms(clean)

        if self.mode == "test":
            seed = self.seed + idx
        elif self.mode == "val":
            seed = self.seed + 150 + idx
        elif self.mode == "train":
            seed = self.seed + 200 + int(torch.randint(2, 999999, (1,)).item())
        elif self.mode == "ssl":
            seed = self.seed + 200 + idx
        else:
            pass

        if self.noise is None:
            noisy = clean.clone()
        else:
            clean, noisy = self.noise_model.apply(clean, seed=seed)

        item["y"] = clean
        item["x"] = noisy

        return item

    def __len__(self):
        return len(self.dataset)
