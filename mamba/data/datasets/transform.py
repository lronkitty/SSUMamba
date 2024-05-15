import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TransformDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        noise,
        bands: int,
        seed: int,
        mode: str,
        transforms=None,
    ):
        super().__init__()
        assert mode in ["train", "test", "val", "ssl"]
        self.dataset = dataset
        self.seed = seed
        self.mode = mode
        self.transforms = transforms
        self.noise = noise
        self.bands = bands
        noise_cls = noise_models.__dict__[self.noise.name]
        self.noise_model = noise_cls(bands=self.bands, **self.noise.params)

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
        else:
            seed = self.seed + 200 + idx

        if self.noise is None:
            noisy = clean.clone()
        else:
            clean, noisy = self.noise_model.apply(clean, seed=seed)

        item["y"] = clean
        item["x"] = noisy
        if self.mode == "test":
            noise = torch.std(noisy - clean)
            item["noise"] = noise.item()
        return item

    def __len__(self):
        return len(self.dataset)
