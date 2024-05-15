import os
from typing import List

import torch
from torch.utils.data import Dataset


class PathsDataset(Dataset):
    def __init__(self, paths: List[str]):
        super().__init__()
        self.paths = paths
        self.img_ids = [
            os.path.split(p)[1].replace(".pth", "") for p in self.paths
        ]

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        clean = torch.load(self.paths[idx])

        return {"y": clean, "img_id": img_id}

    def __len__(self):
        return len(self.paths)
