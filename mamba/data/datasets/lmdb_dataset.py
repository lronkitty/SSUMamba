import os
import lmdb
import pickle
import yaml
from torch.utils.data import Dataset


class LMDBDataset(Dataset):
    def __init__(self, path_lmdb, img_id=None):
        super().__init__()
        self.path_lmdb = path_lmdb
        self.env = lmdb.open(
            self.path_lmdb,
            readonly=True,
            lock=False,
            create=False,
            readahead=False,
        )
        self.img_id = img_id
        with self.env.begin(write=False) as txn:
            self.n_items = txn.stat()["entries"]
        path_meta = os.path.join(self.path_lmdb, "meta.yaml")
        with open(path_meta, "r") as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        assert self.n_items == meta["n_items"]

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            data = txn.get(str(idx).encode("ascii"))
        patch = pickle.loads(data)
        item = {"y": patch}
        if self.img_id is not None:
            item["img_id"] = self.img_id
        return item

    def __len__(self):
        return self.n_items
