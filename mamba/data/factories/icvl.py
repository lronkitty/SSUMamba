import logging
import os
import subprocess
import torch
import h5py
import numpy as np

from mamba.data.normalizers import GlobalMinMax
from .base_factory import DatasetFactory
from .utils import check_filesize, touch
from mamba.data.splits import (
    icvl_train,
    icvl_val,
    icvl_test,
    icvl_crops,
    icvl_rgb,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ICVL(DatasetFactory):
    NAME = "ICVL"
    IMG_SHAPE = (31, 1024, 1024)
    CROPS = icvl_crops
    RGB = icvl_rgb

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.split == 0

        self.f_train = icvl_train
        self.f_val = icvl_val
        self.f_test = icvl_test

    @classmethod
    def download(cls, path_data):
        BASE_URL = "http://icvl.cs.bgu.ac.il/img/hs_pub/"
        path_dataset = os.path.join(path_data, cls.NAME)
        path_raw = os.path.join(path_dataset, "raw")
        path_dl_complete = os.path.join(path_raw, ".download_complete")
        if os.path.exists(path_dl_complete):
            logger.info(f"Dataset downloaded")
            return
        logger.info(f"{path_dl_complete!r} not found, checking filesizes ..")
        os.makedirs(path_raw, exist_ok=True)

        icvl_all = icvl_train + icvl_val + icvl_test
        icvl_all = [f"{fn}.mat" for fn in icvl_all]
        for i, filename in enumerate(icvl_all):
            target = os.path.join(path_raw, filename)
            url = os.path.join(BASE_URL, filename)
            logger.info(
                f"Checking image ({i + 1}/{len(icvl_all)}) : {filename}"
            )
            if os.path.exists(target) and check_filesize(target, url):
                logger.info(f"OK")
                continue
            logger.info(f"Downloading..")
            subprocess.check_call(
                f"wget {url} -O {target}",
                shell=True,
                stdout=subprocess.DEVNULL,
            )

        touch(path_dl_complete)

    def preprocess(self):
        path_source = os.path.join(self.path_data, self.NAME, "raw")
        path_dest = os.path.join(self.path_data, self.NAME, "clean")
        path_complete = os.path.join(path_dest, ".complete")
        if os.path.exists(path_complete):
            return

        os.makedirs(path_dest, exist_ok=True)

        normalizer = GlobalMinMax()
        icvl_all = list(set(self.f_train + self.f_test + self.f_val))

        for i, fn in enumerate(icvl_all):
            path_out = os.path.join(path_dest, f"{fn}.pth")
            if os.path.exists(path_out):
                continue
            logger.info(f"Preprocessing {fn}")
            path_in = os.path.join(path_source, f"{fn}.mat")
            with h5py.File(path_in, "r") as f:
                img = np.array(f["rad"], dtype=np.float32)
            img_torch = torch.tensor(img, dtype=torch.float32)
            img_torch = normalizer.transform(img_torch).clone()
            logger.info(f"shape : {tuple(img_torch.shape)} ")

            torch.save(img_torch, path_out)
            logger.info(
                f"Saved normalized image {i + 1}/{len(icvl_all)} "
                f"to {path_out}"
            )
        touch(path_complete)
        logger.info(f"Dataset preprocessed")
