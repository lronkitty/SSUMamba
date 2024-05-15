import logging
import os
import subprocess
import torch
from imageio import imread
import zipfile
from shutil import copy2

from mamba.data.normalizers import BandMinMaxQuantileStateful
from .base_factory import DatasetFactory
from .utils import check_filesize, touch
from mamba.data.splits import (
    dcmall_train,
    dcmall_test,
    dcmall_val,
    dcmall_crops,
    dcmall_rgb,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

URL = "http://cobweb.ecn.purdue.edu/~biehl/Hyperspectral_Project.zip"


class DCMall(DatasetFactory):
    NAME = "DCMall"
    IMG_SHAPE = (191, 600, 307)
    CROPS = dcmall_crops
    RGB = dcmall_rgb

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.split == 0

        self.f_train = dcmall_train
        self.f_test = dcmall_test
        self.f_val = dcmall_val

    @classmethod
    def download(cls, path_data):
        path_dataset = os.path.join(path_data, cls.NAME)
        path_raw = os.path.join(path_dataset, "raw")
        path_dl = os.path.join(path_raw, ".download_complete")
        if os.path.exists(path_dl):
            return
        os.makedirs(path_raw, exist_ok=True)
        cached = "/tmp/dcmall.zip"
        if not check_filesize(cached, URL):
            logger.info("Downloading DC Mall")
            cmd_dl = f"wget --no-check-certificate {URL!r} -O {cached!r}"
            subprocess.check_call(cmd_dl, shell=True)
        logger.info("Extracting DC Mall..")
        with zipfile.ZipFile(cached, "r") as zip_ref:
            zip_ref.extractall("/tmp/")
        copy2("/tmp/Hyperspectral_Project/dc.tif", path_raw)
        touch(path_dl)
        logger.info("Extraction complete")

    def preprocess(self):

        path_raw = os.path.join(self.path_data, self.NAME, "raw")
        path_clean = os.path.join(self.path_data, self.NAME, "clean")
        path_tif = os.path.join(path_raw, "dc.tif")
        path_done = os.path.join(path_clean, ".done")
        if os.path.exists(path_done):
            return

        full_img = torch.tensor(imread(path_tif), dtype=torch.float)
        logger.debug(f"dcmall_full : {full_img.shape}")

        test = full_img[:, 600:800, 50:250].clone()
        logger.debug(f"dcmall_test : {test.shape}")
        train_0 = full_img[:, :600, :].clone()
        logger.debug(f"dcmall_train_0 : {train_0.shape}")
        train_1 = full_img[:, 800:, :].clone()
        logger.debug(f"dcmall_train_1 : {train_1.shape}")
        val = full_img[:, 600:656, 251:].clone()
        logger.debug(f"dcmall_val : {val.shape}")

        normalizer = BandMinMaxQuantileStateful()

        # fit train
        normalizer.fit([train_0, train_1])
        train_0 = normalizer.transform(train_0)
        train_1 = normalizer.transform(train_1)

        # fit test
        normalizer.fit([test])
        test = normalizer.transform(test)

        # val test
        normalizer.fit([val])
        val = normalizer.transform(val)

        os.makedirs(path_clean, exist_ok=True)
        torch.save(train_0, os.path.join(path_clean, "dcmall_train_0.pth"))
        torch.save(train_1, os.path.join(path_clean, "dcmall_train_1.pth"))
        torch.save(test, os.path.join(path_clean, "dcmall_test.pth"))
        torch.save(val, os.path.join(path_clean, "dcmall_val.pth"))

        logger.debug(f"Images saved to {path_clean}")
        touch(path_done)
