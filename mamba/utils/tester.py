from collections import defaultdict
import logging
import os
from PIL import Image

import json
import matplotlib
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from . import dataloaders_hsi_test
from tqdm import tqdm
import scipy.io
import cv2
from .indexes import MSIQAs

from mamba.models.metrics import (
    mergas,
    mfsim,
    mpsnr,
    msam,
    mssim,
    psnr,
)

matplotlib.use("Agg")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

RGB_DIR = "rgb"
RGB_CROP_DIR = "rgb_cropped"
MAT_DIR = "mat"

def norm(input):
    # min = input.min()
    min = 0
    max = input.max()
    output = (input-min)/(max-min)
    return output

def log_metrics(metrics, log_in=True):
    inout_metrics = list(
        set(
            [
                n.split("_")[0]
                for n in metrics.keys()
                if n.split("_")[1] in ["in", "out"]
            ]
        )
    )
    for name, value in metrics.items():
        if name.split("_")[0] in inout_metrics:
            continue
        if isinstance(value, list):
            value = value[-1]
        logger.info(f"\t{name.upper()} : {value:.4f}")

    for m_name in inout_metrics:
        m_out = metrics[f"{m_name}_out"]
        if log_in:
            m_in = metrics[f"{m_name}_in"]
        else:
            m_in = 0
        if isinstance(m_out, list):
            if log_in:
                m_in = m_in[-1]
            m_out = m_out[-1]
        logger.info(f"\t{m_name.upper()} : in={m_in:.4f}, out={m_out:.4f}")


class Tester:
    def __init__(
        self,
        name,
        save_rgb,
        save_rgb_crop,
        save_raw,
        save_labels,
        seed,
        idx_test,
        test_dir,
        gt_dir,
    ):
        self.save_rgb = save_rgb
        self.save_rgb_crop = save_rgb_crop
        self.save_raw = save_raw
        self.save_labels = save_labels
        self.seed = seed
        self.idx_test = idx_test
        self.test_dir = test_dir
        self.gt_dir = gt_dir

    def eval(self, model, datamodule):
        torch.manual_seed(self.seed)

        self.metrics = {"n_params": model.count_params()[0]}
        self.all_metrics = defaultdict(list)
        if model.__class__.__name__ == "SPCNN_TF":
            dev = "cpu"
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(dev)
        model = model.to(device)

        self.extract_info(datamodule)

        logger.debug(f"Iterating on batches")

        # evaluate on the whole test set
        datamodule.max_test = None
        datamodule.idx_test = self.idx_test
        datamodule.setup("test")
        test_dataloader = datamodule.test_dataloader()
        n_batches = len(test_dataloader)
        test = dataloaders_hsi_test.get_dataloaders([self.test_dir],verbose=True,grey=False)
        n_batches = len(test['test'])
        #{'y': tensor([[[[0.0022, 0....2961]]]]), 'img_id': ['bulb_0822-0909'], 'x': tensor([[[[-0.0473, ....1800]]]])}
        for i,(x,fname) in enumerate(tqdm(test['test'],disable=True)):
            fname=fname[0]
            y=dataloaders_hsi_test.get_gt(self.gt_dir,fname)
            logger.info(f'{x.max()},{x.min()}')
            logger.info(f'{y.max()},{y.min()}')
            batch = {
                'y':torch.unsqueeze(norm(y),0),
                'img_id':[fname],
                'x':x
            }
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logger.info(f'Image ID {i+1}/{n_batches}: {batch["img_id"][0]}')

            model.eval()
            x = batch["x"]
            logger.debug(f"x input shape : {x.shape}")
            with torch.no_grad():
                tic = time.time()

                if model.ssl:
                    logger.info(f"ssl test")
                    bs, c, h, w = batch["x"].shape
                    out = torch.zeros_like(batch["x"])
                    N = int(np.ceil(c / model.n_ssl))
                    for i in range(N):
                        ssl_idx = model.get_ssl_idx(i, c).long()
                        batch["ssl_idx"] = ssl_idx
                        if (
                            model.block_inference
                            and model.block_inference.use_bi
                        ):
                            _out = model.forward_blocks(**batch)
                        else:
                            _out = model.forward(**batch)
                        out[:, ssl_idx] = _out[:, ssl_idx]
                    out = out.float()
                else:
                    if model.block_inference and model.block_inference.use_bi:
                        out = model.forward_blocks(**batch)
                    else:
                        out = model.forward(**batch)
                elapsed = time.time() - tic
                batch["out"] = out.clamp(0, 1)

            logger.debug(f"Inference done")
            self.all_metrics["inference_time"].append(elapsed)

            self.compute_metrics_denoising(**batch)
            logger.info(f"Image metrics :")

            img_id = batch["img_id"][0]

            crop_info = self.get_crop_info(img_id)
            if self.save_raw:
                self._save_raw(**batch)
            if len(crop_info) == 0:
                logger.debug(f"No crop found for {img_id}, not saving to RGB")
                logger.debug(f"{self.img_info}")
                continue

            if self.save_rgb:
                self._save_rgb(**batch)

        self.aggregate_metrics()

    def get_crop_info(self, img_id):
        try:
            return self.img_info[img_id]["crop"]
        except KeyError:
            return []

    def extract_info(self, datamodule):
        logger.debug("Extracting datamodule info..")
        crops = datamodule.dataset_factory.CROPS
        rgb = datamodule.dataset_factory.RGB
        self.img_info = {
            img_id.replace(".", ""): {"crop": crop, "rgb": rgb}
            for (img_id, crop) in crops.items()
        }

    def to_pil(self, x, img_id, crop=False):
        bands = self.img_info[img_id]["rgb"]
        bands = torch.tensor(bands).long()
        x = x[0, bands].permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(np.uint8(255 * x.clip(0, 1)))
        if crop:
            img = img.crop(self.img_info[img_id]["crop"])
        return img

    def compute_metrics_denoising(self, x, y, out, img_id, **kwargs):
        # x : (bs, c, h, w)
        logger.debug("Computing denoising metrics..")
        x = x.clamp(0, 1)
        if torch.any(torch.isnan(y)):
            logger.debug(f"Nan detected in y")
        if torch.any(torch.isnan(x)):
            logger.debug(f"Nan detected in x")
        img_metrics = {}
        with torch.no_grad():
            # logger.debug("Computing PSNR")
            # img_metrics["psnr_in"] = psnr(x, y).item()
            # img_metrics["psnr_out"] = psnr(out, y).item()

            # logger.debug("Computing MPSNR")
            # img_metrics["mpsnr_in"] = mpsnr(x, y).item()
            # img_metrics["mpsnr_out"] = mpsnr(out, y).item()

            # logger.debug("Computing MSSIM")
            # img_metrics["mssim_in"] = mssim(x, y).item()
            # img_metrics["mssim_out"] = mssim(out, y).item()

            # h, w = x.shape[-2:]
            # s = min(h, w)
            # logger.debug(f"Computing MFSIM (s={s})")

            # img_metrics["mfsim_in"] = mfsim(
            #     x[:, :, :s, :s].float(), y[:, :, :s, :s].float()
            # ).item()
            # img_metrics["mfsim_out"] = mfsim(
            #     out[:, :, :s, :s].float(), y[:, :, :s, :s].float()
            # ).item()

            # logger.debug("Computing MERGAS")
            # img_metrics["mergas_in"] = mergas(x, y).item()
            # img_metrics["mergas_out"] = mergas(out, y).item()

            # logger.debug("Computing MSAM")
            # img_metrics["msam_in"] = msam(x, y).item()
            # img_metrics["msam_out"] = msam(out, y).item()

            logger.debug("MSIQAs")
            avg_mpsnr,avg_ssim,avg_sam = MSIQAs(x,y)
            avg_mpsnr_out,avg_ssim_out,avg_sam_out = MSIQAs(out,y)


            img_metrics["MSIQAs_mpsnr_in"] = avg_mpsnr.item()
            img_metrics["MSIQAs_ssim_in"] = avg_ssim.item()
            img_metrics["MSIQAs_sam_in"] = avg_sam.item()
            img_metrics["MSIQAs_mpsnr_out"] = avg_mpsnr_out.item()
            img_metrics["MSIQAs_ssim_out"] = avg_ssim_out.item()
            img_metrics["MSIQAs_sam_out"] = avg_sam_out.item()

        log_metrics(img_metrics)
        for k, v in img_metrics.items():
            self.all_metrics[k].append(v)

        self.metrics[img_id[0]] = img_metrics

    def aggregate_metrics(self):
        global_metrics = {}
        for name, samples in self.all_metrics.items():
            global_metrics[name] = np.mean(samples)
        self.metrics["global"] = global_metrics

        logger.info("-" * 16)
        logger.info("Global metrics :")
        log_metrics(global_metrics)

        with open("test_metrics.json", "w") as f:
            f.write(json.dumps(self.metrics))
            f.close()
        logger.info("Metrics saved to 'test_metrics.json'")
        logger.info(f"Current workdir : {os.getcwd()}")

    def _save_rgb(self, x, out, img_id, y=None, crop=False, **kwargs):
        logger.debug(f"Trying to save RGB")
        img_id = img_id[0]
        folder = RGB_CROP_DIR if crop else RGB_DIR
        os.makedirs(folder, exist_ok=True)
        img_pil = {
            "in": self.to_pil(x, img_id, crop=crop),
            "out": self.to_pil(out, img_id, crop=crop),
        }

        for (cat, pil) in img_pil.items():
            path_img = f"{folder}/{img_id}_{cat}.png"
            pil.save(path_img)
            logger.debug(f"Image saved to {path_img!r}")

    def _save_raw(self, x, out, img_id, y=None, crop=False, **kwargs):
        logger.debug(f"Trying to save mat")
        img_id = img_id[0]
        folder = MAT_DIR
        os.makedirs(folder, exist_ok=True)
        path_img = f"{folder}/{img_id}"
        scipy.io.savemat(path_img,mdict={'srt':out[0].cpu().numpy()})
        logger.debug(f"Image saved to {path_img!r}")