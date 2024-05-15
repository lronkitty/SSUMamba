import logging
import time

import numpy as np
import pytorch_lightning as pl
import torch

from .metrics import mpsnr, mse, psnr
from .utils import PatchesHandler


 
from thop import profile
from torchstat import stat
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseModel(pl.LightningModule):
    def __init__(
        self, optimizer=None, lr_scheduler=None, block_inference=None
    ):
        super().__init__()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.block_inference = block_inference
        self.ssl = 0
        self.n_ssl = 0
        self.automatic_optimization = False
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        # epoch = self.current_epoch
        # sch = self.lr_schedulers()
        # lr = sch.get_last_lr()
        # logger.info(f"Epoch {epoch} : lr={lr},{opt}")

        y = batch.pop("y")
        if self.ssl:
            x = batch["x"]
            bs, c, h, w = x.shape
            ssl_idx = torch.randperm(c)[: self.n_ssl].to(self.device)
            batch["ssl_idx"] = ssl_idx
            out = self.forward(**batch)
            band_out = out[:, ssl_idx]
            band_target = x[:, ssl_idx]
            self.log("train_mse", mse(band_out, band_target))
            self.log("train_psnr", psnr(band_out, band_target))
            self.log("train_mpsnr", mpsnr(band_out.detach(), band_target))
            band_y = y[:, ssl_idx]
            self.log("train_mse_y", mse(band_out, band_y))
            self.log("train_psnr_y", psnr(band_out, band_y))
            self.log("train_mpsnr_y", mpsnr(band_out.detach(), band_y))

            loss = mse(band_out, band_target)

        else:
            out = self.forward(**batch)
            self.log("train_mse", mse(out, y))
            self.log("train_psnr", psnr(out, y))
            self.log("train_mpsnr", mpsnr(out.detach(), y))

            loss = mse(out, y)
        
        self.manual_backward(loss)
        opt.step()

        sch = self.lr_schedulers()
        if self.trainer.is_last_batch:
            epoch = self.current_epoch
            lr = sch.get_last_lr()
            logger.info(f"Epoch {epoch} : lr={lr} \t loss={loss:.6f}")
            sch.step()

    def validation_step(self, batch, batch_idx):
        y = batch.pop("y")
        start = time.time()
        if self.ssl:
            bs, c, h, w = batch["x"].shape
            out = torch.zeros_like(batch["x"])
            N = int(np.ceil(c / self.n_ssl))
            for i in range(N):
                ssl_idx = self.get_ssl_idx(i, c).long()
                batch["ssl_idx"] = ssl_idx
                if self.block_inference and self.block_inference.use_bi:
                    _out = self.forward_blocks(**batch)
                else:
                    _out = self.forward(**batch)
                out[:, ssl_idx] = _out[:, ssl_idx]
        else:
            if self.block_inference and self.block_inference.use_bi:
                out = self.forward_blocks(**batch)
            else:
                out = self.forward(**batch)
        logger.debug(f"Val denoised shape: {out.shape}")
        out = out.clamp(0, 1)
        elapsed = time.time() - start
        _mse = mse(out, y)
        _mpsnr = mpsnr(out, y)
        logger.debug(f"Val mse : {_mse}, mpsnr: {_mpsnr}")
        self.log("val_mse", mse(out, y))
        self.log("val_psnr", psnr(out, y))
        self.log("val_mpsnr", mpsnr(out, y))
        self.log("val_batch_time", elapsed)
        self.log("val_psnr_noise", psnr(batch["x"], y))
        self.log("val_mpsnr_noise", mpsnr(batch["x"], y))

    def get_ssl_idx(self, i, c):
        N = np.ceil(c / self.n_ssl)
        L = int(np.ceil((c - i) / N))
        return i + N * torch.arange(L)

    def test_step(self, batch, batch_idx):
        y = batch.pop("y")
        if self.ssl:
            bs, c, h, w = batch["x"].shape
            out = torch.zeros_like(batch["x"])
            N = int(np.ceil(c / self.n_ssl))
            for i in range(N):
                ssl_idx = self.get_ssl_idx(i, c).long()
                batch["ssl_idx"] = ssl_idx
                if self.block_inference and self.block_inference.use_bi:
                    _out = self.forward_blocks(**batch)
                else:
                    _out = self.forward(**batch)
                out[:, ssl_idx] = _out[:, ssl_idx]
        else:
            if self.block_inference and self.block_inference.use_bi:
                out = self.forward_blocks(**batch)
            else:
                out = self.forward(**batch)
        logger.debug(f"Test denoised shape: {out.shape}")
        out = out.clamp(0, 1)
        self.log("test_mse", mse(out, y))
        self.log("test_psnr", psnr(out, y))
        self.log("test_mpsnr", mpsnr(out, y))
        self.log("test_psnr_noise", psnr(batch["x"], y))
        self.log("test_mpsnr_noise", mpsnr(batch["x"], y))

    def forward_blocks(self, x, **kwargs):
        logger.debug(f"Starting block inference")
        # device_ = x.device
        # x = x.to('cpu')
        # print(x.device)
        block_size = min(
            max(x.shape[-1], x.shape[-2]), self.block_inference.block_size
        )
        patches_handler = PatchesHandler(
            size=(block_size,) * 2,
            channels=x.shape[1],
            stride=block_size - self.block_inference.overlap,
            padding=self.block_inference.padding,
        )

        logger.debug(f"Forward patches handler")
        blocks_in = patches_handler(x, mode="extract").clone()
        blocks_grid = tuple(blocks_in.shape[-2:])
        logger.debug(f"blocks grid : {blocks_in.shape}")

        blocks_out = torch.zeros_like(blocks_in)

        logger.debug(f"Processing blocks {blocks_grid}")
        flops_sum=0
        for i in range(blocks_grid[0]):
            for j in range(blocks_grid[1]):
                print(i*blocks_grid[1]+j,'/',blocks_grid[0]*blocks_grid[1],end='\r')
                # print(blocks_in.device, blocks_out.device)
                tmp_in = blocks_in[:, :, :, :, i, j]

                # flops, params = profile(self, torch.zeros(tmp_in.shape[0],tmp_in.shape[1],tmp_in.shape[2],tmp_in.shape[3]).to(tmp_in.device))
                # flops_sum+=flops

                blocks_ij = self.forward(tmp_in, **kwargs)
                blocks_ij= blocks_ij
                blocks_out[:, :, :, :, i, j] = blocks_ij

                # print(i*blocks_grid[1]+j,'/',blocks_grid[0]*blocks_grid[1],end='\r')
                # # print(blocks_in.device, blocks_out.device)
                # tmp_in = blocks_in[:, :, :, :, i, j].to(device_)
                # blocks_ij = self.forward(tmp_in, **kwargs)
                # blocks_ij= blocks_ij.to('cpu')
                # blocks_out[:, :, :, :, i, j] = blocks_ij

        # print(flops_sum)

        
        x = patches_handler(blocks_out, mode="aggregate")
        logger.debug(f"Blocks aggregated to shape : {tuple(x.shape)}")
        # return x.to(device_)
        return x

    def configure_optimizers(self):
        logger.debug("Configuring optimizer")
        optim_class = torch.optim.__dict__[self.optimizer.class_name]
        optimizer = optim_class(self.parameters(), **self.optimizer.params)

        if self.lr_scheduler is not None:
            scheduler_class = torch.optim.lr_scheduler.__dict__[
                self.lr_scheduler.class_name
            ]
            scheduler = scheduler_class(optimizer, **self.lr_scheduler.params)
        return [optimizer], [scheduler]

    def count_params(self):
        desc = "Model parameters:\n"
        counter = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                count = param.numel()
                # desc += f"\t{name} : {count}\n"
                counter += count
        desc += f"Total number of learnable parameters : {counter}\n"
        logger.info(desc)
        return counter, desc
