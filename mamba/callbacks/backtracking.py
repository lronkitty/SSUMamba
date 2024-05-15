import logging
import os
import os.path as osp

import numpy as np
from pytorch_lightning.callbacks import Callback
import torch


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FORMAT = "{:03d}_{:.4f}.pth"


class Backtracking(Callback):
    def __init__(
        self, dirpath, monitor, mode, period, div_thresh, lr_decay, dummy
    ):
        super().__init__()
        self.dirpath = dirpath
        self.period = period
        self.monitor = monitor
        assert mode in ["min", "max"]
        self.mode = mode
        self.div_thresh = div_thresh
        self.lr_decay = lr_decay
        self.dummy = dummy

        logger.debug(self)

        self.path_prev_metrics = osp.join(self.dirpath, "metrics.pth")
        self.path_prev_state = osp.join(self.dirpath, "state.pth")

        os.makedirs(self.dirpath, exist_ok=True)

    def __repr__(self):
        s = f"Backtracking(dirpath={self.dirpath}, monitor={self.monitor}, "
        s += f"mode={self.mode}, period={self.period}, div_tresh="
        s += (
            f"{self.div_thresh}, lr_decay={self.lr_decay}, dummy={self.dummy})"
        )
        return s

    def on_epoch_end(self, trainer, module):
        epoch = module.current_epoch
        if (epoch % self.period == 0) or self.dummy:
            try:
                value = trainer.callback_metrics[self.monitor].item()
            except KeyError:
                logger.warning(
                    f"Metrics {self.monitor!r} not found for backtracking"
                )
                return
            try:
                prev_metrics = torch.load(self.path_prev_metrics)
                prev_value = prev_metrics["value"]
                prev_epoch = prev_metrics["epoch"]
            # except FileNotFoundError:
            except Exception:
                logger.debug("Backtracking checkpoint not found")
                self.save_state(module, value)
                return
            logger.debug(
                f"Epoch {epoch}, proceeding to verification ({self.monitor}): "
                f"current value={value:.4f}, "
                f"previous value={prev_value:.4f}"
            )
            if self.is_diverging(value, prev_metrics["value"]):
                logger.info(
                    f"Metrics {self.monitor} is diverging, "
                    f"loading weights from epoch {prev_epoch}"
                )
                prev_state = torch.load(self.path_prev_state)
                module.load_state_dict(prev_state["state_dict"])

                lr = trainer.optimizers[0].param_groups[0]["lr"]
                new_lr = lr * self.lr_decay
                trainer.optimizers[0].param_groups[0]["lr"] = new_lr
                logger.info(f"Learning rate decayed from {lr} to {new_lr}")
            else:
                self.save_state(module, value)

    def save_state(self, module, value):
        torch.save(
            {"value": value, "epoch": module.current_epoch},
            self.path_prev_metrics,
        )
        logger.debug(f"Saved metrics to {self.path_prev_metrics!r}")
        torch.save({"state_dict": module.state_dict()}, self.path_prev_state)
        logger.debug(f"Saved module state dict to {self.path_prev_state!r}")

    def is_diverging(self, value, prev_value):
        if self.dummy:
            return np.random.rand() > 0.3
        if self.mode == "min":
            return (value - prev_value) > self.div_thresh
        else:
            return (value - prev_value) < -self.div_thresh
