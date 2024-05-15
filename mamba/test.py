import logging

import torch
from pytorch_lightning import seed_everything

from mamba import models
from mamba.data import DataModule
from mamba.utils import Tester

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device in use : {device}")

    # Fix seed for reproducibility
    logger.info(f"Using random seed {cfg.seed}")
    seed_everything(cfg.seed)

    # Load requested dataloader
    datamodule = DataModule(idx_test=cfg.test.idx_test, **cfg.data.params)
    datamodule.setup(stage="test")

    model_class = models.__dict__[cfg.model.class_name]
    model = model_class.load_from_checkpoint(cfg.ckpt_path)

    model = model.to(device)

    tester = Tester(**cfg.test)
    tester.eval(model, datamodule)
