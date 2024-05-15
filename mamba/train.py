import logging
import os

from omegaconf import errors
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch
from mamba import models
from mamba.data import DataModule
from mamba.utils import Tester
from mamba.callbacks import Backtracking
# from pytorch_lightning.callbacks import TQDMProgressBar

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device in use : {device}")

    # Fix seed for reproducibility
    logger.info(f"Using random seed {cfg.seed}")
    pl.seed_everything(cfg.seed)

    # Load datamodule
    datamodule = DataModule(**cfg.data.params)

    # Logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="tb", name="", version=""
    )

    # Callbacks
    callbacks = [
        cb.ModelCheckpoint(**cfg.checkpoint),
        # cb.ModelCheckpoint(**cfg.checkpoint_best),
        cb.LearningRateMonitor(),
        # cb.ProgressBar(),
        cb.TQDMProgressBar()
    ]
    try:
        logger.info("Loading backtracking config")
        callbacks.append(Backtracking(**cfg.model.backtracking))
        logger.info("Backtracking callback instantiated successfully")
    except (errors.ConfigAttributeError, TypeError):
        logger.info("Backtracking config not found")

    model_class = models.__dict__[cfg.model.class_name]
    # model = model_class.load_from_checkpoint(cfg.load_ckpt).to(device)
    model = model_class(**cfg.model.params)#.to(device)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="tb", name="", version=""
    )

    # Instantiate trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=tb_logger,
        accelerator="gpu",
        strategy='ddp_find_unused_parameters_true',
        # devices=cfg.gpu_ids,
        **cfg.trainer.params,
    )

    # trainer.tune(model)

    # Print model info
    model.count_params()

    # Fit trainer
    
    trainer.fit(model, datamodule=datamodule,ckpt_path=cfg.ckpt_path)

    # Load best checkpoint
    filename_best = os.listdir("best")[0]
    path_best = os.path.join("best", filename_best)
    logger.info(f"Loading best model for testing : {path_best}")
    model.load_state_dict(torch.load(path_best)["state_dict"])

    tester = Tester(**cfg.test)
    tester.eval(model, datamodule=datamodule)
