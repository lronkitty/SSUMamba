### 任意大小。x编码.nomask
from tkinter import X
try:
    from hydra.utils import to_absolute_path
except:
    print("Hydra not found, using relative paths")
    pass
import logging

import torch
import torch.nn as nn

from .base import BaseModel
import mamba.models.layers as layers
# from mamba.models.layers.combinations import *
# from mamba.models.layers.brt_modules import BlockRecurrentAttention
# from mamba.models.layers.network_swinir import *
from mamba.models.layers.vssm.vmamba import VSSM
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class UMamba(BaseModel):
    def __init__(self,base,
        ssl=0,
        n_ssl=0,ckpt=None,):
        super().__init__(**base)
        self.layers_params = layers
        self.ssl = ssl
        self.n_ssl = n_ssl
        logger.debug(f"ssl : {self.ssl}, n_ssl : {self.n_ssl}")

        # self.init_layers()
        model = VSSM(
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        #x.shape = torch.Size([2, 1, 48, 160, 224])
        # skips:
        # torch.Size([2, 32, 48, 160, 224])
        # torch.Size([2, 64, 48, 80, 112])
        # torch.Size([2, 128, 24, 40, 56])
        # torch.Size([2, 256, 12, 20, 28])
        # torch.Size([2, 320, 6, 10, 14])
        # torch.Size([2, 320, 6, 5, 7])
        logger.info(f"Using SSL : {self.ssl}")
        self.ckpt = ckpt
        if self.ckpt is not None:
            try:
                logger.info(f"Loading ckpt {self.ckpt!r}")
                d = torch.load(to_absolute_path(self.ckpt))
                self.load_state_dict(d["state_dict"])
            except:
                print("Could not load ckpt")
                pass

    def forward(self,x, mode=None, img_id=None, sigmas=None, ssl_idx=None, **kwargs
    ):
        x = self.net(x)
        return x
