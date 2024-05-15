from tkinter import X
from hydra.utils import to_absolute_path
import logging

import torch
import torch.nn as nn

from .base import BaseModel
import mamba.models.layers as layers
from mamba.models.layers.combinations import Conv2dBNReLU

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class brcn(BaseModel):
    def __init__(self,base,channels,layers,
        ssl=0,
        n_ssl=0,ckpt=None,):
        super().__init__(**base)
        self.channels = channels
        self.layers_params = layers
        self.ssl = ssl
        self.n_ssl = n_ssl
        logger.debug(f"ssl : {self.ssl}, n_ssl : {self.n_ssl}")

        # self.init_layers()
        self.wf_v1 = Conv2dBNReLU(1,16,3,1,1)
        self.wf_t1 = Conv2dBNReLU(1,16,3,1,1)
        self.wf_r1 = Conv2dBNReLU(16,16,3,1,1)
        self.wb_v1 = Conv2dBNReLU(1,16,3,1,1)
        self.wb_t1 = Conv2dBNReLU(1,16,3,1,1)
        self.wb_r1 = Conv2dBNReLU(16,16,3,1,1)
        self.wf_v2 = Conv2dBNReLU(16,16,3,1,1)
        self.wf_t2 = Conv2dBNReLU(16,16,3,1,1)
        self.wf_r2 = Conv2dBNReLU(16,16,3,1,1)
        self.wb_v2 = Conv2dBNReLU(16,16,3,1,1)
        self.wb_t2 = Conv2dBNReLU(16,16,3,1,1)
        self.wb_r2 = Conv2dBNReLU(16,16,3,1,1)
        self.wf_v3 = Conv2dBNReLU(16,1,3,1,1)
        self.wf_t3 = Conv2dBNReLU(16,1,3,1,1)
        self.wb_v3 = Conv2dBNReLU(16,1,3,1,1)
        self.wb_t3 = Conv2dBNReLU(16,1,3,1,1)
        self.lambda_ = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
        self.lambda_.data.fill_(0.5)
        self.normalized_dict = False

        logger.info(f"Using SSL : {self.ssl}")
        self.ckpt = ckpt
        if self.ckpt is not None:
            logger.info(f"Loading ckpt {self.ckpt!r}")
            d = torch.load(to_absolute_path(self.ckpt))
            self.load_state_dict(d["state_dict"])
    

    def first(self,x):
        hf_1 = torch.empty((x.shape[0],16,x.shape[2],x.shape[3],x.shape[4])).cuda()
        hb_1 = torch.empty((x.shape[0],16,x.shape[2],x.shape[3],x.shape[4])).cuda()
        hf_1[:,:,0,:,:] = self.lambda_*self.wf_v1(x[:,:,0,:,:])
        for i in range(1,x.shape[2]):
            hf_1[:,:,i,:,:] = self.lambda_*(self.wf_v1(x[:,:,i,:,:]) + self.wf_r1(hf_1[:,:,i-1,:,:]) + self.wf_t1(x[:,:,i-1,:,:]))
        hb_1[:,:,-1,:,:] = self.lambda_*self.wb_v1(x[:,:,-1,:,:])
        for i in range(x.shape[2]-2,-1,-1):
            hb_1[:,:,i,:,:] = self.lambda_*(self.wb_v1(x[:,:,i,:,:]) + self.wf_r1(hb_1[:,:,i+1,:,:]) + self.wb_t1(x[:,:,i+1,:,:]))
        return hf_1,hb_1

    def second(self,hf_1,hb_1):
        hf_2 = torch.empty_like(hf_1).cuda()
        hb_2 = torch.empty_like(hf_1).cuda()
        hf_2[:,:,0,:,:] = self.lambda_*self.wf_v2(hf_1[:,:,0,:,:])
        for i in range(1,hf_1.shape[2]):
            hf_2[:,:,i,:,:] = self.lambda_*(self.wf_v2(hf_1[:,:,i,:,:]) + self.wf_r2(hf_2[:,:,i-1,:,:]) + self.wf_t2(hf_1[:,:,i-1,:,:]))
        hb_2[:,:,-1,:,:] = self.lambda_*self.wb_v2(hb_1[:,:,-1,:,:])
        for i in range(hf_1.shape[2]-2,-1,-1):
            hb_2[:,:,i,:,:] = self.lambda_*(self.wb_v2(hb_1[:,:,i,:,:]) + self.wb_r2(hb_2[:,:,i+1,:,:]) + self.wb_t2(hb_1[:,:,i+1,:,:]))
        return hf_2,hb_2

    def output(self,hf_2,hb_2):
        O = torch.empty((hf_2.shape[0],1,hf_2.shape[2],hf_2.shape[3],hf_2.shape[4])).cuda()
        O[:,:,0,:,:] = self.wf_v3(hf_2[:,:,0,:,:]) + self.wb_v3(hb_2[:,:,0,:,:]) + self.wb_t3(hb_2[:,:,1,:,:])
        O[:,:,-1,:,:] = self.wf_v3(hf_2[:,:,-1,:,:])+ self.wf_t3(hb_2[:,:,-2,:,:]) + self.wb_v3(hb_2[:,:,-1,:,:]) 
        for i in range(1,hf_2.shape[2]-1):
            O[:,:,i,:,:] = self.wf_v3(hf_2[:,:,i,:,:]) + self.wf_t3(hb_2[:,:,i-1,:,:]) + self.wb_v3(hb_2[:,:,i,:,:]) + self.wb_t3(hb_2[:,:,i+1,:,:])
        return O

    def forward(self,x):
        x = torch.unsqueeze(x,1)
        hf_1,hb_1 = self.first(x)
        hf_2,hb_2 = self.second(hf_1,hb_1)
        x = self.output(hf_2,hb_2)
        x = torch.squeeze(x,1)
        return x
