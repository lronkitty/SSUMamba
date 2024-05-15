import torch 
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange

class UVMB(nn.Module):
    def __init__(self,c=3,w=256,h=256,t=31):
        super().__init__()
        self.convb  = nn.Sequential(
                    nn.Conv3d(in_channels=c, out_channels=16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(in_channels=16, out_channels=c, kernel_size=3, stride=1, padding=1)
                        )
        self.model1 = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        '''
        self.model2 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

        self.model3 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=w*h*t, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        '''
        self.smooth = nn.Conv3d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        # self.ln = nn.LayerNorm(normalized_shape=c)
        self.softmax = nn.Softmax()
    def forward(self, x):
        b,c,t,w,h = x.shape
        x = self.convb(x) + x
        x_ = rearrange(x, 'b c t w h -> b (t w h) c')
        x = x.reshape(b, -1, c)
        y = self.model1(x).permute(0, 2, 1)
        # z = self.model3(y).permute(0, 2, 1)
        # att = self.softmax(self.model2(x))
        # result = att * z
        output = rearrange(y, 'b c (t w h)-> b c t w h', t=t, w=w, h=h)
        return self.smooth(output)




