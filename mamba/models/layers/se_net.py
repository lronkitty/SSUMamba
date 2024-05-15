import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze and Excitation module

class SqEx(nn.Module):

    def __init__(self, n_features, reduction=31):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

# Residual block using Squeeze and Excitation

class ResBlockSqEx(nn.Module):

    def __init__(self, n_features):
        super(ResBlockSqEx, self).__init__()

        # convolutions

        self.norm1 = nn.BatchNorm2d(n_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm2 = nn.BatchNorm2d(n_features)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        # squeeze and excitation

        self.sqex  = SqEx(n_features)

    def forward(self, x):
        
        x = torch.squeeze(x,1)
        # convolutions
        y = self.conv1(self.relu1(self.norm1(x)))
        y = self.conv2(self.relu2(self.norm2(y)))

        # squeeze and excitation

        y = self.sqex(y)

        # add residuals
        
        y = torch.add(x, y)
        y = torch.unsqueeze(y,1)
        return y

class ResBlockSqEx_res(nn.Module):

    def __init__(self, n_features):
        super(ResBlockSqEx_res, self).__init__()

        # convolutions

        self.norm1 = nn.BatchNorm2d(n_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm2 = nn.BatchNorm2d(n_features)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        # squeeze and excitation

        self.sqex  = SqEx(n_features)

    def forward(self, residual,x):
        
        # convolutions

        x = torch.squeeze(x,1)
        y = self.conv1(self.relu1(self.norm1(x)))
        y = self.conv2(self.relu2(self.norm2(y)))

        # squeeze and excitation

        y = self.sqex(y)

        # add residuals
        
        # y = torch.add(x, y)

        y = torch.unsqueeze(y,1)
        y = torch.add(residual,y)
        return y