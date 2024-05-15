import torch
from . import im2col
def refold(col_data,data_shape, kernel_size, stride,device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    #import torch
    # print(col_data.shape, data_shape[-2])
    col_data = col_data.view(data_shape[-2])
    #print(col_data.shape)
    col_data = col_data.permute(0, 4, 5, 6, 1, 2, 3)
    #print(col_data.shape)
    col_data = col_data.reshape(data_shape[-4])
    #print(col_data.shape) 
    cube_data = im2col.Col2Cube(col_data.to(device),data_shape[1], kernel_size, stride, padding=0, dilation=1, avg=True,input_tensorized=True,device=device)
    #print(cube_data.shape)
    cube_data = cube_data[0,:data_shape[0][0],:data_shape[0][1],:data_shape[0][2]]
    #print(cube_data.shape)

    return cube_data

if __name__ == "__main__":
    import scipy.io as scio
    import os
    import sys
    import numpy as np
    import torch
    os.chdir(sys.path[0])
    col_data_LISTA = scio.loadmat('col_Data.mat')['colData']
    #col_data_LISTA = np.transpose(col_data_LISTA)
    #col_data_LISTA = col_data_LISTA.reshape((1,125,66,66,118))
    col_data_LISTA = torch.from_numpy(col_data_LISTA)
    #scio.savemat('col_data_LISTA.mat', {'colData':col_data_LISTA})
    refold_data = refold(col_data_LISTA,output_size=(206, 206, 31), kernel_size=(56,56,31), stride=(15,15,15), padding=0, dilation=1, avg=True,input_tensorized=True)
    refold_data = refold_data[:,:200,:200,:31]
    refold_data=refold_data.cpu().numpy()
    test = np.sum(refold_data)
    scio.savemat('refold_data.mat', {'refoldData':refold_data})
    #torch.Size([1, 125, 66, 66, 118])186890.55419061845
