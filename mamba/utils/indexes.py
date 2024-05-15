import numpy as np
import torch
# from skimage.metrics import compare_ssim, compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from functools import partial
from scipy import signal

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

class Bandwise(object): 
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        bwindex = []
        for ch in range(C):
            x = torch.squeeze(X[...,ch,:,:].data).cpu().numpy()
            y = torch.squeeze(Y[...,ch,:,:].data).cpu().numpy()
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex

def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
def mse (GT,P):
	"""calculates mean squared error (mse).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.

	:returns:  float -- mse value.
	"""
	# GT,P = _initial_check(GT,P)
	return np.mean((GT.astype(np.float32)-P.astype(np.float32))**2)
cal_bwssim = Bandwise(compare_ssim)
cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=1))


def cal_sam(X, Y, eps=1e-8):
    X = torch.squeeze(X.data).cpu().numpy()
    Y = torch.squeeze(Y.data).cpu().numpy()
    tmp = (np.sum(X*Y, axis=0) + eps) /( (np.sqrt(np.sum(X**2, axis=0)))* (np.sqrt(np.sum(Y**2, axis=0))) + eps)    
    return np.mean(np.real(np.arccos(tmp)))

def cal_ssim(im_true,im_test,eps=13-8):
    # print(im_true.shape)
    im_true=im_true.squeeze(0).squeeze(0).cpu().numpy()
    im_test = im_test.squeeze(0).squeeze(0).cpu().numpy()
    c,_,_=im_true.shape
    bwindex = []
    for i in range(c):
        bwindex.append(ssim(im_true[i,:,:]*255, im_test[i,:,:,]*255))
    return np.mean(bwindex)
def MSIQA(X, Y):

    psnr = np.mean(cal_bwpsnr(X, Y))
    ssim = cal_ssim(X,Y)
    sam = cal_sam(X, Y)
    return psnr, ssim, sam

def MSIQAs(X, Y):
    total_psnr = 0
    total_ssim = 0
    total_sam  = 0
    for x,y in zip(X,Y):
        psnr, ssim, sam = MSIQA(x, y)
        total_psnr += psnr
        total_ssim += ssim
        total_sam  += sam
    avg_mpsnr = total_psnr / X.shape[0]
    avg_ssim = total_ssim / X.shape[0]
    avg_sam  = total_sam / X.shape[0]
    return avg_mpsnr,avg_ssim,avg_sam

if __name__ == "__main__":
    X = np.load("/home/ironkitty/data/paper3/projects/T3SC/tmp/X.npy")
    Y = np.load("/home/ironkitty/data/paper3/projects/T3SC/tmp/Y.npy")
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    msqa = MSIQA(X, Y)
    print(msqa)
    