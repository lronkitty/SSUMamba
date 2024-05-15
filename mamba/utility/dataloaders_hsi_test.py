from torchvision import transforms
from torch.utils.data import Dataset
from os import listdir, path
from PIL import Image
import torch
import math
import torchvision.transforms.functional as TF
import random
from typing import Sequence
from itertools import repeat
import scipy.io as scio
import numpy as np
import torch
import re
from torch._six import container_abcs, string_classes, int_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')
def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data
class MyResize:
    def __init__(self, scale,crop):
        self.scale = scale
        self.crop = crop


    def __call__(self, x):
        bands = x.shape[2]
        if bands > 31:
            bs = int(np.random.rand(1) * bands)
            if bs + 31 > bands:
                bs = bands - 31
            x = x[:, :, bs:bs + 31]
        im_sz=x.shape
        rs=[int(im_sz[0]*self.scale),int(im_sz[1]*self.scale)]
        if rs[0]<self.crop:
            rs[0]=self.crop
        if rs[1] < self.crop:
            rs[1] = self.crop

        im = np.zeros([rs[0], rs[1], im_sz[2]],dtype=x.dtype)
        for i in range(im_sz[2]):
            im[:,:,i]=np.array(Image.fromarray(x[:,:,i]).resize(rs)).T
        return im
class MyRandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return np.flipud(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
class MyRotation90(object):
    def __call__(self, img):
        return np.rot90(img)
        return img
class MyCenterCrop(object):
    def __init__(self, a=512, b=512):
        self.a = a
        self.b = b
    def __call__(self, img):
        _w, _h, _b = img.shape
        c1 = math.ceil((_w - self.a) / 2)-1;
        c2 = math.ceil((_h - self.b) / 2)-1;
        iout = img[c1:c1 + self.a, c2: c2 +  self.b,:];
        return iout
class MyRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return np.fliplr(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
class MyRandomCrop(object):
      def __init__(self, size):
        self.size=size

      def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        _w, _h, _b = img.shape
        x = random.randint(1, _w)
        y = random.randint(1, _h)
        x2 = x + self.size
        y2 = y + self.size
        if x2 > _w:
            x2 = _w
            x = _w - self.size
        if y2 > _h:
            y2 = _h
            y = _h - self.size
        cropImg = img[(x):(x2), (y):(y2), :]
        return cropImg

        # return self.cropit(img,self.size)
        # return img
      def cropit(image, crop_size):
          _w, _h, _b = image.shape
          x = random.randint(1, _w)
          y = random.randint(1, _h)
          x2 = x + crop_size
          y2 = y + crop_size
          if x2 > _w:
              x2 = _w
              x = _w - crop_size
          if y2 > _h:
              y2 = _h
              y = _h - crop_size
          cropImg = image[(x):(x2), (y):(y2), :]
          return cropImg
class MyToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return TF.to_tensor(pic.copy())

    def __repr__(self):
        return self.__class__.__name__ + '()'




class Dataset(Dataset):
    def __init__(self, root_dirs, transform=None, verbose=False, grey=False):
        self.root_dirs = root_dirs
        self.transform = transform
        self.images_path = []
        for cur_path in root_dirs:
            self.images_path += [path.join(cur_path, file) for file in listdir(cur_path) if file.endswith(('tif','png','jpg','jpeg','bmp','mat'))]
        self.verbose = verbose
        self.grey = grey

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_name = self.images_path[idx]

        if self.grey:
            image = Image.open(img_name).convert('L')
        else:
            # image = Image.open(img_name).convert('RGB')
            image = scio.loadmat(img_name)['DataCube'].astype(np.float32)
            # image=image/image.max()
            # image = flipit(flipit(cropit(image,crop_size=128),[0,1]),[1,0])

            # image=transforms.ToPILImage(image)
        if self.transform:
            image = self.transform(image)


        if self.verbose:
            return image, img_name.split('/')[-1]

        return image
def get_gt(gt_path, img_name,verbose=False, grey=False):
    tfs = []
    tfs += [
    # MyRotation90(),
    # MyCenterCrop(),
    MyToTensor()
    ]
    gt_transforms = transforms.Compose(tfs)
    image = scio.loadmat(gt_path+img_name)['DataCube'].astype(np.float32)
    image = gt_transforms(image)
    # image=image/image.max()
    return image

def get_dataloaders(test_path_list, crop_size=96, batch_size=1, downscale=0,
                    drop_last=True, concat=True, n_worker=0, scale_min=0.001, scale_max=0.1, verbose=False, grey=False):

    batch_sizes = {'test':1, 'gt': 1}
    test_transforms = transforms.Compose([MyToTensor()])
    data_transforms = {'test': test_transforms}
    image_datasets = {'test': Dataset(test_path_list, data_transforms['test'], verbose=verbose, grey=grey)}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x],
                                                  num_workers=n_worker,drop_last=drop_last, shuffle=False) for x in ['test']}
    return dataloaders

def flipit(image, axes):

    if axes[0]:
        image = np.fliplr(image)
    if axes[1]:
        image = np.flipud(image)

    return image
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


# def cropit(image, seg=None, margin=5):
#
#     fixedaxes = np.argmin(image.shape[:2])
#     trimaxes = 0 if fixedaxes == 1 else 1
#     trim = image.shape[fixedaxes]
#     center = image.shape[trimaxes] // 2
#     if seg is not None:
#
#         hits = np.where(seg != 0)
#         mins = np.argmin(hits, axis=1)
#         maxs = np.argmax(hits, axis=1)
#
#         if center - (trim // 2) > mins[0]:
#             while center - (trim // 2) > mins[0]:
#                 center = center - 1
#             center = center + margin
#
#         if center + (trim // 2) < maxs[0]:
#             while center + (trim // 2) < maxs[0]:
#                 center = center + 1
#             center = center + margin
#
#     top = max(0, center - (trim // 2))
#     bottom = trim if top == 0 else center + (trim // 2)
#
#     if bottom > image.shape[trimaxes]:
#         bottom = image.shape[trimaxes]
#         top = image.shape[trimaxes] - trim
#
#     if trimaxes == 0:
#         image = image[top: bottom, :, :]
#     else:
#         image = image[:, top: bottom, :]
#
#     if seg is not None:
#         if trimaxes == 0:
#             seg = seg[top: bottom, :, :]
#         else:
#             seg = seg[:, top: bottom, :]
#
#         return image, seg
#     else:
#         return image

