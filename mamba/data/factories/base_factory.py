from abc import ABC, abstractmethod
import logging
import os

from hydra.utils import to_absolute_path
from torch.utils.data import Subset, ConcatDataset
import torchvision.transforms as transforms

from mamba.data.datasets import (
    LMDBDataset,
    NoisyTransformDataset,
    PathsDataset,
)
from mamba.data.patches_utils import generate_patches
from mamba.data.transforms import RandomRot90, RandomSpectralInversion

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DatasetFactory(ABC):
    def __init__(self, noise, path_data, seed, split, bands=None, ssl=False):
        self.noise = noise
        self.path_data = to_absolute_path(path_data)
        self.seed = seed
        self.ssl = ssl
        self.bands = bands
        self.split = split

        # Must be defined in subclasses
        self.f_train = None
        self.f_test = None
        self.f_val = None
        self._setup = False

    def setup(self):
        self.download(self.path_data)
        self.preprocess()
        self._setup = True

    def train(self, transform, **kwargs):
        if transform == 0:
            train_transforms = None
        elif transform == 1:
            train_transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ]
            )
        elif transform == 2:
            train_transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    RandomRot90(),
                ]
            )
        elif transform == 3:
            train_transforms = transforms.Compose(
                [   
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    RandomRot90(),
                    RandomSpectralInversion(),
                ]
            )
        else:
            raise ValueError(f"transform {transform} not recognized")
        return self._train(transforms=train_transforms, **kwargs)

    def _train(self, repeat, **kwargs):
        if self.ssl:
            dataset = self.train_ssl(**kwargs)
        else:
            dataset = self.train_sl(**kwargs)

        logger.debug(f"Len training set : {len(dataset)}")
        if repeat is not None:
            logger.info(f"Repeating training dataset : {repeat}")
            dataset = ConcatDataset([dataset for _ in range(repeat)])
            logger.info(f"Len dataset after repeat : {len(dataset)}")
        return dataset

    def train_sl(self, transforms, **kwargs):
        assert self._setup

        def _db_name(patch_size, subsample, stride, crop_center):
            subsample_str = "-".join([str(s) for s in subsample])
            stride_str = "-".join([str(s) for s in stride])
            return (
                f"{self.split}_ps{patch_size}_"
                f"s{stride_str}_sub{subsample_str}_"
                f"c{crop_center}.db"
            )

        db_name = _db_name(**kwargs)
        path_db = os.path.join(self.path_data, self.NAME, "patches", db_name)
        dataset_paths = PathsDataset(paths=self.full_path_clean(mode="train"))
        generate_patches(
            path_db=path_db,
            dataset=dataset_paths,
            key="y",
            img_shape=self.IMG_SHAPE,
            **kwargs,
        )
        lmdb_dataset = LMDBDataset(path_lmdb=path_db, img_id=self.NAME)
        # assert self.bands == lmdb_dataset.bands
        noisy_transform_dataset = NoisyTransformDataset(
            dataset=lmdb_dataset,
            noise=self.noise,
            seed=self.seed,
            bands=self.bands,
            mode="train",
            transforms=transforms,
        )
        return noisy_transform_dataset

    def train_ssl(self, transforms, **kwargs):
        assert self._setup

        def _db_name(
            noise_str, seed, patch_size, subsample, stride, crop_center
        ):
            subsample_str = "-".join([str(s) for s in subsample])
            stride_str = "-".join([str(s) for s in stride])
            return (
                f"{self.split}_{noise_str}_seed{seed}_ps{patch_size}_"
                f"s{stride_str}_sub{subsample_str}_"
                f"c{crop_center}.db"
            )

        # load images
        dataset_paths = PathsDataset(paths=self.full_path_clean(mode="train"))

        # generate noisy images
        noisy_dataset = NoisyTransformDataset(
            dataset=dataset_paths,
            noise=self.noise,
            seed=self.seed,
            bands=self.bands,
            mode="ssl",
            transforms=None,
        )

        # extract and save noisy patches
        db_name = _db_name(
            noise_str=noisy_dataset.noise_model.__repr__(),
            seed=self.seed,
            **kwargs,
        )
        path_db = os.path.join(self.path_data, self.NAME, "patches", db_name)
        generate_patches(
            path_db=path_db,
            dataset=noisy_dataset,
            key="x",
            img_shape=self.IMG_SHAPE,
            **kwargs,
        )

        # load noisy patches
        lmdb_dataset = LMDBDataset(path_lmdb=path_db, img_id=self.NAME)

        # data augmentation on noisy patches
        transform_dataset = NoisyTransformDataset(
            dataset=lmdb_dataset,
            transforms=transforms,
            noise=None,
            seed=None,
            bands=None,
            mode=None,
        )
        return transform_dataset

    def val(self):
        paths = self.full_path_clean(mode="val")
        dataset = PathsDataset(paths=paths)
        noisy_dataset = NoisyTransformDataset(
            dataset=dataset,
            noise=self.noise,
            seed=self.seed,
            bands=self.bands,
            mode="val",
            transforms=None,
        )
        return noisy_dataset

    def test(self, idx=None):
        paths = self.full_path_clean(mode="test")
        dataset = PathsDataset(paths=paths)
        noisy_dataset = NoisyTransformDataset(
            dataset=dataset,
            noise=self.noise,
            seed=self.seed,
            bands=self.bands,
            mode="test",
            transforms=None,
            compute_noise=True,
        )
        if idx not in [None, ""]:
            logger.debug(f"idx={idx}")
            if isinstance(idx, int):
                idx = [idx]
            noisy_dataset = Subset(noisy_dataset, indices=idx)
        return noisy_dataset

    def full_path_clean(self, mode):
        assert mode in ["train", "test", "val"]
        return [
            os.path.join(self.path_data, self.NAME, "clean", f"{fn}.pth")
            for fn in self.__dict__[f"f_{mode}"]
        ]

    @abstractmethod
    def download(self, path_data):
        pass

    @abstractmethod
    def preprocess(self):
        pass
