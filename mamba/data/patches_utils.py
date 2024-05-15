import logging
import os
import sys
import shutil
import pickle
import yaml

import torch
from typing import List, Tuple
import lmdb

from mamba.data.datasets import (
    get_patch_coords,
    PatchesDataset,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def estimate_storage(
    img_shape, patch_size, stride, subsample, n_images, margin=1.1
):
    """estimage required storage in bytes"""
    c, h, w = img_shape
    # n patches per image
    n_patches = 0
    for _stride, _sub in zip(stride, subsample):
        n_patches += len(
            get_patch_coords(h // _sub, w // _sub, patch_size, _stride)
        )
    logger.debug(f"N_images: {n_images}")
    logger.debug(f"N_patches image: {n_patches}")
    n_patches *= n_images
    logger.debug(f"N_patches total : {n_patches}")
    size_patch = sys.getsizeof(
        pickle.dumps(
            torch.ones(c, patch_size, patch_size, dtype=torch.float32)
        )
    )
    logger.debug(f"Size patch : {size_patch}")
    storage = int(n_patches * size_patch * margin)
    return storage


def apply_centering(img, crop_size):
    assert len(img.shape) == 3
    if crop_size is None:
        return img
    _img = img.clone()

    y_start = (_img.shape[1] - crop_size) // 2
    y_end = (_img.shape[1] + crop_size) // 2

    x_start = (_img.shape[2] - crop_size) // 2
    x_end = (_img.shape[2] + crop_size) // 2

    y_start = max(0, y_start)
    x_start = max(0, x_start)
    _img = _img[:, y_start:y_end, x_start:x_end].clone()
    logger.debug(f"Img centered to shape : {tuple(_img.shape)}")

    return _img


def apply_subsampling(img, subsample):
    if subsample is None:
        return img

    img = img[:, ::subsample, ::subsample].clone()
    logger.debug(f"Img subsampled to shape : {tuple(img.shape)}")
    return img


def generate_patches(
    patch_size: int,
    stride: List[int],
    subsample: List[int],
    dataset: object,
    key: str,
    path_db: str,
    img_shape: Tuple[int],
    crop_center: int,
):
    assert len(stride) == len(subsample)

    if crop_center is not None:
        assert img_shape[1] == crop_center
        assert img_shape[2] == crop_center

    logger.info(f"Path db : {path_db}")
    path_meta = os.path.join(path_db, "meta.yaml")
    if os.path.exists(path_meta):
        logger.info(f"Found db at {path_db}")
        return
    elif os.path.isdir(path_db):
        logger.info(f"Found incomplete database, removing it")
        shutil.rmtree(path_db)
    logger.info(f"Database not found or not complete, starting generation")
    os.makedirs(path_db, exist_ok=True)

    # estimate storage in bytes
    n_bytes = estimate_storage(
        img_shape=img_shape,
        patch_size=patch_size,
        stride=stride,
        subsample=subsample,
        n_images=len(dataset),
    )
    logger.info(f"Required storage: {n_bytes/ (1024 ** 3):.4f} Gib")

    env = lmdb.open(path_db, map_size=n_bytes, writemap=True)

    with env.begin(write=True) as txn:
        idx = 0
        tot_size = 0
        logger.info(f"Iterating through dataset : {dataset}")
        for i, item in enumerate(dataset):
            img_id = item["img_id"]
            logger.info(
                f"Patching img {i + 1}/{len(dataset)} (key={key}): {img_id}"
            )
            img = item[key]
            logger.debug(f"Img shape : {tuple(img.shape)}")
            img = apply_centering(img, crop_center)
            assert len(img.shape) == 3
            logger.debug(f"Current size : {tot_size / (1024 **3):.4f} Gib")
            for _stride, _sub in zip(stride, subsample):
                _img = apply_subsampling(img, _sub)
                patches_dataset = PatchesDataset(
                    img=_img.clone(),
                    patch_size=patch_size,
                    stride=_stride,
                )
                for j, patch in enumerate(patches_dataset):
                    assert tuple(patch.shape) == (
                        img_shape[0],
                        patch_size,
                        patch_size,
                    ), f"Patch shape : {tuple(patch.shape)}"
                    serialized = pickle.dumps(patch)
                    size_patch = sys.getsizeof(serialized)
                    tot_size += size_patch

                    if j == 0:
                        logger.debug(f"Size patch : {size_patch}")
                    txn.put(str(idx).encode("ascii"), serialized)
                    idx += 1
    meta = {
        "bands": int(img_shape[0]),
        "n_items": idx,
        "patch_size": patch_size,
        "stride": "_".join([str(s) for s in stride]),
        "subsample": "_".join([str(s) for s in subsample]),
    }
    logger.debug(f"Meta : {meta}")
    with open(path_meta, "w") as outfile:
        yaml.dump(meta, outfile)
        logger.debug(f"Metadata  written to {path_meta}")
