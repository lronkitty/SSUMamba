import logging
import os

import torch
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseNormalizer:
    def __init__(self):
        assert hasattr(self, "STATEFUL"), "Missing STATEFUL class attribute"

    def fit(self, x):
        raise NotImplementedError

    def transform(self, x):
        raise NotImplementedError

    def get_id(self):
        attributes = [self.__class__.__name__]
        attributes += [
            k[:3] + str(v)
            for k, v in self.__dict__.items()
            if not isinstance(v, torch.Tensor)
        ]
        return "_".join(attributes).replace(".", "")

    def __repr__(self):
        return self.get_id()

    def filename(self):
        return f"{self.get_id()}.pth"

    def save(self, path=None):
        filename = self.filename()
        if path:
            filename = os.path.join(path, filename)
        torch.save(self.__dict__, filename)
        logger.info(f"Normalizer {self} saved to {filename!r}")

    def load(self, path=None):
        filename = self.filename()
        if path:
            filename = os.path.join(path, filename)
        logger.info(f"Loading normalizer {self} from {filename!r}")
        state = torch.load(filename)
        for k, v in state.items():
            setattr(self, k, v)


class GlobalMinMax(BaseNormalizer):
    STATEFUL = False

    def __init__(self, epsilon=0.001):
        super().__init__()
        self.epsilon = epsilon

    def transform(self, x):
        mi = torch.amin(x, dim=(0, 1, 2), keepdim=True)
        ma = torch.amax(x, dim=(0, 1, 2), keepdim=True)
        return (x - mi) / (self.epsilon + (ma - mi))


class BandMinMaxQuantileStateful(BaseNormalizer):
    STATEFUL = True

    def __init__(self, low=0.02, up=0.98, epsilon=0.001):
        super().__init__()
        self.low = low
        self.up = up
        self.epsilon = epsilon

    def fit(self, imgs):
        x_train = []
        for i, img in enumerate(imgs):
            logger.debug(f"Loading img {i + 1}/{len(imgs)}")
            x_train.append(img.flatten(start_dim=1))
        logger.info(f"Concatenating training data ..")
        x_train = torch.cat(x_train, dim=1)
        # x : (c, bs * h * w)
        bands = x_train.shape[0]
        q_global = np.zeros((bands, 2))
        for b in range(bands):
            logger.debug(f"Computing quantile on band {b}")
            q_global[b] = np.percentile(
                x_train[b].cpu().numpy(), q=100 * np.array([self.low, self.up])
            )

        self.q = torch.tensor(q_global, dtype=torch.float32).T[..., None, None]
        logger.debug(f"Quantile fitted with shape {self.q.shape}")

    def transform(self, x):
        # x : (c, h, w)
        x = torch.minimum(x, self.q[1])
        x = torch.maximum(x, self.q[0])
        return (x - self.q[0]) / (self.epsilon + (self.q[1] - self.q[0]))
