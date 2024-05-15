from mamba.utility import *
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseNoise:
    def __init__(self, test=None, ssl=None, seed=None):
        self.name = type(self).__name__
        self.test = test
        self.ssl = ssl
        self.seed = seed

    def update_sigmas(self):
        raise NotImplementedError

    def apply(self, x):
        raise NotImplementedError

    def params_str(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        msg = f"{self.name}_{self.params_str()}_b{self.bands}_s{self.seed}"
        return msg


class ConstantNoise:
    def __init__(self, sigma, bands, **kwargs):
        self.bands = bands
        self.sigma = sigma

        self.sigmas = self.sigma * torch.ones((self.bands, 1, 1)) / 255
        self.sigma_avg = self.sigma / 255

    def apply(self, x, seed, **kwargs):
        generator = np.random.RandomState(seed=seed)
        noise_pixels = torch.tensor(
            generator.randn(*x.shape),
            dtype=torch.float32,
            device=x.device,
        )

        noisy = x + self.sigmas * noise_pixels

        return x, noisy.float()

    def __repr__(self) -> str:
        name = type(self).__name__
        msg = f"{name}_s{self.sigma}"
        return msg


class UniformNoise:
    def __init__(self, sigma_min, sigma_max, **kwargs):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def apply(self, x, seed):
        generator = np.random.RandomState(seed=seed)
        sigmas = torch.tensor(
            generator.rand(x.shape[0], 1, 1),
            dtype=torch.float32,
            device=x.device,
        )
        noise_pixels = torch.tensor(
            generator.randn(*x.shape),
            dtype=torch.float32,
            device=x.device,
        )

        sigmas = self.sigma_min + sigmas * (self.sigma_max - self.sigma_min)
        self.sigmas = sigmas / 255
        self.sigma_avg = (self.sigma_min + self.sigma_max) / (2 * 255)
        noisy = x + self.sigmas * noise_pixels

        return x, noisy

    def __repr__(self) -> str:
        name = type(self).__name__
        msg = f"{name}_min{self.sigma_min}_max{self.sigma_max}"
        return msg


class CorrelatedNoise(BaseNoise):
    def __init__(self, beta, eta, bands, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.eta = eta
        self.bands = bands
        self.update_sigmas()

    def update_sigmas(self):
        idx = np.arange(self.bands)

        sigmas = self.beta * np.exp(
            -((idx / self.bands - 0.5) ** 2) / (4 * self.eta ** 2)
        )
        self.sigmas = torch.tensor(sigmas / 255, dtype=torch.float32).view(
            -1, 1, 1
        )

        self.sigma_avg = self.sigmas.mean()

    def apply(self, x, **kwargs):
        noisy = x + self.sigmas * torch.randn(*x.shape)
        return x, noisy

    def params_str(self):
        return f"beta{self.beta}_eta{self.eta}"


class StripesNoise:
    def __init__(
        self,
        bands,
        ratio_bands=0.33,
        ratio_columns=[0.05, 0.15],
        stripe_intensity=0.5,
        sigma=25,
        **kwargs,
    ):
        self.bands = bands
        self.ratio_bands = ratio_bands
        self.ratio_columns = ratio_columns
        self.stripe_intensity = stripe_intensity
        self.sigma = sigma
        self.std = self.sigma / 255
        self.sigmas = torch.ones(self.bands) * self.std
        self.sigma_avg = self.std

    def stripe_noise(self, c, h, w):
        stripe_noise = torch.zeros((c, 1, w))

        n_bands = int(c * self.ratio_bands)
        bands_affected = self.generator.permutation(c)[:n_bands]
        n_cols = self.generator.randint(
            int(w * self.ratio_columns[0]),
            int(w * self.ratio_columns[1]),
            (len(bands_affected),),
        )
        logger.debug(f"Bands affected : {bands_affected}")
        logger.debug(f"N stripes col : {n_cols}")
        for i, band_idx in enumerate(bands_affected):
            col_idx = self.generator.permutation(w)[: n_cols[i]]
            stripe_noise[band_idx, :, col_idx] = (
                torch.rand(len(col_idx)) - 0.5
            ) * self.stripe_intensity

        return stripe_noise.float()

    def gaussian_noise(self, c, h, w):
        pixel_noise = self.std * torch.tensor(self.generator.randn(c, h, w))
        return pixel_noise.float()

    def apply(self, x, seed, **kwargs):
        self.generator = np.random.RandomState(seed=seed)
        s_noise = self.stripe_noise(*x.shape)
        g_noise = self.gaussian_noise(*x.shape)
        noisy = x + g_noise + s_noise
        return x, noisy

    def __repr__(self):
        name = type(self).__name__
        return (
            f"{name}_rb{self.ratio_bands}-rc{self.ratio_columns[0]}-"
            f"{self.ratio_columns[1]}_"
            f"s{self.sigma}"
        )
        
class MixtureNoise:
    def __init__(self, **kwargs):
        self.add_Dynamic = AddNoiseDynamic(95)
        self.add_Impulse =  AddNoiseImpulse()
        self.add_Stripe =   AddNoiseStripe()
        self.add_Deadline = AddNoiseDeadline()
        
    def apply(self, x, seed):
        noisy = x.cpu().numpy()
        noisy =self.add_Dynamic(noisy)
        noisy =self.add_Impulse(noisy)
        noisy =self.add_Stripe(noisy)
        noisy =self.add_Deadline(noisy)
        

        noisy = torch.from_numpy(noisy).float()
        return x, noisy