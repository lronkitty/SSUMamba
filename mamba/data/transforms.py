import torch


class RandomRot90:
    def __call__(self, x):
        assert len(x.shape) == 3
        k = torch.randint(4, size=(1,)).item()
        x = torch.rot90(x, k=k, dims=(-2, -1))
        return x


class RandomSpectralInversion:
    def __call__(self, x):
        assert len(x.shape) == 3
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=(0,))
        return x
