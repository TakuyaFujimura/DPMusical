import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib.colors import Normalize
from tqdm import tqdm

from util import STFT


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


class DPRemover:
    def __init__(self, config) -> None:
        self.n_iter = config.n_iter
        self.n_input = config.n_input
        self.pooling_type = config.pooling
        assert self.pooling_type in ["mean", "median", "min"]
        self.stft = STFT(**config.stft_cfg)
        self.network = instantiate(config.net_cfg)
        self.network.to(config.device)
        self.device = config.device
        self.optimizer = instantiate(
            {**{"params": self.network.parameters()}, **config.optim_cfg}
        )
        self.loss_func = instantiate(config.loss_cfg)
        if config.show_every is None:
            self.show_every = self.n_iter
        else:
            self.show_every = config.show_every

    def __call__(self, x):
        """Returns denoised signal
        Args:
            x (tensor): waveform (n_time)
        """
        X, angle, pad_list = self.wave_to_image(x)
        Z = self.get_input(X)
        self.train(X, Z)

        self.network.eval()
        with torch.no_grad():
            S = self.network(X)
        s = self.image_to_wave(S, angle, pad_list, len(x))
        return s

    def pad(self, x):
        """Applies zero-padding to adjust the shape
        Args:
            x (tensor): image (1, 1, H, W)
            pad_list (list): list of the padding value
        """
        assert len(x.shape) == 4
        shape = x.shape
        pad_h = 32 - shape[2] % 32
        pad_w = 32 - shape[3] % 32
        m = torch.nn.ZeroPad2d((pad_w, 0, pad_h, 0))
        pad_list = [pad_h, pad_w]
        return m(x), pad_list

    def wave_to_image(self, x):
        """Applies STFT and zero-padding to waveform
        Args:
            x (tensor): waveform (..., n_time)
        Returns:
            X (tensor): amplitude spectrogram
            angle (tensor): phase of the spectrogram
            pad_list (list): list of the padding value
        """
        spectrogram = self.stft(x)
        angle = torch.angle(spectrogram)
        X = torch.abs(spectrogram)[None, None]  # [1, 1, H, W]
        X, pad_list = self.pad(X)
        return X, angle, pad_list

    def pooling(self, X):
        # [B, 1, H, W] -> [1, 1, H, W]
        if self.pooling_type == "mean":
            X = torch.mean(X, dim=0, keepdim=True)
        elif self.pooling_type == "median":
            X = torch.median(X, dim=0, keepdim=True).values
        elif self.pooling_type == "min":
            X = torch.min(X, dim=0, keepdim=True).values
        else:
            raise ValueError(f"{self.pooling_type} is not supported.")
        return X

    def image_to_wave(self, X, angle, pad_list, n_time):
        """Applies iSTFT with orignal phase
        Args:
            X (tensor): multiple amplitude spectrograms (n_input, 1, H, W)
            angle (tensor): angle of the original spectrogram
            pad_list (list): list of the padding value
            n_time (int): number of samples
        Returns:
            x (tensor): waveform (n_time)
        """
        X = X[:, :, pad_list[0] :, pad_list[1] :]
        amp_spec = self.pooling(X)[0, 0]
        spectrogram = amp_spec * torch.exp(1j * angle)
        x = self.stft.inv(spectrogram, n_time)
        return x

    def get_input(self, X):
        Z = torch.rand_like(X).expand(self.n_input, -1, -1, -1)
        return Z

    def train(self, target, net_input):
        """Trains a network
        Args:
            target (tensor): target signal (1, 1, H, W)
            net_input (tensor): input signal (n_input, 1, H, W)
        """
        net_input = net_input.to(self.device)
        self.network.train()
        target = target.to(self.device)
        for i in tqdm(range(1, self.n_iter + 1)):
            net_output = self.network(net_input)
            loss = self.loss_func(net_output, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if i % self.show_every == 0:
                self.network.eval()
                with torch.no_grad():
                    net_output = self.network(net_input)
                    S = self.pooling(net_output)[0, 0]
                self.spec_show(S, str(i))

    def spec_show(self, amp_spec, title=""):
        plt.imshow(
            20 * torch.log10(amp_spec + 1e-8),
            origin="lower",
            cmap="viridis",
            aspect="auto",
            norm=Normalize(vmin=-160, vmax=-30),
        )
        plt.title(title)
        plt.ylabel("Frequency")
        plt.xlabel("Time")
        plt.show()
        plt.clf()
        plt.close()
