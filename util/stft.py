from typing import Optional

import torch


class STFT(torch.nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        window_type: Optional[str] = "hann",
        pad_mode: Optional[str] = "constant",
        device: Optional[str] = "cpu",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad_mode = pad_mode

        if window_type == "hann":
            self.window = torch.hann_window(n_fft, True)
        elif window_type == "hamming":
            self.window = torch.hamming_window(n_fft, True)
        else:
            raise ValueError("window_type must be either hann or hamming")
        self.window = self.window.to(device)

    def forward(self, x):
        X = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            self.window,
            pad_mode=self.pad_mode,
            onesided=True,
            return_complex=True,
        )
        return X

    def inv(self, X, length):
        x = torch.istft(
            X,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            self.window,
            onesided=True,
            length=length,
        )
        return x
