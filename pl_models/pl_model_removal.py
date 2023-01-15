import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchaudio
from hydra.utils import instantiate, to_absolute_path

log = logging.getLogger(__name__)


class DPRemovalModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.musical = torchaudio.load(config.model.musical_path)[0]
        clean = torchaudio.load(config.model.clean_path)[0]
        num_input = config.model.num_input
        self.reduction = config.model.reduction
        self.musical_spec = self.stft(self.musical).expand(num_input, 1, -1, -1)
        target = torch.abs(self.musical_spec)
        self.target, self.pad1, self.pad2 = self.pad32(target)
        self.clean = clean
        self.net_input = torch.rand_like(self.target)  # Sampled from Uniform(0,1)

        self.save_hyperparameters()
        self.sampling_rate = config.sampling_rate
        self.channels = config.channels
        self.bits_per_sample = config.bits_per_sample
        self.network = instantiate(config.model.net_cfg)
        self.stft = instantiate(config.model.stft_cfg)
        self.optim_cfg = config.model.optim_cfg
        self.losses = torch.nn.ModuleDict({})
        for loss_tag, loss_config in config.model.losses.items():
            self.losses[loss_tag] = instantiate(loss_config)
        self.main_loss = config.model.main_loss

    def pad32(self, x):
        assert len(x.shape) == 4
        shape = x.shape
        pad1 = 32 - shape[2] % 32
        pad2 = 32 - shape[3] % 32
        m = torch.nn.ZeroPad2d((pad2, 0, pad1, 0))
        return m(x), pad1, pad2

    def forward(self, net_input):
        return self.network(net_input)

    def _to_time(self, net_output):
        net_output = net_output[:, 0, self.pad1 :, self.pad2 :]
        # (num_input, n_freq, n_frame)
        if self.reduction == "mean":
            amp_spec = net_output.mean(dim=0)
        elif self.reduction == "median":
            amp_spec = torch.median(net_output, dim=0).values
        elif self.reduction == "min":
            amp_spec = torch.min(net_output, dim=0).values
        else:
            raise ValueError(f"{self.reduction} is not supported.")
        spec = amp_spec * torch.exp(1j * torch.angle(self.musical_spec))
        time = self.stft.inv(spec, len(self.musical))
        return time

    def training_step(self, batch, batch_idx):
        net_output = self(self.net_input)
        output_time = self._to_time(net_output)
        for loss_tag, loss_func in self.losses.items():
            if loss_tag == self.main_loss:
                loss_val = loss_func(self.target, net_output)
                loss = loss_val
            else:
                loss_val = loss_func(self.clean, output_time)
            self.log(f"train/{loss_tag}", loss_val, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        net_output = self(self.net_input)
        output_time = self._to_time(net_output)
        for loss_tag, loss_func in self.losses.items():
            if loss_tag == self.main_loss:
                loss_val = loss_func(self.target, net_output)
            else:
                loss_val = loss_func(self.clean, output_time)
            self.log(f"valid/{loss_tag}", loss_val, on_epoch=True)

    def test_step(self, batch, batch_idx):
        net_output = self(self.net_input)
        output_time = self._to_time(net_output)
        for loss_tag, loss_func in self.losses.items():
            if loss_tag == self.main_loss:
                loss_val = loss_func(self.target, net_output)
            else:
                loss_val = loss_func(self.clean, output_time)
            self.log(f"valid/{loss_tag}", loss_val, on_epoch=True)

        savedir = Path(to_absolute_path(f"{self.trainer.log_dir}/results/test"))
        savedir.mkdir(parents=True, exist_ok=True)
        output_path = savedir / "output.wav"
        output = output_time.detach().cpu().reshape(self.channels, -1)
        torchaudio.save(
            str(output_path),
            output,
            self.sampling_rate,
            bits_per_sample=self.bits_per_sample,
        )

    def configure_optimizers(self):
        optimizer = instantiate({**{"params": self.parameters()}, **self.optim_cfg})
        return optimizer
