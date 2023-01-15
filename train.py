import logging

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar

from pl_models import DPRemovalModel

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg) -> None:
    exp_name = HydraConfig().get().run.dir
    log.info(f"Start experiment: {exp_name}")

    pl.seed_everything(cfg.seed)

    torch.autograd.set_detect_anomaly(True)

    log.info("Create new model")
    model = DPRemovalModel(cfg)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=f"{cfg.path.exp_root}/{cfg.name}/tensorboard", name=""
    )
    trainer = instantiate(
        cfg.trainer,
        callbacks=[TQDMProgressBar(refresh_rate=cfg.refresh_rate)],
        logger=tb_logger,
        check_val_every_n_epoch=cfg.every_n_epochs_valid,
    )

    log.info("Start Training")
    trainer.fit(model)


if __name__ == "__main__":
    main()
