import os

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path
from importlib import import_module

# Single agent
# python eval.py model=model_forecast data_root=/home/ailab/AILabDataset/01_Open_Dataset/24_Argoverse2/motion_forecasting batch_size=64 'checkpoint="./checkpoints/model_forecast_finetune.ckpt"'
# python eval.py model=model_forecast data_root=/home/ailab/AILabDataset/01_Open_Dataset/24_Argoverse2/motion_forecasting batch_size=64 'checkpoint="./checkpoints/model_forecast_finetune.ckpt"' test=true

# Multi agent
# python eval.py model=model_forecast_multiagent data_root=/home/ailab/AILabDataset/01_Open_Dataset/24_Argoverse2/motion_forecasting batch_size=64 'checkpoint="./checkpoints/multiagent_baseline.ckpt"'
# python eval.py model=model_forecast_multiagent data_root=/home/ailab/AILabDataset/01_Open_Dataset/24_Argoverse2/motion_forecasting batch_size=64 'checkpoint="./checkpoints/multiagent_baseline.ckpt"' test=true

@hydra.main(version_base=None, config_path="./conf/", config_name="config")
def main(conf):
    pl.seed_everything(conf.seed)

    checkpoint = to_absolute_path(conf.checkpoint)
    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"

    model_path = conf.model.target._target_
    module = import_module(model_path[: model_path.rfind(".")])
    Model: pl.LightningModule = getattr(module, model_path[model_path.rfind(".") + 1 :])
    model = Model.load_from_checkpoint(checkpoint)

    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu",
        devices=conf.gpus,
        max_epochs=1,
        limit_val_batches=conf.limit_val_batches,
        limit_test_batches=conf.limit_test_batches,
    )

    datamodule: pl.LightningDataModule = instantiate(conf.datamodule, test=conf.test)

    if not conf.test:
        trainer.validate(model, datamodule)
    else:
        trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
