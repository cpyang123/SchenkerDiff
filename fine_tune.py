import graph_tool as gt
import os
import pathlib
import warnings
import numpy as np

import random
import pickle

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from src.diffusion import diffusion_utils

import src.utils
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from src.diffusion_model import LiftedDenoisingDiffusion
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
import src.utils
from torch_geometric.utils import  to_dense_batch
from src.datasets.schenker_dataset import SchenkerDiffHeteroGraphData
import torch.nn.functional as F
from src.schenker_gnn.config import DEVICE

warnings.filterwarnings("ignore", category=PossibleUserWarning)

torch.set_float32_matmul_precision('medium')


import os
import torch.distributed as dist

# Set environment variables required by the env:// init_method
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "2900"

# if not dist.is_initialized():
#     dist.init_process_group(backend="gloo", init_method="env://", rank=0, world_size=1)


from src.datasets.schenker_dataset import SchenkerGraphDataModule, SchenkerDatasetInfos
from src.analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
from src.analysis.visualization import NonMolecularVisualization

from hydra import initialize, compose
from omegaconf import OmegaConf



def main():
    # Initialize Hydra with the desired config path and version_base
    with initialize(config_path="../SchenkerDiff/configs", version_base="1.3"):
        # Compose the configuration by specifying the config name
        cfg = compose(config_name="config")

    dataset_config = cfg["dataset"]
    datamodule = SchenkerGraphDataModule(cfg, is_tune = True)
    sampling_metrics = PlanarSamplingMetrics(datamodule)

    dataset_infos = SchenkerDatasetInfos(datamodule, dataset_config)
    train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
    visualization_tools = NonMolecularVisualization()

    if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}


    loaded_model = DiscreteDenoisingDiffusion.load_from_checkpoint(checkpoint_path= "last.ckpt", **model_kwargs)




    # Run this if you want to fine tune: 

    name = cfg.general.name
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    datamodule_tune = SchenkerGraphDataModule(cfg, is_tune = True)
    trainer = Trainer(gradient_clip_val = cfg.train.clip_grad,
                    strategy = "ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                    accelerator = 'gpu' if use_gpu else 'cpu',
                    devices = cfg.general.gpus if use_gpu else 1,
                    max_epochs = 100,
                    check_val_every_n_epoch = cfg.general.check_val_every_n_epochs,
                    fast_dev_run = cfg.general.name == 'debug',
                    enable_progress_bar = True,
                    log_every_n_steps = 50 if name != 'debug' else 1,
                    logger = [])

    trainer.fit(loaded_model, datamodule = datamodule_tune, ckpt_path = cfg.general.resume)


    trainer.save_checkpoint("fine_tuned.ckpt")


if __name__ == "__main__":
    main()
