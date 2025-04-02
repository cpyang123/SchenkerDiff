import graph_tool as gt
import os
import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

import src.utils
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from src.diffusion_model import LiftedDenoisingDiffusion
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures

warnings.filterwarnings("ignore", category=PossibleUserWarning)

torch.set_float32_matmul_precision('medium')


@hydra.main(version_base='1.3', config_path='../SchenkerDiff/configs', config_name='config')
def main(cfg: DictConfig):