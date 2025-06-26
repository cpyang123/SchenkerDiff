import graph_tool as gt
import os
import pathlib
import warnings
import numpy as np

import random
import pickle

import torch

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
import os
import torch.distributed as dist

from src.datasets.schenker_dataset import SchenkerGraphDataModule, SchenkerDatasetInfos
from src.analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
from src.analysis.visualization import NonMolecularVisualization

from hydra import initialize, compose
from omegaconf import OmegaConf



def initialize_model():
    # Set environment variables required by the env:// init_method
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    warnings.filterwarnings("ignore", category=PossibleUserWarning)
    torch.set_float32_matmul_precision('medium')
    torch.cuda.empty_cache()

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="env://", rank=0, world_size=1)

    # Initialize Hydra with the desired config path and version_base
    with initialize(config_path="../SchenkerDiff/configs", version_base="1.3"):
        # Compose the configuration by specifying the config name
        cfg = compose(config_name="config")

    dataset_config = cfg["dataset"]
    datamodule = SchenkerGraphDataModule(cfg)
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

    loaded_model = DiscreteDenoisingDiffusion.load_from_checkpoint(checkpoint_path="last-v1.ckpt", **model_kwargs)

    return loaded_model


def main():
    batch_size = 1
    keep_chain = 10
    number_chain_steps = 100

    loaded_model = initialize_model()

    """
    :param batch_id: int
    :param batch_size: int
    :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
    :param save_final: int: number of predictions to save to file
    :param keep_chain: int: number of chains to save to file
    :param keep_chain_steps: number of timesteps to save for each chain
    :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
    """
    E, r, names, n_nodes_list = sample_r_E(batch_size)
    print(E.shape)
    num_nodes = torch.tensor([int(x) for x in n_nodes_list]).to(loaded_model.device)
    if num_nodes is None:
        n_nodes = loaded_model.node_dist.sample_n(batch_size, loaded_model.device)
    elif type(num_nodes) == int:
        n_nodes = num_nodes * torch.ones(batch_size, device=loaded_model.device, dtype=torch.int)
    else:
        assert isinstance(num_nodes, torch.Tensor)
        n_nodes = num_nodes
    n_max = torch.max(n_nodes).item()
    # Build the masks
    arange = torch.arange(n_max, device=loaded_model.device).unsqueeze(0).expand(batch_size, -1)
    node_mask = arange < n_nodes.unsqueeze(1)

    # Sample a piece, and use the R matrix from that
    # Get a random sample from the data
    # pass through Stephen's script to get the S matrix, and the R matrix through the data processing (process_file_for_GUI)

    # Sample noise  -- z has size (n_samples, n_nodes, n_features)
    z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=loaded_model.limit_dist, node_mask=node_mask)
    X, _, y = z_T.X, z_T.E, z_T.y

    E_transpose = E.permute(0, 2, 1, 3)  # Shape remains (bs, n_nodes, n_nodes, 2)

    # Symmetrize using max operation (ensures strongest connection remains)
    E = torch.maximum(E, E_transpose).to(DEVICE)
    r = r.to(DEVICE)

    assert (E == torch.transpose(E, 1, 2)).all()
    assert number_chain_steps < loaded_model.T
    chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
    chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

    chain_X = torch.zeros(chain_X_size)
    chain_E = torch.zeros(chain_E_size)

    # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
    for s_int in reversed(range(0, loaded_model.T)):
        s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
        t_array = s_array + 1
        s_norm = s_array / loaded_model.T
        t_norm = t_array / loaded_model.T

        # Sample z_s
        sampled_s, discrete_sampled_s = loaded_model.sample_p_zs_given_zt(s_norm, t_norm, X, E, r, y, node_mask)
        X, _, y = sampled_s.X, sampled_s.E, sampled_s.y

        discrete_sampled_s_E, _ = loaded_model.apply_node_mask_E_r(E, r, node_mask)

        # Save the first keep_chain graphs
        write_index = (s_int * number_chain_steps) // loaded_model.T
        chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
        chain_E[write_index] = discrete_sampled_s_E[:keep_chain]

    # # Sample
    sampled_s = sampled_s.mask(node_mask, collapse=True)

    X, _, y = sampled_s.X, sampled_s.E, sampled_s.y

    E, _ = loaded_model.apply_node_mask_E_r(E, r, node_mask)

    # Prepare the chain for saving
    if keep_chain > 0:
        final_X_chain = X[:keep_chain]
        final_E_chain = E[:keep_chain]

        chain_X[0] = final_X_chain  # Overwrite last frame with the resulting X, E
        chain_E[0] = final_E_chain

        chain_X = diffusion_utils.reverse_tensor(chain_X)
        chain_E = diffusion_utils.reverse_tensor(chain_E)

        # Repeat last frame to see final sample better
        chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
        chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
        assert chain_X.size(0) == (number_chain_steps + 10)

    molecule_list = []
    for i in range(batch_size):
        n = n_nodes[i]
        atom_types = X[i, :n].cpu()
        edge_types = E[i, :n, :n].cpu()
        rhythm_types = r[i, :n, :].cpu()
        sample_names = names[i]
        molecule_list.append([atom_types, edge_types, rhythm_types, sample_names])

def sample_r_E(batch_size):
    """
    Samples `batch_size` random pickle files from the directory
    where i is a random integer between 0 and 1080. Each pickle file contains a dictionary
    that is converted to a PyG Data object using the pre-defined function `hetero_to_data`.

    The PyG Data object is expected to have at least the following attributes:
    - x: Tensor of node features with shape (num_nodes, feature_dim)
    - edge_index: LongTensor with shape (2, num_edges)
    - edge_attr: Tensor with shape (num_edges, 2) representing edge attributes
    - r: Tensor with shape (num_nodes, dr) representing additional node-level features

    For each sample, the function creates:
    - An adjacency tensor E_sample of shape (n_nodes, n_nodes, 2) where each edge's attribute
        is placed at the corresponding (u, v) location. If the original graph has fewer than
        `n_nodes` nodes, the tensors are padded with zeros; if it has more, they are truncated.
    - A node feature tensor r_sample of shape (n_nodes, dr) similarly padded or truncated.

    Finally, the function stacks these into:
    - E_tensor: Tensor of shape (batch_size, n_nodes, n_nodes, 2)
    - r_tensor: Tensor of shape (batch_size, n_nodes, dr)

    Returns:
        E_tensor, r_tensor
    """
    E_list = []
    r_list = []
    name_list = []
    node_sizes = []

    # get samples from OOS distribution
    np.random.seed(42)
    n_samples = 76

    # Randomly select 90 indices for the test set
    # test_indices = np.random.choice(n_samples, 150, replace=False)
    test_indices = np.array([1])

    for _ in range(batch_size):
        # Select a random index between 0 and 1080 (inclusive)

        idx = 1
        file_path = f"data/schenker/processed/heterdatacleaned/processed/{idx}_processed.pt"

        # Load the pickle file containing a dictionary
        data_dict = torch.load(file_path)

        # Convert dictionary to a PyG Data object using the provided function
        data = SchenkerDiffHeteroGraphData.hetero_to_data(data_dict)

        # Determine the actual number of nodes in the current sample
        m = data.x.shape[0]

        # Initialize an adjacency tensor for this sample
        E_sample = torch.zeros((m, m, 10))
        # Fill in the edge attributes: iterate over each edge
        for i in range(data.edge_index.shape[1]):
            u = data.edge_index[0, i].item()
            v = data.edge_index[1, i].item()
            # Only consider nodes within the allowed range (pad/truncate as needed)
            if u < m and v < m:
                E_sample[u, v, :] = data.edge_attr[i, :]

        # Process the r tensor (node-level additional features)
        dr = data.r.shape[1]  # feature dimension of r
        r_sample = torch.zeros((m, dr))
        # Copy available node features; pad with zeros if necessary or truncate if too many nodes
        r_sample[:m, :] = data.r[:m, :]

        # Append this sample's results to the lists
        E_list.append(E_sample)
        r_list.append(r_sample)
        name_list.append(data_dict['name'])
        node_sizes.append(m)

    # Stack all samples to form the batch tensors
    # Determine the maximum number of nodes in the batch
    max_nodes = max(tensor.shape[0] for tensor in r_list)

    # Pad the E_list tensors to shape (max_nodes, max_nodes, 3)
    E_padded = []
    for e in E_list:
        n = e.shape[0]
        # F.pad expects pad in the format: (pad_last_dim_left, pad_last_dim_right,
        # pad_second_last_dim_left, pad_second_last_dim_right, ...)
        # For a tensor of shape (n, n, 3): pad last dimension (3) with (0,0),
        # second dimension with (0, max_nodes-n), and first dimension with (0, max_nodes-n).
        pad_amount = (0, 0, 0, max_nodes - n, 0, max_nodes - n)
        E_padded.append(F.pad(e, pad_amount))

    # Stack the padded tensors along a new batch dimension
    E_tensor = torch.stack(E_padded, dim=0)  # Shape: (batch_size, max_nodes, max_nodes, 3)

    # Pad the r_list tensors to shape (max_nodes, dr)
    r_padded = []
    for r in r_list:
        n = r.shape[0]
        # For a tensor of shape (n, dr), pad the first dimension with (0, max_nodes-n)
        pad_amount = (0, 0, 0, max_nodes - n)
        r_padded.append(F.pad(r, pad_amount))

    # Stack the padded tensors along the batch dimension
    r_tensor = torch.stack(r_padded, dim=0)  # Shape: (batch_size, max_nodes, dr)

    return E_tensor, r_tensor, name_list, node_sizes


if __name__ == "__main__":
    main()
