import os
import pathlib

import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url, HeteroData

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
import concurrent.futures
import glob
import pickle
from copy import deepcopy
from pathlib import Path
import math

import music21
import re
from music21 import key, converter

import music21.converter
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
import src.pyScoreParser.score_as_graph as score_graph
from src.datasets.data_maps import *
from errors import PickledError
from src.pyScoreParser.musicxml_parser.mxp import MusicXMLDocument, exception
from src.pyScoreParser.musicxml_parser.mxp.note import Note

import torch.nn.functional as F

from src.schenker_gnn.for_diffusion.infer_structure_from_rhythm import load_score, extract_structure_sparse


TRAIN_NAMES = "train-names.txt"
SAVE_FOLDER = "processed_data"
TEST_NAMES = "test-names.txt"
TEST_SAVE_FOLDER = "processed_data_test"



class EnharmonicError(Exception):
    def __init__(self, message):
        super().__init__(message)

class SchenkerGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split = None, root = "", transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.split = split
        self.num_graphs = 200  # Adjust based on your data
        self.dataset = SchenkerDiffHeteroGraphData(root=root, train_names=dataset_name)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.processed_file_names = self.dataset.processed_file_names
        self.data_list = dataset.data_list
        
    def len(self):
        return len(self.processed_file_names)

    def __len__(self):
        return self.len()

    def get(self, idx):
        return self.data_list[idx]

    def __getitem__(self, idx):
        return self.get(idx)

    @property
    def processed_file_names(self):
        np.random.seed(42)
        n_samples = 150

        # Randomly select 90 indices for the test set
        test_indices = np.random.choice(n_samples, 30, replace=False)

        if self.test_mode:
            return [f'{i}_processed.pt' for i in test_indices]

        # For training, use the complement of the test indices
        train_indices = np.setdiff1d(np.arange(n_samples), test_indices)[5:]
        if self.split == 'ture':
            return [f'{i}_processed.pt' for i in range(9)]
        return [f'{i}_processed.pt' for i in train_indices]

    def download(self):
        # If data is already prepared locally, you can skip this
        print("Data should be available locally. Skipping download.")

    def process(self):
        # Load your dataset (similar to `process_file` in `HeteroGraphData`)
        self.dataset.process()



class SchenkerGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, is_tune = False, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        with open(os.path.join(root_path, TRAIN_NAMES), "r") as file:
            names = file.readlines()
        names = [line.strip() for line in names if line[0] != "#"]

        with open(os.path.join(root_path, TRAIN_NAMES), "r") as file:
            names = file.readlines()
        names = [line.strip() for line in names if line[0] != "#"]

        if is_tune: 
            datasets = {'train': SchenkerDiffHeteroGraphData(dataset_name=self.cfg.dataset.name, train_names=names,
                                                 split='tune', root=root_path),
                        'val': SchenkerDiffHeteroGraphData(dataset_name=self.cfg.dataset.name, train_names=names,
                                            split='val', root=root_path),
                        'test': SchenkerDiffHeteroGraphData(dataset_name=self.cfg.dataset.name, train_names=names,
                                            split='test', root=root_path)}
        else:
            datasets = {'train': SchenkerDiffHeteroGraphData(dataset_name=self.cfg.dataset.name, train_names=names,
                                                    split='train', root=root_path),
                        'val': SchenkerDiffHeteroGraphData(dataset_name=self.cfg.dataset.name, train_names=names,
                                            split='val', root=root_path),
                        'test': SchenkerDiffHeteroGraphData(dataset_name=self.cfg.dataset.name, train_names=names,
                                            split='test', root=root_path)}
            
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class SchenkerDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        # self.node_types = torch.tensor([1])               # There are no node types 
        self.node_types = self.datamodule.node_types()      # LIKE HELL THERE AREN'T!
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)




INTERVAL_EDGES = [1, 2, 3, 4, 5, 8]
NUM_DEPTHS = 12
# NUM_FEATURES = 42
NUM_FEATURES = 18
NUM_EDGE_TYPES = 25
MAX_LEN = 40
INCLUDE_GLOBAL_NODES = False

class SchenkerDiffHeteroGraphData(Dataset):
    def __init__(self,
                 root,
                 dataset_name = None,
                 train_names=None,
                 transform=None,
                 pre_transform=None,
                 test_mode=False,
                 split = "train",
                 voice_mode=False,
                 include_depth_edges=True,
                 include_global_nodes=INCLUDE_GLOBAL_NODES
                 ):
        """
        root: where my dataset should be stored: it will automatically saved at root/processed
        """
        self.test_mode = False if split == 'train' else True
        self.fine_tune = False if split != 'tune' else True
        self.voice_mode = voice_mode
        self.train_names = train_names
        self.include_depth_edges = include_depth_edges
        self.include_global_nodes = include_global_nodes
        self.dataset_name = dataset_name
        super(SchenkerDiffHeteroGraphData, self).__init__(root, transform = transform, pre_transform = pre_transform)
        self.data_list = []

        for file_name in self.processed_file_names:
            file_path = os.path.join(self.processed_dir, file_name)
            if os.path.isfile(file_path):
                self.data_list.append(torch.load(file_path, weights_only = False))
            # else:
            #     print(f"Missing processed file: {file_path}")
        self.root = root
        self.scale_degree_map = SCALE_DEGREE_MAP

        with open('scale_degree_map.pkl', 'wb') as f:
            pickle.dump(self.scale_degree_map, f)

    def len(self):
        return len(self.processed_file_names)

    def __len__(self):
        return self.len()

    def get(self, idx):
        r_data = self.hetero_to_data(self.data_list[idx])
        return r_data

    def __getitem__(self, idx):
        return self.get(idx)

    @property
    def processed_file_names(self):
        np.random.seed(42)
        n_samples = 150

        # Randomly select 90 indices for the test set
        test_indices = np.random.choice(n_samples, 30, replace=False)

        if self.test_mode:
            return [f'{i}_processed.pt' for i in test_indices]

        # For training, use the complement of the test indices
        train_indices = np.setdiff1d(np.arange(n_samples), test_indices)[5:]

        if self.fine_tune:
            return [f'{i}_processed.pt' for i in range(9)]
        return [f'{i}_processed.pt' for i in train_indices]

    @staticmethod
    def one_hot_convert(mapped_pitch, num_class):
        # number of samples, number of class
        one_hot_encoded = np.zeros((len(mapped_pitch), num_class))
        for i, pitch in enumerate(mapped_pitch):
            one_hot_encoded[i, pitch] = 1
        return SchenkerDiffHeteroGraphData.to_float_tensor(one_hot_encoded)

    @staticmethod
    def to_float_tensor(array):
        if array.dtype == np.object_:
            array = np.array(array, dtype=float)
        return torch.tensor(array, dtype=torch.float)
    
    @staticmethod
    def resize_tensor(x, target_rows=MAX_LEN, target_cols=52):
        assert torch.all((x == 0) | (x == 1)), "Tensor contains values other than 0 or 1."

        current_rows, current_cols = x.shape
        
        
        # Pad or truncate the columns to target_cols
        if current_cols < target_cols:
            x = torch.nn.functional.pad(x, (0, target_cols - current_cols))
        elif current_cols > target_cols:
            x = x[:, :target_cols] # Truncate columns
        
        # Pad or truncate the rows to target_rows
        if current_rows < target_rows:
            # one-hot row for padding
            one_hot_row = torch.zeros(1, target_cols)
            one_hot_row[0, -1] = 1

            padding_rows = target_rows - current_rows
            padding_tensor = one_hot_row.repeat(padding_rows, 1)
            
            # Concatenate padding to tensor
            x = torch.cat([x, padding_tensor], dim=0)
        elif current_rows > target_rows:
            x = x[:target_rows, :] # Truncate rows
        
        return x
    
    @staticmethod
    def hetero_to_data(hetero_dict):
        # Initialize
        # x = self.resize_tensor(hetero_data['note']['x'], target_rows = MAX_LEN, target_cols = NUM_FEATURES + 1)
        hetero_data = hetero_dict['data']
        x =  hetero_data['note']['x']
        r = hetero_data['note']['r']

        # Add depth information to the R matrix
        # make a zero‐column of shape [..., 1]

        zeros = torch.zeros(*r.shape[:-1], 1, dtype=r.dtype, device=r.device)

        # concatenate on the last dim
        r = torch.cat([r, zeros], dim=-1)  # now shape [..., D+1]
        # if hetero_dict['s_edge_attr'] is None or hetero_dict['s_edge_attr'] == []:
        #     s_attr = torch.empty((0, 1))
        # else:
        #     s_attr = hetero_dict['s_edge_attr'].unsqueeze(-1)


        # # For s_edge_index: if None, create an empty tensor.
        # # Often, edge indices are expected to have shape (2, num_edges), so we use (2, 0) with type long.
        # if hetero_dict['s_edge_index'] is None or hetero_dict['s_edge_attr'] == []:
        #     s_inx = torch.empty((2, 0), dtype=torch.long)
        # else:
        #     s_inx = hetero_dict['s_edge_index']
        
        edge_indices = []
        edge_attrs = []
        edge_types = list(hetero_data.edge_types)[:] # getting rid of the first 'note'
        # print(edge_types)
        one_hot_dict = {edge_type: idx for idx, edge_type in enumerate(edge_types)}
        # print(one_hot_dict)
        
        existing = set()
        edge_indices = []
        edge_attrs = []

        for edge_type in edge_types:
            edge_index = hetero_data[edge_type]['edge_index']  # shape [2, E]
            src, dst = edge_index[0], edge_index[1]

            # collect the indices of columns we haven't seen
            new_cols = []
            for i in range(edge_index.size(1)):
                u, v = int(src[i]), int(dst[i])
                
                if (u, v) not in existing:

                    # # Add depth information to R
                    m = re.search(r'\d+', str(edge_type))
                    if not m:
                        pass
                    else:
                        num = int(m.group(0))
                        if u >= r.shape[0] or v >= r.shape[0]:
                            print(hetero_dict['name'])
                        if r[u, -1].item() == 0:
                            # make sure new_val is a tensor on the same device/dtype as r
                            r[u, -1] = torch.tensor(num, dtype=r.dtype, device=r.device)
                        if r[v, -1].item() == 0:
                            # make sure new_val is a tensor on the same device/dtype as r
                            r[v, -1] = torch.tensor(num, dtype=r.dtype, device=r.device)

                    existing.add((u, v))
                    new_cols.append(i)

            # if there are no new edges, skip
            if not new_cols:
                continue

            # build a filtered edge_index of shape [2, len(new_cols)]
            idx = torch.tensor(new_cols, dtype=torch.long)
            filtered_ei = edge_index[:, idx]
            edge_indices.append(filtered_ei)

            # one‐hot for this edge_type
            one_hot = torch.zeros(NUM_EDGE_TYPES)
            one_hot[one_hot_dict[edge_type]] = 1
            # repeat one_hot for each new edge
            edge_attr = one_hot.repeat(filtered_ei.size(1), 1)
            edge_attrs.append(edge_attr)

        edge_indices = torch.cat(edge_indices, dim=1)  
        edge_attrs = torch.cat(edge_attrs, dim=0) 

        # Normalize the depth dimension in the R tensor:
        # 1. grab the last‑dimension values
        depth_dim = r[:, -1]
        max_val = depth_dim.max()
        if max_val != 0:
            # in‑place replace the last column with normalized values
            r[:, -1] = depth_dim / max_val
        
        assert torch.all((x == 0) | (x == 1)), "Tensor contains values other than 0 or 1."
        data = Data(x=x, edge_index=edge_indices, edge_attr=edge_attrs, \
                     y=torch.zeros([1, 0]), r = r)
        
        # if 'asap-dataset' in hetero_data['name']:
        #     data = Data(x=x, edge_index=edge_indices, edge_attr=edge_attrs, \
        #              y=torch.zeros([1, 0]), r = r)
        # else: 
        #     final_indicies, final_attrs = SchenkerDiffHeteroGraphData.concat_adjacencies(edge_indices, edge_attrs, s_inx,s_attr, t_edges, b_edges)

        #     data = Data(x=x, edge_index=final_indicies, edge_attr=final_attrs, \
        #                 y=torch.zeros([1, 0]), r = r)

        return data

    @staticmethod
    def concat_adjacencies(edge_index1, edge_attr1, edge_index2, edge_attr2):
        """
        Concatenates two sets of edges and assigns a new two-class one-hot encoding.

        Parameters:
        edge_index1 (Tensor): Tensor of shape [2, num_edges1] for the first adjacency.
        edge_attr1 (Tensor): Tensor of shape [num_edges1, ?]. (Ignored, as only one class is present.)
        edge_index2 (Tensor): Tensor of shape [2, num_edges2] for the second adjacency.
        edge_attr2 (Tensor): Tensor of shape [num_edges2, ?]. (Ignored, as only one class is present.)

        Returns:
        combined_edge_index (Tensor): Concatenated edge indices tensor of shape [2, num_edges1 + num_edges2].
        combined_edge_attr (Tensor): Concatenated edge attributes tensor of shape [num_edges1 + num_edges2, 2],
                                    where edges from the first set are encoded as [0, 1, 0] and from the
                                    second set as [0, 0, 1].
        """
        num_edges1 = edge_index1.shape[1]
        num_edges2 = edge_index2.shape[1]
        
        # For the first set, assign the one-hot encoding [1, 0] to every edge.
        one_hot1 = torch.tensor([0, 1, 0], dtype=torch.float).repeat(num_edges1, 1)
        # For the second set, assign the one-hot encoding [0, 1] to every edge.
        one_hot2 = torch.tensor([0, 0, 1], dtype=torch.float).repeat(num_edges2, 1)

        # Convert each edge in edge_index1 into a tuple and store in a set for fast lookup.
        edges1 = {tuple(edge_index1[:, i].tolist()) for i in range(num_edges1)}
        
        # Filter edge_index2: only keep edges not already in edges1.
        filtered_edges2 = []
        filtered_one_hot2 = []
        for i in range(num_edges2):
            edge_tuple = tuple(edge_index2[:, i].tolist())
            if edge_tuple not in edges1:
                filtered_edges2.append(edge_index2[:, i].unsqueeze(1))
                filtered_one_hot2.append(one_hot2[i].unsqueeze(0))
        
        # If there are any filtered edges, concatenate them; otherwise create empty tensors.
        if filtered_edges2:
            filtered_edge_index2 = torch.cat(filtered_edges2, dim=1)
            filtered_one_hot2 = torch.cat(filtered_one_hot2, dim=0)
        else:
            filtered_edge_index2 = torch.empty((2, 0), dtype=edge_index1.dtype, device=edge_index1.device)
            filtered_one_hot2 = torch.empty((0, 3), dtype=one_hot2.dtype, device=edge_index1.device)
        
        
        # Concatenate the edge indices along the column dimension.
        combined_edge_index = torch.cat([edge_index1, filtered_edge_index2], dim=1)
        # Concatenate the new one-hot encoded attributes along the row dimension.
        combined_edge_attr = torch.cat([one_hot1, filtered_one_hot2], dim=0)
        
        return combined_edge_index, combined_edge_attr

    @staticmethod
    def add_interval_edges(notes: list[Note], edge_indices, intervals: list[int] = INTERVAL_EDGES):
        interval_up_from_to = {interval: [] for interval in intervals}
        interval_down_from_to = {interval: [] for interval in intervals}

        for i, current_note in enumerate(notes):
            ups_found = {interval: False for interval in intervals}
            downs_found = {interval: False for interval in intervals}
            for interval in intervals:
                for j, potential_next_note in enumerate(notes):
                    if potential_next_note.state_fixed.time_position <= current_note.state_fixed.time_position: continue
                    lower, upper = DIATONIC_TO_CHROMATIC_INTERVAL[interval]
                    if lower <= potential_next_note.pitch[1] - current_note.pitch[1] <= upper and not ups_found[interval]:
                        ups_found[interval] = True
                        interval_up_from_to[interval].append([i, j])
                    if -upper <= potential_next_note.pitch[1] - current_note.pitch[1] <= -lower and not downs_found[interval]:
                        downs_found[interval] = True
                        interval_down_from_to[interval].append([i, j])
                    if ups_found[interval] and downs_found[interval]:
                        break

        for interval in intervals:
            edge_indices[f"up{interval}"] = interval_up_from_to[interval]
            edge_indices[f"down{interval}"] = interval_down_from_to[interval]
        return edge_indices

    @staticmethod
    def initialize_edge_indices(hetero_data, edge_indices):
        edge_indices = {
            edge_type: torch.tensor(np.array(edges), dtype=torch.long).t().contiguous()
            for edge_type, edges in edge_indices.items()
        }
        edge_indices = [
            (('note', edge_type, 'note'), edge_indices)
            for edge_type, edge_indices in edge_indices.items()
        ]

        # Initialize edge weights to 1
        for edge_type, edge_index in edge_indices:
            if edge_index.numel() == 0:
                continue
            hetero_data[edge_type].edge_index = edge_index
            num_edges = hetero_data[edge_type].edge_index.shape[1]
            edge_weights = torch.ones(num_edges)
            hetero_data[edge_type].edge_attr = edge_weights
        
        return hetero_data

    @staticmethod
    def open_pickle(pkl_file):
        with open(pkl_file, 'rb') as f:
            data_dict = pickle.load(f)
        if len(data_dict['t_edges'][0]) == 0 or len(data_dict['b_edges'][0]) == 0:
            raise PickledError(f"pickle file {pkl_file} has treble or bass of length 0")
        return data_dict

    def prepare_depth_edges_bass_to_treble(self, pkl_file, hetero_data_original):
        data_dict = self.open_pickle(pkl_file)
        data_response_pairs = []
        edge_indices = {f"{k}_depth{d}": [] for k in ["treble", "bass"] for d in range(NUM_DEPTHS)}
        for depth in range(NUM_DEPTHS):
            try:
                treble_edges = data_dict['t_edges'][depth]
                bass_edges = data_dict['b_edges'][depth]
            except IndexError:
                continue

            data_response_pairs, edge_indices = self.process_voice_edges(
                "bass", bass_edges, hetero_data_original, edge_indices, depth, data_response_pairs
            )
            data_response_pairs, edge_indices = self.process_voice_edges(
                "treble", treble_edges, hetero_data_original, edge_indices, depth, data_response_pairs
            )
        return data_response_pairs

    def process_voice_edges(self, voice, edges, hetero_data_original, edge_indices, depth, data_response_pairs):
        for i, edge in enumerate(edges):
            hetero_data = deepcopy(hetero_data_original)
            edge_indices[f"{voice}_depth{depth}"] = edges[:i]
            hetero_data = self.initialize_edge_indices(hetero_data, edge_indices)
            response = torch.tensor(edges[i], dtype=torch.long)
            data_response_pairs.append((hetero_data, response))
        edge_indices[f"{voice}_depth{depth}"] = edges
        return data_response_pairs, edge_indices

    def prepare_depth_edges_alternating(self, pkl_file, hetero_data_original):
        data_dict = self.open_pickle(pkl_file)
        data_response_pairs = []
        edge_indices = {f"{k}_depth{d}": [] for k in ["treble", "bass"] for d in range(NUM_DEPTHS)}
        for depth in range(NUM_DEPTHS):
            treble_edges = data_dict['t_edges'][depth]
            bass_edges = data_dict['b_edges'][depth]

            for treble_idx in range(len(treble_edges)):
                for bass_idx in range(1, len(bass_edges)):
                    hetero_data = deepcopy(hetero_data_original)

                    edge_indices[f"treble_depth{depth}"] = treble_edges[:treble_idx]
                    edge_indices[f"bass_depth{depth}"] = bass_edges[:bass_idx]

                    hetero_data_treble = self.initialize_edge_indices(hetero_data, edge_indices)
                    hetero_data_bass = self.initialize_edge_indices(hetero_data, edge_indices)

                    treble_response = torch.tensor(treble_edges[treble_idx], dtype=torch.long) if treble_idx < len(treble_edges) else None
                    bass_response = torch.tensor(bass_edges[bass_idx], dtype=torch.long) if bass_idx < len(bass_edges) else None

                    if treble_response is not None:
                        data_response_pairs.append((hetero_data_treble, treble_response))
                    if bass_response is not None:
                        data_response_pairs.append((hetero_data_bass, bass_response))

        return data_response_pairs

    @staticmethod
    def get_metric_strengths(pyscoreparser_notes: list[Note]):
        time_signature = pyscoreparser_notes[0].state_fixed.time_signature
        if time_signature is None:
            raise ValueError("Time signature is None")
        time_signature_str = f"{time_signature.numerator}/{time_signature.denominator}"
        time_signature_QL = time_signature.numerator / time_signature.denominator * 4
        curr_total_QL = 0
        metric_strengths = []
        #TODO: Handle anacrusis
        for note in pyscoreparser_notes:
            measure_placement_QL = curr_total_QL % time_signature_QL
            metric_map_time_sig = METRIC_STRENGTH_QUARTER_ONSET[time_signature_str]
            try:
                metric_strengths.append(metric_map_time_sig[measure_placement_QL] - 1)
            except KeyError:
                metric_strengths.append(5)
            curr_total_QL += SECONDS_TO_QL(note.note_duration.seconds)
        return metric_strengths

    @staticmethod
    def add_voice_and_depth_edges(pkl_file, edge_indices):

        if pkl_file:
            with open(pkl_file, 'rb') as f:
                data_dict = pickle.load(f)

            for voice, edge_lists in data_dict.items():
                for i in range(NUM_DEPTHS):
                    edge_indices[f'{"treble" if voice[0] == "t" else "bass"}_depth{i}'] = data_dict[voice][i] \
                        if i < len(data_dict[voice]) \
                        else []

        return edge_indices

    @staticmethod
    def get_scale_degrees(pyscoreparser_notes, key_signature: music21.key.Key):
        tonic = key_signature.tonic
        pitches = [music21.pitch.Pitch(note.pitch[0].replace("x", "##").replace("bb", "--")) for note in pyscoreparser_notes]
        pitches = [
            pitch
            if tonic.name in [p.name for p in pitches]
            else pitch.getEnharmonic()
            for pitch in pitches
        ]
        intervals = [music21.interval.Interval(tonic, pitch) for pitch in pitches]
        intervals = [
            i.directedName
            if i.direction is music21.interval.Direction.ASCENDING
            else i.complement.directedName
            for i in intervals
        ]
        try:
            intervals_mapped = [SCALE_DEGREE_MAP[i] for i in intervals]
        except KeyError as e:
            raise EnharmonicError(f"Enharmonic issue found {e}: \n"
                                  f"tonic: {tonic.name} \n"
                                  f"pitches: {[p.name for p in pitches]}")
        return intervals_mapped
    
    @staticmethod
    def encode_voice_positions(notes):
        """
        Given a list of notes (each with a .voice attribute),
        returns two lists of length len(notes):
        - low_to_high: 0.0 for the lowest voice up to 1.0 for the highest
        - high_to_low: 1.0 for the lowest voice down to 0.0 for the highest

        If there’s only one voice, low_to_high will be all 0.0 and high_to_low all 1.0.

        :param notes: list of note objects with a .voice attribute
        :return: (low_to_high, high_to_low)
        """
        # 1. Find all distinct voices and sort them (lowest → highest)
        unique_voices = sorted({note.voice for note in notes})
        num_voices = len(unique_voices)

        # 2. Build a mapping: voice → index (0 for lowest, num_voices-1 for highest)
        voice_to_index = {voice: idx for idx, voice in enumerate(unique_voices)}

        # 3. Generate the two encodings
        low_to_high = []
        high_to_low = []
        if num_voices > 1:
            for note in notes:
                idx = voice_to_index[note.voice]
                # normalize into [0,1]
                norm = idx / (num_voices - 1)
                low_to_high.append(norm)
                high_to_low.append(1.0 - norm)
        else:
            # edge case: only one voice
            low_to_high = [0.0] * len(notes)
            high_to_low = [1.0] * len(notes)

        return low_to_high, high_to_low


    @staticmethod
    def process_file_nodes(hetero_data, pyscoreparser_notes, key_signature: music21.key.Key, include_global_nodes=INCLUDE_GLOBAL_NODES):
        offsets = [note.state_fixed.time_position for note in pyscoreparser_notes]
        durations = [note.note_duration.seconds for note in pyscoreparser_notes]

        low_to_high, high_to_low = SchenkerDiffHeteroGraphData.encode_voice_positions(pyscoreparser_notes)

        rhythmic_features = {
            "metric_strength": SchenkerDiffHeteroGraphData.get_metric_strengths(pyscoreparser_notes),
            "duration": np.array([duration / np.max(durations) for duration in durations]),
            "offsets": np.array([offset / np.max(offsets) for offset in offsets]),
            "voice_high_low": np.array([ encoding for encoding in high_to_low]),
            "voice_low_high": np.array([ encoding for encoding in low_to_high]),
        }

        rhythmic_features["metric_strength"] = SchenkerDiffHeteroGraphData.one_hot_convert(rhythmic_features["metric_strength"], 6)
        rhythmic_features["duration"] = SchenkerDiffHeteroGraphData.to_float_tensor(rhythmic_features["duration"]).unsqueeze(1)
        rhythmic_features["offsets"] = SchenkerDiffHeteroGraphData.to_float_tensor(rhythmic_features["offsets"]).unsqueeze(1)
        rhythmic_features["voice_high_low"] = SchenkerDiffHeteroGraphData.to_float_tensor(rhythmic_features["voice_high_low"]).unsqueeze(1)
        rhythmic_features["voice_low_high"] = SchenkerDiffHeteroGraphData.to_float_tensor(rhythmic_features["voice_low_high"]).unsqueeze(1)

        node_features = {
            # "pitch_class": [PITCH_CLASS_MAP[note.pitch[0]] for note in pyscoreparser_notes],
            # "metric_strength": SchenkerDiffHeteroGraphData.get_metric_strengths(pyscoreparser_notes),
            # "midi": np.array([(note.pitch[1] - 21) / 88 for note in pyscoreparser_notes]),
            # "midi": np.array([note.pitch[1] for note in pyscoreparser_notes]),
            # "duration": np.array([duration / np.max(durations) for duration in durations]),
            # "offsets": np.array([offset / np.max(offsets) for offset in offsets]),
            "scale_degrees": SchenkerDiffHeteroGraphData.get_scale_degrees(pyscoreparser_notes, key_signature)
        }
        # node_features["pitch_class"] = SchenkerDiffHeteroGraphData.one_hot_convert(node_features["pitch_class"], len(PITCH_CLASS_MAP))
        # node_features["metric_strength"] = SchenkerDiffHeteroGraphData.one_hot_convert(node_features["metric_strength"], 6)
        # node_features["midi"] = self.one_hot_convert(node_features["midi"], 88)
        # node_features["midi"] = HeteroGraphData.to_float_tensor(node_features["midi"]).unsqueeze(1)
        # node_features["duration"] = SchenkerDiffHeteroGraphData.to_float_tensor(node_features["duration"]).unsqueeze(1)
        # node_features["offsets"] = SchenkerDiffHeteroGraphData.to_float_tensor(node_features["offsets"]).unsqueeze(1)
        node_features["scale_degrees"] = SchenkerDiffHeteroGraphData.one_hot_convert(node_features["scale_degrees"], len(SCALE_DEGREE_MAP))

        note_features = torch.cat([feature for feature in node_features.values()], dim=1)
        rhythmic_features = torch.cat([feature for feature in rhythmic_features.values()], dim=1)

        # if include_global_nodes:
        #     global_nodes = torch.zeros(4, NUM_FEATURES, dtype=torch.float)
        #     note_features = torch.cat([note_features, global_nodes], dim=0)



        notes_graph = score_graph.make_edge(pyscoreparser_notes)
        hetero_data['note'].x = note_features
        hetero_data['note'].r = rhythmic_features

        return hetero_data, notes_graph

    @staticmethod
    def add_global_node_edges(pyscoreparser_notes, edge_indices):
        num_actual_notes = len(pyscoreparser_notes)
        for i, (voice, end) in enumerate([(v, e) for v in ['treble', 'bass'] for e in ['start', 'end']]):
            edge_indices[f"global_{end}_{voice}"] = [
                [n, num_actual_notes + i] for n in range(num_actual_notes)
            ]
        return edge_indices
    
    @staticmethod
    def filter_non_overlapping(edges):
        """
        Given edges as pairs (u, v), treat each as an undirected interval [min(u,v), max(u,v)].
        Returns a new list where any edge that overlaps a previously kept edge is discarded.
        """
        accepted = []
        for u, v in edges:
            a, b = sorted((u, v))
            # check for any partial‐overlap with an accepted (c,d):
            #   a < c < b < d  (new contains start of old)
            # or c < a < d < b  (old contains start of new)
            partial = False
            for c, d in accepted:
                if (a < c < b < d) or (c < a < d < b):
                    partial = True
                    break
            if not partial:
                accepted.append((a, b))
        return accepted
    
    @staticmethod
    def infer_edge_depths(edges):
        """
        Given a list of edges as (u,v) with u<v and no partial overlaps,
        returns a dict of edge and its depth
        """
        edges = SchenkerDiffHeteroGraphData.filter_non_overlapping(edges)

        # sort the edges
        edges_sorted = sorted(
            edges,
            key=lambda x: math.fabs(x[0] - x[1])
        )
        
        # find maximum index within edges
        max_u = max(u for u, _ in edges_sorted)
        max_v = max(v for _, v in edges_sorted)
        n = max(max_u, max_v)
        nodes = (n + 1) * [-1]
        
        # itterate through from the smallest span edges to the largest
        # and update the node depths one by one
        for u, v in edges_sorted:
            # if math.fabs(u - v) == 1:
            nodes[u] = max(nodes[u], 0)
            nodes[v] = max(nodes[v], 0)
            if math.fabs(u - v) > 1:
                nodes[u] = max(max(nodes[u+1:v]) + 1, nodes[u])
                nodes[v] = max(max(nodes[u+1:v]) + 1, nodes[v])
    
        while True:
            prev_nodes = [i for i in nodes]
            for u, v in edges_sorted:
                # if math.fabs(u - v) == 1:
                nodes[u] = max(nodes[u], 0)
                nodes[v] = max(nodes[v], 0)
                if math.fabs(u - v) > 1:
                    nodes[u] = max(max(nodes[u+1:v]) + 1, nodes[u])
                    nodes[v] = max(max(nodes[u+1:v]) + 1, nodes[v])
            if prev_nodes == nodes:
                break
            else:
                prev_nodes = [i for i in nodes]


        
        # the depth of an edge is the smallest depth of its connected nodes
        res = dict()
        # for u, v in edges_sorted:
        #     # if math.fabs(u - v) == 1:
        #     #     if 1 in res:
        #     #         if (u,v) not in res[1]:
        #     #             res[1].append((u,v))
        #     #     else:
        #     #         res[1] = [(u,v)]
        #     # else:
        #     start = max(nodes[u+1:v] + [0]) + 1
        #     end   = min(nodes[v], nodes[u]) + 1
        #     for i in range(start, end):
        #         # if i <= math.fabs(u - v):
        #         if i in res:
        #             if (u,v) not in res[i]:
        #                 res[i].append((u,v))
        #         else:
        #             res[i] = [(u,v)]
                

        conn = {}
        n = len(nodes)
        max_h = max(nodes)
        
        # For each level ℓ, scan the base list and link consecutive nodes present at ℓ
        for l in range(max_h + 1):
            prev = None
            for i in range(n):
                if nodes[i] >= l:
                    if prev is not None:
                        conn.setdefault((prev, i), []).append(l)
                    prev = i
            # at end of this level, reset prev for next ℓ

        for edge in edges_sorted:
            for depth in conn[(edge)]:
                if depth in res:
                    res[depth].append(edge)
                else:
                    res[depth] = [edge]


        # print(res)
        return res


    @staticmethod
    def process_file_edges(
            hetero_data, notes_graph, pyscoreparser_notes, include_depth_edges,
            pkl_file=None, include_global_nodes=INCLUDE_GLOBAL_NODES, xml_file = None
    ):
        edge_indices = {k: [] for k in [
            "onset",
            # "voice",
            "forward",
            # "slur",
            "melisma",
            # "rest",
        ]}
        # edge_indices = HeteroGraphData.add_interval_edges(pyscoreparser_notes, edge_indices)

        if include_depth_edges:
            # Check if we have analysis for the file, if not, use the prediction from SchenkerGNN
            if pkl_file:
                edge_indices = SchenkerDiffHeteroGraphData.add_voice_and_depth_edges(pkl_file, edge_indices)
                # treble_edges = []
                # for key, edges in edge_indices.items():
                #     if key.startswith("treble_depth"):
                #         treble_edges.extend(edges)
                # treble_edges = {
                #     (edge[0], edge[1])
                #     for edge in treble_edges
                # }
                # treble_edges= list(treble_edges)
                # treble_depths = SchenkerDiffHeteroGraphData.infer_edge_depths(treble_edges)

            else:
                # [TODO]: Re-enable when we have new schenkerlink modek
                # analysis_treble, analysis_bass, node_list = load_score(str(xml_file))
                # clean_bass = [edge for edge in analysis_bass if edge not in analysis_treble]
                # treble_depths = SchenkerDiffHeteroGraphData.infer_edge_depths(analysis_treble) if analysis_treble != [] else []
                # bass_depths = SchenkerDiffHeteroGraphData.infer_edge_depths(clean_bass) if clean_bass != [] else []
                # edge_dict = {"treble": treble_depths, "bass": bass_depths}
                # for i in range(NUM_DEPTHS):
                #     for voice in ["treble", "bass"]:
                #         edge_indices[f'{voice}_depth{i}'] = edge_dict[voice][i] \
                #             if i in edge_dict[voice] \
                #             else []
                for i in range(NUM_DEPTHS):
                    for voice in ["treble", "bass"]:
                        edge_indices[f'{voice}_depth{i}'] = []


                # s_edge_index, s_edge_attr = extract_structure_sparse(analysis_treble, analysis_bass, node_list)
        # if include_global_nodes:
        #     edge_indices = HeteroGraphData.add_global_node_edges(pyscoreparser_notes, edge_indices)

        for edge in notes_graph:
            from_to = edge[:2]
            edge_type = edge[2]
            if edge_type in edge_indices.keys():
                edge_indices[edge_type].append(from_to)

        return SchenkerDiffHeteroGraphData.initialize_edge_indices(hetero_data, edge_indices)

    @staticmethod
    def check_overlapping_notes(pyscoreparser_notes, name):
        file_name = name.split('/')[-1]
        overlapping_pairs = []
        for i in range(len(pyscoreparser_notes)-1):
            for j in range(i+1, len(pyscoreparser_notes)):
                if pyscoreparser_notes[i].state_fixed.time_position == pyscoreparser_notes[j].state_fixed.time_position:
                    overlapping_pairs.append((i,j))
        output_dir = Path('inference/overlapping_edges')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{file_name}.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(overlapping_pairs, f)
        return

    @staticmethod
    def extract_voices(pkl_file, notes, include_global_nodes=INCLUDE_GLOBAL_NODES):
        num_nodes = len(notes)  # Try max over treble/bass indices if it doesn't work

        with open(pkl_file, 'rb') as f:
            edges = pickle.load(f)

        # Recursively unpack t_edges, b_edges to get the node indices under treble, bass voices resp.
        # Union = both, complement of both = inner voice, other wise just whichever treble/bass

        # Multilabel classification 3 -- treble, bass, inner (+ implicit class: treble + bass)
        t_edges = edges["t_edges"][0]
        b_edges = edges["b_edges"][0]

        # Keep track of treble and bass notes
        treble_indices = list(set([i for pair in t_edges for i in pair]))
        bass_indices = list(set([i for pair in b_edges for i in pair]))
        if include_global_nodes:
            treble_indices += [num_nodes, num_nodes+1]
            bass_indices += [num_nodes+2, num_nodes+3]
        inner_indices = [i for i in range(num_nodes) if (i not in treble_indices and i not in bass_indices)]

        treble_indices = torch.LongTensor(treble_indices)
        bass_indices = torch.LongTensor(bass_indices)
        inner_indices = torch.LongTensor(inner_indices)

        # num_nodes x 3 torch tensor
        ground_truth = torch.zeros((num_nodes + 4, 3)) if include_global_nodes else torch.zeros((num_nodes, 3))
        ground_truth[treble_indices, 0] = 1
        ground_truth[bass_indices, 1] = 1
        ground_truth[inner_indices, 2] = 1


        return ground_truth

    @staticmethod
    def process_file_for_GUI(xml_file,  include_depth_edges=False, include_global_nodes=True):
        xml = MusicXMLDocument(xml_file)
        pyscoreparser_notes = xml.get_notes()
        music21_score = music21.converter.parse(str(xml_file))
        # key_signature = music21_score.analyze('key')
        ks_elem = music21_score.flat.getElementsByClass(music21.key.KeySignature)[0]
        # 3) convert that KeySignature into a Key object
        key_signature: music21.key.Key = ks_elem.asKey()

        hetero_data = HeteroData()
        hetero_data, notes_graph = SchenkerDiffHeteroGraphData.process_file_nodes(
            hetero_data, pyscoreparser_notes, key_signature, include_global_nodes
        )
        hetero_data = SchenkerDiffHeteroGraphData.process_file_edges(
            hetero_data, notes_graph, pyscoreparser_notes, include_depth_edges,
            pkl_file=None, include_global_nodes=include_global_nodes
        )

        # ground_truth_voice = self.extract_voices(pkl_file, pyscoreparser_notes)

        # if 'asap-dataset' in str(xml_file):
        #     s_edge_index, s_edge_attr = [], []
        # else:
        
        analysis_treble, analysis_bass, node_list = load_score(str(xml_file))
        s_edge_index, s_edge_attr = extract_structure_sparse(analysis_treble, analysis_bass, node_list)
        
        
        data_dict = {
            "name": str(xml_file).removesuffix('.xml'),
            "data": hetero_data,
            # "voice": ground_truth_voice,
            # "s_edge_index": s_edge_index,
            # "s_edge_attr": s_edge_attr
        }

        return data_dict

    def process_file(self, xml_file, pkl_file, index, include_depth_edges, save_data=True):
        xml_file = Path(xml_file)
        XMLDocument = MusicXMLDocument(str(xml_file))
        pyscoreparser_notes = XMLDocument.get_notes()
        music21_score = music21.converter.parse(str(xml_file))
        # key_signature = music21_score.analyze('key')
        ks_elem = music21_score.flat.getElementsByClass(music21.key.KeySignature)[0]
        # 3) convert that KeySignature into a Key object
        key_signature: music21.key.Key = ks_elem.asKey()


        self.check_overlapping_notes(pyscoreparser_notes, str(xml_file).removesuffix('.xml'))

        hetero_data = HeteroData()
        hetero_data, notes_graph = SchenkerDiffHeteroGraphData.process_file_nodes(
            hetero_data, pyscoreparser_notes, key_signature
        )
        hetero_data = SchenkerDiffHeteroGraphData.process_file_edges(
            hetero_data, notes_graph, pyscoreparser_notes,
            include_depth_edges=include_depth_edges, pkl_file=pkl_file, include_global_nodes=INCLUDE_GLOBAL_NODES,
            xml_file = str(xml_file)
        )

        # ground_truth_voice = self.extract_voices(pkl_file, pyscoreparser_notes, include_depth_edges)

        # with open(pkl_file, 'rb') as f:
        #     edges = pickle.load(f)d

        # # Recursively unpack t_edges, b_edges to get the node indices under treble, bass voices resp.
        # # Union = both, complement of both = inner voice, other wise just whichever treble/bass

        # # Multilabel classification 3 -- treble, bass, inner (+ implicit class: treble + bass)
        # t_edges = edges["t_edges"][0]
        # b_edges = edges["b_edges"][0]

        # # analysis_treble, analysis_bass, node_list = load_score(str(xml_file))
        # # s_edge_index, s_edge_attr = extract_structure_sparse(analysis_treble, analysis_bass, node_list)

        for key, value in hetero_data.items():
            # Check if value is None.
            if value is None:
                raise ValueError(f"Value for '{key}' is None.")
            
            # Check if value is a tensor with no elements.
            if isinstance(value, torch.Tensor) and value.numel() == 0:
                raise ValueError(f"Tensor for '{key}' is empty.")
            
            # Check for empty collection types (list, tuple, set, dict).
            if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
                raise ValueError(f"Collection for '{key}' is empty.")

        if hetero_data[('note', 'forward', 'note')]['edge_index'].numel() == 0:
            raise ValueError(f"Tensor for '{('node', 'forward', 'node')}' is empty.")

        # if 'asap-dataset' in str(xml_file):
        #     analysis_treble, analysis_bass, node_list = load_score(str(xml_file))
        #     s_edge_index, s_edge_attr = extract_structure_sparse(analysis_treble, analysis_bass, node_list)
        # else:
        #     edge_index = {}
        #     temp_edge_index = self.add_voice_and_depth_edges(pkl_file, edge_index)
        #     s_edge_index = []
        #     for edges_list in temp_edge_index.values():
        #         s_edge_index.extend(edges_list)
        #     s_edge_attr = [1] * len(s_edge_index)
        #     s_edge_index = torch.tensor(s_edge_index, dtype=torch.long).t().contiguous()
        #     s_edge_attr = torch.tensor(s_edge_attr, dtype=torch.float)

        data_dict = { 
            "name": str(xml_file).removesuffix('.xml'), 
            "data": hetero_data, 
            # "voice": ground_truth_voice,  
            # "s_edge_index": s_edge_index, 
            # "s_edge_attr": s_edge_attr 
        }

        if save_data:
            torch.save(data_dict, os.path.join(self.processed_dir, f'{index}_processed.pt'))
            self.data_list.append(data_dict)

        return data_dict

    def process(self):

        self.data_list = []
        index = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for directory in self.train_names:
                pkl_files = []
                # filepath = f"../{directory}/**/*" if self.test_mode else f"{directory}/**/*"
                filepath = f"../../../{directory}/**/*" if self.test_mode else f"../../../SchenkerDiff/{directory}/**/*"
                pkl_files.extend(glob.glob(filepath + ".pkl", recursive=True))
                if len(pkl_files) > 0:
                    # For inference
                    pkl_file = pkl_files[0]
                else:
                    pkl_file = None
                xml_files = []
                xml_files.extend(glob.glob(filepath + ".xml", recursive=True))
                for xml_file in xml_files:
                    if index % 100 == 0:
                        print(f"Processing file {xml_file}")
                    future = executor.submit(self.process_file, xml_file, pkl_file, index,
                                           include_depth_edges=self.include_depth_edges)
                    try:
                        result = future.result(timeout=10)
                    except (EnharmonicError, ValueError,TypeError, KeyError, IndexError, music21.analysis.discrete.DiscreteAnalysisException, exception.MusicXMLParseException, concurrent.futures.TimeoutError) as e:
                        print(f"Skipping {xml_file} due to error: {e}")
                        continue
                    index += 1

    def hetero_to_networkx(self, obj_idx):
        edge_type_color = {
            'forward': 'red',
            'onset': 'green',
            'sustain': 'blue',
            'rest': 'yellow',
        }

        file_name = self.processed_file_names[obj_idx]
        file_path = os.path.join(self.processed_dir, file_name)
        hetero_data = torch.load(file_path)
        G = nx.DiGraph()

        # Add nodes
        for node_type in hetero_data.node_types:
            for node_id in range(hetero_data[node_type].NUM_NODES):
                G.add_node(f"{node_type}_{node_id}", type=node_type)

        # Add edges with colors
        for edge_type in hetero_data.edge_types:
            color = edge_type_color.get(edge_type[1], 'black')
            for source, target in hetero_data[edge_type].edge_index.t().numpy():
                src_node = f"{edge_type[0]}_{source}"
                tgt_node = f"{edge_type[2]}_{target}"
                G.add_edge(src_node, tgt_node, type=edge_type[1], color=color)

        return G


if __name__ == "__main__":
    with open("debug_instance.txt", "r") as file:
        names = file.readlines()
        names = [line.strip() for line in names if line[0] != "#"]

    dataset = SchenkerDiffHeteroGraphData(root="processed/heterdatacleaned/", train_names=names)
