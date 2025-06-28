import glob
import os
import pickle
from collections import Counter
from copy import deepcopy
from pathlib import Path

import music21.converter
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Dataset, HeteroData

import src.pyScoreParser.score_as_graph as score_graph
from src.schenker_gnn.config import INTERVAL_EDGES, NUM_DEPTHS, NUM_FEATURES, INCLUDE_GLOBAL_NODES, ABLATIONS, DEVICE
from src.schenker_gnn.data_maps import *
from errors import PickledError
from src.pyScoreParser.musicxml_parser.mxp import MusicXMLDocument
from src.pyScoreParser.musicxml_parser.mxp.note import Note


class EnharmonicError(Exception):
    def __init__(self, message):
        super().__init__(message)


class HeteroGraphData(Dataset):
    def __init__(self,
                 root,
                 train_names=None,
                 transform=None,
                 pre_transform=None,
                 test_mode=False,
                 voice_mode=False,
                 include_depth_edges=True,
                 include_global_nodes=INCLUDE_GLOBAL_NODES
                 ):
        """
        root: where my dataset should be stored: it will automatically saved at root/processed
        """
        self.test_mode = test_mode
        self.voice_mode = voice_mode
        self.train_names = train_names
        self.include_depth_edges = include_depth_edges
        self.include_global_nodes = include_global_nodes
        self.scale_degree_counter = Counter()
        super(HeteroGraphData, self).__init__(root, transform = transform, pre_transform = pre_transform)
        self.data_list = []

        for file_name in self.processed_file_names:
            file_path = os.path.join(self.processed_dir, file_name)
            if os.path.isfile(file_path):
                self.data_list.append(torch.load(file_path))
            #else:
                #print(f"Missing processed file: {file_path}")
        self.root = root


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
        if self.test_mode:
            return [f'{i}_processed.pt' for i in range(144)]
        return [f'{i}_processed.pt' for i in range(1070)]

    @staticmethod
    def one_hot_convert(mapped_pitch, num_class):
        # number of samples, number of class
        one_hot_encoded = np.zeros((len(mapped_pitch), num_class))
        for i, pitch in enumerate(mapped_pitch):
            one_hot_encoded[i, pitch] = 1
        return HeteroGraphData.to_float_tensor(one_hot_encoded)

    @staticmethod
    def to_float_tensor(array):
        if array.dtype == np.object_:
            array = np.array(array, dtype=float)
        return torch.tensor(array, dtype=torch.float)

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
        with open(pkl_file, 'rb') as f:
            data_dict = pickle.load(f)

        for voice, edge_lists in data_dict.items():
            for i in range(NUM_DEPTHS):
                edge_indices[f'{"treble" if voice[0] == "t" else "bass"}_depth{i}'] = data_dict[voice][i] \
                    if i < len(data_dict[voice]) \
                    else []

        return edge_indices

    @staticmethod
    def get_scale_degrees(pyscoreparser_notes, key_signature: music21.key.Key, counter=None):
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
        if counter is not None:
            for i in intervals:
                counter[i] += 1
        return intervals_mapped

    @staticmethod
    def process_file_nodes(hetero_data, pyscoreparser_notes, key_signature: music21.key.Key, include_global_nodes=INCLUDE_GLOBAL_NODES, counter=None):
        offsets = [note.state_fixed.time_position for note in pyscoreparser_notes]
        durations = [note.note_duration.seconds for note in pyscoreparser_notes]

        node_features = {
            # "pitch_class": [PITCH_CLASS_MAP[note.pitch[0]] for note in pyscoreparser_notes],
            "metric_strength": HeteroGraphData.get_metric_strengths(pyscoreparser_notes),
            # "midi": np.array([(note.pitch[1] - 21) / 88 for note in pyscoreparser_notes]),
            # "midi": np.array([note.pitch[1] for note in pyscoreparser_notes]),
            # "duration": np.array([duration / np.max(durations) for duration in durations]),
            "duration": np.array(durations),
            "offsets": np.array([offset / np.max(offsets) for offset in offsets]),
            # "scale_degrees": HeteroGraphData.get_scale_degrees(pyscoreparser_notes, key_signature, counter)
        }
        # node_features["pitch_class"] = HeteroGraphData.one_hot_convert(node_features["pitch_class"], len(PITCH_CLASS_MAP))
        node_features["metric_strength"] = HeteroGraphData.one_hot_convert(node_features["metric_strength"], 6)
        # node_features["midi"] = self.one_hot_convert(node_features["midi"], 88)
        # node_features["midi"] = HeteroGraphData.to_float_tensor(node_features["midi"]).unsqueeze(1)
        node_features["duration"] = HeteroGraphData.to_float_tensor(node_features["duration"]).unsqueeze(1)
        node_features["offsets"] = HeteroGraphData.to_float_tensor(node_features["offsets"]).unsqueeze(1)
        # node_features["scale_degrees"] = HeteroGraphData.one_hot_convert(node_features["scale_degrees"], len(SCALE_DEGREE_MAP))

        note_features = torch.cat([feature for feature in node_features.values()], dim=1)

        if include_global_nodes:
            global_nodes = torch.zeros(4, NUM_FEATURES, dtype=torch.float)
            note_features = torch.cat([note_features, global_nodes], dim=0)

        notes_graph = score_graph.make_edge(pyscoreparser_notes)
        hetero_data['note'].x = note_features

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
    def process_file_edges(
            hetero_data, notes_graph, pyscoreparser_notes, include_depth_edges,
            pkl_file=None, include_global_nodes=INCLUDE_GLOBAL_NODES
    ):
        edge_indices = {k: [] for k in [
            "onset",
            # "voice",
            "forward",
            # "slur",
            # "sustain",
            # "rest",
        ]}
        # edge_indices = HeteroGraphData.add_interval_edges(pyscoreparser_notes, edge_indices)
        if include_depth_edges:
            edge_indices = HeteroGraphData.add_voice_and_depth_edges(pkl_file, edge_indices)
        if include_global_nodes:
            edge_indices = HeteroGraphData.add_global_node_edges(pyscoreparser_notes, edge_indices)

        for edge in notes_graph:
            from_to = edge[:2]
            edge_type = edge[2]
            if edge_type in edge_indices.keys():
                edge_indices[edge_type].append(from_to)

        return HeteroGraphData.initialize_edge_indices(hetero_data, edge_indices)

    @staticmethod
    def check_overlapping_notes(pyscoreparser_notes, name):
        file_name = name.split('\\')[-1]
        # file_name = name.split('/')[-1]
        overlapping_pairs = []
        for i in range(len(pyscoreparser_notes)-1):
            for j in range(i+1, len(pyscoreparser_notes)):
                if pyscoreparser_notes[i].state_fixed.time_position == pyscoreparser_notes[j].state_fixed.time_position:
                    overlapping_pairs.append((i, j))
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
    def process_file_for_GUI(xml_file, include_depth_edges=False, include_global_nodes=True):
        xml = MusicXMLDocument(xml_file)
        pyscoreparser_notes = xml.get_notes()
        music21_score = music21.converter.parse(str(xml_file))
        key_signature = music21_score.analyze('key')

        hetero_data = HeteroData()
        hetero_data, notes_graph = HeteroGraphData.process_file_nodes(
            hetero_data, pyscoreparser_notes, key_signature, include_global_nodes
        )
        hetero_data = HeteroGraphData.process_file_edges(
            hetero_data, notes_graph, pyscoreparser_notes, include_depth_edges,
            pkl_file=None, include_global_nodes=include_global_nodes
        )

        # pkl_file = xml_file[:-4] + ".pkl"
        # ground_truth_voice = HeteroGraphData.extract_voices(pkl_file, pyscoreparser_notes)

        return {
            "name": str(xml_file).removesuffix('.xml'),
            # "voice": ground_truth_voice,
            "data": hetero_data
        }

    def process_file(self, xml_file, pkl_file, index, include_depth_edges, save_data=True):
        xml_file = Path(xml_file)
        XMLDocument = MusicXMLDocument(str(xml_file))
        pyscoreparser_notes = XMLDocument.get_notes()
        music21_score = music21.converter.parse(str(xml_file))
        key_signature = music21_score.analyze('key')

        self.check_overlapping_notes(pyscoreparser_notes, str(xml_file).removesuffix('.xml'))

        hetero_data = HeteroData()
        hetero_data, notes_graph = HeteroGraphData.process_file_nodes(
            hetero_data, pyscoreparser_notes, key_signature, counter=self.scale_degree_counter
        )
        hetero_data = HeteroGraphData.process_file_edges(
            hetero_data, notes_graph, pyscoreparser_notes,
            include_depth_edges=include_depth_edges, pkl_file=pkl_file, include_global_nodes=INCLUDE_GLOBAL_NODES,
        )

        ground_truth_voice = self.extract_voices(pkl_file, pyscoreparser_notes)

        data_dict = {
            "name": str(xml_file).removesuffix('.xml'),
            "data": hetero_data,
            "voice": ground_truth_voice
        }
        if save_data:
            torch.save(data_dict, os.path.join(self.processed_dir, f'{index}_processed.pt'))
            self.data_list.append(data_dict)
        return data_dict

    def process(self):
        self.data_list = []
        index = 0
        for directory in self.train_names:
            pkl_files = []
            # filepath = f"../{directory}/**/*" if self.test_mode else f"{directory}/**/*"
            filepath = f"{directory}/**/*"
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
                try:
                    self.process_file(xml_file, pkl_file, index, include_depth_edges=self.include_depth_edges)
                except EnharmonicError as e:
                    print(e)
                    continue
                index += 1
        print(self.scale_degree_counter)

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

    dataset = HeteroGraphData(root="processed/heterdatacleaned/", train_names=names)
