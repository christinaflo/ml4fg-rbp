import os
import subprocess
import random
import numpy as np
import pandas as pd
import networkx as nx
import forgi.graph.bulge_graph as fgb

import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data


def run_rnashape(sequence):
    """Taken from https://github.com/xypan1232/iDeepS/"""
    cmd = 'echo "%s" | ./RNAshapes -t %d -c %d -# %d' % (sequence, 5, 10, 1)
    runout = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if runout.returncode != 0:
        raise Exception(runout.stderr.decode("utf-8"))
    out = runout.stdout.decode("utf-8")
    text = out.strip().split('\n')
    seq_info = text[0]
    if 'configured to print' in text[-1]:
        struct_text = text[-2]
    else:
        struct_text = text[1]
    # shape:
    structur = struct_text.split()[1]
    return structur


class CLIPDataset(InMemoryDataset):
    def __init__(self, root, exp_id, is_training, include_graph=False,
                 transform=None, pre_transform=None, pre_filter=None):
        self.exp_id = exp_id
        self.is_training = is_training
        self.include_graph = include_graph
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        exp_path = os.path.join(self.root, 'processed', str(self.exp_id))
        parent_dir = 'train' if self.is_training else 'test'
        if self.include_graph:
            return os.path.join(exp_path, f'{parent_dir}_graph')
        return os.path.join(exp_path, parent_dir)

    @property
    def raw_file_names(self):
        data_split = 'train' if self.is_training else 'test'
        return [f'clipseq_dataset_{self.exp_id}_{data_split}.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        exp_data = pd.read_csv(self.raw_paths[0])

        nucleotide_types = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4}
        ss_types = {'F': 0, 'T': 1, 'I': 2, 'H': 3, 'M': 4, 'S': 5}

        data_list = []
        for i, row in exp_data.iterrows():
            seq = row['seq']
            if self.include_graph or not row['ss']:
                structur = run_rnashape(row['seq'].replace('T', 'U'))
                bg = fgb.BulgeGraph.from_dotbracket(dotbracket_str=structur, seq=seq.replace('T', 'U'))
                graph = bg.to_networkx()
                ss_seq = bg.to_element_string()
                H = nx.Graph()
                H.add_nodes_from(sorted(graph.nodes(data=True)))
                H.add_edges_from(graph.edges(data=True))
                dist_matrix = np.array(nx.floyd_warshall_numpy(H))
                # adj_mat = np.array(nx.to_numpy_matrix(H))
            else:
                ss_seq = row['ss']
                dist_matrix = None

            seq = torch.tensor([nucleotide_types[x] for x in seq], dtype=torch.long)
            struct = torch.tensor([ss_types[x.upper()] for x in ss_seq], dtype=torch.long)
            label = torch.tensor(row['label'], dtype=torch.long)
            label_one_hot = F.one_hot(label, num_classes=2).to(dtype=torch.float32)

            if self.include_graph:
                edge_attr = torch.tensor(dist_matrix, dtype=torch.float32)
                data = Data(seq=seq, struct=struct, label=label, label_one_hot=label_one_hot,
                            edge_attr=edge_attr)
            else:
                data = Data(seq=seq, struct=struct, label=label, label_one_hot=label_one_hot)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def train_val_split(self, validation_size=0.2, shuffle=False, ideeps_split=True):
        """Taken from https://github.com/xypan1232/iDeepS/"""
        if ideeps_split:
            classes = self.data['label'].numpy()
            num_samples = len(classes)
            classes = np.array(classes)
            classes_unique = np.unique(classes)
            num_classes = len(classes_unique)
            indices = np.arange(num_samples)
            # indices_folds=np.zeros([num_samples],dtype=int)
            training_indice = []
            training_label = []
            validation_indice = []
            validation_label = []
            for cl in classes_unique:
                indices_cl = indices[classes == cl]
                num_samples_cl = len(indices_cl)

                # split this class into k parts
                if shuffle:
                    random.shuffle(indices_cl)  # in-place shuffle

                # module and residual
                num_samples_each_split = int(num_samples_cl * validation_size)
                res = num_samples_cl - num_samples_each_split

                training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
                training_label = training_label + [cl] * res

                validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
                validation_label = validation_label + [cl] * num_samples_each_split

            training_index = np.arange(len(training_label))
            random.shuffle(training_index)
            training_indice = np.array(training_indice)[training_index]
            #training_label = np.array(training_label)[training_index]

            validation_index = np.arange(len(validation_label))
            random.shuffle(validation_index)
            validation_indice = np.array(validation_indice)[validation_index]
            #validation_label = np.array(validation_label)[validation_index]

            train_dataset = self.index_select(training_indice)
            val_dataset = self.index_select(validation_indice)
            return train_dataset, val_dataset

        train_budget = int(self.len() * (1 - validation_size))
        perm = self.shuffle()
        return perm[:train_budget], perm[train_budget:]

