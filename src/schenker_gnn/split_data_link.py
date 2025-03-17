import math
import random
import torch
from config import NUM_DEPTHS


def link_split(data, val_ratio=0.05, test_ratio=0.1, depth=None):
    assert 'batch' not in data  # no batch-mode
    assert depth is not None, "Depth must be specified"

    train_row, train_col = [], []
    val_row, val_col = [], []
    test_row, test_col = [], []

    for d in range(depth + 1):
        for voice in ['treble', 'bass']:
            edge_type = ('note', f"{voice}_depth{d}", 'note')
            if edge_type in data.edge_types:
                edge_indices = data[edge_type].edge_index
                train_row.append(edge_indices[0])
                train_col.append(edge_indices[1])

    if train_row and train_col:
        row = torch.cat(train_row)
        col = torch.cat(train_col)
    else:
        row = torch.tensor([], dtype=torch.long)
        col = torch.tensor([], dtype=torch.long)

    # remove edges with higher depth
    for d in range(depth + 1, NUM_DEPTHS):
        for voice in ['treble', 'bass']:
            edge_type = ('note', f"{voice}_depth{d}", 'note')
            if edge_type in data.edge_types:
                del data[edge_type]

    train_row = row.clone()
    train_col = col.clone()

    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)

    val_row = torch.cat([torch.tensor(val_row), r])
    val_col = torch.cat([torch.tensor(val_col), c])

    # test edges
    r, c = row[n_v:n_v + n_t], col[n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    test_row = torch.cat([test_row, r])
    test_col = torch.cat([test_col, c])

    # training edges
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    train_row = torch.cat([train_row, r])
    train_col = torch.cat([train_col, c])

    # include all other edge types
    for key in data.keys:
        if "edge_index" in key and not key.startswith(("treble_depth", "bass_depth")):
            other_edge_index = data[key]
            train_row = torch.cat([train_row, other_edge_index[0]])
            train_col = torch.cat([train_col, other_edge_index[1]])

            val_row = torch.cat([val_row, other_edge_index[0]])
            val_col = torch.cat([val_col, other_edge_index[1]])

            test_row = torch.cat([test_row, other_edge_index[0]])
            test_col = torch.cat([test_col, other_edge_index[1]])

    # negative edges
    num_nodes = data.num_nodes
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero().t()
    perm = random.sample(range(neg_row.size(0)), min(n_v + n_t, neg_row.size(0)))
    perm = torch.tensor(perm)
    perm = perm.to(torch.long)
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data


if __name__ == "__main__":
    from data_processing import HeterGraph

    # Load the dataset
    with open("train-names.txt", "r") as file:
        train_names = file.readlines()
        train_names = [line.strip() for line in train_names if line[0] != "#"]

    dataset = HeterGraph(root="processed/heterdatacleaned/", train_names=train_names)

    data = dataset[0]['data']
    depth = 2
    split_data = link_split(data, depth=depth)

    # Print the edges after the split
    print("Training positive edges:", split_data.train_pos_edge_index)
    print("Validation positive edges:", split_data.val_pos_edge_index)
    print("Test positive edges:", split_data.test_pos_edge_index)
    print("Training negative edge mask:", split_data.train_neg_adj_mask)
    print("Validation negative edges:", split_data.val_neg_edge_index)
    print("Test negative edges:", split_data.test_neg_edge_index)
