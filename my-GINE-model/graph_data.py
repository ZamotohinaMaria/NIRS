import json
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


def normalize_label(value) -> int:
    if pd.isna(value):
        raise ValueError("Empty label")

    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)

    if isinstance(value, float):
        if int(value) in (0, 1):
            return int(value)

    s = str(value).strip().lower()
    benign_values = {"0", "benign", "goodware", "normal", "clean", "false"}
    malware_values = {"1", "malware", "malicious", "bad", "trojan", "virus", "true"}

    if s in benign_values:
        return 0
    if s in malware_values:
        return 1

    raise ValueError(f"Unrecognized label value: {value}")


def split_api_sequence(seq: str, delimiter_regex: str) -> List[str]:
    if pd.isna(seq):
        return []

    s = str(seq).strip()
    if not s:
        return []

    if (s.startswith("[") and s.endswith("]")) or (s.startswith('["') and s.endswith('"]')):
        try:
            parsed = json.loads(s.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    tokens = re.split(delimiter_regex, s)
    return [t.strip() for t in tokens if t.strip()]


def build_vocab(sequences: List[List[str]], min_token_freq: int = 1) -> Dict[str, int]:
    counter = Counter()
    for seq in sequences:
        counter.update(seq)

    vocab = {"<UNK>": 0}
    for token, freq in counter.items():
        if freq >= min_token_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def sequence_to_ids(seq: List[str], vocab: Dict[str, int], max_seq_len: int) -> List[int]:
    seq = seq[:max_seq_len]
    return [vocab.get(tok, 0) for tok in seq]


def make_node_features(api_ids: List[int], vocab_size: int) -> torch.Tensor:
    n = len(api_ids)
    x = torch.zeros((n, vocab_size), dtype=torch.float32)
    for i, api_id in enumerate(api_ids):
        x[i, api_id] = 1.0
    return x


def build_graph_from_sequence(
    api_ids: List[int],
    label: int,
    vocab_size: int,
    undirected: bool = False,
    add_skip_edges: bool = False,
) -> Data:
    if len(api_ids) == 0:
        api_ids = [0]

    x = make_node_features(api_ids, vocab_size)

    edges = []
    edge_attr = []

    for i in range(len(api_ids) - 1):
        edges.append([i, i + 1])
        edge_attr.append([1.0, 0.0, 0.0])

        if undirected:
            edges.append([i + 1, i])
            edge_attr.append([0.0, 1.0, 0.0])

    if add_skip_edges and len(api_ids) >= 3:
        for i in range(len(api_ids) - 2):
            edges.append([i, i + 2])
            edge_attr.append([0.0, 0.0, 1.0])

            if undirected:
                edges.append([i + 2, i])
                edge_attr.append([0.0, 0.0, 1.0])

    if not edges:
        edges = [[0, 0]]
        edge_attr = [[1.0, 0.0, 0.0]]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def make_graph_dataset(
    df: pd.DataFrame,
    label_col: str,
    seq_col: str,
    delimiter_regex: str,
    min_token_freq: int,
    max_seq_len: int,
    undirected: bool,
    add_skip_edges: bool,
) -> Tuple[List[Data], Dict[str, int]]:
    parsed_sequences = []
    labels = []

    for _, row in df.iterrows():
        try:
            label = normalize_label(row[label_col])
            tokens = split_api_sequence(row[seq_col], delimiter_regex)
            if len(tokens) == 0:
                continue
            parsed_sequences.append(tokens)
            labels.append(label)
        except Exception:
            continue

    if len(parsed_sequences) == 0:
        raise ValueError("No valid sequences after parsing")

    vocab = build_vocab(parsed_sequences, min_token_freq=min_token_freq)
    vocab_size = len(vocab)

    graphs = []
    for seq, label in zip(parsed_sequences, labels):
        api_ids = sequence_to_ids(seq, vocab, max_seq_len=max_seq_len)
        data = build_graph_from_sequence(
            api_ids=api_ids,
            label=label,
            vocab_size=vocab_size,
            undirected=undirected,
            add_skip_edges=add_skip_edges,
        )
        graphs.append(data)

    return graphs, vocab


def split_dataset(graphs: List[Data], test_size: float, val_size: float, seed: int):
    labels = [int(g.y.item()) for g in graphs]

    train_val_idx, test_idx = train_test_split(
        np.arange(len(graphs)),
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=seed,
        stratify=train_val_labels,
    )

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    return train_graphs, val_graphs, test_graphs


def print_dataset_stats(graphs: List[Data], split_name: str) -> None:
    labels = [int(g.y.item()) for g in graphs]
    counter = Counter(labels)
    avg_nodes = np.mean([g.num_nodes for g in graphs]) if graphs else 0.0
    avg_edges = np.mean([g.num_edges for g in graphs]) if graphs else 0.0

    print(f"\n[{split_name}]")
    print(f"Samples: {len(graphs)}")
    print(f"Class distribution: {dict(counter)}")
    print(f"Avg nodes: {avg_nodes:.2f}")
    print(f"Avg edges: {avg_edges:.2f}")
