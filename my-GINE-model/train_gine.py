#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_gine.py

Обучение GINE на MalbehavD-V1 / похожем CSV с последовательностями API-вызовов.

Что делает:
1) читает CSV
2) ищет колонку с меткой и колонку с API sequence
3) строит словарь API
4) превращает каждую последовательность в граф PyG
5) обучает GINE для graph classification
6) считает Accuracy / Precision / Recall / F1 / ROC-AUC

Пример запуска:
python train_gine.py \
    --csv /path/to/MalBehavD-V1-dataset.csv \
    --epochs 30 \
    --batch-size 64 \
    --hidden-dim 128 \
    --lr 1e-3

Если автоопределение колонок не сработало:
python train_gine.py \
    --csv /path/to/data.csv \
    --label-col label \
    --seq-col api_sequence
"""

import os
import re
import math
import json
import random
import argparse
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool

# -----------------------------
# Утилиты
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GINE on MalbehavD-V1-like dataset")
    parser.add_argument("--csv", type=str, required=True, help="Путь к CSV-файлу")
    parser.add_argument("--label-col", type=str, default=None, help="Имя колонки с меткой")
    parser.add_argument("--seq-col", type=str, default=None, help="Имя колонки с последовательностью API")
    parser.add_argument("--id-col", type=str, default=None, help="Необязательная колонка с id/sample name")
    parser.add_argument("--delimiter-regex", type=str, default=r"[,\s;|>]+",
                        help=r"Regex для разбиения строки API sequence")
    parser.add_argument("--min-token-freq", type=int, default=1,
                        help="Минимальная частота API для попадания в словарь")
    parser.add_argument("--max-seq-len", type=int, default=400,
                        help="Обрезать последовательности длиннее этого значения")
    parser.add_argument("--undirected", action="store_true",
                        help="Добавлять обратные рёбра")
    parser.add_argument("--add-skip-edges", action="store_true",
                        help="Добавлять skip-edges i -> i+2")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Доля от train+val части")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default="best_gine.pt")
    parser.add_argument("--vocab-path", type=str, default="api_vocab.json")
    return parser.parse_args()


def normalize_label(value) -> int:
    """
    Приводит разные варианты меток к 0/1:
    benign, goodware, normal -> 0
    malware, malicious, bad -> 1
    """
    if pd.isna(value):
        raise ValueError("Пустая метка")

    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)

    if isinstance(value, float):
        if int(value) in (0, 1):
            return int(value)

    s = str(value).strip().lower()

    benign_values = {"0", "benign", "goodware", "normal", "clean"}
    malware_values = {"1", "malware", "malicious", "bad", "trojan", "virus"}

    if s in benign_values:
        return 0
    if s in malware_values:
        return 1

    # Иногда встречается "False/True"
    if s == "false":
        return 0
    if s == "true":
        return 1

    raise ValueError(f"Не удалось распознать метку: {value}")


def guess_label_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "label", "class", "target", "y", "malware", "is_malware",
        "category", "type"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]

    # эвристика: колонка с 2 уникальными значениями
    for col in df.columns:
        vals = df[col].dropna().astype(str).str.lower().unique().tolist()
        if 1 < len(vals) <= 6:
            joined = set(vals)
            if joined & {"benign", "malware", "0", "1", "true", "false"}:
                return col
    return None


def guess_sequence_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "api_sequence", "sequence", "api_calls", "api_call_sequence",
        "calls", "trace", "behavior", "behaviour", "log", "api"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]

    # эвристика: самая длинная текстовая колонка
    text_cols = []
    for col in df.columns:
        if df[col].dtype == object:
            avg_len = df[col].dropna().astype(str).map(len).mean()
            if not math.isnan(avg_len):
                text_cols.append((col, avg_len))
    if text_cols:
        text_cols.sort(key=lambda x: x[1], reverse=True)
        return text_cols[0][0]
    return None


def split_api_sequence(seq: str, delimiter_regex: str) -> List[str]:
    if pd.isna(seq):
        return []

    s = str(seq).strip()
    if not s:
        return []

    # Иногда последовательность хранится как Python-list / JSON-list
    if (s.startswith("[") and s.endswith("]")) or (s.startswith('["') and s.endswith('"]')):
        try:
            parsed = json.loads(s.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    tokens = re.split(delimiter_regex, s)
    tokens = [t.strip() for t in tokens if t.strip()]
    return tokens


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
    """
    Простой и стабильный baseline:
    x = one-hot(id API)
    """
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
    add_skip_edges: bool = False
) -> Data:
    """
    Один образец -> один граф:
    узлы = API calls
    ребра = переходы по времени
    edge_attr:
      [1,0,0] = forward
      [0,1,0] = backward
      [0,0,1] = skip
    """
    if len(api_ids) == 0:
        # Защита от пустых последовательностей
        api_ids = [0]

    x = make_node_features(api_ids, vocab_size)

    edges = []
    edge_attr = []

    # обычные forward edges
    for i in range(len(api_ids) - 1):
        edges.append([i, i + 1])
        edge_attr.append([1.0, 0.0, 0.0])

        if undirected:
            edges.append([i + 1, i])
            edge_attr.append([0.0, 1.0, 0.0])

    # skip edges через один шаг
    if add_skip_edges and len(api_ids) >= 3:
        for i in range(len(api_ids) - 2):
            edges.append([i, i + 2])
            edge_attr.append([0.0, 0.0, 1.0])

            if undirected:
                edges.append([i + 2, i])
                edge_attr.append([0.0, 0.0, 1.0])

    if not edges:
        # последовательность длины 1: делаем self-loop
        edges = [[0, 0]]
        edge_attr = [[1.0, 0.0, 0.0]]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    y = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data


# -----------------------------
# Модель GINE
# -----------------------------

class GINEMalwareClassifier(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.node_encoder = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_add_pool(x, batch)
        out = self.head(x)
        return out


# -----------------------------
# Обучение и оценка
# -----------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.cross_entropy(logits, batch.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_true = []
    all_pred = []
    all_prob = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_true.extend(batch.y.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)

    return compute_metrics(y_true, y_pred, y_prob)


# -----------------------------
# Подготовка данных
# -----------------------------

def load_and_prepare_dataframe(csv_path: str, label_col: Optional[str], seq_col: Optional[str]) -> Tuple[pd.DataFrame, str, str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV не найден: {csv_path}")

    df = pd.read_csv(csv_path)

    if label_col is None:
        label_col = guess_label_column(df)
    if seq_col is None:
        seq_col = guess_sequence_column(df)

    if label_col is None:
        raise ValueError("Не удалось автоматически определить колонку с меткой. Укажи --label-col")
    if seq_col is None:
        raise ValueError("Не удалось автоматически определить колонку с последовательностью. Укажи --seq-col")

    if label_col not in df.columns:
        raise ValueError(f"Колонка метки не найдена: {label_col}")
    if seq_col not in df.columns:
        raise ValueError(f"Колонка последовательности не найдена: {seq_col}")

    df = df[[label_col, seq_col]].copy()
    df = df.dropna(subset=[label_col, seq_col]).reset_index(drop=True)
    return df, label_col, seq_col


def make_graph_dataset(
    df: pd.DataFrame,
    label_col: str,
    seq_col: str,
    delimiter_regex: str,
    min_token_freq: int,
    max_seq_len: int,
    undirected: bool,
    add_skip_edges: bool
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
        raise ValueError("После парсинга не осталось валидных последовательностей")

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
            add_skip_edges=add_skip_edges
        )
        graphs.append(data)

    return graphs, vocab


def split_dataset(graphs: List[Data], test_size: float, val_size: float, seed: int):
    labels = [int(g.y.item()) for g in graphs]

    train_val_idx, test_idx = train_test_split(
        np.arange(len(graphs)),
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )

    train_val_labels = [labels[i] for i in train_val_idx]

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=seed,
        stratify=train_val_labels
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


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    print("Loading CSV...")
    df, label_col, seq_col = load_and_prepare_dataframe(
        csv_path=args.csv,
        label_col=args.label_col,
        seq_col=args.seq_col
    )

    print(f"Detected label column: {label_col}")
    print(f"Detected sequence column: {seq_col}")
    print(f"Rows after dropna: {len(df)}")

    print("Building graph dataset...")
    graphs, vocab = make_graph_dataset(
        df=df,
        label_col=label_col,
        seq_col=seq_col,
        delimiter_regex=args.delimiter_regex,
        min_token_freq=args.min_token_freq,
        max_seq_len=args.max_seq_len,
        undirected=args.undirected,
        add_skip_edges=args.add_skip_edges
    )

    with open(args.vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Saved vocab to: {args.vocab_path}")

    train_graphs, val_graphs, test_graphs = split_dataset(
        graphs=graphs,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed
    )

    print_dataset_stats(train_graphs, "TRAIN")
    print_dataset_stats(val_graphs, "VAL")
    print_dataset_stats(test_graphs, "TEST")

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    num_node_features = train_graphs[0].num_node_features
    edge_dim = train_graphs[0].edge_attr.size(1)

    device = torch.device(args.device)
    model = GINEMalwareClassifier(
        num_node_features=num_node_features,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_f1 = -1.0
    best_epoch = -1

    print("\nStart training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), args.save_path)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Saved best model to: {args.save_path}")

    print("\nEvaluating best checkpoint on test...")
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    print("\n[Test metrics]")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()