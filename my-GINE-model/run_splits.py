#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader

from data_io import guess_label_column
from dataset_adapters import load_and_prepare_with_adapter
from graph_data import (
    build_graph_from_sequence,
    make_graph_dataset,
    normalize_label,
    sequence_to_ids,
    split_api_sequence,
)
from model import GINEMalwareClassifier
from trainer import train_model
from utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple train/eval splits and save aggregated metrics"
    )
    parser.add_argument("--csv", type=str, required=True, help="Source dataset CSV")
    parser.add_argument("--label-col", type=str, default=None, help="Label column in source CSV")
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="auto",
        choices=["auto", "single_seq", "wide_api"],
    )
    parser.add_argument("--seq-col", type=str, default=None)
    parser.add_argument("--api-col-regex", type=str, default=r"^\d+$")
    parser.add_argument("--include-unnamed-api-cols", action="store_true")

    parser.add_argument("--train-per-class", type=int, default=650)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)

    parser.add_argument("--delimiter-regex", type=str, default=r"[,\s;|>]+")
    parser.add_argument("--min-token-freq", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=400)
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--add-skip-edges", action="store_true")

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument(
        "--output-root",
        type=str,
        default="runs/multi_split_eval",
        help="A new experiment folder will be created inside this root",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def split_raw_dataset(
    df: pd.DataFrame,
    label_col: str,
    train_per_class: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tmp = df.copy()
    tmp["_norm_label"] = tmp[label_col].map(normalize_label)
    tmp = tmp.dropna(subset=["_norm_label"]).reset_index(drop=True)

    classes = sorted(tmp["_norm_label"].unique().tolist())
    if classes != [0, 1]:
        raise ValueError(f"Expected binary labels 0/1 after normalization, got: {classes}")

    sampled_indices: List[int] = []
    for cls in classes:
        cls_idx = tmp[tmp["_norm_label"] == cls].index
        if len(cls_idx) < train_per_class:
            raise ValueError(
                f"Class {cls} has only {len(cls_idx)} rows, but train-per-class={train_per_class}"
            )
        picked = (
            tmp.loc[cls_idx]
            .sample(n=train_per_class, random_state=seed)
            .index
            .tolist()
        )
        sampled_indices.extend(picked)

    train_df = tmp.loc[sampled_indices].drop(columns=["_norm_label"]).reset_index(drop=True)
    eval_df = tmp.drop(index=sampled_indices).drop(columns=["_norm_label"]).reset_index(drop=True)
    return train_df, eval_df


def make_eval_graphs_with_fixed_vocab(
    df: pd.DataFrame,
    label_col: str,
    seq_col: str,
    delimiter_regex: str,
    vocab: Dict[str, int],
    max_seq_len: int,
    undirected: bool,
    add_skip_edges: bool,
):
    graphs = []
    for _, row in df.iterrows():
        try:
            label = normalize_label(row[label_col])
            tokens = split_api_sequence(row[seq_col], delimiter_regex)
            if not tokens:
                continue
            api_ids = sequence_to_ids(tokens, vocab, max_seq_len=max_seq_len)
            graph = build_graph_from_sequence(
                api_ids=api_ids,
                label=label,
                vocab_size=len(vocab),
                undirected=undirected,
                add_skip_edges=add_skip_edges,
            )
            graphs.append(graph)
        except Exception:
            continue
    return graphs


@torch.no_grad()
def evaluate_with_confusion(model, loader, device, threshold: float):
    model.eval()
    all_true: List[int] = []
    all_prob: List[float] = []
    all_pred: List[int] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist()
        preds = [int(p >= threshold) for p in probs]
        labels = batch.y.detach().cpu().numpy().tolist()

        all_true.extend([int(x) for x in labels])
        all_prob.extend([float(x) for x in probs])
        all_pred.extend([int(x) for x in preds])

    y_true = np.array(all_true)
    y_prob = np.array(all_prob)
    y_pred = np.array(all_pred)

    from trainer import compute_metrics

    metrics = compute_metrics(y_true, y_pred, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    metrics["pred_benign"] = int((y_pred == 0).sum())
    metrics["pred_malware"] = int((y_pred == 1).sum())
    metrics["true_benign"] = int((y_true == 0).sum())
    metrics["true_malware"] = int((y_true == 1).sum())
    return metrics


def write_run_report(path: str, seed: int, train_size: int, eval_size: int, metrics: Dict[str, float]) -> None:
    lines = [
        f"Seed: {seed}",
        f"Train samples: {train_size}",
        f"Eval samples: {eval_size}",
        "",
        "[Metrics]",
        f"accuracy: {metrics['accuracy']:.4f}",
        f"precision: {metrics['precision']:.4f}",
        f"recall: {metrics['recall']:.4f}",
        f"f1: {metrics['f1']:.4f}",
        f"roc_auc: {metrics['roc_auc']:.4f}",
        "",
        "[Confusion matrix]",
        "rows=true, cols=pred",
        "         pred_0  pred_1",
        f"true_0   {metrics['tn']:<6}  {metrics['fp']}",
        f"true_1   {metrics['fn']:<6}  {metrics['tp']}",
        "",
        "[Counts]",
        f"True benign (0): {metrics['true_benign']}",
        f"True malware (1): {metrics['true_malware']}",
        f"Pred benign (0): {metrics['pred_benign']}",
        f"Pred malware (1): {metrics['pred_malware']}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    set_seed(args.base_seed)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_root, f"exp_{timestamp}")
    splits_dir = os.path.join(exp_dir, "splits")
    models_dir = os.path.join(exp_dir, "models")
    vocab_dir = os.path.join(exp_dir, "vocab")
    metrics_dir = os.path.join(exp_dir, "metrics")
    summary_dir = os.path.join(exp_dir, "summary")
    for d in [splits_dir, models_dir, vocab_dir, metrics_dir, summary_dir]:
        ensure_dir(d)

    raw_df = pd.read_csv(args.csv)
    label_col = args.label_col or guess_label_column(raw_df)
    if label_col is None:
        raise ValueError("Could not detect label column. Pass --label-col")
    if label_col not in raw_df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    all_rows: List[Dict[str, float]] = []
    device = torch.device(args.device)

    print(f"Experiment dir: {exp_dir}")
    print(f"Runs: {args.num_runs}, train-per-class: {args.train_per_class}")

    for i in range(args.num_runs):
        seed = args.base_seed + i
        set_seed(seed)
        print(f"\n=== Run {i + 1}/{args.num_runs} | seed={seed} ===")

        train_raw, eval_raw = split_raw_dataset(
            df=raw_df,
            label_col=label_col,
            train_per_class=args.train_per_class,
            seed=seed,
        )

        train_csv = os.path.join(splits_dir, f"seed{seed}_train.csv")
        eval_csv = os.path.join(splits_dir, f"seed{seed}_eval.csv")
        train_raw.to_csv(train_csv, index=False)
        eval_raw.to_csv(eval_csv, index=False)

        train_df, train_label_col, train_seq_col, _ = load_and_prepare_with_adapter(
            csv_path=train_csv,
            label_col=label_col,
            seq_col=args.seq_col,
            dataset_format=args.dataset_format,
            api_col_regex=args.api_col_regex,
            include_unnamed_api_cols=args.include_unnamed_api_cols,
        )
        eval_df, eval_label_col, eval_seq_col, _ = load_and_prepare_with_adapter(
            csv_path=eval_csv,
            label_col=label_col,
            seq_col=args.seq_col,
            dataset_format=args.dataset_format,
            api_col_regex=args.api_col_regex,
            include_unnamed_api_cols=args.include_unnamed_api_cols,
        )

        train_graphs, vocab = make_graph_dataset(
            df=train_df,
            label_col=train_label_col,
            seq_col=train_seq_col,
            delimiter_regex=args.delimiter_regex,
            min_token_freq=args.min_token_freq,
            max_seq_len=args.max_seq_len,
            undirected=args.undirected,
            add_skip_edges=args.add_skip_edges,
        )
        eval_graphs = make_eval_graphs_with_fixed_vocab(
            df=eval_df,
            label_col=eval_label_col,
            seq_col=eval_seq_col,
            delimiter_regex=args.delimiter_regex,
            vocab=vocab,
            max_seq_len=args.max_seq_len,
            undirected=args.undirected,
            add_skip_edges=args.add_skip_edges,
        )
        if len(eval_graphs) == 0:
            raise ValueError("Eval split has no valid graphs after preprocessing")

        vocab_path = os.path.join(vocab_dir, f"seed{seed}_api_vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        model_path = os.path.join(models_dir, f"seed{seed}_g_h{args.hidden_dim}l{args.num_layers}.pt")

        train_loader = DataLoader(
            train_graphs,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        eval_loader = DataLoader(
            eval_graphs,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        model = GINEMalwareClassifier(
            num_node_features=train_graphs[0].num_node_features,
            edge_dim=train_graphs[0].edge_attr.size(1),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=2,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_epoch, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            save_path=model_path,
        )

        model.load_state_dict(torch.load(model_path, map_location=device))
        metrics = evaluate_with_confusion(model, eval_loader, device=device, threshold=args.threshold)
        metrics["seed"] = seed
        metrics["train_samples"] = len(train_graphs)
        metrics["eval_samples"] = len(eval_graphs)
        metrics["best_epoch"] = best_epoch
        all_rows.append(metrics)

        report_path = os.path.join(metrics_dir, f"seed{seed}_metrics.txt")
        write_run_report(
            path=report_path,
            seed=seed,
            train_size=len(train_graphs),
            eval_size=len(eval_graphs),
            metrics=metrics,
        )
        print(
            f"run seed={seed} | acc={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f}"
        )

    summary_df = pd.DataFrame(all_rows)
    summary_csv = os.path.join(summary_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    agg_lines = ["[Aggregate metrics]"]
    for col in metric_cols:
        values = summary_df[col].astype(float)
        agg_lines.append(
            f"{col}: mean={values.mean():.4f}, std={values.std(ddof=0):.4f}, "
            f"min={values.min():.4f}, max={values.max():.4f}"
        )

    agg_txt = os.path.join(summary_dir, "aggregate.txt")
    with open(agg_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(agg_lines) + "\n")

    print("\nCompleted.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Aggregate TXT: {agg_txt}")
    print(f"All artifacts saved in: {exp_dir}")


if __name__ == "__main__":
    main()
