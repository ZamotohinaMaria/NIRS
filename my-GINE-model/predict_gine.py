#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from hashlib import sha1
from typing import List, Optional, Tuple

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch_geometric.loader import DataLoader

from data_io import guess_label_column, guess_sequence_column
from graph_data import build_graph_from_sequence, normalize_label, sequence_to_ids, split_api_sequence
from model import GINEMalwareClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict malware probability with trained GINE")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .pt model checkpoint")
    parser.add_argument("--vocab-path", type=str, required=True, help="Path to api_vocab.json")

    parser.add_argument("--input-seq", type=str, default=None, help="Single API sequence string")
    parser.add_argument("--input-csv", type=str, default=None, help="CSV with samples to classify")
    parser.add_argument("--label-col", type=str, default=None, help="Label column for evaluation metrics on CSV")

    parser.add_argument(
        "--dataset-format",
        type=str,
        default="auto",
        choices=["auto", "single_seq", "wide_api"],
        help="Input CSV format",
    )
    parser.add_argument("--seq-col", type=str, default=None, help="Sequence column for single_seq format")
    parser.add_argument(
        "--api-col-regex",
        type=str,
        default=r"^\d+$",
        help="Regex for API columns in wide_api format",
    )
    parser.add_argument(
        "--include-unnamed-api-cols",
        action="store_true",
        help="Include non-empty trailing Unnamed:* columns in wide_api format",
    )
    parser.add_argument("--id-col", type=str, default=None, help="Optional ID column (kept for compatibility)")

    parser.add_argument("--hidden-dim", type=int, default=None, help="Model hidden dim (if not parsable from filename)")
    parser.add_argument("--num-layers", type=int, default=None, help="Model layers (if not parsable from filename)")
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--delimiter-regex", type=str, default=r"[,\s;|>]+")
    parser.add_argument("--max-seq-len", type=int, default=400)
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--add-skip-edges", action="store_true")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for malware class")

    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument(
        "--predictions-path",
        type=str,
        default=None,
        help="Output metrics text path for batch prediction. If omitted, auto-generated under <output-dir>/predictions",
    )
    return parser.parse_args()


def resolve_output_path(path: str, output_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(output_dir, path)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def parse_arch_from_model_name(model_path: str) -> Tuple[Optional[int], Optional[int]]:
    model_name = os.path.basename(model_path)
    match = re.search(r"g_h(\d+)l(\d+)_", model_name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def collect_wide_api_cols(
    df: pd.DataFrame,
    api_col_regex: str,
    include_unnamed_api_cols: bool,
) -> List[str]:
    pattern = re.compile(api_col_regex)
    api_cols = [c for c in df.columns if pattern.match(str(c))]
    api_cols = sorted(api_cols, key=lambda c: int(c))

    if include_unnamed_api_cols:
        unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:") and df[c].notna().any()]
        unnamed_cols = sorted(
            unnamed_cols,
            key=lambda c: int(re.search(r"(\d+)$", c).group(1)) if re.search(r"(\d+)$", c) else 10**9,
        )
        api_cols.extend(unnamed_cols)
    return api_cols


def row_to_sequence(values: List[object]) -> str:
    tokens: List[str] = []
    for value in values:
        if pd.isna(value):
            continue
        token = str(value).strip()
        if token and token.lower() != "nan":
            tokens.append(token)
    return " ".join(tokens)


def resolve_csv_sequences(
    csv_path: str,
    dataset_format: str,
    label_col: Optional[str],
    seq_col: Optional[str],
    api_col_regex: str,
    include_unnamed_api_cols: bool,
    id_col: Optional[str],
) -> Tuple[pd.DataFrame, List[str], str, Optional[List[object]], Optional[str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if dataset_format == "auto":
        numeric_cols = [c for c in df.columns if re.match(api_col_regex, str(c))]
        dataset_format = "wide_api" if len(numeric_cols) >= 10 else "single_seq"

    out_df = pd.DataFrame(index=df.index)
    if id_col:
        if id_col not in df.columns:
            raise ValueError(f"id column not found: {id_col}")
        out_df[id_col] = df[id_col]
    else:
        out_df["row_id"] = df.index

    if dataset_format == "single_seq":
        if seq_col is None:
            seq_col = guess_sequence_column(df)
        if seq_col is None or seq_col not in df.columns:
            raise ValueError("Could not resolve sequence column for single_seq format")
        seq_series = df[seq_col].fillna("").astype(str).map(str.strip)
        sequences = seq_series.tolist()
        source = seq_col
    elif dataset_format == "wide_api":
        api_cols = collect_wide_api_cols(df, api_col_regex=api_col_regex, include_unnamed_api_cols=include_unnamed_api_cols)
        if not api_cols:
            raise ValueError("No API columns found for wide_api format")
        sequences = df[api_cols].apply(lambda r: row_to_sequence(r.tolist()), axis=1).tolist()
        source = f"{len(api_cols)} wide API columns"
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

    resolved_label_col = label_col
    if resolved_label_col is None:
        resolved_label_col = guess_label_column(df)
    label_values: Optional[List[object]] = None
    if resolved_label_col is not None and resolved_label_col in df.columns:
        label_values = df[resolved_label_col].tolist()

    return out_df, sequences, source, label_values, resolved_label_col


def build_graphs_from_sequences(
    sequences: List[str],
    vocab: dict,
    delimiter_regex: str,
    max_seq_len: int,
    undirected: bool,
    add_skip_edges: bool,
):
    graphs = []
    kept_indices = []
    for idx, seq in enumerate(sequences):
        tokens = split_api_sequence(seq, delimiter_regex)
        if not tokens:
            continue
        api_ids = sequence_to_ids(tokens, vocab, max_seq_len=max_seq_len)
        graph = build_graph_from_sequence(
            api_ids=api_ids,
            label=0,
            vocab_size=len(vocab),
            undirected=undirected,
            add_skip_edges=add_skip_edges,
        )
        graphs.append(graph)
        kept_indices.append(idx)
    return graphs, kept_indices


@torch.no_grad()
def predict_graphs(model, graphs, batch_size: int, device: torch.device) -> List[float]:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    probs: List[float] = []
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
        probs.extend([float(x) for x in p])
    return probs


def build_auto_predictions_path(args: argparse.Namespace, output_dir: str) -> str:
    model_tag = sanitize_token(os.path.splitext(os.path.basename(args.model_path))[0]) or "model"
    input_tag = sanitize_token(os.path.splitext(os.path.basename(args.input_csv))[0]) if args.input_csv else "single"
    signature = f"{model_tag}|{input_tag}|{args.threshold}|{args.max_seq_len}|{int(args.undirected)}|{int(args.add_skip_edges)}"
    sig_hash = sha1(signature.encode("utf-8")).hexdigest()[:8]
    filename = f"metrics_{model_tag}_{input_tag}_{sig_hash}.txt"
    return os.path.join(output_dir, "predictions", filename)


def format_float(value: float) -> str:
    return f"{value:.4f}"


def build_metrics_report(
    rows_in_csv: int,
    rows_non_empty: int,
    threshold: float,
    probs_all: List[float],
    preds_all: List[int],
    y_true_eval: Optional[List[int]],
    y_pred_eval: Optional[List[int]],
    y_prob_eval: Optional[List[float]],
):
    pred_malware = int(sum(preds_all))
    pred_benign = int(len(preds_all) - pred_malware)
    lines: List[str] = []

    lines.append("[Prediction summary]")
    lines.append(f"Rows in CSV: {rows_in_csv}")
    lines.append(f"Rows with non-empty sequence: {rows_non_empty}")
    lines.append(f"Threshold: {threshold:.4f}")
    lines.append(f"Predicted benign (0): {pred_benign}")
    lines.append(f"Predicted malware (1): {pred_malware}")
    lines.append(f"Mean malware probability: {sum(probs_all) / max(len(probs_all), 1):.4f}")

    has_eval = (
        y_true_eval is not None
        and y_pred_eval is not None
        and y_prob_eval is not None
        and len(y_true_eval) > 0
    )
    if not has_eval:
        lines.append("")
        lines.append("Evaluation metrics are unavailable: no valid labels were found.")
        return lines

    lines.append(f"Rows with valid labels for evaluation: {len(y_true_eval)}")

    acc = accuracy_score(y_true_eval, y_pred_eval)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_eval, y_pred_eval, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true_eval, y_prob_eval)
    except Exception:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true_eval, y_pred_eval, labels=[0, 1]).ravel()
    true_benign = int(sum(1 for x in y_true_eval if x == 0))
    true_malware = int(sum(1 for x in y_true_eval if x == 1))

    lines.append("")
    lines.append("[Evaluation metrics]")
    lines.append(f"accuracy: {format_float(float(acc))}")
    lines.append(f"precision: {format_float(float(precision))}")
    lines.append(f"recall: {format_float(float(recall))}")
    lines.append(f"f1: {format_float(float(f1))}")
    lines.append(f"roc_auc: {format_float(float(auc))}" if auc == auc else "roc_auc: nan")

    lines.append("")
    lines.append("[Confusion matrix]")
    lines.append("rows=true, cols=pred")
    lines.append("         pred_0  pred_1")
    lines.append(f"true_0   {tn:<6}  {fp}")
    lines.append(f"true_1   {fn:<6}  {tp}")

    lines.append("")
    lines.append("[Counts]")
    lines.append(f"True benign (0): {true_benign}")
    lines.append(f"True malware (1): {true_malware}")
    lines.append(f"Pred benign (0): {pred_benign}")
    lines.append(f"Pred malware (1): {pred_malware}")

    return lines


def main():
    args = parse_args()

    if bool(args.input_seq) == bool(args.input_csv):
        raise ValueError("Set exactly one input source: --input-seq OR --input-csv")

    hidden_dim, num_layers = parse_arch_from_model_name(args.model_path)
    if args.hidden_dim is not None:
        hidden_dim = args.hidden_dim
    if args.num_layers is not None:
        num_layers = args.num_layers
    if hidden_dim is None or num_layers is None:
        raise ValueError(
            "Could not infer hidden_dim/num_layers from model filename. "
            "Pass --hidden-dim and --num-layers explicitly."
        )

    with open(args.vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    device = torch.device(args.device)
    model = GINEMalwareClassifier(
        num_node_features=len(vocab),
        edge_dim=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    if args.input_seq:
        graphs, _ = build_graphs_from_sequences(
            sequences=[args.input_seq],
            vocab=vocab,
            delimiter_regex=args.delimiter_regex,
            max_seq_len=args.max_seq_len,
            undirected=args.undirected,
            add_skip_edges=args.add_skip_edges,
        )
        if not graphs:
            raise ValueError("Input sequence became empty after tokenization")
        prob = predict_graphs(model=model, graphs=graphs, batch_size=1, device=device)[0]
        pred = int(prob >= args.threshold)
        print(f"malware_prob: {prob:.6f}")
        print(f"pred_label: {pred}")
        return

    _, sequences, source, label_values, resolved_label_col = resolve_csv_sequences(
        csv_path=args.input_csv,
        dataset_format=args.dataset_format,
        label_col=args.label_col,
        seq_col=args.seq_col,
        api_col_regex=args.api_col_regex,
        include_unnamed_api_cols=args.include_unnamed_api_cols,
        id_col=args.id_col,
    )
    print(f"Resolved input source: {source}")
    print(f"Rows in CSV: {len(sequences)}")

    graphs, kept_indices = build_graphs_from_sequences(
        sequences=sequences,
        vocab=vocab,
        delimiter_regex=args.delimiter_regex,
        max_seq_len=args.max_seq_len,
        undirected=args.undirected,
        add_skip_edges=args.add_skip_edges,
    )
    print(f"Rows with non-empty sequence after parsing: {len(graphs)}")
    if not graphs:
        raise ValueError("No valid sequences for prediction")

    probs = predict_graphs(model=model, graphs=graphs, batch_size=args.batch_size, device=device)
    preds = [int(p >= args.threshold) for p in probs]

    y_true_eval: Optional[List[int]] = None
    y_pred_eval: Optional[List[int]] = None
    y_prob_eval: Optional[List[float]] = None
    if label_values is not None:
        y_true_eval = []
        y_prob_eval = []
        y_pred_eval = []
        for local_idx, original_idx in enumerate(kept_indices):
            try:
                label = normalize_label(label_values[original_idx])
            except Exception:
                continue
            y_true_eval.append(label)
            y_prob_eval.append(probs[local_idx])
            y_pred_eval.append(preds[local_idx])
        if len(y_true_eval) == 0:
            y_true_eval = None
            y_pred_eval = None
            y_prob_eval = None

    report_lines = build_metrics_report(
        rows_in_csv=len(sequences),
        rows_non_empty=len(graphs),
        threshold=args.threshold,
        probs_all=probs,
        preds_all=preds,
        y_true_eval=y_true_eval,
        y_pred_eval=y_pred_eval,
        y_prob_eval=y_prob_eval,
    )
    if resolved_label_col:
        report_lines.insert(1, f"Resolved label column: {resolved_label_col}")

    if args.predictions_path:
        pred_path = resolve_output_path(args.predictions_path, args.output_dir)
    else:
        pred_path = build_auto_predictions_path(args, output_dir=args.output_dir)
    ensure_parent_dir(pred_path)
    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print("\n".join(report_lines))
    print("")
    print(f"Saved prediction metrics to: {pred_path}")


if __name__ == "__main__":
    main()
