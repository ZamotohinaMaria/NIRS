#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import contextlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from hashlib import sha1
from typing import Any, Dict

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from cli import parse_args
from dataset_adapters import load_and_prepare_with_adapter
from graph_data import make_graph_dataset, print_dataset_stats, split_dataset
from model import GINEMalwareClassifier
from trainer import (
    evaluate,
    find_best_threshold,
    fit_temperature_scaler,
    probs_from_logits,
    train_model,
)
from utils import set_seed


class TeeStream:
    def __init__(self, primary_stream, mirror_stream=None):
        self.primary_stream = primary_stream
        self.mirror_stream = mirror_stream

    def write(self, data: str) -> None:
        self.primary_stream.write(data)
        if self.mirror_stream is not None:
            try:
                self.mirror_stream.write(data)
            except Exception:
                self.mirror_stream = None

    def flush(self) -> None:
        self.primary_stream.flush()
        if self.mirror_stream is not None:
            try:
                self.mirror_stream.flush()
            except Exception:
                self.mirror_stream = None


def resolve_output_path(path: str, output_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(output_dir, path)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def _fmt_num(value) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace(".", "p").replace("-", "m")


def _metric_tag(value: float) -> str:
    if value != value:  # nan
        return "na"
    return f"{int(round(value * 1000)):03d}"


def parse_threshold_grid(spec: str) -> np.ndarray:
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("Empty --threshold-grid")

    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --threshold-grid '{spec}'. Expected start:end:points format."
            )
        start = float(parts[0])
        end = float(parts[1])
        points = int(parts[2])
        if points < 2:
            raise ValueError("--threshold-grid points must be >= 2")
        grid = np.linspace(start, end, points)
    else:
        values = [float(p.strip()) for p in spec.split(",") if p.strip()]
        if not values:
            raise ValueError("No threshold values parsed from --threshold-grid")
        grid = np.array(values, dtype=float)

    valid = [float(v) for v in grid if 0.0 < float(v) < 1.0]
    if not valid:
        raise ValueError("Threshold grid must contain values strictly between 0 and 1")
    return np.array(sorted(set(valid)), dtype=float)


def build_auto_model_path(args, output_dir: str) -> str:
    signature = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "min_token_freq": args.min_token_freq,
        "undirected": int(args.undirected),
        "add_skip_edges": int(args.add_skip_edges),
    }
    signature_raw = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature_hash = sha1(signature_raw).hexdigest()[:8]
    file_name = (
        f"g_h{args.hidden_dim}l{args.num_layers}"
        f"_b{args.batch_size}e{args.epochs}"
        f"_lr{_fmt_num(args.lr)}"
        f"_s{args.seed}_{signature_hash}.pt"
    )
    return os.path.join(output_dir, "models", file_name)


def finalize_auto_model_path(save_path: str, best_epoch: int, best_val_f1: float) -> str:
    base, ext = os.path.splitext(save_path)
    candidate = f"{base}_be{best_epoch}_vf{_metric_tag(best_val_f1)}{ext}"
    if candidate == save_path:
        return save_path
    if os.path.exists(candidate):
        candidate = f"{base}_be{best_epoch}_vf{_metric_tag(best_val_f1)}_{sha1(candidate.encode('utf-8')).hexdigest()[:6]}{ext}"
    os.replace(save_path, candidate)
    return candidate


def build_auto_results_path(args, adapter_info: dict, output_dir: str) -> str:
    csv_stem = os.path.splitext(os.path.basename(args.csv))[0]
    dataset_name = _sanitize_token(csv_stem) or "dataset"
    fmt = _sanitize_token(adapter_info.get("resolved_format", "auto"))

    signature = {
        "dataset": dataset_name,
        "format": fmt,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_seq_len": args.max_seq_len,
        "min_token_freq": args.min_token_freq,
        "undirected": int(args.undirected),
        "add_skip_edges": int(args.add_skip_edges),
        "test_size": args.test_size,
        "val_size": args.val_size,
        "seed": args.seed,
    }
    signature_raw = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature_hash = sha1(signature_raw).hexdigest()[:10]

    filename = (
        f"{dataset_name}_{fmt}_hd{args.hidden_dim}_l{args.num_layers}_"
        f"bs{args.batch_size}_ep{args.epochs}_lr{_fmt_num(args.lr)}_"
        f"wd{_fmt_num(args.weight_decay)}_seed{args.seed}_{signature_hash}.txt"
    )
    return os.path.join(output_dir, "results", filename)


def collect_split_stats(graphs) -> dict:
    labels = [int(g.y.item()) for g in graphs]
    counter = Counter(labels)
    avg_nodes = np.mean([g.num_nodes for g in graphs]) if graphs else 0.0
    avg_edges = np.mean([g.num_edges for g in graphs]) if graphs else 0.0
    return {
        "samples": len(graphs),
        "class_distribution": dict(counter),
        "avg_nodes": float(avg_nodes),
        "avg_edges": float(avg_edges),
    }


def format_split_block(split_name: str, stats: dict) -> list[str]:
    return [
        f"[{split_name}]",
        f"Samples: {stats['samples']}",
        f"Class distribution: {stats['class_distribution']}",
        f"Avg nodes: {stats['avg_nodes']:.2f}",
        f"Avg edges: {stats['avg_edges']:.2f}",
    ]


def build_auto_train_log_path(args, output_dir: str) -> str:
    dataset_name = _sanitize_token(os.path.splitext(os.path.basename(args.csv))[0]) or "dataset"
    started_at = datetime.now().strftime("%M-%H__%d-%m-%Y")
    file_name = f"{dataset_name}+e{args.epochs}+b{args.batch_size}+{started_at}.txt"
    return os.path.join(output_dir, "models", file_name)


def _is_checkpoint_dict(payload: Any) -> bool:
    return isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict)


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if _is_checkpoint_dict(payload):
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Unsupported checkpoint format: expected state_dict or checkpoint dict with 'state_dict'.")


def save_checkpoint_dict(
    path: str,
    state_dict: Dict[str, torch.Tensor],
    model_args: Dict[str, Any],
    vocab_size: int,
    threshold: float = 0.5,
    temperature: float = 1.0,
) -> None:
    checkpoint = {
        "schema_version": 1,
        "state_dict": state_dict,
        "model_args": model_args,
        "vocab_size": int(vocab_size),
        "threshold": float(threshold),
        "temperature": float(temperature),
    }
    torch.save(checkpoint, path)


def run_training(args) -> None:
    set_seed(args.seed)

    auto_model_name = args.save_path is None
    if auto_model_name:
        save_path = build_auto_model_path(args, output_dir=args.output_dir)
    else:
        save_path = resolve_output_path(args.save_path, args.output_dir)
    vocab_path = resolve_output_path(args.vocab_path, args.output_dir)
    ensure_parent_dir(save_path)
    ensure_parent_dir(vocab_path)

    print("Loading CSV...")
    df, label_col, seq_col, adapter_info = load_and_prepare_with_adapter(
        csv_path=args.csv,
        label_col=args.label_col,
        seq_col=args.seq_col,
        dataset_format=args.dataset_format,
        api_col_regex=args.api_col_regex,
        include_unnamed_api_cols=args.include_unnamed_api_cols,
    )

    print(f"Adapter format: {adapter_info['resolved_format']}")
    print(f"Source sequence columns: {adapter_info['source_sequence_columns']}")
    print(f"Raw rows: {adapter_info['raw_rows']}")
    print(f"Prepared rows: {adapter_info['prepared_rows']}")
    print(f"Detected label column: {label_col}")
    print(f"Detected sequence column: {seq_col}")
    print(f"Rows after preparation: {len(df)}")

    print("Building graph dataset...")
    graphs, vocab = make_graph_dataset(
        df=df,
        label_col=label_col,
        seq_col=seq_col,
        delimiter_regex=args.delimiter_regex,
        min_token_freq=args.min_token_freq,
        max_seq_len=args.max_seq_len,
        undirected=args.undirected,
        add_skip_edges=args.add_skip_edges,
    )

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Saved vocab to: {vocab_path}")

    if args.no_internal_split:
        train_graphs = graphs
        val_graphs = []
        test_graphs = []
        print("\nInternal split disabled (--no-internal-split).")
        print_dataset_stats(train_graphs, "TRAIN_ALL")
    else:
        train_graphs, val_graphs, test_graphs = split_dataset(
            graphs=graphs,
            test_size=args.test_size,
            val_size=args.val_size,
            seed=args.seed,
        )
        print_dataset_stats(train_graphs, "TRAIN")
        print_dataset_stats(val_graphs, "VAL")
        print_dataset_stats(test_graphs, "TEST")

    train_stats = collect_split_stats(train_graphs)
    val_stats = collect_split_stats(val_graphs) if val_graphs else None
    test_stats = collect_split_stats(test_graphs) if test_graphs else None

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = (
        DataLoader(
            val_graphs,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if len(val_graphs) > 0
        else None
    )
    test_loader = (
        DataLoader(
            test_graphs,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if len(test_graphs) > 0
        else None
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
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_epoch, best_val_score = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        save_path=save_path,
        monitor_metric=args.select_metric,
    )

    tuned_threshold = 0.5
    tuned_temperature = 1.0
    tuned_val_metrics = None
    threshold_metric_score = float("nan")

    if val_loader is not None:
        model.load_state_dict(_extract_state_dict(torch.load(save_path, map_location=device)))
        _, val_details = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            threshold=0.5,
            temperature=1.0,
            return_details=True,
        )
        val_y_true = val_details["y_true"]
        val_logits = val_details["logits"]

        if args.calibration == "temperature":
            tuned_temperature = fit_temperature_scaler(
                logits=val_logits,
                y_true=val_y_true,
                device=device,
            )
        val_probs = probs_from_logits(val_logits, temperature=tuned_temperature)
        threshold_grid = parse_threshold_grid(args.threshold_grid)
        tuned_threshold, threshold_metric_score = find_best_threshold(
            y_true=val_y_true,
            y_prob=val_probs,
            metric=args.select_metric,
            threshold_grid=threshold_grid,
        )
        tuned_val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            threshold=tuned_threshold,
            temperature=tuned_temperature,
        )

    print(f"\nBest epoch: {best_epoch}")
    if val_loader is not None:
        print(f"Best val {args.select_metric.upper()}: {best_val_score:.4f}")
        print(f"Tuned threshold ({args.select_metric}): {tuned_threshold:.4f}")
        print(f"Temperature: {tuned_temperature:.4f}")
        if tuned_val_metrics is not None:
            print(
                "Tuned val metrics | "
                f"f1={tuned_val_metrics['f1']:.4f} "
                f"mcc={tuned_val_metrics['mcc']:.4f} "
                f"pr_auc={tuned_val_metrics['pr_auc']:.4f} "
                f"brier={tuned_val_metrics['brier']:.4f}"
            )
    else:
        print(f"Best val {args.select_metric.upper()}: n/a (no internal split)")

    if auto_model_name:
        save_path = finalize_auto_model_path(save_path, best_epoch=best_epoch, best_val_f1=best_val_score)

    raw_payload = torch.load(save_path, map_location="cpu")
    state_dict = _extract_state_dict(raw_payload)
    save_checkpoint_dict(
        path=save_path,
        state_dict=state_dict,
        model_args={
            "num_node_features": int(num_node_features),
            "edge_dim": int(edge_dim),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "num_classes": 2,
            "dropout": float(args.dropout),
        },
        vocab_size=len(vocab),
        threshold=tuned_threshold,
        temperature=tuned_temperature,
    )

    print(f"Saved best model to: {save_path}")
    test_metrics = None
    if test_loader is not None:
        print("\nEvaluating best checkpoint on test...")
        best_payload = torch.load(save_path, map_location=device)
        model.load_state_dict(_extract_state_dict(best_payload))
        test_metrics = evaluate(model, test_loader, device)

        print("\n[Test metrics]")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")
    else:
        print("\nTest evaluation skipped (no internal split).")

    if args.results_path:
        results_path = resolve_output_path(args.results_path, args.output_dir)
    else:
        results_path = build_auto_results_path(args, adapter_info=adapter_info, output_dir=args.output_dir)
    ensure_parent_dir(results_path)

    report_lines: list[str] = []
    report_lines.extend(format_split_block("TRAIN", train_stats))
    if val_stats is not None:
        report_lines.append("")
        report_lines.extend(format_split_block("VAL", val_stats))
    if test_stats is not None:
        report_lines.append("")
        report_lines.extend(format_split_block("TEST", test_stats))
    report_lines.append("")
    report_lines.append(f"Best epoch: {best_epoch}")
    if val_loader is not None:
        report_lines.append(f"Best val {args.select_metric.upper()}: {best_val_score:.4f}")
        report_lines.append(f"Tuned threshold ({args.select_metric}): {tuned_threshold:.4f}")
        report_lines.append(f"Temperature: {tuned_temperature:.4f}")
        if threshold_metric_score == threshold_metric_score:
            report_lines.append(f"Tuning score: {threshold_metric_score:.4f}")
        if tuned_val_metrics is not None:
            report_lines.append("")
            report_lines.append("[Validation metrics @ tuned threshold]")
            for k, v in tuned_val_metrics.items():
                report_lines.append(f"{k}: {v:.4f}")
    else:
        report_lines.append(f"Best val {args.select_metric.upper()}: n/a (no internal split)")
    report_lines.append(f"Saved best model to: {save_path}")
    if test_metrics is not None:
        report_lines.append("")
        report_lines.append("Evaluating best checkpoint on test...")
        report_lines.append("")
        report_lines.append("[Test metrics]")
        for k, v in test_metrics.items():
            report_lines.append(f"{k}: {v:.4f}")
    else:
        report_lines.append("")
        report_lines.append("Test evaluation skipped (no internal split).")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Saved run summary to: {results_path}")


def main():
    args = parse_args()
    train_log_path = build_auto_train_log_path(args, output_dir=args.output_dir)
    log_file = None
    try:
        ensure_parent_dir(train_log_path)
        log_file = open(train_log_path, "w", encoding="utf-8")
    except Exception as exc:
        print(
            f"[warn] Failed to create training log file '{train_log_path}': {exc}. "
            "Continuing without file logging.",
            file=sys.stderr,
        )

    if log_file is None:
        run_training(args)
        return

    with log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Training log file: {train_log_path}")
            run_training(args)


if __name__ == "__main__":
    main()
