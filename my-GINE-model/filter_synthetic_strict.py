#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from dataset_adapters import load_and_prepare_with_adapter


SPECIAL_TOKENS = {"START", "END", "PAD"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict filtering for synthetic API sequences: "
            "length filter, exact dedup, near-dup vs real, near-dup inside synthetic, optional class balance."
        )
    )
    parser.add_argument("--real-csv", required=True, help="Path to real/reference CSV")
    parser.add_argument("--synthetic-csv", required=True, help="Path to synthetic CSV")
    parser.add_argument("--output-csv", required=True, help="Path to filtered output CSV")
    parser.add_argument("--report-json", default=None, help="Optional path to filtering report JSON")

    parser.add_argument("--label-col", default="labels", help="Label column name")
    parser.add_argument("--seq-col", default=None, help="Sequence column for single_seq format")
    parser.add_argument(
        "--dataset-format",
        default="auto",
        choices=["auto", "single_seq", "wide_api"],
        help="Dataset adapter mode",
    )
    parser.add_argument("--api-col-regex", default=r"^\d+$", help="Regex for wide API columns")
    parser.add_argument(
        "--include-unnamed-api-cols",
        action="store_true",
        help="Include non-empty trailing Unnamed:* columns in wide format",
    )
    parser.add_argument("--delimiter-regex", default=r"[,\s;|>]+", help="Tokenizer regex")

    parser.add_argument("--length-q-low", type=float, default=0.05, help="Low quantile for length filter from real")
    parser.add_argument("--length-q-high", type=float, default=0.95, help="High quantile for length filter from real")
    parser.add_argument("--min-length", type=int, default=2, help="Absolute min length after cleanup")
    parser.add_argument("--max-length", type=int, default=100000, help="Absolute max length after cleanup")

    parser.add_argument(
        "--drop-special-tokens",
        default="yes",
        choices=["yes", "no"],
        help="Remove START/END/PAD tokens before filtering",
    )
    parser.add_argument(
        "--remove-exact-in-real",
        default="yes",
        choices=["yes", "no"],
        help="Drop synthetic rows that exactly match any real sequence",
    )
    parser.add_argument(
        "--near-dup-threshold-real",
        type=float,
        default=0.95,
        help="Drop synthetic if cosine similarity to nearest real is >= threshold",
    )
    parser.add_argument(
        "--near-dup-threshold-self",
        type=float,
        default=0.95,
        help="Cluster near duplicates inside synthetic with this cosine threshold and keep one per cluster",
    )
    parser.add_argument(
        "--self-neighbors",
        type=int,
        default=10,
        help="Neighbors for synthetic-vs-synthetic near-dup graph",
    )

    parser.add_argument(
        "--balance-labels",
        default="yes",
        choices=["yes", "no"],
        help="Balance labels 0/1 after filtering",
    )
    parser.add_argument(
        "--target-per-label",
        type=int,
        default=0,
        help="If >0, cap each label to this size (after balancing logic)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balanced sampling")
    return parser.parse_args()


def normalize_label(value) -> int:
    if pd.isna(value):
        raise ValueError("empty label")
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
    raise ValueError(f"unsupported label value: {value}")


def split_tokens(seq: str, delimiter_regex: str) -> List[str]:
    if pd.isna(seq):
        return []
    s = str(seq).strip()
    if not s:
        return []
    parts = re.split(delimiter_regex, s)
    return [t.strip() for t in parts if t.strip()]


def preprocess_df(df: pd.DataFrame, label_col: str, seq_col: str, delimiter_regex: str, drop_special_tokens: bool) -> pd.DataFrame:
    rows = []
    for raw_label, raw_seq in zip(df[label_col].tolist(), df[seq_col].tolist()):
        try:
            label = normalize_label(raw_label)
        except Exception:
            continue
        tokens = split_tokens(raw_seq, delimiter_regex=delimiter_regex)
        if drop_special_tokens:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        if len(tokens) == 0:
            continue
        rows.append(
            {
                "label": int(label),
                "sequence": " ".join(tokens),
                "length": len(tokens),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["label", "sequence", "length"])
    return pd.DataFrame(rows)


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.real_csv):
        raise FileNotFoundError(f"real CSV not found: {args.real_csv}")
    if not os.path.exists(args.synthetic_csv):
        raise FileNotFoundError(f"synthetic CSV not found: {args.synthetic_csv}")

    report_path = args.report_json
    if report_path is None:
        base, _ = os.path.splitext(args.output_csv)
        report_path = base + ".filter_report.json"

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)

    real_prepared, real_label_col, real_seq_col, real_info = load_and_prepare_with_adapter(
        csv_path=args.real_csv,
        label_col=args.label_col,
        seq_col=args.seq_col,
        dataset_format=args.dataset_format,
        api_col_regex=args.api_col_regex,
        include_unnamed_api_cols=args.include_unnamed_api_cols,
    )
    syn_prepared, syn_label_col, syn_seq_col, syn_info = load_and_prepare_with_adapter(
        csv_path=args.synthetic_csv,
        label_col=args.label_col,
        seq_col=args.seq_col,
        dataset_format=args.dataset_format,
        api_col_regex=args.api_col_regex,
        include_unnamed_api_cols=args.include_unnamed_api_cols,
    )

    real_df = preprocess_df(
        real_prepared,
        label_col=real_label_col,
        seq_col=real_seq_col,
        delimiter_regex=args.delimiter_regex,
        drop_special_tokens=(args.drop_special_tokens == "yes"),
    )
    syn_df = preprocess_df(
        syn_prepared,
        label_col=syn_label_col,
        seq_col=syn_seq_col,
        delimiter_regex=args.delimiter_regex,
        drop_special_tokens=(args.drop_special_tokens == "yes"),
    )

    if len(real_df) == 0:
        raise ValueError("No valid rows in real dataset after preprocessing")
    if len(syn_df) == 0:
        raise ValueError("No valid rows in synthetic dataset after preprocessing")

    real_lengths = real_df["length"].to_numpy(dtype=float)
    q_low_len = int(np.floor(np.quantile(real_lengths, args.length_q_low)))
    q_high_len = int(np.ceil(np.quantile(real_lengths, args.length_q_high)))
    low_len = max(args.min_length, q_low_len)
    high_len = min(args.max_length, q_high_len)
    if low_len > high_len:
        low_len, high_len = args.min_length, args.max_length

    stats: Dict[str, object] = {
        "created_at": datetime.now().isoformat(),
        "real_csv": os.path.abspath(args.real_csv),
        "synthetic_csv": os.path.abspath(args.synthetic_csv),
        "output_csv": os.path.abspath(args.output_csv),
        "report_json": os.path.abspath(report_path),
        "adapter_real": real_info,
        "adapter_synthetic": syn_info,
        "params": {
            "length_q_low": args.length_q_low,
            "length_q_high": args.length_q_high,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "near_dup_threshold_real": args.near_dup_threshold_real,
            "near_dup_threshold_self": args.near_dup_threshold_self,
            "self_neighbors": args.self_neighbors,
            "drop_special_tokens": args.drop_special_tokens,
            "remove_exact_in_real": args.remove_exact_in_real,
            "balance_labels": args.balance_labels,
            "target_per_label": args.target_per_label,
            "seed": args.seed,
        },
        "counts": {},
    }

    stats["counts"]["real_preprocessed"] = int(len(real_df))
    stats["counts"]["synthetic_preprocessed"] = int(len(syn_df))

    # 1) Length filter.
    syn_df = syn_df[(syn_df["length"] >= low_len) & (syn_df["length"] <= high_len)].reset_index(drop=True)
    stats["counts"]["after_length_filter"] = int(len(syn_df))
    stats["length_filter"] = {
        "real_q_low_len": int(q_low_len),
        "real_q_high_len": int(q_high_len),
        "applied_low_len": int(low_len),
        "applied_high_len": int(high_len),
    }
    if len(syn_df) == 0:
        raise ValueError("No rows left after length filter")

    # 2) Exact duplicate filters.
    if args.remove_exact_in_real == "yes":
        real_set = set(real_df["sequence"].tolist())
        syn_df = syn_df[~syn_df["sequence"].isin(real_set)].reset_index(drop=True)
    stats["counts"]["after_exact_vs_real"] = int(len(syn_df))
    if len(syn_df) == 0:
        raise ValueError("No rows left after exact-vs-real filter")

    syn_df = syn_df.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)
    stats["counts"]["after_exact_internal"] = int(len(syn_df))
    if len(syn_df) == 0:
        raise ValueError("No rows left after exact internal dedup")

    # 3) Near duplicate vs real (cosine over TF-IDF).
    vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"[^ ]+")
    combined = real_df["sequence"].tolist() + syn_df["sequence"].tolist()
    x_all = vectorizer.fit_transform(combined)
    x_real = x_all[: len(real_df)]
    x_syn = x_all[len(real_df) :]

    nn_real = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn_real.fit(x_real)
    dist_real, _ = nn_real.kneighbors(x_syn, return_distance=True)
    sim_real = 1.0 - dist_real[:, 0]
    keep_mask_real = sim_real < args.near_dup_threshold_real
    syn_df = syn_df[keep_mask_real].reset_index(drop=True)
    stats["counts"]["after_near_vs_real"] = int(len(syn_df))
    stats["near_vs_real"] = {
        "mean_sim": float(np.mean(sim_real)) if len(sim_real) else 0.0,
        "p95_sim": float(np.quantile(sim_real, 0.95)) if len(sim_real) else 0.0,
        "max_sim": float(np.max(sim_real)) if len(sim_real) else 0.0,
        "dropped_count": int(np.sum(~keep_mask_real)),
    }
    if len(syn_df) == 0:
        raise ValueError("No rows left after near-dup vs real filter")

    # 4) Near duplicate inside synthetic (connected components on NN graph).
    if len(syn_df) >= 2:
        x_syn = vectorizer.transform(syn_df["sequence"].tolist())
        k = max(2, min(args.self_neighbors, len(syn_df)))
        nn_syn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        nn_syn.fit(x_syn)
        dist_syn, idx_syn = nn_syn.kneighbors(x_syn, return_distance=True)

        dsu = DSU(len(syn_df))
        for i in range(len(syn_df)):
            for t in range(1, k):
                j = int(idx_syn[i, t])
                if j <= i:
                    continue
                sim = 1.0 - float(dist_syn[i, t])
                if sim >= args.near_dup_threshold_self:
                    dsu.union(i, j)

        comps: Dict[int, List[int]] = {}
        for i in range(len(syn_df)):
            r = dsu.find(i)
            comps.setdefault(r, []).append(i)

        keep_idx = [min(v) for v in comps.values()]
        keep_idx = sorted(keep_idx)
        syn_df = syn_df.iloc[keep_idx].reset_index(drop=True)
        stats["near_internal"] = {
            "components": int(len(comps)),
            "kept_after_components": int(len(keep_idx)),
            "dropped_by_components": int(sum(len(v) - 1 for v in comps.values())),
        }
    else:
        stats["near_internal"] = {"components": int(len(syn_df)), "kept_after_components": int(len(syn_df)), "dropped_by_components": 0}

    stats["counts"]["after_near_internal"] = int(len(syn_df))
    if len(syn_df) == 0:
        raise ValueError("No rows left after internal near-dup filter")

    # 5) Optional class balance.
    if args.balance_labels == "yes":
        rs = np.random.RandomState(args.seed)
        g0 = syn_df[syn_df["label"] == 0].copy()
        g1 = syn_df[syn_df["label"] == 1].copy()
        n0, n1 = len(g0), len(g1)
        if n0 == 0 or n1 == 0:
            raise ValueError(f"Cannot balance labels: class counts are label0={n0}, label1={n1}")
        target = min(n0, n1)
        if args.target_per_label > 0:
            target = min(target, args.target_per_label)

        g0 = g0.iloc[rs.permutation(n0)[:target]]
        g1 = g1.iloc[rs.permutation(n1)[:target]]
        syn_df = pd.concat([g0, g1], axis=0).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        stats["balance"] = {"enabled": True, "target_per_label": int(target), "label0_before": int(n0), "label1_before": int(n1)}
    else:
        stats["balance"] = {"enabled": False}

    stats["counts"]["final"] = int(len(syn_df))
    stats["label_counts_final"] = {
        "label0": int((syn_df["label"] == 0).sum()),
        "label1": int((syn_df["label"] == 1).sum()),
    }

    out_df = pd.DataFrame({"labels": syn_df["label"].astype(int), "api_sequence": syn_df["sequence"].astype(str)})
    out_df.to_csv(args.output_csv, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] Filtered rows: {len(out_df)}")
    print(f"[OK] Saved CSV: {args.output_csv}")
    print(f"[OK] Saved report: {report_path}")
    print(f"[INFO] Final label counts: {stats['label_counts_final']}")


if __name__ == "__main__":
    main()

