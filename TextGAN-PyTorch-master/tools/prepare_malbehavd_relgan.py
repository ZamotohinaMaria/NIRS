#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare MalbehavD-V1 CSV into RelGAN text files (one model per class).
"""

import argparse
import csv
import json
import os
import random
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to MalBehavD-V1-dataset.csv")
    parser.add_argument("--output_root", default="dataset", help="TextGAN dataset root directory")
    parser.add_argument("--dataset_prefix", default="MalbehavD-V1", help="Prefix for output dataset names")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction for test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min_seq_len", type=int, default=2, help="Drop samples shorter than this")
    return parser.parse_args()


def read_csv_sequences(csv_path: str, min_seq_len: int):
    by_label: Dict[int, List[List[str]]] = {0: [], 1: []}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        if len(headers) < 3:
            raise ValueError("CSV must have at least 3 columns: sha256, labels, api_calls...")
        api_cols = headers[2:]

        for row in reader:
            if len(row) < 3:
                continue
            label = int(row[1])
            api_values = row[2:]
            if len(api_values) < len(api_cols):
                api_values = api_values + [""] * (len(api_cols) - len(api_values))
            seq = [v.strip() for v in api_values if v.strip()]
            if len(seq) >= min_seq_len:
                by_label[label].append(seq)

    return by_label, api_cols


def split_train_test(samples: List[List[str]], test_ratio: float, seed: int):
    rng = random.Random(seed)
    items = samples[:]
    rng.shuffle(items)

    test_n = int(round(len(items) * test_ratio))
    test_n = max(1, min(test_n, len(items) - 1))
    test = items[:test_n]
    train = items[test_n:]
    return train, test


def write_txt(path: str, samples: List[List[str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for seq in samples:
            f.write(" ".join(seq))
            f.write("\n")


def main():
    args = parse_args()
    by_label, api_cols = read_csv_sequences(args.input_csv, args.min_seq_len)

    label_to_name = {0: "goodware", 1: "malware"}
    test_dir = os.path.join(args.output_root, "testdata")
    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    summary = {
        "input_csv": os.path.abspath(args.input_csv),
        "dataset_prefix": args.dataset_prefix,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "min_seq_len": args.min_seq_len,
        "api_columns": api_cols,
        "classes": {},
    }

    for label, class_name in label_to_name.items():
        train, test = split_train_test(by_label[label], args.test_ratio, args.seed + label)
        dataset_name = f"{args.dataset_prefix}_{class_name}"

        train_path = os.path.join(args.output_root, f"{dataset_name}.txt")
        test_path = os.path.join(test_dir, f"{dataset_name}_test.txt")

        write_txt(train_path, train)
        write_txt(test_path, test)

        summary["classes"][class_name] = {
            "label": label,
            "dataset_name": dataset_name,
            "train_size": len(train),
            "test_size": len(test),
            "train_path": os.path.abspath(train_path),
            "test_path": os.path.abspath(test_path),
        }
        print(f"[ok] {class_name}: train={len(train)} test={len(test)}")

    meta_path = os.path.join(args.output_root, f"{args.dataset_prefix}_relgan_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[ok] metadata: {os.path.abspath(meta_path)}")


if __name__ == "__main__":
    main()
