#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert generated RelGAN text samples back to MalbehavD-V1 CSV layout.
"""

import argparse
import csv
import hashlib
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_txt", required=True, help="Generated samples text file")
    parser.add_argument("--template_csv", required=True, help="Original MalbehavD-V1 CSV to copy header schema")
    parser.add_argument("--output_csv", required=True, help="Output CSV file")
    parser.add_argument("--label", type=int, required=True, choices=[0, 1], help="Label for all generated rows")
    return parser.parse_args()


def read_header(template_csv):
    with open(template_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    if len(header) < 3 or header[0] != "sha256" or header[1] != "labels":
        raise ValueError("Unexpected template header. Expected: sha256,labels,0,1,...")
    return header


def main():
    args = parse_args()
    header = read_header(args.template_csv)
    api_cols = header[2:]
    max_calls = len(api_cols)

    rows = []
    with open(args.samples_txt, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            tokens = [t for t in line.strip().split() if t and t not in ("BOS", "EOS")]
            tokens = tokens[:max_calls]

            digest_input = f"{args.label}|{i}|{' '.join(tokens)}".encode("utf-8")
            sample_sha = hashlib.sha256(digest_input).hexdigest()
            padded = tokens + [""] * (max_calls - len(tokens))
            rows.append([sample_sha, str(args.label)] + padded)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[ok] exported {len(rows)} rows -> {os.path.abspath(args.output_csv)}")


if __name__ == "__main__":
    main()
