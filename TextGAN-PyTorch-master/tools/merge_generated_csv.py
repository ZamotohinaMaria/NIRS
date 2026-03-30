#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge generated goodware and malware CSV files into one MalbehavD-V1-like CSV.
"""

import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goodware_csv", required=True)
    parser.add_argument("--malware_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def main():
    args = parse_args()
    h1, r1 = read_csv(args.goodware_csv)
    h2, r2 = read_csv(args.malware_csv)
    if h1 != h2:
        raise ValueError("Headers do not match between CSV files.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(h1)
        writer.writerows(r1)
        writer.writerows(r2)

    print(f"[ok] merged {len(r1)} + {len(r2)} rows -> {os.path.abspath(args.output_csv)}")


if __name__ == "__main__":
    main()
