import argparse
import csv
import json
import os
import random
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare MalbehavD-V1 CSV into train/valid/test txt files for Diffusion-LM."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to MalBehavD-V1 CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../datasets/malbehav",
        help="Output directory for prepared files.",
    )
    parser.add_argument(
        "--skip_columns",
        type=str,
        default="sha256,labels",
        help="Comma-separated list of columns to exclude from API sequence.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="labels",
        help="Label column name in CSV.",
    )
    parser.add_argument(
        "--include_label_token",
        type=str,
        default="yes",
        choices=["yes", "no"],
        help="If yes, prepend LABEL_<value> token to each sequence.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.05,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Random seed for split.",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=2,
        help="Minimum token length to keep a sequence.",
    )
    parser.add_argument(
        "--use_all_for_train",
        type=str,
        default="no",
        choices=["yes", "no"],
        help="If yes, put all sequences into train and mirror a tiny subset into valid/test.",
    )
    return parser.parse_args()


def build_sequences(input_csv, skip_columns, label_column, include_label_token, min_tokens):
    rows = []
    label_counter = Counter()
    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames if reader.fieldnames is not None else []
        for row in reader:
            tokens = []
            label_value = str(row.get(label_column, "")).strip()
            if include_label_token == "yes" and label_value != "":
                tokens.append(f"LABEL_{label_value}")
            for col in fieldnames:
                if col in skip_columns:
                    continue
                val = str(row.get(col, "")).strip()
                if val == "" or val.lower() == "nan":
                    continue
                tokens.append(val)
            if len(tokens) >= min_tokens:
                rows.append(tokens)
                if label_value != "":
                    label_counter[label_value] += 1
    return rows, label_counter


def write_split(path, sequences):
    with open(path, "w", encoding="utf-8") as f:
        for tokens in sequences:
            f.write(" ".join(tokens) + "\n")


def main():
    args = parse_args()
    skip_columns = {x.strip() for x in args.skip_columns.split(",") if x.strip() != ""}
    sequences, label_counter = build_sequences(
        input_csv=args.input_csv,
        skip_columns=skip_columns,
        label_column=args.label_column,
        include_label_token=args.include_label_token,
        min_tokens=args.min_tokens,
    )

    if len(sequences) == 0:
        raise ValueError("No usable sequences found. Check input CSV and options.")

    random.seed(args.seed)
    random.shuffle(sequences)

    n_total = len(sequences)
    if args.use_all_for_train == "yes":
        train_data = sequences
        # Keep non-empty eval splits to avoid empty-dataloader issues in training loop.
        n_valid = max(1, int(n_total * 0.01))
        n_test = max(1, int(n_total * 0.01))
        valid_data = sequences[:n_valid]
        test_data = sequences[n_valid:n_valid + n_test]
    else:
        if args.train_ratio <= 0 or args.valid_ratio < 0:
            raise ValueError("train_ratio must be > 0 and valid_ratio must be >= 0")
        if args.train_ratio + args.valid_ratio >= 1.0:
            raise ValueError("train_ratio + valid_ratio must be < 1.0")
        n_train = int(n_total * args.train_ratio)
        n_valid = int(n_total * args.valid_ratio)
        train_data = sequences[:n_train]
        valid_data = sequences[n_train:n_train + n_valid]
        test_data = sequences[n_train + n_valid:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "malbehav_train.txt")
    valid_path = os.path.join(args.output_dir, "malbehav_valid.txt")
    test_path = os.path.join(args.output_dir, "malbehav_test.txt")
    meta_path = os.path.join(args.output_dir, "malbehav_stats.json")

    write_split(train_path, train_data)
    write_split(valid_path, valid_data)
    write_split(test_path, test_data)

    all_tokens = Counter()
    for seq in sequences:
        all_tokens.update(seq)

    stats = {
        "total_sequences": n_total,
        "train_sequences": len(train_data),
        "valid_sequences": len(valid_data),
        "test_sequences": len(test_data),
        "unique_tokens": len(all_tokens),
        "estimated_vocab_size_threshold_gt10": 4 + sum(1 for _, c in all_tokens.items() if c > 10),
        "label_counts": dict(label_counter),
        "paths": {
            "train": train_path,
            "valid": valid_path,
            "test": test_path,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
