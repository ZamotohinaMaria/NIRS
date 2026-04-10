import argparse
import csv
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MalBehavD CSV to Diffusion-LM ROCStory jsonl format."
    )
    parser.add_argument("--input_csv", type=str, required=True, help="Path to MalBehavD CSV.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="diffusion_lm/ROCstory",
        help="Output dir for roc_train.json and roc_valid.json.",
    )
    parser.add_argument(
        "--valid_size",
        type=int,
        default=500,
        help="Number of samples for roc_valid.json.",
    )
    parser.add_argument("--seed", type=int, default=101, help="Shuffle seed.")
    parser.add_argument(
        "--label_filter",
        type=str,
        default="all",
        choices=["all", "malware", "benign"],
        help="Use all labels or only one class.",
    )
    return parser.parse_args()


def keep_row(label, label_filter):
    if label_filter == "all":
        return True
    if label_filter == "malware":
        return str(label).strip() == "1"
    if label_filter == "benign":
        return str(label).strip() == "0"
    return True


def extract_sequences(input_csv, label_filter):
    sequences = []
    with open(input_csv, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not keep_row(row.get("labels", ""), label_filter):
                continue
            tokens = []
            for key, value in row.items():
                if key in ("sha256", "labels"):
                    continue
                token = (value or "").strip()
                if token:
                    tokens.append(token)
            if tokens:
                sequences.append(tokens)
    return sequences


def write_jsonl(path, sequences):
    with open(path, "w", encoding="utf-8") as f:
        for tokens in sequences:
            print(json.dumps([" ".join(tokens)]), file=f)


def main():
    args = parse_args()
    sequences = extract_sequences(args.input_csv, args.label_filter)
    if len(sequences) == 0:
        raise ValueError("No sequences extracted. Check input CSV and --label_filter.")

    random.Random(args.seed).shuffle(sequences)

    valid_size = max(1, min(args.valid_size, len(sequences) - 1))
    valid_sequences = sequences[:valid_size]
    train_sequences = sequences[valid_size:]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "roc_train.json")
    valid_path = os.path.join(args.out_dir, "roc_valid.json")
    write_jsonl(train_path, train_sequences)
    write_jsonl(valid_path, valid_sequences)

    uniq_tokens = set()
    for seq in sequences:
        uniq_tokens.update(seq)

    print(f"Total sequences: {len(sequences)}")
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Valid sequences: {len(valid_sequences)}")
    print(f"Unique API tokens: {len(uniq_tokens)}")
    print(f"Vocab with START/END/UNK/PAD: {len(uniq_tokens) + 4}")
    print(f"Wrote: {train_path}")
    print(f"Wrote: {valid_path}")


if __name__ == "__main__":
    main()
