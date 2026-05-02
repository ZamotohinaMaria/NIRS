import argparse
import csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Diffusion-LM generated txt samples into Malbehav-like CSV."
    )
    parser.add_argument("--input_txt", type=str, required=True, help="Path to generated .txt file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path.")
    parser.add_argument(
        "--drop_special_tokens",
        type=str,
        default="yes",
        choices=["yes", "no"],
        help="Drop START/END/PAD/UNK tokens if present.",
    )
    parser.add_argument(
        "--with_sha256_column",
        type=str,
        default="yes",
        choices=["yes", "no"],
        help="Include sha256 column as synthetic ids.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    special = {"START", "END", "PAD", "UNK"}

    rows = []
    max_len = 0
    with open(args.input_txt, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            tokens = [x for x in line.strip().split() if x != ""]
            if args.drop_special_tokens == "yes":
                tokens = [x for x in tokens if x not in special]

            label = ""
            if len(tokens) > 0 and tokens[0].startswith("LABEL_"):
                label = tokens[0].replace("LABEL_", "", 1)
                tokens = tokens[1:]

            max_len = max(max_len, len(tokens))
            rows.append((i, label, tokens))

    header = []
    if args.with_sha256_column == "yes":
        header.append("sha256")
    header.append("labels")
    header.extend([str(i) for i in range(max_len)])

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, label, tokens in rows:
            row = []
            if args.with_sha256_column == "yes":
                row.append(f"generated_{idx:08d}")
            row.append(label)
            if len(tokens) < max_len:
                tokens = tokens + [""] * (max_len - len(tokens))
            row.extend(tokens)
            writer.writerow(row)

    print(f"saved {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
