import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GINE on MalbehavD-V1-like dataset")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="auto",
        choices=["auto", "single_seq", "wide_api"],
        help="Dataset adapter: auto-detect, single sequence column, or wide API columns",
    )
    parser.add_argument("--label-col", type=str, default=None, help="Label column name")
    parser.add_argument(
        "--seq-col",
        type=str,
        default=None,
        help="Sequence column name (used by single_seq adapter)",
    )
    parser.add_argument(
        "--api-col-regex",
        type=str,
        default=r"^\d+$",
        help="Regex for API columns in wide_api adapter",
    )
    parser.add_argument(
        "--include-unnamed-api-cols",
        action="store_true",
        help="Include non-empty trailing Unnamed:* columns in wide_api adapter",
    )
    parser.add_argument("--id-col", type=str, default=None, help="Optional id/sample name column")
    parser.add_argument(
        "--delimiter-regex",
        type=str,
        default=r"[,\s;|>]+",
        help=r"Regex used to split API sequence string",
    )
    parser.add_argument(
        "--min-token-freq",
        type=int,
        default=1,
        help="Minimum API frequency to keep in vocabulary",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=400,
        help="Truncate sequences longer than this value",
    )
    parser.add_argument("--undirected", action="store_true", help="Add reverse edges")
    parser.add_argument("--add-skip-edges", action="store_true", help="Add skip edges i -> i+2")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation fraction from train+val")
    parser.add_argument(
        "--no-internal-split",
        action="store_true",
        help="Train on all provided rows without internal train/val/test split",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Base directory for outputs (used for relative --save-path, --vocab-path, --results-path)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Model checkpoint path. If omitted, auto-generated under <output-dir>/models",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="vocab/api_vocab.json",
        help="Vocabulary JSON path (relative paths are resolved inside --output-dir)",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Run summary text path. If omitted, auto-generated under <output-dir>/results",
    )
    return parser.parse_args()
