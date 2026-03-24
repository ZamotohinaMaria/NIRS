import math
import os
from typing import Optional, Set, Tuple

import pandas as pd


def guess_label_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "label",
        "class",
        "target",
        "y",
        "malware",
        "is_malware",
        "category",
        "type",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]

    for col in df.columns:
        vals = df[col].dropna().astype(str).str.lower().unique().tolist()
        if 1 < len(vals) <= 6:
            joined = set(vals)
            if joined & {"benign", "malware", "0", "1", "true", "false"}:
                return col
    return None


def guess_sequence_column(df: pd.DataFrame, exclude_cols: Optional[Set[str]] = None) -> Optional[str]:
    exclude_cols = exclude_cols or set()
    candidates = [
        "api_sequence",
        "sequence",
        "api_calls",
        "api_call_sequence",
        "calls",
        "trace",
        "behavior",
        "behaviour",
        "log",
        "api",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map and lower_map[c] not in exclude_cols:
            return lower_map[c]

    text_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        lower_col = col.lower()
        # Skip common id/hash fields to avoid selecting them as sequence columns.
        if lower_col in {"sha256", "sha1", "md5", "hash", "sample_id", "id"}:
            continue
        if df[col].dtype == object:
            avg_len = df[col].dropna().astype(str).map(len).mean()
            if not math.isnan(avg_len):
                text_cols.append((col, avg_len))
    if text_cols:
        text_cols.sort(key=lambda x: x[1], reverse=True)
        return text_cols[0][0]
    return None


def load_and_prepare_dataframe(
    csv_path: str,
    label_col: Optional[str],
    seq_col: Optional[str],
) -> Tuple[pd.DataFrame, str, str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if label_col is None:
        label_col = guess_label_column(df)
    if seq_col is None:
        seq_col = guess_sequence_column(df, exclude_cols={label_col} if label_col is not None else None)

    if label_col is None:
        raise ValueError("Could not detect label column automatically. Pass --label-col")
    if seq_col is None:
        raise ValueError("Could not detect sequence column automatically. Pass --seq-col")

    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")
    if seq_col not in df.columns:
        raise ValueError(f"Sequence column not found: {seq_col}")

    df = df[[label_col, seq_col]].copy()
    df = df.dropna(subset=[label_col, seq_col]).reset_index(drop=True)
    return df, label_col, seq_col
