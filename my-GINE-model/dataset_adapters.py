import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_io import guess_label_column, guess_sequence_column


def _sort_numeric_columns(columns: List[str]) -> List[str]:
    return sorted(columns, key=lambda c: int(c))


def _sort_unnamed_columns(columns: List[str]) -> List[str]:
    def _key(col: str) -> int:
        match = re.search(r"(\d+)$", col)
        if match:
            return int(match.group(1))
        return 10**9

    return sorted(columns, key=_key)


def _row_to_sequence(row: pd.Series) -> str:
    tokens: List[str] = []
    for value in row.tolist():
        if pd.isna(value):
            continue
        token = str(value).strip()
        if token and token.lower() != "nan":
            tokens.append(token)
    return " ".join(tokens)


def _collect_wide_api_columns(
    df: pd.DataFrame,
    api_col_regex: str,
    include_unnamed_api_cols: bool,
) -> List[str]:
    pattern = re.compile(api_col_regex)
    api_cols = [c for c in df.columns if pattern.match(str(c))]
    api_cols = _sort_numeric_columns(api_cols)

    if include_unnamed_api_cols:
        unnamed_cols = [
            c for c in df.columns if str(c).startswith("Unnamed:") and df[c].notna().any()
        ]
        api_cols.extend(_sort_unnamed_columns(unnamed_cols))

    return api_cols


def _contiguous_numeric_prefix_len(columns: List[str]) -> int:
    numeric = sorted([int(c) for c in columns if str(c).isdigit()])
    count = 0
    for value in numeric:
        if value == count:
            count += 1
        else:
            break
    return count


def _auto_detect_format(df: pd.DataFrame, api_col_regex: str) -> str:
    pattern = re.compile(api_col_regex)
    numeric_cols = [c for c in df.columns if pattern.match(str(c))]
    prefix_len = _contiguous_numeric_prefix_len(numeric_cols)

    if prefix_len >= 10:
        return "wide_api"
    return "single_seq"


def _prepare_single_seq(
    df: pd.DataFrame,
    label_col: Optional[str],
    seq_col: Optional[str],
) -> Tuple[pd.DataFrame, str, str, Dict[str, str]]:
    if label_col is None:
        label_col = guess_label_column(df)
    if label_col is None:
        raise ValueError("Could not detect label column automatically. Pass --label-col")
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    if seq_col is None:
        seq_col = guess_sequence_column(df, exclude_cols={label_col})
    if seq_col is None:
        raise ValueError("Could not detect sequence column automatically. Pass --seq-col")
    if seq_col not in df.columns:
        raise ValueError(f"Sequence column not found: {seq_col}")

    out_df = df[[label_col, seq_col]].copy()
    out_df = out_df.dropna(subset=[label_col, seq_col]).reset_index(drop=True)
    info = {"format": "single_seq", "source_sequence_columns": seq_col}
    return out_df, label_col, seq_col, info


def _prepare_wide_api(
    df: pd.DataFrame,
    label_col: Optional[str],
    api_col_regex: str,
    include_unnamed_api_cols: bool,
) -> Tuple[pd.DataFrame, str, str, Dict[str, str]]:
    if label_col is None:
        label_col = guess_label_column(df)
    if label_col is None:
        raise ValueError("Could not detect label column automatically. Pass --label-col")
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    api_cols = _collect_wide_api_columns(
        df=df,
        api_col_regex=api_col_regex,
        include_unnamed_api_cols=include_unnamed_api_cols,
    )
    if not api_cols:
        raise ValueError(
            f"No API columns found with regex '{api_col_regex}'. "
            "Adjust --api-col-regex or use --dataset-format single_seq."
        )

    out_df = pd.DataFrame()
    out_df[label_col] = df[label_col]
    out_df["api_sequence"] = df[api_cols].apply(_row_to_sequence, axis=1)
    out_df = out_df.dropna(subset=[label_col]).reset_index(drop=True)
    out_df = out_df[out_df["api_sequence"].astype(str).str.len() > 0].reset_index(drop=True)

    info = {
        "format": "wide_api",
        "source_sequence_columns": f"{len(api_cols)} columns",
    }
    return out_df, label_col, "api_sequence", info


def load_and_prepare_with_adapter(
    csv_path: str,
    label_col: Optional[str],
    seq_col: Optional[str],
    dataset_format: str,
    api_col_regex: str,
    include_unnamed_api_cols: bool,
) -> Tuple[pd.DataFrame, str, str, Dict[str, str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    if dataset_format == "auto":
        resolved_format = _auto_detect_format(raw_df, api_col_regex=api_col_regex)
    else:
        resolved_format = dataset_format

    if resolved_format == "single_seq":
        prepared_df, out_label_col, out_seq_col, info = _prepare_single_seq(
            df=raw_df,
            label_col=label_col,
            seq_col=seq_col,
        )
    elif resolved_format == "wide_api":
        prepared_df, out_label_col, out_seq_col, info = _prepare_wide_api(
            df=raw_df,
            label_col=label_col,
            api_col_regex=api_col_regex,
            include_unnamed_api_cols=include_unnamed_api_cols,
        )
    else:
        raise ValueError(
            f"Unsupported dataset format: {resolved_format}. "
            "Use one of: auto, single_seq, wide_api."
        )

    info["resolved_format"] = resolved_format
    info["raw_rows"] = str(len(raw_df))
    info["prepared_rows"] = str(len(prepared_df))
    return prepared_df, out_label_col, out_seq_col, info
