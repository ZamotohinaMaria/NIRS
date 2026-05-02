#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive generation-quality checks for API-sequence datasets.

Checks implemented:
1) Duplicate and leakage checks (exact + near duplicates)
2) Diversity and coverage checks
3) Statistical closeness checks (lengths, token/ngram JS-divergence, MMD)
4) Label-consistency checks with an independent classifier
5) TSTR/TRTS and augmentation utility protocol
6) Stability across multiple random seeds
7) OOD detector check (Isolation Forest)
8) Privacy/memorization risk proxy checks
"""

import argparse
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict quality checks for generated API-sequence datasets.")
    parser.add_argument("--real-csv", required=True, help="Path to real/reference CSV")
    parser.add_argument("--synthetic-csv", required=True, help="Path to generated/synthetic CSV")
    parser.add_argument("--label-col", default="labels", help="Label column name")
    parser.add_argument("--seq-col", default=None, help="Optional single sequence column name")
    parser.add_argument("--api-col-regex", default=r"^\d+$", help="Regex for wide API columns")
    parser.add_argument(
        "--include-unnamed-api-cols",
        action="store_true",
        help="Include non-empty trailing Unnamed:* columns for wide format",
    )
    parser.add_argument("--delimiter-regex", default=r"[,\s;|>]+", help="Tokenizer regex for sequence strings")
    parser.add_argument("--test-size", type=float, default=0.2, help="Real holdout ratio for protocols")
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated seeds for stability checks")
    parser.add_argument("--max-features", type=int, default=12000, help="TF-IDF max_features")
    parser.add_argument(
        "--near-dup-threshold",
        type=float,
        default=0.98,
        help="Cosine-similarity threshold for near-duplicate alarm",
    )
    parser.add_argument("--mmd-sample-size", type=int, default=800, help="Per-domain sample size for MMD")
    parser.add_argument(
        "--ood-contamination",
        type=float,
        default=0.05,
        help="IsolationForest contamination parameter",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "runs", "generation_quality"),
        help="Output directory",
    )
    parser.add_argument("--thr-exact-dup-pass", type=float, default=0.01)
    parser.add_argument("--thr-exact-dup-warn", type=float, default=0.05)
    parser.add_argument("--thr-near-dup-pass", type=float, default=0.10)
    parser.add_argument("--thr-near-dup-warn", type=float, default=0.25)
    parser.add_argument("--thr-js-uni-pass", type=float, default=0.20)
    parser.add_argument("--thr-js-uni-warn", type=float, default=0.35)
    parser.add_argument("--thr-tstr-f1-ratio-pass", type=float, default=0.90)
    parser.add_argument("--thr-tstr-f1-ratio-warn", type=float, default=0.75)
    parser.add_argument("--thr-aug-f1-drop-warn", type=float, default=-0.03)
    parser.add_argument("--thr-aug-f1-drop-fail", type=float, default=-0.07)
    parser.add_argument("--thr-ood-gap-pass", type=float, default=0.10)
    parser.add_argument("--thr-ood-gap-warn", type=float, default=0.20)
    parser.add_argument("--thr-mia-auc-pass", type=float, default=0.60)
    parser.add_argument("--thr-mia-auc-warn", type=float, default=0.70)
    parser.add_argument("--thr-fn-delta-warn", type=float, default=20.0)
    parser.add_argument("--thr-fn-delta-fail", type=float, default=60.0)
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


def collect_wide_api_cols(df: pd.DataFrame, api_col_regex: str, include_unnamed_api_cols: bool) -> List[str]:
    pattern = re.compile(api_col_regex)
    api_cols = [c for c in df.columns if pattern.match(str(c))]
    api_cols = sorted(api_cols, key=lambda c: int(c))
    if include_unnamed_api_cols:
        unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:") and df[c].notna().any()]
        unnamed_cols = sorted(
            unnamed_cols,
            key=lambda c: int(re.search(r"(\d+)$", c).group(1)) if re.search(r"(\d+)$", c) else 10**9,
        )
        api_cols.extend(unnamed_cols)
    return api_cols


def row_to_sequence(values: Sequence[object]) -> str:
    tokens: List[str] = []
    for v in values:
        if pd.isna(v):
            continue
        t = str(v).strip()
        if t and t.lower() != "nan":
            tokens.append(t)
    return " ".join(tokens)


def prepare_dataset(
    csv_path: str,
    label_col: str,
    seq_col: Optional[str],
    api_col_regex: str,
    include_unnamed_api_cols: bool,
    delimiter_regex: str,
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    raw = pd.read_csv(csv_path)
    if label_col not in raw.columns:
        raise ValueError(f"label column '{label_col}' not found in {csv_path}")

    if seq_col is not None:
        if seq_col not in raw.columns:
            raise ValueError(f"seq column '{seq_col}' not found in {csv_path}")
        seq_series = raw[seq_col].fillna("").astype(str).map(str.strip)
    else:
        api_cols = collect_wide_api_cols(raw, api_col_regex=api_col_regex, include_unnamed_api_cols=include_unnamed_api_cols)
        if not api_cols:
            raise ValueError("No API columns found and no --seq-col was provided")
        seq_series = raw[api_cols].apply(lambda r: row_to_sequence(r.tolist()), axis=1)

    out_rows = []
    for seq, lbl_raw in zip(seq_series.tolist(), raw[label_col].tolist()):
        tokens = split_tokens(seq, delimiter_regex=delimiter_regex)
        if not tokens:
            continue
        try:
            lbl = normalize_label(lbl_raw)
        except Exception:
            continue
        out_rows.append(
            {
                "sequence": " ".join(tokens),
                "tokens": tokens,
                "length": len(tokens),
                "label": int(lbl),
            }
        )
    if not out_rows:
        raise ValueError(f"No valid rows after preprocessing: {csv_path}")
    return pd.DataFrame(out_rows)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if y_prob is not None:
        out["roc_auc"] = safe_auc(y_true, y_prob)
    else:
        out["roc_auc"] = float("nan")
    return out


def entropy_from_counter(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = np.array([v / total for v in counter.values()], dtype=float)
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def js_divergence_from_counters(c1: Counter, c2: Counter) -> float:
    keys = list(set(c1.keys()) | set(c2.keys()))
    if not keys:
        return 0.0
    p = np.array([c1.get(k, 0.0) for k in keys], dtype=float)
    q = np.array([c2.get(k, 0.0) for k in keys], dtype=float)
    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(np.where(p > 0, p * np.log2((p + 1e-12) / (m + 1e-12)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log2((q + 1e-12) / (m + 1e-12)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def ngram_counter(seqs_tokens: Iterable[List[str]], n: int) -> Counter:
    cnt: Counter = Counter()
    for toks in seqs_tokens:
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            cnt[tuple(toks[i : i + n])] += 1
    return cnt


def ks_statistic_2samp(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    all_vals = np.sort(np.concatenate([x_sorted, y_sorted]))
    cdf_x = np.searchsorted(x_sorted, all_vals, side="right") / len(x_sorted)
    cdf_y = np.searchsorted(y_sorted, all_vals, side="right") / len(y_sorted)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def approx_wasserstein_1d(x: np.ndarray, y: np.ndarray, n_quantiles: int = 500) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    q = np.linspace(0.0, 1.0, n_quantiles)
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    return float(np.mean(np.abs(xq - yq)))


def summarize_lengths(lengths: np.ndarray) -> Dict[str, float]:
    return {
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
    }


def exact_and_near_duplicates(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    max_features: int,
    near_dup_threshold: float,
) -> Dict[str, object]:
    real_set = set(real_df["sequence"].tolist())
    synth_seq = synth_df["sequence"].tolist()
    exact_matches = np.array([s in real_set for s in synth_seq], dtype=bool)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    X_real = vectorizer.fit_transform(real_df["sequence"].tolist())
    X_synth = vectorizer.transform(synth_seq)
    nbrs = NearestNeighbors(n_neighbors=1, metric="cosine")
    nbrs.fit(X_real)
    dist, idx = nbrs.kneighbors(X_synth, return_distance=True)
    sim = 1.0 - dist[:, 0]

    return {
        "exact_duplicate_count": int(exact_matches.sum()),
        "exact_duplicate_ratio": float(exact_matches.mean()),
        "near_duplicate_threshold": near_dup_threshold,
        "near_duplicate_count": int((sim >= near_dup_threshold).sum()),
        "near_duplicate_ratio": float((sim >= near_dup_threshold).mean()),
        "nearest_similarity_mean": float(np.mean(sim)),
        "nearest_similarity_p95": float(np.percentile(sim, 95)),
        "nearest_similarity_max": float(np.max(sim)),
        "sample_nearest_real_indices": idx[:10, 0].tolist(),
    }


def diversity_and_coverage(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict[str, object]:
    real_tokens = [tok for seq in real_df["tokens"].tolist() for tok in seq]
    synth_tokens = [tok for seq in synth_df["tokens"].tolist() for tok in seq]

    real_cnt = Counter(real_tokens)
    synth_cnt = Counter(synth_tokens)

    real_uni = set(real_cnt.keys())
    synth_uni = set(synth_cnt.keys())
    shared_uni = real_uni & synth_uni

    real_uni_seq = set(real_df["sequence"].tolist())
    synth_uni_seq = set(synth_df["sequence"].tolist())

    return {
        "real": {
            "rows": int(len(real_df)),
            "unique_sequences": int(len(real_uni_seq)),
            "unique_sequence_ratio": float(len(real_uni_seq) / len(real_df)),
            "unique_tokens": int(len(real_uni)),
            "token_entropy_bits": entropy_from_counter(real_cnt),
            "length_stats": summarize_lengths(real_df["length"].to_numpy()),
        },
        "synthetic": {
            "rows": int(len(synth_df)),
            "unique_sequences": int(len(synth_uni_seq)),
            "unique_sequence_ratio": float(len(synth_uni_seq) / len(synth_df)),
            "unique_tokens": int(len(synth_uni)),
            "token_entropy_bits": entropy_from_counter(synth_cnt),
            "length_stats": summarize_lengths(synth_df["length"].to_numpy()),
        },
        "coverage": {
            "shared_unique_tokens": int(len(shared_uni)),
            "synthetic_token_coverage_in_real": float(len(shared_uni) / max(len(synth_uni), 1)),
            "synthetic_token_novelty_vs_real": float(len(synth_uni - real_uni) / max(len(synth_uni), 1)),
            "synthetic_sequence_novelty_vs_real": float(len(synth_uni_seq - real_uni_seq) / max(len(synth_uni_seq), 1)),
        },
    }


def distribution_closeness(real_df: pd.DataFrame, synth_df: pd.DataFrame, mmd_sample_size: int) -> Dict[str, object]:
    real_len = real_df["length"].to_numpy(dtype=float)
    synth_len = synth_df["length"].to_numpy(dtype=float)

    uni_real = Counter(tok for seq in real_df["tokens"].tolist() for tok in seq)
    uni_synth = Counter(tok for seq in synth_df["tokens"].tolist() for tok in seq)
    bi_real = ngram_counter(real_df["tokens"].tolist(), n=2)
    bi_synth = ngram_counter(synth_df["tokens"].tolist(), n=2)

    # Approximate MMD in TF-IDF embedding space
    n = min(mmd_sample_size, len(real_df), len(synth_df))
    real_sample = real_df.sample(n=n, random_state=42)["sequence"].tolist()
    synth_sample = synth_df.sample(n=n, random_state=42)["sequence"].tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=6000)
    X = vec.fit_transform(real_sample + synth_sample)
    X = normalize(X, norm="l2")
    xr = X[:n].toarray()
    xs = X[n : 2 * n].toarray()
    # median heuristic for gamma
    joined = np.vstack([xr[: min(n, 250)], xs[: min(n, 250)]])
    d2 = np.sum((joined[:, None, :] - joined[None, :, :]) ** 2, axis=2)
    median_sq_dist = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / max(median_sq_dist, 1e-9)
    k_xx = np.exp(-gamma * np.sum((xr[:, None, :] - xr[None, :, :]) ** 2, axis=2))
    k_yy = np.exp(-gamma * np.sum((xs[:, None, :] - xs[None, :, :]) ** 2, axis=2))
    k_xy = np.exp(-gamma * np.sum((xr[:, None, :] - xs[None, :, :]) ** 2, axis=2))
    mmd2 = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())

    return {
        "length_distribution": {
            "ks_statistic": ks_statistic_2samp(real_len, synth_len),
            "approx_wasserstein": approx_wasserstein_1d(real_len, synth_len),
        },
        "token_distribution": {
            "js_divergence_unigram": js_divergence_from_counters(uni_real, uni_synth),
            "js_divergence_bigram": js_divergence_from_counters(bi_real, bi_synth),
        },
        "embedding_space": {
            "mmd_rbf_squared": mmd2,
            "mmd_sample_size_per_domain": int(n),
        },
    }


def fit_predict_lr(
    train_text: List[str],
    train_y: np.ndarray,
    test_text: List[str],
    max_features: int,
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, train_y)
    probs = clf.predict_proba(X_test)[:, 1]
    pred = (probs >= 0.5).astype(int)
    return pred, probs


def tstr_trts_augmentation_once(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    test_size: float,
    seed: int,
    max_features: int,
) -> Dict[str, Dict[str, float]]:
    x_real = real_df["sequence"].tolist()
    y_real = real_df["label"].to_numpy(dtype=int)
    x_synth = synth_df["sequence"].tolist()
    y_synth = synth_df["label"].to_numpy(dtype=int)

    x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(
        x_real,
        y_real,
        test_size=test_size,
        random_state=seed,
        stratify=y_real,
    )

    pred, prob = fit_predict_lr(x_train_r, y_train_r, x_test_r, max_features=max_features)
    baseline = binary_metrics(y_test_r, pred, prob)

    x_aug = list(x_train_r) + x_synth
    y_aug = np.concatenate([y_train_r, y_synth])
    pred, prob = fit_predict_lr(x_aug, y_aug, x_test_r, max_features=max_features)
    aug = binary_metrics(y_test_r, pred, prob)

    pred, prob = fit_predict_lr(x_synth, y_synth, x_test_r, max_features=max_features)
    tstr = binary_metrics(y_test_r, pred, prob)

    pred, prob = fit_predict_lr(x_train_r, y_train_r, x_synth, max_features=max_features)
    trts = binary_metrics(y_synth, pred, prob)

    return {
        "baseline_real_to_real": baseline,
        "augmented_real_plus_synth_to_real": aug,
        "tstr_synth_to_real": tstr,
        "trts_real_to_synth": trts,
    }


def aggregate_metric_blocks(blocks: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    names = list(blocks[0].keys())
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "fn", "fp"]
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name in names:
        out[name] = {}
        for mk in metric_keys:
            vals = np.array([float(b[name][mk]) for b in blocks], dtype=float)
            out[name][mk] = {
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
            }
    return out


def ood_check(real_df: pd.DataFrame, synth_df: pd.DataFrame, max_features: int, contamination: float) -> Dict[str, float]:
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    x_real = vec.fit_transform(real_df["sequence"].tolist())
    x_synth = vec.transform(synth_df["sequence"].tolist())
    detector = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=1,
    )
    detector.fit(x_real)
    pred_real = detector.predict(x_real)   # +1 inlier, -1 outlier
    pred_synth = detector.predict(x_synth)
    score_real = detector.score_samples(x_real)
    score_synth = detector.score_samples(x_synth)
    return {
        "real_outlier_ratio": float(np.mean(pred_real == -1)),
        "synthetic_outlier_ratio": float(np.mean(pred_synth == -1)),
        "score_mean_real": float(np.mean(score_real)),
        "score_mean_synthetic": float(np.mean(score_synth)),
        "score_gap_real_minus_synthetic": float(np.mean(score_real) - np.mean(score_synth)),
    }


def privacy_memorization_proxy(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    test_size: float,
    max_features: int,
    near_dup_threshold: float,
) -> Dict[str, float]:
    x_real = real_df["sequence"].tolist()
    y_real = real_df["label"].to_numpy(dtype=int)
    x_train, x_holdout, _, _ = train_test_split(
        x_real,
        y_real,
        test_size=test_size,
        random_state=42,
        stratify=y_real,
    )

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    x_all = x_train + x_holdout + synth_df["sequence"].tolist()
    vec.fit(x_all)
    X_synth = vec.transform(synth_df["sequence"].tolist())
    X_train = vec.transform(x_train)
    X_hold = vec.transform(x_holdout)

    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(X_synth)
    d_train, _ = nn.kneighbors(X_train, return_distance=True)
    d_hold, _ = nn.kneighbors(X_hold, return_distance=True)
    sim_train = 1.0 - d_train[:, 0]
    sim_hold = 1.0 - d_hold[:, 0]

    y_member = np.concatenate([np.ones_like(sim_train), np.zeros_like(sim_hold)])
    scores = np.concatenate([sim_train, sim_hold])
    mia_auc = safe_auc(y_member, scores)

    synth_set = set(synth_df["sequence"].tolist())
    exact_train = np.mean([s in synth_set for s in x_train])
    exact_hold = np.mean([s in synth_set for s in x_holdout])

    return {
        "membership_inference_auc_via_nn_similarity": float(mia_auc),
        "mean_nn_similarity_train_members": float(np.mean(sim_train)),
        "mean_nn_similarity_holdout_nonmembers": float(np.mean(sim_hold)),
        "near_dup_ratio_train_members": float(np.mean(sim_train >= near_dup_threshold)),
        "near_dup_ratio_holdout_nonmembers": float(np.mean(sim_hold >= near_dup_threshold)),
        "exact_overlap_ratio_train_members": float(exact_train),
        "exact_overlap_ratio_holdout_nonmembers": float(exact_hold),
    }


def make_text_report(report: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("[Generation Quality Report]")
    lines.append(f"created_at: {report['meta']['created_at']}")
    lines.append(f"real_rows: {report['meta']['real_rows']}, synthetic_rows: {report['meta']['synthetic_rows']}")
    lines.append("")

    dup = report["duplicate_and_leakage"]
    lines.append("[1] Duplicate and leakage checks")
    lines.append(f"exact_duplicate_ratio: {dup['exact_duplicate_ratio']:.4f}")
    lines.append(f"near_duplicate_ratio(>={dup['near_duplicate_threshold']}): {dup['near_duplicate_ratio']:.4f}")
    lines.append(f"nearest_similarity_mean: {dup['nearest_similarity_mean']:.4f}")
    lines.append("")

    div = report["diversity_and_coverage"]
    lines.append("[2] Diversity and coverage")
    lines.append(
        "synthetic_unique_sequence_ratio: "
        f"{div['synthetic']['unique_sequence_ratio']:.4f}, "
        f"token_novelty_vs_real: {div['coverage']['synthetic_token_novelty_vs_real']:.4f}"
    )
    lines.append("")

    dist = report["distribution_closeness"]
    lines.append("[3] Statistical closeness")
    lines.append(
        "length_ks: "
        f"{dist['length_distribution']['ks_statistic']:.4f}, "
        f"length_wasserstein: {dist['length_distribution']['approx_wasserstein']:.4f}"
    )
    lines.append(
        "js_unigram: "
        f"{dist['token_distribution']['js_divergence_unigram']:.4f}, "
        f"js_bigram: {dist['token_distribution']['js_divergence_bigram']:.4f}, "
        f"mmd2: {dist['embedding_space']['mmd_rbf_squared']:.6f}"
    )
    lines.append("")

    util = report["stability_protocol_aggregate"]
    lines.append("[4] Utility protocol (mean over seeds)")
    for name in ["baseline_real_to_real", "augmented_real_plus_synth_to_real", "tstr_synth_to_real", "trts_real_to_synth"]:
        lines.append(
            f"{name}: "
            f"acc={util[name]['accuracy']['mean']:.4f}, "
            f"f1={util[name]['f1']['mean']:.4f}, "
            f"recall={util[name]['recall']['mean']:.4f}, "
            f"fn={util[name]['fn']['mean']:.1f}"
        )
    lines.append("")

    ood = report["ood_check"]
    lines.append("[5] OOD check")
    lines.append(
        f"synthetic_outlier_ratio={ood['synthetic_outlier_ratio']:.4f}, "
        f"real_outlier_ratio={ood['real_outlier_ratio']:.4f}, "
        f"score_gap_real_minus_synthetic={ood['score_gap_real_minus_synthetic']:.4f}"
    )
    lines.append("")

    priv = report["privacy_risk_proxy"]
    lines.append("[6] Privacy/memorization proxy")
    lines.append(
        f"MIA_AUC={priv['membership_inference_auc_via_nn_similarity']:.4f}, "
        f"mean_sim_member={priv['mean_nn_similarity_train_members']:.4f}, "
        f"mean_sim_nonmember={priv['mean_nn_similarity_holdout_nonmembers']:.4f}"
    )
    lines.append("")

    verdict = report["verdict"]
    lines.append("[7] Verdict")
    lines.append(f"overall_status: {verdict['overall_status']}")
    lines.append(f"summary: {verdict['summary']}")
    lines.append("")
    lines.append("criteria:")
    for c in verdict["criteria"]:
        lines.append(
            f"- {c['name']}: {c['status']} | value={c['value']} | "
            f"pass_if={c['pass_rule']} | warn_if={c['warn_rule']}"
        )
    return "\n".join(lines) + "\n"


def parse_seeds(seeds_raw: str) -> List[int]:
    out = []
    for part in seeds_raw.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        raise ValueError("No valid seeds")
    return out


def evaluate_verdict(report: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    criteria: List[Dict[str, str]] = []

    def add_criterion(name: str, value: float, status: str, pass_rule: str, warn_rule: str) -> None:
        criteria.append(
            {
                "name": name,
                "status": status,
                "value": f"{value:.6f}",
                "pass_rule": pass_rule,
                "warn_rule": warn_rule,
            }
        )

    dup = report["duplicate_and_leakage"]
    exact_dup = float(dup["exact_duplicate_ratio"])
    if exact_dup <= args.thr_exact_dup_pass:
        st = "PASS"
    elif exact_dup <= args.thr_exact_dup_warn:
        st = "WARN"
    else:
        st = "FAIL"
    add_criterion(
        "exact_duplicate_ratio",
        exact_dup,
        st,
        f"<= {args.thr_exact_dup_pass}",
        f"<= {args.thr_exact_dup_warn}",
    )

    near_dup = float(dup["near_duplicate_ratio"])
    if near_dup <= args.thr_near_dup_pass:
        st = "PASS"
    elif near_dup <= args.thr_near_dup_warn:
        st = "WARN"
    else:
        st = "FAIL"
    add_criterion(
        "near_duplicate_ratio",
        near_dup,
        st,
        f"<= {args.thr_near_dup_pass}",
        f"<= {args.thr_near_dup_warn}",
    )

    js_uni = float(report["distribution_closeness"]["token_distribution"]["js_divergence_unigram"])
    if js_uni <= args.thr_js_uni_pass:
        st = "PASS"
    elif js_uni <= args.thr_js_uni_warn:
        st = "WARN"
    else:
        st = "FAIL"
    add_criterion(
        "js_divergence_unigram",
        js_uni,
        st,
        f"<= {args.thr_js_uni_pass}",
        f"<= {args.thr_js_uni_warn}",
    )

    agg = report["stability_protocol_aggregate"]
    baseline_f1 = float(agg["baseline_real_to_real"]["f1"]["mean"])
    tstr_f1 = float(agg["tstr_synth_to_real"]["f1"]["mean"])
    tstr_f1_ratio = tstr_f1 / baseline_f1 if baseline_f1 > 0 else 0.0
    if tstr_f1_ratio >= args.thr_tstr_f1_ratio_pass:
        st = "PASS"
    elif tstr_f1_ratio >= args.thr_tstr_f1_ratio_warn:
        st = "WARN"
    else:
        st = "FAIL"
    add_criterion(
        "tstr_f1_ratio_vs_baseline",
        tstr_f1_ratio,
        st,
        f">= {args.thr_tstr_f1_ratio_pass}",
        f">= {args.thr_tstr_f1_ratio_warn}",
    )

    aug_f1 = float(agg["augmented_real_plus_synth_to_real"]["f1"]["mean"])
    aug_delta_f1 = aug_f1 - baseline_f1
    if aug_delta_f1 >= args.thr_aug_f1_drop_warn:
        st = "PASS"
    elif aug_delta_f1 >= args.thr_aug_f1_drop_fail:
        st = "WARN"
    else:
        st = "FAIL"
    add_criterion(
        "augmentation_delta_f1",
        aug_delta_f1,
        st,
        f">= {args.thr_aug_f1_drop_warn}",
        f">= {args.thr_aug_f1_drop_fail}",
    )

    baseline_fn = float(agg["baseline_real_to_real"]["fn"]["mean"])
    aug_fn = float(agg["augmented_real_plus_synth_to_real"]["fn"]["mean"])
    fn_delta = aug_fn - baseline_fn
    if fn_delta <= args.thr_fn_delta_warn:
        st = "PASS"
    elif fn_delta <= args.thr_fn_delta_fail:
        st = "WARN"
    else:
        st = "FAIL"
    add_criterion(
        "augmentation_fn_delta",
        fn_delta,
        st,
        f"<= {args.thr_fn_delta_warn}",
        f"<= {args.thr_fn_delta_fail}",
    )

    ood = report["ood_check"]
    ood_gap = float(ood["synthetic_outlier_ratio"] - ood["real_outlier_ratio"])
    if ood_gap <= args.thr_ood_gap_pass:
        st = "PASS"
    elif ood_gap <= args.thr_ood_gap_warn:
        st = "WARN"
    else:
        st = "FAIL"
    add_criterion(
        "ood_outlier_ratio_gap",
        ood_gap,
        st,
        f"<= {args.thr_ood_gap_pass}",
        f"<= {args.thr_ood_gap_warn}",
    )

    mia_auc = float(report["privacy_risk_proxy"]["membership_inference_auc_via_nn_similarity"])
    if mia_auc <= args.thr_mia_auc_pass:
        st = "PASS"
    elif mia_auc <= args.thr_mia_auc_warn:
        st = "WARN"
    else:
        st = "FAIL"
    add_criterion(
        "privacy_mia_auc_proxy",
        mia_auc,
        st,
        f"<= {args.thr_mia_auc_pass}",
        f"<= {args.thr_mia_auc_warn}",
    )

    statuses = [c["status"] for c in criteria]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses:
        overall = "WARN"
    else:
        overall = "PASS"

    summary = (
        f"PASS={statuses.count('PASS')}, "
        f"WARN={statuses.count('WARN')}, "
        f"FAIL={statuses.count('FAIL')}"
    )
    return {
        "overall_status": overall,
        "summary": summary,
        "criteria": criteria,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    real_df = prepare_dataset(
        csv_path=args.real_csv,
        label_col=args.label_col,
        seq_col=args.seq_col,
        api_col_regex=args.api_col_regex,
        include_unnamed_api_cols=args.include_unnamed_api_cols,
        delimiter_regex=args.delimiter_regex,
    )
    synth_df = prepare_dataset(
        csv_path=args.synthetic_csv,
        label_col=args.label_col,
        seq_col=args.seq_col,
        api_col_regex=args.api_col_regex,
        include_unnamed_api_cols=args.include_unnamed_api_cols,
        delimiter_regex=args.delimiter_regex,
    )

    seeds = parse_seeds(args.seeds)

    per_seed = []
    for seed in seeds:
        per_seed.append(
            tstr_trts_augmentation_once(
                real_df=real_df,
                synth_df=synth_df,
                test_size=args.test_size,
                seed=seed,
                max_features=args.max_features,
            )
        )

    report = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "real_csv": os.path.abspath(args.real_csv),
            "synthetic_csv": os.path.abspath(args.synthetic_csv),
            "real_rows": int(len(real_df)),
            "synthetic_rows": int(len(synth_df)),
            "label_col": args.label_col,
            "test_size": args.test_size,
            "seeds": seeds,
        },
        "duplicate_and_leakage": exact_and_near_duplicates(
            real_df=real_df,
            synth_df=synth_df,
            max_features=args.max_features,
            near_dup_threshold=args.near_dup_threshold,
        ),
        "diversity_and_coverage": diversity_and_coverage(real_df=real_df, synth_df=synth_df),
        "distribution_closeness": distribution_closeness(
            real_df=real_df,
            synth_df=synth_df,
            mmd_sample_size=args.mmd_sample_size,
        ),
        "stability_protocol_per_seed": per_seed,
        "stability_protocol_aggregate": aggregate_metric_blocks(per_seed),
        "ood_check": ood_check(
            real_df=real_df,
            synth_df=synth_df,
            max_features=args.max_features,
            contamination=args.ood_contamination,
        ),
        "privacy_risk_proxy": privacy_memorization_proxy(
            real_df=real_df,
            synth_df=synth_df,
            test_size=args.test_size,
            max_features=args.max_features,
            near_dup_threshold=args.near_dup_threshold,
        ),
    }
    report["verdict"] = evaluate_verdict(report, args)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"gen_quality_{ts}"
    json_path = os.path.join(args.output_dir, f"{base}.json")
    txt_path = os.path.join(args.output_dir, f"{base}.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(make_text_report(report))

    print(f"[ok] report json: {json_path}")
    print(f"[ok] report text: {txt_path}")


if __name__ == "__main__":
    main()
