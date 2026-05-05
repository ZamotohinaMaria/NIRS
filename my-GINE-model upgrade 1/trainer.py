from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = float("nan")
    try:
        brier = brier_score_loss(y_true, y_prob)
    except Exception:
        brier = float("nan")

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "mcc": float(mcc),
        "brier": float(brier),
        "roc_auc": float(roc_auc),
    }


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.cross_entropy(logits, batch.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def collect_logits_labels(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true = []
    all_logits = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_true.extend(batch.y.cpu().numpy().tolist())
        all_logits.extend(logits.detach().cpu().numpy().tolist())

    y_true = np.array(all_true, dtype=np.int64)
    logits_np = np.array(all_logits, dtype=np.float32)
    return y_true, logits_np


def probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected logits shape [N, 2], got {logits.shape}")
    temp = max(float(temperature), 1e-6)
    logits_t = logits / temp
    logits_t = logits_t - logits_t.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_t)
    probs = exp_logits[:, 1] / np.clip(exp_logits.sum(axis=1), a_min=1e-12, a_max=None)
    return probs.astype(np.float64)


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    threshold_grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    if threshold_grid is None:
        threshold_grid = np.linspace(0.05, 0.95, 37)

    best_threshold = 0.5
    best_score = -1.0
    for threshold in threshold_grid:
        y_pred = (y_prob >= float(threshold)).astype(int)
        if metric == "mcc":
            score = matthews_corrcoef(y_true, y_pred)
        else:
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            score = float(f1)

        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_score


def fit_temperature_scaler(
    logits: np.ndarray,
    y_true: np.ndarray,
    device: torch.device,
    max_iter: int = 100,
) -> float:
    if logits.shape[0] == 0:
        return 1.0

    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(y_true, dtype=torch.long, device=device)
    temperature = torch.nn.Parameter(torch.ones(1, device=device))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_t / torch.clamp(temperature, min=1e-6), labels_t)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        value = float(torch.clamp(temperature.detach(), min=1e-6).item())
        if np.isfinite(value) and value > 0:
            return value
    except Exception:
        pass

    return 1.0


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    threshold: float = 0.5,
    temperature: float = 1.0,
    return_details: bool = False,
):
    y_true, logits = collect_logits_labels(model, loader, device)
    y_prob = probs_from_logits(logits, temperature=temperature)
    y_pred = (y_prob >= float(threshold)).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    if not return_details:
        return metrics

    details = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "logits": logits,
    }
    return metrics, details


def train_model(
    model,
    train_loader,
    val_loader: Optional[object],
    optimizer,
    device,
    epochs: int,
    save_path: str,
    monitor_metric: str = "f1",
) -> Tuple[int, float]:
    best_val_score = -1.0
    best_epoch = -1
    best_train_loss = float("inf")

    print("\nStart training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch:03d} | "
                f"loss={train_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"val_pr_auc={val_metrics['pr_auc']:.4f} | "
                f"val_mcc={val_metrics['mcc']:.4f} | "
                f"val_brier={val_metrics['brier']:.4f} | "
                f"val_auc={val_metrics['roc_auc']:.4f}"
            )

            current_score = float(val_metrics.get(monitor_metric, float("nan")))
            if np.isfinite(current_score) and current_score > best_val_score:
                best_val_score = current_score
                best_epoch = epoch
                torch.save(model.state_dict(), save_path)
        else:
            print(f"Epoch {epoch:03d} | loss={train_loss:.4f}")
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_epoch = epoch
                torch.save(model.state_dict(), save_path)

    if val_loader is None:
        best_val_score = float("nan")

    return best_epoch, best_val_score
