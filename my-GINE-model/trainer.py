from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
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
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()

    all_true = []
    all_pred = []
    all_prob = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_true.extend(batch.y.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)

    return compute_metrics(y_true, y_pred, y_prob)


def train_model(model, train_loader, val_loader, optimizer, device, epochs: int, save_path: str) -> Tuple[int, float]:
    best_val_f1 = -1.0
    best_epoch = -1

    print("\nStart training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

    return best_epoch, best_val_f1
