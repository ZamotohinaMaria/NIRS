import argparse
import json
import math
import os
import random
from functools import partial
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, set_seed


def _read_roc_jsonl(path: str) -> List[List[str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)[0]
            rows.append(text.split())
    return rows


def _encode_rows(rows: List[List[str]], vocab: dict, max_len: int) -> List[List[int]]:
    start_id = vocab["START"]
    end_id = vocab["END"]
    unk_id = vocab["UNK"]
    encoded = []
    for tokens in rows:
        ids = [start_id] + [vocab.get(tok, unk_id) for tok in tokens] + [end_id]
        ids = ids[:max_len]
        encoded.append(ids)
    return encoded


class SeqDataset(Dataset):
    def __init__(self, sequences: List[List[int]]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def _collate(batch, pad_id):
    max_len = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_t = torch.tensor(seq, dtype=torch.long)
        input_ids[i, : len(seq)] = seq_t
        attention_mask[i, : len(seq)] = 1
    labels = input_ids.clone()
    labels[labels == pad_id] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        valid_tokens = (batch["labels"] != -100).sum().item()
        total_loss += out.loss.item() * valid_tokens
        total_tokens += valid_tokens
    if total_tokens == 0:
        return float("inf"), float("inf")
    nll = total_loss / total_tokens
    ppl = math.exp(min(nll, 50.0))
    return nll, ppl


def main():
    parser = argparse.ArgumentParser(description="Train AR LM for Diffusion-LM AR evaluation.")
    parser.add_argument("--roc_train", type=str, required=True, help="Directory with roc_train.json and roc_valid.json")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to diffusion vocab.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir for save_pretrained")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=169)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    required = ["START", "END", "UNK", "PAD"]
    for token in required:
        if token not in vocab:
            raise ValueError(f"Token '{token}' is missing in vocab: {args.vocab_path}")

    train_path = os.path.join(args.roc_train, "roc_train.json")
    valid_path = os.path.join(args.roc_train, "roc_valid.json")
    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        raise FileNotFoundError(f"Expected files: {train_path} and {valid_path}")

    train_rows = _read_roc_jsonl(train_path)
    valid_rows = _read_roc_jsonl(valid_path)
    train_ids = _encode_rows(train_rows, vocab, args.max_len)
    valid_ids = _encode_rows(valid_rows, vocab, args.max_len)

    pad_id = vocab["PAD"]
    train_ds = SeqDataset(train_ids)
    valid_ds = SeqDataset(valid_ids)
    collate_fn = partial(_collate, pad_id=pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=args.max_len,
        n_ctx=args.max_len,
        n_embd=args.d_model,
        n_layer=args.n_layers,
        n_head=args.n_heads,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        bos_token_id=vocab["START"],
        eos_token_id=vocab["END"],
        pad_token_id=pad_id,
    )
    model = GPT2LMHeadModel(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_nll = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_tokens = 0
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            valid_tokens = (batch["labels"] != -100).sum().item()
            running_loss += loss.item() * valid_tokens
            running_tokens += valid_tokens

        train_nll = running_loss / max(running_tokens, 1)
        train_ppl = math.exp(min(train_nll, 50.0))
        valid_nll, valid_ppl = evaluate(model, valid_loader, args.device)
        print(
            f"epoch {epoch}/{args.epochs} | "
            f"train_nll={train_nll:.4f} train_ppl={train_ppl:.2f} | "
            f"valid_nll={valid_nll:.4f} valid_ppl={valid_ppl:.2f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_nll": train_nll,
                "train_ppl": train_ppl,
                "valid_nll": valid_nll,
                "valid_ppl": valid_ppl,
            }
        )

        if valid_nll < best_valid_nll:
            best_valid_nll = valid_nll
            best_epoch = epoch
            best_dir = os.path.join(args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            with open(os.path.join(best_dir, "train_args.json"), "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2)

    model.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"best_epoch": best_epoch, "best_valid_nll": best_valid_nll, "history": history},
            f,
            indent=2,
        )
    print(f"Training finished. Best epoch: {best_epoch}, best valid_nll: {best_valid_nll:.4f}")
    print(f"Saved final model to: {args.output_dir}")
    print(f"Saved best model to: {os.path.join(args.output_dir, 'best')}")


if __name__ == "__main__":
    main()
