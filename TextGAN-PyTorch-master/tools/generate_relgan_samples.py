#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate synthetic API-call sequences from a trained RelGAN generator checkpoint.
"""

import argparse
import os

import torch

import config as cfg
from models.RelGAN_G import RelGAN_G
from utils.text_process import load_dict, tensor_to_tokens, text_process, write_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. MalbehavD-V1_malware")
    parser.add_argument("--checkpoint", required=True, help="Path to generator checkpoint (.pt)")
    parser.add_argument("--output_txt", required=True, help="Output text file path")
    parser.add_argument("--num_samples", type=int, default=1285, help="Number of sequences to generate")
    parser.add_argument("--batch_size", type=int, default=64, help="Sampling batch size")
    parser.add_argument("--model_type", default="vanilla", choices=["vanilla", "LSTM"])
    parser.add_argument("--mem_slots", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--head_size", type=int, default=128)
    parser.add_argument("--gen_embed_dim", type=int, default=64)
    parser.add_argument("--gen_hidden_dim", type=int, default=64)
    parser.add_argument("--cuda", type=int, default=1, help="Use CUDA if available (1/0)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    return parser.parse_args()


def main():
    args = parse_args()

    train_path = os.path.join("dataset", f"{args.dataset}.txt")
    test_path = os.path.join("dataset", "testdata", f"{args.dataset}_test.txt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Dataset file not found: {train_path}")

    seq_len, vocab_size = text_process(train_path, test_path if os.path.exists(test_path) else None)
    cfg.dataset = args.dataset
    cfg.max_seq_len = seq_len
    cfg.vocab_size = vocab_size
    cfg.model_type = args.model_type

    use_cuda = bool(args.cuda) and torch.cuda.is_available()
    cfg.CUDA = use_cuda
    cfg.device = args.device if use_cuda else -1

    if use_cuda:
        torch.cuda.set_device(cfg.device)

    _, idx2word = load_dict(args.dataset)

    gen = RelGAN_G(
        args.mem_slots,
        args.num_heads,
        args.head_size,
        args.gen_embed_dim,
        args.gen_hidden_dim,
        cfg.vocab_size,
        cfg.max_seq_len,
        cfg.padding_idx,
        gpu=use_cuda,
    )

    map_location = f"cuda:{cfg.device}" if use_cuda else "cpu"
    state = torch.load(args.checkpoint, map_location=map_location)
    gen.load_state_dict(state)
    if use_cuda:
        gen = gen.cuda()
    gen.eval()

    with torch.no_grad():
        samples = gen.sample(args.num_samples, args.batch_size)
    tokens = tensor_to_tokens(samples, idx2word)
    tokens = [[t for t in sent if t not in (cfg.start_token, cfg.padding_token)] for sent in tokens]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_txt)), exist_ok=True)
    write_tokens(args.output_txt, tokens)
    print(f"[ok] generated {len(tokens)} samples -> {os.path.abspath(args.output_txt)}")


if __name__ == "__main__":
    main()
