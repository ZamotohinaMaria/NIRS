"""
Train a diffusion model on images.
"""

import argparse
import json, torch, os
from collections import Counter
import numpy as np
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.text_datasets import load_data_text
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from transformers import AutoTokenizer
from improved_diffusion.train_util import TrainLoop
from transformers import set_seed
from functools import partial
from improved_diffusion.test_util import get_weights, compute_logp
from improved_diffusion.rounding import load_models, load_tokenizer
import torch.distributed as dist
import wandb

def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _infer_roc_vocab_size(roc_train_dir: str, min_token_freq: int):
    train_path = os.path.join(roc_train_dir, "roc_train.json")
    if not os.path.exists(train_path):
        return None

    min_token_freq = max(1, int(min_token_freq))
    counter = Counter()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, list) and len(payload) > 0:
                text = str(payload[0])
            elif isinstance(payload, str):
                text = payload
            else:
                continue
            counter.update(tok for tok in text.split() if tok)

    kept_tokens = sum(1 for _, freq in counter.items() if freq >= min_token_freq)
    # START/END/UNK/PAD
    return kept_tokens + 4


def _apply_roc_profile(args):
    if args.modality not in {"roc", "roc-aug"}:
        return []
    if not _as_bool(getattr(args, "apply_roc_profile", True)):
        return []

    changes = []

    def _set_arg(name, new_value):
        old_value = getattr(args, name)
        if old_value != new_value:
            setattr(args, name, new_value)
            changes.append(f"{name}: {old_value} -> {new_value}")

    resumed = bool(args.resume_checkpoint)
    if resumed:
        changes.append("resume checkpoint detected: keep hyperparameters for compatibility")
        if args.early_stop_patience_steps > 0:
            _set_arg("early_stop_patience_steps", 0)
        return changes

    if args.training_mode == "emb":
        _set_arg("training_mode", "e2e")
    if args.in_channel < 128:
        _set_arg("in_channel", 128)
    if args.out_channel != args.in_channel:
        _set_arg("out_channel", args.in_channel)
    if args.padding_mode != "pad":
        _set_arg("padding_mode", "pad")
    if not args.predict_xstart:
        _set_arg("predict_xstart", True)
    if args.diffusion_steps < 2000:
        _set_arg("diffusion_steps", 2000)
    if args.lr_anneal_steps < 200000:
        _set_arg("lr_anneal_steps", 400000)
    if args.weight_decay != 0.0:
        _set_arg("weight_decay", 0.0)
    if args.early_stop_patience_steps > 0:
        _set_arg("early_stop_patience_steps", 0)
    if args.min_token_freq != 1:
        _set_arg("min_token_freq", 1)
    if _as_bool(getattr(args, "auto_vocab_size", True)) and str(args.training_mode).startswith("e2e"):
        inferred_vocab_size = _infer_roc_vocab_size(args.roc_train, args.min_token_freq)
        if inferred_vocab_size is not None:
            _set_arg("vocab_size", inferred_vocab_size)
        else:
            changes.append(
                f"vocab_size: keep {args.vocab_size} (cannot find {args.roc_train}/roc_train.json)"
            )

    return changes


def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    dist_util.setup_dist() # DEBUG **
    os.makedirs(args.checkpoint_path, exist_ok=True)
    logger.configure(dir=args.checkpoint_path)
    for update_msg in _apply_roc_profile(args):
        logger.log(f"roc profile | {update_msg}")


    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev()) #  DEBUG **
    # model.cuda() #  DEBUG **

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'the parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "diffusion_lm"),
        name=args.checkpoint_path,
    )
    wandb.config.update(args.__dict__, allow_val_change=True)

    if args.experiment_mode == 'conditional_gen':
        assert args.modality in ['e2e']
        assert args.padding_mode == 'pad'

    logger.log("creating data loader...")
    if args.modality == 'image':
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )
        data_valid = None

    else:
        print('load data', '*'*50)
        if args.modality == 'roc-aug' or args.modality == 'commonGen-aug':
            tokenizer = load_tokenizer(args.modality, args.experiment, 'predictability/diffusion_models_v7/diff_roc_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd108_xstart')
            rev_tokenizer = {v: k for k, v in tokenizer.items()}
            print(len(rev_tokenizer), 'loading from tokenizer. ')
        elif args.use_bert_tokenizer == 'yes':
            rev_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            rev_tokenizer = None

        if args.experiment == 'random1':
            args.experiment = 'random'
            print('loading from the vocabs here.')
            assert args.in_channel == 64
            assert args.modality == 'roc'
            model22 = torch.nn.Embedding(args.vocab_size, args.in_channel)
            model22_weight = torch.load('predictability/diffusion_models_v7/diff_roc-aug_pad_rand64_'
                                        'transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd108_xstart_e2e/'
                                        'ema_0.9999_200000.pt', map_location='cpu')['word_embedding.weight']
            model22.weight = model22_weight
            model22.weight.requires_grad=False
        else:
            model22 = None

        data = load_data_text(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            data_args = args,
            task_mode=args.modality,
            padding_mode=args.padding_mode, #block, pad
            load_vocab=rev_tokenizer,
            model=model22,
        )
        next(data)
        model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                        args.checkpoint_path, extra_args=args)
        if args.modality == 'book' or args.use_bert_tokenizer == 'yes':
            rev_tokenizer = tokenizer # BERT tokenizer BPE.
        else:
            rev_tokenizer = {v: k for k, v in tokenizer.items()}

        data_valid = load_data_text(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            deterministic=True,
            data_args=args,
            task_mode=args.modality,
            padding_mode=args.padding_mode,  # block, pad
            split='valid',
            load_vocab=rev_tokenizer,
            model=model2,
        )

    # dist.barrier()
    # import time
    # while not os.path.exists(os.path.join(args.checkpoint_path, 'vocab.json')):
    #     time.sleep(1)
    def get_mapping_func(args, diffusion, data):
        model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                        args.checkpoint_path, extra_args=args)
        model3 = get_weights(model2, args)
        model3 = model3.to(dist_util.dev())
        print(model3, model3.weight.requires_grad)
        mapping_func = partial(compute_logp, args, model3)
        diffusion.mapping_func = mapping_func
        return mapping_func

    if args.modality != 'image' and not str(args.training_mode).startswith('e2e'):
        get_mapping_func(args, diffusion, data)
    else:
        logger.log("skipping mapping function setup for image/e2e training mode.")

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval,
        early_stop_patience_steps=args.early_stop_patience_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        seed=101,
        gradient_clipping=-1.0,
        eval_interval=2000,
        early_stop_patience_steps=0,
        checkpoint_path='diff_models'
    )
    text_defaults = dict(modality='text',
                         dataset_name='wikitext',
                         dataset_config_name='wikitext-2-raw-v1',
                         config='',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress',model_arch='conv-unet',
                         roc_train='diffusion_lm/ROCstory',#'diffusion_lm/ROCstory/ROCstory17.csv',
                         wiki_train='diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
                         e2e_train='e2e_data',
                         yelp_train='diffusion_lm/yelpnlg-resources/yelpnlg-corpus',
                          commonGen_train = 'diffusion_lm/common-gen/commongen_data',
                          emb_scale_factor=1.0, noise_level=0.0, cache_mode='no', use_bert_tokenizer='no',
                          padding_mode='block',
                          min_token_freq=1,
                          apply_roc_profile=True,
                          auto_vocab_size=True,
                          preprocessing_num_workers=1)
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
