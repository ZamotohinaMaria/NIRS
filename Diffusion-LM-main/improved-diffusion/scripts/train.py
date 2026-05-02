"""
Train a diffusion model on images.
"""

import argparse
import json, torch, os
import numpy as np
import tempfile
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
LOCAL_TRANSFORMERS_SRC = os.path.join(REPO_ROOT, "transformers", "src")

if os.path.isdir(LOCAL_TRANSFORMERS_SRC) and LOCAL_TRANSFORMERS_SRC not in sys.path:
    sys.path.insert(0, LOCAL_TRANSFORMERS_SRC)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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


class TeeStream:
    def __init__(self, stream, filepath):
        self.stream = stream
        self.file = open(filepath, "a", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.stream.flush()
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def setup_console_tee(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    tee_path = os.path.join(log_dir, "console.log")
    sys.stdout = TeeStream(sys.stdout, tee_path)
    sys.stderr = TeeStream(sys.stderr, tee_path)


def ensure_file_exists(path, name):
    if path is None or str(path).strip() == "":
        raise ValueError(f"{name} is empty")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def ensure_dir_writable(path, name):
    if path is None or str(path).strip() == "":
        raise ValueError(f"{name} is empty")
    os.makedirs(path, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix="._write_test_", delete=True):
            pass
    except Exception as e:
        raise PermissionError(f"{name} is not writable: {path}. Error: {e}")


def validate_training_paths(args):
    ensure_dir_writable(args.checkpoint_path, "checkpoint_path")

    if args.modality == "malbehav":
        ensure_file_exists(args.malbehav_train, "malbehav_train")
        if str(args.malbehav_valid).strip() != "":
            ensure_file_exists(args.malbehav_valid, "malbehav_valid")
        if str(args.malbehav_test).strip() != "":
            ensure_file_exists(args.malbehav_test, "malbehav_test")


def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    dist_util.setup_dist() # DEBUG **
    validate_training_paths(args)
    if dist.get_rank() == 0:
        setup_console_tee(args.checkpoint_path)
    logger.configure(dir=args.checkpoint_path)
    if dist.get_rank() == 0:
        logger.log(f"console tee logging to {os.path.join(args.checkpoint_path, 'console.log')}")


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
        print(model3, model3.weight.requires_grad)
        mapping_func = partial(compute_logp, args, model3.cuda())
        diffusion.mapping_func = mapping_func
        return mapping_func

    get_mapping_func(args, diffusion, data)

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
        eval_num_batches=args.eval_num_batches,
        early_stop_patience_eval=args.early_stop_patience_eval,
        early_stop_min_delta=args.early_stop_min_delta,
        save_best_model=(args.save_best_model == "yes"),
        best_model_filename=args.best_model_filename,
        save_optimizer_state=(args.save_optimizer_state == "yes"),
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
        eval_num_batches=4,
        early_stop_patience_eval=5,
        early_stop_min_delta=0.0,
        checkpoint_path='diff_models',
        save_best_model='yes',
        best_model_filename='best_model.pt',
        save_optimizer_state='yes',
    )
    text_defaults = dict(modality='text',
                         dataset_name='wikitext',
                         dataset_config_name='wikitext-2-raw-v1',
                         config='diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k128_trainc20000.yaml',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress',model_arch='conv-unet',
                         roc_train='diffusion_lm/ROCstory',#'diffusion_lm/ROCstory/ROCstory17.csv',
                         wiki_train='diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
                         e2e_train='e2e_data',
                         yelp_train='diffusion_lm/yelpnlg-resources/yelpnlg-corpus',
                         commonGen_train = 'diffusion_lm/common-gen/commongen_data',
                         malbehav_train='',
                         malbehav_valid='',
                         malbehav_test='',
                         malbehav_skip_columns='sha256,labels',
                         malbehav_label_column='labels',
                         malbehav_include_label='yes',
                         emb_scale_factor=1.0, noise_level=0.0, cache_mode='no', use_bert_tokenizer='no',
                         padding_mode='block',
                         preprocessing_num_workers=1)
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
