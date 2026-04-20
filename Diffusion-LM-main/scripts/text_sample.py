"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
import sys
import math
import subprocess
import time

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer

from improved_diffusion.test_util import get_weights, denoised_fn_round

from improved_diffusion import dist_util, logger
from functools import partial
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def _normalize_text(text):
    return " ".join(text.strip().split())


def _distinct_n(tokenized_texts, n):
    total = 0
    uniq = set()
    for tokens in tokenized_texts:
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            uniq.add(tuple(tokens[i:i + n]))
            total += 1
    if total == 0:
        return 0.0
    return len(uniq) / total


def _memorization_rate_exact(generated_texts, train_jsonl_path):
    if not train_jsonl_path or not os.path.exists(train_jsonl_path):
        return None

    train_set = set()
    with open(train_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)[0]
            train_set.add(_normalize_text(text))

    if len(generated_texts) == 0:
        return 0.0

    memorized = 0
    for text in generated_texts:
        if _normalize_text(text) in train_set:
            memorized += 1
    return memorized / len(generated_texts)


def _compute_generation_metrics(word_lst, args):
    tokenized = [w.strip().split() for w in word_lst if w.strip()]
    normalized_texts = [_normalize_text(w) for w in word_lst if w.strip()]
    metrics = {
        "num_generated": len(normalized_texts),
        "avg_length": (sum(len(x) for x in tokenized) / len(tokenized)) if tokenized else 0.0,
        "distinct_1": _distinct_n(tokenized, 1),
        "distinct_2": _distinct_n(tokenized, 2),
        "unique_sequence_ratio": (len(set(normalized_texts)) / len(normalized_texts)) if normalized_texts else 0.0,
        "memorization_rate_exact_train": None,
    }

    if hasattr(args, "modality") and args.modality in ["roc", "roc-aug"] and hasattr(args, "roc_train"):
        train_path = os.path.join(args.roc_train, "roc_train.json")
        metrics["memorization_rate_exact_train"] = _memorization_rate_exact(normalized_texts, train_path)

    return metrics


def _safe_read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_valid_bpd_eval(args):
    return _run_bpd_eval(args, "valid")


def _run_bpd_eval(args, split):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nll_script = os.path.join(script_dir, "nll.py")
    cmd = [
        sys.executable,
        nll_script,
        "--model_path", args.model_path,
        "--batch_size", str(args.nll_batch_size),
        "--num_samples", str(args.nll_num_samples),
        "--split", split,
        "--out_dir", args.nll_out_dir,
        "--clamp", args.nll_clamp,
    ]
    subprocess.run(cmd, check=True)
    model_dir = os.path.split(args.model_path)[0]
    if "ema" in args.model_path:
        json_path = os.path.join(model_dir, f"ema_score_{split}_nll.json")
    elif args.nll_clamp == "noclamp":
        json_path = os.path.join(model_dir, f"score_{split}_nll_noclamp.json")
    else:
        json_path = os.path.join(model_dir, f"score_{split}_nll.json")
    return _safe_read_json(json_path)


def _run_ar_eval(args, generated_path):
    if not args.ar_model_path:
        return None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ppl_script = os.path.join(script_dir, "ppl_under_ar.py")
    cmd = [
        sys.executable,
        ppl_script,
        "--model_path", args.model_path,
        "--modality", args.modality,
        "--experiment", args.experiment,
        "--model_name_or_path", args.ar_model_path,
        "--input_text", generated_path,
        "--mode", "eval",
    ]
    subprocess.run(cmd, check=True)
    model_dir = os.path.split(args.model_path)[0]
    if "infill" in generated_path:
        json_path = os.path.join(model_dir, "infill_score_decode.json")
    elif "ema" in args.model_path:
        json_path = os.path.join(model_dir, "ema_score_decode.json")
    else:
        json_path = os.path.join(model_dir, "score_decode.json")
    result = _safe_read_json(json_path)
    if result is None:
        return None
    if "score_decode" in result:
        result["ar_perplexity"] = math.exp(result["score_decode"])
    return result


def main():
    set_seed(101)
    args = create_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dist_util.setup_dist()
    logger.configure(dir=args.out_dir)

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)
    args.sigma_small = True

    # args.diffusion_steps = 200 #500  # DEBUG

    if args.experiment == 'random1': args.experiment = 'random'
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    # diffusion.rescale_timesteps = False  # DEBUG --> REMOVE
    print(diffusion.rescale_timesteps, 'a marker for whether we are in the debug mode')
    model.to(dist_util.dev())
    model.eval() # DEBUG



    if args.experiment_mode == 'conditional_gen':
        from improved_diffusion.text_datasets import load_data_text
        model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                        os.path.split(args.model_path)[0])
        print('conditional generation mode --> load data')
        rev_tokenizer = {v: k for k, v in tokenizer.items()}

        # print(rev_tokenizer)
        data = load_data_text(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            data_args=args,
            model=model2,
            deterministic=True,
            task_mode=args.modality,
            padding_mode=args.padding_mode,  # block, pad
            split=args.split,
            load_vocab=rev_tokenizer,
        )

    model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                    os.path.split(args.model_path)[0])
    if args.training_mode.startswith('e2e'):
        print('e2e, load the right model embeddings', '*'*80)
        model2.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

    logger.log("sampling...")
    all_images = []
    all_labels = []
    print(args.num_samples)
    model3 = get_weights(model2, args)
    model3 = model3.to(dist_util.dev())

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    total_target = args.num_samples * args.mbr_sample
    partial_out_path = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.partial.npy")
    progress_path = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.progress.json")
    partial_arr = None
    if dist.get_rank() == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        partial_shape = (total_target, args.image_size ** 2, args.in_channel)
        partial_arr = np.lib.format.open_memmap(
            partial_out_path, mode="w+", dtype=np.float32, shape=partial_shape
        )
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "partial_file": partial_out_path,
                    "written_samples": 0,
                    "target_samples": total_target,
                    "completed": False,
                },
                f,
                indent=2,
            )
        logger.log(f"streaming partial samples to {partial_out_path}")

    generated_so_far = 0
    batch_idx = 0
    sample_start_time = time.time()
    while generated_so_far < total_target:
        batch_idx += 1
        batch_start_time = time.time()
        model_kwargs = {}
        if args.experiment_mode == 'conditional_gen':
            batch, model_kwargs = next(data)
            model_kwargs.pop('input_ids')
            if args.mbr_sample > 1:
                model_kwargs = {k: v.to(dist_util.dev()).repeat_interleave(args.mbr_sample, dim=0) for k, v in model_kwargs.items()}
            else:
                model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
            print([(k, v.shape) for (k,v) in model_kwargs.items()])
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        if args.model_arch == '1d-unet':
            if args.mbr_sample > 1 and args.experiment_mode == 'conditional_gen':
                sample_shape = (args.batch_size * args.mbr_sample, args.in_channel, args.image_size ** 2)
            else:
                sample_shape = (args.batch_size,  args.in_channel, args.image_size ** 2)
        else:
            if args.mbr_sample > 1 and args.experiment_mode == 'conditional_gen':
                sample_shape = (args.batch_size * args.mbr_sample, args.image_size ** 2, args.in_channel)
            else:
                sample_shape = (args.batch_size, args.image_size ** 2, args.in_channel)
        print(sample_shape)
        sample = sample_fn(
            model,
            sample_shape,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model3) if args.clamp == 'clamp' else None,
            model_kwargs=model_kwargs,
            top_p =args.top_p,
        )

        if args.model_arch == '1d-unet':
            print(sample.shape)
            sample = sample.permute(0, 2, 1)
        print(sample.shape)

        # if diffusion.training_mode.startswith('e2e'):
        #     word_lst_e2e = []
        #     print('decoding for e2e', )
        #     print(sample.shape)
        #     x_t = sample
        #     if args.model_arch == 'conv-unet':
        #         reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
        #     else:
        #         reshaped_x_t = x_t
        #     logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        #     cands = th.topk(logits, k=1, dim=-1)
        #     sample = cands.indices
        #     tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
        #     for seq in cands.indices:
        #         if isinstance(tokenizer, dict):
        #             tokens = " ".join([tokenizer[x[0].item()] for x in seq])
        #         else:
        #             tokens = tokenizer.decode(seq.squeeze(-1))
        #         word_lst_e2e.append(tokens)


        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        batch_np = np.concatenate([sample.cpu().numpy() for sample in gathered_samples], axis=0)
        remaining = total_target - generated_so_far
        take_count = min(remaining, batch_np.shape[0])
        if take_count <= 0:
            break
        batch_np = batch_np[:take_count]
        all_images.append(batch_np)

        if dist.get_rank() == 0 and partial_arr is not None:
            partial_arr[generated_so_far:generated_so_far + take_count] = batch_np.astype(np.float32, copy=False)
            partial_arr.flush()
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "partial_file": partial_out_path,
                        "written_samples": generated_so_far + take_count,
                        "target_samples": total_target,
                        "completed": False,
                    },
                    f,
                    indent=2,
                )

        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        generated_so_far += take_count
        elapsed = time.time() - sample_start_time
        per_batch = time.time() - batch_start_time
        avg_per_batch = elapsed / max(batch_idx, 1)
        remaining_batches = max(0, math.ceil((total_target - generated_so_far) / args.batch_size))
        eta_sec = remaining_batches * avg_per_batch
        progress_pct = 100.0 * generated_so_far / max(total_target, 1)
        logger.log(
            f"created {generated_so_far}/{total_target} samples "
            f"({progress_pct:.1f}%) | batch_time={per_batch:.1f}s | eta={eta_sec/60:.1f}m"
        )

    arr = np.concatenate(all_images, axis=0)
    print(arr.shape, 'full shape')
    arr = arr[:total_target]

    if diffusion.training_mode.startswith('e2e'):
        word_lst_e2e = []
        print('decoding for e2e', )
        print(arr.shape)
        x_t = th.tensor(arr).to(dist_util.dev())
        if args.model_arch == 'conv-unet':
            reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
        else:
            reshaped_x_t = x_t
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)
        sample = cands.indices
        tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
        for seq in cands.indices:
            if isinstance(tokenizer, dict):
                tokens = " ".join([tokenizer[x[0].item()] for x in seq])
            else:
                tokens = tokenizer.decode(seq.squeeze(-1))
            word_lst_e2e.append(tokens)

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.npz")
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "partial_file": partial_out_path,
                    "written_samples": int(arr.shape[0]),
                    "target_samples": total_target,
                    "completed": True,
                    "final_npz": out_path,
                },
                f,
                indent=2,
            )

    dist.barrier()
    logger.log("sampling complete")

    if args.verbose == 'yes':
        logger.log('decode by rounding. ')
        print('load_models')
        if diffusion.training_mode.startswith('e2e'):
            word_lst = word_lst_e2e
        else:
            set_seed(101)
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                           os.path.split(args.model_path)[0])
            print('rounding')
            word_lst = rounding_func(args.experiment, arr, model, tokenizer,
                                     emb_scale_factor=args.emb_scale_factor)

        out_path2 = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.txt")
        fout = open(out_path2, 'w')

        for (xx) in zip( word_lst):
            # print('---' * 30)
            # print(tokenizer.decode(gg.tolist()))
            # print('xx' * 30)
            print(xx[0], file=fout)
            # print('---' * 30)
        fout.close()
        print(f'written the decoded output to {out_path2}')

        ##############
        out_path2 = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.json")
        fout = open(out_path2, 'w')
        for (xx) in zip(word_lst):
            print(json.dumps(xx), file=fout)
        fout.close()
        print(f'written the decoded output to {out_path2}')

        metrics_path = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.metrics.json")
        metrics = _compute_generation_metrics(word_lst, args)
        if args.compute_all_metrics:
            args.compute_valid_bpd = True
            args.compute_train_bpd = True
            args.compute_ar_ppl = True

        if args.compute_valid_bpd:
            try:
                nll_result = _run_bpd_eval(args, "valid")
                if nll_result is not None:
                    for key, value in nll_result.items():
                        metrics[f"valid_{key}"] = value
            except Exception as e:
                metrics["valid_bpd_error"] = str(e)
        if args.compute_train_bpd:
            try:
                nll_result = _run_bpd_eval(args, "train")
                if nll_result is not None:
                    for key, value in nll_result.items():
                        metrics[f"train_{key}"] = value
            except Exception as e:
                metrics["train_bpd_error"] = str(e)
        if args.compute_ar_ppl:
            try:
                ar_result = _run_ar_eval(args, out_path2)
                if ar_result is not None:
                    for key, value in ar_result.items():
                        metrics[f"ar_{key}"] = value
            except Exception as e:
                metrics["ar_ppl_error"] = str(e)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f'written generation metrics to {metrics_path}')




def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=50,#10000,
        batch_size=64,
        use_ddim=False,
        mbr_sample=1,
        model_path="",
        model_arch='conv-unet',
        verbose='yes',
        out_dir="generation_outputs"
    )
    text_defaults = dict(modality='text',
                         dataset_name='wikitext',
                         dataset_config_name='wikitext-2-raw-v1',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress', model_arch='trans-unet',
                         preprocessing_num_workers=1,
                         emb_scale_factor=1.0, top_p=-1., split='valid', clamp='clamp',
                         compute_all_metrics=False, compute_valid_bpd=False, compute_train_bpd=False,
                         compute_ar_ppl=False, ar_model_path="",
                         nll_batch_size=64, nll_num_samples=256, nll_out_dir="scores", nll_clamp="clamp")
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    # defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
