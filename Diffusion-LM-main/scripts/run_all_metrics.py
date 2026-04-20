import argparse
import json
import math
import os
import subprocess
import sys


def _safe_read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_bpd_eval(args, split):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nll_script = os.path.join(script_dir, "nll.py")
    os.makedirs(args.nll_out_dir, exist_ok=True)
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
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    model_dir = os.path.split(args.model_path)[0]
    if "ema" in args.model_path:
        json_path = os.path.join(model_dir, f"ema_score_{split}_nll.json")
    elif args.nll_clamp == "noclamp":
        json_path = os.path.join(model_dir, f"score_{split}_nll_noclamp.json")
    else:
        json_path = os.path.join(model_dir, f"score_{split}_nll.json")
    return _safe_read_json(json_path)


def _run_ar_eval(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ppl_script = os.path.join(script_dir, "ppl_under_ar.py")
    cmd = [
        sys.executable,
        ppl_script,
        "--model_path", args.model_path,
        "--modality", args.modality,
        "--experiment", args.experiment,
        "--model_name_or_path", args.ar_model_path,
        "--input_text", args.generated_json,
        "--mode", "eval",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    model_dir = os.path.split(args.model_path)[0]
    if "infill" in args.generated_json:
        json_path = os.path.join(model_dir, "infill_score_decode.json")
    elif "ema" in args.model_path:
        json_path = os.path.join(model_dir, "ema_score_decode.json")
    else:
        json_path = os.path.join(model_dir, "score_decode.json")
    result = _safe_read_json(json_path)
    if result is not None and "score_decode" in result:
        result["ar_perplexity"] = math.exp(result["score_decode"])
    return result


def main():
    parser = argparse.ArgumentParser(description="Run all generation metrics in one command.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to diffusion checkpoint (.pt).")
    parser.add_argument("--generated_json", type=str, required=True, help="Path to generated *.json from text_sample.py")
    parser.add_argument("--ar_model_path", type=str, required=True, help="Path to AR model dir (save_pretrained format).")
    parser.add_argument("--modality", type=str, default="roc")
    parser.add_argument("--experiment", type=str, default="random")

    parser.add_argument("--nll_batch_size", type=int, default=64)
    parser.add_argument("--nll_num_samples", type=int, default=256)
    parser.add_argument("--nll_out_dir", type=str, default="scores")
    parser.add_argument("--nll_clamp", type=str, default="clamp", choices=["clamp", "noclamp"])

    parser.add_argument("--skip_valid_bpd", action="store_true")
    parser.add_argument("--skip_train_bpd", action="store_true")
    parser.add_argument("--skip_ar_ppl", action="store_true")
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if not os.path.exists(args.generated_json):
        raise FileNotFoundError(f"Generated json not found: {args.generated_json}")
    if not args.skip_ar_ppl and not os.path.exists(args.ar_model_path):
        raise FileNotFoundError(f"AR model path not found: {args.ar_model_path}")

    metrics = {
        "model_path": args.model_path,
        "generated_json": args.generated_json,
        "nll_out_dir": args.nll_out_dir,
    }

    if not args.skip_valid_bpd:
        try:
            result = _run_bpd_eval(args, "valid")
            if result is not None:
                for k, v in result.items():
                    metrics[f"valid_{k}"] = v
        except Exception as e:
            metrics["valid_bpd_error"] = str(e)

    if not args.skip_train_bpd:
        try:
            result = _run_bpd_eval(args, "train")
            if result is not None:
                for k, v in result.items():
                    metrics[f"train_{k}"] = v
        except Exception as e:
            metrics["train_bpd_error"] = str(e)

    if not args.skip_ar_ppl:
        try:
            result = _run_ar_eval(args)
            if result is not None:
                for k, v in result.items():
                    metrics[f"ar_{k}"] = v
        except Exception as e:
            metrics["ar_ppl_error"] = str(e)

    if args.out_json:
        out_path = args.out_json
    else:
        out_path = args.generated_json + ".all_metrics.json"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Written combined metrics to: {out_path}")


if __name__ == "__main__":
    main()

