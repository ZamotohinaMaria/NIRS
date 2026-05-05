import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime


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


def setup_console_tee(log_dir, filename):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    sys.stdout = TeeStream(sys.stdout, log_path)
    sys.stderr = TeeStream(sys.stderr, log_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Malbehav sequences by label. "
            "Supports one label (0/1) or both labels in one run."
        )
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to diffusion checkpoint .pt")
    parser.add_argument(
        "--target_label",
        type=str,
        default="both",
        choices=["0", "1", "both"],
        help="0=benign, 1=malware, both=collect both labels simultaneously.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Target count per label (or for single label mode).",
    )
    parser.add_argument("--samples_per_round", type=int, default=512, help="Raw samples per round")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for text_sample.py")
    parser.add_argument("--top_p", type=float, default=-1.0, help="Nucleus sampling parameter")
    parser.add_argument("--seed_start", type=int, default=101, help="Start seed; incremented each round")
    parser.add_argument("--max_rounds", type=int, default=50, help="Maximum sampling rounds")
    parser.add_argument("--out_dir", type=str, default="generation_outputs", help="Final output directory")
    parser.add_argument("--tmp_dir", type=str, default="generation_outputs/_tmp_label_sampling", help="Temp directory")
    parser.add_argument("--text_sample_script", type=str, default="scripts/text_sample.py", help="Path to text_sample.py")
    return parser.parse_args()


def newest_txt_file(path):
    cands = glob.glob(os.path.join(path, "*.txt"))
    if len(cands) == 0:
        return None
    return sorted(cands, key=os.path.getmtime)[-1]


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


def main():
    args = parse_args()
    ensure_file_exists(args.model_path, "model_path")
    ensure_file_exists(args.text_sample_script, "text_sample_script")
    ensure_dir_writable(args.out_dir, "out_dir")
    ensure_dir_writable(args.tmp_dir, "tmp_dir")
    setup_console_tee(args.out_dir, "generation_by_label_console.log")

    collect_both = args.target_label == "both"
    targets = ["0", "1"] if collect_both else [args.target_label]
    accepted = {label: [] for label in targets}
    total_seen = 0

    for round_idx in range(args.max_rounds):
        round_seed = args.seed_start + round_idx
        for old in glob.glob(os.path.join(args.tmp_dir, "*")):
            if os.path.isdir(old):
                shutil.rmtree(old, ignore_errors=True)
            else:
                try:
                    os.remove(old)
                except OSError:
                    pass

        cmd = [
            sys.executable,
            args.text_sample_script,
            "--model_path", args.model_path,
            "--batch_size", str(args.batch_size),
            "--num_samples", str(args.samples_per_round),
            "--top_p", str(args.top_p),
            "--out_dir", args.tmp_dir,
            "--verbose", "yes",
            "--seed", str(round_seed),
        ]

        print("RUN:", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise RuntimeError(f"text_sample.py failed in round {round_idx}")

        txt_path = newest_txt_file(args.tmp_dir)
        if txt_path is None:
            raise RuntimeError("No .txt samples produced by text_sample.py")

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                seq = line.strip()
                if seq == "":
                    continue
                total_seen += 1
                toks = seq.split()
                if len(toks) == 0:
                    continue

                # text_sample.py usually emits: START LABEL_X ... END PAD PAD ...
                # Normalize to: LABEL_X ... (without START/END/PAD tail).
                if toks[0] == "START":
                    toks = toks[1:]
                if len(toks) == 0:
                    continue
                if "END" in toks:
                    toks = toks[:toks.index("END")]
                toks = [t for t in toks if t != "PAD"]
                if len(toks) == 0:
                    continue

                normalized = " ".join(toks)
                for label in targets:
                    target_prefix = f"LABEL_{label}"
                    if normalized.startswith(target_prefix + " ") or normalized == target_prefix:
                        if len(accepted[label]) < args.num_samples:
                            accepted[label].append(normalized)
                        break

        progress = " ".join(
            [f"label{label}={len(accepted[label])}/{args.num_samples}" for label in targets]
        )
        print(f"round={round_idx + 1}/{args.max_rounds} seed={round_seed} {progress} seen={total_seen}")
        if all(len(accepted[label]) >= args.num_samples for label in targets):
            break

    if not any(len(accepted[label]) > 0 for label in targets):
        raise RuntimeError("No accepted labeled samples found. Increase training/sampling budget.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for label in targets:
        rows = accepted[label][:args.num_samples]
        out_txt = os.path.join(args.out_dir, f"malbehav_label{label}_{len(rows)}_{stamp}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(row + "\n")
        acceptance_rate = (len(rows) / max(total_seen, 1)) * 100.0
        print(f"saved={out_txt}")
        print(
            f"label={label} accepted={len(rows)} total_seen={total_seen} "
            f"acceptance_rate={acceptance_rate:.2f}%"
        )


if __name__ == "__main__":
    main()
