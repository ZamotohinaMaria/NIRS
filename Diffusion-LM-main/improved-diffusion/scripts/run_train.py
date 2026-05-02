import sys
import os
import argparse
import tempfile
import subprocess
import shlex
import importlib.util
from datetime import datetime

def ensure_dir_writable(path, name):
    if path is None or str(path).strip() == "":
        raise ValueError(f"{name} is empty")
    os.makedirs(path, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix="._write_test_", delete=True):
            pass
    except Exception as e:
        raise PermissionError(f"{name} is not writable: {path}. Error: {e}")


def ensure_modules_available(module_names):
    missing = []
    for mod in module_names:
        if importlib.util.find_spec(mod) is None:
            missing.append(mod)
    if missing:
        raise ModuleNotFoundError(
            "Missing Python modules: "
            + ", ".join(missing)
            + ". Install them before training, e.g. "
            + f"pip install {' '.join(missing)}"
        )


def ensure_huggingface_hub_compat():
    try:
        import huggingface_hub as hf_hub
    except Exception as e:
        raise ModuleNotFoundError(
            f"Failed to import huggingface_hub: {e}. "
            "Install compatible version: pip install huggingface_hub==0.4.0"
        )
    if not hasattr(hf_hub, "HfFolder"):
        ver = getattr(hf_hub, "__version__", "unknown")
        raise RuntimeError(
            "Incompatible huggingface_hub version detected: "
            f"{ver}. This project expects API with HfFolder.\n"
            "Run: pip install huggingface_hub==0.4.0"
        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--experiment', type=str, default='random', help='no-rep=gpt2gen, no-zipfs, has-rep=regular, rm-window-rep')

    parser.add_argument('--model_arch', type=str, default='conv-unet', help='')
    parser.add_argument('--modality', type=str, default='synth', help='')
    parser.add_argument('--noise_schedule', type=str, default='cosine', help='')
    parser.add_argument('--loss_type', type=str, default='Lsimple', help='')
    parser.add_argument('--dropout', type=str, default='0.1', help='')
    parser.add_argument('--weight_decay', type=str, default=0.0, help='')

    parser.add_argument('--image_size', type=int, default=13, help='')
    parser.add_argument('--hidden_size', type=int, default=128, help='')
    parser.add_argument('--in_channel', type=int, default=16, help='')
    parser.add_argument('--m', type=int, default=3, help='')
    parser.add_argument('--k', type=int, default=32, help='')
    parser.add_argument('--lr_anneal_steps', type=int, default=40000, help='')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='')

    parser.add_argument('--lr', type=float, default=1e-04, help='')
    parser.add_argument('--bsz', type=int, default=64, help='')
    parser.add_argument('--diff_steps', type=int, default=4000, help='')
    parser.add_argument('--save_interval', type=int, default=50000, help='save checkpoint every N steps')
    parser.add_argument('--eval_interval', type=int, default=2000, help='evaluate every N steps')
    parser.add_argument('--eval_num_batches', type=int, default=4, help='validation batches per evaluation')
    parser.add_argument('--early_stop_patience_eval', type=int, default=5, help='stop if no eval improvement for N checks')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0, help='minimum eval improvement to reset patience')
    parser.add_argument('--padding_mode', type=str, default='block', help='')
    parser.add_argument('--seed', type=int, default=101, help='') # old is 42
    parser.add_argument('--checkpoint_root', type=str, default='runs/models', help='root directory for runs')

    parser.add_argument('--notes', type=str, default=None, help='')
    parser.add_argument('--submit', type=str, default='no', help='')
    parser.add_argument('--use_big', type=str, default='no', help='')
    parser.add_argument('--app', type=str, default='', help='')


    args = parser.parse_args()

    # Fast dependency check to fail before long training startup.
    ensure_modules_available([
        "sacremoses",
        "blobfile",
        "mpi4py",
        "datasets",
        "spacy",
        "wandb",
    ])
    ensure_huggingface_hub_compat()

    folder_name = args.checkpoint_root
    ensure_dir_writable(folder_name, "checkpoint_root")



    if args.loss_type == 'Lsimple':
        train_setup = " --use_kl False --learn_sigma False "
    elif args.loss_type == 'Lhybrid':
        train_setup = " --use_kl False --learn_sigma True "
    elif args.loss_type == 'Lvlb':
        train_setup = " --use_kl True --learn_sigma True "
    else:
        assert False


    if args.experiment == 'random':
        exp_m = 'rand'
    if args.experiment == 'random1':
        exp_m = 'pred'
    elif args.experiment == 'gpt2_pre_compress':
        exp_m = 'emb'
    elif args.experiment == 'glove':
        exp_m = 'glo'


    if args.modality == 'synth' or args.modality =='synth_trans':

        Model_FILE = f"diff_{args.modality}{args.k}_{args.m}_{exp_m}{args.in_channel}_{args.model_arch}_lr{args.lr}" \
                     f"_{args.diff_steps}_{args.noise_schedule}_{args.loss_type}_h{args.hidden_size}" \
                     f"_s{args.num_res_blocks}_sd{args.seed}"

    elif args.modality == 'roc' or args.modality == 'roc-aug' or args.modality == 'book' \
            or args.modality == 'simple-wiki' or args.modality == 'e2e-tgt' or args.modality == 'e2e'\
            or args.modality == 'yelp' or args.modality == 'commonGen' or args.modality == 'commonGen-aug' \
            or args.modality == 'malbehav':

        Model_FILE = f"diff_{args.modality}_{args.padding_mode}_{exp_m}{args.in_channel}_{args.model_arch}_lr{args.lr}_{args.weight_decay}" \
                     f"_{args.diff_steps}_{args.noise_schedule}_{args.loss_type}_h{args.hidden_size}" \
                     f"_s{args.num_res_blocks}_d{args.dropout}_sd{args.seed}"


    elif args.modality == 'pos':
        Model_FILE = f"diff_{args.modality}{0}_{0}_{exp_m}{args.in_channel}_{args.model_arch}_lr{args.lr}" \
                     f"_{args.diff_steps}_{args.noise_schedule}_{args.loss_type}_h{args.hidden_size}" \
                     f"_s{args.num_res_blocks}_sd{args.seed}"

    elif args.modality == 'image' or args.modality == 'permuted_image':
        Model_FILE = f"diff_{args.modality}{0}_{0}_{exp_m}{args.in_channel}_{args.model_arch}_lr{args.lr}" \
                     f"_{args.diff_steps}_{args.noise_schedule}_{args.loss_type}_h{args.hidden_size}" \
                     f"_s{args.num_res_blocks}_sd{args.seed}"

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    arch_short = {"transformer": "trf", "conv-unet": "cunet", "1d-unet": "u1d"}.get(args.model_arch, args.model_arch)
    prefix = args.notes if args.notes else args.modality
    run_name = (
        f"{prefix}_{arch_short}"
        f"_ic{args.in_channel}"
        f"_bs{args.bsz}"
        f"_s{args.seed}"
    )
    run_name += f"_{ts}"
    Model_FILE = os.path.join(folder_name, run_name)
    ensure_dir_writable(Model_FILE, "run_dir")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    train_script = os.path.join(script_dir, "train.py")

    command_list = [
        sys.executable, train_script,
        "--checkpoint_path", Model_FILE,
        "--model_arch", args.model_arch,
        "--modality", args.modality,
        "--save_interval", str(args.save_interval),
        "--lr", str(args.lr),
        "--eval_interval", str(args.eval_interval),
        "--eval_num_batches", str(args.eval_num_batches),
        "--early_stop_patience_eval", str(args.early_stop_patience_eval),
        "--early_stop_min_delta", str(args.early_stop_min_delta),
        "--batch_size", str(args.bsz),
        "--diffusion_steps", str(args.diff_steps),
        "--noise_schedule", args.noise_schedule,
        "--image_size", str(args.image_size),
        "--num_channels", str(args.hidden_size),
        "--seed", str(args.seed),
        "--dropout", str(args.dropout),
        "--in_channel", str(args.in_channel),
        "--out_channel", str(args.in_channel),
        "--padding_mode", args.padding_mode,
        "--experiment", args.experiment,
        "--lr_anneal_steps", str(args.lr_anneal_steps),
        "--weight_decay", str(args.weight_decay),
        "--num_res_blocks", str(args.num_res_blocks),
    ]

    if args.loss_type == 'Lsimple':
        command_list.extend(["--use_kl", "False", "--learn_sigma", "False"])
    elif args.loss_type == 'Lhybrid':
        command_list.extend(["--use_kl", "False", "--learn_sigma", "True"])
    elif args.loss_type == 'Lvlb':
        command_list.extend(["--use_kl", "True", "--learn_sigma", "True"])

    if args.app.strip():
        command_list.extend(shlex.split(args.app))

    env = os.environ.copy()
    env["OPENAI_LOGDIR"] = Model_FILE
    env["TOKENIZERS_PARALLELISM"] = "false"

    with open(os.path.join(Model_FILE, 'train_command.cmd'), 'w') as f:
        print(" ".join(command_list), file=f)

    with open(os.path.join(Model_FILE, 'train_command.txt'), 'w') as f:
        print(" ".join(command_list), file=f)

    print(" ".join(command_list))
    if args.submit == 'no':
        subprocess.run(command_list, env=env, cwd=project_dir, check=False)
    # #
    elif args.submit == 'yes':
        print("submit=yes is remapped to local run in this Windows-default build.")
        subprocess.run(command_list, env=env, cwd=project_dir, check=False)
