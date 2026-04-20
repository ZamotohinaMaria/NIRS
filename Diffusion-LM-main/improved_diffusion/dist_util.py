"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 1 #8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    comm = MPI.COMM_WORLD
    # Windows builds commonly do not include NCCL even when CUDA is available.
    # Use NCCL only when it is actually available; otherwise fallback to gloo.
    use_nccl = th.cuda.is_available() and dist.is_nccl_available()
    backend = "nccl" if use_nccl else "gloo"

    # For single-machine runs, binding to localhost is the most robust default.
    # Allow explicit override via env var when needed.
    hostname = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    try:
        dist.init_process_group(backend=backend, init_method="env://")
    except RuntimeError:
        # Final fallback for environments where NCCL was selected but unavailable
        # at runtime for any reason.
        if backend != "gloo":
            dist.init_process_group(backend="gloo", init_method="env://")
        else:
            raise


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
