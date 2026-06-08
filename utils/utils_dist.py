import os

import torch


def _distributed_options(config):
    if isinstance(config, dict):
        return bool(config.get("enabled", False)), str(
            config.get("backend", "auto")
        ).lower()
    return bool(config), "auto"


def _resolve_backend(backend):
    if backend != "auto":
        return backend
    if torch.cuda.is_available() and torch.distributed.is_nccl_available():
        return "nccl"
    return "gloo"


def init_distributed(config=False):
    enabled, backend = _distributed_options(config)
    if not enabled:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), False

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), False

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend=_resolve_backend(backend))
    return torch.device(f"cuda:{local_rank}"), True


def cleanup_distributed():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def reduce_mean_tensor(tensor):
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    tensor = tensor.detach().clone()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= torch.distributed.get_world_size()
    return tensor


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model
