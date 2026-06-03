import os

import torch


def init_distributed(enabled=False):
    if not enabled:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), False

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), False

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return torch.device(f"cuda:{local_rank}"), True


def cleanup_distributed():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model
