from datetime import datetime
from pathlib import Path

import torch


def build_checkpoint_state(
    config,
    model,
    optimizer,
    scheduler,
    iteration,
    epoch,
    best_val_psnr,
    history,
    unwrap_fn,
):
    return {
        "iteration": int(iteration),
        "epoch": int(epoch),
        "total_iterations": int(config["train"]["total_iterations"]),
        "model": unwrap_fn(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val_psnr": float(best_val_psnr),
        "history": history,
        "config": config,
    }


def load_checkpoint_state(checkpoint, model, optimizer, scheduler, unwrap_fn, steps_per_epoch):
    model_state = checkpoint.get("model") or checkpoint.get("model_state_dict")
    if model_state is None:
        raise KeyError("checkpoint does not contain model state.")
    unwrap_fn(model).load_state_dict(model_state)

    optimizer_state = checkpoint.get("optimizer") or checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler") or checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    if "iteration" not in checkpoint and "epoch" in checkpoint:
        checkpoint["iteration"] = (int(checkpoint["epoch"]) + 1) * steps_per_epoch


def checkpoint_iteration(checkpoint, steps_per_epoch):
    if "iteration" in checkpoint:
        return int(checkpoint["iteration"])
    if "epoch" in checkpoint:
        return (int(checkpoint["epoch"]) + 1) * steps_per_epoch
    return 0


def prepare_run_dir(config, resume=None):
    exp_cfg = config.setdefault("experiment", {})
    result_root = Path(exp_cfg.get("result_root", "result"))
    run_prefix = exp_cfg.get("run_prefix", "run")
    output_dir = exp_cfg.get("output_dir")

    if resume:
        resume_path = Path(resume)
        if resume_path.is_dir():
            checkpoint_path = resume_path / "last.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            return resume_path, checkpoint_path
        if resume_path.is_file():
            run_dir = Path(output_dir) if output_dir else _new_run_dir(result_root, run_prefix)
            return run_dir, resume_path
        raise FileNotFoundError(f"resume must be a .pt file or run directory: {resume}")

    if output_dir:
        return Path(output_dir), None
    return _new_run_dir(result_root, run_prefix), None


def save_checkpoint(state, run_dir, name="last.pt"):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name
    torch.save(state, path)
    return path


def _new_run_dir(result_root, run_prefix):
    result_root = Path(result_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = result_root / f"{run_prefix}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = result_root / f"{run_prefix}_{timestamp}_{suffix}"
        suffix += 1
    return run_dir
