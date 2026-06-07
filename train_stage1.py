import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from datasets.stage1_dataset import build_stage1_loader
from models.stage1_model import build_stage1_model
from utils import (
    Stage1AuxLoss,
    build_logger,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    load_config,
    save_config,
    save_history,
    set_seed,
    unwrap_model,
)
from utils.utils_torch_load import torch_load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage1 gyro model.")
    parser.add_argument("--config", default="config/stage1.yaml")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def new_run_dir(config, resume=None):
    if resume:
        return Path(resume).parent.parent if Path(resume).is_file() else Path(resume)
    exp_cfg = config["experiment"]
    if exp_cfg.get("output_dir"):
        return Path(exp_cfg["output_dir"])
    result_root = Path(exp_cfg.get("result_root", "result"))
    run_prefix = exp_cfg.get("run_prefix", "stage1")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = result_root / f"{run_prefix}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = result_root / f"{run_prefix}_{timestamp}_{suffix}"
        suffix += 1
    return run_dir


def resolve_resume_checkpoint(resume):
    if not resume:
        return None
    resume_path = Path(resume)
    if resume_path.is_dir():
        candidates = [
            resume_path / "checkpoints" / "latest.pt",
            resume_path / "checkpoints" / "last.pt",
            resume_path / "last.pt",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Missing checkpoint under resume directory: {candidates[0]}")
    if resume_path.is_file():
        return resume_path
    raise FileNotFoundError(f"resume must be a .pt file or run directory: {resume}")


def build_optimizer(config, parameters):
    optim_cfg = config.get("optimizer", {})
    name = optim_cfg.get("name", "adamw").lower()
    lr = float(optim_cfg.get("lr", 1e-4))
    betas = tuple(float(v) for v in optim_cfg.get("betas", [0.9, 0.999]))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer.name: {name}")


def build_scheduler(config, optimizer, steps_per_epoch, epochs):
    sched_cfg = config.get("scheduler", {})
    name = sched_cfg.get("name", "onecycle").lower()
    if name in ("none", "null"):
        return None, "none"
    if name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(sched_cfg.get("max_lr", config.get("optimizer", {}).get("lr", 1e-4))),
            epochs=int(epochs),
            steps_per_epoch=int(steps_per_epoch),
            pct_start=float(sched_cfg.get("pct_start", 0.3)),
            div_factor=float(sched_cfg.get("div_factor", 25.0)),
            final_div_factor=float(sched_cfg.get("final_div_factor", 10000.0)),
        )
        return scheduler, "iteration"
    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get("t_max") or epochs),
            eta_min=float(sched_cfg.get("eta_min", 1e-7)),
        )
        return scheduler, "epoch"
    raise ValueError(f"Unknown scheduler.name: {name}")


def reduce_metrics(metrics, device):
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return metrics
    keys = list(metrics.keys())
    values = torch.tensor([metrics[key] for key in keys], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return {key: float(value.detach().cpu()) for key, value in zip(keys, values)}


def metrics_from_sums(metrics):
    count = max(1.0, metrics["count"])
    result = {
        "loss": metrics["loss_sum"] / count,
        "gyro_loss": metrics["gyro_loss_sum"] / count,
        "aux_loss": metrics["aux_loss_sum"] / count,
        "mae": metrics["mae_sum"] / count,
        "rmse": metrics["rmse_sum"] / count,
        "count": int(metrics["count"]),
    }
    grad_count = max(1.0, metrics.get("grad_norm_count", 0.0))
    if metrics.get("grad_norm_count", 0.0) > 0:
        result["grad_norm"] = metrics.get("grad_norm_sum", 0.0) / grad_count
    for key, value in metrics.items():
        if key not in (
            "loss_sum",
            "gyro_loss_sum",
            "aux_loss_sum",
            "mae_sum",
            "rmse_sum",
            "count",
            "grad_norm_sum",
            "grad_norm_count",
        ):
            result[key] = int(value)
    return result


def update_running(running, loss_metrics, batch_size):
    running["loss_sum"] += float(loss_metrics["loss"].detach().cpu()) * batch_size
    running["gyro_loss_sum"] += float(loss_metrics["gyro_loss"].detach().cpu()) * batch_size
    running["aux_loss_sum"] += float(loss_metrics["aux_loss"].detach().cpu()) * batch_size
    running["mae_sum"] += float(loss_metrics["mae"].detach().cpu()) * batch_size
    running["rmse_sum"] += float(loss_metrics["rmse"].detach().cpu()) * batch_size
    running["count"] += batch_size


def empty_metrics():
    return {
        "loss_sum": 0.0,
        "gyro_loss_sum": 0.0,
        "aux_loss_sum": 0.0,
        "mae_sum": 0.0,
        "rmse_sum": 0.0,
        "count": 0.0,
        "pred_nonfinite": 0.0,
        "target_nonfinite": 0.0,
        "pose_nonfinite": 0.0,
        "loss_nonfinite": 0.0,
        "grad_norm_sum": 0.0,
        "grad_norm_count": 0.0,
        "grad_clipped": 0.0,
        "grad_nonfinite": 0.0,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, show_progress=False):
    was_training = model.training
    model.eval()
    metrics = empty_metrics()
    batches = tqdm(loader, desc="val", leave=False, disable=not show_progress)
    for batch in batches:
        image = batch["image"].to(device, non_blocking=True).float()
        target_gyro = batch["gyro"].to(device, non_blocking=True).float()
        timestamp_window = batch["timestamp_window"].to(device, non_blocking=True).float()
        focal_length = batch["focal_length"].to(device, non_blocking=True).float()
        outputs = model(image, focal_length=focal_length, return_aux=True)
        loss, loss_metrics, _ = criterion(outputs, target_gyro, timestamp_window)
        batch_size = image.shape[0]

        pred_gyro = outputs["gyro"]
        pred_finite = torch.isfinite(pred_gyro).flatten(1).all(dim=1)
        target_finite = torch.isfinite(target_gyro).flatten(1).all(dim=1)
        metrics["pred_nonfinite"] += int((~pred_finite).sum().detach().cpu())
        metrics["target_nonfinite"] += int((~target_finite).sum().detach().cpu())
        if "pose" in outputs:
            pose_finite = torch.isfinite(outputs["pose"]).flatten(1).all(dim=1)
            metrics["pose_nonfinite"] += int((~pose_finite).sum().detach().cpu())
        if not torch.isfinite(loss):
            metrics["loss_nonfinite"] += batch_size

        update_running(metrics, loss_metrics, batch_size)
        if show_progress:
            batches.set_postfix(metrics_from_sums(metrics))

    if was_training:
        model.train()
    return metrics_from_sums(reduce_metrics(metrics, device))


def save_checkpoint(path, config, model, optimizer, scheduler, epoch, best_val_loss, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_val_loss": float(best_val_loss),
            "history": history,
            "config": config,
        },
        path,
    )


def save_best_metrics(run_dir, epoch, metrics):
    data = {
        "epoch": int(epoch),
        "best_val_loss": float(metrics["loss"]),
        "mae_at_best_loss": float(metrics["mae"]),
        "rmse_at_best_loss": float(metrics["rmse"]),
        "aux_loss_at_best_loss": float(metrics["aux_loss"]),
        "count": int(metrics.get("count", 0)),
    }
    path = Path(run_dir) / "best_metrics.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch_load_checkpoint(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    if checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def main():
    args = parse_args()
    config = load_config(args.config)
    resume = args.resume or config.get("train", {}).get("resume")
    if resume and Path(resume).is_dir() and (Path(resume) / "config.yaml").exists():
        config = load_config(Path(resume) / "config.yaml")
        config.setdefault("train", {})["resume"] = resume
    set_seed(config.get("train", {}).get("seed"))

    device, distributed = init_distributed(config.get("distributed", {}))
    resume = args.resume or config.get("train", {}).get("resume")
    resume_checkpoint = resolve_resume_checkpoint(resume)
    run_dir = new_run_dir(config, resume=resume)
    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, run_dir / "config.yaml")

    logger = build_logger(
        config["experiment"]["name"],
        run_dir / "log.txt" if is_main_process() else None,
        enabled=is_main_process(),
    )
    logger.info(f"run_dir={run_dir}")
    logger.info(f"device={device}, distributed={distributed}")

    train_dataset, train_loader, train_sampler = build_stage1_loader(
        config,
        split=config["dataset"].get("split", "train"),
        distributed=distributed,
        device=device,
        is_train=True,
    )
    val_loader = None
    if config.get("validation", {}).get("enabled", True):
        _, val_loader, _ = build_stage1_loader(
            config,
            split=config["validation"].get("split", "val"),
            distributed=False,
            device=device,
            is_train=False,
        )

    model = build_stage1_model(config).to(device)
    if is_main_process() and hasattr(model, "pretrained_report"):
        logger.info(f"pretrained={model.pretrained_report}")
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    loss_cfg = config.get("loss", {})
    criterion = Stage1AuxLoss(
        gyro_loss=loss_cfg.get("gyro_loss", config.get("train", {}).get("loss", "smooth_l1")),
        aux_loss=loss_cfg.get("aux_loss", "smooth_l1"),
        aux_weight=loss_cfg.get("aux_weight", 0.05),
        default_dt=config.get("time", {}).get("default_dt", 1.0 / 240.0),
        target_norm_weight=loss_cfg.get("target_norm_weight", 0.0),
        target_norm_reference=loss_cfg.get("target_norm_reference", 2.5),
        target_norm_max_weight=loss_cfg.get("target_norm_max_weight", 3.0),
    ).to(device)
    optimizer = build_optimizer(config, model.parameters())
    epochs = int(config["train"].get("epochs", 50))
    scheduler, scheduler_step = build_scheduler(config, optimizer, len(train_loader), epochs)

    start_epoch = 0
    best_val_loss = float("inf")
    history = []
    if resume_checkpoint:
        checkpoint = load_checkpoint(resume_checkpoint, model, optimizer, scheduler, device)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        history = list(checkpoint.get("history", []))
        logger.info(f"resumed_from={resume_checkpoint}, start_epoch={start_epoch}")

    logger.info(
        f"train_samples={len(train_dataset)}, steps_per_epoch={len(train_loader)}, "
        f"epochs={epochs}, batch_size={config['dataset'].get('batch_size', 8)}"
    )

    log_interval = int(config["train"].get("log_interval", 100))
    checkpoint_interval = int(config["train"].get("checkpoint_interval", 5))
    val_interval = int(config.get("validation", {}).get("interval", 1))
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 0.0) or 0.0)
    skip_nonfinite_loss = bool(config["train"].get("skip_nonfinite_loss", True))
    for epoch in range(start_epoch, epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        running = empty_metrics()
        progress = tqdm(
            train_loader,
            desc=f"train epoch {epoch + 1}/{epochs}",
            disable=not is_main_process(),
        )
        for step, batch in enumerate(progress, start=1):
            image = batch["image"].to(device, non_blocking=True).float()
            target_gyro = batch["gyro"].to(device, non_blocking=True).float()
            timestamp_window = batch["timestamp_window"].to(device, non_blocking=True).float()
            focal_length = batch["focal_length"].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(image, focal_length=focal_length, return_aux=True)
            loss, loss_metrics, _ = criterion(outputs, target_gyro, timestamp_window)
            batch_size = image.shape[0]

            pred_gyro = outputs["gyro"]
            pred_finite = torch.isfinite(pred_gyro).flatten(1).all(dim=1)
            target_finite = torch.isfinite(target_gyro).flatten(1).all(dim=1)
            running["pred_nonfinite"] += int((~pred_finite).sum().detach().cpu())
            running["target_nonfinite"] += int((~target_finite).sum().detach().cpu())
            if "pose" in outputs:
                pose_finite = torch.isfinite(outputs["pose"]).flatten(1).all(dim=1)
                running["pose_nonfinite"] += int((~pose_finite).sum().detach().cpu())

            if not torch.isfinite(loss):
                running["loss_nonfinite"] += batch_size
                optimizer.zero_grad(set_to_none=True)
                if skip_nonfinite_loss:
                    if is_main_process() and (step % log_interval == 0 or step == len(train_loader)):
                        metrics = metrics_from_sums(running)
                        progress.set_postfix(loss=metrics["loss"], mae=metrics["mae"])
                    continue

            loss.backward()
            if grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_norm,
                    error_if_nonfinite=False,
                )
                if torch.isfinite(grad_norm):
                    grad_norm_value = float(grad_norm.detach().cpu())
                    running["grad_norm_sum"] += grad_norm_value
                    running["grad_norm_count"] += 1.0
                    if grad_norm_value > grad_clip_norm:
                        running["grad_clipped"] += 1.0
                else:
                    running["grad_nonfinite"] += batch_size
                    optimizer.zero_grad(set_to_none=True)
                    if skip_nonfinite_loss:
                        continue
            optimizer.step()
            if scheduler is not None and scheduler_step == "iteration":
                scheduler.step()

            update_running(running, loss_metrics, batch_size)

            if is_main_process() and (step % log_interval == 0 or step == len(train_loader)):
                metrics = metrics_from_sums(running)
                lr = optimizer.param_groups[0]["lr"]
                progress.set_postfix(
                    loss=metrics["loss"],
                    mae=metrics["mae"],
                    aux=metrics["aux_loss"],
                    lr=lr,
                    grad=metrics.get("grad_norm", 0.0),
                )

        if scheduler is not None and scheduler_step == "epoch":
            scheduler.step()

        train_metrics = metrics_from_sums(reduce_metrics(running, device))
        train_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append({"split": "train", "epoch": epoch + 1, **train_metrics})
        if is_main_process():
            logger.info(
                f"epoch={epoch + 1}/{epochs} train_loss={train_metrics['loss']:.6f} "
                f"train_gyro_loss={train_metrics['gyro_loss']:.6f} "
                f"train_aux_loss={train_metrics['aux_loss']:.6f} "
                f"train_mae={train_metrics['mae']:.6f}"
            )

        should_validate = val_loader is not None and ((epoch + 1) % val_interval == 0)
        if should_validate:
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                show_progress=is_main_process(),
            )
            history.append({"split": "val", "epoch": epoch + 1, **val_metrics})
            if is_main_process():
                nonfinite_msg = ""
                if any(
                    int(val_metrics.get(key, 0)) > 0
                    for key in ("pred_nonfinite", "target_nonfinite", "pose_nonfinite", "loss_nonfinite")
                ):
                    nonfinite_msg = (
                        f" pred_nonfinite={int(val_metrics.get('pred_nonfinite', 0))}"
                        f" target_nonfinite={int(val_metrics.get('target_nonfinite', 0))}"
                        f" pose_nonfinite={int(val_metrics.get('pose_nonfinite', 0))}"
                        f" loss_nonfinite={int(val_metrics.get('loss_nonfinite', 0))}"
                    )
                logger.info(
                    f"epoch={epoch + 1}/{epochs} val_loss={val_metrics['loss']:.6f} "
                    f"val_gyro_loss={val_metrics['gyro_loss']:.6f} "
                    f"val_aux_loss={val_metrics['aux_loss']:.6f} "
                    f"val_mae={val_metrics['mae']:.6f}{nonfinite_msg}"
                )
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    save_checkpoint(
                        run_dir / "checkpoints" / "best.pt",
                        config,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        best_val_loss,
                        history,
                    )
                    save_best_metrics(run_dir, epoch + 1, val_metrics)
                    logger.info(f"saved best checkpoint | loss={best_val_loss:.6f}")

        if is_main_process():
            save_history(history, run_dir)

        if is_main_process() and ((epoch + 1) % checkpoint_interval == 0 or epoch + 1 == epochs):
            save_checkpoint(
                run_dir / "checkpoints" / "latest.pt",
                config,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                history,
            )

    cleanup_distributed()


if __name__ == "__main__":
    main()
