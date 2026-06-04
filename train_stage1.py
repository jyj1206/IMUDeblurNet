import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from datasets.stage1_gyro_dataset import build_stage1_loader
from models.stage1_gyro_estimation_model import build_stage1_model
from utils import (
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/stage1_gyro.yaml")
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


def build_loss(name):
    name = name.lower()
    if name == "mse":
        return torch.nn.MSELoss()
    if name in ("l1", "mae"):
        return torch.nn.L1Loss()
    if name in ("smooth_l1", "huber"):
        return torch.nn.SmoothL1Loss()
    raise ValueError(f"Unknown train.loss: {name}")


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
    keys = ["loss_sum", "mae_sum", "count"]
    values = torch.tensor([metrics[key] for key in keys], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return {key: float(value.detach().cpu()) for key, value in zip(keys, values)}


def metrics_from_sums(metrics):
    count = max(1.0, metrics["count"])
    return {
        "loss": metrics["loss_sum"] / count,
        "mae": metrics["mae_sum"] / count,
        "count": int(metrics["count"]),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, show_progress=False):
    was_training = model.training
    model.eval()
    metrics = {"loss_sum": 0.0, "mae_sum": 0.0, "count": 0.0}
    batches = tqdm(loader, desc="val", leave=False, disable=not show_progress)
    for batch in batches:
        image = batch["image"].to(device, non_blocking=True).float()
        target_gyro = batch["gyro"].to(device, non_blocking=True).float()
        pred_gyro = model(image)["gyro"]
        loss = criterion(pred_gyro, target_gyro)
        batch_size = image.shape[0]
        metrics["loss_sum"] += float(loss.detach().cpu()) * batch_size
        metrics["mae_sum"] += float((pred_gyro - target_gyro).abs().mean().detach().cpu()) * batch_size
        metrics["count"] += batch_size
        if show_progress:
            current = metrics_from_sums(metrics)
            batches.set_postfix(loss=current["loss"], mae=current["mae"])
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


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    if checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("train", {}).get("seed"))

    device, distributed = init_distributed(config.get("distributed", {}))
    resume = args.resume or config.get("train", {}).get("resume")
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
        logger.info(f"pretrained_backbone={model.pretrained_report}")
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    criterion = build_loss(config["train"].get("loss", "smooth_l1")).to(device)
    optimizer = build_optimizer(config, model.parameters())
    epochs = int(config["train"].get("epochs", 50))
    scheduler, scheduler_step = build_scheduler(config, optimizer, len(train_loader), epochs)

    start_epoch = 0
    best_val_loss = float("inf")
    history = []
    if resume:
        checkpoint = load_checkpoint(resume, model, optimizer, scheduler, device)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        history = list(checkpoint.get("history", []))
        logger.info(f"resumed_from={resume}, start_epoch={start_epoch}")

    logger.info(
        f"train_samples={len(train_dataset)}, steps_per_epoch={len(train_loader)}, "
        f"epochs={epochs}, batch_size={config['dataset'].get('batch_size', 8)}"
    )

    log_interval = int(config["train"].get("log_interval", 100))
    checkpoint_interval = int(config["train"].get("checkpoint_interval", 5))
    val_interval = int(config.get("validation", {}).get("interval", 1))
    for epoch in range(start_epoch, epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        running = {"loss_sum": 0.0, "mae_sum": 0.0, "count": 0.0}
        progress = tqdm(
            train_loader,
            desc=f"train epoch {epoch + 1}/{epochs}",
            disable=not is_main_process(),
        )
        for step, batch in enumerate(progress, start=1):
            image = batch["image"].to(device, non_blocking=True).float()
            target_gyro = batch["gyro"].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            pred_gyro = model(image)["gyro"]
            loss = criterion(pred_gyro, target_gyro)
            loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler_step == "iteration":
                scheduler.step()

            batch_size = image.shape[0]
            running["loss_sum"] += float(loss.detach().cpu()) * batch_size
            running["mae_sum"] += float((pred_gyro - target_gyro).abs().mean().detach().cpu()) * batch_size
            running["count"] += batch_size

            if is_main_process() and (step % log_interval == 0 or step == len(train_loader)):
                metrics = metrics_from_sums(running)
                lr = optimizer.param_groups[0]["lr"]
                progress.set_postfix(loss=metrics["loss"], mae=metrics["mae"], lr=lr)

        if scheduler is not None and scheduler_step == "epoch":
            scheduler.step()

        train_metrics = metrics_from_sums(reduce_metrics(running, device))
        train_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append({"split": "train", "epoch": epoch + 1, **train_metrics})
        if is_main_process():
            logger.info(
                f"epoch={epoch + 1}/{epochs} train_loss={train_metrics['loss']:.6f} "
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
                logger.info(
                    f"epoch={epoch + 1}/{epochs} val_loss={val_metrics['loss']:.6f} "
                    f"val_mae={val_metrics['mae']:.6f}"
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
