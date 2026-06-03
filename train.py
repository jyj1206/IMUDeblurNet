import argparse
import math
from pathlib import Path

import torch
from tqdm import tqdm

from datasets import build_dataloader
from models.stage2_deblur_model import build_model
from utils import (
    append_history,
    batch_psnr,
    batch_ssim,
    build_checkpoint_state,
    build_criterion,
    build_logger,
    build_optimizer,
    build_scheduler,
    checkpoint_iteration,
    cleanup_distributed,
    evaluate_model,
    init_distributed,
    interval_due,
    is_main_process,
    load_checkpoint_state,
    load_config,
    normalize_config,
    prepare_run_dir,
    resolve_training_length,
    save_checkpoint,
    save_config,
    save_history,
    unwrap_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/stage2_deblur.yaml")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def sync_run_state(run_dir, resume_checkpoint, distributed):
    if not distributed:
        return run_dir, resume_checkpoint

    payload = [
        str(run_dir) if is_main_process() else None,
        str(resume_checkpoint) if is_main_process() and resume_checkpoint else None,
    ]
    torch.distributed.broadcast_object_list(payload, src=0)
    return Path(payload[0]), Path(payload[1]) if payload[1] else None


def main():
    args = parse_args()
    cfg = normalize_config(load_config(args.config))
    resume = args.resume or cfg.get("train", {}).get("resume")

    if resume and Path(resume).is_dir() and (Path(resume) / "config.yaml").exists():
        cfg = normalize_config(load_config(Path(resume) / "config.yaml"))
        cfg.setdefault("train", {})["resume"] = resume

    device, distributed = init_distributed(cfg.get("distributed", {}))
    run_dir, resume_checkpoint = prepare_run_dir(cfg, resume)
    run_dir, resume_checkpoint = sync_run_state(run_dir, resume_checkpoint, distributed)

    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(
        cfg["experiment"]["name"],
        run_dir / "train.log" if is_main_process() else None,
        enabled=is_main_process(),
    )
    logger.info(f"run_dir={run_dir}")
    logger.info(f"device={device}, distributed={distributed}")

    train_dataset, loader, sampler = build_dataloader(
        cfg,
        split=cfg["dataset"].get("split", "train"),
        distributed=distributed,
        device=device,
        is_train=True,
    )

    val_loader = None
    validation_cfg = cfg.get("validation", {})
    if validation_cfg.get("enabled", True):
        _, val_loader, _ = build_dataloader(
            cfg,
            split=validation_cfg.get("split", "val"),
            distributed=False,
            device=device,
            is_train=False,
        )

    total_iterations, epochs = resolve_training_length(cfg, len(loader))
    if is_main_process():
        save_config(cfg, run_dir / "config.yaml")
    logger.info(
        f"train_samples={len(train_dataset)}, steps_per_epoch={len(loader)}, "
        f"epochs={epochs}, total_iterations={total_iterations}"
    )

    model = build_model(cfg).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    criterion = build_criterion(cfg["train"].get("loss", "psnr")).to(device)
    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = build_scheduler(cfg, optimizer, total_iterations=total_iterations)

    start_iteration = 0
    best_val_psnr = -math.inf
    history = []
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        load_checkpoint_state(ckpt, model, optimizer, scheduler, unwrap_model, len(loader))
        start_iteration = checkpoint_iteration(ckpt, len(loader))
        best_val_psnr = float(ckpt.get("best_val_psnr", best_val_psnr))
        history = list(ckpt.get("history", []))
        logger.info(f"resumed_from={resume_checkpoint}, start_iteration={start_iteration}")

    train_cfg = cfg["train"]
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 1000))
    log_interval = int(train_cfg.get("log_interval", 100))
    validation_interval = int(validation_cfg.get("interval", checkpoint_interval))
    max_val_batches = validation_cfg.get("max_batches")

    current_iteration = int(start_iteration)
    start_epoch = current_iteration // len(loader)
    first_epoch_offset = current_iteration % len(loader)
    recent_losses = []
    progress = tqdm(
        total=total_iterations,
        initial=current_iteration,
        desc="train",
        disable=not is_main_process(),
    )

    for epoch in range(start_epoch, epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()

        for batch_idx, batch in enumerate(loader):
            if epoch == start_epoch and batch_idx < first_epoch_offset:
                continue
            if current_iteration >= total_iterations:
                break

            current_iteration += 1
            current_epoch = (current_iteration - 1) // len(loader)

            blur = batch["lq"].to(device, non_blocking=True).float()
            sharp = batch["gt"].to(device, non_blocking=True).float()
            motion_field = batch["motion_field"].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)

            pred = model(blur, motion_field, current_epoch)
            loss = criterion(pred, sharp)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            recent_losses.append(float(loss.detach().cpu()))
            if interval_due(current_iteration, log_interval) or current_iteration == total_iterations:
                train_metrics = {
                    "loss": sum(recent_losses) / len(recent_losses),
                    "psnr": float(batch_psnr(pred.detach(), sharp.detach()).detach().cpu()),
                    "ssim": float(batch_ssim(pred.detach(), sharp.detach()).detach().cpu()),
                }
                recent_losses.clear()
                if is_main_process():
                    append_history(history, "train", current_iteration, train_metrics)
                    progress.set_postfix(train_loss=train_metrics["loss"])

            if val_loader is not None and (
                interval_due(current_iteration, validation_interval)
                or current_iteration == total_iterations
            ):
                val_metrics = evaluate_model(
                    model,
                    val_loader,
                    criterion,
                    device,
                    epoch=current_epoch,
                    max_batches=max_val_batches,
                )
                if is_main_process():
                    append_history(history, "val", current_iteration, val_metrics)
                    logger.info(
                        f"iter={current_iteration}/{total_iterations} "
                        f"val_loss={val_metrics['loss']:.6f} "
                        f"val_psnr={val_metrics['psnr']:.4f} "
                        f"val_ssim={val_metrics['ssim']:.4f}"
                    )
                    if val_metrics["psnr"] > best_val_psnr:
                        best_val_psnr = val_metrics["psnr"]
                        state = build_checkpoint_state(
                            cfg,
                            model,
                            optimizer,
                            scheduler,
                            current_iteration,
                            current_epoch,
                            best_val_psnr,
                            history,
                            unwrap_model,
                        )
                        save_checkpoint(state, run_dir, "best.pt")

            if is_main_process() and (
                interval_due(current_iteration, checkpoint_interval)
                or current_iteration == total_iterations
            ):
                state = build_checkpoint_state(
                    cfg,
                    model,
                    optimizer,
                    scheduler,
                    current_iteration,
                    current_epoch,
                    best_val_psnr,
                    history,
                    unwrap_model,
                )
                save_checkpoint(state, run_dir, "last.pt")
                save_checkpoint(state, run_dir, f"iter_{current_iteration:08d}.pt")
                save_history(history, run_dir)

            progress.update(1)

        first_epoch_offset = 0
        if current_iteration >= total_iterations:
            break

    progress.close()
    if is_main_process():
        final_epoch = max(0, (current_iteration - 1) // len(loader))
        state = build_checkpoint_state(
            cfg,
            model,
            optimizer,
            scheduler,
            current_iteration,
            final_epoch,
            best_val_psnr,
            history,
            unwrap_model,
        )
        save_checkpoint(state, run_dir, "last.pt")
        save_history(history, run_dir)
        logger.info(f"finished total_iterations={current_iteration}, best_val_psnr={best_val_psnr:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
