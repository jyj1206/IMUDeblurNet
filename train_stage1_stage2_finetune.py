import argparse
import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from datasets.stage1_stage2_finetune_dataset import build_stage1_stage2_finetune_loader
from models.stage1_stage2_finetune_model import build_stage1_stage2_finetune_model
from utils import (
    append_history,
    batch_psnr,
    batch_ssim,
    build_logger,
    build_scheduler,
    cleanup_distributed,
    init_distributed,
    interval_due,
    is_main_process,
    load_config,
    load_eval_config,
    prepare_run_dir,
    reduce_mean_tensor,
    resolve_training_length,
    save_checkpoint,
    save_config,
    save_history,
    set_seed,
    unwrap_model,
)
from utils.utils_eval import load_model_weights
from utils.utils_eval_config import upgrade_stage1_config_names
from utils.utils_loss import (
    build_stage1_stage2_finetune_loss,
    cmf_epe,
)
from utils.utils_stage_pipeline import camera_matrix_from_config
from utils.utils_torch_load import torch_load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end Stage1 -> differentiable CMF -> Stage2 fine-tuning."
    )
    parser.add_argument("--config", default="config/stage1_stage2_finetune.yaml")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def _load_component_config(component_cfg, checkpoint_key, normalize=False):
    checkpoint = component_cfg.get("checkpoint") or checkpoint_key
    config_path = component_cfg.get("config")
    cfg, source = load_eval_config(config_path, checkpoint, normalize=normalize)
    return cfg, source


def _set_requires_grad(module, enabled):
    for param in module.parameters():
        param.requires_grad = bool(enabled)


def _freeze_unused_stage1_aux(stage1):
    for name in ("flow_decoder", "depth_decoder", "pose_solver"):
        module = getattr(stage1, name, None)
        if module is not None:
            _set_requires_grad(module, False)


def _parameter_groups(model, config):
    train_cfg = config.get("train", {})
    stage1_cfg = config.get("stage1", {})
    stage2_cfg = config.get("stage2", {})
    default_lr = float(config.get("optimizer", {}).get("lr", train_cfg.get("lr", 1e-5)))
    stage1_lr = float(stage1_cfg.get("lr", train_cfg.get("stage1_lr", default_lr)))
    stage2_lr = float(stage2_cfg.get("lr", train_cfg.get("stage2_lr", default_lr)))

    groups = []
    stage1_params = [p for p in model.stage1.parameters() if p.requires_grad]
    stage2_params = [p for p in model.stage2.parameters() if p.requires_grad]
    if stage1_params:
        groups.append({"params": stage1_params, "lr": stage1_lr, "name": "stage1"})
    if stage2_params:
        groups.append({"params": stage2_params, "lr": stage2_lr, "name": "stage2"})
    if not groups:
        raise ValueError("No trainable parameters. Check stage1.freeze/stage2.freeze.")
    return groups


def _build_optimizer(config, model):
    optim_cfg = config.get("optimizer", {})
    name = str(optim_cfg.get("name", "adamw")).lower()
    betas = tuple(float(v) for v in optim_cfg.get("betas", [0.9, 0.999]))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    groups = _parameter_groups(model, config)
    if name == "adamw":
        return torch.optim.AdamW(groups, betas=betas, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(groups, betas=betas, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer.name: {name}")


def _make_state(
    config, model, optimizer, scheduler, iteration, epoch, best_val_psnr, history
):
    return {
        "iteration": int(iteration),
        "epoch": int(epoch),
        "total_iterations": int(config["train"]["total_iterations"]),
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val_psnr": float(best_val_psnr),
        "history": history,
        "config": config,
    }


def _load_resume(path, model, optimizer, scheduler, device):
    checkpoint = torch_load_checkpoint(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    if checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def _save_best_metrics(run_dir, iteration, metrics):
    data = {
        "iteration": int(iteration),
        "best_psnr": float(metrics["psnr"]),
        "ssim_at_best_psnr": float(metrics["ssim"]),
        "loss_at_best_psnr": float(metrics["loss"]),
        "gyro_mae_at_best_psnr": float(metrics["gyro_mae"]),
        "cmf_epe_at_best_psnr": float(metrics["cmf_epe"]),
        "count": int(metrics.get("count", 0)),
    }
    path = Path(run_dir) / "best_metrics.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        if key == "meta":
            moved[key] = value
        elif torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _forward(model, batch):
    return model(
        batch["stage1_image"].float(),
        batch["lq"].float(),
        batch["timestamp_window"].float(),
        crop_origin_yx=batch.get("crop_origin_yx"),
    )


def _batch_metrics(outputs, batch, loss_metrics):
    pred = outputs["pred"]
    sharp = batch["gt"].float()
    target_gyro = batch["gyro"].float()
    target_cmf = batch.get("target_cmf")
    metrics = {
        "loss": float(loss_metrics["loss"].detach().cpu()),
        "image_loss": float(loss_metrics["image_loss"].detach().cpu()),
        "gyro_loss": float(loss_metrics["gyro_loss"].detach().cpu()),
        "cmf_loss": float(loss_metrics["cmf_loss"].detach().cpu()),
        "smooth_loss": float(loss_metrics["smooth_loss"].detach().cpu()),
        "psnr": float(batch_psnr(pred.detach(), sharp.detach()).detach().cpu()),
        "ssim": float(batch_ssim(pred.detach(), sharp.detach()).detach().cpu()),
        "gyro_mae": float(
            (outputs["pred_gyro"].detach() - target_gyro).abs().mean().detach().cpu()
        ),
    }
    if target_cmf is not None:
        metrics["cmf_epe"] = float(
            cmf_epe(outputs["cmf"].detach(), target_cmf.float()).detach().cpu()
        )
    else:
        metrics["cmf_epe"] = 0.0
    return metrics


def _reduce_log_metrics(metrics, device):
    reduced = {}
    for key, value in metrics.items():
        tensor = torch.tensor(float(value), device=device)
        reduced[key] = float(reduce_mean_tensor(tensor).detach().cpu())
    return reduced


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None, show_progress=False):
    was_training = model.training
    model.eval()
    sums = {
        "loss": 0.0,
        "image_loss": 0.0,
        "gyro_loss": 0.0,
        "cmf_loss": 0.0,
        "smooth_loss": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "gyro_mae": 0.0,
        "cmf_epe": 0.0,
    }
    count = 0
    total = len(loader) if max_batches is None else min(len(loader), int(max_batches))
    progress = tqdm(
        loader, total=total, desc="val", leave=False, disable=not show_progress
    )
    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = _move_batch(batch, device)
        outputs = _forward(model, batch)
        loss_metrics = criterion(outputs, batch)
        metrics = _batch_metrics(outputs, batch, loss_metrics)
        batch_size = int(batch["lq"].shape[0])
        for key in sums:
            sums[key] += metrics[key] * batch_size
        count += batch_size
        if show_progress:
            progress.set_postfix(
                loss=sums["loss"] / max(1, count), psnr=sums["psnr"] / max(1, count)
            )
    if was_training:
        model.train()
    if count <= 0:
        return {**{key: 0.0 for key in sums}, "count": 0}
    return {**{key: value / count for key, value in sums.items()}, "count": count}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    resume = args.resume or cfg.get("train", {}).get("resume")
    if resume and Path(resume).is_dir() and (Path(resume) / "config.yaml").exists():
        cfg = load_config(Path(resume) / "config.yaml")
        cfg.setdefault("train", {})["resume"] = resume
    cfg = upgrade_stage1_config_names(cfg)

    set_seed(cfg.get("train", {}).get("seed"))
    stage1_cfg, stage1_source = _load_component_config(
        cfg.get("stage1", {}), None, normalize=False
    )
    stage2_cfg, stage2_source = _load_component_config(
        cfg.get("stage2", {}), None, normalize=True
    )
    cfg["stage1_resolved_config"] = stage1_cfg
    cfg["stage2_resolved_config"] = stage2_cfg
    cfg["stage1_config_source"] = stage1_source
    cfg["stage2_config_source"] = stage2_source

    camera_matrix = camera_matrix_from_config(cfg)
    device, distributed = init_distributed(cfg.get("distributed", {}))
    run_dir, resume_checkpoint = prepare_run_dir(cfg, resume)
    if distributed:
        payload = [
            str(run_dir) if is_main_process() else None,
            str(resume_checkpoint) if is_main_process() and resume_checkpoint else None,
        ]
        torch.distributed.broadcast_object_list(payload, src=0)
        run_dir = Path(payload[0])
        resume_checkpoint = Path(payload[1]) if payload[1] else None

    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, run_dir / "config.yaml")
    logger = build_logger(
        cfg["experiment"]["name"],
        run_dir / "log.txt" if is_main_process() else None,
        enabled=is_main_process(),
    )
    logger.info(f"run_dir={run_dir}")
    logger.info(f"device={device}, distributed={distributed}")

    train_dataset, train_loader, train_sampler = build_stage1_stage2_finetune_loader(
        cfg,
        stage1_cfg,
        split=cfg["dataset"].get("split", "train"),
        distributed=distributed,
        device=device,
        is_train=True,
    )
    val_loader = None
    if cfg.get("validation", {}).get("enabled", True):
        _, val_loader, _ = build_stage1_stage2_finetune_loader(
            cfg,
            stage1_cfg,
            split=cfg["validation"].get("split", "val"),
            distributed=False,
            device=device,
            is_train=False,
        )

    model = build_stage1_stage2_finetune_model(
        stage1_config=stage1_cfg,
        stage2_config=stage2_cfg,
        motion_downsample=cfg.get("dataset", {}).get("motion_downsample", 2),
        default_dt=cfg.get("time", {}).get("default_dt", 1.0 / 240.0),
        camera_matrix=camera_matrix,
    ).to(device)

    if bool(cfg.get("train", {}).get("freeze_unused_stage1_aux", True)):
        _freeze_unused_stage1_aux(model.stage1)
    if cfg.get("stage1", {}).get("freeze", False):
        _set_requires_grad(model.stage1, False)
    if cfg.get("stage2", {}).get("freeze", False):
        _set_requires_grad(model.stage2, False)

    if resume_checkpoint is None:
        stage1_report = load_model_weights(
            model.stage1,
            cfg.get("stage1", {}).get("checkpoint"),
            device=device,
            strict=bool(cfg.get("stage1", {}).get("strict", True)),
        )
        stage2_report = load_model_weights(
            model.stage2,
            cfg.get("stage2", {}).get("checkpoint"),
            device=device,
            strict=bool(cfg.get("stage2", {}).get("strict", True)),
        )
        logger.info(f"loaded_stage1={stage1_report}")
        logger.info(f"loaded_stage2={stage2_report}")

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            find_unused_parameters=bool(
                cfg.get("distributed", {}).get("find_unused_parameters", False)
            ),
        )

    total_iterations, epochs = resolve_training_length(cfg, len(train_loader))
    optimizer = _build_optimizer(cfg, unwrap_model(model))
    scheduler = build_scheduler(
        cfg, optimizer, total_iterations=total_iterations, total_epochs=epochs
    )
    criterion = build_stage1_stage2_finetune_loss(cfg).to(device)

    start_iteration = 0
    best_val_psnr = -math.inf
    history = []
    if resume_checkpoint:
        checkpoint = _load_resume(
            resume_checkpoint, model, optimizer, scheduler, device
        )
        start_iteration = int(checkpoint.get("iteration", 0))
        best_val_psnr = float(checkpoint.get("best_val_psnr", best_val_psnr))
        history = list(checkpoint.get("history", []))
        logger.info(
            f"resumed_from={resume_checkpoint}, start_iteration={start_iteration}"
        )

    logger.info(
        f"train_samples={len(train_dataset)}, steps_per_epoch={len(train_loader)}, "
        f"epochs={epochs}, total_iterations={total_iterations}, "
        f"batch_size={cfg['dataset'].get('batch_size')}"
    )

    train_cfg = cfg["train"]
    log_interval = int(train_cfg.get("log_interval", 100))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 1000))
    validation_interval = int(
        cfg.get("validation", {}).get("interval", checkpoint_interval)
    )
    max_val_batches = cfg.get("validation", {}).get("max_batches")
    grad_clip = train_cfg.get("grad_clip_norm")
    grad_clip = float(grad_clip) if grad_clip is not None else None

    current_iteration = int(start_iteration)
    start_epoch = current_iteration // len(train_loader)
    first_epoch_offset = current_iteration % len(train_loader)
    recent_losses = []
    last_log_time = time.time()
    progress = tqdm(
        total=total_iterations,
        initial=current_iteration,
        desc="stage1->stage2 finetune",
        disable=not is_main_process(),
    )

    for epoch in range(start_epoch, epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_had_update = False
        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < first_epoch_offset:
                continue
            if current_iteration >= total_iterations:
                break
            current_iteration += 1
            current_epoch = (current_iteration - 1) // len(train_loader)
            batch = _move_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)
            outputs = _forward(model, batch)
            loss_metrics = criterion(outputs, batch)
            loss = loss_metrics["loss"]
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    unwrap_model(model).parameters(), grad_clip
                )
            optimizer.step()
            epoch_had_update = True
            progress.update(1)
            recent_losses.append(float(loss.detach().cpu()))

            if (
                interval_due(current_iteration, log_interval)
                or current_iteration == total_iterations
            ):
                metrics = _batch_metrics(outputs, batch, loss_metrics)
                metrics["loss"] = sum(recent_losses) / max(1, len(recent_losses))
                metrics = _reduce_log_metrics(metrics, device)
                metrics["lr_stage1"] = float(optimizer.param_groups[0]["lr"])
                metrics["lr_stage2"] = float(optimizer.param_groups[-1]["lr"])
                recent_losses.clear()
                if is_main_process():
                    append_history(history, "train", current_iteration, metrics)
                    elapsed = time.time() - last_log_time
                    last_log_time = time.time()
                    progress.set_postfix(
                        loss=metrics["loss"],
                        psnr=metrics["psnr"],
                        gyro=metrics["gyro_mae"],
                    )
                    logger.info(
                        f"train iter={current_iteration}/{total_iterations} "
                        f"loss={metrics['loss']:.6f} image={metrics['image_loss']:.6f} "
                        f"gyro={metrics['gyro_loss']:.6f} cmf={metrics['cmf_loss']:.6f} "
                        f"psnr={metrics['psnr']:.4f} ssim={metrics['ssim']:.6f} "
                        f"gyro_mae={metrics['gyro_mae']:.6f} cmf_epe={metrics['cmf_epe']:.6f} "
                        f"lr1={metrics['lr_stage1']:.3e} lr2={metrics['lr_stage2']:.3e} elapsed={elapsed:.1f}s"
                    )
                    save_history(history, run_dir)

            if val_loader is not None and (
                interval_due(current_iteration, validation_interval)
                or current_iteration == total_iterations
            ):
                val_metrics = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    max_batches=max_val_batches,
                    show_progress=is_main_process(),
                )
                if is_main_process():
                    append_history(history, "val", current_iteration, val_metrics)
                    logger.info(
                        f"iter={current_iteration}/{total_iterations} "
                        f"val_loss={val_metrics['loss']:.6f} "
                        f"val_psnr={val_metrics['psnr']:.4f} "
                        f"val_ssim={val_metrics['ssim']:.6f} "
                        f"val_gyro_mae={val_metrics['gyro_mae']:.6f} "
                        f"val_cmf_epe={val_metrics['cmf_epe']:.6f}"
                    )
                    is_best = val_metrics["psnr"] > best_val_psnr
                    if is_best:
                        best_val_psnr = val_metrics["psnr"]
                    state = _make_state(
                        cfg,
                        model,
                        optimizer,
                        scheduler,
                        current_iteration,
                        current_epoch,
                        best_val_psnr,
                        history,
                    )
                    save_checkpoint(state, run_dir, "latest.pt")
                    if is_best:
                        save_checkpoint(state, run_dir, "best.pt")
                        _save_best_metrics(run_dir, current_iteration, val_metrics)
                        logger.info(f"saved best checkpoint | psnr={best_val_psnr:.6f}")
                    save_history(history, run_dir)

            if is_main_process() and (
                interval_due(current_iteration, checkpoint_interval)
                or current_iteration == total_iterations
            ):
                state = _make_state(
                    cfg,
                    model,
                    optimizer,
                    scheduler,
                    current_iteration,
                    current_epoch,
                    best_val_psnr,
                    history,
                )
                save_checkpoint(state, run_dir, "latest.pt")
                save_history(history, run_dir)
                logger.info(f"saved latest checkpoint | iter={current_iteration}")

        first_epoch_offset = 0
        if scheduler is not None and epoch_had_update:
            scheduler.step()
        if current_iteration >= total_iterations:
            break

    progress.close()
    if is_main_process():
        final_epoch = max(0, (current_iteration - 1) // len(train_loader))
        state = _make_state(
            cfg,
            model,
            optimizer,
            scheduler,
            current_iteration,
            final_epoch,
            best_val_psnr,
            history,
        )
        save_checkpoint(state, run_dir, "latest.pt")
        save_history(history, run_dir)
        logger.info(
            f"finished total_iterations={current_iteration}, best_val_psnr={best_val_psnr:.4f}"
        )

    cleanup_distributed()


if __name__ == "__main__":
    main()
