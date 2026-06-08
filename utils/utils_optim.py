import torch


def build_optimizer(config, parameters):
    optim_cfg = config.get("optimizer", {})
    train_cfg = config.get("train", {})

    name = optim_cfg.get("name", "adam").lower()
    lr = float(optim_cfg.get("lr", train_cfg.get("lr", 1e-4)))
    betas = tuple(float(v) for v in optim_cfg.get("betas", [0.9, 0.999]))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    if name == "adam":
        return torch.optim.Adam(
            parameters, lr=lr, betas=betas, weight_decay=weight_decay
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=lr, betas=betas, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer.name: {name}")


def build_scheduler(config, optimizer, total_iterations=None, total_epochs=None):
    scheduler_cfg = config.get("scheduler", {})
    train_cfg = config.get("train", {})
    name = scheduler_cfg.get("name", "cosine").lower()

    if name in ("none", "null"):
        return None
    if name == "cosine":
        t_max = int(
            scheduler_cfg.get("t_max")
            or total_epochs
            or train_cfg.get("epochs")
            or total_iterations
            or train_cfg.get("total_iterations")
            or 50
        )
        eta_min = float(scheduler_cfg.get("eta_min", 1e-7))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
    raise ValueError(f"Unknown scheduler.name: {name}")
