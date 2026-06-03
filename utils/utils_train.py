import math
import random

import numpy as np
import torch


def resolve_training_length(config, steps_per_epoch):
    train_cfg = config.setdefault("train", {})
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive.")

    iterations = train_cfg.get("iterations")
    if iterations is not None:
        total_iterations = int(iterations)
        epochs = math.ceil(total_iterations / steps_per_epoch)
    else:
        epochs = int(train_cfg.get("epochs", 1))
        total_iterations = epochs * steps_per_epoch

    train_cfg["epochs"] = epochs
    train_cfg["total_iterations"] = total_iterations
    return total_iterations, epochs


def interval_due(iteration, interval):
    if interval is None:
        return False
    interval = int(interval)
    return interval > 0 and iteration % interval == 0


def set_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
