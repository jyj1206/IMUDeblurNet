import csv
import json
from pathlib import Path


def append_history(history, split, iteration, metrics):
    row = {"split": split, "iteration": int(iteration)}
    row.update({key: float(value) for key, value in metrics.items()})
    history.append(row)


def save_history(history, run_dir):
    run_dir = Path(run_dir)
    metric_dir = run_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    train_rows = [row for row in history if row["split"] == "train"]
    val_rows = [row for row in history if row["split"] == "val"]
    _save_split_csv(
        train_rows,
        metric_dir / "train_log.csv",
        [
            "epoch",
            "iteration",
            "loss",
            "loss_last",
            "mae",
            "gyro_x_mae",
            "gyro_y_mae",
            "gyro_z_mae",
            "psnr",
            "ssim",
            "lr",
            "count",
        ],
    )
    _save_split_csv(
        val_rows,
        metric_dir / "validation_log.csv",
        [
            "epoch",
            "iteration",
            "loss",
            "mae",
            "gyro_x_mae",
            "gyro_y_mae",
            "gyro_z_mae",
            "psnr",
            "ssim",
            "count",
        ],
    )
    _plot_metric_curves(metric_dir, train_rows, val_rows)


def _save_split_csv(rows, path, preferred_fields):
    fields = list(preferred_fields)
    for row in rows:
        for key in row:
            if key not in {"split"} and key not in fields:
                fields.append(key)

    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _plot_metric_curves(metric_dir, train_rows, val_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = _metric_names(train_rows, val_rows)
    labels = {
        "loss": "Loss",
        "loss_last": "Last Loss",
        "mae": "MAE",
        "gyro_x_mae": "Gyro X MAE",
        "gyro_y_mae": "Gyro Y MAE",
        "gyro_z_mae": "Gyro Z MAE",
        "psnr": "PSNR (dB)",
        "ssim": "SSIM",
        "lr": "Learning Rate",
    }
    specs = [
        (metric, labels.get(metric, metric), f"{metric}_curve.png")
        for metric in metrics
    ]
    for metric, ylabel, filename in specs:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        plotted = False
        x_label = "Epoch"
        for split, rows in [("train", train_rows), ("validation", val_rows)]:
            rows = [row for row in rows if metric in row]
            if not rows:
                continue
            x_key = _x_key(rows)
            if x_key == "iteration":
                x_label = "Iteration"
            axis.plot(
                [row[x_key] for row in rows],
                [row[metric] for row in rows],
                label=f"{split} {metric}",
                linewidth=1.8,
                marker="o" if split == "validation" else None,
            )
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        axis.set_xlabel(x_label)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(metric_dir / filename, dpi=150)
        plt.close(fig)


def _metric_names(*row_groups):
    skip = {"split", "epoch", "iteration", "count"}
    preferred = [
        "loss",
        "mae",
        "gyro_x_mae",
        "gyro_y_mae",
        "gyro_z_mae",
        "psnr",
        "ssim",
        "loss_last",
        "lr",
    ]
    names = []
    for rows in row_groups:
        for row in rows:
            for key, value in row.items():
                if key in skip or value == "":
                    continue
                if key not in names:
                    names.append(key)
    return [key for key in preferred if key in names] + [
        key for key in names if key not in preferred
    ]


def _x_key(rows):
    if rows and all("iteration" in row for row in rows):
        return "iteration"
    return "epoch"
