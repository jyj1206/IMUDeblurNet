import csv
from pathlib import Path


def append_history(history, split, iteration, metrics):
    row = {"split": split, "iteration": int(iteration)}
    row.update({key: float(value) for key, value in metrics.items()})
    history.append(row)


def save_history(history, run_dir):
    run_dir = Path(run_dir)
    metric_dir = run_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [row for row in history if row["split"] == "train"]
    val_rows = [row for row in history if row["split"] == "val"]
    _save_split_csv(
        train_rows,
        metric_dir / "train_log.csv",
        ["iteration", "loss", "loss_last", "psnr", "ssim", "lr"],
    )
    _save_split_csv(
        val_rows,
        metric_dir / "validation_log.csv",
        ["iteration", "loss", "psnr", "ssim", "count"],
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

    specs = [
        ("loss", "Loss", "loss_curve.png"),
        ("psnr", "PSNR (dB)", "psnr_curve.png"),
        ("ssim", "SSIM", "ssim_curve.png"),
    ]
    for metric, ylabel, filename in specs:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        plotted = False
        for split, rows in [("train", train_rows), ("validation", val_rows)]:
            rows = [row for row in rows if metric in row]
            if not rows:
                continue
            axis.plot(
                [row["iteration"] for row in rows],
                [row[metric] for row in rows],
                label=f"{split} {metric}",
                linewidth=1.8,
                marker="o" if split == "validation" else None,
            )
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        axis.set_xlabel("Iteration")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(metric_dir / filename, dpi=150)
        plt.close(fig)
