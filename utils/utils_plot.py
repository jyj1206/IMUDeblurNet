import csv
import json
from pathlib import Path


def append_history(history, split, iteration, metrics):
    row = {"split": split, "iteration": int(iteration)}
    row.update({key: float(value) for key, value in metrics.items()})
    history.append(row)


def save_history(history, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_history_json(history, run_dir / "metrics_history.json")
    _save_history_csv(history, run_dir / "metrics_history.csv")
    _plot_history(history, run_dir / "metrics.png")


def _save_history_json(history, path):
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def _save_history_csv(history, path):
    fields = ["split", "iteration", "loss", "psnr", "ssim"]
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field, "") for field in fields})


def _plot_history(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [row for row in history if row["split"] == "train"]
    val = [row for row in history if row["split"] == "val"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for axis, metric in zip(axes, ["loss", "psnr", "ssim"]):
        for split, rows in [("train", train), ("val", val)]:
            rows = [row for row in rows if metric in row]
            if rows:
                axis.plot(
                    [row["iteration"] for row in rows],
                    [row[metric] for row in rows],
                    label=split,
                )
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
