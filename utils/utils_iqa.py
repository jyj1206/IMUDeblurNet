from contextlib import contextmanager
import warnings

import torch


FULL_REFERENCE_METRICS = {"lpips", "topiq_fr"}
NO_REFERENCE_METRICS = {"niqe", "topiq_nr"}


def normalize_iqa_metric_names(names, realblur_preset=False, has_target=True):
    metric_names = []
    if realblur_preset:
        if has_target:
            metric_names.extend(["lpips", "niqe", "topiq_fr"])
        else:
            metric_names.extend(["niqe", "topiq_nr"])
    metric_names.extend(names or [])

    normalized = []
    for name in metric_names:
        key = str(name).strip().lower().replace("-", "_")
        if key == "topiq":
            key = "topiq_fr"
        if key not in FULL_REFERENCE_METRICS and key not in NO_REFERENCE_METRICS:
            raise ValueError(
                f"Unknown IQA metric: {name}. "
                "Use one of: lpips, niqe, topiq, topiq_fr, topiq_nr."
            )
        if not has_target and key in FULL_REFERENCE_METRICS:
            continue
        if key not in normalized:
            normalized.append(key)
    return normalized


class Stage2IqaMetrics:
    def __init__(self, metric_names, device):
        self.metric_names = list(metric_names)
        self.lpips_model = None
        self.pyiqa_models = {}

        with _quiet_iqa_library_warnings():
            if "lpips" in self.metric_names:
                try:
                    import lpips
                except ImportError as exc:
                    raise ImportError(
                        "LPIPS metric needs the 'lpips' package. "
                        "Install it with: pip install lpips"
                    ) from exc
                self.lpips_model = lpips.LPIPS(net="alex").to(device).eval()

            pyiqa_names = [name for name in self.metric_names if name in {"niqe", "topiq_fr", "topiq_nr"}]
            if pyiqa_names:
                try:
                    import pyiqa
                except ImportError as exc:
                    raise ImportError(
                        "NIQE/TOPIQ metrics need the 'pyiqa' package. "
                        "Install it with: pip install pyiqa"
                    ) from exc
                for name in pyiqa_names:
                    self.pyiqa_models[name] = pyiqa.create_metric(
                        name,
                        device=device,
                        as_loss=False,
                    )

    @torch.no_grad()
    def __call__(self, pred, target=None):
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0) if target is not None else None
        values = {}

        if "lpips" in self.metric_names:
            if target is None:
                raise ValueError("LPIPS needs a target image.")
            pred_lpips = pred * 2.0 - 1.0
            target_lpips = target * 2.0 - 1.0
            values["lpips"] = _as_batch_values(
                self.lpips_model(pred_lpips, target_lpips),
                pred.shape[0],
            )

        for name, model in self.pyiqa_models.items():
            if name in FULL_REFERENCE_METRICS:
                if target is None:
                    raise ValueError(f"{name} needs a target image.")
                metric_value = model(pred, target)
            else:
                metric_value = model(pred)
            values[name] = _as_batch_values(metric_value, pred.shape[0])

        return values


def _as_batch_values(value, batch_size):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.detach().float().cpu()
    batch_size = int(batch_size)
    flat = value.reshape(-1)
    if flat.numel() == 1:
        return flat.repeat(batch_size)
    if flat.numel() == batch_size:
        return flat
    if flat.numel() % batch_size == 0:
        return flat.reshape(batch_size, -1).mean(dim=1)
    raise ValueError(
        f"Metric output cannot be mapped to batch values: "
        f"shape={tuple(value.shape)}, batch_size={batch_size}"
    )


@contextmanager
def _quiet_iqa_library_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The parameter 'pretrained' is deprecated.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Arguments other than a weight enum or `None` for 'weights' are deprecated.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Importing from timm\.models\.layers is deprecated.*",
            category=FutureWarning,
        )
        yield
