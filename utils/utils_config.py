from copy import deepcopy


def normalize_config(config):
    config = deepcopy(config)

    paths = config.get("paths", {})
    data = config.get("data", {})
    motion = config.get("motion_field", {})

    if paths:
        experiment = config.setdefault("experiment", {})
        experiment["name"] = paths.get("run_name", experiment.get("name", "stage2_motion_field_deblur"))
        experiment["result_root"] = paths.get("log_root", experiment.get("result_root", "result"))
        experiment["run_prefix"] = paths.get("run_prefix", experiment.get("run_prefix", "run"))
        if paths.get("output_dir") is not None:
            experiment["output_dir"] = paths["output_dir"]

        train = config.setdefault("train", {})
        if paths.get("resume") is not None:
            train["resume"] = paths["resume"]

    if data:
        dataset = config.setdefault("dataset", {})
        dataset["root"] = data.get(
            "dataset_root",
            paths.get("dataset_root", dataset.get("root", "data/IMUBlur")),
        )
        dataset["split"] = data.get("train_split", dataset.get("split", "train"))
        dataset["metadata_name"] = data.get("metadata_name", dataset.get("metadata_name", "metadata.csv"))
        dataset["patch_size"] = data.get("patch_size", dataset.get("patch_size", 256))
        dataset["batch_size"] = data.get("batch_size", dataset.get("batch_size", 4))
        dataset["num_workers"] = data.get("num_workers", dataset.get("num_workers", 0))

        validation = config.setdefault("validation", {})
        validation["enabled"] = data.get("use_validation", validation.get("enabled", True))
        validation["split"] = data.get("val_split", validation.get("split", "val"))
        validation["batch_size"] = data.get("val_batch_size", validation.get("batch_size", 1))
        validation["num_workers"] = data.get("val_workers", validation.get("num_workers", 0))
        if data.get("val_max_batches") is not None:
            validation["max_batches"] = data["val_max_batches"]

    if motion:
        dataset = config.setdefault("dataset", {})
        dataset["motion_field_root"] = motion.get("root", dataset.get("motion_field_root"))
        dataset["motion_field_dir"] = motion.get("dir", dataset.get("motion_field_dir", "camera_motion_field"))
        dataset["motion_field_ext"] = motion.get("ext", dataset.get("motion_field_ext", "npy"))
        dataset["motion_downsample"] = motion.get("downsample", dataset.get("motion_downsample", 2))

    model = config.setdefault("model", {})
    model_args = dict(model.get("args") or {})
    model_aliases = {
        "img_channels": "img_channel",
        "image_channels": "img_channel",
        "motion_channels": "motion_channel",
    }
    for source, target in model_aliases.items():
        if source in model and target not in model_args:
            model_args[target] = model[source]
        if source in model_args and target not in model_args:
            model_args[target] = model_args.pop(source)
    for key in (
        "img_channel",
        "motion_channel",
        "width",
        "middle_blk_num",
        "enc_blk_nums",
        "dec_blk_nums",
    ):
        if key in model and key not in model_args:
            model_args[key] = model[key]
    if model_args:
        model["args"] = model_args

    train = config.setdefault("train", {})
    if "total_iters" in train:
        train["iterations"] = train["total_iters"]
    elif "total_iterations" in train:
        train["iterations"] = train["total_iterations"]
    else:
        train.setdefault("iterations", None)

    if "log_every" in train:
        train["log_interval"] = train["log_every"]
    if "save_every" in train:
        train["checkpoint_interval"] = train["save_every"]

    validation = config.setdefault("validation", {})
    if "val_every" in train:
        validation["interval"] = train["val_every"]
    elif "every" in validation:
        validation["interval"] = validation["every"]

    train.setdefault("loss", "psnr")
    train.setdefault("log_interval", 100)
    train.setdefault("checkpoint_interval", 1000)
    train.setdefault("resume", None)
    validation.setdefault("enabled", True)
    validation.setdefault("interval", train["checkpoint_interval"])
    validation.setdefault("max_batches", None)

    return config
