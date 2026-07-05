import json
import os
import random
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

import numpy as np

from utils.log import logger, model_tracker


REQUIRED_KEYS = ["id", "dataset_params", "preprocessor_params", "model_params", "trainer_params"]
LABEL_SWEEP_DEFAULT_TYPES = ("specific", "composite", "factor", "weighted")
LABEL_SWEEP_ALL_ALIASES = {"all", "alltypes", "alllabeltypes", "alllabels"}
SUMMARY_METRICS = ("accuracy", "precision", "recall", "f1", "auc")
EEG_ONLY_SUMMARY_BRANCHES = (("eeg", ""),)
MULTIMODAL_SUMMARY_BRANCHES = (("eeg", "eeg"), ("survey", "survey"), ("fusion", "fusion"))


def verify_config(config):
    for key in REQUIRED_KEYS:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")


def configure_reproducibility(config):
    seed = config.get("seed")
    if seed is None:
        return

    seed = int(seed)
    deterministic = bool(config.get("deterministic", True))

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    except Exception as exc:
        logger.log("reproducibility_warning", str(exc))

    logger.log("seed", seed)
    logger.log("deterministic", deterministic)


def _normalize_label_type(value):
    normalized = str(value).strip().lower()
    normalized = normalized.replace("-", "").replace("_", "").replace(" ", "")
    return normalized


def _format_label_folder_name(raw_label):
    if raw_label is None:
        return "Run"

    normalized = _normalize_label_type(raw_label)
    canonical_map = {
        "specific": "Specific",
        "composite": "Composite",
        "factor": "Factor",
        "weighted": "Weighted",
    }
    if normalized in canonical_map:
        return canonical_map[normalized]

    label_text = str(raw_label).strip()
    if not label_text:
        return "Run"

    safe_chars = []
    for ch in label_text:
        if ch.isalnum():
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe_label = "".join(safe_chars).strip("_")
    return safe_label or "Run"


def _format_run_family_folder_name(config):
    trainer_name = str(config.get("trainer_params", {}).get("name", "")).strip().lower()
    dataset_name = str(config.get("dataset_params", {}).get("name", "")).strip().lower()
    if "multimodal" in trainer_name or "multimodal" in dataset_name:
        return "multimodal"
    return "eeg_only"


def _format_architecture_folder_name(config, model_name):
    dataset_params = config.get("dataset_params", {})
    eeg_loader_name = str(
        dataset_params.get("eeg_loader_name", dataset_params.get("eeg_source", ""))
    ).strip().lower()

    is_raw = model_name.lower().startswith("raw") or eeg_loader_name in {"raw", "rawcsv", "raw_csv"}
    prefix = "1D" if is_raw else "2D"
    base_name = model_name[3:] if model_name.lower().startswith("raw") else model_name
    return f"{prefix}_{base_name}"


def _build_saved_model_folder_name(config, model_name, fold_data, fold_idx):
    dataset_params = config.get("dataset_params", {})
    run_family = _format_run_family_folder_name(config)
    architecture_name = _format_architecture_folder_name(config, model_name)
    label_name = _format_label_folder_name(dataset_params.get("labels_csv"))

    cv_fold = fold_data.get("cv_fold_in_repeat")
    cv_repeat = fold_data.get("cv_repeat_index")
    if cv_fold is not None:
        if cv_repeat is not None and int(cv_repeat) > 1:
            fold_folder = f"Repeat_{int(cv_repeat)}_CV_{int(cv_fold)}"
        else:
            fold_folder = f"CV_{int(cv_fold)}"
        return os.path.join(run_family, architecture_name, label_name, fold_folder)

    if fold_data.get("cv_fold") is not None:
        fold_folder = f"CV_{int(fold_data['cv_fold'])}"
        return os.path.join(run_family, architecture_name, label_name, fold_folder)

    if dataset_params.get("labels_csv") is not None:
        return os.path.join(run_family, architecture_name, label_name, "Run")

    return os.path.join(run_family, architecture_name, f"Run_{int(fold_idx)}")


def _resolve_label_sweep_types(config):
    dataset_params = config.get("dataset_params", {})
    label_sweep_cfg = config.get("label_sweep", {})

    raw_labels_csv = dataset_params.get("labels_csv", dataset_params.get("lables_csv"))
    labels_csv_norm = _normalize_label_type(raw_labels_csv) if raw_labels_csv is not None else ""
    shorthand_all = labels_csv_norm in LABEL_SWEEP_ALL_ALIASES
    enabled = bool(label_sweep_cfg.get("enabled", False)) or shorthand_all
    if not enabled:
        return None

    configured_types = label_sweep_cfg.get("label_types", dataset_params.get("label_types"))
    if configured_types is None:
        configured_types = list(LABEL_SWEEP_DEFAULT_TYPES)

    if not isinstance(configured_types, (list, tuple)):
        raise ValueError("label_sweep.label_types must be a list of label type names.")

    valid_norm_to_name = {_normalize_label_type(name): name for name in LABEL_SWEEP_DEFAULT_TYPES}
    resolved = []
    for raw_type in configured_types:
        norm = _normalize_label_type(raw_type)
        if norm not in valid_norm_to_name:
            raise ValueError(
                f"Unsupported label type '{raw_type}'. "
                f"Supported values: {', '.join(LABEL_SWEEP_DEFAULT_TYPES)}."
            )
        canonical = valid_norm_to_name[norm]
        if canonical not in resolved:
            resolved.append(canonical)

    if not resolved:
        raise ValueError("label_sweep requires at least one label type.")

    return resolved


def _resolve_seed_sweep(config):
    seed_sweep_cfg = config.get("seed_sweep", {})
    if not bool(seed_sweep_cfg.get("enabled", False)):
        return None

    seeds = seed_sweep_cfg.get("seeds")
    if seeds is None or not isinstance(seeds, (list, tuple)):
        raise ValueError("seed_sweep.seeds must be a list of integer seeds.")

    resolved = []
    for seed in seeds:
        seed_int = int(seed)
        if seed_int not in resolved:
            resolved.append(seed_int)

    if not resolved:
        raise ValueError("seed_sweep requires at least one seed.")

    return resolved


def _resolve_summary_branches(config):
    trainer_name = str(config.get("trainer_params", {}).get("name", "")).strip().lower()
    dataset_name = str(config.get("dataset_params", {}).get("name", "")).strip().lower()

    if "multimodal" in trainer_name or "multimodal" in dataset_name:
        return MULTIMODAL_SUMMARY_BRANCHES
    return EEG_ONLY_SUMMARY_BRANCHES


def _extract_metric_for_summary(entry, metric_name, prefix=""):
    metric_suffix = f"{prefix}_{metric_name}" if prefix else metric_name
    candidates = (
        f"cv_avg_test_{metric_suffix}",
        f"test_{metric_suffix}",
        f"cv_avg_val_{metric_suffix}",
        f"val_{metric_suffix}",
        f"train_{metric_suffix}",
    )
    for key in candidates:
        if key in entry and entry[key] is not None:
            return key, entry[key]
    return None, None


def _format_metric(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric_bundle(row, branch_name):
    return "/".join(_format_metric(row.get(f"{branch_name}_{metric}")) for metric in SUMMARY_METRICS)


def run_config(
    config,
    data_map,
    preprocessor_map,
    model_map,
    trainer_map,
    save_model=False,
    log_filename_override=None,
    clear_logger=False,
):
    verify_config(config)
    config_copy = deepcopy(config)
    if log_filename_override:
        config_copy["log_filename"] = log_filename_override
    if clear_logger:
        logger.clear()

    dataset_params = config_copy["dataset_params"]
    preprocessor_params = config_copy["preprocessor_params"]
    model_params = config_copy["model_params"]
    trainer_params = config_copy["trainer_params"]
    summary_branches = _resolve_summary_branches(config_copy)
    config_seed = config_copy.get("seed")
    if config_seed is not None and preprocessor_params.get("seed") is None:
        preprocessor_params["seed"] = int(config_seed)

    logger.log('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.log('config_id', config_copy.get("id", "N/A"))
    configure_reproducibility(config_copy)

    model_name = model_params.get("name")
    if model_name not in model_map:
        raise ValueError(f"Model '{model_name}' not found in model_map.")

    dataset_retriever = dataset_params.get("name")
    if dataset_retriever not in data_map:
        raise ValueError(f"Data retriever '{dataset_retriever}' not found in data_map.")
    X, y, metadata = data_map[dataset_retriever](dataset_params, metadata={})

    preprocessor_name = preprocessor_params.get("name")
    if preprocessor_name not in preprocessor_map:
        raise ValueError(f"Preprocessor '{preprocessor_name}' not found in preprocessor_map.")
    data, metadata = preprocessor_map[preprocessor_name](preprocessor_params, X, y, metadata)

    fold_data_list = data if isinstance(data, list) else [data]

    fold_metrics = []
    for fold_idx, fold_data in enumerate(fold_data_list, start=1):
        # Reinitialize tracker per fold so model saving works with CV runs.
        model_tracker.set_config(config_copy)
        if len(fold_data_list) > 1:
            logger.log('cv_fold', fold_idx)
            logger.log('cv_folds', len(fold_data_list))
            if fold_data.get('cv_fold_in_repeat') is not None:
                logger.log('cv_fold_in_repeat', fold_data.get('cv_fold_in_repeat'))
            if fold_data.get('cv_repeat_index') is not None:
                logger.log('cv_repeat', fold_data.get('cv_repeat_index'))
            if fold_data.get('cv_total_folds') is not None:
                logger.log('cv_total_folds', fold_data.get('cv_total_folds'))

        save_folder_name = _build_saved_model_folder_name(config_copy, model_name, fold_data, fold_idx)
        model_tracker.set_model_name(save_folder_name, save_model)

        model = model_map[model_name](model_params, metadata)
        model_tracker.set_model(model)

        trainer_name = trainer_params.get("name")
        if trainer_name not in trainer_map:
            raise ValueError(f"Trainer '{trainer_name}' not found in trainer_map.")
        trainer = trainer_map[trainer_name](trainer_params, model, fold_data, metadata)
        trained_model = trainer.run()

        model_tracker.set_model(trained_model)
        if getattr(trainer, "survey_model", None) is not None:
            model_tracker.set_auxiliary_artifact("survey_model", trainer.survey_model)
        if save_model:
            logger.log('model_save_path', model_tracker.get_model_info_save_path())
            model_tracker.save_model_details()
        else:
            model_tracker.reset_tracker()

        if len(fold_data_list) > 1:
            split_name = None
            if fold_data.get('test_loader') is not None:
                split_name = 'test'
            elif fold_data.get('val_loader') is not None:
                split_name = 'val'
            if split_name:
                entry = logger.build_entry_dict()
                metrics = {}
                for _branch_name, branch_prefix in summary_branches:
                    for metric in SUMMARY_METRICS:
                        metric_suffix = f"{branch_prefix}_{metric}" if branch_prefix else metric
                        key = f"{split_name}_{metric_suffix}"
                        if key in entry and entry[key] is not None:
                            metrics[metric_suffix] = entry[key]
                if metrics:
                    fold_metrics.append((split_name, metrics))

    if fold_metrics:
        split_name = fold_metrics[0][0]
        print(f"\nCross-validation summary ({split_name}):")
        for branch_name, branch_prefix in summary_branches:
            branch_values_found = False
            for metric in SUMMARY_METRICS:
                metric_suffix = f"{branch_prefix}_{metric}" if branch_prefix else metric
                values = [m.get(metric_suffix) for _, m in fold_metrics if m.get(metric_suffix) is not None]
                if values:
                    avg_value = float(np.mean(values))
                    logger.log(f"cv_avg_{split_name}_{metric_suffix}", avg_value)
                    branch_values_found = True
                    label_prefix = "Fusion" if branch_name == "fusion" else branch_name.upper()
                    print(f"Mean {label_prefix} {metric}: {avg_value:.4f}")
            if branch_values_found:
                print()

    logger_filename = config_copy.get("log_filename", "default_log.csv")
    logger.save(logger_filename)
    return logger.build_entry_dict()


def run_pipeline_config(
    config,
    data_map,
    preprocessor_map,
    model_map,
    trainer_map,
    save_model=False,
    log_filename_override=None,
):
    config = deepcopy(config)
    if log_filename_override:
        config["log_filename"] = log_filename_override

    summary_branches = _resolve_summary_branches(config)
    sweep_types = _resolve_label_sweep_types(config)
    seed_sweep = _resolve_seed_sweep(config)
    if sweep_types or seed_sweep:
        base_config = deepcopy(config)
        base_id = str(base_config.get("id", "config"))
        sweep_rows = []
        label_types = sweep_types if sweep_types is not None else [None]
        seeds = seed_sweep if seed_sweep is not None else [base_config.get("seed")]

        if sweep_types:
            print(f"Label sweep enabled: {', '.join(sweep_types)}")
        if seed_sweep:
            print(f"Seed sweep enabled: {', '.join(str(seed) for seed in seeds)}")

        total_runs = len(label_types) * len(seeds)
        run_idx = 0
        for seed in seeds:
            for label_type in label_types:
                run_idx += 1
                run_cfg = deepcopy(base_config)
                run_cfg.setdefault("dataset_params", {})
                run_cfg.setdefault("preprocessor_params", {})
                run_id = base_id

                if label_type is not None:
                    run_cfg["dataset_params"]["labels_csv"] = label_type
                    run_id = f"{run_id}_{label_type}"
                effective_label = run_cfg["dataset_params"].get("labels_csv")

                if seed is not None:
                    seed = int(seed)
                    run_cfg["seed"] = seed
                    run_cfg["preprocessor_params"]["seed"] = seed
                if seed_sweep:
                    run_id = f"{run_id}_seed{seed}"
                run_cfg["id"] = run_id

                print(
                    f"\nRun {run_idx}/{total_runs} -> "
                    f"labels_csv={effective_label}, seed={seed}"
                )
                entry = run_config(
                    run_cfg,
                    data_map,
                    preprocessor_map,
                    model_map,
                    trainer_map,
                    save_model=save_model,
                    clear_logger=True,
                )

                row = {
                    "label_type": effective_label,
                    "seed": seed,
                    "label_source": entry.get("label_source"),
                }
                for branch_name, branch_prefix in summary_branches:
                    for metric_name in SUMMARY_METRICS:
                        metric_key, metric_value = _extract_metric_for_summary(
                            entry, metric_name, prefix=branch_prefix
                        )
                        row[f"{branch_name}_{metric_name}"] = metric_value
                        row[f"{branch_name}_{metric_name}_key"] = metric_key
                sweep_rows.append(row)

        print("\nLabel sweep summary:")
        header = f"{'label':<10} {'seed':>6}"
        for branch_name, _branch_prefix in summary_branches:
            header += f" {f'{branch_name}(acc/p/r/f1/auc)':>38}"
        header += "  source"
        print(header)
        for row in sweep_rows:
            line = f"{row['label_type']:<10} {str(row.get('seed', '-')):>6}"
            for branch_name, _branch_prefix in summary_branches:
                line += f" {_format_metric_bundle(row, branch_name):>38}"
            line += f"  {row.get('label_source', 'N/A')}"
            print(line)

        if seed_sweep:
            grouped = defaultdict(list)
            for row in sweep_rows:
                grouped[row["label_type"]].append(row)

            print("\nSeed-averaged summary (mean +/- std):")
            print(
                f"{'label':<10} {'model':<8} "
                f"{'accuracy':>17} {'precision':>17} {'recall':>17} {'f1':>17} {'auc':>17}"
            )
            first_branch_name = summary_branches[0][0]
            for label_type in sorted(grouped.keys()):
                label_rows = grouped[label_type]
                for branch_name, _branch_prefix in summary_branches:
                    metric_cells = []
                    for metric_name in SUMMARY_METRICS:
                        key = f"{branch_name}_{metric_name}"
                        values = [_to_float_or_none(r.get(key)) for r in label_rows]
                        values = [v for v in values if v is not None]
                        if not values:
                            metric_cells.append(f"{'-':>17}")
                            continue
                        mean_v = float(np.mean(values))
                        std_v = float(np.std(values))
                        metric_cells.append(f"{mean_v:>8.4f} +/- {std_v:<6.4f}")
                    label_cell = label_type if branch_name == first_branch_name else ""
                    print(f"{label_cell:<10} {branch_name:<8} {' '.join(metric_cells)}")

        return {"label_sweep_results": sweep_rows}

    return run_config(
        config,
        data_map,
        preprocessor_map,
        model_map,
        trainer_map,
        save_model=save_model,
    )


def start_pipeline(config_file, data_map, preprocessor_map, model_map, trainer_map, save_model=False):
    if not config_file.endswith('.json'):
        raise ValueError("config_file must be a .json file")
    if os.path.isabs(config_file) or os.path.exists(config_file):
        config_path = config_file
    else:
        config_path = os.path.join('run_configs', config_file)

    with open(config_path, 'r') as f:
        config = json.load(f)

    return run_pipeline_config(
        config,
        data_map,
        preprocessor_map,
        model_map,
        trainer_map,
        save_model=save_model,
    )
