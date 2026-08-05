import copy
import random
from datetime import datetime

import numpy as np
import torch

from utils.log import logger, model_tracker
from utils.pipeline_setup import load_json_config


METRIC_SUFFIXES = ("loss", "accuracy", "precision", "recall", "f1", "auc")
METRIC_PREFIX_PRIORITY = (
    "cv_avg_test_",
    "cv_avg_val_",
    "test_",
    "val_",
    "train_",
)
MODEL_ARTIFACT_NAME_MAP = {
    "ChronoNet": "2D_ChronoNet",
    "CNNGRU": "2D_CNNGRU",
    "CNNLSTM": "2D_CNNLSTM",
    "RawChronoNet": "1D_ChronoNet",
    "RawCNNGRU": "1D_CNNGRU",
    "RawCNNLSTM": "1D_CNNLSTM",
}
LABEL_NAME_MAP = {
    "specific": "Specific",
    "composite": "Composite",
    "factor": "Factor",
    "weighted": "Weighted",
}


def _is_metric_key(key):
    return any(key.endswith(f"_{suffix}") or key == suffix for suffix in METRIC_SUFFIXES)


def _coerce_numeric(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _timestamp_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _set_random_seed(seed, deterministic=False):
    if seed is None:
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _resolve_component(component_params, component_map, component_kind):
    component_name = component_params.get("name")
    if not component_name:
        raise ValueError(f"Missing {component_kind} name in config.")
    if component_name not in component_map:
        raise KeyError(f"Unknown {component_kind} '{component_name}'.")
    return component_map[component_name]


def _normalize_label_value(label_value):
    if label_value is None:
        return None
    normalized = str(label_value).strip().lower()
    normalized = normalized.replace("-", "").replace("_", "").replace(" ", "")
    return normalized


def _build_model_artifact_name(config, run_metadata):
    model_name = config.get("model_params", {}).get("name")
    model_dir = MODEL_ARTIFACT_NAME_MAP.get(model_name, str(model_name or "model"))

    label_mode = config.get("dataset_params", {}).get("labels_csv")
    label_dir = LABEL_NAME_MAP.get(_normalize_label_value(label_mode), str(label_mode or "Label"))

    path_parts = []
    artifact_family = config.get("model_artifact_family")
    if artifact_family:
        path_parts.append(str(artifact_family))
    path_parts.extend([model_dir, label_dir])

    cv_fold = run_metadata.get("cv_fold")
    cv_repeat = run_metadata.get("cv_repeat", run_metadata.get("cv_repeat_index"))
    if cv_fold is not None:
        fold_name = f"CV_{int(cv_fold)}"
        if cv_repeat is not None and int(cv_repeat) > 1:
            fold_name = f"Repeat_{int(cv_repeat)}\\{fold_name}"
        path_parts.append(fold_name)

    return "\\".join(path_parts)


def _log_run_header(config):
    logger.log("timestamp", _timestamp_now())
    logger.log("config_id", config.get("id", "config"))
    if "seed" in config:
        logger.log("seed", config.get("seed"))
    if "deterministic" in config:
        logger.log("deterministic", config.get("deterministic"))


def _collect_split_metric_values(entry, split_prefix):
    metric_values = {}
    for key, value in entry.items():
        if not key.startswith(split_prefix) or not _is_metric_key(key):
            continue
        numeric_value = _coerce_numeric(value)
        if numeric_value is None:
            continue
        metric_values[key] = numeric_value
    return metric_values


def _aggregate_cv_metrics(fold_entries):
    if not fold_entries:
        return {}

    split_prefix = None
    for candidate in ("test_", "val_"):
        if any(
            key.startswith(candidate) and _is_metric_key(key)
            for entry in fold_entries
            for key in entry.keys()
        ):
            split_prefix = candidate
            break

    if split_prefix is None:
        return {}

    collected = {}
    for entry in fold_entries:
        for key, value in _collect_split_metric_values(entry, split_prefix).items():
            collected.setdefault(key, []).append(value)

    return {f"cv_avg_{key}": float(sum(values) / len(values)) for key, values in collected.items() if values}


def _flatten_primary_metrics(entry):
    flattened = {}
    for prefix in METRIC_PREFIX_PRIORITY:
        for key, value in entry.items():
            if not key.startswith(prefix) or not _is_metric_key(key):
                continue
            numeric_value = _coerce_numeric(value)
            if numeric_value is None:
                continue
            alias = key[len(prefix):]
            if alias not in flattened:
                flattened[alias] = numeric_value
    return flattened


def _build_fold_metadata(base_metadata, fold_data):
    fold_metadata = dict(base_metadata)
    fold_metadata.update(
        {
            "cv_fold": fold_data.get("cv_fold"),
            "cv_fold_in_repeat": fold_data.get("cv_fold_in_repeat"),
            "cv_repeat": fold_data.get("cv_repeat"),
            "cv_repeat_index": fold_data.get("cv_repeat_index"),
            "cv_total_folds": fold_data.get("cv_total_folds"),
        }
    )
    return fold_metadata


def _get_enabled_sweep_values(config, sweep_name, values_key):
    sweep_config = config.get(sweep_name) or {}
    if not sweep_config.get("enabled"):
        return None

    values = list(sweep_config.get(values_key) or [])
    if not values:
        raise ValueError(f"{sweep_name}.enabled is true, but no {values_key} were provided.")
    return values


def _aggregate_numeric_rows(rows, *, ignored_keys=()):
    aggregated = {}
    metric_names = sorted(
        {key for row in rows for key in row.keys() if key not in set(ignored_keys)}
    )
    for metric_name in metric_names:
        metric_values = [row[metric_name] for row in rows if metric_name in row]
        if metric_values:
            aggregated[metric_name] = float(sum(metric_values) / len(metric_values))
    return aggregated


def _build_seed_sweep_config(config, seed):
    seed_config = copy.deepcopy(config)
    seed_config["seed"] = int(seed)
    seed_config["id"] = f"{seed_config.get('id', 'config')}_seed_{int(seed)}"
    seed_config["seed_sweep"] = {"enabled": False}
    return seed_config


def _build_label_sweep_config(config, label_type):
    label_config = copy.deepcopy(config)
    label_config["label_sweep"] = {"enabled": False}
    label_config.setdefault("dataset_params", {})
    label_config["dataset_params"]["labels_csv"] = label_type
    label_config["id"] = f"{label_config.get('id', 'config')}_{label_type}"
    return label_config


def _run_one_training_pass(config, data, metadata, model_map, trainer_map, save_model):
    model_params = copy.deepcopy(config.get("model_params", {}))
    trainer_params = copy.deepcopy(config.get("trainer_params", {}))

    model_class = _resolve_component(model_params, model_map, "model")
    trainer_class = _resolve_component(trainer_params, trainer_map, "trainer")

    model_tracker.reset_tracker()
    model = model_class(model_params, metadata)
    model_tracker.set_model(model)
    model_tracker.set_config(config)

    artifact_name = _build_model_artifact_name(config, metadata)
    model_tracker.set_model_name(artifact_name, save_model=save_model)
    if model_tracker.filepath is not None:
        logger.log("model_save_path", model_tracker.filepath)

    trainer = trainer_class(trainer_params, model, data, metadata)
    trained_model = trainer.run()

    if save_model:
        model_tracker.set_model(trained_model)
        if model_tracker.filepath is not None:
            logger.log("model_save_path", model_tracker.filepath)
        model_tracker.save_model_details()

    return trained_model


def _run_single_config(config, data_map, preprocessor_map, model_map, trainer_map, save_model=False, log_filename_override=None):
    logger.clear()
    model_tracker.reset_tracker()

    config = copy.deepcopy(config)
    seed = config.get("seed")
    deterministic = bool(config.get("deterministic", False))
    _set_random_seed(seed, deterministic=deterministic)
    _log_run_header(config)

    dataset_params = copy.deepcopy(config.get("dataset_params", {}))
    preprocessor_params = copy.deepcopy(config.get("preprocessor_params", {}))

    dataset_loader = _resolve_component(dataset_params, data_map, "dataset")
    preprocessor = _resolve_component(preprocessor_params, preprocessor_map, "preprocessor")

    metadata = {}
    X, y, metadata = dataset_loader(dataset_params, metadata)
    if seed is not None and "seed" not in metadata:
        metadata["seed"] = int(seed)

    processed_data, metadata = preprocessor(preprocessor_params, X, y, metadata)

    if isinstance(processed_data, list):
        fold_entries = []
        for fold_data in processed_data:
            _set_random_seed(seed, deterministic=deterministic)
            fold_metadata = _build_fold_metadata(metadata, fold_data)

            fold_context = {
                "cv_fold": fold_data.get("cv_fold"),
                "cv_fold_in_repeat": fold_data.get("cv_fold_in_repeat"),
                "cv_repeat": fold_data.get("cv_repeat", fold_data.get("cv_repeat_index")),
                "cv_total_folds": fold_data.get("cv_total_folds"),
            }
            for key, value in fold_context.items():
                if value is not None:
                    logger.log(key, value)

            _run_one_training_pass(config, fold_data, fold_metadata, model_map, trainer_map, save_model)
            fold_entries.append(dict(logger.build_entry_dict()))
            model_tracker.reset_tracker()

        for key, value in _aggregate_cv_metrics(fold_entries).items():
            logger.log(key, value)

        result = dict(logger.build_entry_dict())
        result["fold_results"] = fold_entries
    else:
        _run_one_training_pass(config, processed_data, dict(metadata), model_map, trainer_map, save_model)
        result = dict(logger.build_entry_dict())
        model_tracker.reset_tracker()

    log_filename = log_filename_override or config.get("log_filename")
    if log_filename:
        logger.save(log_filename)

    logger.clear()
    return result


def _run_seed_sweep(config, data_map, preprocessor_map, model_map, trainer_map, save_model=False, log_filename_override=None):
    seeds = _get_enabled_sweep_values(config, "seed_sweep", "seeds")
    if seeds is None:
        return _run_single_config(
            config,
            data_map,
            preprocessor_map,
            model_map,
            trainer_map,
            save_model=save_model,
            log_filename_override=log_filename_override,
        )

    seed_results = []
    for seed in seeds:
        run_result = _run_single_config(
            _build_seed_sweep_config(config, seed),
            data_map,
            preprocessor_map,
            model_map,
            trainer_map,
            save_model=save_model,
            log_filename_override=log_filename_override,
        )
        seed_results.append({"seed": int(seed), **_flatten_primary_metrics(run_result)})

    return {
        "config_id": config.get("id", "config"),
        "seed_sweep_results": seed_results,
        **_aggregate_numeric_rows(seed_results, ignored_keys=("seed",)),
    }


def run_pipeline_config(config, data_map, preprocessor_map, model_map, trainer_map, save_model=False, log_filename_override=None):
    config = copy.deepcopy(config)
    label_types = _get_enabled_sweep_values(config, "label_sweep", "label_types")
    if label_types is None:
        return _run_seed_sweep(
            config,
            data_map,
            preprocessor_map,
            model_map,
            trainer_map,
            save_model=save_model,
            log_filename_override=log_filename_override,
        )

    sweep_results = []
    for label_type in label_types:
        label_config = _build_label_sweep_config(config, label_type)
        run_result = _run_seed_sweep(
            label_config,
            data_map,
            preprocessor_map,
            model_map,
            trainer_map,
            save_model=save_model,
            log_filename_override=log_filename_override,
        )
        sweep_row = {
            "label_type": label_type,
            "config_id": run_result.get("config_id", label_config["id"]),
            **_flatten_primary_metrics(run_result),
        }
        if "fold_results" in run_result:
            sweep_row["fold_results"] = run_result["fold_results"]
        if "seed_sweep_results" in run_result:
            sweep_row["seed_sweep_results"] = run_result["seed_sweep_results"]
        sweep_results.append(sweep_row)

    return {
        "config_id": config.get("id", "config"),
        "label_sweep_results": sweep_results,
    }


def start_pipeline(config_file, data_map, preprocessor_map, model_map, trainer_map, save_model=False):
    config, _config_path = load_json_config(config_file)
    return run_pipeline_config(
        config,
        data_map,
        preprocessor_map,
        model_map,
        trainer_map,
        save_model=save_model,
    )
