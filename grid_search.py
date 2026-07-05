import argparse
import hashlib
import itertools
import json
import os
from copy import deepcopy

from utils.lib_pipe import run_pipeline_config
from dataset_retrievers.tfrecord_retriever import load_belonging_tfrecords
from dataset_retrievers.raw_csv_retriever import load_belonging_raw_csvs
from dataset_retrievers.tfrecord_multimodal_retriever import (
    load_belonging_multimodal,
    load_belonging_multimodal_raw_csvs,
    load_belonging_multimodal_tfrecords,
)
from preprocessors.tfrecord_processor import tfrecord_preprocessor
from architectures.chrononet import ChronoNet
from architectures.cnn_gru import CNNGRU
from architectures.cnn_lstm import CNNLSTM
from architectures.raw_cnn_gru import RawCNNGRU
from architectures.raw_cnn_lstm import RawCNNLSTM
from architectures.raw_chrononet import RawChronoNet
from trainers.belonging_trainer import BelongingTrainer
from trainers.belonging_multimodal_trainer import BelongingMultimodalTrainer


DATA_MAP = {
    "load_belonging_tfrecords": load_belonging_tfrecords,
    "load_belonging_raw_csvs": load_belonging_raw_csvs,
    "load_belonging_multimodal": load_belonging_multimodal,
    "load_belonging_multimodal_raw_csvs": load_belonging_multimodal_raw_csvs,
    "load_belonging_multimodal_tfrecords": load_belonging_multimodal_tfrecords,
}
PREPROCESSOR_MAP = {
    "tfrecord_preprocessor": tfrecord_preprocessor,
    "sequence_preprocessor": tfrecord_preprocessor,
}
MODEL_MAP = {
    "ChronoNet": ChronoNet,
    "CNNGRU": CNNGRU,
    "CNNLSTM": CNNLSTM,
    "RawCNNGRU": RawCNNGRU,
    "RawCNNLSTM": RawCNNLSTM,
    "RawChronoNet": RawChronoNet,
}
TRAINER_MAP = {
    "BelongingTrainer": BelongingTrainer,
    "BelongingMultimodalTrainer": BelongingMultimodalTrainer,
}


def _set_by_path(config, path, value):
    parts = path.split(".")
    if not parts:
        raise ValueError("Empty parameter path.")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            raise KeyError(f"Invalid path segment '{part}' in '{path}'.")
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _load_config(config_file):
    if not config_file.endswith(".json"):
        raise ValueError("config_file must be a .json file")
    if os.path.isabs(config_file) or os.path.exists(config_file):
        config_path = config_file
    else:
        config_path = os.path.join("run_configs", config_file)
    with open(config_path, "r") as f:
        return json.load(f), config_path


def _grid_signature(config_path, base_config, metric, params, trials, max_trials):
    config_snapshot = deepcopy(base_config)
    config_snapshot.pop("grid_search", None)
    payload = {
        "config_path": os.path.abspath(config_path),
        "config_snapshot": config_snapshot,
        "metric": metric,
        "params": params,
        "trials": trials,
        "max_trials": max_trials,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _state_path(grid_search, base_config, signature):
    filename = grid_search.get("state_file")
    if filename:
        if os.path.isabs(filename):
            return filename
        if os.path.dirname(filename):
            return filename
        return os.path.join("logs", filename)

    base_id = base_config.get("id", "config")
    safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base_id)
    return os.path.join("logs", f"grid_search_state_{safe_id}_{signature[:8]}.json")


def _load_state(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _save_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _extract_metric(entry, metric):
    if metric in entry and entry[metric] is not None:
        return metric, entry[metric]

    candidates = [
        f"cv_avg_val_{metric}",
        f"cv_avg_test_{metric}",
        f"val_{metric}",
        f"test_{metric}",
        f"train_{metric}",
    ]
    for key in candidates:
        if key in entry and entry[key] is not None:
            return key, entry[key]
    return None, None


def _extract_sweep_metric(result, metric, aggregation):
    rows = result.get("label_sweep_results", [])
    values = []
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        values.append(float(value))

    if not values:
        return None, None

    if aggregation != "mean":
        raise ValueError(f"Unsupported grid_search.score_aggregation '{aggregation}'. Use 'mean'.")

    return f"label_sweep_{aggregation}_{metric}", float(sum(values) / len(values))


def _extract_trial_score(result, metric, aggregation):
    if isinstance(result, dict) and "label_sweep_results" in result:
        return _extract_sweep_metric(result, metric, aggregation)
    return _extract_metric(result, metric)


def run_grid_search(config_file=None, config=None, grid_search_override=None, save_model=False):
    if config is None:
        if config_file is None:
            raise ValueError("Either config_file or config must be provided.")
        base_config, config_path = _load_config(config_file)
    else:
        base_config = deepcopy(config)
        config_path = config_file or f"{base_config.get('id', 'config')}.json"

    if grid_search_override is not None:
        base_config["grid_search"] = deepcopy(grid_search_override)

    grid_search = base_config.get("grid_search")
    if not grid_search:
        raise ValueError("Config must include a 'grid_search' section.")

    metric = grid_search.get("metric", "accuracy")
    score_aggregation = grid_search.get("score_aggregation", "mean")
    explicit_trials = grid_search.get("trials")
    params = grid_search.get("params", {})
    param_items = []
    trial_param_sets = []

    if explicit_trials:
        trial_param_sets = [dict(trial) for trial in explicit_trials]
    else:
        if not params:
            raise ValueError("Config must include either 'grid_search.params' or 'grid_search.trials'.")
        param_items = list(params.items())
        values_list = [values if isinstance(values, list) else [values] for _, values in param_items]
        combos = list(itertools.product(*values_list))
        trial_param_sets = [
            {path: value for (path, _), value in zip(param_items, combo)}
            for combo in combos
        ]

    max_trials = grid_search.get("max_trials")
    if max_trials is not None:
        trial_param_sets = trial_param_sets[: int(max_trials)]

    signature = _grid_signature(config_path, base_config, metric, params, explicit_trials, max_trials)
    state_path = _state_path(grid_search, base_config, signature)
    resume = grid_search.get("resume", True)

    log_filename = grid_search.get("log_filename")
    if not log_filename:
        base_log = base_config.get("log_filename", "default_log.csv")
        log_filename = f"grid_search_{base_log}"

    print(f"Grid search: {len(trial_param_sets)} trials, metric={metric}, config={config_path}")

    best_score = None
    best_trial = None
    results = []
    completed = set()

    if resume:
        state = _load_state(state_path)
        if state and state.get("signature") == signature:
            completed = set(state.get("completed_trials", []))
            results = state.get("results", [])
            best_trial = state.get("best_trial")
            best_score = best_trial.get("score") if best_trial else None
            if completed:
                print(f"Resuming: {len(completed)} completed trial(s) from {state_path}")
        elif state:
            print(f"State file signature mismatch; starting fresh: {state_path}")

    for idx, trial_params in enumerate(trial_param_sets, start=1):
        if idx in completed:
            print(f"\nTrial {idx}/{len(trial_param_sets)} (skipped, already completed)")
            continue

        trial_config = deepcopy(base_config)
        for path, value in trial_params.items():
            _set_by_path(trial_config, path, value)

        trial_config["id"] = f"{trial_config.get('id', 'config')}_grid_{idx}"

        print(f"\nTrial {idx}/{len(trial_param_sets)}")
        for path, value in trial_params.items():
            print(f"  {path} = {value}")

        result = run_pipeline_config(
            trial_config,
            DATA_MAP,
            PREPROCESSOR_MAP,
            MODEL_MAP,
            TRAINER_MAP,
            save_model=save_model,
            log_filename_override=log_filename,
        )

        metric_key, score = _extract_trial_score(result, metric, score_aggregation)
        if score is None:
            raise RuntimeError(f"Metric '{metric}' not found in logs for trial {idx}.")

        results.append(
            {
                "trial": idx,
                "metric_key": metric_key,
                "score": score,
                "params": trial_params,
            }
        )

        if best_score is None or score > best_score:
            best_score = score
            best_trial = results[-1]

        print(f"  {metric_key} = {score:.4f}")

        completed.add(idx)
        state_payload = {
            "signature": signature,
            "completed_trials": sorted(completed),
            "results": results,
            "best_trial": best_trial,
        }
        _save_state(state_path, state_payload)

    if best_trial:
        print("\nBest trial:")
        print(f"  trial = {best_trial['trial']}")
        print(f"  {best_trial['metric_key']} = {best_trial['score']:.4f}")
        for path, value in best_trial["params"].items():
            print(f"  {path} = {value}")

    return {
        "config_path": config_path,
        "log_filename": log_filename,
        "state_path": state_path,
        "best_trial": best_trial,
        "results": results,
        "metric": metric,
        "score_aggregation": score_aggregation,
        "base_config": base_config,
    }


def main():
    parser = argparse.ArgumentParser(description="Grid search for pipeline configs")
    parser.add_argument("config_file", nargs="?", default="belonging_config_chrononet_tfrecord.json")
    parser.add_argument("-m", "--models", action="store_true", help="Save model artifacts and plots per trial")
    args = parser.parse_args()
    run_grid_search(config_file=args.config_file, save_model=args.models)


if __name__ == "__main__":
    main()
