import argparse
import itertools
import json
import os
from copy import deepcopy

from utils.lib_pipe import run_config
from dataset_retrievers.spectrogram_retriever import load_belonging_spectrograms
from dataset_retrievers.tfrecord_retriever import load_belonging_tfrecords
from preprocessors.spectrogram_processor import spectrogram_preprocessor
from preprocessors.tfrecord_processor import tfrecord_preprocessor
from architectures.chrononet import ChronoNet
from trainers.belonging_trainer import BelongingTrainer


DATA_MAP = {
    "load_belonging_spectrograms": load_belonging_spectrograms,
    "load_belonging_tfrecords": load_belonging_tfrecords,
}
PREPROCESSOR_MAP = {
    "spectrogram_preprocessor": spectrogram_preprocessor,
    "tfrecord_preprocessor": tfrecord_preprocessor,
}
MODEL_MAP = {
    "ChronoNet": ChronoNet,
}
TRAINER_MAP = {
    "BelongingTrainer": BelongingTrainer,
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


def _extract_metric(entry, metric):
    candidates = [
        f"cv_avg_val_{metric}",
        f"cv_avg_test_{metric}",
        f"val_{metric}",
        f"test_{metric}",
        f"train_{metric}",
    ]
    for key in candidates:
        if key in entry:
            return key, entry[key]
    return None, None


def _load_config(config_file):
    if not config_file.endswith(".json"):
        raise ValueError("config_file must be a .json file")
    if os.path.isabs(config_file) or os.path.exists(config_file):
        config_path = config_file
    else:
        config_path = os.path.join("run_configs", config_file)
    with open(config_path, "r") as f:
        return json.load(f), config_path


def main():
    parser = argparse.ArgumentParser(description="Grid search for pipeline configs")
    parser.add_argument("config_file", nargs="?", default="belonging_config_chrononet_tfrecord.json")
    args = parser.parse_args()

    base_config, config_path = _load_config(args.config_file)
    grid_search = base_config.get("grid_search")
    if not grid_search:
        raise ValueError("Config must include a 'grid_search' section.")

    metric = grid_search.get("metric", "accuracy")
    params = grid_search.get("params", {})
    if not params:
        raise ValueError("'grid_search.params' is required.")

    param_items = list(params.items())
    values_list = [values if isinstance(values, list) else [values] for _, values in param_items]
    combos = list(itertools.product(*values_list))

    max_trials = grid_search.get("max_trials")
    if max_trials is not None:
        combos = combos[: int(max_trials)]

    log_filename = grid_search.get("log_filename")
    if not log_filename:
        base_log = base_config.get("log_filename", "default_log.csv")
        log_filename = f"grid_search_{base_log}"

    print(f"Grid search: {len(combos)} trials, metric={metric}, config={config_path}")

    best_score = None
    best_trial = None
    results = []

    for idx, combo in enumerate(combos, start=1):
        trial_config = deepcopy(base_config)
        trial_params = {}
        for (path, _), value in zip(param_items, combo):
            _set_by_path(trial_config, path, value)
            trial_params[path] = value

        base_id = trial_config.get("id", "config")
        trial_config["id"] = f"{base_id}_grid_{idx}"

        print(f"\nTrial {idx}/{len(combos)}")
        for path, value in trial_params.items():
            print(f"  {path} = {value}")

        entry = run_config(
            trial_config,
            DATA_MAP,
            PREPROCESSOR_MAP,
            MODEL_MAP,
            TRAINER_MAP,
            save_model=False,
            log_filename_override=log_filename,
            clear_logger=True,
        )

        metric_key, score = _extract_metric(entry, metric)
        if score is None:
            raise RuntimeError(f"Metric '{metric}' not found in logs for trial {idx}.")

        results.append({
            "trial": idx,
            "metric_key": metric_key,
            "score": score,
            "params": trial_params,
        })

        if best_score is None or score > best_score:
            best_score = score
            best_trial = results[-1]

        print(f"  {metric_key} = {score:.4f}")

    if best_trial:
        print("\nBest trial:")
        print(f"  trial = {best_trial['trial']}")
        print(f"  {best_trial['metric_key']} = {best_trial['score']:.4f}")
        for path, value in best_trial["params"].items():
            print(f"  {path} = {value}")


if __name__ == "__main__":
    main()
