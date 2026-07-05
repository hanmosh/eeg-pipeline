import argparse
import csv
import json
import os
from copy import deepcopy

from grid_search import run_grid_search


SUITE_ITEMS = (
    {
        "name": "2D ChronoNet",
        "config_file": "run_configs/belonging_config_multimodal_tfrecord.json",
        "hidden_param_path": "model_params.gru_hidden_size",
    },
    {
        "name": "2D CNNGRU",
        "config_file": "run_configs/belonging_config_multimodal_cnn_gru_tfrecord.json",
        "hidden_param_path": "model_params.gru_hidden_size",
    },
    {
        "name": "2D CNNLSTM",
        "config_file": "run_configs/belonging_config_multimodal_cnn_lstm_tfrecord.json",
        "hidden_param_path": "model_params.lstm_hidden_size",
    },
    {
        "name": "1D RawChronoNet",
        "config_file": "run_configs/belonging_config_multimodal_raw_chrononet_csv.json",
        "hidden_param_path": "model_params.gru_hidden_size",
    },
    {
        "name": "1D RawCNNGRU",
        "config_file": "run_configs/belonging_config_multimodal_switchable.json",
        "hidden_param_path": "model_params.gru_hidden_size",
    },
    {
        "name": "1D RawCNNLSTM",
        "config_file": "run_configs/belonging_config_multimodal_raw_cnn_lstm_csv.json",
        "hidden_param_path": "model_params.lstm_hidden_size",
    },
)

DEFAULT_BEST_CONFIG_DIR = os.path.join("run_configs", "grid_search_best_training")
DEFAULT_SUMMARY_JSON = os.path.join("logs", "grid_search_multimodal_training_suite_summary.json")
DEFAULT_SUMMARY_CSV = os.path.join("logs", "grid_search_multimodal_training_suite_summary.csv")

TRIAL_PRESETS = {
    "coarse_grid_2": (
        {"learning_rate": 0.0003, "dropout_rate": 0.3, "hidden_size": 16},
        {"learning_rate": 0.001, "dropout_rate": 0.2, "hidden_size": 8},
    ),
    "coarse_grid_3": (
        {"learning_rate": 0.0003, "dropout_rate": 0.3, "hidden_size": 16},
        {"learning_rate": 0.001, "dropout_rate": 0.2, "hidden_size": 8},
        {"learning_rate": 0.0001, "dropout_rate": 0.4, "hidden_size": 32},
    ),
}


def _set_by_path(config, path, value):
    parts = path.split(".")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _safe_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _build_preset_trials(item, args):
    trials = []
    for preset_trial in TRIAL_PRESETS[args.trial_preset]:
        trials.append(
            {
                "trainer_params.learning_rate": preset_trial["learning_rate"],
                "model_params.dropout_rate": preset_trial["dropout_rate"],
                item["hidden_param_path"]: preset_trial["hidden_size"],
            }
        )
    return trials


def _build_grid_override(item, args):
    config_filename = os.path.basename(item["config_file"])
    config_stem = os.path.splitext(config_filename)[0]
    run_suffix = "multimodal_training"
    if args.trial_preset:
        run_suffix = f"{run_suffix}_{_safe_name(args.trial_preset)}"

    grid_override = {
        "metric": "fusion_accuracy",
        "score_aggregation": "mean",
        "resume": True,
        "max_trials": args.max_trials,
        "log_filename": f"grid_search_{config_stem}_{run_suffix}.csv",
        "state_file": f"grid_search_state_{config_stem}_{run_suffix}.json",
    }

    if args.trial_preset:
        grid_override["trials"] = _build_preset_trials(item, args)
    else:
        grid_override["params"] = {
            "trainer_params.learning_rate": args.learning_rates,
            "model_params.dropout_rate": args.dropout_rates,
            item["hidden_param_path"]: args.hidden_sizes,
        }

    return grid_override


def _normalize_base_config(config, args):
    config = deepcopy(config)
    trainer_params = config.setdefault("trainer_params", {})
    trainer_params["fusion_mode"] = args.fusion_mode
    trainer_params["survey_weight"] = args.survey_weight
    trainer_params["fusion_threshold"] = args.fusion_threshold
    return config


def _save_best_config(result, output_dir):
    best_trial = result.get("best_trial")
    if not best_trial:
        return None

    best_config = deepcopy(result["base_config"])
    best_config.pop("grid_search", None)

    for path, value in best_trial["params"].items():
        _set_by_path(best_config, path, value)

    best_config["id"] = f"{best_config.get('id', 'config')}_best_training"

    os.makedirs(output_dir, exist_ok=True)
    config_filename = os.path.basename(result["config_path"])
    config_stem = os.path.splitext(config_filename)[0]
    output_path = os.path.join(output_dir, f"{config_stem}_best_training.json")
    with open(output_path, "w") as f:
        json.dump(best_config, f, indent=4)
    return output_path


def _write_summary(summary_rows, summary_json_path, summary_csv_path):
    os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    fieldnames = [
        "suite_name",
        "config_file",
        "metric",
        "metric_key",
        "score",
        "learning_rate",
        "dropout_rate",
        "hidden_size_param",
        "hidden_size",
        "state_path",
        "log_filename",
        "best_config_path",
    ]
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Run the same multimodal training-parameter grid search for all six architectures."
    )
    parser.add_argument("-m", "--models", action="store_true", help="Save model artifacts and plots per trial")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional cap on the number of trials per model")
    parser.add_argument(
        "--trial-preset",
        choices=sorted(TRIAL_PRESETS.keys()),
        help="Use a curated coarse-grid trial list instead of the full Cartesian grid",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.0001, 0.0003, 0.001],
        help="Learning-rate grid shared by all models",
    )
    parser.add_argument(
        "--dropout-rates",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4],
        help="Dropout-rate grid shared by all models",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="Hidden-size grid shared by all models",
    )
    parser.add_argument(
        "--fusion-mode",
        default="fixed_weight",
        help="Shared fusion mode applied to every model during this training-parameter search",
    )
    parser.add_argument(
        "--survey-weight",
        type=float,
        default=0.55,
        help="Shared survey weight applied to every model during this training-parameter search",
    )
    parser.add_argument(
        "--fusion-threshold",
        type=float,
        default=0.5,
        help="Shared fusion threshold applied to every model during this training-parameter search",
    )
    parser.add_argument(
        "--best-config-dir",
        default=DEFAULT_BEST_CONFIG_DIR,
        help="Directory where best-config snapshots will be written",
    )
    parser.add_argument(
        "--summary-json",
        default=DEFAULT_SUMMARY_JSON,
        help="JSON summary output path",
    )
    parser.add_argument(
        "--summary-csv",
        default=DEFAULT_SUMMARY_CSV,
        help="CSV summary output path",
    )
    args = parser.parse_args()

    if args.trial_preset:
        preset_suffix = _safe_name(args.trial_preset)
        if args.best_config_dir == DEFAULT_BEST_CONFIG_DIR:
            args.best_config_dir = os.path.join("run_configs", f"grid_search_best_training_{preset_suffix}")
        if args.summary_json == DEFAULT_SUMMARY_JSON:
            args.summary_json = os.path.join(
                "logs", f"grid_search_multimodal_training_suite_summary_{preset_suffix}.json"
            )
        if args.summary_csv == DEFAULT_SUMMARY_CSV:
            args.summary_csv = os.path.join(
                "logs", f"grid_search_multimodal_training_suite_summary_{preset_suffix}.csv"
            )

    summary_rows = []
    total = len(SUITE_ITEMS)

    for idx, item in enumerate(SUITE_ITEMS, start=1):
        print(f"\nSuite run {idx}/{total}: {item['name']}")
        base_config = _normalize_base_config(_load_json(item["config_file"]), args)
        grid_override = _build_grid_override(item, args)
        result = run_grid_search(
            config_file=item["config_file"],
            config=base_config,
            grid_search_override=grid_override,
            save_model=args.models,
        )

        best_trial = result.get("best_trial")
        best_config_path = _save_best_config(result, args.best_config_dir)

        row = {
            "suite_name": item["name"],
            "config_file": item["config_file"],
            "metric": result.get("metric"),
            "metric_key": best_trial.get("metric_key") if best_trial else None,
            "score": best_trial.get("score") if best_trial else None,
            "learning_rate": None,
            "dropout_rate": None,
            "hidden_size_param": item["hidden_param_path"],
            "hidden_size": None,
            "state_path": result.get("state_path"),
            "log_filename": result.get("log_filename"),
            "best_config_path": best_config_path,
        }
        if best_trial:
            row["learning_rate"] = best_trial["params"].get("trainer_params.learning_rate")
            row["dropout_rate"] = best_trial["params"].get("model_params.dropout_rate")
            row["hidden_size"] = best_trial["params"].get(item["hidden_param_path"])
        summary_rows.append(row)

    _write_summary(summary_rows, args.summary_json, args.summary_csv)

    print("\nSuite summary saved:")
    print(f"  JSON: {args.summary_json}")
    print(f"  CSV:  {args.summary_csv}")
    print(f"  Best configs: {args.best_config_dir}")


if __name__ == "__main__":
    main()
