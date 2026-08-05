import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from statistics import stdev
from typing import Any, Dict, List, Optional, Tuple, Union


SuiteConfig = Tuple[str, str]

LABEL_DISPLAY_NAMES = {
    "specific": "Specific",
    "composite": "Composite",
    "factor": "Factor",
    "weighted": "Weighted",
}
LABEL_ORDER = {label: idx for idx, label in enumerate(("specific", "composite", "factor", "weighted"))}
ARCHITECTURE_ORDER = {
    "chrononet": 0,
    "cnn_gru": 1,
    "cnn_lstm": 2,
}
SECTION_ORDER = {
    "raw_1d": 0,
    "scalogram_2d": 1,
}
SECTION_TITLES = (
    ("raw_1d", "RAW 1D EEG Models"),
    ("scalogram_2d", "SCALOGRAM 2D Models"),
)
TABLE_HEADER = (
    "| Label | Model | EEG Acc | EEG Prec | EEG Rec | EEG F1 | EEG AUC | "
    "Survey Acc | Survey Prec | Survey Rec | Survey F1 | Survey AUC | "
    "Fusion Acc | Fusion Prec | Fusion Rec | Fusion F1 | Fusion AUC |"
)
TABLE_DIVIDER = (
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)


@dataclass(frozen=True)
class SuiteDefinition:
    window_size: Union[str, int]
    configs: Tuple[SuiteConfig, ...]


@dataclass(frozen=True)
class SuiteRunSummary:
    architecture_slug: str
    config_path: str
    config: Dict[str, Any]
    result: Dict[str, Any]


SUITES = {
    "full_128": SuiteDefinition(
        window_size=128,
        configs=(
            ("chrononet_2d", "run_configs/belonging_config_multimodal_tfrecord_128_full_compare.json"),
            ("cnn_lstm_2d", "run_configs/belonging_config_multimodal_cnn_lstm_tfrecord_128_full_compare.json"),
            ("cnn_gru_2d", "run_configs/belonging_config_multimodal_cnn_gru_tfrecord_128_full_compare.json"),
            ("chrononet_1d", "run_configs/belonging_config_multimodal_raw_chrononet_csv_128_full_compare.json"),
            ("cnn_lstm_1d", "run_configs/belonging_config_multimodal_raw_cnn_lstm_csv_128_full_compare.json"),
            ("cnn_gru_1d", "run_configs/belonging_config_multimodal_raw_cnn_gru_csv_128_full_compare.json"),
        ),
    ),
    "full_256": SuiteDefinition(
        window_size=256,
        configs=(
            ("chrononet_2d", "run_configs/belonging_config_multimodal_tfrecord_256_full_compare.json"),
            ("cnn_lstm_2d", "run_configs/belonging_config_multimodal_cnn_lstm_tfrecord_256_full_compare.json"),
            ("cnn_gru_2d", "run_configs/belonging_config_multimodal_cnn_gru_tfrecord_256_full_compare.json"),
            ("chrononet_1d", "run_configs/belonging_config_multimodal_raw_chrononet_csv_256_4_30hz.json"),
            ("cnn_lstm_1d", "run_configs/belonging_config_multimodal_raw_cnn_lstm_csv_256_4_30hz.json"),
            ("cnn_gru_1d", "run_configs/belonging_config_multimodal_raw_cnn_gru_csv_256_4_30hz.json"),
        ),
    ),
    "full_512": SuiteDefinition(
        window_size=512,
        configs=(
            ("chrononet_2d", "run_configs/belonging_config_multimodal_tfrecord_512_full_compare.json"),
            ("cnn_lstm_2d", "run_configs/belonging_config_multimodal_cnn_lstm_tfrecord_512_full_compare.json"),
            ("cnn_gru_2d", "run_configs/belonging_config_multimodal_cnn_gru_tfrecord_512_full_compare.json"),
            ("chrononet_1d", "run_configs/belonging_config_multimodal_raw_chrononet_csv_512_full_compare.json"),
            ("cnn_lstm_1d", "run_configs/belonging_config_multimodal_raw_cnn_lstm_csv_512_full_compare.json"),
            ("cnn_gru_1d", "run_configs/belonging_config_multimodal_raw_cnn_gru_csv_512_full_compare.json"),
        ),
    ),
}


def build_suite_parser(window_size, total_runs):
    parser = argparse.ArgumentParser(
        description=(
            "Run the multimodal comparison suite on the generated "
            f"{window_size} setup ({total_runs} architectures)."
        )
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save trained model artifacts for each run.",
    )
    return parser


def _build_log_filename(window_size, architecture_slug):
    return f"multimodal_full_compare_{window_size}_{architecture_slug}.csv"


def _prepare_config(config_path, architecture_slug):
    from utils.pipeline_setup import load_json_config

    config, _ = load_json_config(config_path)
    config = deepcopy(config)
    config["id"] = f"{config.get('id', architecture_slug)}_{architecture_slug}_suite"
    return config


def _load_pipeline_runtime():
    from utils.lib_pipe import run_pipeline_config
    from utils.log import logger
    from utils.pipeline_setup import DATA_MAP, MODEL_MAP, PREPROCESSOR_MAP, TRAINER_MAP

    return logger, run_pipeline_config, DATA_MAP, PREPROCESSOR_MAP, MODEL_MAP, TRAINER_MAP


def _format_metric_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _format_accuracy_value(mean_value: Optional[float], std_value: Optional[float]) -> str:
    mean_text = _format_metric_value(mean_value)
    if not mean_text:
        return ""
    if std_value is None:
        return mean_text
    return f"{mean_text} +/- {_format_metric_value(std_value)}"


def _metric_candidates(metric_name: str) -> Tuple[str, ...]:
    return (
        metric_name,
        f"cv_avg_test_{metric_name}",
        f"cv_avg_val_{metric_name}",
        f"test_{metric_name}",
        f"val_{metric_name}",
        f"train_{metric_name}",
    )


def _get_metric_value(entry: Dict[str, Any], metric_name: str) -> Optional[float]:
    for key in _metric_candidates(metric_name):
        if key not in entry:
            continue
        value = entry.get(key)
        if value is None or value == "":
            continue
        return float(value)
    return None


def _compute_fold_accuracy_sd(label_result: Dict[str, Any], metric_name: str) -> Optional[float]:
    fold_results = label_result.get("fold_results") or []
    values: List[float] = []
    for fold_result in fold_results:
        value = _get_metric_value(fold_result, metric_name)
        if value is not None:
            values.append(value)
    if len(values) < 2:
        return None
    return float(stdev(values))


def _section_name(architecture_slug: str) -> str:
    return "raw_1d" if architecture_slug.endswith("_1d") else "scalogram_2d"


def _architecture_order(architecture_slug: str) -> int:
    for prefix, order in ARCHITECTURE_ORDER.items():
        if architecture_slug.startswith(prefix):
            return order
    return len(ARCHITECTURE_ORDER)


def _model_display_name(config: Dict[str, Any], architecture_slug: str) -> str:
    model_name = str(config.get("model_params", {}).get("name", architecture_slug))
    model_prefix = "1D" if architecture_slug.endswith("_1d") else "2D"
    return f"{model_prefix} {model_name}"


def _label_display_name(label_type: Any) -> str:
    normalized = str(label_type or "").strip().lower()
    return LABEL_DISPLAY_NAMES.get(normalized, str(label_type))


def _build_suite_rows(suite_run_summaries: List[SuiteRunSummary]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for suite_run in suite_run_summaries:
        label_results = suite_run.result.get("label_sweep_results") or []
        if not label_results:
            label_results = [
                {
                    **suite_run.result,
                    "label_type": suite_run.config.get("dataset_params", {}).get("labels_csv"),
                }
            ]

        for label_result in label_results:
            label_type = label_result.get("label_type")
            normalized_label = str(label_type or "").strip().lower()
            rows.append(
                {
                    "section": _section_name(suite_run.architecture_slug),
                    "section_order": SECTION_ORDER[_section_name(suite_run.architecture_slug)],
                    "label": _label_display_name(label_type),
                    "label_order": LABEL_ORDER.get(normalized_label, len(LABEL_ORDER)),
                    "model": _model_display_name(suite_run.config, suite_run.architecture_slug),
                    "model_order": _architecture_order(suite_run.architecture_slug),
                    "eeg_accuracy": _get_metric_value(label_result, "eeg_accuracy"),
                    "eeg_accuracy_sd": _compute_fold_accuracy_sd(label_result, "eeg_accuracy"),
                    "eeg_precision": _get_metric_value(label_result, "eeg_precision"),
                    "eeg_recall": _get_metric_value(label_result, "eeg_recall"),
                    "eeg_f1": _get_metric_value(label_result, "eeg_f1"),
                    "eeg_auc": _get_metric_value(label_result, "eeg_auc"),
                    "survey_accuracy": _get_metric_value(label_result, "survey_accuracy"),
                    "survey_accuracy_sd": _compute_fold_accuracy_sd(label_result, "survey_accuracy"),
                    "survey_precision": _get_metric_value(label_result, "survey_precision"),
                    "survey_recall": _get_metric_value(label_result, "survey_recall"),
                    "survey_f1": _get_metric_value(label_result, "survey_f1"),
                    "survey_auc": _get_metric_value(label_result, "survey_auc"),
                    "fusion_accuracy": _get_metric_value(label_result, "fusion_accuracy"),
                    "fusion_accuracy_sd": _compute_fold_accuracy_sd(label_result, "fusion_accuracy"),
                    "fusion_precision": _get_metric_value(label_result, "fusion_precision"),
                    "fusion_recall": _get_metric_value(label_result, "fusion_recall"),
                    "fusion_f1": _get_metric_value(label_result, "fusion_f1"),
                    "fusion_auc": _get_metric_value(label_result, "fusion_auc"),
                }
            )

    rows.sort(key=lambda row: (row["section_order"], row["label_order"], row["model_order"], row["model"]))
    return rows


def _build_table_row(row: Dict[str, Any]) -> str:
    values = [
        row["label"],
        row["model"],
        _format_accuracy_value(row["eeg_accuracy"], row["eeg_accuracy_sd"]),
        _format_metric_value(row["eeg_precision"]),
        _format_metric_value(row["eeg_recall"]),
        _format_metric_value(row["eeg_f1"]),
        _format_metric_value(row["eeg_auc"]),
        _format_accuracy_value(row["survey_accuracy"], row["survey_accuracy_sd"]),
        _format_metric_value(row["survey_precision"]),
        _format_metric_value(row["survey_recall"]),
        _format_metric_value(row["survey_f1"]),
        _format_metric_value(row["survey_auc"]),
        _format_accuracy_value(row["fusion_accuracy"], row["fusion_accuracy_sd"]),
        _format_metric_value(row["fusion_precision"]),
        _format_metric_value(row["fusion_recall"]),
        _format_metric_value(row["fusion_f1"]),
        _format_metric_value(row["fusion_auc"]),
    ]
    return "| " + " | ".join(values) + " |"


def build_suite_summary_markdown(window_size: Union[str, int], suite_run_summaries: List[SuiteRunSummary]) -> str:
    rows = _build_suite_rows(suite_run_summaries)
    if not rows:
        return ""

    best_row = max(
        rows,
        key=lambda row: float("-inf") if row["fusion_accuracy"] is None else row["fusion_accuracy"],
    )

    lines = [
        f"# {window_size} Suite Result Tables",
        "",
        "Means are taken directly from the suite run that just completed. "
        "Accuracy columns are reported as mean +/- SD across CV test folds when fold-level metrics are available.",
        "",
        "Best overall fusion result: "
        f"{best_row['label']} / {best_row['model']} with fusion accuracy "
        f"{_format_accuracy_value(best_row['fusion_accuracy'], best_row['fusion_accuracy_sd'])}, "
        f"fusion F1 {_format_metric_value(best_row['fusion_f1'])}, "
        f"and fusion AUC {_format_metric_value(best_row['fusion_auc'])}.",
    ]

    for section_name, section_title in SECTION_TITLES:
        section_rows = [row for row in rows if row["section"] == section_name]
        if not section_rows:
            continue
        lines.extend(["", f"## {section_title}", TABLE_HEADER, TABLE_DIVIDER])
        lines.extend(_build_table_row(row) for row in section_rows)

    return "\n".join(lines).rstrip()


def _run_suite_config(size_text, run_index, total_runs, architecture_slug, config_path, save_models, runtime):
    logger, run_pipeline_config, data_map, preprocessor_map, model_map, trainer_map = runtime
    log_filename = _build_log_filename(size_text, architecture_slug)
    config = _prepare_config(config_path, architecture_slug)
    print(f"[{run_index}/{total_runs}] {architecture_slug} (log: logs/{log_filename})")
    logger.clear()
    result = run_pipeline_config(
        config,
        data_map,
        preprocessor_map,
        model_map,
        trainer_map,
        save_model=save_models,
        log_filename_override=log_filename,
    )
    print()
    return SuiteRunSummary(
        architecture_slug=architecture_slug,
        config_path=config_path,
        config=config,
        result=result,
    )


def get_suite_definition(suite_name):
    try:
        return SUITES[suite_name]
    except KeyError as exc:
        available = ", ".join(sorted(SUITES))
        raise ValueError(f"Unknown suite '{suite_name}'. Available suites: {available}") from exc


def build_suite_main(suite_name):
    def main(argv=None):
        run_named_multimodal_suite(suite_name, argv=argv)

    return main


def run_multimodal_suite(window_size, suite_configs, argv=None):
    total_runs = len(suite_configs)
    args = build_suite_parser(window_size, total_runs).parse_args(argv)
    size_text = str(window_size)
    runtime = _load_pipeline_runtime()
    suite_run_summaries: List[SuiteRunSummary] = []

    print(
        f"Starting {size_text} multimodal comparison suite at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"Total config runs: {total_runs}")
    print()

    for run_index, (architecture_slug, config_path) in enumerate(suite_configs, start=1):
        suite_run_summaries.append(
            _run_suite_config(
                size_text,
                run_index,
                total_runs,
                architecture_slug,
                config_path,
                args.save_models,
                runtime,
            )
        )

    print(
        f"Completed {size_text} multimodal comparison suite at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    summary_markdown = build_suite_summary_markdown(window_size, suite_run_summaries)
    if summary_markdown:
        print()
        print(summary_markdown)


def run_named_multimodal_suite(suite_name, argv=None):
    suite = get_suite_definition(suite_name)
    run_multimodal_suite(suite.window_size, suite.configs, argv=argv)
