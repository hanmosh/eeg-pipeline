import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from utils.log import logger
from dataset_retrievers.tfrecord_retriever import (
    _ALWAYS_SKIP_PERSON_IDS,
    _load_csv_labels_map,
    _question_selected,
    _resolve_effective_max_windows,
    _resolve_included_questions,
    _resolve_max_windows_per_question,
    _resolve_question_mode,
    _resolve_uncapped_questions,
)


_RAW_FILENAME_PATTERN = re.compile(
    r"^eeg_(?P<person>.+)_(?P<question>\d+)_(?P<timestamp>\d+)\.csv$"
)


def _parse_raw_filename(filename):
    match = _RAW_FILENAME_PATTERN.match(filename)
    if not match:
        return None, None, None
    person_id = match.group("person")
    try:
        question_num = int(match.group("question"))
        timestamp = int(match.group("timestamp"))
    except ValueError:
        return None, None, None
    return person_id, question_num, timestamp


def _segment_signal(samples, samples_per_window, window_stride):
    total_samples = int(samples.shape[0])
    if total_samples < samples_per_window:
        return np.empty((0, samples.shape[1], samples_per_window), dtype=np.float32)

    windows = []
    for start in range(0, total_samples - samples_per_window + 1, window_stride):
        end = start + samples_per_window
        windows.append(samples[start:end].T)

    if not windows:
        return np.empty((0, samples.shape[1], samples_per_window), dtype=np.float32)
    return np.asarray(windows, dtype=np.float32)


def _resolve_signal_filter(dataset_params):
    low_hz = dataset_params.get("bandpass_low_hz")
    high_hz = dataset_params.get("bandpass_high_hz")
    if low_hz is None and high_hz is None:
        return None

    sample_rate_hz = float(dataset_params.get("sampling_rate_hz", 256.0))
    if sample_rate_hz <= 0:
        raise ValueError("dataset_params.sampling_rate_hz must be greater than zero.")

    nyquist_hz = sample_rate_hz / 2.0
    low_hz = float(low_hz) if low_hz is not None else None
    high_hz = float(high_hz) if high_hz is not None else None

    if low_hz is not None and low_hz <= 0:
        raise ValueError("dataset_params.bandpass_low_hz must be greater than zero.")
    if high_hz is not None and high_hz <= 0:
        raise ValueError("dataset_params.bandpass_high_hz must be greater than zero.")
    if low_hz is not None and low_hz >= nyquist_hz:
        raise ValueError(
            "dataset_params.bandpass_low_hz must be below the Nyquist frequency."
        )
    if high_hz is not None and high_hz >= nyquist_hz:
        raise ValueError(
            "dataset_params.bandpass_high_hz must be below the Nyquist frequency."
        )
    if low_hz is not None and high_hz is not None and low_hz >= high_hz:
        raise ValueError(
            "dataset_params.bandpass_low_hz must be lower than bandpass_high_hz."
        )

    if low_hz is not None and high_hz is not None:
        normalized_cutoff = [low_hz / nyquist_hz, high_hz / nyquist_hz]
        filter_type = "bandpass"
    elif low_hz is not None:
        normalized_cutoff = low_hz / nyquist_hz
        filter_type = "highpass"
    else:
        normalized_cutoff = high_hz / nyquist_hz
        filter_type = "lowpass"

    return {
        "sample_rate_hz": sample_rate_hz,
        "low_hz": low_hz,
        "high_hz": high_hz,
        "filter_type": filter_type,
        "sos": butter(4, normalized_cutoff, btype=filter_type, output="sos"),
    }


def _apply_signal_filter(samples, signal_filter, csv_path):
    if signal_filter is None:
        return samples

    try:
        filtered = sosfiltfilt(signal_filter["sos"], samples, axis=0)
    except ValueError as exc:
        low_hz = signal_filter.get("low_hz")
        high_hz = signal_filter.get("high_hz")
        raise ValueError(
            f"Unable to apply raw EEG filter ({low_hz}, {high_hz}) Hz to {csv_path}: {exc}"
        ) from exc
    return np.asarray(filtered, dtype=np.float32)


def load_belonging_raw_csvs(dataset_params, metadata):
    raw_data_dir = dataset_params.get("raw_data_dir")
    channels = dataset_params.get("channels", ["TP9", "AF7", "AF8", "TP10"])
    question_mode = _resolve_question_mode(dataset_params)
    max_windows_per_question = _resolve_max_windows_per_question(dataset_params)
    included_questions = _resolve_included_questions(dataset_params)
    uncapped_questions = _resolve_uncapped_questions(dataset_params)
    csv_labels_map, label_col, label_source_text = _load_csv_labels_map(dataset_params)
    signal_filter = _resolve_signal_filter(dataset_params)

    if not raw_data_dir:
        raise ValueError("dataset_params must include 'raw_data_dir'")
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"Raw EEG dir not found: {raw_data_dir}")

    samples_per_window = int(dataset_params.get("samples_per_window", 256))
    window_stride = int(dataset_params.get("window_stride", samples_per_window))
    if samples_per_window <= 0:
        raise ValueError("dataset_params.samples_per_window must be greater than zero.")
    if window_stride <= 0:
        raise ValueError("dataset_params.window_stride must be greater than zero.")


    csv_paths = glob.glob(os.path.join(raw_data_dir, "**", "per_question", "eeg_*.csv"), recursive=True)
    if not csv_paths:
        raise RuntimeError(f"No per-question raw EEG CSV files found in {raw_data_dir}")

    parsed_paths = []
    for csv_path in csv_paths:
        filename = os.path.basename(csv_path)
        person_id, question_num, timestamp = _parse_raw_filename(filename)
        if person_id is None or question_num is None or timestamp is None:
            continue
        parsed_paths.append((person_id, question_num, timestamp, csv_path))
    parsed_paths.sort(key=lambda item: (item[0], item[1], item[2]))

    person_to_windows = {}
    person_to_label = {}
    skipped_question = []
    skipped_missing_label = []
    skipped_missing_people = set()
    skipped_forced_person_csvs = []
    skipped_forced_person_ids = set()
    skipped_short_csvs = []
    trimmed_question_csvs = 0
    trimmed_windows_removed = 0
    total_windows = 0
    for person_id, question_num, _timestamp, csv_path in parsed_paths:
        if person_id in _ALWAYS_SKIP_PERSON_IDS:
            skipped_forced_person_csvs.append(csv_path)
            skipped_forced_person_ids.add(person_id)
            continue
        if not _question_selected(question_num, question_mode, included_questions):
            skipped_question.append(csv_path)
            continue

        if csv_labels_map is not None:
            if person_id not in csv_labels_map:
                skipped_missing_label.append(csv_path)
                skipped_missing_people.add(person_id)
                continue
            label = csv_labels_map[person_id]
        else:
            raise ValueError(
                "Raw CSV loading requires dataset_params.labels_csv / labels_lookup_csv."
            )

        frame = pd.read_csv(csv_path)
        missing_channels = [channel for channel in channels if channel not in frame.columns]
        if missing_channels:
            raise ValueError(
                f"Missing channels {missing_channels} in raw EEG CSV {csv_path}."
            )

        samples = frame[channels].to_numpy(dtype=np.float32)
        samples = _apply_signal_filter(samples, signal_filter, csv_path)
        windows = _segment_signal(samples, samples_per_window, window_stride)
        if windows.shape[0] == 0:
            skipped_short_csvs.append(csv_path)
            continue

        effective_max_windows = _resolve_effective_max_windows(
            question_num,
            max_windows_per_question,
            uncapped_questions,
        )
        if effective_max_windows is not None and windows.shape[0] > effective_max_windows:
            trimmed_question_csvs += 1
            trimmed_windows_removed += int(windows.shape[0] - effective_max_windows)
            windows = windows[:effective_max_windows]

        total_windows += int(windows.shape[0])
        person_to_windows.setdefault(person_id, []).append(windows)
        if person_id in person_to_label and person_to_label[person_id] != label:
            raise ValueError(
                f"Label mismatch for person {person_id}: {person_to_label[person_id]} vs {label} in {csv_path}."
            )
        person_to_label[person_id] = label

    if skipped_question:
        logger.log("skipped_question_csvs", len(skipped_question))
    if skipped_forced_person_csvs:
        logger.log("skipped_forced_person_csvs", len(skipped_forced_person_csvs))
        logger.log("skipped_forced_person_ids", len(skipped_forced_person_ids))
    if skipped_missing_label:
        logger.log("skipped_missing_label_csvs", len(skipped_missing_label))
        logger.log("skipped_missing_label_people", len(skipped_missing_people))
    if skipped_short_csvs:
        logger.log("skipped_short_csvs", len(skipped_short_csvs))
    if trimmed_question_csvs:
        logger.log("trimmed_question_csvs", trimmed_question_csvs)
        logger.log("trimmed_windows_removed", trimmed_windows_removed)
    if not person_to_windows:
        raise RuntimeError("No raw EEG windows were loaded from the cleaned CSV files.")

    person_ids = sorted(person_to_windows.keys())
    windows_list = [
        np.concatenate([np.asarray(segment, dtype=np.float32) for segment in person_to_windows[pid]], axis=0)
        for pid in person_ids
    ]
    labels = [person_to_label[pid] for pid in person_ids]

    unique_labels, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique_labels, counts):
        logger.log(f"class_{lbl}_count", cnt)
    logger.log("total_windows", total_windows)
    logger.log("num_people", len(set(person_ids)))
    logger.log("question_mode", question_mode)
    logger.log("included_questions", sorted(included_questions))
    logger.log("uncapped_questions", sorted(uncapped_questions))
    logger.log("samples_per_window", samples_per_window)
    logger.log("window_stride", window_stride)
    logger.log("label_source", f"csv:{label_source_text}")
    if signal_filter is not None:
        logger.log("sampling_rate_hz", signal_filter["sample_rate_hz"])
        logger.log("bandpass_low_hz", signal_filter["low_hz"])
        logger.log("bandpass_high_hz", signal_filter["high_hz"])
        logger.log("signal_filter_type", signal_filter["filter_type"])

    metadata.update({
        "num_people": len(set(person_ids)),
        "num_windows": total_windows,
        "channels": channels,
        "num_channels": len(channels),
        "sample_length": samples_per_window,
        "num_classes": len(set(labels)),
        "question_mode": question_mode,
        "included_questions": sorted(included_questions),
        "uncapped_questions": sorted(uncapped_questions),
        "max_windows_per_question": max_windows_per_question,
        "label_col": label_col,
        "label_source": f"csv:{label_source_text}",
        "samples_per_window": samples_per_window,
        "window_stride": window_stride,
        "trimmed_question_csvs": trimmed_question_csvs,
        "trimmed_windows_removed": trimmed_windows_removed,
        "input_type": "raw",
        "sampling_rate_hz": signal_filter["sample_rate_hz"] if signal_filter is not None else None,
        "bandpass_low_hz": signal_filter["low_hz"] if signal_filter is not None else None,
        "bandpass_high_hz": signal_filter["high_hz"] if signal_filter is not None else None,
        "signal_filter_type": signal_filter["filter_type"] if signal_filter is not None else None,
    })

    X = {
        "windows": windows_list,
        "person_ids": person_ids,
    }
    y = np.array(labels)
    return X, y, metadata
