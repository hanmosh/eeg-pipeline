import glob
import os
import re

import numpy as np
import pandas as pd

from utils.log import logger
from dataset_retrievers.tfrecord_retriever import (
    _ALWAYS_SKIP_PERSON_IDS,
    _load_csv_labels_map,
    _question_in_range,
    _resolve_max_windows_per_question,
    _resolve_question_mode,
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


def load_belonging_raw_csvs(dataset_params, metadata):
    raw_data_dir = dataset_params.get("raw_data_dir")
    channels = dataset_params.get("channels", ["TP9", "AF7", "AF8", "TP10"])
    question_mode = _resolve_question_mode(dataset_params)
    max_windows_per_question = _resolve_max_windows_per_question(dataset_params)
    csv_labels_map, label_col, label_source_text = _load_csv_labels_map(dataset_params)

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

    max_people = dataset_params.get("max_people")
    if max_people is not None:
        max_people = int(max_people)
        if max_people <= 0:
            raise ValueError("dataset_params.max_people must be greater than zero.")

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
        if not _question_in_range(question_num, question_mode):
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
        if max_people is not None and person_id not in person_to_label and len(person_to_label) >= max_people:
            continue

        frame = pd.read_csv(csv_path)
        missing_channels = [channel for channel in channels if channel not in frame.columns]
        if missing_channels:
            raise ValueError(
                f"Missing channels {missing_channels} in raw EEG CSV {csv_path}."
            )

        samples = frame[channels].to_numpy(dtype=np.float32)
        windows = _segment_signal(samples, samples_per_window, window_stride)
        if windows.shape[0] == 0:
            skipped_short_csvs.append(csv_path)
            continue

        if max_windows_per_question is not None and windows.shape[0] > max_windows_per_question:
            trimmed_question_csvs += 1
            trimmed_windows_removed += int(windows.shape[0] - max_windows_per_question)
            windows = windows[:max_windows_per_question]

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
    windows_list = [np.concatenate(person_to_windows[pid], axis=0) for pid in person_ids]
    labels = [person_to_label[pid] for pid in person_ids]

    unique_labels, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique_labels, counts):
        logger.log(f"class_{lbl}_count", cnt)
    logger.log("total_windows", total_windows)
    logger.log("num_people", len(set(person_ids)))
    logger.log("question_mode", question_mode)
    logger.log("samples_per_window", samples_per_window)
    logger.log("window_stride", window_stride)
    logger.log("label_source", f"csv:{label_source_text}")

    metadata.update({
        "num_people": len(set(person_ids)),
        "num_windows": total_windows,
        "channels": channels,
        "num_channels": len(channels),
        "sample_length": samples_per_window,
        "num_classes": len(set(labels)),
        "question_mode": question_mode,
        "max_windows_per_question": max_windows_per_question,
        "label_col": label_col,
        "label_source": f"csv:{label_source_text}",
        "samples_per_window": samples_per_window,
        "window_stride": window_stride,
        "trimmed_question_csvs": trimmed_question_csvs,
        "trimmed_windows_removed": trimmed_windows_removed,
        "input_type": "raw",
    })

    X = {
        "windows": windows_list,
        "person_ids": person_ids,
    }
    y = np.array(labels)
    return X, y, metadata
