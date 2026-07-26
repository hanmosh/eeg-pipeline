from collections import Counter
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from utils.log import logger


DEFAULT_FOLD_MANIFEST_COLUMNS = {
    "SpecificLabel": "specific_fold",
    "CompositeLabel": "composite_fold",
    "FactorLabel": "factor_fold",
    "WeightedDiffLabel": "weighted_fold",
}

DEFAULT_SPLIT_SCOPE = "inter_subject"


class TrainScalogramSpecAugment:
    """Mild SpecAugment-style masking for [T, C, H, W] scalogram sequences.

    We treat H as the frequency axis and W as the time axis. The same sampled
    mask is applied across all windows/channels in the sequence item to keep the
    augmentation conservative for this small dataset.
    """

    def __init__(
        self,
        *,
        p=0.1,
        freq_mask_param=5,
        time_mask_param=10,
        num_freq_masks=1,
        num_time_masks=1,
        mask_value=0.0,
    ):
        self.p = float(p)
        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)
        self.num_freq_masks = int(num_freq_masks)
        self.num_time_masks = int(num_time_masks)
        self.mask_value = float(mask_value)

        if not 0.0 <= self.p <= 1.0:
            raise ValueError("preprocessor_params.specaugment.p must be between 0 and 1.")
        if self.freq_mask_param < 0:
            raise ValueError("preprocessor_params.specaugment.freq_mask_param must be >= 0.")
        if self.time_mask_param < 0:
            raise ValueError("preprocessor_params.specaugment.time_mask_param must be >= 0.")
        if self.num_freq_masks < 0:
            raise ValueError("preprocessor_params.specaugment.num_freq_masks must be >= 0.")
        if self.num_time_masks < 0:
            raise ValueError("preprocessor_params.specaugment.num_time_masks must be >= 0.")

    def _sample_mask(self, axis_size, mask_param):
        if axis_size <= 0 or mask_param <= 0:
            return None
        max_width = min(int(axis_size), int(mask_param))
        if max_width <= 0:
            return None

        mask_width = int(torch.randint(0, max_width + 1, (1,)).item())
        if mask_width <= 0:
            return None
        start_max = axis_size - mask_width
        start = 0 if start_max <= 0 else int(torch.randint(0, start_max + 1, (1,)).item())
        return start, start + mask_width

    def __call__(self, windows):
        if windows.ndim != 4:
            return windows
        if self.p <= 0.0 or torch.rand(1).item() > self.p:
            return windows

        augmented = windows.clone()
        _, _, height, width = augmented.shape

        for _ in range(self.num_freq_masks):
            mask_range = self._sample_mask(height, self.freq_mask_param)
            if mask_range is None:
                continue
            start, end = mask_range
            augmented[:, :, start:end, :] = self.mask_value

        for _ in range(self.num_time_masks):
            mask_range = self._sample_mask(width, self.time_mask_param)
            if mask_range is None:
                continue
            start, end = mask_range
            augmented[:, :, :, start:end] = self.mask_value

        return augmented


class TrainScalogramGaussianNoise:
    """Small additive Gaussian noise for normalized [T, C, H, W] scalograms."""

    def __init__(
        self,
        *,
        p=0.1,
        std=0.01,
        clamp_min=0.0,
        clamp_max=1.0,
    ):
        self.p = float(p)
        self.std = float(std)
        self.clamp_min = None if clamp_min is None else float(clamp_min)
        self.clamp_max = None if clamp_max is None else float(clamp_max)

        if not 0.0 <= self.p <= 1.0:
            raise ValueError("preprocessor_params.gaussian_noise.p must be between 0 and 1.")
        if self.std < 0.0:
            raise ValueError("preprocessor_params.gaussian_noise.std must be >= 0.")
        if (
            self.clamp_min is not None
            and self.clamp_max is not None
            and self.clamp_min > self.clamp_max
        ):
            raise ValueError("preprocessor_params.gaussian_noise.clamp_min must be <= clamp_max.")

    def __call__(self, windows):
        if windows.ndim != 4:
            return windows
        if self.p <= 0.0 or self.std <= 0.0 or torch.rand(1).item() > self.p:
            return windows

        augmented = windows + (torch.randn_like(windows) * self.std)
        if self.clamp_min is None and self.clamp_max is None:
            return augmented
        min_value = self.clamp_min if self.clamp_min is not None else float("-inf")
        max_value = self.clamp_max if self.clamp_max is not None else float("inf")
        return torch.clamp(augmented, min=min_value, max=max_value)


class TrainScalogramTransformChain:
    """Applies multiple train-only scalogram transforms in sequence."""

    def __init__(self, transforms):
        self.transforms = [transform for transform in transforms if transform is not None]

    def __call__(self, windows):
        augmented = windows
        for transform in self.transforms:
            augmented = transform(augmented)
        return augmented


class TFRecordSequenceDataset(Dataset):
    """Sequences of window tensors for each participant."""

    def __init__(
        self,
        person_ids,
        person_to_windows,
        person_to_label,
        sequence_length=None,
        sequence_stride=None,
        max_windows_per_person=None,
        downsample=False,
        window_transform=None,
    ):
        self.person_ids = [str(pid) for pid in person_ids]
        self.person_to_windows = {}
        self.person_to_label = person_to_label
        self.sequence_length = sequence_length or 0
        self.sequence_stride = sequence_stride or 0
        self.max_windows_per_person = max_windows_per_person
        self.downsample = downsample
        self.window_transform = window_transform

        self.sequences = []
        self.sequence_labels = []
        self.labels = []
        self.num_windows = 0

        for pid in self.person_ids:
            windows = person_to_windows.get(pid)
            if windows is None:
                continue
            windows = self._maybe_downsample(windows)
            if windows is None or len(windows) == 0:
                continue

            self.person_to_windows[pid] = windows
            if self.sequence_length > 0:
                stride = self.sequence_stride if self.sequence_stride > 0 else self.sequence_length
                for start in range(0, len(windows), stride):
                    end = min(start + self.sequence_length, len(windows))
                    if end <= start:
                        continue
                    self._add_sequence(pid, start, end)
            else:
                self._add_sequence(pid, 0, len(windows))

    def __len__(self):
        return len(self.sequences)

    def _maybe_downsample(self, windows):
        if self.max_windows_per_person is None or self.max_windows_per_person <= 0:
            return windows
        if len(windows) <= self.max_windows_per_person:
            return windows
        if self.downsample:
            start_max = len(windows) - self.max_windows_per_person
            start = np.random.randint(0, start_max + 1)
        else:
            start = 0
        return windows[start:start + self.max_windows_per_person]

    def _add_sequence(self, pid, start, end):
        label = int(self.person_to_label[pid])
        self.sequences.append((pid, start, end))
        self.sequence_labels.append(label)
        length = end - start
        self.labels.append(label)
        self.num_windows += length

    def __getitem__(self, idx):
        pid, start, end = self.sequences[idx]
        windows = self.person_to_windows[pid][start:end]
        windows = torch.as_tensor(windows, dtype=torch.float32)
        if self.window_transform is not None:
            windows = self.window_transform(windows)
        label = self.sequence_labels[idx]
        return windows, label, pid


def sequence_collate_fn(batch):
    windows, labels, pids = zip(*batch)
    lengths = torch.tensor([w.size(0) for w in windows], dtype=torch.long)
    max_len = int(lengths.max()) if len(lengths) > 0 else 0
    batch_size = len(windows)

    if batch_size == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long), lengths, []

    sample_dim = windows[0].dim()
    if sample_dim == 4:
        channels, height, width = windows[0].shape[1:]
        padded_windows = torch.zeros((batch_size, max_len, channels, height, width), dtype=windows[0].dtype)
        for i, win in enumerate(windows):
            length = win.size(0)
            padded_windows[i, :length] = win
    elif sample_dim == 3:
        channels, sample_length = windows[0].shape[1:]
        padded_windows = torch.zeros((batch_size, max_len, channels, sample_length), dtype=windows[0].dtype)
        for i, win in enumerate(windows):
            length = win.size(0)
            padded_windows[i, :length] = win
    else:
        raise ValueError(
            f"Unsupported sample shape {tuple(windows[0].shape)}. "
            "Expected [T, C, H, W] or [T, C, L]."
        )

    labels = torch.tensor(labels, dtype=torch.long)
    return padded_windows, labels, lengths, list(pids)


def _validate_stratified_splits(n_total, test_split, val_split, num_classes, class_counts):
    if num_classes <= 1:
        return

    def _split_size(n, split):
        return int(np.ceil(n * split)) if split > 0 else 0

    n_test = _split_size(n_total, test_split)
    n_train_val = n_total - n_test
    n_val = _split_size(n_train_val, val_split) if n_train_val > 0 else 0
    n_train = n_total - n_test - n_val

    if n_test > 0 and n_test < num_classes:
        raise ValueError(
            f"test_split too small: n_test={n_test}, num_classes={num_classes}."
        )
    if n_val > 0 and n_val < num_classes:
        raise ValueError(
            f"val_split too small: n_val={n_val}, num_classes={num_classes}."
        )
    if n_train < num_classes:
        raise ValueError(
            f"Not enough training samples: n_train={n_train}, num_classes={num_classes}."
        )

    min_count = min(class_counts.values()) if class_counts else 0
    if test_split > 0 and val_split > 0 and min_count < 3:
        raise ValueError(
            f"Each class needs at least 3 people for train/val/test. Smallest class has {min_count}."
        )
    if (test_split > 0) != (val_split > 0) and min_count < 2:
        raise ValueError(
            f"Each class needs at least 2 people for train+test or train+val. Smallest class has {min_count}."
        )


def _resolve_active_label_col(metadata):
    for key in ("label_col", "survey_label_col"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _resolve_fold_manifest_column(preprocessor_params, metadata):
    explicit = preprocessor_params.get("fold_manifest_column")
    if explicit:
        return str(explicit)

    label_col = _resolve_active_label_col(metadata)
    column_map = preprocessor_params.get("fold_manifest_columns")
    if isinstance(column_map, dict) and label_col in column_map:
        return str(column_map[label_col])

    if label_col in DEFAULT_FOLD_MANIFEST_COLUMNS:
        return DEFAULT_FOLD_MANIFEST_COLUMNS[label_col]

    if label_col is None:
        raise ValueError(
            "fold_manifest_csv requires metadata to include an active label column. "
            "Expected metadata['label_col'] or metadata['survey_label_col']."
        )

    raise ValueError(
        f"No fold-manifest column mapping found for label column '{label_col}'. "
        "Set preprocessor_params.fold_manifest_column or fold_manifest_columns."
    )


def _load_fold_manifest_spec(preprocessor_params, metadata, unique_person_ids, cv_folds, cv_repeats, test_split):
    manifest_path = preprocessor_params.get("fold_manifest_csv")
    if not manifest_path:
        return None

    if cv_folds <= 1:
        raise ValueError("fold_manifest_csv requires preprocessor_params.cv_folds > 1.")
    if cv_repeats != 1:
        raise ValueError("fold_manifest_csv currently supports only cv_repeats = 1.")
    if float(test_split) != 0.0:
        raise ValueError("fold_manifest_csv currently requires test_split = 0.0.")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Fold manifest not found: {manifest_path}")

    id_col = str(preprocessor_params.get("fold_manifest_id_col", "FileName"))
    fold_col = _resolve_fold_manifest_column(preprocessor_params, metadata)

    manifest_df = pd.read_csv(manifest_path)
    if id_col not in manifest_df.columns:
        raise ValueError(f"Fold manifest must include ID column '{id_col}'.")
    if fold_col not in manifest_df.columns:
        raise ValueError(f"Fold manifest must include fold column '{fold_col}'.")

    manifest_df[id_col] = manifest_df[id_col].astype(str).str.strip()
    manifest_df = manifest_df[manifest_df[id_col] != ""].copy()

    duplicated_ids = manifest_df[manifest_df[id_col].duplicated()][id_col].unique().tolist()
    if duplicated_ids:
        raise ValueError(
            f"Fold manifest contains duplicate participant IDs, e.g. '{duplicated_ids[0]}'."
        )

    expected_people = [str(pid) for pid in unique_person_ids]
    expected_people_set = set(expected_people)
    manifest_df = manifest_df[manifest_df[id_col].isin(expected_people_set)].copy()

    present_people = set(manifest_df[id_col])
    missing_people = [pid for pid in expected_people if pid not in present_people]
    if missing_people:
        preview = ", ".join(missing_people[:5])
        raise ValueError(
            "Fold manifest is missing participant assignments for "
            f"{len(missing_people)} person(s), e.g. {preview}."
        )

    manifest_df[fold_col] = pd.to_numeric(manifest_df[fold_col], errors="coerce")
    if manifest_df[fold_col].isna().any():
        bad_rows = manifest_df.loc[manifest_df[fold_col].isna(), id_col].tolist()
        raise ValueError(
            f"Fold manifest column '{fold_col}' contains non-numeric values, e.g. '{bad_rows[0]}'."
        )
    manifest_df[fold_col] = manifest_df[fold_col].astype(int)

    invalid_folds = sorted(
        fold_num
        for fold_num in manifest_df[fold_col].unique().tolist()
        if fold_num < 1 or fold_num > cv_folds
    )
    if invalid_folds:
        raise ValueError(
            f"Fold manifest column '{fold_col}' contains fold IDs outside 1..{cv_folds}: {invalid_folds}"
        )

    assigned_folds = set(manifest_df[fold_col].tolist())
    missing_fold_ids = [fold_num for fold_num in range(1, cv_folds + 1) if fold_num not in assigned_folds]
    if missing_fold_ids:
        raise ValueError(
            f"Fold manifest column '{fold_col}' is missing assignments for fold(s): {missing_fold_ids}"
        )

    ordered_people = manifest_df[id_col].to_numpy(dtype=str)
    fold_to_people = {}
    for fold_num in range(1, cv_folds + 1):
        fold_people = manifest_df.loc[manifest_df[fold_col] == fold_num, id_col].to_numpy(dtype=str)
        if len(fold_people) == 0:
            raise ValueError(f"Fold manifest column '{fold_col}' has no participants assigned to fold {fold_num}.")
        fold_to_people[fold_num] = fold_people

    return {
        "path": manifest_path,
        "id_col": id_col,
        "fold_col": fold_col,
        "ordered_people": ordered_people,
        "fold_to_people": fold_to_people,
    }


def _build_generator(seed_value):
    if seed_value is None:
        return None
    gen = torch.Generator()
    gen.manual_seed(int(seed_value))
    return gen


def _normalize_split_scope(value):
    if value is None:
        return DEFAULT_SPLIT_SCOPE

    normalized = str(value).strip().lower()
    normalized = normalized.replace("-", "").replace("_", "").replace(" ", "")
    if normalized in {"intersubject", "participantwise", "personwise", "subjectwise"}:
        return "inter_subject"

    raise ValueError(
        "Only inter-subject splitting is supported. Remove preprocessor_params.split_scope "
        "or set it to 'inter_subject'."
    )


def _prepare_sequence_inputs(X, y):
    scalograms_list = X.get("scalograms")
    if scalograms_list is None:
        scalograms_list = X.get("windows")
    person_ids = X.get("person_ids")
    if scalograms_list is None or person_ids is None:
        raise ValueError("X must contain 'scalograms' or 'windows', and 'person_ids'")
    if len(scalograms_list) != len(person_ids) or len(y) != len(person_ids):
        raise ValueError("Mismatch between scalograms, person_ids, and labels lengths")

    person_to_windows = {}
    person_to_label = {}
    for pid, scalograms, label in zip(person_ids, scalograms_list, y):
        pid_str = str(pid)
        person_to_windows[pid_str] = np.asarray(scalograms, dtype=np.float32)
        if pid_str not in person_to_label:
            person_to_label[pid_str] = int(label)

    unique_person_ids = np.array(list(person_to_windows.keys()))
    unique_labels = np.array([person_to_label[str(pid)] for pid in unique_person_ids])
    class_counts = Counter(unique_labels)
    return person_to_windows, person_to_label, unique_person_ids, unique_labels, class_counts


def _make_sequence_dataset(
    person_ids,
    person_to_windows,
    person_to_label,
    dataset_kwargs,
    *,
    downsample=False,
    window_transform=None,
):
    return TFRecordSequenceDataset(
        person_ids,
        person_to_windows,
        person_to_label,
        downsample=downsample,
        window_transform=window_transform,
        **dataset_kwargs,
    )


def _make_sequence_loader(dataset, batch_size, shuffle, *, generator=None):
    if dataset is None:
        return None

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": 0,
        "collate_fn": sequence_collate_fn,
    }
    if generator is not None:
        loader_kwargs["generator"] = generator
    return DataLoader(dataset, **loader_kwargs)


def _build_cv_split_plan(manifest_spec, cv_people, cv_labels, cv_folds, cv_repeats, effective_seed):
    if manifest_spec is not None:
        split_plan = []
        ordered_people = manifest_spec["ordered_people"]
        for fold_num in range(1, cv_folds + 1):
            val_persons = np.asarray(manifest_spec["fold_to_people"][fold_num], dtype=str)
            val_people_set = set(val_persons.tolist())
            train_persons = np.asarray(
                [pid for pid in ordered_people if pid not in val_people_set],
                dtype=str,
            )
            split_plan.append((fold_num, 1, fold_num, train_persons, val_persons))
        return cv_folds, split_plan

    if cv_repeats > 1:
        splitter = RepeatedStratifiedKFold(
            n_splits=cv_folds,
            n_repeats=cv_repeats,
            random_state=effective_seed,
        )
        total_cv_folds = cv_folds * cv_repeats
        split_plan = []
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(cv_people, cv_labels), start=1):
            repeat_idx = ((fold_idx - 1) // cv_folds) + 1
            fold_in_repeat = ((fold_idx - 1) % cv_folds) + 1
            train_persons = cv_people[train_idx]
            val_persons = cv_people[val_idx]
            split_plan.append((fold_idx, repeat_idx, fold_in_repeat, train_persons, val_persons))
        return total_cv_folds, split_plan

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=effective_seed)
    split_plan = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(cv_people, cv_labels), start=1):
        train_persons = cv_people[train_idx]
        val_persons = cv_people[val_idx]
        split_plan.append((fold_idx, 1, fold_idx, train_persons, val_persons))
    return cv_folds, split_plan


def _log_class_counts(split_name, labels):
    if len(labels) == 0:
        return

    counter = Counter(labels)
    for label, count in counter.items():
        logger.log(f"{split_name}_class_{label}_count", count)


def _resolve_train_scalogram_specaugment(preprocessor_params):
    raw_config = preprocessor_params.get("specaugment")
    if raw_config is None:
        return None, {
            "enabled": False,
            "p": 0.0,
            "freq_mask_param": 0,
            "time_mask_param": 0,
            "num_freq_masks": 0,
            "num_time_masks": 0,
            "mask_value": 0.0,
        }
    if not isinstance(raw_config, dict):
        raise ValueError("preprocessor_params.specaugment must be a dictionary when provided.")

    config = {
        "enabled": bool(raw_config.get("enabled", False)),
        "p": float(raw_config.get("p", 0.1)),
        "freq_mask_param": int(raw_config.get("freq_mask_param", 5)),
        "time_mask_param": int(raw_config.get("time_mask_param", 10)),
        "num_freq_masks": int(raw_config.get("num_freq_masks", 1)),
        "num_time_masks": int(raw_config.get("num_time_masks", 1)),
        "mask_value": float(raw_config.get("mask_value", 0.0)),
    }
    if not config["enabled"]:
        return None, config

    augmenter = TrainScalogramSpecAugment(
        p=config["p"],
        freq_mask_param=config["freq_mask_param"],
        time_mask_param=config["time_mask_param"],
        num_freq_masks=config["num_freq_masks"],
        num_time_masks=config["num_time_masks"],
        mask_value=config["mask_value"],
    )
    return augmenter, config


def _resolve_train_scalogram_gaussian_noise(preprocessor_params):
    raw_config = preprocessor_params.get("gaussian_noise")
    if raw_config is None:
        return None, {
            "enabled": False,
            "p": 0.0,
            "std": 0.0,
            "clamp_min": 0.0,
            "clamp_max": 1.0,
        }
    if not isinstance(raw_config, dict):
        raise ValueError("preprocessor_params.gaussian_noise must be a dictionary when provided.")

    config = {
        "enabled": bool(raw_config.get("enabled", False)),
        "p": float(raw_config.get("p", 0.1)),
        "std": float(raw_config.get("std", 0.01)),
        "clamp_min": raw_config.get("clamp_min", 0.0),
        "clamp_max": raw_config.get("clamp_max", 1.0),
    }
    if config["clamp_min"] is not None:
        config["clamp_min"] = float(config["clamp_min"])
    if config["clamp_max"] is not None:
        config["clamp_max"] = float(config["clamp_max"])
    if not config["enabled"]:
        return None, config

    augmenter = TrainScalogramGaussianNoise(
        p=config["p"],
        std=config["std"],
        clamp_min=config["clamp_min"],
        clamp_max=config["clamp_max"],
    )
    return augmenter, config


def _compose_train_scalogram_transforms(*transforms):
    active_transforms = [transform for transform in transforms if transform is not None]
    if not active_transforms:
        return None
    if len(active_transforms) == 1:
        return active_transforms[0]
    return TrainScalogramTransformChain(active_transforms)


def _make_optional_sequence_dataset(
    person_ids,
    person_to_windows,
    person_to_label,
    dataset_kwargs,
    *,
    downsample=False,
    window_transform=None,
):
    if person_ids is None or len(person_ids) == 0:
        return None
    return _make_sequence_dataset(
        person_ids,
        person_to_windows,
        person_to_label,
        dataset_kwargs,
        downsample=downsample,
        window_transform=window_transform,
    )


def _labels_for_people(person_ids, person_to_label):
    return [person_to_label[str(pid)] for pid in person_ids]


def _summarize_binary_labels(labels):
    pos_count = int(sum(labels))
    neg_count = len(labels) - pos_count
    return pos_count, neg_count


def _resolve_train_scalogram_augmentation(preprocessor_params):
    train_specaugment, train_specaugment_config = _resolve_train_scalogram_specaugment(preprocessor_params)
    train_gaussian_noise, train_gaussian_noise_config = _resolve_train_scalogram_gaussian_noise(preprocessor_params)

    train_window_transform = _compose_train_scalogram_transforms(
        train_specaugment,
        train_gaussian_noise,
    )
    augmentation_metadata = {
        "train_specaugment_enabled": bool(train_specaugment_config.get("enabled")),
        "train_specaugment_p": float(train_specaugment_config.get("p", 0.0)),
        "train_specaugment_freq_mask_param": int(train_specaugment_config.get("freq_mask_param", 0)),
        "train_specaugment_time_mask_param": int(train_specaugment_config.get("time_mask_param", 0)),
        "train_specaugment_num_freq_masks": int(train_specaugment_config.get("num_freq_masks", 0)),
        "train_specaugment_num_time_masks": int(train_specaugment_config.get("num_time_masks", 0)),
        "train_specaugment_mask_value": float(train_specaugment_config.get("mask_value", 0.0)),
        "train_gaussian_noise_enabled": bool(train_gaussian_noise_config.get("enabled")),
        "train_gaussian_noise_p": float(train_gaussian_noise_config.get("p", 0.0)),
        "train_gaussian_noise_std": float(train_gaussian_noise_config.get("std", 0.0)),
        "train_gaussian_noise_clamp_min": train_gaussian_noise_config.get("clamp_min"),
        "train_gaussian_noise_clamp_max": train_gaussian_noise_config.get("clamp_max"),
    }
    return train_window_transform, augmentation_metadata


def _log_metadata_values(values):
    for key, value in values.items():
        logger.log(key, value)


def _build_shared_preprocessor_metadata(
    *,
    split_scope,
    batch_size,
    downsample_train,
    max_windows_per_person,
    sequence_length,
    sequence_stride,
    effective_seed,
    augmentation_metadata,
):
    return {
        "split_scope": split_scope,
        "batch_size": batch_size,
        "downsample_train": downsample_train,
        "max_windows_per_person": max_windows_per_person,
        "sequence_length": sequence_length,
        "sequence_stride": sequence_stride,
        "seed": effective_seed,
        **augmentation_metadata,
    }


def _print_cv_fold_summary(
    *,
    fold_idx,
    cv_folds,
    repeat_idx,
    cv_repeats,
    fold_in_repeat,
    total_cv_folds,
    train_people_count,
    train_windows,
    val_people_count,
    val_windows,
    train_pos,
    train_neg,
    val_pos,
    val_neg,
):
    if cv_repeats > 1:
        print(
            f"Repeat {repeat_idx}/{cv_repeats}, Fold {fold_in_repeat}/{cv_folds} "
            f"(global {fold_idx}/{total_cv_folds}): "
            f"Train {train_people_count} people ({train_windows} windows), "
            f"Val {val_people_count} people ({val_windows} windows)"
        )
    else:
        print(
            f"Fold {fold_idx}/{cv_folds}: "
            f"Train {train_people_count} people ({train_windows} windows), "
            f"Val {val_people_count} people ({val_windows} windows)"
        )
    print(
        f"Fold {fold_idx}: Train {train_pos} pos / {train_neg} neg | "
        f"Val {val_pos} pos / {val_neg} neg"
    )


def tfrecord_preprocessor(preprocessor_params, X, y, metadata):
    """TFRecord sequence preprocessing (participant-wise splits)."""
    logger.log_dict(preprocessor_params)

    test_split = preprocessor_params.get("test_split", 0.2)
    val_split = preprocessor_params.get("val_split", 0.2)
    batch_size = preprocessor_params.get("batch_size", 16)
    cv_folds = preprocessor_params.get("cv_folds", 0)
    cv_repeats = int(preprocessor_params.get("cv_repeats", 1))
    downsample_train = preprocessor_params.get("downsample_train", False)
    max_windows_per_person = preprocessor_params.get("max_windows_per_person", None)
    sequence_length = preprocessor_params.get("sequence_length", None)
    sequence_stride = preprocessor_params.get("sequence_stride", None)
    cv_fold_as_test = bool(preprocessor_params.get("cv_fold_as_test", False))
    seed = preprocessor_params.get("seed", None)
    effective_seed = int(seed) if seed is not None else None
    split_scope = _normalize_split_scope(preprocessor_params.get("split_scope"))

    train_window_transform, augmentation_metadata = _resolve_train_scalogram_augmentation(preprocessor_params)
    dataset_kwargs = {
        "sequence_length": sequence_length,
        "sequence_stride": sequence_stride,
        "max_windows_per_person": max_windows_per_person,
    }
    _log_metadata_values(augmentation_metadata)

    person_to_windows, person_to_label, unique_person_ids, unique_labels, class_counts = _prepare_sequence_inputs(
        X, y
    )
    num_classes = len(class_counts)
    manifest_spec = _load_fold_manifest_spec(
        preprocessor_params,
        metadata,
        unique_person_ids,
        cv_folds,
        cv_repeats,
        test_split,
    )
    if manifest_spec is not None:
        logger.log("fold_manifest_csv", manifest_spec["path"])
        logger.log("fold_manifest_column", manifest_spec["fold_col"])
    logger.log("split_scope", split_scope)

    shared_metadata = _build_shared_preprocessor_metadata(
        split_scope=split_scope,
        batch_size=batch_size,
        downsample_train=downsample_train,
        max_windows_per_person=max_windows_per_person,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        effective_seed=effective_seed,
        augmentation_metadata=augmentation_metadata,
    )

    if cv_folds and cv_folds > 1:
        if cv_repeats < 1:
            raise ValueError("cv_repeats must be >= 1 when cv_folds > 1.")
        if test_split < 0 or test_split >= 1:
            raise ValueError("test_split must be in [0, 1) when cv_folds > 1.")

        cv_people = unique_person_ids
        cv_labels = unique_labels
        holdout_test_people = np.array([], dtype=str)
        holdout_test_labels = np.array([], dtype=int)

        if manifest_spec is not None:
            cv_people = np.asarray(manifest_spec["ordered_people"], dtype=str)
            cv_labels = np.array(_labels_for_people(cv_people, person_to_label))
        elif test_split > 0:
            cv_people, holdout_test_people = train_test_split(
                unique_person_ids,
                test_size=test_split,
                random_state=effective_seed,
                stratify=unique_labels,
            )
            cv_labels = np.array(_labels_for_people(cv_people, person_to_label))
            holdout_test_labels = np.array(_labels_for_people(holdout_test_people, person_to_label))

        if cv_folds > len(cv_people):
            raise ValueError(
                "cv_folds cannot exceed number of people available for CV "
                "after holdout test split."
            )

        cv_class_counts = Counter(cv_labels)
        min_cv_class = min(cv_class_counts.values()) if cv_class_counts else 0
        if min_cv_class < cv_folds:
            raise ValueError(
                f"cv_folds={cv_folds} is too large for stratified CV after holdout split. "
                f"Smallest CV class has {min_cv_class} people."
            )

        total_cv_folds, split_plan = _build_cv_split_plan(
            manifest_spec,
            cv_people,
            cv_labels,
            cv_folds,
            cv_repeats,
            effective_seed,
        )

        fold_data = []
        test_dataset = _make_optional_sequence_dataset(
            holdout_test_people,
            person_to_windows,
            person_to_label,
            dataset_kwargs,
            downsample=False,
        )
        test_loader = _make_sequence_loader(test_dataset, batch_size, shuffle=False)

        if test_dataset is not None:
            print(
                f"Holdout Test: {len(holdout_test_people)} people "
                f"({test_dataset.num_windows} windows)"
            )

        for fold_idx, repeat_idx, fold_in_repeat, train_persons, val_persons in split_plan:
            train_labels = _labels_for_people(train_persons, person_to_label)
            val_labels = _labels_for_people(val_persons, person_to_label)
            train_pos, train_neg = _summarize_binary_labels(train_labels)
            val_pos, val_neg = _summarize_binary_labels(val_labels)

            train_dataset = _make_sequence_dataset(
                train_persons,
                person_to_windows,
                person_to_label,
                dataset_kwargs,
                downsample=downsample_train,
                window_transform=train_window_transform,
            )
            val_dataset = _make_sequence_dataset(
                val_persons,
                person_to_windows,
                person_to_label,
                dataset_kwargs,
                downsample=False,
            )

            _print_cv_fold_summary(
                fold_idx=fold_idx,
                cv_folds=cv_folds,
                repeat_idx=repeat_idx,
                cv_repeats=cv_repeats,
                fold_in_repeat=fold_in_repeat,
                total_cv_folds=total_cv_folds,
                train_people_count=len(train_persons),
                train_windows=train_dataset.num_windows,
                val_people_count=len(val_persons),
                val_windows=val_dataset.num_windows,
                train_pos=train_pos,
                train_neg=train_neg,
                val_pos=val_pos,
                val_neg=val_neg,
            )

            train_loader = _make_sequence_loader(
                train_dataset,
                shuffle=True,
                batch_size=batch_size,
                generator=_build_generator(effective_seed + fold_idx) if effective_seed is not None else None,
            )
            val_loader = _make_sequence_loader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
            )

            fold_data.append(
                {
                    "train_loader": train_loader,
                    "val_loader": None if cv_fold_as_test else val_loader,
                    "test_loader": val_loader if cv_fold_as_test else test_loader,
                    "cv_fold": fold_idx,
                    "cv_fold_in_repeat": fold_in_repeat,
                    "cv_repeat_index": repeat_idx,
                    "cv_total_folds": total_cv_folds,
                }
            )

        metadata.update(
            {
                **shared_metadata,
                "cv_folds": cv_folds,
                "cv_repeats": cv_repeats,
                "cv_total_folds": total_cv_folds,
                "cv_fold_as_test": cv_fold_as_test,
                "test_split": test_split,
                "num_test_people": len(holdout_test_people),
                "num_test_windows": test_dataset.num_windows if test_dataset is not None else 0,
                "has_holdout_test": test_dataset is not None,
                "fold_manifest_used": manifest_spec is not None,
                "fold_manifest_csv": manifest_spec["path"] if manifest_spec is not None else None,
                "fold_manifest_column": manifest_spec["fold_col"] if manifest_spec is not None else None,
            }
        )

        if test_dataset is not None and len(holdout_test_labels) > 0:
            _log_class_counts("test", holdout_test_labels.tolist())

        return fold_data, metadata

    _validate_stratified_splits(
        len(unique_person_ids), test_split, val_split, num_classes, class_counts
    )

    if test_split > 0:
        train_val_persons, test_persons = train_test_split(
            unique_person_ids,
            test_size=test_split,
            random_state=effective_seed,
            stratify=unique_labels,
        )
    else:
        train_val_persons = unique_person_ids
        test_persons = np.array([], dtype=str)

    if val_split > 0:
        train_val_labels = np.array(_labels_for_people(train_val_persons, person_to_label))
        train_persons, val_persons = train_test_split(
            train_val_persons,
            test_size=val_split,
            random_state=effective_seed,
            stratify=train_val_labels,
        )
    else:
        train_persons = train_val_persons
        val_persons = np.array([], dtype=str)

    train_dataset = _make_sequence_dataset(
        train_persons,
        person_to_windows,
        person_to_label,
        dataset_kwargs,
        downsample=downsample_train,
        window_transform=train_window_transform,
    )
    val_dataset = _make_optional_sequence_dataset(
        val_persons,
        person_to_windows,
        person_to_label,
        dataset_kwargs,
        downsample=False,
    )
    test_dataset = _make_optional_sequence_dataset(
        test_persons,
        person_to_windows,
        person_to_label,
        dataset_kwargs,
        downsample=False,
    )

    train_windows = train_dataset.num_windows
    val_windows = val_dataset.num_windows if val_dataset is not None else 0
    test_windows = test_dataset.num_windows if test_dataset is not None else 0

    print(
        f"Train: {len(train_persons)} people ({train_windows} windows), "
        f"Val: {len(val_persons)} people ({val_windows} windows), "
        f"Test: {len(test_persons)} people ({test_windows} windows)"
    )

    train_loader = _make_sequence_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=_build_generator(effective_seed),
    )
    val_loader = _make_sequence_loader(val_dataset, batch_size, shuffle=False)
    test_loader = _make_sequence_loader(test_dataset, batch_size, shuffle=False)

    metadata.update(
        {
            **shared_metadata,
            "num_train_samples": train_windows,
            "num_val_samples": val_windows,
            "num_test_samples": test_windows,
            "num_train_windows": train_windows,
            "num_val_windows": val_windows,
            "num_test_windows": test_windows,
            "num_train_people": len(train_persons),
            "num_val_people": len(val_persons),
            "num_test_people": len(test_persons),
        }
    )

    for split_name, split_labels in [
        ("train", train_dataset.labels),
        ("val", val_dataset.labels if val_dataset is not None else []),
        ("test", test_dataset.labels if test_dataset is not None else []),
    ]:
        _log_class_counts(split_name, split_labels)

    data = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }

    return data, metadata

