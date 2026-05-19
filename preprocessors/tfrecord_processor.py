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
    ):
        self.person_ids = [str(pid) for pid in person_ids]
        self.person_to_windows = {}
        self.person_to_label = person_to_label
        self.sequence_length = sequence_length or 0
        self.sequence_stride = sequence_stride or 0
        self.max_windows_per_person = max_windows_per_person
        self.downsample = downsample

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
    else:
        raise ValueError(
            f"Unsupported sample shape {tuple(windows[0].shape)}. "
            "Expected [T, C, H, W]."
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


def tfrecord_preprocessor(preprocessor_params, X, y, metadata):
    """TFRecord sequence preprocessing (participant-wise splits)."""
    logger.log_dict(preprocessor_params)

    test_split = preprocessor_params.get('test_split', 0.2)
    val_split = preprocessor_params.get('val_split', 0.2)
    batch_size = preprocessor_params.get('batch_size', 16)
    cv_folds = preprocessor_params.get('cv_folds', 0)
    cv_repeats = int(preprocessor_params.get('cv_repeats', 1))
    downsample_train = preprocessor_params.get('downsample_train', False)
    max_windows_per_person = preprocessor_params.get('max_windows_per_person', None)
    sequence_length = preprocessor_params.get('sequence_length', None)
    sequence_stride = preprocessor_params.get('sequence_stride', None)
    cv_fold_as_test = bool(preprocessor_params.get('cv_fold_as_test', False))
    seed = preprocessor_params.get('seed', None)
    effective_seed = int(seed) if seed is not None else None

    def _build_generator(seed_value):
        if seed_value is None:
            return None
        gen = torch.Generator()
        gen.manual_seed(int(seed_value))
        return gen

    scalograms_list = X.get('scalograms')
    person_ids = X.get('person_ids')
    if scalograms_list is None or person_ids is None:
        raise ValueError("X must contain 'scalograms' and 'person_ids'")
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
        logger.log('fold_manifest_csv', manifest_spec['path'])
        logger.log('fold_manifest_column', manifest_spec['fold_col'])

    if cv_folds and cv_folds > 1:
        if cv_repeats < 1:
            raise ValueError("cv_repeats must be >= 1 when cv_folds > 1.")
        if test_split < 0 or test_split >= 1:
            raise ValueError("test_split must be in [0, 1) when cv_folds > 1.")

        cv_people = unique_person_ids
        cv_labels = unique_labels
        holdout_test_people = np.array([])
        holdout_test_labels = np.array([])

        if manifest_spec is not None:
            cv_people = np.asarray(manifest_spec['ordered_people'], dtype=str)
            cv_labels = np.array([person_to_label[str(pid)] for pid in cv_people])
        elif test_split > 0:
            cv_people, holdout_test_people = train_test_split(
                unique_person_ids,
                test_size=test_split,
                random_state=effective_seed,
                stratify=unique_labels,
            )
            cv_labels = np.array([person_to_label[str(pid)] for pid in cv_people])
            holdout_test_labels = np.array([person_to_label[str(pid)] for pid in holdout_test_people])

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

        if manifest_spec is not None:
            total_cv_folds = cv_folds
            split_plan = []
            ordered_people = manifest_spec['ordered_people']
            for fold_num in range(1, cv_folds + 1):
                val_persons = np.asarray(manifest_spec['fold_to_people'][fold_num], dtype=str)
                val_people_set = set(val_persons.tolist())
                train_persons = np.asarray(
                    [pid for pid in ordered_people if pid not in val_people_set],
                    dtype=str,
                )
                split_plan.append((fold_num, 1, fold_num, train_persons, val_persons))
        elif cv_repeats > 1:
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
        else:
            splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=effective_seed)
            total_cv_folds = cv_folds
            split_plan = []
            for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(cv_people, cv_labels), start=1):
                train_persons = cv_people[train_idx]
                val_persons = cv_people[val_idx]
                split_plan.append((fold_idx, 1, fold_idx, train_persons, val_persons))

        fold_data = []
        test_dataset = TFRecordSequenceDataset(
            holdout_test_people,
            person_to_windows,
            person_to_label,
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
            max_windows_per_person=max_windows_per_person,
            downsample=False,
        ) if len(holdout_test_people) > 0 else None
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=sequence_collate_fn,
        ) if test_dataset is not None else None

        if test_dataset is not None:
            print(
                f"Holdout Test: {len(holdout_test_people)} people "
                f"({test_dataset.num_windows} windows)"
            )

        for fold_idx, repeat_idx, fold_in_repeat, train_persons, val_persons in split_plan:
            train_labels = [person_to_label[str(pid)] for pid in train_persons]
            val_labels = [person_to_label[str(pid)] for pid in val_persons]
            train_pos = int(sum(train_labels))
            val_pos = int(sum(val_labels))
            train_neg = len(train_labels) - train_pos
            val_neg = len(val_labels) - val_pos

            train_dataset = TFRecordSequenceDataset(
                train_persons,
                person_to_windows,
                person_to_label,
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                max_windows_per_person=max_windows_per_person,
                downsample=downsample_train,
            )
            val_dataset = TFRecordSequenceDataset(
                val_persons,
                person_to_windows,
                person_to_label,
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                max_windows_per_person=max_windows_per_person,
                downsample=False,
            )

            if cv_repeats > 1:
                print(
                    f"Repeat {repeat_idx}/{cv_repeats}, Fold {fold_in_repeat}/{cv_folds} "
                    f"(global {fold_idx}/{total_cv_folds}): "
                    f"Train {len(train_persons)} people ({train_dataset.num_windows} windows), "
                    f"Val {len(val_persons)} people ({val_dataset.num_windows} windows)"
                )
            else:
                print(
                    f"Fold {fold_idx}/{cv_folds}: "
                    f"Train {len(train_persons)} people ({train_dataset.num_windows} windows), "
                    f"Val {len(val_persons)} people ({val_dataset.num_windows} windows)"
                )
            print(
                f"Fold {fold_idx}: Train {train_pos} pos / {train_neg} neg | "
                f"Val {val_pos} pos / {val_neg} neg"
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=sequence_collate_fn,
                generator=_build_generator(effective_seed + fold_idx) if effective_seed is not None else None,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=sequence_collate_fn,
            )

            fold_data.append({
                'train_loader': train_loader,
                'val_loader': None if cv_fold_as_test else val_loader,
                'test_loader': val_loader if cv_fold_as_test else test_loader,
                'cv_fold': fold_idx,
                'cv_fold_in_repeat': fold_in_repeat,
                'cv_repeat_index': repeat_idx,
                'cv_total_folds': total_cv_folds,
            })

        metadata.update({
            'cv_folds': cv_folds,
            'cv_repeats': cv_repeats,
            'cv_total_folds': total_cv_folds,
            'batch_size': batch_size,
            'downsample_train': downsample_train,
            'max_windows_per_person': max_windows_per_person,
            'sequence_length': sequence_length,
            'sequence_stride': sequence_stride,
            'cv_fold_as_test': cv_fold_as_test,
            'seed': effective_seed,
            'test_split': test_split,
            'num_test_people': len(holdout_test_people),
            'num_test_windows': test_dataset.num_windows if test_dataset is not None else 0,
            'has_holdout_test': test_dataset is not None,
            'fold_manifest_used': manifest_spec is not None,
            'fold_manifest_csv': manifest_spec['path'] if manifest_spec is not None else None,
            'fold_manifest_column': manifest_spec['fold_col'] if manifest_spec is not None else None,
        })

        if test_dataset is not None and len(holdout_test_labels) > 0:
            counter = Counter(holdout_test_labels.tolist())
            for label, count in counter.items():
                logger.log(f'test_class_{label}_count', count)

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
        test_persons = np.array([])

    if val_split > 0:
        train_val_labels = np.array([person_to_label[str(pid)] for pid in train_val_persons])
        train_persons, val_persons = train_test_split(
            train_val_persons,
            test_size=val_split,
            random_state=effective_seed,
            stratify=train_val_labels,
        )
    else:
        train_persons = train_val_persons
        val_persons = np.array([])

    train_dataset = TFRecordSequenceDataset(
        train_persons,
        person_to_windows,
        person_to_label,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        max_windows_per_person=max_windows_per_person,
        downsample=downsample_train,
    )
    val_dataset = TFRecordSequenceDataset(
        val_persons,
        person_to_windows,
        person_to_label,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        max_windows_per_person=max_windows_per_person,
        downsample=False,
    ) if len(val_persons) > 0 else None
    test_dataset = TFRecordSequenceDataset(
        test_persons,
        person_to_windows,
        person_to_label,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        max_windows_per_person=max_windows_per_person,
        downsample=False,
    ) if len(test_persons) > 0 else None

    train_windows = train_dataset.num_windows
    val_windows = val_dataset.num_windows if val_dataset is not None else 0
    test_windows = test_dataset.num_windows if test_dataset is not None else 0

    print(
        f"Train: {len(train_persons)} people ({train_windows} windows), "
        f"Val: {len(val_persons)} people ({val_windows} windows), "
        f"Test: {len(test_persons)} people ({test_windows} windows)"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=sequence_collate_fn,
        generator=_build_generator(effective_seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=sequence_collate_fn,
    ) if val_dataset is not None else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=sequence_collate_fn,
    ) if test_dataset is not None else None

    metadata.update({
        'num_train_samples': train_windows,
        'num_val_samples': val_windows,
        'num_test_samples': test_windows,
        'num_train_windows': train_windows,
        'num_val_windows': val_windows,
        'num_test_windows': test_windows,
        'num_train_people': len(train_persons),
        'num_val_people': len(val_persons),
        'num_test_people': len(test_persons),
        'batch_size': batch_size,
        'downsample_train': downsample_train,
        'max_windows_per_person': max_windows_per_person,
        'sequence_length': sequence_length,
        'sequence_stride': sequence_stride,
        'seed': effective_seed,
    })

    for split_name, split_labels in [
        ('train', train_dataset.labels),
        ('val', val_dataset.labels if val_dataset is not None else []),
        ('test', test_dataset.labels if test_dataset is not None else []),
    ]:
        if len(split_labels) > 0:
            counter = Counter(split_labels)
            for label, count in counter.items():
                logger.log(f'{split_name}_class_{label}_count', count)

    data = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
    }

    return data, metadata
