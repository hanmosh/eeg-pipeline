from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from utils.log import logger


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

    channels, height, width = windows[0].shape[1:]
    padded_windows = torch.zeros((batch_size, max_len, channels, height, width), dtype=windows[0].dtype)

    for i, win in enumerate(windows):
        length = win.size(0)
        padded_windows[i, :length] = win

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


def tfrecord_preprocessor(preprocessor_params, X, y, metadata):
    """TFRecord sequence preprocessing (participant-wise splits)."""
    logger.log_dict(preprocessor_params)

    test_split = preprocessor_params.get('test_split', 0.2)
    val_split = preprocessor_params.get('val_split', 0.2)
    batch_size = preprocessor_params.get('batch_size', 16)
    cv_folds = preprocessor_params.get('cv_folds', 0)
    downsample_train = preprocessor_params.get('downsample_train', False)
    max_windows_per_person = preprocessor_params.get('max_windows_per_person', None)
    sequence_length = preprocessor_params.get('sequence_length', None)
    sequence_stride = preprocessor_params.get('sequence_stride', None)

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

    if cv_folds and cv_folds > 1:
        if cv_folds > len(unique_person_ids):
            raise ValueError("cv_folds cannot exceed number of unique people")
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_data = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(unique_person_ids, unique_labels), start=1):
            train_persons = unique_person_ids[train_idx]
            val_persons = unique_person_ids[val_idx]

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
                downsample=False,
            )

            print(
                f"Fold {fold_idx}/{cv_folds}: "
                f"Train {len(train_persons)} people ({train_dataset.num_windows} windows), "
                f"Val {len(val_persons)} people ({val_dataset.num_windows} windows)"
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=sequence_collate_fn,
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
                'val_loader': val_loader,
                'test_loader': None,
                'cv_fold': fold_idx,
            })

        metadata.update({
            'cv_folds': cv_folds,
            'batch_size': batch_size,
            'downsample_train': downsample_train,
            'max_windows_per_person': max_windows_per_person,
            'sequence_length': sequence_length,
            'sequence_stride': sequence_stride,
        })

        return fold_data, metadata

    _validate_stratified_splits(
        len(unique_person_ids), test_split, val_split, num_classes, class_counts
    )

    if test_split > 0:
        train_val_persons, test_persons = train_test_split(
            unique_person_ids,
            test_size=test_split,
            random_state=42,
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
            random_state=42,
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
        downsample=False,
    ) if len(val_persons) > 0 else None
    test_dataset = TFRecordSequenceDataset(
        test_persons,
        person_to_windows,
        person_to_label,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
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
