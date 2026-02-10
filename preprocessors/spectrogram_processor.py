import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from utils.log import logger


class SpectrogramSequenceDataset(Dataset):
    """Sequences of window tensors for each participant."""

    def __init__(
        self,
        person_ids,
        person_to_groups,
        person_to_label,
        channels,
        image_size,
        sequence_length=None,
        sequence_stride=None,
        max_windows_per_person=None,
        downsample=False,
    ):
        self.person_ids = [str(pid) for pid in person_ids]
        self.person_to_groups = person_to_groups
        self.person_to_label = person_to_label
        self.channels = channels
        self.image_size = image_size
        self.sequence_length = sequence_length or 0
        self.sequence_stride = sequence_stride or 0
        self.max_windows_per_person = max_windows_per_person
        self.downsample = downsample

        self.sequences = []
        self.sequence_labels = []
        self.sequence_pids = []
        self.labels = []
        self.num_windows = 0

        for pid in self.person_ids:
            groups = list(self.person_to_groups[pid])
            groups = sorted(groups, key=self._window_sort_key)
            groups = self._maybe_downsample(groups)
            if not groups:
                continue

            if self.sequence_length > 0:
                stride = self.sequence_stride if self.sequence_stride > 0 else self.sequence_length
                for start in range(0, len(groups), stride):
                    seq = groups[start:start + self.sequence_length]
                    if not seq:
                        continue
                    self._add_sequence(pid, seq)
            else:
                self._add_sequence(pid, groups)

    def __len__(self):
        return len(self.sequences)

    def _add_sequence(self, pid, seq):
        label = int(self.person_to_label[pid])
        self.sequences.append(seq)
        self.sequence_labels.append(label)
        self.sequence_pids.append(pid)
        self.labels.append(label)
        self.num_windows += len(seq)

    def _window_sort_key(self, group):
        window = group.get('window')
        try:
            return (0, int(window))
        except (TypeError, ValueError):
            return (1, str(window))

    def _maybe_downsample(self, groups):
        if not self.downsample:
            return groups
        if self.max_windows_per_person is None or self.max_windows_per_person <= 0:
            return groups
        if len(groups) <= self.max_windows_per_person:
            return groups
        start_max = len(groups) - self.max_windows_per_person
        start = np.random.randint(0, start_max + 1)
        return groups[start:start + self.max_windows_per_person]

    def __getitem__(self, idx):
        groups = self.sequences[idx]
        label = self.sequence_labels[idx]
        pid = self.sequence_pids[idx]

        windows = []
        for group in groups:
            channels_data = []
            for channel in self.channels:
                img_path = group.get(channel)
                if img_path is None or not os.path.exists(img_path):
                    img = np.zeros(self.image_size, dtype=np.float32)
                else:
                    img = Image.open(img_path).convert('L')
                    img = np.array(img, dtype=np.float32) / 255.0
                channels_data.append(img)

            window = np.stack(channels_data, axis=0)
            windows.append(window)

        windows = torch.FloatTensor(np.stack(windows, axis=0))
        return windows, int(label), pid


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


def group_images_by_window(image_paths, person_ids, channels):
    image_info = []
    for img_path, person_id in zip(image_paths, person_ids):
        basename = os.path.basename(img_path)
        parts = basename.replace('.png', '').split('_')
        channel = None
        window = None
        for part in parts:
            if part in channels:
                channel = part
            if part.startswith('win'):
                window = part.replace('win', '')
        if channel and window:
            image_info.append({
                'path': img_path,
                'person_id': person_id,
                'channel': channel,
                'window': window,
            })

    groups_dict = {}
    for info in image_info:
        key = (info['person_id'], info['window'])
        if key not in groups_dict:
            groups_dict[key] = {'person_id': info['person_id'], 'window': info['window']}
        groups_dict[key][info['channel']] = info['path']

    complete_groups = []
    group_person_ids = []
    for (person_id, _window), group in groups_dict.items():
        if all(ch in group for ch in channels):
            complete_groups.append(group)
            group_person_ids.append(person_id)

    return complete_groups, group_person_ids


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


def spectrogram_preprocessor(preprocessor_params, X, y, metadata):
    """Window-level sequence preprocessing (participant-wise splits)."""
    logger.log_dict(preprocessor_params)

    test_split = preprocessor_params.get('test_split', 0.2)
    val_split = preprocessor_params.get('val_split', 0.2)
    batch_size = preprocessor_params.get('batch_size', 16)
    cv_folds = preprocessor_params.get('cv_folds', 0)
    downsample_train = preprocessor_params.get('downsample_train', False)
    max_windows_per_person = preprocessor_params.get('max_windows_per_person', None)
    sequence_length = preprocessor_params.get('sequence_length', None)
    sequence_stride = preprocessor_params.get('sequence_stride', None)

    channels = metadata.get('channels', ['TP9', 'AF7', 'AF8', 'TP10'])
    image_size = metadata.get('image_size', (64, 64))

    image_paths = X['images']
    person_ids = X['person_ids']

    image_groups, group_person_ids = group_images_by_window(image_paths, person_ids, channels)

    person_to_label = {}
    for person_id, label in zip(person_ids, y):
        person_id = str(person_id)
        if person_id not in person_to_label:
            person_to_label[person_id] = label

    person_to_groups = defaultdict(list)
    for group, pid in zip(image_groups, group_person_ids):
        person_to_groups[str(pid)].append(group)

    unique_person_ids = np.array(list(person_to_groups.keys()))
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

            train_dataset = SpectrogramSequenceDataset(
                train_persons,
                person_to_groups,
                person_to_label,
                channels=channels,
                image_size=image_size,
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                max_windows_per_person=max_windows_per_person,
                downsample=downsample_train,
            )
            val_dataset = SpectrogramSequenceDataset(
                val_persons,
                person_to_groups,
                person_to_label,
                channels=channels,
                image_size=image_size,
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

    train_dataset = SpectrogramSequenceDataset(
        train_persons,
        person_to_groups,
        person_to_label,
        channels=channels,
        image_size=image_size,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        max_windows_per_person=max_windows_per_person,
        downsample=downsample_train,
    )
    val_dataset = SpectrogramSequenceDataset(
        val_persons,
        person_to_groups,
        person_to_label,
        channels=channels,
        image_size=image_size,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        downsample=False,
    ) if len(val_persons) > 0 else None
    test_dataset = SpectrogramSequenceDataset(
        test_persons,
        person_to_groups,
        person_to_label,
        channels=channels,
        image_size=image_size,
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
