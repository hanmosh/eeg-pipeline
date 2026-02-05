import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold
from collections import Counter, defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from utils.log import logger


class SpectrogramDataset(Dataset):
    """PyTorch Dataset for loading spectrograms grouped by window across all channels"""
    
    def __init__(
        self,
        image_groups,
        labels,
        channels=['TP9', 'AF7', 'AF8', 'TP10'],
        transform=None,
        image_size=(64, 64)
    ):
        """
        Args:
            image_groups: List of dicts
            labels: Array of labels for each image group
            channels: List of channel names
            transform: Optional transform to apply to images
        """
        self.image_groups = image_groups
        self.labels = labels
        self.channels = channels
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_groups)
    
    def __getitem__(self, idx):
        # Load all 4 channel images for this window
        channels_data = []
        
        for channel in self.channels:
            img_path = self.image_groups[idx].get(channel)
            if img_path is None or not os.path.exists(img_path):
                # blank image
                img = np.zeros(self.image_size, dtype=np.float32)
            else:
                img = Image.open(img_path).convert('L')
                img = np.array(img, dtype=np.float32) / 255.0
            
            channels_data.append(img)
        
        image = np.stack(channels_data, axis=0)
        
        if self.transform:
            image = self.transform(image)
        
        image = torch.FloatTensor(image)
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return image, label


class SpectrogramPersonDataset(Dataset):
    """PyTorch Dataset for loading all window groups for a person"""

    def __init__(
        self,
        person_ids,
        person_to_groups,
        person_to_label,
        channels=['TP9', 'AF7', 'AF8', 'TP10'],
        transform=None,
        image_size=(64, 64),
        max_windows_per_person=None,
        downsample=False
    ):
        self.person_ids = [str(pid) for pid in person_ids]
        self.person_to_groups = person_to_groups
        self.person_to_label = person_to_label
        self.channels = channels
        self.transform = transform
        self.image_size = image_size
        self.max_windows_per_person = max_windows_per_person
        self.downsample = downsample

        self.labels = [self.person_to_label[pid] for pid in self.person_ids]

    def __len__(self):
        return len(self.person_ids)

    def _select_groups(self, groups):
        if not self.downsample:
            return groups
        if self.max_windows_per_person is None or self.max_windows_per_person <= 0:
            return groups
        if len(groups) <= self.max_windows_per_person:
            return groups
        idx = np.random.choice(len(groups), self.max_windows_per_person, replace=False)
        return [groups[i] for i in idx]

    def __getitem__(self, idx):
        person_id = self.person_ids[idx]
        groups = self.person_to_groups[person_id]
        groups = self._select_groups(groups)

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

            if self.transform:
                window = self.transform(window)

            windows.append(window)

        windows = np.stack(windows, axis=0)
        windows = torch.FloatTensor(windows)
        label = torch.LongTensor([self.person_to_label[person_id]])[0]

        return windows, label


def person_collate_fn(batch):
    windows, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(windows), labels


def group_images_by_window(image_paths, person_ids, channels):
    """
    Group images by person and window number across all channels.
    
    Returns:
        List of dictionaries, containing {channel: path} for one window
        List of person_ids for each group
    """
    # Extract person_id, channel, and window number
    image_info = []
    for img_path, person_id in zip(image_paths, person_ids):
        basename = os.path.basename(img_path)
        parts = basename.replace('.png', '').split('_')
        
        # Find channel and window parts
        channel = None
        window = None
        for i, part in enumerate(parts):
            if part in channels:
                channel = part
            if part.startswith('win'):
                window = part.replace('win', '')
        
        if channel and window:
            image_info.append({
                'path': img_path,
                'person_id': person_id,
                'channel': channel,
                'window': window
            })
    
    # Group by person_id and window
    groups_dict = {}
    for info in image_info:
        key = (info['person_id'], info['window'])
        if key not in groups_dict:
            groups_dict[key] = {'person_id': info['person_id']}
        groups_dict[key][info['channel']] = info['path']
    
    # Only include groups with all 4 channels -- should be all for this case
    complete_groups = []
    group_person_ids = []
    
    for (person_id, window), group in groups_dict.items():
        if all(ch in group for ch in channels):
            complete_groups.append(group)
            group_person_ids.append(person_id)
    
    return complete_groups, group_person_ids


def spectrogram_preprocessor(preprocessor_params, X, y, metadata):
    """
    Preprocess spectrogram data for training.
    
    Args:
        preprocessor_params (dict): test_split, val_split, batch_size, split_by_person,
            downsample_train, max_windows_per_person
        X (dict): Dict with images and person_ids
        y (array): Labels
        metadata (dict): Metadata dict
        
    Returns:
        data (dict): Dict with train/val/test dataloaders
        metadata (dict): Updated metadata
    """
    
    logger.log_dict(preprocessor_params)
    
    test_split = preprocessor_params.get('test_split', 0.2)
    val_split = preprocessor_params.get('val_split', 0.2)
    batch_size = preprocessor_params.get('batch_size', 16)
    force_single_person_batch = preprocessor_params.get('force_single_person_batch', False)
    split_by_person = preprocessor_params.get('split_by_person', True)
    cv_folds = preprocessor_params.get('cv_folds', 0)
    channels = metadata.get('channels', ['TP9', 'AF7', 'AF8', 'TP10'])
    image_size = metadata.get('image_size', (64, 64))
    downsample_train = preprocessor_params.get('downsample_train', False)
    max_windows_per_person = preprocessor_params.get('max_windows_per_person', None)

    if force_single_person_batch and batch_size != 1:
        print("force_single_person_batch enabled: overriding batch_size to 1")
        batch_size = 1
    
    image_paths = X['images']
    person_ids = X['person_ids']
    
    print("Grouping images by window across channels...")
    image_groups, group_person_ids = group_images_by_window(
        image_paths, person_ids, channels
    )
    
    print(f"Created {len(image_groups)} complete window groups")
    
    # Align labels with groups
    person_to_label = {}
    
    for person_id, label in zip(person_ids, y):
        person_id = str(person_id)
        if person_id not in person_to_label:
            person_to_label[person_id] = label

    person_to_groups = defaultdict(list)
    for group, pid in zip(image_groups, group_person_ids):
        person_to_groups[str(pid)].append(group)
    
    # Split data
    unique_person_ids = np.array(list(person_to_groups.keys()))
    unique_labels = np.array([person_to_label[str(pid)] for pid in unique_person_ids])

    def resolve_max_windows(person_ids):
        if not downsample_train:
            return max_windows_per_person
        if max_windows_per_person is not None and max_windows_per_person > 0:
            return max_windows_per_person
        counts = [len(person_to_groups[str(pid)]) for pid in person_ids]
        if not counts:
            return None
        return int(min(counts))

    def count_windows(person_ids):
        return int(sum(len(person_to_groups[str(pid)]) for pid in person_ids))

    if cv_folds and cv_folds > 1:
        if cv_folds > len(unique_person_ids):
            raise ValueError("cv_folds cannot exceed number of unique people")
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_data = []
        for fold_idx, (train_person_idx, val_person_idx) in enumerate(
            skf.split(unique_person_ids, unique_labels), start=1
        ):
            train_persons = unique_person_ids[train_person_idx]
            val_persons = unique_person_ids[val_person_idx]

            resolved_max_windows = resolve_max_windows(train_persons)

            print(
                f"Fold {fold_idx}/{cv_folds}: "
                f"Train {len(train_persons)} people ({count_windows(train_persons)} windows), "
                f"Val {len(val_persons)} people ({count_windows(val_persons)} windows)"
            )

            train_dataset = SpectrogramPersonDataset(
                train_persons,
                person_to_groups,
                person_to_label,
                channels=channels,
                image_size=image_size,
                max_windows_per_person=resolved_max_windows,
                downsample=downsample_train
            )
            val_dataset = SpectrogramPersonDataset(
                val_persons,
                person_to_groups,
                person_to_label,
                channels=channels,
                image_size=image_size,
                downsample=False
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=person_collate_fn
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=person_collate_fn
            )

            fold_data.append({
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': None,
                'cv_fold': fold_idx
            })

        metadata.update({
            'cv_folds': cv_folds,
            'batch_size': batch_size,
            'downsample_train': downsample_train,
            'max_windows_per_person': max_windows_per_person
        })

        return fold_data, metadata
    
    # First split: train+val vs test
    if test_split > 0:
        train_val_persons, test_persons = train_test_split(
            unique_person_ids,
            test_size=test_split,
            random_state=42,
            stratify=unique_labels
        )
    else:
        train_val_persons = unique_person_ids
        test_persons = np.array([])
    
    # Second split: train vs val
    if val_split > 0:
        train_val_labels = np.array([person_to_label[str(pid)] for pid in train_val_persons])
        train_persons, val_persons = train_test_split(
            train_val_persons,
            test_size=val_split,
            random_state=42,
            stratify=train_val_labels
        )
    else:
        train_persons = train_val_persons
        val_persons = np.array([])
    
    resolved_max_windows = resolve_max_windows(train_persons)

    train_labels = np.array([person_to_label[str(pid)] for pid in train_persons])
    val_labels = np.array([person_to_label[str(pid)] for pid in val_persons]) if len(val_persons) > 0 else np.array([])
    test_labels = np.array([person_to_label[str(pid)] for pid in test_persons]) if len(test_persons) > 0 else np.array([])

    train_windows = count_windows(train_persons)
    val_windows = count_windows(val_persons) if len(val_persons) > 0 else 0
    test_windows = count_windows(test_persons) if len(test_persons) > 0 else 0

    print(
        f"Train: {len(train_persons)} people ({train_windows} windows), "
        f"Val: {len(val_persons)} people ({val_windows} windows), "
        f"Test: {len(test_persons)} people ({test_windows} windows)"
    )

    # Create datasets
    train_dataset = SpectrogramPersonDataset(
        train_persons,
        person_to_groups,
        person_to_label,
        channels=channels,
        image_size=image_size,
        max_windows_per_person=resolved_max_windows,
        downsample=downsample_train
    )
    val_dataset = SpectrogramPersonDataset(
        val_persons,
        person_to_groups,
        person_to_label,
        channels=channels,
        image_size=image_size,
        downsample=False
    ) if len(val_persons) > 0 else None
    test_dataset = SpectrogramPersonDataset(
        test_persons,
        person_to_groups,
        person_to_label,
        channels=channels,
        image_size=image_size,
        downsample=False
    ) if len(test_persons) > 0 else None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=person_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=person_collate_fn
    ) if val_dataset is not None else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=person_collate_fn
    ) if test_dataset is not None else None

    # Update metadata
    metadata.update({
        'num_train_samples': len(train_persons),
        'num_val_samples': len(val_persons),
        'num_test_samples': len(test_persons),
        'num_train_windows': train_windows,
        'num_val_windows': val_windows,
        'num_test_windows': test_windows,
        'num_train_people': len(train_persons),
        'num_val_people': len(val_persons),
        'num_test_people': len(test_persons),
        'batch_size': batch_size,
        'downsample_train': downsample_train,
        'max_windows_per_person': resolved_max_windows
    })
    
    # Log class distribution (per person)
    for split_name, split_labels in [('train', train_labels), ('val', val_labels), ('test', test_labels)]:
        if len(split_labels) > 0:
            counter = Counter(split_labels)
            for label, count in counter.items():
                logger.log(f'{split_name}_class_{label}_count', count)
    
    data = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }
    
    return data, metadata
