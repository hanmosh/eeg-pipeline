import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from utils.log import logger


class SpectrogramDataset(Dataset):
    """PyTorch Dataset for loading spectrograms grouped by window across all channels"""
    
    def __init__(self, image_groups, labels, channels=['TP9', 'AF7', 'AF8', 'TP10'], 
                 transform=None):
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
        
    def __len__(self):
        return len(self.image_groups)
    
    def __getitem__(self, idx):
        # Load all 4 channel images for this window
        channels_data = []
        
        for channel in self.channels:
            img_path = self.image_groups[idx].get(channel)
            if img_path is None or not os.path.exists(img_path):
                # blank image
                img = np.zeros((64, 64), dtype=np.float32)
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
        preprocessor_params (dict): test_split, val_split, batch_size, split_by_person
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
    split_by_person = preprocessor_params.get('split_by_person', True)
    channels = metadata.get('channels', ['TP9', 'AF7', 'AF8', 'TP10'])
    
    image_paths = X['images']
    person_ids = X['person_ids']
    
    print("Grouping images by window across channels...")
    image_groups, group_person_ids = group_images_by_window(
        image_paths, person_ids, channels
    )
    
    print(f"Created {len(image_groups)} complete window groups")
    
    # Align labels with groups
    unique_persons = list(set(group_person_ids))
    person_to_label = {}
    
    for person_id, label in zip(person_ids, y):
        if person_id not in person_to_label:
            person_to_label[person_id] = label
    
    group_labels = np.array([person_to_label[pid] for pid in group_person_ids])
    
    # Split data
    unique_person_ids = np.array(list(set(group_person_ids)))
    unique_labels = np.array([person_to_label[pid] for pid in unique_person_ids])
    
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
        train_val_labels = np.array([person_to_label[pid] for pid in train_val_persons])
        train_persons, val_persons = train_test_split(
            train_val_persons,
            test_size=val_split,
            random_state=42,
            stratify=train_val_labels
        )
    else:
        train_persons = train_val_persons
        val_persons = np.array([])
    
    #indices for each split
    train_idx = [i for i, pid in enumerate(group_person_ids) if pid in train_persons]
    val_idx = [i for i, pid in enumerate(group_person_ids) if pid in val_persons]
    test_idx = [i for i, pid in enumerate(group_person_ids) if pid in test_persons]
    
    # Create datasets
    train_groups = [image_groups[i] for i in train_idx]
    train_labels = group_labels[train_idx]
    
    val_groups = [image_groups[i] for i in val_idx] if len(val_idx) > 0 else []
    val_labels = group_labels[val_idx] if len(val_idx) > 0 else np.array([])
    
    test_groups = [image_groups[i] for i in test_idx] if len(test_idx) > 0 else []
    test_labels = group_labels[test_idx] if len(test_idx) > 0 else np.array([])
    
    print(f"Train: {len(train_groups)} windows, Val: {len(val_groups)} windows, Test: {len(test_groups)} windows")
    
    train_people = set([group_person_ids[i] for i in train_idx])
    val_people = set([group_person_ids[i] for i in val_idx]) if len(val_idx) > 0 else set()
    test_people = set([group_person_ids[i] for i in test_idx]) if len(test_idx) > 0 else set()
    
    print(f"Train people: {len(train_people)}, Val people: {len(val_people)}, Test people: {len(test_people)}")
    
    # Create datasets
    train_dataset = SpectrogramDataset(train_groups, train_labels, channels=channels)
    val_dataset = SpectrogramDataset(val_groups, val_labels, channels=channels) if len(val_groups) > 0 else None
    test_dataset = SpectrogramDataset(test_groups, test_labels, channels=channels) if len(test_groups) > 0 else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    ) if val_dataset is not None else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    ) if test_dataset is not None else None
    
    # Update metadata
    metadata.update({
        'num_train_samples': len(train_groups),
        'num_val_samples': len(val_groups),
        'num_test_samples': len(test_groups),
        'num_train_people': len(train_people),
        'num_val_people': len(val_people),
        'num_test_people': len(test_people),
        'batch_size': batch_size
    })
    
    # Log class distribution
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