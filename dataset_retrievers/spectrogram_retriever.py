import os
import glob
import numpy as np
import pandas as pd
from utils.log import logger


def load_belonging_spectrograms(dataset_params, metadata):
    """Load spectrogram image paths and participant labels."""
    spectrograms_dir = dataset_params.get('spectrograms_dir', '../scalogram-data/64x64')
    labels_csv = dataset_params.get('labels_csv', 'GT1_labels.csv')
    label_col = dataset_params.get('label_col', 'GT1')
    channels = dataset_params.get('channels', ['TP9', 'AF7', 'AF8', 'TP10'])

    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    if not os.path.exists(spectrograms_dir):
        raise FileNotFoundError(f"Spectrograms dir not found: {spectrograms_dir}")

    labels_df = pd.read_csv(labels_csv)
    if 'student_id' not in labels_df.columns:
        raise ValueError("Labels CSV must have 'student_id' column")
    if label_col not in labels_df.columns:
        raise ValueError(f"Labels CSV must have '{label_col}' column")

    labels_df['student_id'] = labels_df['student_id'].astype(str)
    labels_map = dict(zip(labels_df['student_id'], labels_df[label_col]))

    image_paths = []
    person_ids = []
    labels = []

    person_folders = [
        f for f in os.listdir(spectrograms_dir)
        if os.path.isdir(os.path.join(spectrograms_dir, f))
    ]

    for person_id in person_folders:
        if person_id not in labels_map:
            continue
        label = labels_map[person_id]
        person_dir = os.path.join(spectrograms_dir, person_id)

        for channel in channels:
            pattern = os.path.join(person_dir, f"*_{person_id}_{channel}_win*.png")
            for img_path in sorted(glob.glob(pattern)):
                image_paths.append(img_path)
                person_ids.append(person_id)
                labels.append(label)

    if not image_paths:
        raise RuntimeError("No spectrogram images found matching labels")

    sub_dir_name = os.path.basename(spectrograms_dir)
    if sub_dir_name in ('64x64', '64'):
        image_size = (64, 64)
    elif sub_dir_name in ('128x128', '128'):
        image_size = (128, 128)
    else:
        raise ValueError(f"Unknown image size directory: {sub_dir_name}")

    metadata.update({
        'num_images': len(image_paths),
        'num_people': len(set(person_ids)),
        'channels': channels,
        'num_channels': len(channels),
        'image_size': image_size,
        'num_classes': len(set(labels)),
    })

    unique_labels, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique_labels, counts):
        logger.log(f'class_{lbl}_count', cnt)
    logger.log('total_images', len(image_paths))
    logger.log('num_people', len(set(person_ids)))

    X = {
        'images': image_paths,
        'person_ids': person_ids,
    }
    y = np.array(labels)

    return X, y, metadata
