import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from utils.log import logger


def load_belonging_spectrograms(dataset_params, metadata):
    """
    Load pregenerated spectrograms from the scalogram-data directory.
    
    Args:
        dataset_params (dict): Parameters for dataset retrieval including:
            - spectrograms_dir: Path to directory containing spectrograms
            - labels_csv: Path to CSV file with student_id and belonging labels
            - channels: List of channel names
        metadata (dict): Metadata dictionary to be updated with relevant dataset info.
        
    Returns:
        X (dict): Dictionary with 'images' (list of image paths) and 'person_ids' (list of IDs)
        y (array): Belonging labels (0 or 1)
        metadata (dict): Updated metadata with dataset info.
    """
    
    spectrograms_dir = dataset_params.get('spectrograms_dir', '../scalogram-data/64x64')
    labels_csv = dataset_params.get('labels_csv', 'GT1_labels.csv')
    channels = dataset_params.get('channels', ['TP9', 'AF7', 'AF8', 'TP10'])
    allowed_ids = dataset_params.get('ten_percent_ids', None)
    if allowed_ids is None:
        allowed_ids = dataset_params.get('ten_percent_ids', None)
    
    logger.log('spectrograms_dir', spectrograms_dir)
    logger.log('labels_csv', labels_csv)
    
    # Load labels
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    
    labels_df = pd.read_csv(labels_csv)
    
    # Standardize column names
    if 'student_id' in labels_df.columns:
        id_col = 'student_id'
    else:
        raise ValueError("Labels CSV must have 'student_id' column")
    
    if 'GT1' in labels_df.columns:
        label_col = 'GT1'
    else:
        raise ValueError("Labels CSV must have 'GT1' column")
    
    labels_df[id_col] = labels_df[id_col].astype(str)
    labels_map = dict(zip(labels_df[id_col], labels_df[label_col]))
    
    # Collect image paths and labels
    image_paths = []
    person_ids = []
    labels = []
    
    if not os.path.exists(spectrograms_dir):
        raise FileNotFoundError(f"Spectrograms dir not found: {spectrograms_dir}")
    
    person_folders = [f for f in os.listdir(spectrograms_dir) 
                     if os.path.isdir(os.path.join(spectrograms_dir, f))]

    if allowed_ids is not None:
        allowed_ids_set = {str(pid) for pid in allowed_ids}
        if not allowed_ids_set:
            raise ValueError("allowed_ids is provided but empty")
        person_folders = [pid for pid in person_folders if pid in allowed_ids_set]
        logger.log('allowed_ids_count', len(allowed_ids_set))
    
    print(f"Found {len(person_folders)} person folders")
    
    for person_id in person_folders:
        person_dir = os.path.join(spectrograms_dir, person_id)
        
        # Check if this person has a label
        if person_id not in labels_map:
            print(f"Warning: No label found for {person_id}, skipping")
            continue
        
        label = labels_map[person_id]
        
        # Find all spectrogram images for this person
        for channel in channels:
            pattern = os.path.join(person_dir, f"*_{person_id}_{channel}_win*.png")
            channel_images = sorted(glob.glob(pattern))
            
            if not channel_images:
                print(f"Warning: No images found for {person_id}, channel {channel}")
                continue

            for img_path in channel_images:
                image_paths.append(img_path)
                person_ids.append(person_id)
                labels.append(label)
    
    if not image_paths:
        raise RuntimeError("No spectrogram images found matching labels")
    
    print(f"Loaded {len(image_paths)} spectrogram images from {len(set(person_ids))} people")

    #get sub-directory name for identifying image size
    sub_dir_name = os.path.basename(spectrograms_dir)
    
    if sub_dir_name == '64x64' or sub_dir_name == '64':
        image_size = (64, 64)
    elif sub_dir_name == '128x128' or sub_dir_name == '128':
        image_size = (128, 128)
    else:
        raise ValueError(f"Unknown image size directory: {sub_dir_name}")
    
    # Update metadata
    metadata.update({
        "num_images": len(image_paths),
        "num_people": len(set(person_ids)),
        "channels": channels,
        "num_channels": len(channels),
        "image_size": image_size,
        "num_classes": len(set(labels))
    })
    
    # Log statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique_labels, counts):
        logger.log(f'class_{lbl}_count', cnt)
    
    logger.log('total_images', len(image_paths))
    logger.log('num_people', len(set(person_ids)))
    
    X = {
        'images': image_paths,
        'person_ids': person_ids
    }
    y = np.array(labels)
    
    return X, y, metadata
