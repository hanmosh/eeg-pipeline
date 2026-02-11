import os
import glob
import re
import numpy as np

from utils.log import logger


_TF_IMPORT_ERROR = (
    "TensorFlow is required to decode serialized tensors. "
    "Install it or re-export scalograms/shape as raw numpy bytes."
)

_FILENAME_PATTERN = re.compile(
    r"^eeg_(?P<person>[^_]+)_(?P<question>\d+)_(?P<timestamp>\d+)_scalograms\.tfrecord$"
)


def _parse_filename(filename):
    match = _FILENAME_PATTERN.match(filename)
    if not match:
        return None, None, None
    person_id = match.group('person')
    try:
        question_num = int(match.group('question'))
    except ValueError:
        return None, None, None
    try:
        timestamp = int(match.group('timestamp'))
    except ValueError:
        return None, None, None
    return person_id, question_num, timestamp


def _coerce_bytes(value, field_name):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, np.ndarray):
        if value.size == 1:
            item = value.item()
            if isinstance(item, (bytes, bytearray, memoryview)):
                return bytes(item)
        if value.dtype == np.uint8:
            return value.tobytes()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected single value for '{field_name}', got {len(value)}")
        return _coerce_bytes(value[0], field_name)
    raise TypeError(f"Unsupported type for '{field_name}': {type(value)}")


def _coerce_int(value, field_name):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError(f"Empty value for '{field_name}'")
        value = value.ravel()[0]
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(f"Empty value for '{field_name}'")
        value = value[0]
    return int(value)


def _get_tensorflow():
    try:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf  # type: ignore
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass
    except Exception:
        return None
    return tf


def _try_parse_tensorproto(byte_data, dtype):
    tf = _get_tensorflow()
    if tf is None:
        return None
    try:
        tensor = tf.io.parse_tensor(byte_data, out_type=tf.dtypes.as_dtype(dtype))
        return tensor.numpy()
    except Exception:
        return None


def _decode_shape(shape_value, tfrecord_path):
    if isinstance(shape_value, (list, tuple, np.ndarray)) and not isinstance(shape_value, (bytes, bytearray, memoryview)):
        shape = np.array(shape_value, dtype=int).ravel()
        if shape.size in (3, 4) and np.all(shape > 0):
            return shape

    shape_bytes = _coerce_bytes(shape_value, 'shape')
    for dtype in (np.int32, np.int64):
        shape = np.frombuffer(shape_bytes, dtype=dtype)
        if shape.size in (3, 4) and np.all(shape > 0):
            return shape.astype(int)

    parsed = _try_parse_tensorproto(shape_bytes, np.int32)
    if parsed is None:
        parsed = _try_parse_tensorproto(shape_bytes, np.int64)
    if parsed is None:
        raise ValueError(
            f"Unable to decode 'shape' from {tfrecord_path}. {_TF_IMPORT_ERROR}"
        )

    shape = np.array(parsed, dtype=int).ravel()
    if shape.size not in (3, 4) or not np.all(shape > 0):
        raise ValueError(f"Invalid 'shape' decoded from {tfrecord_path}: {shape}")
    return shape


def _decode_scalograms(scalogram_value, shape, tfrecord_path):
    expected = int(np.prod(shape))
    if isinstance(scalogram_value, np.ndarray) and scalogram_value.dtype != np.object_:
        array = np.asarray(scalogram_value, dtype=np.float32)
        if array.size == expected:
            return array.reshape(tuple(shape))
        if tuple(array.shape) == tuple(shape):
            return array

    scalogram_bytes = _coerce_bytes(scalogram_value, 'scalograms')
    flat = np.frombuffer(scalogram_bytes, dtype=np.float32)
    if flat.size == expected:
        return flat.reshape(tuple(shape))

    parsed = _try_parse_tensorproto(scalogram_bytes, np.float32)
    if parsed is None:
        raise ValueError(
            f"Unable to decode 'scalograms' from {tfrecord_path}. {_TF_IMPORT_ERROR}"
        )

    array = np.array(parsed, dtype=np.float32)
    if array.size == expected:
        return array.reshape(tuple(shape))
    if tuple(array.shape) != tuple(shape):
        raise ValueError(
            f"Decoded scalograms shape {array.shape} does not match expected {tuple(shape)} "
            f"for {tfrecord_path}."
        )
    return array


def _load_tfrecord_records_tensorflow(tfrecord_path, compression_type=None):
    tf = _get_tensorflow()
    if tf is None:
        return None
    feature_description = {
        'scalograms': tf.io.FixedLenFeature([], tf.string),
        'shape': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    records = []
    dataset = tf.data.TFRecordDataset([tfrecord_path], compression_type=compression_type)
    for raw_record in dataset:
        example = tf.io.parse_single_example(raw_record, feature_description)
        records.append({
            'scalograms': example['scalograms'].numpy(),
            'shape': example['shape'].numpy(),
            'label': example['label'].numpy(),
        })
    return records


def _load_tfrecord_records(tfrecord_path, compression_type=None):
    if compression_type:
        records = _load_tfrecord_records_tensorflow(tfrecord_path, compression_type=compression_type)
        if records is None:
            raise ImportError(
                "TensorFlow is required to read compressed TFRecords. "
                "Install TensorFlow or remove the compression setting."
            )
        return records

    records = None
    try:
        from tfrecord.tfrecord_loader import tfrecord_loader  # type: ignore
        description = {
            'scalograms': 'byte',
            'shape': 'byte',
            'label': 'int',
        }
        records = list(tfrecord_loader(tfrecord_path, None, description))
        if records:
            return records
    except Exception:
        records = None

    records_tf = _load_tfrecord_records_tensorflow(tfrecord_path)
    if records_tf is not None:
        return records_tf

    if records is not None:
        return records

    raise ImportError(
        "The 'tfrecord' package is not installed and TensorFlow is unavailable. "
        "Install with `pip install tfrecord` or install TensorFlow to read TFRecords."
    )


def load_belonging_tfrecords(dataset_params, metadata):
    """Load TFRecord scalograms and labels per participant session."""
    tfrecords_dir = dataset_params.get('tfrecords_dir')
    compression_type = dataset_params.get('tfrecords_compression')
    channels = dataset_params.get('channels', ['TP9', 'AF7', 'AF8', 'TP10'])

    if not tfrecords_dir:
        raise ValueError("dataset_params must include 'tfrecords_dir'")
    if not os.path.exists(tfrecords_dir):
        raise FileNotFoundError(f"TFRecords dir not found: {tfrecords_dir}")

    tfrecord_paths = glob.glob(os.path.join(tfrecords_dir, '*.tfrecord'))
    if not tfrecord_paths:
        raise RuntimeError(f"No TFRecord files found in {tfrecords_dir}")
    parsed_paths = []
    for tfrecord_path in tfrecord_paths:
        filename = os.path.basename(tfrecord_path)
        person_id, question_num, timestamp = _parse_filename(filename)
        if person_id is None or question_num is None or timestamp is None:
            raise ValueError(
                f"Unable to parse person/question/timestamp from TFRecord filename: {filename}"
            )
        parsed_paths.append((person_id, question_num, timestamp, tfrecord_path))
    parsed_paths.sort(key=lambda item: (item[0], item[1], item[2]))

    person_to_scalograms = {}
    person_to_label = {}
    skipped_empty = []
    skipped_question = []

    image_size = None
    num_channels = None
    total_windows = 0

    for person_id, question_num, _timestamp, tfrecord_path in parsed_paths:
        filename = os.path.basename(tfrecord_path)
        if question_num > 33:
            skipped_question.append(tfrecord_path)
            continue
        records = _load_tfrecord_records(tfrecord_path, compression_type=compression_type)
        if len(records) == 0:
            skipped_empty.append(tfrecord_path)
            continue
        if len(records) != 1:
            raise ValueError(
                f"Expected exactly 1 record in {tfrecord_path}, found {len(records)}. "
                "If the TFRecord was written with compression, set "
                "`dataset_params.tfrecords_compression` to 'GZIP' or 'ZLIB'."
            )
        record = records[0]

        shape_value = record.get('shape')
        scalogram_value = record.get('scalograms')
        label = _coerce_int(record.get('label'), 'label')

        shape = _decode_shape(shape_value, tfrecord_path)
        if shape.size != 4:
            raise ValueError(
                f"Expected shape with 4 dims (N, C, H, W) in {tfrecord_path}, got {shape}."
            )

        scalograms = _decode_scalograms(scalogram_value, shape, tfrecord_path)
        scalograms = np.asarray(scalograms, dtype=np.float32)

        n_windows, n_channels, height, width = scalograms.shape
        if num_channels is None:
            num_channels = n_channels
        elif num_channels != n_channels:
            raise ValueError(
                f"Inconsistent channel counts: expected {num_channels}, got {n_channels} in {tfrecord_path}."
            )
        if image_size is None:
            image_size = (height, width)
        elif image_size != (height, width):
            raise ValueError(
                f"Inconsistent image sizes: expected {image_size}, got {(height, width)} in {tfrecord_path}."
            )

        total_windows += int(n_windows)
        person_to_scalograms.setdefault(person_id, []).append(scalograms)
        if person_id in person_to_label and person_to_label[person_id] != label:
            raise ValueError(
                f"Label mismatch for person {person_id}: "
                f"{person_to_label[person_id]} vs {label} in {tfrecord_path}."
            )
        person_to_label[person_id] = label

    if channels and num_channels is not None and len(channels) != num_channels:
        raise ValueError(
            f"Configured channels length ({len(channels)}) does not match TFRecord channels ({num_channels})."
        )
    if skipped_empty:
        logger.log('skipped_empty_tfrecords', len(skipped_empty))
    if skipped_question:
        logger.log('skipped_question_tfrecords', len(skipped_question))
    if not person_to_scalograms:
        raise RuntimeError(
            "No non-empty TFRecord files found. "
            "If the TFRecords were written with compression, set "
            "`dataset_params.tfrecords_compression` to 'GZIP' or 'ZLIB'."
        )

    person_ids = sorted(person_to_scalograms.keys())
    scalograms_list = [np.concatenate(person_to_scalograms[pid], axis=0) for pid in person_ids]
    labels = [person_to_label[pid] for pid in person_ids]

    unique_labels, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique_labels, counts):
        logger.log(f'class_{lbl}_count', cnt)
    logger.log('total_windows', total_windows)
    logger.log('num_people', len(set(person_ids)))

    metadata.update({
        'num_people': len(set(person_ids)),
        'num_windows': total_windows,
        'channels': channels,
        'num_channels': num_channels or len(channels),
        'image_size': image_size,
        'num_classes': len(set(labels)),
    })

    X = {
        'scalograms': scalograms_list,
        'person_ids': person_ids,
    }
    y = np.array(labels)

    return X, y, metadata
