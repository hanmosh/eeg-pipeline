import csv
import os
import glob
import re
import numpy as np

from utils.log import logger
from utils.text_utils import normalize_token


_TF_IMPORT_ERROR = (
    "TensorFlow is required to decode serialized tensors. "
    "Install it or re-export scalograms/shape as raw numpy bytes."
)
_DEFAULT_ALL_NLP_CSV = os.path.join(
    "data", "belonging", "andrews_40_subjects", "AllNLPData.csv"
)

_FILENAME_PATTERN = re.compile(
    r"^eeg_(?P<person>.+)_(?P<question>\d+)_(?P<timestamp>\d+)(?:_scalograms)?\.tfrecord$"
)
_ALWAYS_SKIP_PERSON_IDS = set()


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


def _resolve_question_mode(dataset_params):
    raw_mode = dataset_params.get('question_mode', 'cognitive')
    if raw_mode is None:
        raw_mode = 'cognitive'

    normalized = normalize_token(raw_mode)

    cognitive_aliases = {'cognitive', 'congnitive', 'congitive'}
    non_cognitive_aliases = {'noncognitive', 'noncongnitive', 'noncongitive'}

    if normalized in cognitive_aliases:
        return 'cognitive'
    if normalized in non_cognitive_aliases:
        return 'non_cognitive'

    raise ValueError(
        "Invalid question mode. Set dataset_params.question_mode to "
        "'cognitive' or 'non cognitive'."
    )


def _question_in_range(question_num, question_mode):
    if question_mode == 'cognitive':
        return 33 <= question_num < 39
    return 1 <= question_num < 33


def _coerce_question_numbers(raw_value, setting_name):
    if raw_value is None:
        return set()
    if isinstance(raw_value, (list, tuple, set)):
        values = list(raw_value)
    else:
        values = [raw_value]

    resolved = set()
    for value in values:
        try:
            question_num = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{setting_name} must contain only positive integers.") from exc
        if question_num <= 0:
            raise ValueError(f"{setting_name} must contain only positive integers.")
        resolved.add(question_num)
    return resolved


def _resolve_included_questions(dataset_params):
    return _coerce_question_numbers(
        dataset_params.get('included_questions'),
        'dataset_params.included_questions',
    )


def _resolve_uncapped_questions(dataset_params):
    return _coerce_question_numbers(
        dataset_params.get('uncapped_questions'),
        'dataset_params.uncapped_questions',
    )


def _question_selected(question_num, question_mode, included_questions):
    return question_num in included_questions or _question_in_range(question_num, question_mode)


def _resolve_effective_max_windows(question_num, max_windows_per_question, uncapped_questions):
    if question_num in uncapped_questions:
        return None
    return max_windows_per_question


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


def _resolve_label_source(dataset_params):
    labels_csv = dataset_params.get('labels_csv')
    if not labels_csv:
        return None

    normalized = str(labels_csv).strip().lower()
    normalized = normalized.replace('-', '').replace('_', '').replace(' ', '')

    mode_to_label_col = {
        'specific': 'SpecificLabel',
        'composite': 'CompositeLabel',
        'factor': 'FactorLabel',
        'weighted': 'WeightedDiffLabel',
    }
    if normalized in mode_to_label_col:
        label_modes_csv = dataset_params.get('labels_lookup_csv', _DEFAULT_ALL_NLP_CSV)
        return {
            'csv_path': label_modes_csv,
            'label_col': mode_to_label_col[normalized],
            'id_candidates': ('id', 'student_id', 'FileName', 'filename'),
            'mode': normalized,
        }

    raise ValueError(
        f"Unsupported labels_csv '{labels_csv}'. "
        "Use one of: specific, composite, factor, weighted. "
        f"Labels are loaded from {dataset_params.get('labels_lookup_csv', _DEFAULT_ALL_NLP_CSV)}."
    )


def _resolve_max_windows_per_question(dataset_params):
    raw_value = dataset_params.get('max_windows_per_question')
    if raw_value is None:
        return None
    try:
        max_windows = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "dataset_params.max_windows_per_question must be a positive integer or null."
        ) from exc
    if max_windows <= 0:
        raise ValueError("dataset_params.max_windows_per_question must be greater than zero.")
    return max_windows


def _load_csv_labels_map(dataset_params):
    source = _resolve_label_source(dataset_params)
    if source is None:
        return None, None, None

    labels_csv = source['csv_path']
    label_col = source['label_col']
    id_candidates = source['id_candidates']

    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")

    labels_map = {}
    with open(labels_csv, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        id_col = next((col for col in id_candidates if col in fieldnames), None)
        if id_col is None:
            raise ValueError(
                f"Labels CSV must have one of these ID columns: {', '.join(id_candidates)}"
            )
        if label_col not in reader.fieldnames:
            raise ValueError(f"Labels CSV must have '{label_col}' column")

        for row in reader:
            person_id = str(row.get(id_col, '')).strip()
            raw_label = str(row.get(label_col, '')).strip()
            if not person_id:
                continue
            if raw_label == '':
                raise ValueError(
                    f"Missing label for student_id '{person_id}' in column '{label_col}'."
                )
            try:
                labels_map[person_id] = int(float(raw_label))
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric label '{raw_label}' for student_id '{person_id}' "
                    f"in column '{label_col}'."
                ) from exc

    if not labels_map:
        raise ValueError(f"No labels were loaded from {labels_csv}")

    source_text = f"{labels_csv}:{label_col}"
    if source.get('mode'):
        source_text = f"{source_text}:mode={source['mode']}"
    return labels_map, label_col, source_text


def _get_tensorflow():
    try:
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


def _load_tfrecord_records_tensorflow(tfrecord_path):
    tf = _get_tensorflow()
    if tf is None:
        return None
    records = []
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(bytes(raw_record.numpy()))
        record = {}
        for key, feature in example.features.feature.items():
            kind = feature.WhichOneof('kind')
            if kind == 'bytes_list':
                values = [bytes(value) for value in feature.bytes_list.value]
                record[key] = values[0] if len(values) == 1 else np.array(values, dtype=object)
            elif kind == 'float_list':
                record[key] = np.array(feature.float_list.value, dtype=np.float32)
            elif kind == 'int64_list':
                record[key] = np.array(feature.int64_list.value, dtype=np.int64)
        records.append(record)
    return records


def _load_tfrecord_records(tfrecord_path):
    records = None
    reader_available = False
    reader_error = None
    try:
        try:
            from tfrecord.tfrecord_loader import tfrecord_loader  # type: ignore
        except Exception:
            from tfrecord.reader import tfrecord_loader  # type: ignore
        reader_available = True
        records = list(tfrecord_loader(tfrecord_path, None, None))
        return records
    except Exception as exc:
        reader_error = exc
        records = None

    records_tf = _load_tfrecord_records_tensorflow(tfrecord_path)
    if records_tf is not None:
        return records_tf

    if records is not None:
        return records

    if reader_available and reader_error is not None:
        raise RuntimeError(f"Failed to read TFRecord '{tfrecord_path}': {reader_error}") from reader_error

    raise ImportError(
        "The 'tfrecord' package is not installed and TensorFlow is unavailable. "
        "Install with `pip install tfrecord` or install TensorFlow to read TFRecords."
    )


def _resolve_recursive_tfrecord_search(dataset_params):
    raw_value = dataset_params.get('tfrecords_recursive', False)
    if isinstance(raw_value, bool):
        return raw_value

    normalized = str(raw_value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off', ''}:
        return False

    raise ValueError(
        "dataset_params.tfrecords_recursive must be a boolean-like value."
    )


def _resolve_tfrecord_search_dirs(dataset_params):
    raw_dir = dataset_params.get('tfrecords_dir')
    if raw_dir is None:
        raise ValueError("dataset_params must include 'tfrecords_dir'")

    normalized = str(raw_dir).strip()
    if not normalized:
        raise ValueError("dataset_params.tfrecords_dir must be a non-empty path.")

    return (normalized,)


def load_belonging_tfrecords(dataset_params, metadata):
    """Load TFRecord scalograms and labels per participant session."""
    tfrecord_search_dirs = _resolve_tfrecord_search_dirs(dataset_params)
    recursive_search = _resolve_recursive_tfrecord_search(dataset_params)
    channels = dataset_params.get('channels', ['TP9', 'AF7', 'AF8', 'TP10'])
    question_mode = _resolve_question_mode(dataset_params)
    max_windows_per_question = _resolve_max_windows_per_question(dataset_params)
    included_questions = _resolve_included_questions(dataset_params)
    uncapped_questions = _resolve_uncapped_questions(dataset_params)
    csv_labels_map, label_col, label_source_text = _load_csv_labels_map(dataset_params)

    for search_dir in tfrecord_search_dirs:
        if not os.path.exists(search_dir):
            raise FileNotFoundError(f"TFRecords dir not found: {search_dir}")

    tfrecord_paths = []
    for search_dir in tfrecord_search_dirs:
        tfrecord_pattern = (
            os.path.join(search_dir, '**', '*.tfrecord')
            if recursive_search else os.path.join(search_dir, '*.tfrecord')
        )
        tfrecord_paths.extend(glob.glob(tfrecord_pattern, recursive=recursive_search))

    tfrecord_paths = sorted(dict.fromkeys(tfrecord_paths))
    if not tfrecord_paths:
        raise RuntimeError(
            "No TFRecord files found in the configured TFRecord search directories."
        )

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
    skipped_missing_label = []
    skipped_missing_people = set()
    skipped_forced_person_tfrecords = []
    skipped_forced_person_ids = set()
    skipped_unreadable = []
    trimmed_question_tfrecords = 0
    trimmed_windows_removed = 0

    image_size = None
    num_channels = None
    total_windows = 0

    for person_id, question_num, _timestamp, tfrecord_path in parsed_paths:
        if person_id in _ALWAYS_SKIP_PERSON_IDS:
            skipped_forced_person_tfrecords.append(tfrecord_path)
            skipped_forced_person_ids.add(person_id)
            continue

        if not _question_selected(question_num, question_mode, included_questions):
            skipped_question.append(tfrecord_path)
            continue

        try:
            records = _load_tfrecord_records(tfrecord_path)
        except Exception as exc:
            skipped_unreadable.append((tfrecord_path, str(exc)))
            continue
        if len(records) == 0:
            skipped_empty.append(tfrecord_path)
            continue
        if len(records) != 1:
            raise ValueError(
                f"Expected exactly 1 record in {tfrecord_path}, found {len(records)}."
            )
        record = records[0]

        shape_value = record.get('shape')
        scalogram_value = record.get('scalograms')
        if csv_labels_map is not None:
            if person_id not in csv_labels_map:
                skipped_missing_label.append(tfrecord_path)
                skipped_missing_people.add(person_id)
                continue
            label = csv_labels_map[person_id]
        else:
            if 'label' not in record:
                raise ValueError(
                    f"TFRecord '{tfrecord_path}' does not contain a 'label' field. "
                    "Provide dataset_params.labels_csv / labels_lookup_csv or export labels into the TFRecord."
                )
            label = _coerce_int(record.get('label'), 'label')

        shape = _decode_shape(shape_value, tfrecord_path)
        if shape.size != 4:
            raise ValueError(
                f"Expected shape with 4 dims (N, C, H, W) in {tfrecord_path}, got {shape}."
            )

        scalograms = _decode_scalograms(scalogram_value, shape, tfrecord_path)
        scalograms = np.asarray(scalograms, dtype=np.float32)

        n_windows, n_channels, height, width = scalograms.shape
        effective_max_windows = _resolve_effective_max_windows(
            question_num,
            max_windows_per_question,
            uncapped_questions,
        )
        if effective_max_windows is not None and n_windows > effective_max_windows:
            trimmed_question_tfrecords += 1
            trimmed_windows_removed += int(n_windows - effective_max_windows)
            scalograms = scalograms[:effective_max_windows]
            n_windows = int(scalograms.shape[0])
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
    if skipped_unreadable:
        logger.log('skipped_unreadable_tfrecords', len(skipped_unreadable))
        logger.log(
            'skipped_unreadable_tfrecord_sample',
            ' | '.join(
                f"{os.path.basename(path)} :: {error}" for path, error in skipped_unreadable[:3]
            ),
        )
    if skipped_empty:
        logger.log('skipped_empty_tfrecords', len(skipped_empty))
    if skipped_question:
        logger.log('skipped_question_tfrecords', len(skipped_question))
    if skipped_forced_person_tfrecords:
        logger.log('skipped_forced_person_tfrecords', len(skipped_forced_person_tfrecords))
        logger.log('skipped_forced_person_ids', len(skipped_forced_person_ids))
    if skipped_missing_label:
        logger.log('skipped_missing_label_tfrecords', len(skipped_missing_label))
        logger.log('skipped_missing_label_people', len(skipped_missing_people))
    if trimmed_question_tfrecords:
        logger.log('trimmed_question_tfrecords', trimmed_question_tfrecords)
        logger.log('trimmed_windows_removed', trimmed_windows_removed)
    if not person_to_scalograms:
        raise RuntimeError(
            "No non-empty TFRecord files found in the configured TFRecord search directories."
        )

    person_ids = sorted(person_to_scalograms.keys())
    scalograms_list = [
        np.concatenate(
            [np.asarray(segment, dtype=np.float32) for segment in person_to_scalograms[pid]],
            axis=0,
        )
        for pid in person_ids
    ]
    labels = [person_to_label[pid] for pid in person_ids]

    unique_labels, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique_labels, counts):
        logger.log(f'class_{lbl}_count', cnt)
    logger.log('total_windows', total_windows)
    logger.log('num_people', len(set(person_ids)))
    logger.log('question_mode', question_mode)
    logger.log('included_questions', sorted(included_questions))
    logger.log('uncapped_questions', sorted(uncapped_questions))
    logger.log('tfrecords_recursive', recursive_search)
    logger.log('tfrecord_search_dirs_count', len(tfrecord_search_dirs))
    if csv_labels_map is not None:
        logger.log('label_source', f"csv:{label_source_text}")
    else:
        logger.log('label_source', 'tfrecord_label_field')

    metadata.update({
        'num_people': len(set(person_ids)),
        'num_windows': total_windows,
        'channels': channels,
        'num_channels': num_channels or len(channels),
        'image_size': image_size,
        'num_classes': len(set(labels)),
        'question_mode': question_mode,
        'included_questions': sorted(included_questions),
        'uncapped_questions': sorted(uncapped_questions),
        'tfrecords_recursive': recursive_search,
        'tfrecord_search_dirs': list(tfrecord_search_dirs),
        'max_windows_per_question': max_windows_per_question,
        'label_col': label_col,
        'label_source': f"csv:{label_source_text}" if csv_labels_map is not None else 'tfrecord_label_field',
        'skipped_forced_person_tfrecords': len(skipped_forced_person_tfrecords),
        'skipped_forced_person_ids': len(skipped_forced_person_ids),
        'skipped_unreadable_tfrecords': len(skipped_unreadable),
        'skipped_missing_label_tfrecords': len(skipped_missing_label),
        'skipped_missing_label_people': len(skipped_missing_people),
        'trimmed_question_tfrecords': trimmed_question_tfrecords,
        'trimmed_windows_removed': trimmed_windows_removed,
    })

    X = {
        'scalograms': scalograms_list,
        'person_ids': person_ids,
    }
    y = np.array(labels)

    return X, y, metadata

