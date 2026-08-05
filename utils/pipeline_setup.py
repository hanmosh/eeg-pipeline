import json
import os

from architectures.chrononet import ChronoNet
from architectures.cnn_gru import CNNGRU
from architectures.cnn_lstm import CNNLSTM
from architectures.raw_cnn_gru import RawCNNGRU
from architectures.raw_cnn_lstm import RawCNNLSTM
from architectures.raw_chrononet import RawChronoNet
from dataset_retrievers.raw_csv_retriever import load_belonging_raw_csvs
from dataset_retrievers.tfrecord_multimodal_retriever import (
    load_belonging_multimodal_raw_csvs,
    load_belonging_multimodal_tfrecords,
)
from dataset_retrievers.tfrecord_retriever import load_belonging_tfrecords
from preprocessors.tfrecord_processor import tfrecord_preprocessor
from trainers.belonging_multimodal_trainer import BelongingMultimodalTrainer
from trainers.belonging_trainer import BelongingTrainer


DATA_MAP = {
    "load_belonging_tfrecords": load_belonging_tfrecords,
    "load_belonging_raw_csvs": load_belonging_raw_csvs,
    "load_belonging_multimodal_raw_csvs": load_belonging_multimodal_raw_csvs,
    "load_belonging_multimodal_tfrecords": load_belonging_multimodal_tfrecords,
}

PREPROCESSOR_MAP = {
    "tfrecord_preprocessor": tfrecord_preprocessor,
}

MODEL_MAP = {
    "ChronoNet": ChronoNet,
    "CNNGRU": CNNGRU,
    "CNNLSTM": CNNLSTM,
    "RawCNNGRU": RawCNNGRU,
    "RawCNNLSTM": RawCNNLSTM,
    "RawChronoNet": RawChronoNet,
}

TRAINER_MAP = {
    "BelongingTrainer": BelongingTrainer,
    "BelongingMultimodalTrainer": BelongingMultimodalTrainer,
}


def set_by_path(config, path, value):
    parts = path.split(".")
    if not parts:
        raise ValueError("Empty parameter path.")

    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            raise KeyError(f"Invalid path segment '{part}' in '{path}'.")
        cursor = cursor[part]
    cursor[parts[-1]] = value


def resolve_config_path(config_file):
    if not config_file.endswith(".json"):
        raise ValueError("config_file must be a .json file")
    if os.path.isabs(config_file) or os.path.exists(config_file):
        return config_file
    return os.path.join("run_configs", config_file)


def load_json_config(config_file):
    config_path = resolve_config_path(config_file)
    with open(config_path, "r") as f:
        return json.load(f), config_path
