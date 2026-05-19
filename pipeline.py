import argparse

from utils.lib_pipe import start_pipeline
from dataset_retrievers.tfrecord_retriever import load_belonging_tfrecords
from dataset_retrievers.tfrecord_multimodal_retriever import load_belonging_multimodal_tfrecords
from preprocessors.tfrecord_processor import tfrecord_preprocessor
from architectures.chrononet import ChronoNet
from trainers.belonging_trainer import BelongingTrainer
from trainers.belonging_multimodal_trainer import BelongingMultimodalTrainer

DATA_MAP = {
    "load_belonging_tfrecords": load_belonging_tfrecords,
    "load_belonging_multimodal_tfrecords": load_belonging_multimodal_tfrecords,
}
PREPROCESSOR_MAP = {
    "tfrecord_preprocessor": tfrecord_preprocessor,
}
MODEL_MAP = {
    "ChronoNet": ChronoNet,
}
TRAINER_MAP = {
    "BelongingTrainer": BelongingTrainer,
    "BelongingMultimodalTrainer": BelongingMultimodalTrainer,
}

parser = argparse.ArgumentParser(description="Run ChronoNet training pipeline")
parser.add_argument("config_file", nargs="?", default="belonging_config_chrononet_tfrecord.json")
parser.add_argument("-m", "--models", action="store_true", help="Save trained model artifacts")
args = parser.parse_args()

start_pipeline(args.config_file, DATA_MAP, PREPROCESSOR_MAP, MODEL_MAP, TRAINER_MAP, save_model=args.models)
