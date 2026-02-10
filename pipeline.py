import argparse

from utils.lib_pipe import start_pipeline
from dataset_retrievers.spectrogram_retriever import load_belonging_spectrograms
from preprocessors.spectrogram_processor import spectrogram_preprocessor
from architectures.chrononet import ChronoNet
from trainers.belonging_trainer import BelongingTrainer

DATA_MAP = {
    "load_belonging_spectrograms": load_belonging_spectrograms,
}
PREPROCESSOR_MAP = {
    "spectrogram_preprocessor": spectrogram_preprocessor,
}
MODEL_MAP = {
    "ChronoNet": ChronoNet,
}
TRAINER_MAP = {
    "BelongingTrainer": BelongingTrainer,
}

parser = argparse.ArgumentParser(description="Run ChronoNet training pipeline")
parser.add_argument("config_file", nargs="?", default="belonging_config_chrononet.json")
parser.add_argument("-m", "--models", action="store_true", help="Save trained model artifacts")
args = parser.parse_args()

start_pipeline(args.config_file, DATA_MAP, PREPROCESSOR_MAP, MODEL_MAP, TRAINER_MAP, save_model=args.models)
