import argparse
import os
from utils.lib_pipe import start_pipeline

from dataset_retrievers.spectrogram_retriever import load_belonging_spectrograms
from preprocessors.spectrogram_processor import spectrogram_preprocessor
from architectures.belonging_architecture import BelongingCNN
from trainers.belonging_trainer import BelongingTrainer

#MAPS from string names to classes
#maps from data retrieval/generation name to function
DATA_MAP = {
     "load_belonging_spectrograms": load_belonging_spectrograms
}
# Maps from preprocessor name to preprocessor function
PREPROCESSOR_MAP = {
    "spectrogram_preprocessor": spectrogram_preprocessor
}
# Maps from model name to model class
MODEL_MAP = {
    "BelongingCNN": BelongingCNN,
}
# maps from trainer name to trainer class
TRAINER_MAP = {
    "BelongingTrainer": BelongingTrainer
}


#use argparse to get command line arguments
parser = argparse.ArgumentParser(description="Run grid search training pipeline")

# Optional positional arguments with defaults
parser.add_argument("config_file", nargs="?", default="noise_comp_config.json", help="Path to config JSON")
parser.add_argument("-m", "--models", action="store_true", help="Keep trained models and their relevant information") # not implemented
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")


args = parser.parse_args()

start_pipeline(args.config_file, DATA_MAP, PREPROCESSOR_MAP, MODEL_MAP, TRAINER_MAP, save_model = args.models)