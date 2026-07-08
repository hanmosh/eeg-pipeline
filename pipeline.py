import argparse

from utils.lib_pipe import start_pipeline
from utils.pipeline_setup import DATA_MAP, MODEL_MAP, PREPROCESSOR_MAP, TRAINER_MAP


def build_parser():
    parser = argparse.ArgumentParser(description="Run EEG training pipeline")
    parser.add_argument("config_file", nargs="?", default="belonging_config_chrononet_tfrecord.json")
    parser.add_argument("-m", "--models", action="store_true", help="Save trained model artifacts")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    start_pipeline(
        args.config_file,
        DATA_MAP,
        PREPROCESSOR_MAP,
        MODEL_MAP,
        TRAINER_MAP,
        save_model=args.models,
    )


if __name__ == "__main__":
    main()
