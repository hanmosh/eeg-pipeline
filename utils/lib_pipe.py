import json
import os
from copy import deepcopy
from datetime import datetime

import numpy as np

from utils.log import logger, model_tracker


REQUIRED_KEYS = ["id", "dataset_params", "preprocessor_params", "model_params", "trainer_params"]


def verify_config(config):
    for key in REQUIRED_KEYS:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")


def run_config(
    config,
    data_map,
    preprocessor_map,
    model_map,
    trainer_map,
    save_model=False,
    log_filename_override=None,
    clear_logger=False,
):
    verify_config(config)
    config_copy = deepcopy(config)
    if log_filename_override:
        config_copy["log_filename"] = log_filename_override
    if clear_logger:
        logger.clear()

    dataset_params = config_copy["dataset_params"]
    preprocessor_params = config_copy["preprocessor_params"]
    model_params = config_copy["model_params"]
    trainer_params = config_copy["trainer_params"]

    logger.log('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.log('config_id', config_copy.get("id", "N/A"))

    model_name = model_params.get("name")
    if model_name not in model_map:
        raise ValueError(f"Model '{model_name}' not found in model_map.")
    model_tracker.set_model_name(model_name, save_model)
    model_tracker.set_config(config_copy)

    dataset_retriever = dataset_params.get("name")
    if dataset_retriever not in data_map:
        raise ValueError(f"Data retriever '{dataset_retriever}' not found in data_map.")
    X, y, metadata = data_map[dataset_retriever](dataset_params, metadata={})

    preprocessor_name = preprocessor_params.get("name")
    if preprocessor_name not in preprocessor_map:
        raise ValueError(f"Preprocessor '{preprocessor_name}' not found in preprocessor_map.")
    data, metadata = preprocessor_map[preprocessor_name](preprocessor_params, X, y, metadata)

    fold_data_list = data if isinstance(data, list) else [data]

    fold_metrics = []
    for fold_idx, fold_data in enumerate(fold_data_list, start=1):
        if len(fold_data_list) > 1:
            logger.log('cv_fold', fold_idx)
            logger.log('cv_folds', len(fold_data_list))

        model = model_map[model_name](model_params, metadata)
        model_tracker.set_model(model)

        trainer_name = trainer_params.get("name")
        if trainer_name not in trainer_map:
            raise ValueError(f"Trainer '{trainer_name}' not found in trainer_map.")
        trainer = trainer_map[trainer_name](trainer_params, model, fold_data, metadata)
        trained_model = trainer.run()

        model_tracker.set_model(trained_model)
        if save_model:
            logger.log('model_save_path', model_tracker.get_model_info_save_path())
            model_tracker.save_model_details()
        else:
            model_tracker.reset_tracker()

        if len(fold_data_list) > 1:
            split_name = None
            if fold_data.get('test_loader') is not None:
                split_name = 'test'
            elif fold_data.get('val_loader') is not None:
                split_name = 'val'
            if split_name:
                entry = logger.build_entry_dict()
                metrics = {}
                for metric in ('accuracy', 'precision', 'recall', 'f1', 'auc'):
                    key = f'{split_name}_{metric}'
                    if key in entry:
                        metrics[metric] = entry[key]
                if metrics:
                    fold_metrics.append((split_name, metrics))

    if fold_metrics:
        split_name = fold_metrics[0][0]
        print(f"\nCross-validation summary ({split_name}):")
        for metric in ('accuracy', 'precision', 'recall', 'f1', 'auc'):
            values = [m.get(metric) for _, m in fold_metrics if m.get(metric) is not None]
            if values:
                avg_value = float(np.mean(values))
                logger.log(f'cv_avg_{split_name}_{metric}', avg_value)
                print(f"Mean {metric}: {avg_value:.4f}")

    logger_filename = config_copy.get("log_filename", "default_log.csv")
    logger.save(logger_filename)
    return logger.build_entry_dict()


def start_pipeline(config_file, data_map, preprocessor_map, model_map, trainer_map, save_model=False):
    if not config_file.endswith('.json'):
        raise ValueError("config_file must be a .json file")
    if os.path.isabs(config_file) or os.path.exists(config_file):
        config_path = config_file
    else:
        config_path = os.path.join('run_configs', config_file)

    with open(config_path, 'r') as f:
        config = json.load(f)

    return run_config(
        config,
        data_map,
        preprocessor_map,
        model_map,
        trainer_map,
        save_model=save_model,
    )
