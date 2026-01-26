import json
from architectures import *
from preprocessors import *
from trainers import *
from itertools import product
from copy import deepcopy
from utils.log import logger
from utils.log import model_tracker
from datetime import datetime



def verify_config(config):
    """
    Verify that the config dictionary contains all required keys.
    Raise ValueError if any required key is missing.

    Parameters:
        config (dict): Configuration dictionary to verify.

    Returns:
        None
    """
    required_keys = ["id", "dataset_params", "preprocessor_params", "model_params", "trainer_params"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    # Additional verification logic can be added here

def replace_nested_value(d, dotted_key, value):
    """
    Replace a value in a nested dictionary given a dotted key.

    Parameters:
        d (dict): The dictionary to modify.
        dotted_key (str): The dotted key representing the path to the value.
        value: The new value to set.

    Returns:
        None
    """
    keys = dotted_key.split('.')
    for k in keys[:-1]:
        d = d[k]  # assumes path exists
    d[keys[-1]] = value


def get_nested_value(d, dotted_key):
    """
    Retrieve a value from a nested dictionary given a dotted key.

    Parameters:
        d (dict): The dictionary to search.
        dotted_key (str): The dotted key representing the path to the value.
        
    Returns:
        The value found at the specified path.
    """
    keys = dotted_key.split('.')
    for k in keys:
        d = d[k]  # assumes path exists
    return d

def start_pipeline(config_file, data_map, preprocessor_map, model_map, trainer_map, save_model = False):
    #open and load config file
    with open('run_configs/' + config_file, 'r') as f:
        config = json.load(f)

    # Verify config structure for required keys
    verify_config(config)

    # Generate grid combinations if grid_keys are specified
    grid_keys = config.get("grid_keys", [])
    if grid_keys:
        grid_values = [get_nested_value(config, key) for key in grid_keys]
        grid_combinations = list(product(*grid_values))
    else:
        grid_combinations = [()]

    total_combos = len(grid_combinations)

    # Iterate over each combination in the grid
    for combo in grid_combinations:

        print("\n" + "="*50)
        print(f"Training configuration {len(grid_combinations) - total_combos + 1} of {len(grid_combinations)}")

        # Create a deep copy of the original config for modification
        config_copy = deepcopy(config)

        # Replace grid keys with current combination values
        for i, key in enumerate(grid_keys):
            replace_nested_value(config_copy, key, combo[i])

        print(f"Training with configuration: { {key: combo[i] for i, key in enumerate(grid_keys)} }")
        print("="*50 + "\n")

        dataset_params = config_copy["dataset_params"]
        preprocessor_params = config_copy["preprocessor_params"]
        model_params = config_copy["model_params"]
        trainer_params = config_copy["trainer_params"]

        # Log the current timestamp and configuration ID
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.log('timestamp', timestamp)
        logger.log('config_id', config_copy.get("id", "N/A"))

        # set the model name in the model tracker
        model_name = model_params.get("name", None)
        if model_name is None or model_name not in model_map:
            raise ValueError(f"Model '{model_name}' not found in model_map.")
        model_tracker.set_model_name(model_name, save_model)

        #set the config in the model tracker
        model_tracker.set_config(config_copy)

        # Data retrieval/generation
        dataset_retriever = dataset_params.get("name", None)
        if dataset_retriever is None or dataset_retriever not in data_map:
            raise ValueError(f"Data retriever '{dataset_retriever}' not found in data_map.")
        
        # data retriever:
        # params: dataset params from config, metadata (empty dict to start)
        # returns: X, y, metadata (with any additional info needed for preprocessing/modeling)
        X, y, metadata = data_map[dataset_retriever](dataset_params, metadata={})

        # Preprocessing
        preprocessor_name = preprocessor_params.get("name", None)
        if preprocessor_name is None or preprocessor_name not in preprocessor_map:
            raise ValueError(f"Preprocessor '{preprocessor_name}' not found in preprocessor_map.")
        
        # preprocessor:
        # params: preprocessor params from config, X, y, metadata
        # returns: data(a dictionary of split datasets or data loaders), metadata (with any additional info needed for modeling)
        # Using a dictionary to allow for multiple data splits or loaders (e.g. having seperate test/train/val sets or loaders)
        data, metadata = preprocessor_map[preprocessor_name](preprocessor_params, X, y, metadata)

        if isinstance(data, list):
            fold_data_list = data
        else:
            fold_data_list = [data]

        for fold_idx, fold_data in enumerate(fold_data_list, start=1):
            if len(fold_data_list) > 1:
                logger.log('cv_fold', fold_idx)
                logger.log('cv_folds', len(fold_data_list))

            # Model instantiation
            # model_name already retrieved above
            
            # model:
            # params: model params from config, metadata (if needed, such as input shape)
            # returns: instantiated model
            model = model_map[model_name](model_params, metadata)

            #set the model in the model tracker
            model_tracker.set_model(model)

            # Trainer instantiation and training
            trainer_name = trainer_params.get("name", None)
            if trainer_name is None or trainer_name not in trainer_map:
                raise ValueError(f"Trainer '{trainer_name}' not found in trainer_map.")
            
            # trainer:
            # params: trainer params from config, model, data (dictionary from preprocessor), metadata
            # NOTE: trainer should also handle validation procedure (along with dataloader if needed)
            trainer = trainer_map[trainer_name](trainer_params, model, fold_data, metadata)

            # trainer.train():
            # returns: trained model
            trained_model = trainer.run()

            # Finalize logging and save results
            model_tracker.set_model(trained_model)
            if save_model:
                logger.log('model_save_path', model_tracker.get_model_info_save_path())
                model_tracker.save_model_details()
            else:
                model_tracker.reset_tracker()

            # Save and reset logger for next run
            logger_filename = config_copy.get("log_filename", "default_log.csv")
            logger.save(logger_filename)

        total_combos -= 1


        


        
    
