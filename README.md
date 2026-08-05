# Machine Learning Pipeline for Applied ML in Scientific Research

### Author: Hannah Moshtaghi | B.S. + M.S. California Polytechnic State University, San Luis Obispo

This repository contains a machine learning pipeline designed for applied machine learning tasks in scientific research. The pipeline includes data collection, preprocessing, model training, evaluation, and logging components to facilitate reproducible and efficient ML workflows.

This pipeline offers a skeleton structure that can be customized for various scientific datasets and ML models, maintaining modularity and scalability.

Adapted from Colin J. DuHamel's [pipeline](https://github.com/cjduhamel/pipeline) 

## Usage

#### Requirements
- Python 3.7+
- Required libraries: numpy, pandas, matplotlib
- Optional Libraries (to run the examples): scikit-learn, pytorch

### Basics
This repository contains several key files/directories:
- pipeline.py: Main pipeline script to run the ML workflow.
- utils/: Utility functions for pipeline, logging, and tracking operations 
    - lib_pipe.py: Core pipeline functions.
    - lib_logger.py: Logging and tracking utilities.

To run the pipeline, execute the following command in your terminal:
```bash
python pipeline.py config_file results_file [-m] [-v]
```
- `config_file`: Filename for the configuration JSON file (default: `noise_comp_config.json`). Must be located in the `run_configs/` directory.
- `-m`: Optional flag to save trained models and their relevant information. (Default: False)
- `-v`: Optional flag to enable verbose output during execution. (Default: False)

To run the saved multimodal suites, use one of:
```bash
python run_multimodal_full_128_suite.py
python run_multimodal_full_256_suite.py
python run_multimodal_full_512_suite.py
```
You can also add tags to those commands to do things like saving the models. 

These suites utilize a labeling method that was derived from the student responses.

### Configuration
- `id`: run name.
- `log_filename`: CSV log file.
- `seed`, `deterministic`: reproducibility settings.
- `model_artifact_family`: saved-model folder label.
- `label_sweep.{enabled,label_types}`: runs the label set in one config.
- `seed_sweep.{enabled,seeds}`: optional repeated runs over multiple seeds.
- `dataset_params`: loader name, label source, survey CSV path, EEG source/path settings, question filters, channel list, window sizes, and raw-signal filter settings.
- `preprocessor_params`: split/CV settings, batch size, train downsampling/window caps, seed, and optional train-only `specaugment` / `gaussian_noise` blocks.
- `model_params`: model name plus convolution, hidden-size, dropout, batch-norm, and sequence-window settings.
- `trainer_params`: trainer name plus epochs, learning rate, weight decay, patience, decision threshold, validation use, and multimodal fusion settings.

Names must match the maps in `utils/pipeline_setup.py`.

### Experiment Information

The models were trained using EEG and survey data collected while participants completed a computing-belonging questionnaire. The survey implementation is available in the [RscaSurvey repository](https://github.com/meoragiusiano/RscaSurvey), and the processed features and labels are available in [`AllNLPData.csv`](https://github.com/hanmosh/eeg-pipeline/blob/main/data/belonging/andrews_40_subjects/AllNLPData.csv).

Four binary label strategies were evaluated:

- **Specific:** Based on the original Q1 membership item. Responses 5--7 were labeled as belonging, while responses 1--4 were labeled as not belonging.
- **Composite:** Mean of `membership` (Q1--Q3), `accept_pos` (Q4--Q7), reverse-coded `accept_neg_rev` (Q8--Q11), `emotion` (Q12--Q16), and `trust_instr` (Q17--Q22). Scores were split at the median.
- **Factor:** One-factor analysis of the same five features used for the Composite label. Factor scores greater than or equal to zero were labeled as belonging.
- **Weighted:** Weighted combination of `membership` (Q1--Q3), `conf_prog` (Q22), `anticipated_perf` (Q24), `self_efficacy` (Q25), `tool_comfort` (Q27), and `preparedness` (Q30). Feature weights were based on the absolute mean difference between the Specific-label classes, and scores were split at the median.

Features used to construct each label were excluded from that label's survey-model inputs to reduce label leakage.

## Building out the Pipeline
The pipeline is designed to be modular, and it is up to you to implement the specific components. You will need to create your own data loaders, preprocessors, models, and trainers. The pipeline expects these components to adhere to specific interfaces, which you can find in the example implementations provided. Explanations of these components are as follows:

- **Data Retrievers**: Responsible for retrieving and loading datasets based on the provided parameters. Must be a function that has the following input/output structure:
    - Input: 
        - `dataset_params`: Dictionary of parameters for dataset loading. (Comes from config file)
        - `metadata`: An empty dictionary that can be populated with additional information during data loading to be used later in the pipeline.
    - Output:
        - `X`: Features dataset.
        - `y`: Target labels.
        - `metadata`: Updated metadata dictionary.

- **Preprocessors**: Handle data preprocessing tasks such as normalization, feature extraction, and data augmentation. Must be a function that has the following input/output structure:
    - Input:
        - `preprocessor_params`: Dictionary of parameters for data preprocessing. (Comes from config file)
        - `X`: Features dataset.
        - `y`: Target labels.
        - `metadata`: Metadata dictionary from data loading.
    - Output:
        - `data`: A dictionary containing preprocessed data, can contain any representation of the data (multiple splits, dataloaders, etc) but must be prepared for use by the trainer.
        - `metadata`: Updated metadata dictionary.

- **Model Architectures**: Define the machine learning models to be used for training and evaluation. Must be a class that has the following input structure:
    - Instance Initialization:
        - `model_params`: Dictionary of parameters for the model. (Comes from config file)
        - `metadata`: Metadata dictionary from preprocessing.

- **Trainers**: Manage the training and evaluation of the models. Must be a class that has the following input structure:
    - Instance Initialization:
        - `trainer_params`: Dictionary of parameters for model training. (Comes from config file)
        - `model`: An instance of the model architecture.
        - `data`: Dictionary of preprocessed data from the preprocessor.
        - `metadata`: Metadata dictionary from preprocessing.
    - Required Methods:
        - `run()`: Method to run the model training/evaluation process. Returns the trained model.

## Using the Grid Search Functionality
The pipeline supports grid search for hyperparameter tuning. To utilize this feature, pass an array of values for any parameter in the configuration file. Second, specify a key `grid_keys` in the configuration file, which is an array of strings representing the parameter names to include in the grid search using dot notation for nested parameters (e.g., `trainer_params.learning_rate`). The pipeline will automatically generate all combinations of the specified parameters and execute the training process for each combination.

For example usages, refer to the example configuration files in the `run_configs/` directory.

## Logging and Model Tracking
The pipeline includes two important logging and tracking features, both implemented in `utils/log.py`. 

There are two main features. Examples of their usage can be found in the examples provided. An explanation of each feature is as follows:
### CSV Logger:

A lightweight CSV logger that can record the reults of each training run, along with relevant hyperparameters and metrics

To use the logger, simply import the global logger instance via `from utils.log import logger`. The logger has the following methods to be used within your components:

- `logger.log(key, value)`: Logs a key-value pair to the current run. (Will show up in the csv file as a column-value entry)
- `logger.log_dict(dict)`: Logs all key-value pairs in the provided dictionary to the current run.

The logged information will be saved to a CSV file specified in the configuration file under the key `log_filename`. The logger will automatically log a timestamp, the ID from the config file, and (if the model is being saved) the path to the saved model.

Examples of the logger's usage can be found in the example implementations provided in the repository.

### Model Tracker:

A model tracking utility that saves trained models along with their relevant metadata and hyperparameters for future reference. This, unlike the CSV logger, is designed to save training and evaluation information, such as training curves, model architecture details, and performance metrics. 

To use the traker, simply import the global tracker instance via `from utils.log import model_tracker`. The tracker has the following methods to be used within your components:

- `model_tracker.track_metric(metric_name, value)`: Tracks a metric value for a given metric name. If the metric name exists, the value will be appended to the list of values for that metric, otherwise, a new entry will be created with a new list.
- `model_tracker.add_metrics(metric_name, values)`: Adds a list of values to the specified metric name. If the metric name exists, the values will be extended to the existing list, otherwise, a new entry will be created with the provided list.
- `model_tracker.get_metrics()`: Returns the dictionary of all tracked metrics.
- `model_tracker.get_metric(metric_name)`: Returns the list of values for the specified metric name.
- `plot_metric(metric_name, x_range=None, x_label=None, y_label=None)`: Plots the specified metric over the tracked values. Optionally, you can provide an x_range (list of x values), x_label, and y_label for the plot. If x_range is not provided, the x values will be the indices of the tracked values. The plot is saved in the model's traacking directory (generated dynamically).
- `model_tracker.plot_metrics(metric_names, x_range=None, x_label='Epochs', y_label=None)`: Plots multiple specified metrics on the same plot. Optionally, you can provide an x_range (list of x values), x_label, and y_label for the plot. If x_range is not provided, the x values will be the indices of the tracked values. The plot is saved in the model's traacking directory (generated dynamically).

If model saving is enabled (via the `-m` flag when running the pipeline), the trained model will be saved along with its tracked metrics, configuration parameters, and any generated plots. The models are saved to a dynamically generated directory name based on the model architecture name.

NOTE: If model saving is not enabled, the model tracker will still function, but tracked metrics and plots will not be saved. Please ensure relevant information is logged using the CSV logger for future reference. Tracked metrics can be accessed during runtime via the model_tracker instance.


## Further Reference and Future Updates

To understand the implementation details and to see updated documentation, please refer to the code comments and docstrings within the repository. The README will be updated to the best of my ability, but the code itself will always have the most accurate and up-to-date information.
