import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import json
from collections import OrderedDict

class Logger:
    def __init__(self):
        self.logs = [] # List of key-value pairs to log in a single row

    def log(self, key, value):
        """
        Add a key-value pair to the log entry.
        Parameters:
            entry (dict): Key-value pair to log.
        Returns:
            None
        """
        self.logs.append((key, value))

    def log_dict(self, entry_dict):
        """
        Add Multiple key-value pairs from a dictionary to the log entry.
        Parameters:
            entry_dict (dict): Dictionary of key-value pairs to log.
        Returns:
            None
        """
        for entry in entry_dict:
            self.logs.append((entry, entry_dict[entry]))

    def build_entry_dict(self):
        """
        Build the current log entry as an ordered dictionary.
        Parameters:
            None
        Returns:
            dict: The current log entry as a dictionary.
        """
        entry_dict = OrderedDict()
        for key, value in self.logs:
            entry_dict[key] = value
        return entry_dict

    def clear(self):
        """
        Clear the current log entry
        Parameters:
            None
        Returns:
            the log entry that was cleared
        """
        cleared_logs = self.logs
        self.logs = []
        return cleared_logs

    def save(self, filename):
        """
        Save the logged entry to a CSV file. If the file exists, append to it.
        Files are saved in the 'logs/' directory.
        Once saved, the internal log entry list is cleared.
        Parameters:
            filepname (str): name of the CSV file.
        Returns:
            None
        """
        filepath = f"logs/{filename}"
        df = pd.DataFrame(self.build_entry_dict(), index=[0])
        try:
            #check if logs/ exists, if not create it
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            existing_df = pd.read_csv(filepath)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(filepath, index=False)
        except FileNotFoundError:
            df.to_csv(filepath, index=False)

#instantiate a global logger
logger = Logger()

class ModelTracker:
    """
    Class for tracking model performance metrics over training epochs.
    When training and validation is completed, metrics can be retrieved, plotted, and/or saved.
    """
    def __init__(self):
        self.model = None
        self.model_name = None
        self.metrics = {}
        self.config = None
        self.filepath = None
        self.save_model = False

    def set_model_name(self, name, save_model=False):
        """
        Set the name of the model being tracked. And generates the filepath for saving model-related files (if save_model is True).
        Parameters:
            name (str): The name of the model.
            save_model (bool): Whether the model will be saved. If True, filepath is generated.
        Returns:
            None
        """
        self.model_name = name
        self.save_model = bool(save_model)
        if self.save_model:
            self.filepath = self.get_filepath()
        else:
            self.filepath = None


    def set_model(self, model):
        """
        Set the model being tracked.
        Parameters:
            model: The model instance to track.
        Returns:
            None
        """
        self.model = model

    def set_config(self, config):
        """
        Set the configuration used for training the model.
        Parameters:
            config (dict): The configuration dictionary.
        Returns:
            None
        """
        self.config = config

    def track_metric(self, metric_name, value):
        """
        Track a metric value for a given metric name.
        If metric name does exist, append the value to the list. Otherwise, create a new list.
        (when these are saved, single value metrics will be saved as a single value, multi-value metrics as a list)
        Parameters:
            metric_name (str): The name of the metric to track.
            value: The metric value to record.
        Returns:
            None
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def add_metric(self, metric_name, values):
        """
        Add multiple metric values for a given metric name.
        If metric name does exist, extend the list with the new values. Otherwise, create a new list.
        Parameters:
            metric_name (str): The name of the metric to add values for.
            values (list): The list of metric values to add.
        Returns:
            None
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].extend(values)

    def get_metrics(self):
        """
        Retrieve all tracked metrics.
        Parameters:
            None
        Returns:
            dict: A dictionary of all tracked metrics.
        """
        return self.metrics
    
    def get_metric(self, metric_name):
        """
        Retrieve tracked values for a specific metric.
        Parameters:
            metric_name (str): The name of the metric to retrieve.
        Returns:
            list: A list of tracked values for the specified metric.
        """
        return self.metrics.get(metric_name, [])
    
    def get_filepath(self):
        """
        Generate the filepath for saving model-related files. If the standard filepath has already been used, add a numeric suffix to avoid overwriting.
        Files are saved in 'models/{model_name}/' directory.
        Parameters:
            None
        Returns:
            str: The filepath for the model.
        """
        if self.model_name:
            base_filepath = f'models/{self.model_name}'
            filepath = base_filepath + '/'
            suffix = 1
            while os.path.exists(filepath):
                filepath = f"{base_filepath}_{suffix}/"
                suffix += 1
            os.makedirs(filepath, exist_ok=True)
            return filepath
        else:
            raise ValueError("Model name not set. Cannot generate filepath.")
    
    def get_model_info_save_path(self):
        """
        Get the full path where the model details would be saved.
        Parameters:
            None
        Returns:
            str: The full path to save the model.
        """
        if self.model and self.filepath:
            return self.filepath
        else:
            raise ValueError("Model or filepath not set. Cannot get model save path.")
    
    def plot_metric(self, metric_name, x_range = None, x_label='Epochs', y_label=None):
        """
        Plot the tracked values for a specific metric over some time value (e.g. epochs, seconds).
        Saved as a PNG file in the models/model_name directory.
        A model name must be set to save the plot.
        Parameters:
            metric_name (str): The name of the metric to plot.
            x_range (list, optional): The x-axis values corresponding to the metric values. Defaults to None (uses index).
            x_label (str, optional): Label for the x-axis. Defaults to 'Epochs'.
            y_label (str, optional): Label for the y-axis. Defaults to None (uses metric name).
        Returns:
            None
        """

        # First check if model details are to be saved
        if not self.save_model:
            print("Model saving not enabled. Skipping plot saving.")
            return

        # Proceed with plotting
        y_values = self.get_metric(metric_name)
        if not y_values:
            print(f"No data to plot for metric: {metric_name}")
            return
        
        if x_range is None:
            x_range = list(range(1, len(y_values) + 1))
        
        plt.figure()
        plt.plot(x_range, y_values, marker='o')
        plt.title(f'{metric_name} over {x_label}')
        plt.xlabel(x_label)
        plt.ylabel(y_label if y_label else metric_name)
        plt.grid()
        if self.filepath:
            plt.savefig(f'{self.filepath}{metric_name}_plot.png')
        else:
            raise ValueError("Model Name not set. Cannot save plot.")
        plt.close()

    def plot_metrics(self, metric_names, x_range = None, x_label='Epochs', y_label=None):
        """
        Plot multiple tracked metrics over some time value (e.g. epochs, seconds).
        Each metric is plotted on the same graph.
        Saved as a PNG file in the models/model_name directory.
        A model name must be set to save the plot.
        Parameters:
            metric_names (list): List of metric names to plot.
            x_range (tuple, optional): The range (min, max) of x-axis values. Corresponding x-values to the metric are calculated based on this range. Defaults to None (uses index).
            x_label (str, optional): Label for the x-axis. Defaults to 'Epochs'.
            y_label (str, optional): Label for the y-axis. Defaults to None (uses metric names).
        Returns:
            None
        """
        # First check if model details are to be saved
        if not self.save_model:
            print("Model saving not enabled. Skipping plot saving.")
            return
        
        plt.figure()
        for metric_name in metric_names:
            y_values = self.get_metric(metric_name)
            if not y_values:
                print(f"No data to plot for metric: {metric_name}")
                continue
            
            if x_range is None:
                x_vals = list(range(1, len(y_values) + 1))
            else:
                x_start, x_end = x_range
                x_vals = list(np.linspace(x_start, x_end, num=len(y_values)))
                
            
            plt.plot(x_vals, y_values, marker='o', label=metric_name)
        
        plt.title(f'Metrics over {x_label}')
        plt.xlabel(x_label)
        plt.ylabel(y_label if y_label else 'Metrics')
        plt.legend()
        plt.grid()
        plot_name = '_'.join(metric_names) + '_plot.png'
        if self.filepath:
            plt.savefig(f'{self.filepath}{plot_name}')
        else:
            raise ValueError("Model Name not set. Cannot save plot.")
        plt.close()

    def reset_tracker(self):
        """
        Reset the model tracker by clearing the model, model name, metrics, config, and filepath.
        Parameters:
            None
        Returns:
            None
        """
        self.model = None
        self.model_name = None
        self.metrics = {}
        self.config = None
        self.filepath = None
        self.save_model = False

    def save_model_details(self):
        """
        Save the model details and tracked metrics to a text file along with the configuration used for training.
        A model name must be set to save the details.
        Parameters:
            filename (str, optional): The name of the file to save the details. Defaults to 'model_details.txt'.
        Returns:
            None
        """
        # First check if model details are to be saved
        if not self.save_model: #this condition should never be satisfied (unless changes were made)
            print("Model saving not enabled. Skipping saving of details.")
            return

        if not self.model:
            print("No model set to save details for.")
            return
        if not self.filepath:
            raise ValueError("Model Name not set. Cannot save model details.")
        
        model_name = self.model_name if self.model_name else "model"
        filepath = f'{self.filepath}model_details.txt'
        with open(filepath, 'w') as f:
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Model Details:\n{str(self.model)}\n\n")
            f.write("Configuration:\n")
            if self.config:
                # dictionary is nested, so save as json string
                config_str = json.dumps(self.config, indent=4)
                f.write(config_str + "\n\n")
            f.write("\nTracked Metrics:\n")
            for metric_name, values in self.metrics.items():
                f.write(f"{metric_name}: {values if len(values) > 1 else values[0]}\n")

            #write a timestamp
            f.write(f"\nSaved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        #save the config as a json file
        if self.config:
            config_filepath = f'{self.filepath}config.json'
            with open(config_filepath, 'w') as f:
                json.dump(self.config, f, indent=4)


        # clear tracked metrics after saving
        self.reset_tracker()


    
#instantiate a global model tracker
model_tracker = ModelTracker()

    
