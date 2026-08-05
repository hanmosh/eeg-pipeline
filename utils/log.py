import json
import os
import pickle
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Logger:
    def __init__(self):
        self.logs = []

    def log(self, key, value):
        self.logs.append((key, value))

    def log_dict(self, entry_dict):
        for entry, value in entry_dict.items():
            self.logs.append((entry, value))

    def build_entry_dict(self):
        entry_dict = OrderedDict()
        for key, value in self.logs:
            entry_dict[key] = value
        return entry_dict

    def clear(self):
        cleared_logs = self.logs
        self.logs = []
        return cleared_logs

    def _resolve_log_path(self, filename):
        if os.path.isabs(filename):
            return filename
        if os.path.dirname(filename):
            return filename
        return os.path.join("logs", filename)

    def _coerce_log_value(self, value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple, set, dict)):
            return json.dumps(value, default=str)
        return value

    def save(self, filename):
        filepath = self._resolve_log_path(filename)
        entry_dict = {
            key: self._coerce_log_value(value)
            for key, value in self.build_entry_dict().items()
        }
        df = pd.DataFrame([entry_dict])
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            combined_df = pd.concat([existing_df, df], ignore_index=True, sort=False)
            combined_df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)


logger = Logger()


class ModelTracker:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.metrics = {}
        self.config = None
        self.filepath = None
        self.save_model = False
        self.auxiliary_artifacts = {}

    def set_model_name(self, name, save_model=False):
        self.model_name = name
        self.save_model = bool(save_model)
        if self.save_model:
            self.filepath = self.get_filepath()
        else:
            self.filepath = None

    def set_model(self, model):
        self.model = model

    def set_config(self, config):
        self.config = config

    def set_auxiliary_artifact(self, name, artifact):
        self.auxiliary_artifacts[name] = artifact

    def track_metric(self, metric_name, value):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def add_metric(self, metric_name, values):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].extend(values)

    def add_metrics(self, metric_name, values):
        self.add_metric(metric_name, values)

    def get_metrics(self):
        return self.metrics

    def get_metric(self, metric_name):
        return self.metrics.get(metric_name, [])

    def get_filepath(self):
        if self.model_name:
            base_filepath = f"models/{self.model_name}"
            filepath = base_filepath + "/"
            suffix = 1
            while os.path.exists(filepath):
                filepath = f"{base_filepath}_{suffix}/"
                suffix += 1
            os.makedirs(filepath, exist_ok=True)
            return filepath
        raise ValueError("Model name not set. Cannot generate filepath.")

    def get_model_info_save_path(self):
        if self.model and self.filepath:
            return self.filepath
        raise ValueError("Model or filepath not set. Cannot get model save path.")

    def plot_metric(self, metric_name, x_range=None, x_label="Epochs", y_label=None):
        if not self.save_model:
            print("Model saving not enabled. Skipping plot saving.")
            return

        y_values = self.get_metric(metric_name)
        if not y_values:
            print(f"No data to plot for metric: {metric_name}")
            return

        if x_range is None:
            x_range = list(range(1, len(y_values) + 1))

        plt.figure()
        plt.plot(x_range, y_values, marker="o")
        plt.title(f"{metric_name} over {x_label}")
        plt.xlabel(x_label)
        plt.ylabel(y_label if y_label else metric_name)
        plt.grid()
        if self.filepath:
            plt.savefig(f"{self.filepath}{metric_name}_plot.png")
        else:
            raise ValueError("Model Name not set. Cannot save plot.")
        plt.close()

    def plot_metrics(self, metric_names, x_range=None, x_label="Epochs", y_label=None):
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

            plt.plot(x_vals, y_values, marker="o", label=metric_name)

        plt.title(f"Metrics over {x_label}")
        plt.xlabel(x_label)
        plt.ylabel(y_label if y_label else "Metrics")
        plt.legend()
        plt.grid()
        plot_name = "_".join(metric_names) + "_plot.png"
        if self.filepath:
            plt.savefig(f"{self.filepath}{plot_name}")
        else:
            raise ValueError("Model Name not set. Cannot save plot.")
        plt.close()

    def reset_tracker(self):
        self.model = None
        self.model_name = None
        self.metrics = {}
        self.config = None
        self.filepath = None
        self.save_model = False
        self.auxiliary_artifacts = {}

    def save_model_details(self):
        if not self.save_model:
            print("Model saving not enabled. Skipping saving of details.")
            return

        if not self.model:
            print("No model set to save details for.")
            return
        if not self.filepath:
            raise ValueError("Model Name not set. Cannot save model details.")

        model_name = self.model_name if self.model_name else "model"
        filepath = f"{self.filepath}model_details.txt"
        with open(filepath, "w") as f:
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Model Details:\n{str(self.model)}\n\n")
            f.write("Configuration:\n")
            if self.config:
                config_str = json.dumps(self.config, indent=4)
                f.write(config_str + "\n\n")
            f.write("\nTracked Metrics:\n")
            for metric_name, values in self.metrics.items():
                f.write(f"{metric_name}: {values if len(values) > 1 else values[0]}\n")
            f.write(f"\nSaved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if self.config:
            config_filepath = f"{self.filepath}config.json"
            with open(config_filepath, "w") as f:
                json.dump(self.config, f, indent=4)

        try:
            import torch

            if hasattr(self.model, "state_dict"):
                weights_filepath = f"{self.filepath}model_state_dict.pt"
                torch.save(self.model.state_dict(), weights_filepath)
        except Exception as exc:
            with open(filepath, "a") as f:
                f.write(f"\nmodel_save_warning: {exc}\n")

        if self.metrics:
            metrics_filepath = f"{self.filepath}metrics.json"
            serializable_metrics = {
                metric_name: values if len(values) > 1 else values[0]
                for metric_name, values in self.metrics.items()
            }
            with open(metrics_filepath, "w") as f:
                json.dump(serializable_metrics, f, indent=4)

        for artifact_name, artifact in self.auxiliary_artifacts.items():
            artifact_path = f"{self.filepath}{artifact_name}.pkl"
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact, f)

        self.reset_tracker()


model_tracker = ModelTracker()
