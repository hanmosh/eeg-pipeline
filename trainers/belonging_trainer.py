import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, roc_auc_score, confusion_matrix

from utils.log import logger, model_tracker


class BelongingTrainer:
    def __init__(self, trainer_params, model, data, metadata):
        self.trainer_params = trainer_params
        self.model = model
        self.data = data
        self.metadata = metadata
        self.decision_threshold = trainer_params.get('decision_threshold', 0.5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def _compute_loss_and_stats(self, outputs, labels, lengths, criterion):
        if outputs.dim() == 3:
            batch_size, seq_len, num_classes = outputs.shape
            logits = outputs.view(batch_size * seq_len, num_classes)
            labels_flat = labels.view(batch_size * seq_len)
            if lengths is not None:
                mask = (torch.arange(seq_len, device=labels.device)
                        .unsqueeze(0) < lengths.unsqueeze(1))
                mask_flat = mask.view(-1)
                loss_vec = criterion(logits, labels_flat)
                loss = loss_vec[mask_flat].mean()
                logits = logits[mask_flat]
                labels_use = labels_flat[mask_flat]
            else:
                loss = criterion(logits, labels_flat).mean()
                labels_use = labels_flat
        else:
            loss = criterion(outputs, labels).mean()
            logits = outputs
            labels_use = labels

        probs = torch.softmax(logits, dim=1)
        if probs.shape[1] == 2 and self.decision_threshold is not None:
            preds = (probs[:, 1] >= self.decision_threshold).long()
        else:
            preds = torch.argmax(logits, dim=1)
        return loss, preds, probs, labels_use

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                inputs, labels, lengths = batch
                return inputs, labels, lengths, None
            if len(batch) == 4:
                inputs, labels, lengths, person_ids = batch
                return inputs, labels, lengths, person_ids
        raise ValueError("Expected batch to be (inputs, labels, lengths[, person_ids]).")

    def _aggregate_participant_probs(self, probs, labels, person_ids):
        person_prob_sums = {}
        person_counts = {}
        person_labels = {}

        for pid, prob, label in zip(person_ids, probs, labels):
            pid = str(pid)
            if pid not in person_prob_sums:
                person_prob_sums[pid] = prob.astype(float)
                person_counts[pid] = 1
                person_labels[pid] = int(label)
            else:
                person_prob_sums[pid] += prob
                person_counts[pid] += 1
                if person_labels[pid] != int(label):
                    print(
                        f"Warning: inconsistent labels for participant {pid} "
                        f"({person_labels[pid]} vs {int(label)})."
                    )

        agg_pids = sorted(person_prob_sums.keys())
        agg_probs = np.vstack([person_prob_sums[pid] / person_counts[pid] for pid in agg_pids])
        agg_labels = np.array([person_labels[pid] for pid in agg_pids], dtype=int)
        return agg_labels, agg_probs, agg_pids

    def _print_participant_table(self, split_name, pids, labels, probs):
        if probs.ndim != 2 or probs.shape[0] == 0:
            return
        preds = np.argmax(probs, axis=1)
        if probs.shape[1] == 2:
            prob_vals = probs[:, 1]
            prob_header = "P(class=1)"
        else:
            prob_vals = probs.max(axis=1)
            prob_header = "P(pred)"

        print(f"\n{split_name.capitalize()} Participant Results:")
        print(f"{'participant':>12}  {'label':>5}  {'pred':>5}  {prob_header:>10}")
        for pid, label, pred, prob in zip(pids, labels, preds, prob_vals):
            print(f"{pid:>12}  {label:>5d}  {pred:>5d}  {prob:>10.4f}")
    def _compute_prf(self, labels, preds):
        labels_arr = np.array(labels, dtype=int)
        preds_arr = np.array(preds, dtype=int)
        if labels_arr.size == 0:
            return 0.0, 0.0, 0.0, 'binary'
        unique_labels = np.unique(labels_arr)
        average = 'binary' if len(unique_labels) == 2 else 'macro'
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_arr, preds_arr, average=average, zero_division=0
        )
        return precision, recall, f1, average

    def _predict_from_probs(self, probs):
        if probs.ndim == 2 and probs.shape[1] == 2 and self.decision_threshold is not None:
            return (probs[:, 1] >= self.decision_threshold).astype(int)
        return np.argmax(probs, axis=1)

    def _plot_roc_curve(self, labels, probs, split_name='test'):
        if not model_tracker.save_model:
            print("Model saving not enabled. Skipping ROC curve plot.")
            return
        if model_tracker.filepath is None:
            print("Model filepath not set. Skipping ROC curve plot.")
            return

        labels = np.array(labels, dtype=int)
        probs = np.array(probs, dtype=float)
        if labels.size == 0 or probs.size == 0:
            print(f"{split_name.capitalize()} ROC Curve: no samples, skipping plot.")
            return
        if len(np.unique(labels)) < 2:
            print(f"{split_name.capitalize()} ROC Curve: only one class present, skipping plot.")
            return

        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{split_name.capitalize()} ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{model_tracker.filepath}{split_name}_roc_curve.png')
        plt.close()

    def run(self):
        trained_model = self.train()
        if self.data['test_loader'] is not None:
            self.evaluate(self.data['test_loader'], split_name='test')
        elif self.data['val_loader'] is not None:
            # For CV runs (no held-out test), report validation confusion matrix.
            self.evaluate(self.data['val_loader'], split_name='val')

        if model_tracker.save_model:
            model_tracker.plot_metrics(["train_loss", "val_loss"])
            model_tracker.plot_metrics(["train_accuracy", "val_accuracy"])

        return trained_model

    def train(self):
        logger.log_dict(self.trainer_params)

        num_epochs = self.trainer_params.get('num_epochs', 50)
        learning_rate = self.trainer_params.get('learning_rate', 1e-3)
        weight_decay = self.trainer_params.get('weight_decay', 1e-4)
        patience = self.trainer_params.get('patience', 10)

        train_labels = np.array(self.data['train_loader'].dataset.labels, dtype=int)
        class_counts = np.bincount(train_labels)
        class_counts = np.where(class_counts == 0, 1, class_counts)
        class_weights = class_counts.sum() / (len(class_counts) * class_counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=self.device),
            reduction='none'
        )
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

        train_loader = self.data['train_loader']
        val_loader = self.data['val_loader']

        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            self.model.train()
            train_losses = []
            train_preds = []
            train_labels = []
            train_probs = []
            train_person_ids = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                inputs, labels, lengths, person_ids = self._unpack_batch(batch)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs.to(self.device), lengths=lengths)
                loss, preds, _probs, labels_use = self._compute_loss_and_stats(
                    outputs, labels, lengths, criterion
                )

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_preds.extend(preds.detach().cpu().numpy())
                train_labels.extend(labels_use.detach().cpu().numpy())
                train_probs.extend(_probs.detach().cpu().numpy())
                if person_ids is not None:
                    train_person_ids.extend(list(person_ids))

            avg_train_loss = np.mean(train_losses)
            train_labels = np.array(train_labels, dtype=int)
            train_preds = np.array(train_preds, dtype=int)
            train_probs = np.array(train_probs, dtype=float)
            use_participant_train = len(train_person_ids) == len(train_labels) and len(train_person_ids) > 0
            if use_participant_train:
                participant_labels, participant_probs, _ = self._aggregate_participant_probs(
                    train_probs, train_labels, train_person_ids
                )
                participant_preds = self._predict_from_probs(participant_probs)
                train_labels_for_metrics = participant_labels
                train_preds_for_metrics = participant_preds
                train_metric_level = 'participant'
            else:
                train_labels_for_metrics = train_labels
                train_preds_for_metrics = train_preds
                train_metric_level = 'sequence'

            train_accuracy = accuracy_score(train_labels_for_metrics, train_preds_for_metrics)
            train_precision, train_recall, train_f1, train_avg = self._compute_prf(
                train_labels_for_metrics, train_preds_for_metrics
            )

            model_tracker.track_metric('train_loss', avg_train_loss)
            model_tracker.track_metric('train_accuracy', train_accuracy)
            model_tracker.track_metric('train_precision', train_precision)
            model_tracker.track_metric('train_recall', train_recall)
            model_tracker.track_metric('train_f1', train_f1)

            if val_loader is not None:
                val_loss, val_accuracy, val_precision, val_recall, val_f1, val_avg, val_metric_level = self.validate(
                    val_loader, criterion
                )
                model_tracker.track_metric('val_loss', val_loss)
                model_tracker.track_metric('val_accuracy', val_accuracy)
                model_tracker.track_metric('val_precision', val_precision)
                model_tracker.track_metric('val_recall', val_recall)
                model_tracker.track_metric('val_f1', val_f1)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                print(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc ({train_metric_level}): {train_accuracy:.4f}, "
                    f"Train P/R/F1 ({train_avg}): {train_precision:.4f}/{train_recall:.4f}/{train_f1:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc ({val_metric_level}): {val_accuracy:.4f}, "
                    f"Val P/R/F1 ({val_avg}, {val_metric_level}): {val_precision:.4f}/{val_recall:.4f}/{val_f1:.4f}"
                )

                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                print(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc ({train_metric_level}): {train_accuracy:.4f}, "
                    f"Train P/R/F1 ({train_avg}): {train_precision:.4f}/{train_recall:.4f}/{train_f1:.4f}"
                )

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Restored best model from validation")

        logger.log('final_train_loss', avg_train_loss)
        logger.log('final_train_accuracy', train_accuracy)
        if val_loader is not None:
            logger.log('best_val_loss', best_val_loss)

        return self.model

    def validate(self, val_loader, criterion):
        self.model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        val_probs = []
        val_person_ids = []

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels, lengths, person_ids = self._unpack_batch(batch)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                outputs = self.model(inputs.to(self.device), lengths=lengths)
                loss, preds, _probs, labels_use = self._compute_loss_and_stats(
                    outputs, labels, lengths, criterion
                )

                val_losses.append(loss.item())
                val_preds.extend(preds.detach().cpu().numpy())
                val_labels.extend(labels_use.detach().cpu().numpy())
                val_probs.extend(_probs.detach().cpu().numpy())
                if person_ids is not None:
                    val_person_ids.extend(list(person_ids))

        avg_val_loss = np.mean(val_losses)
        val_labels = np.array(val_labels, dtype=int)
        val_preds = np.array(val_preds, dtype=int)
        val_probs = np.array(val_probs, dtype=float)

        use_participant_val = len(val_person_ids) == len(val_labels) and len(val_person_ids) > 0
        if use_participant_val:
            participant_labels, participant_probs, _ = self._aggregate_participant_probs(
                val_probs, val_labels, val_person_ids
            )
            participant_preds = self._predict_from_probs(participant_probs)
            val_labels_for_metrics = participant_labels
            val_preds_for_metrics = participant_preds
            val_metric_level = 'participant'
        else:
            val_labels_for_metrics = val_labels
            val_preds_for_metrics = val_preds
            val_metric_level = 'sequence'

        val_accuracy = accuracy_score(val_labels_for_metrics, val_preds_for_metrics)
        val_precision, val_recall, val_f1, val_avg = self._compute_prf(
            val_labels_for_metrics, val_preds_for_metrics
        )
        return avg_val_loss, val_accuracy, val_precision, val_recall, val_f1, val_avg, val_metric_level

    def evaluate(self, test_loader, split_name='test'):
        self.model.eval()
        test_preds = []
        test_labels = []
        test_probs = []
        test_person_ids = []

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels, lengths, person_ids = self._unpack_batch(batch)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                outputs = self.model(inputs.to(self.device), lengths=lengths)
                _loss, preds, probs, labels_use = self._compute_loss_and_stats(
                    outputs, labels, lengths, nn.CrossEntropyLoss(reduction='none')
                )

                test_preds.extend(preds.detach().cpu().numpy())
                test_labels.extend(labels_use.detach().cpu().numpy())
                test_probs.extend(probs.detach().cpu().numpy())
                if person_ids is not None:
                    test_person_ids.extend(list(person_ids))

        test_labels = np.array(test_labels, dtype=int)
        test_preds = np.array(test_preds, dtype=int)
        test_probs = np.array(test_probs, dtype=float)

        use_participant_level = len(test_person_ids) == len(test_labels) and len(test_person_ids) > 0
        if use_participant_level:
            participant_labels, participant_probs, participant_ids = self._aggregate_participant_probs(
                test_probs, test_labels, test_person_ids
            )
            participant_preds = self._predict_from_probs(participant_probs)
            accuracy = accuracy_score(participant_labels, participant_preds)
            precision, recall, f1, _avg = self._compute_prf(participant_labels, participant_preds)
            cm = confusion_matrix(participant_labels, participant_preds)
            auc = None
            if participant_probs.shape[1] == 2 and len(np.unique(participant_labels)) == 2:
                auc = roc_auc_score(participant_labels, participant_probs[:, 1])
        else:
            accuracy = accuracy_score(test_labels, test_preds)
            precision, recall, f1, _avg = self._compute_prf(test_labels, test_preds)
            cm = confusion_matrix(test_labels, test_preds)
            auc = None
            if test_probs.ndim == 2 and test_probs.shape[1] == 2 and len(np.unique(test_labels)) == 2:
                auc = roc_auc_score(test_labels, test_probs[:, 1])

        logger.log(f'{split_name}_accuracy', accuracy)
        logger.log(f'{split_name}_precision', precision)
        logger.log(f'{split_name}_recall', recall)
        logger.log(f'{split_name}_f1', f1)
        if auc is not None:
            logger.log(f'{split_name}_auc', auc)

        print(f"\n{split_name.capitalize()} Set Results:")
        if use_participant_level:
            print("Metrics computed at participant level.")
        else:
            print("Metrics computed at sequence level (participant IDs unavailable).")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if auc is not None:
            print(f"AUC: {auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)

        if use_participant_level and participant_probs.shape[1] == 2:
            self._plot_roc_curve(participant_labels, participant_probs[:, 1], split_name=split_name)
        elif (not use_participant_level) and test_probs.ndim == 2 and test_probs.shape[1] == 2:
            self._plot_roc_curve(test_labels, test_probs[:, 1], split_name=split_name)
        if use_participant_level:
            self._print_participant_table(split_name, participant_ids, participant_labels, participant_probs)

        return self.model
