import csv
import os

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trainers.belonging_trainer import BelongingTrainer
from utils.log import logger, model_tracker


class _ConstantSurveyModel:
    def __init__(self, positive_probability):
        self.positive_probability = float(positive_probability)

    def predict_proba(self, X):
        p1 = np.full((len(X),), self.positive_probability, dtype=float)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


class BelongingMultimodalTrainer(BelongingTrainer):
    def __init__(self, trainer_params, model, data, metadata):
        super().__init__(trainer_params, model, data, metadata)

        self.survey_weight = float(trainer_params.get("survey_weight", 0.7))
        if self.survey_weight < 0.0 or self.survey_weight > 1.0:
            raise ValueError("trainer_params.survey_weight must be between 0 and 1.")

        self.fusion_threshold = float(trainer_params.get("fusion_threshold", 0.5))
        if self.fusion_threshold < 0.0 or self.fusion_threshold > 1.0:
            raise ValueError("trainer_params.fusion_threshold must be between 0 and 1.")

        fusion_mode = str(trainer_params.get("fusion_mode", "fixed_weight")).strip().lower()
        if fusion_mode in {"fixed", "weighted"}:
            fusion_mode = "fixed_weight"
        if fusion_mode not in {"fixed_weight", "gated"}:
            raise ValueError("trainer_params.fusion_mode must be 'fixed_weight' or 'gated'.")
        self.fusion_mode = fusion_mode

        self.gated_confidence_margin = float(trainer_params.get("gated_confidence_margin", 0.0))
        if self.gated_confidence_margin < 0.0:
            raise ValueError("trainer_params.gated_confidence_margin must be non-negative.")

        # Keep threshold behavior fixed to match pallavi_recreation.py.
        self.decision_threshold = self.fusion_threshold
        self.threshold_strategy = None
        self.tuned_threshold = None

        self.survey_seed = int(trainer_params.get("survey_seed", metadata.get("seed", 42) or 42))
        self.survey_features_by_person = metadata.get("survey_features_by_person")
        if not self.survey_features_by_person:
            raise ValueError(
                "Missing survey_features_by_person in metadata. "
                "Use load_belonging_multimodal_tfrecords with this trainer."
            )
        self.survey_model = None

    def _save_participant_predictions(
        self,
        split_name,
        participant_ids,
        labels,
        eeg_probs,
        survey_probs,
        fused_scores,
        eeg_preds,
        survey_preds,
        fused_preds,
        fusion_sources,
    ):
        if len(participant_ids) == 0:
            return

        context = logger.build_entry_dict()
        filepath = os.path.join("logs", "belonging_multimodal_participant_predictions_v2.csv")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        fieldnames = [
            "timestamp",
            "config_id",
            "cv_fold",
            "cv_folds",
            "cv_fold_in_repeat",
            "cv_repeat",
            "cv_total_folds",
            "split_name",
            "question_mode",
            "label_source",
            "survey_label_col",
            "eeg_loader_name",
            "fusion_mode",
            "survey_weight",
            "fusion_threshold",
            "participant",
            "label",
            "eeg_prob",
            "eeg_pred",
            "survey_prob",
            "survey_pred",
            "fusion_prob",
            "fusion_pred",
            "fusion_decision_source",
            "eeg_correct",
            "survey_correct",
            "fusion_correct",
            "eeg_rescues_survey",
            "fusion_rescues_survey",
            "fusion_hurts_survey",
        ]

        file_exists = os.path.exists(filepath)
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for pid, label, eeg_prob, survey_prob, fusion_prob, eeg_pred, survey_pred, fusion_pred, fusion_source in zip(
                participant_ids,
                labels,
                eeg_probs[:, 1],
                survey_probs[:, 1],
                fused_scores,
                eeg_preds,
                survey_preds,
                fused_preds,
                fusion_sources,
            ):
                eeg_correct = int(eeg_pred == label)
                survey_correct = int(survey_pred == label)
                fusion_correct = int(fusion_pred == label)
                writer.writerow(
                    {
                        "timestamp": context.get("timestamp"),
                        "config_id": context.get("config_id"),
                        "cv_fold": context.get("cv_fold"),
                        "cv_folds": context.get("cv_folds"),
                        "cv_fold_in_repeat": context.get("cv_fold_in_repeat"),
                        "cv_repeat": context.get("cv_repeat"),
                        "cv_total_folds": context.get("cv_total_folds"),
                        "split_name": split_name,
                        "question_mode": context.get("question_mode"),
                        "label_source": context.get("label_source"),
                        "survey_label_col": context.get("survey_label_col"),
                        "eeg_loader_name": context.get("eeg_loader_name"),
                        "fusion_mode": self.fusion_mode,
                        "survey_weight": self.survey_weight,
                        "fusion_threshold": self.fusion_threshold,
                        "participant": pid,
                        "label": int(label),
                        "eeg_prob": float(eeg_prob),
                        "eeg_pred": int(eeg_pred),
                        "survey_prob": float(survey_prob),
                        "survey_pred": int(survey_pred),
                        "fusion_prob": float(fusion_prob),
                        "fusion_pred": int(fusion_pred),
                        "fusion_decision_source": fusion_source,
                        "eeg_correct": eeg_correct,
                        "survey_correct": survey_correct,
                        "fusion_correct": fusion_correct,
                        "eeg_rescues_survey": int((not survey_correct) and eeg_correct),
                        "fusion_rescues_survey": int((not survey_correct) and fusion_correct),
                        "fusion_hurts_survey": int(survey_correct and (not fusion_correct)),
                    }
                )

    def _print_rescue_summary(self, labels, eeg_preds, survey_preds, fused_preds):
        survey_wrong = survey_preds != labels
        survey_mistakes = int(np.sum(survey_wrong))
        eeg_rescues = int(np.sum(survey_wrong & (eeg_preds == labels)))
        fusion_rescues = int(np.sum(survey_wrong & (fused_preds == labels)))
        fusion_hurts = int(np.sum((survey_preds == labels) & (fused_preds != labels)))

        print("\nSurvey Rescue Summary:")
        print(f"Survey mistakes: {survey_mistakes}")
        print(f"EEG alone rescues: {eeg_rescues}")
        print(f"Fusion rescues: {fusion_rescues}")
        print(f"Fusion hurts correct survey decisions: {fusion_hurts}")

    def _compute_fused_outputs(self, survey_probs, eeg_probs, survey_preds, eeg_preds):
        base_scores = (
            self.survey_weight * survey_probs[:, 1]
            + (1.0 - self.survey_weight) * eeg_probs[:, 1]
        )

        if self.fusion_mode == "fixed_weight":
            fused_scores = base_scores
            fusion_sources = np.full((len(base_scores),), "fixed_weight", dtype=object)
            return fused_scores, (fused_scores >= self.fusion_threshold).astype(int), fusion_sources

        fused_scores = base_scores.copy()
        fusion_sources = np.full((len(base_scores),), "blend", dtype=object)

        survey_conf = np.abs(survey_probs[:, 1] - self.fusion_threshold)
        eeg_conf = np.abs(eeg_probs[:, 1] - self.fusion_threshold)
        disagree = survey_preds != eeg_preds
        use_eeg = disagree & (eeg_conf >= (survey_conf + self.gated_confidence_margin))
        use_survey = disagree & ~use_eeg

        fused_scores[use_eeg] = eeg_probs[use_eeg, 1]
        fused_scores[use_survey] = survey_probs[use_survey, 1]
        fusion_sources[use_eeg] = "eeg"
        fusion_sources[use_survey] = "survey"

        fused_preds = (fused_scores >= self.fusion_threshold).astype(int)
        return fused_scores, fused_preds, fusion_sources

    def run(self):
        trained_model = self.train()
        self.survey_model = self._fit_survey_model()

        if self.data["test_loader"] is not None:
            self.evaluate(self.data["test_loader"], split_name="test")
        elif self.data["val_loader"] is not None:
            self.evaluate(self.data["val_loader"], split_name="val")

        if model_tracker.save_model:
            model_tracker.plot_metrics(["train_loss", "val_loss"])
            model_tracker.plot_metrics(["train_accuracy", "val_accuracy"])
        return trained_model

    def _upsample_minority(self, X, y):
        y = np.asarray(y, dtype=int)
        if len(y) == 0:
            return X, y
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) <= 1:
            return X, y

        rng = np.random.RandomState(self.survey_seed)
        max_count = int(np.max(counts))
        X_parts = [X]
        y_parts = [y]

        for cls, count in zip(classes, counts):
            if int(count) >= max_count:
                continue
            cls_idx = np.where(y == cls)[0]
            sample_idx = rng.choice(cls_idx, size=max_count - int(count), replace=True)
            X_parts.append(X[sample_idx])
            y_parts.append(y[sample_idx])

        return np.vstack(X_parts), np.concatenate(y_parts)

    def _fit_survey_model(self):
        train_dataset = self.data["train_loader"].dataset
        person_to_label = train_dataset.person_to_label
        train_person_ids = [str(pid) for pid in train_dataset.person_to_windows.keys()]
        if not train_person_ids:
            raise RuntimeError("No train participants available for survey model fitting.")

        missing = [pid for pid in train_person_ids if pid not in self.survey_features_by_person]
        if missing:
            raise ValueError(
                f"Missing NLP features for {len(missing)} train participant(s), e.g. {missing[0]}."
            )

        X_train = np.vstack([self.survey_features_by_person[pid] for pid in train_person_ids]).astype(np.float32)
        y_train = np.array([int(person_to_label[pid]) for pid in train_person_ids], dtype=int)
        unique_labels = np.unique(y_train)

        if len(unique_labels) < 2:
            constant_prob = float(unique_labels[0])
            logger.log("survey_model_type", "constant")
            logger.log("survey_train_people", len(train_person_ids))
            return _ConstantSurveyModel(constant_prob)

        X_train_up, y_train_up = self._upsample_minority(X_train, y_train)
        survey_model = Pipeline(
            [
                # Impute inside the fold to avoid train/test leakage from global statistics.
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=5000, random_state=self.survey_seed)),
            ]
        )
        survey_model.fit(X_train_up, y_train_up)
        logger.log("survey_model_type", "logreg")
        logger.log("survey_train_people", len(train_person_ids))
        logger.log("survey_train_people_upsampled", len(y_train_up))
        return survey_model

    def _collect_eeg_participant_outputs(self, data_loader):
        self.model.eval()
        all_labels = []
        all_probs = []
        all_person_ids = []

        with torch.no_grad():
            for batch in data_loader:
                inputs, labels, lengths, person_ids = self._unpack_batch(batch)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                outputs = self.model(inputs.to(self.device), lengths=lengths)
                _loss, _preds, probs, labels_use = self._compute_loss_and_stats(
                    outputs, labels, lengths, torch.nn.CrossEntropyLoss(reduction="none")
                )

                all_labels.extend(labels_use.detach().cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())
                if person_ids is not None:
                    all_person_ids.extend([str(pid) for pid in person_ids])

        all_labels = np.array(all_labels, dtype=int)
        all_probs = np.array(all_probs, dtype=float)
        if len(all_person_ids) != len(all_labels):
            raise ValueError("Participant IDs are required for multimodal evaluation.")

        participant_labels, participant_probs, participant_ids = self._aggregate_participant_probs(
            all_probs, all_labels, all_person_ids
        )
        return participant_labels, participant_probs, participant_ids

    def _get_survey_probs(self, participant_ids):
        missing = [pid for pid in participant_ids if pid not in self.survey_features_by_person]
        if missing:
            raise ValueError(
                f"Missing NLP features for {len(missing)} evaluation participant(s), e.g. {missing[0]}."
            )
        X_eval = np.vstack([self.survey_features_by_person[pid] for pid in participant_ids]).astype(np.float32)
        return np.asarray(self.survey_model.predict_proba(X_eval), dtype=float)

    def _compute_auc(self, labels, scores):
        labels = np.asarray(labels, dtype=int)
        scores = np.asarray(scores, dtype=float)
        if len(np.unique(labels)) < 2:
            return None
        return float(roc_auc_score(labels, scores))

    def _log_split_metrics(self, split_name, prefix, accuracy, precision, recall, f1, auc):
        key_prefix = f"{split_name}_{prefix}" if prefix else split_name
        logger.log(f"{key_prefix}_accuracy", accuracy)
        logger.log(f"{key_prefix}_precision", precision)
        logger.log(f"{key_prefix}_recall", recall)
        logger.log(f"{key_prefix}_f1", f1)
        if auc is not None:
            logger.log(f"{key_prefix}_auc", auc)

    def evaluate(self, data_loader, split_name="test"):
        if self.survey_model is None:
            self.survey_model = self._fit_survey_model()

        labels, eeg_probs, participant_ids = self._collect_eeg_participant_outputs(data_loader)
        survey_probs = self._get_survey_probs(participant_ids)

        logger.log(f"{split_name}_fusion_mode", self.fusion_mode)
        logger.log(f"{split_name}_gated_confidence_margin", self.gated_confidence_margin)

        eeg_preds = (eeg_probs[:, 1] >= self.fusion_threshold).astype(int)
        survey_preds = (survey_probs[:, 1] >= self.fusion_threshold).astype(int)

        fused_scores, fused_preds, fusion_sources = self._compute_fused_outputs(
            survey_probs, eeg_probs, survey_preds, eeg_preds
        )

        eeg_accuracy = accuracy_score(labels, eeg_preds)
        eeg_precision, eeg_recall, eeg_f1, _ = self._compute_prf(labels, eeg_preds)
        eeg_auc = self._compute_auc(labels, eeg_probs[:, 1])

        survey_accuracy = accuracy_score(labels, survey_preds)
        survey_precision, survey_recall, survey_f1, _ = self._compute_prf(labels, survey_preds)
        survey_auc = self._compute_auc(labels, survey_probs[:, 1])

        fusion_accuracy = accuracy_score(labels, fused_preds)
        fusion_precision, fusion_recall, fusion_f1, _ = self._compute_prf(labels, fused_preds)
        fusion_auc = self._compute_auc(labels, fused_scores)

        self._log_split_metrics(
            split_name, "fusion", fusion_accuracy, fusion_precision, fusion_recall, fusion_f1, fusion_auc
        )
        self._log_split_metrics(
            split_name, "eeg", eeg_accuracy, eeg_precision, eeg_recall, eeg_f1, eeg_auc
        )
        self._log_split_metrics(
            split_name, "survey", survey_accuracy, survey_precision, survey_recall, survey_f1, survey_auc
        )
        logger.log(f"{split_name}_threshold_used", self.fusion_threshold)
        logger.log(f"{split_name}_survey_weight", self.survey_weight)

        print(f"\n{split_name.capitalize()} Set Results (Multimodal):")
        print(f"Fusion mode: {self.fusion_mode}")
        print(f"Survey weight: {self.survey_weight:.3f}, EEG weight: {1.0 - self.survey_weight:.3f}")
        print(f"Threshold: {self.fusion_threshold:.3f}")
        if self.fusion_mode == "gated":
            print(f"Gated confidence margin: {self.gated_confidence_margin:.3f}")
            print(
                f"Fusion decisions -> blend: {int(np.sum(fusion_sources == 'blend'))}, "
                f"survey: {int(np.sum(fusion_sources == 'survey'))}, "
                f"eeg: {int(np.sum(fusion_sources == 'eeg'))}"
            )
        print(
            f"EEG      -> Acc: {eeg_accuracy:.4f}, P/R/F1: "
            f"{eeg_precision:.4f}/{eeg_recall:.4f}/{eeg_f1:.4f}"
            + (f", AUC: {eeg_auc:.4f}" if eeg_auc is not None else "")
        )
        print(
            f"Survey   -> Acc: {survey_accuracy:.4f}, P/R/F1: "
            f"{survey_precision:.4f}/{survey_recall:.4f}/{survey_f1:.4f}"
            + (f", AUC: {survey_auc:.4f}" if survey_auc is not None else "")
        )
        print(
            f"Fusion   -> Acc: {fusion_accuracy:.4f}, P/R/F1: "
            f"{fusion_precision:.4f}/{fusion_recall:.4f}/{fusion_f1:.4f}"
            + (f", AUC: {fusion_auc:.4f}" if fusion_auc is not None else "")
        )
        self._print_rescue_summary(labels, eeg_preds, survey_preds, fused_preds)
        print("\nFusion Confusion Matrix:")
        print(confusion_matrix(labels, fused_preds))

        self._save_participant_predictions(
            split_name,
            participant_ids,
            labels,
            eeg_probs,
            survey_probs,
            fused_scores,
            eeg_preds,
            survey_preds,
            fused_preds,
            fusion_sources,
        )

        if fusion_auc is not None:
            self._plot_roc_curve(labels, fused_scores, split_name=split_name)

        return self.model
