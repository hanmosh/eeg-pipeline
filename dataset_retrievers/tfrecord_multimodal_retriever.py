import os

import numpy as np
import pandas as pd

from dataset_retrievers.tfrecord_retriever import load_belonging_tfrecords
from dataset_retrievers.raw_csv_retriever import load_belonging_raw_csvs
from utils.log import logger


LABEL_MODE_TO_COL = {
    "specific": "SpecificLabel",
    "composite": "CompositeLabel",
    "factor": "FactorLabel",
    "weighted": "WeightedDiffLabel",
}

NOTEBOOK_LABEL_COLUMNS = [
    "FileName",
    "SpecificLabel",
    "BelongingCompositeScore",
    "CompositeLabel",
    "BelongingFactorScore",
    "FactorLabel",
    "WeightedScore",
    "WeightedDiffLabel",
]

# Keep the weighted-label leakage drop aligned with the historical notebook logic.
NOTEBOOK_WEIGHTED_DROP_COLUMNS = [
    "preparedness",
    "anticipated_perf",
    "self_efficacy",
    "tool_comfort",
    "membership",
    "conf_prog",
]

BELONGING_LABEL_DROP_COLUMNS = [
    "membership",
    "accept_pos",
    "accept_neg_rev",
    "emotion",
    "trust_instr",
    "conf_prog",
]

LABEL_CONSTRUCTION_FEATURES = {
    # SpecificLabel is constructed from Q1, which is not present as an input feature here.
    "SpecificLabel": [],
    "CompositeLabel": BELONGING_LABEL_DROP_COLUMNS,
    "FactorLabel": BELONGING_LABEL_DROP_COLUMNS,
    "WeightedDiffLabel": NOTEBOOK_WEIGHTED_DROP_COLUMNS,
}

NLP_DEFAULT_PATHS = [
    os.path.join("data", "belonging", "andrews_40_subjects", "AllNLPData.csv"),
]


def _normalize_label_mode(value):
    if value is None:
        return ""
    normalized = str(value).strip().lower()
    return normalized.replace("-", "").replace("_", "").replace(" ", "")


def _resolve_nlp_csv_path(dataset_params):
    configured = dataset_params.get("nlp_csv")
    candidates = []
    if configured:
        candidates.append(configured)
    candidates.extend(NLP_DEFAULT_PATHS)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    candidate_text = ", ".join(candidates)
    raise FileNotFoundError(
        "Unable to find NLP feature CSV. Checked: "
        f"{candidate_text}. Set dataset_params.nlp_csv to an existing AllNLPData.csv."
    )


def _resolve_id_column(df):
    for col in ("FileName", "id", "student_id", "filename"):
        if col in df.columns:
            return col
    raise ValueError("NLP CSV must include one of: FileName, id, student_id, filename.")


def _resolve_label_col(dataset_params):
    raw = dataset_params.get("labels_csv", dataset_params.get("lables_csv"))
    normalized = _normalize_label_mode(raw)
    return LABEL_MODE_TO_COL.get(normalized)


def _resolve_active_survey_columns(all_columns, label_col):
    drop_cols = set(NOTEBOOK_LABEL_COLUMNS)
    drop_cols.update(LABEL_CONSTRUCTION_FEATURES.get(label_col, []))
    return [c for c in all_columns if c not in drop_cols]


def _resolve_eeg_loader(dataset_params):
    configured = dataset_params.get("eeg_loader_name", dataset_params.get("eeg_source"))
    if configured is not None:
        normalized = _normalize_label_mode(configured)
        if normalized in {"tfrecord", "scalogram", "scalograms"}:
            return load_belonging_tfrecords, "tfrecord"
        if normalized in {"raw", "rawcsv"}:
            return load_belonging_raw_csvs, "raw_csv"
        raise ValueError(
            f"Unsupported dataset_params.eeg_loader_name '{configured}'. "
            "Use 'tfrecord' or 'raw_csv'."
        )

    if dataset_params.get("raw_data_dir"):
        return load_belonging_raw_csvs, "raw_csv"
    return load_belonging_tfrecords, "tfrecord"


def _load_belonging_multimodal(dataset_params, metadata):
    eeg_loader, eeg_loader_name = _resolve_eeg_loader(dataset_params)
    X, y, metadata = eeg_loader(dataset_params, metadata)
    person_ids = [str(pid) for pid in X["person_ids"]]
    eeg_key = "scalograms" if "scalograms" in X else "windows"
    eeg_windows = X[eeg_key]
    y = np.asarray(y, dtype=int)

    nlp_csv_path = _resolve_nlp_csv_path(dataset_params)
    nlp_df = pd.read_csv(nlp_csv_path)
    id_col = _resolve_id_column(nlp_df)

    nlp_df[id_col] = nlp_df[id_col].astype(str).str.strip()
    nlp_df = nlp_df[nlp_df[id_col] != ""].copy()
    nlp_df = nlp_df.drop_duplicates(subset=[id_col], keep="last")
    nlp_df = nlp_df.set_index(id_col)

    keep_indices = []
    missing_nlp_people = []
    for idx, pid in enumerate(person_ids):
        if pid in nlp_df.index:
            keep_indices.append(idx)
        else:
            missing_nlp_people.append(pid)

    if not keep_indices:
        raise RuntimeError(
            "No participant overlap between TFRecords and NLP CSV after ID alignment."
        )

    if missing_nlp_people:
        logger.log("skipped_missing_nlp_people", len(missing_nlp_people))
        person_ids = [person_ids[i] for i in keep_indices]
        eeg_windows = [eeg_windows[i] for i in keep_indices]
        y = y[keep_indices]

    label_col = _resolve_label_col(dataset_params)
    active_survey_cols = _resolve_active_survey_columns(nlp_df.columns, label_col)
    if not active_survey_cols:
        raise ValueError(
            "No usable NLP survey features found. "
            "Check AllNLPData.csv columns and leakage-drop configuration."
        )

    survey_frame = nlp_df.loc[person_ids, active_survey_cols].apply(pd.to_numeric, errors="coerce")

    survey_features = survey_frame.to_numpy(dtype=np.float32)
    survey_features_by_person = {
        pid: survey_features[idx]
        for idx, pid in enumerate(person_ids)
    }

    if label_col and label_col in nlp_df.columns:
        nlp_labels = pd.to_numeric(nlp_df.loc[person_ids, label_col], errors="coerce")
        if nlp_labels.isna().any():
            raise ValueError(f"Found non-numeric values in NLP label column '{label_col}'.")
        nlp_labels = nlp_labels.astype(int).to_numpy()
        mismatch_count = int(np.sum(nlp_labels != y))
        logger.log("nlp_label_mismatch_count", mismatch_count)
        if mismatch_count > 0:
            raise ValueError(
                f"Label mismatch between TFRecord labels and NLP column '{label_col}' "
                f"for {mismatch_count} participant(s)."
            )

    X = {
        eeg_key: eeg_windows,
        "person_ids": person_ids,
        "survey_features": survey_features,
        "survey_feature_names": active_survey_cols,
    }
    metadata.update(
        {
            "num_people": len(person_ids),
            "nlp_csv": nlp_csv_path,
            "survey_feature_names": active_survey_cols,
            "num_survey_features": len(active_survey_cols),
            "survey_features_by_person": survey_features_by_person,
            "survey_label_col": label_col,
            "eeg_loader_name": eeg_loader_name,
        }
    )
    logger.log("nlp_csv", nlp_csv_path)
    logger.log("num_survey_features", len(active_survey_cols))
    logger.log("survey_label_col", label_col)
    logger.log("eeg_loader_name", eeg_loader_name)

    return X, y, metadata


def load_belonging_multimodal(dataset_params, metadata):
    return _load_belonging_multimodal(dataset_params, metadata)


def load_belonging_multimodal_tfrecords(dataset_params, metadata):
    params = dict(dataset_params)
    params.setdefault("eeg_loader_name", "tfrecord")
    return _load_belonging_multimodal(params, metadata)


def load_belonging_multimodal_raw_csvs(dataset_params, metadata):
    params = dict(dataset_params)
    params.setdefault("eeg_loader_name", "raw_csv")
    return _load_belonging_multimodal(params, metadata)
