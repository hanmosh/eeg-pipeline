LABEL_FOLDER_NAMES = {
    "specific": "Specific",
    "composite": "Composite",
    "factor": "Factor",
    "weighted": "Weighted",
}


def normalize_token(value):
    if value is None:
        return ""
    normalized = str(value).strip().lower()
    return normalized.replace("-", "").replace("_", "").replace(" ", "")


def sanitize_path_component(value, default=""):
    text = "" if value is None else str(value).strip()
    if not text:
        return default

    safe_name = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return safe_name or default


def canonical_label_folder_name(raw_label, default="Run"):
    normalized = normalize_token(raw_label)
    if normalized in LABEL_FOLDER_NAMES:
        return LABEL_FOLDER_NAMES[normalized]
    return sanitize_path_component(raw_label, default=default)
