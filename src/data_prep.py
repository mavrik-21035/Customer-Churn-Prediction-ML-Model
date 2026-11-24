"""Input cleaning, schema discovery, and leakage-safe preprocessing."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)
MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "?"}
POSITIVE_LABEL_HINTS = ("yes", "true", "1", "churn", "churned", "positive", "y")


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Read a CSV with a clear error when the supplied path is invalid."""
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {csv_path}")
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read CSV at {csv_path}: {exc}") from exc


def clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim text, normalize missing tokens, and parse numeric-looking strings."""
    if frame.empty:
        raise ValueError("The input dataset has no rows.")
    cleaned = frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    if len(set(cleaned.columns)) != len(cleaned.columns):
        raise ValueError("Column names must be unique after trimming whitespace.")

    for column in cleaned.columns:
        series = cleaned[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue
        text = series.astype("string").str.strip()
        text = text.mask(text.str.casefold().isin(MISSING_TOKENS), pd.NA)
        numeric_text = (
            text.str.replace(r"[$£€,\s]", "", regex=True)
            .str.replace("%", "", regex=False)
        )
        numeric_values = pd.to_numeric(numeric_text, errors="coerce")
        observed_count = int(text.notna().sum())
        numeric_count = int(numeric_values.notna().sum())
        if observed_count and numeric_count / observed_count >= 0.95:
            cleaned[column] = numeric_values
        else:
            cleaned[column] = text
    return cleaned


def detect_id_columns(frame: pd.DataFrame, target: str) -> list[str]:
    """Detect likely identifiers without dropping ordinary low-cardinality data."""
    id_columns: list[str] = []
    for column in frame.columns:
        if column == target:
            continue
        normalized = re.sub(r"[^a-z0-9]+", "", column.casefold())
        named_id = (
            normalized in {
                "id", "identifier", "customerid", "userid", "accountid",
                "subscriberid", "recordid", "rowid",
            }
            or bool(re.search(r"(?:^|[_\s-])(?:id|identifier)$", column, re.I))
            or bool(re.search(r"[a-z](?:ID|Identifier)$", column))
            or "accountnumber" in normalized
            or "customernumber" in normalized
        )
        values = frame[column].dropna()
        unique_ratio = values.nunique() / len(values) if len(values) else 0.0
        high_cardinality_text = (
            (pd.api.types.is_object_dtype(frame[column])
             or pd.api.types.is_string_dtype(frame[column]))
            and len(values) >= 20
            and unique_ratio >= 0.98
        )
        if named_id or high_cardinality_text:
            id_columns.append(column)
    return id_columns


def resolve_positive_label(
    target: pd.Series, requested_label: str | None = None
) -> str:
    """Resolve the positive class label for a binary target."""
    if target.isna().any():
        raise ValueError("The target column contains missing values.")
    labels = [str(value) for value in target.unique()]
    if len(labels) != 2:
        raise ValueError(
            f"Expected a binary target with exactly 2 classes; found {labels}."
        )
    if requested_label is not None:
        match = next(
            (
                label for label in labels
                if label.casefold() == requested_label.casefold()
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"Positive label {requested_label!r} is not present. "
                f"Available labels: {labels}."
            )
        return match
    for hint in POSITIVE_LABEL_HINTS:
        match = next(
            (label for label in labels if label.casefold() == hint), None
        )
        if match is not None:
            return match
    inferred = sorted(labels)[-1]
    LOGGER.warning(
        "Positive target label was not recognizable; using %r. "
        "Pass --positive-label to override.",
        inferred,
    )
    return inferred


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create modest, schema-conditional telecom features when source fields exist."""
    features = frame.copy()
    normalized_names = {
        re.sub(r"[^a-z0-9]+", "", column.casefold()): column
        for column in features.columns
    }
    tenure_column = normalized_names.get("tenure") or normalized_names.get(
        "tenuremonths"
    )
    if (
        tenure_column is not None
        and pd.api.types.is_numeric_dtype(features[tenure_column])
    ):
        tenure = features[tenure_column]
        features["tenure_band"] = pd.cut(
            tenure,
            bins=[-np.inf, 6, 12, 24, 48, np.inf],
            labels=[
                "0-6 months",
                "7-12 months",
                "13-24 months",
                "25-48 months",
                "49+ months",
            ],
        ).astype("string")
        features["is_new_customer"] = (tenure <= 6).astype("int8")

    monthly_column = normalized_names.get("monthlycharges") or normalized_names.get(
        "monthlycharge"
    )
    total_column = normalized_names.get("totalcharges") or normalized_names.get(
        "totalcharge"
    )
    if (
        tenure_column is not None
        and monthly_column is not None
        and total_column is not None
        and all(
            pd.api.types.is_numeric_dtype(features[column])
            for column in (tenure_column, monthly_column, total_column)
        )
    ):
        tenure = features[tenure_column].replace(0, np.nan)
        features["average_historical_monthly_charge"] = (
            features[total_column] / tenure
        )
    return features


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build imputers and encoders fitted only within each training fold."""
    numeric_columns = features.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                strategy="most_frequent", keep_empty_features=True
            )),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("No usable feature columns remain after excluding target/IDs.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model_frame(
    frame: pd.DataFrame,
    target: str,
    drop_columns: list[str],
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Select model features and align inference data to the training schema."""
    excluded = set(drop_columns) | {target}
    features = engineer_features(frame.drop(
        columns=[column for column in excluded if column in frame.columns]
    ))
    if expected_columns is not None:
        features = features.reindex(columns=expected_columns)
    if features.shape[1] == 0:
        raise ValueError("No model features remain after excluding target/ID columns.")
    return features

