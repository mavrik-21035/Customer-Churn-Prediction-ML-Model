"""Cross-validated model comparison, selection, and threshold tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_prep import build_preprocessor

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Fitted model and held-out data needed for fair evaluation."""

    pipeline: Pipeline
    model_name: str
    threshold: float
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    comparison: pd.DataFrame
    searches: dict[str, GridSearchCV]


def _model_searches(
    features: pd.DataFrame, positive_weight: float
) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    preprocessor = build_preprocessor(features)
    shared = [("preprocessor", preprocessor)]
    return {
        "Logistic Regression": (
            Pipeline(shared + [("model", LogisticRegression(
                class_weight="balanced", max_iter=2000, solver="liblinear"
            ))]),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            Pipeline(shared + [("model", RandomForestClassifier(
                class_weight="balanced_subsample",
                n_estimators=250,
                random_state=42,
                n_jobs=1,
            ))]),
            {
                "model__max_depth": [None, 12],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "XGBoost": (
            Pipeline(shared + [("model", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=positive_weight,
                n_estimators=180,
                tree_method="hist",
                random_state=42,
                n_jobs=1,
            ))]),
            {
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
            },
        ),
        "LightGBM": (
            Pipeline(shared + [("model", LGBMClassifier(
                objective="binary",
                class_weight="balanced",
                n_estimators=180,
                verbosity=-1,
                random_state=42,
                n_jobs=1,
            ))]),
            {
                "model__num_leaves": [15, 31],
                "model__learning_rate": [0.05, 0.1],
            },
        ),
    }


def _best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], np.finfo(float).eps)
    )
    return float(thresholds[int(np.nanargmax(f1))])


def train_models(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    cv_folds: int = 3,
) -> TrainingResult:
    """Tune four classifiers using training-only CV; reserve a stratified test set."""
    if target.nunique() != 2:
        raise ValueError("Training requires a binary target encoded as 0/1.")
    if target.value_counts().min() < cv_folds + 1:
        raise ValueError(
            f"Each target class needs at least {cv_folds + 1} records "
            "for the requested stratified split/CV."
        )
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target.astype(int),
        test_size=test_size,
        random_state=42,
        stratify=target.astype(int),
    )
    minority_count = int(y_train.value_counts().min())
    actual_folds = min(cv_folds, minority_count)
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    positive_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    searches: dict[str, GridSearchCV] = {}

    for name, (pipeline, parameter_grid) in _model_searches(
        X_train, positive_weight
    ).items():
        LOGGER.info("Tuning %s", name)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        searches[name] = search

    rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    for name, search in searches.items():
        oof_probabilities = cross_val_predict(
            clone(search.best_estimator_),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        model_threshold = _best_f1_threshold(y_train, oof_probabilities)
        thresholds[name] = model_threshold
        test_probabilities = search.best_estimator_.predict_proba(X_test)[:, 1]
        test_predictions = (test_probabilities >= model_threshold).astype(int)
        true_negative, false_positive, false_negative, true_positive = (
            confusion_matrix(y_test, test_predictions, labels=[0, 1]).ravel()
        )
        rows.append({
            "model": name,
            "cv_roc_auc_mean": float(search.best_score_),
            "cv_roc_auc_std": float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
            "test_roc_auc": float(roc_auc_score(y_test, test_probabilities)),
            "test_pr_auc": float(average_precision_score(y_test, test_probabilities)),
            "test_precision": float(
                precision_score(y_test, test_predictions, zero_division=0)
            ),
            "test_recall": float(
                recall_score(y_test, test_predictions, zero_division=0)
            ),
            "test_f1": float(f1_score(y_test, test_predictions, zero_division=0)),
            "test_threshold": model_threshold,
            "test_true_negative": int(true_negative),
            "test_false_positive": int(false_positive),
            "test_false_negative": int(false_negative),
            "test_true_positive": int(true_positive),
            "best_parameters": str(search.best_params_),
        })
    comparison = pd.DataFrame(rows).sort_values(
        "cv_roc_auc_mean", ascending=False
    ).reset_index(drop=True)
    best_name = str(comparison.iloc[0]["model"])
    best_pipeline = searches[best_name].best_estimator_
    threshold = thresholds[best_name]
    LOGGER.info(
        "Selected %s (mean CV ROC-AUC %.4f); OOF F1 threshold %.4f",
        best_name,
        comparison.iloc[0]["cv_roc_auc_mean"],
        threshold,
    )
    return TrainingResult(
        pipeline=best_pipeline,
        model_name=best_name,
        threshold=threshold,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        comparison=comparison,
        searches=searches,
    )
