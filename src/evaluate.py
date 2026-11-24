"""Held-out performance evaluation and model explainability artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from src.train import TrainingResult

LOGGER = logging.getLogger(__name__)


def evaluate_model(
    result: TrainingResult, output_dir: str | Path
) -> dict[str, float]:
    """Evaluate the selected model once on the untouched test split."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    probability = result.pipeline.predict_proba(result.X_test)[:, 1]
    prediction = (probability >= result.threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(result.y_test, probability)),
        "pr_auc": float(average_precision_score(result.y_test, probability)),
        "precision": float(precision_score(result.y_test, prediction, zero_division=0)),
        "recall": float(recall_score(result.y_test, prediction, zero_division=0)),
        "f1": float(f1_score(result.y_test, prediction, zero_division=0)),
        "threshold": float(result.threshold),
        "test_rows": int(len(result.y_test)),
    }
    pd.DataFrame([metrics]).to_csv(destination / "test_metrics.csv", index=False)
    result.comparison.to_csv(destination / "model_comparison.csv", index=False)
    matrix = confusion_matrix(result.y_test, prediction, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["actual_retained", "actual_churn"],
        columns=["predicted_retained", "predicted_churn"],
    ).to_csv(destination / "confusion_matrix.csv")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    RocCurveDisplay.from_predictions(result.y_test, probability, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(result.y_test, probability, ax=axes[1])
    ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=["Retained", "Churn"]
    ).plot(ax=axes[2], colorbar=False)
    axes[2].set_title(f"Confusion matrix @ {result.threshold:.3f}")
    figure.tight_layout()
    figure.savefig(destination / "model_evaluation.png", dpi=150)
    plt.close(figure)
    return metrics


def explain_model(
    result: TrainingResult,
    output_dir: str | Path,
    customer_ids: pd.DataFrame | None = None,
    high_risk_count: int = 10,
) -> pd.DataFrame:
    """Try SHAP, with a model/permutation importance fallback."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    preprocessor = result.pipeline.named_steps["preprocessor"]
    estimator = result.pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    transformed_train = preprocessor.transform(result.X_train)
    transformed_test = preprocessor.transform(result.X_test)
    high_risk_values: np.ndarray | None = None
    method = "SHAP"

    try:
        import shap

        background_rows = min(100, transformed_train.shape[0])
        explanation_rows = min(200, transformed_test.shape[0])
        explainer = shap.Explainer(
            estimator,
            transformed_train[:background_rows],
            feature_names=feature_names,
        )
        shap_values = explainer(transformed_test[:explanation_rows])
        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            values = values[:, :, 1] if values.shape[-1] == 2 else values.mean(axis=-1)
        if values.ndim != 2 or values.shape[1] != len(feature_names):
            raise ValueError(f"Unexpected SHAP array shape: {values.shape}")
        importance = np.abs(values).mean(axis=0)
        probabilities = result.pipeline.predict_proba(result.X_test)[:, 1]
        high_risk_positions = np.argsort(probabilities)[::-1][
            :min(high_risk_count, len(probabilities))
        ]
        local_shap_values = explainer(transformed_test[high_risk_positions])
        high_risk_values = np.asarray(local_shap_values.values)
        if high_risk_values.ndim == 3:
            high_risk_values = (
                high_risk_values[:, :, 1]
                if high_risk_values.shape[-1] == 2
                else high_risk_values.mean(axis=-1)
            )
        if high_risk_values.ndim != 2:
            raise ValueError(
                f"Unexpected local SHAP array shape: {high_risk_values.shape}"
            )
        pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
            "method": "mean_absolute_shap",
        }).sort_values("importance", ascending=False).to_csv(
            destination / "feature_importance.csv", index=False
        )
        figure = plt.figure(figsize=(9, 6))
        shap.summary_plot(
            values,
            transformed_test[:explanation_rows],
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(destination / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close(figure)
    except Exception as exc:
        LOGGER.warning(
            "SHAP explainability unavailable (%s); using model/permutation "
            "importance instead.",
            exc,
        )
        method = "model_or_permutation_importance"
        if hasattr(estimator, "feature_importances_"):
            importance = np.asarray(estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            importance = np.abs(np.asarray(estimator.coef_)).mean(axis=0)
        else:
            permutation = permutation_importance(
                result.pipeline,
                result.X_test,
                result.y_test,
                scoring="roc_auc",
                n_repeats=5,
                random_state=42,
                n_jobs=1,
            )
            importance = permutation.importances_mean
            feature_names = result.X_test.columns.to_numpy()
        importance_frame = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
            "method": method,
        }).sort_values("importance", ascending=False)
        importance_frame.to_csv(destination / "feature_importance.csv", index=False)
        figure, axis = plt.subplots(figsize=(9, 6))
        top = importance_frame.head(20).sort_values("importance")
        axis.barh(top["feature"], top["importance"], color="#4472C4")
        axis.set(
            title="Global model feature importance",
            xlabel="Importance",
            ylabel="",
        )
        figure.tight_layout()
        figure.savefig(destination / "shap_summary.png", dpi=150)
        plt.close(figure)

    probabilities = result.pipeline.predict_proba(result.X_test)[:, 1]
    high_risk_positions = np.argsort(probabilities)[::-1][:high_risk_count]
    if customer_ids is None:
        customer_ids = pd.DataFrame({"record_id": result.X_test.index.to_numpy()})
    risk_identifiers = customer_ids.iloc[high_risk_positions].reset_index(drop=True)
    if high_risk_values is not None:
        rows: list[dict[str, Any]] = []
        for rank, (position, local_values) in enumerate(
            zip(high_risk_positions, high_risk_values), start=1
        ):
            top_features = np.argsort(np.abs(local_values))[::-1][:5]
            for feature_position in top_features:
                row = {
                    "risk_rank": rank,
                    "test_row_position": int(position),
                    "churn_probability": float(probabilities[position]),
                    "feature": str(feature_names[feature_position]),
                    "shap_value": float(local_values[feature_position]),
                }
                row.update(risk_identifiers.iloc[rank - 1].to_dict())
                rows.append(row)
        pd.DataFrame(rows).to_csv(
            destination / "high_risk_customer_explanations.csv", index=False
        )
    else:
        important = pd.read_csv(destination / "feature_importance.csv").head(10)
        high_risk = result.X_test.iloc[high_risk_positions].reset_index(drop=True)
        high_risk.insert(0, "churn_probability", probabilities[high_risk_positions])
        high_risk = pd.concat([risk_identifiers, high_risk], axis=1)
        high_risk.to_csv(destination / "high_risk_customer_profiles.csv", index=False)
        important.to_csv(destination / "global_drivers_for_high_risk.csv", index=False)
    return pd.read_csv(destination / "feature_importance.csv")


def save_model(
    result: TrainingResult,
    output_path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Persist the fitted preprocessing/model pipeline and scoring metadata."""
    payload = {
        "pipeline": result.pipeline,
        "model_name": result.model_name,
        "threshold": result.threshold,
        **metadata,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output)
    LOGGER.info("Saved model artifact to %s", output)
