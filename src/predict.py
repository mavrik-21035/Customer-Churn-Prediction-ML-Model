"""Score new customer records using a previously saved model artifact."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_prep import clean_dataframe, make_model_frame, read_dataset

LOGGER = logging.getLogger(__name__)


def predict_frame(frame: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    """Return scores while tolerating absent columns and unseen categories."""
    cleaned = clean_dataframe(frame)
    feature_columns = model_bundle["feature_columns"]
    raw_features = make_model_frame(
        cleaned,
        target=model_bundle["target"],
        drop_columns=model_bundle["drop_columns"],
        expected_columns=feature_columns,
    )
    probabilities = model_bundle["pipeline"].predict_proba(raw_features)[:, 1]
    threshold = float(model_bundle["threshold"])
    positive_label = model_bundle["positive_label"]
    negative_label = model_bundle["negative_label"]
    prediction = np.where(probabilities >= threshold, positive_label, negative_label)
    ids = model_bundle["id_columns"]
    available_ids = [column for column in ids if column in cleaned.columns]
    if available_ids:
        output = cleaned[available_ids].copy()
    else:
        output = pd.DataFrame({"record_id": np.arange(len(cleaned))})
    output["churn_probability"] = probabilities
    output["predicted_churn"] = prediction
    output["risk_tier"] = pd.cut(
        probabilities,
        bins=[-np.inf, 0.33, 0.67, np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
    ).astype(str)
    return output


def score_frame(
    frame: pd.DataFrame, model_bundle: dict, output_path: str | Path
) -> pd.DataFrame:
    """Score records and write the predictions CSV."""
    output = predict_frame(frame, model_bundle)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    LOGGER.info("Saved %s scored records to %s", len(output), destination)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to model.joblib.")
    parser.add_argument("--data", required=True, help="CSV to score.")
    parser.add_argument("--output", default="outputs/scored_predictions.csv")
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    bundle = joblib.load(arguments.model)
    score_frame(read_dataset(arguments.data), bundle, arguments.output)


if __name__ == "__main__":
    main()
