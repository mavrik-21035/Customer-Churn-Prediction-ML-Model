"""Focused tests for schema cleaning, ID detection, and inference alignment."""

import unittest

import numpy as np
import pandas as pd

from src.data_prep import (
    build_preprocessor,
    clean_dataframe,
    detect_id_columns,
    engineer_features,
    make_model_frame,
    resolve_positive_label,
)


class DataPreparationTests(unittest.TestCase):
    def test_numeric_text_and_whitespace_missing_values_are_cleaned(self) -> None:
        frame = pd.DataFrame({
            "amount": [" $1,200.50 ", "  ", "$42.00"],
            "category": [" yes ", "No", "N/A"],
        })

        cleaned = clean_dataframe(frame)

        self.assertAlmostEqual(cleaned.loc[0, "amount"], 1200.5)
        self.assertTrue(pd.isna(cleaned.loc[1, "amount"]))
        self.assertEqual(cleaned.loc[0, "category"], "yes")
        self.assertTrue(pd.isna(cleaned.loc[2, "category"]))

    def test_identifiers_and_positive_class_are_detected(self) -> None:
        frame = pd.DataFrame({
            "customerID": ["A", "B", "C"],
            "valid": ["yes", "no", "yes"],
            "Churn": ["No", "Yes", "No"],
        })

        self.assertEqual(detect_id_columns(frame, "Churn"), ["customerID"])
        self.assertEqual(resolve_positive_label(frame["Churn"]), "Yes")

    def test_feature_engineering_and_inference_schema_alignment(self) -> None:
        training = pd.DataFrame({
            "tenure": [0, 4, 20],
            "MonthlyCharges": [20.0, 40.0, 60.0],
            "TotalCharges": [0.0, 160.0, 1200.0],
            "Contract": ["month", "month", "year"],
        })
        engineered = engineer_features(training)
        self.assertIn("tenure_band", engineered.columns)
        self.assertTrue(
            np.isnan(engineered.loc[0, "average_historical_monthly_charge"])
        )

        preprocessor = build_preprocessor(engineered)
        preprocessor.fit(engineered)
        model_frame = make_model_frame(
            training.drop(columns=["Contract"]),
            target="Churn",
            drop_columns=[],
            expected_columns=engineered.columns.tolist(),
        )
        transformed = preprocessor.transform(model_frame)
        unseen_category = training.copy()
        unseen_category["Contract"] = "new-contract"
        unseen_frame = make_model_frame(
            unseen_category,
            target="Churn",
            drop_columns=[],
            expected_columns=engineered.columns.tolist(),
        )
        unseen_transformed = preprocessor.transform(unseen_frame)

        self.assertEqual(transformed.shape[0], len(training))
        self.assertEqual(
            transformed.shape[1], len(preprocessor.get_feature_names_out())
        )
        self.assertEqual(unseen_transformed.shape, transformed.shape)

    def test_numeric_target_positive_label_can_be_selected(self) -> None:
        target = pd.Series([0, 1, 0, 1])
        self.assertEqual(resolve_positive_label(target, "1"), "1")


if __name__ == "__main__":
    unittest.main()
