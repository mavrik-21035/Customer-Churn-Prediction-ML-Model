"""Run customer churn EDA, model comparison, evaluation, and risk scoring."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data_prep import (
    clean_dataframe,
    detect_id_columns,
    make_model_frame,
    read_dataset,
    resolve_positive_label,
)
from src.eda import run_eda
from src.evaluate import evaluate_model, explain_model, save_model
from src.predict import score_frame
from src.train import train_models

LOGGER = logging.getLogger(__name__)


def _resolve_column(frame_columns: list[str], requested: str) -> str:
    lookup = {column.casefold(): column for column in frame_columns}
    resolved = lookup.get(requested.strip().casefold())
    if resolved is None:
        raise ValueError(
            f"Target column {requested!r} not found. Available columns: {frame_columns}"
        )
    return resolved


def _parse_columns(values: list[str] | None) -> list[str]:
    return [value.strip() for value in (values or []) if value.strip()]


def _retention_action(feature: str) -> str:
    normalized = "".join(
        character for character in feature.casefold() if character.isalnum()
    )
    if any(token in normalized for token in ("contract", "tenure", "newcustomer")):
        return (
            "Offer early-life onboarding and a clear, no-pressure review of "
            "contract options before renewal; test term incentives against margin."
        )
    if any(token in normalized for token in ("charge", "price", "spend", "bill")):
        return (
            "Offer a bill review and right-sized plan comparison, explain recent "
            "charges, and test targeted offers against a no-discount holdout."
        )
    if any(
        token in normalized for token in ("support", "security", "backup", "device")
    ):
        return (
            "Offer proactive setup/troubleshooting for the relevant support or "
            "protection service, then follow up to confirm the issue was resolved."
        )
    if any(token in normalized for token in ("payment", "billing", "paperless")):
        return (
            "Review payment friction and billing preferences with the customer; "
            "test an assisted payment-method change or reminder where appropriate."
        )
    if any(token in normalized for token in ("internet", "streaming", "service")):
        return (
            "Run a needs-based service review for the segment and test a relevant "
            "service-quality or plan-fit intervention before offering a discount."
        )
    return (
        "Contact this segment for a brief needs and satisfaction review, identify "
        "the specific friction point, and test one targeted remedy."
    )


def _run_playbook(
    insights: list[str], output_path: Path, churn_rate: float
) -> None:
    playbook = [
        "# Retention playbook",
        "",
        f"Observed dataset churn rate: **{churn_rate:.2%}**.",
        "",
        "Prioritize the following evidence-backed hypotheses. Validate them with "
        "controlled retention experiments before scaling.",
        "",
    ]
    for index, insight in enumerate(insights, start=1):
        feature = insight.partition(":")[0]
        playbook.extend([
            f"## {index}. {insight}",
            "",
            "- **Who to target:** Customers in the higher-risk segment "
            "described above; "
            "use the scored list and refresh it before each campaign.",
            f"- **Action to test:** {_retention_action(feature)}",
            "- **Next step:** Run a randomized holdout test; track "
            "incremental retention, "
            "contact cost, and margin over a defined follow-up window.",
            "",
        ])
    playbook.extend([
        "## Operational use",
        "",
        "- Start with the High tier, then test Medium-tier customers when contact "
        "capacity allows. Tier boundaries are descriptive probability bands, not "
        "guarantees of churn.",
        "- Compare interventions against a no-contact holdout and monitor calibration "
        "and subgroup performance after deployment.",
        "- Investigate missing data and shifts in customer mix before acting "
        "on changes "
        "in predicted risk.",
        "",
    ])
    output_path.write_text("\n".join(playbook), encoding="utf-8")


def run_pipeline(arguments: argparse.Namespace) -> None:
    raw = read_dataset(arguments.data)
    cleaned = clean_dataframe(raw)
    target = _resolve_column(list(cleaned.columns), arguments.target)
    positive_label = resolve_positive_label(
        cleaned[target], arguments.positive_label
    )
    id_columns = (
        detect_id_columns(cleaned, target)
        if arguments.auto_detect_ids
        else []
    )
    explicit_drop = _parse_columns(arguments.drop_cols)
    missing_drop = sorted(set(explicit_drop) - set(cleaned.columns))
    if missing_drop:
        raise ValueError(f"Drop columns not found in CSV: {missing_drop}")
    drop_columns = list(dict.fromkeys(id_columns + explicit_drop))
    negative_label = next(
        str(value)
        for value in cleaned[target].unique()
        if str(value) != positive_label
    )
    target_binary = (cleaned[target].astype(str) == positive_label).astype("int8")
    features = make_model_frame(cleaned, target, drop_columns)
    if features.isna().all(axis=1).all():
        raise ValueError("Every row has missing values for every model feature.")

    output_dir = Path(arguments.output_dir)
    eda_dir = output_dir / "eda"
    evaluation_dir = output_dir / "evaluation"
    insights = run_eda(
        cleaned, target, positive_label, eda_dir, excluded_columns=drop_columns
    )
    churn_rate = float(target_binary.mean())
    _run_playbook(insights, output_dir / "retention_playbook.md", churn_rate)

    result = train_models(
        features,
        target_binary,
        test_size=0.2,
        cv_folds=arguments.cv_folds,
    )
    metrics = evaluate_model(result, evaluation_dir)
    customer_ids = (
        cleaned.loc[result.X_test.index, id_columns].reset_index(drop=True)
        if id_columns
        else None
    )
    importance = explain_model(
        result, evaluation_dir, customer_ids=customer_ids
    )
    metadata = {
        "target": target,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "id_columns": id_columns,
        "drop_columns": drop_columns,
        "feature_columns": features.columns.tolist(),
        "raw_columns": cleaned.columns.tolist(),
    }
    model_path = output_dir / "model.joblib"
    save_model(result, model_path, metadata)
    bundle = {
        "pipeline": result.pipeline,
        "threshold": result.threshold,
        **metadata,
    }
    score_frame(cleaned, bundle, output_dir / "scored_predictions.csv")

    summary = [
        "# Churn modeling run",
        "",
        f"- Dataset: `{arguments.data}`",
        f"- Target: `{target}`; positive class: `{positive_label}`.",
        f"- Dropped identifier/leakage columns: {drop_columns or 'none'}",
        f"- Selected model: **{result.model_name}**.",
        f"- OOF F1 threshold (training data only): **{result.threshold:.4f}**.",
        f"- Held-out test metrics: `{metrics}`.",
        f"- Top global feature driver: **{importance.iloc[0]['feature']}**.",
        f"- Model artifact: `{model_path}`.",
        "",
        "See `eda/eda_report.md`, `evaluation/model_comparison.csv`, "
        "`evaluation/test_metrics.csv`, and `retention_playbook.md` for details.",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    LOGGER.info("Selected %s; held-out metrics: %s", result.model_name, metrics)
    LOGGER.info("Top model drivers: %s", importance.head(5)["feature"].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Input CSV path.",
    )
    parser.add_argument("--target", default="Churn", help="Binary target column.")
    parser.add_argument(
        "--positive-label",
        help="Positive/churn class; inferred from known labels when omitted.",
    )
    parser.add_argument(
        "--drop-cols",
        nargs="*",
        default=[],
        help="Additional ID or leakage columns excluded from model features.",
    )
    parser.add_argument(
        "--auto-detect-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically exclude ID-like columns (default: enabled).",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cv-folds", type=int, default=3)
    arguments = parser.parse_args()
    if not 2 <= arguments.cv_folds <= 10:
        parser.error("--cv-folds must be between 2 and 10.")
    return arguments


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run_pipeline(parse_args())
    except (FileNotFoundError, ValueError, ImportError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc
