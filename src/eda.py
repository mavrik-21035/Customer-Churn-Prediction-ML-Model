"""Generate data quality summaries, churn segment analysis, and plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)


def _categorical_analysis(
    frame: pd.DataFrame,
    target: str,
    positive_label: str,
    output_dir: Path,
    excluded_columns: set[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    insight_candidates: list[dict[str, Any]] = []
    feature_columns = [
        column for column in frame.columns
        if column != target and column not in excluded_columns and (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
        )
    ]
    for column in feature_columns:
        values = frame[column].astype("string").fillna("(Missing)")
        grouped = pd.DataFrame(
            {"segment": values, "churn": frame[target].astype(str) == positive_label}
        ).groupby("segment", dropna=False)["churn"].agg(["sum", "count", "mean"])
        grouped = grouped.rename(
            columns={"sum": "churners", "count": "customers", "mean": "churn_rate"}
        ).reset_index()
        grouped.insert(0, "feature", column)
        records.extend(grouped.to_dict(orient="records"))
        eligible = grouped[grouped["customers"] >= max(20, int(len(frame) * 0.01))]
        if len(eligible) >= 2:
            lowest = eligible.loc[eligible["churn_rate"].idxmin()]
            highest = eligible.loc[eligible["churn_rate"].idxmax()]
            insight_candidates.append({
                "kind": "categorical",
                "feature": column,
                "description": (
                    f"{column}: {highest['segment']} has "
                    f"{highest['churn_rate']:.1%} churn "
                    f"(n={int(highest['customers'])}) versus "
                    f"{lowest['segment']} at {lowest['churn_rate']:.1%} "
                    f"(n={int(lowest['customers'])})."
                ),
                "rate_gap": float(highest["churn_rate"] - lowest["churn_rate"]),
                "support": int(highest["customers"] + lowest["customers"]),
            })
        if grouped["customers"].sum() > 0:
            plot_data = grouped.nlargest(12, "customers").sort_values("churn_rate")
            figure, axis = plt.subplots(figsize=(9, max(3.5, len(plot_data) * 0.35)))
            sns.barplot(
                data=plot_data,
                x="churn_rate",
                y="segment",
                ax=axis,
                color="#4472C4",
            )
            axis.set(title=f"Churn rate by {column}", xlabel="Churn rate", ylabel="")
            formatter = plt.FuncFormatter(
                lambda value, _: f"{value:.0%}"
            )
            axis.xaxis.set_major_formatter(formatter)
            figure.tight_layout()
            figure.savefig(output_dir / f"churn_by_{_safe_name(column)}.png", dpi=140)
            plt.close(figure)
    segment_table = pd.DataFrame(records)
    if not segment_table.empty:
        segment_table["churn_rate"] = segment_table["churn_rate"].astype(float)
    return segment_table, insight_candidates


def _safe_name(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_" for character in value
    )


def _numeric_analysis(
    frame: pd.DataFrame,
    target: str,
    positive_label: str,
    output_dir: Path,
    excluded_columns: set[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    numeric_columns = [
        column for column in frame.select_dtypes(include="number").columns
        if column != target and column not in excluded_columns
    ]
    records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    binary_target = frame[target].astype(str) == positive_label
    for column in numeric_columns:
        churn_values = frame.loc[binary_target, column].dropna()
        retained_values = frame.loc[~binary_target, column].dropna()
        churn_mean = float(churn_values.mean()) if not churn_values.empty else np.nan
        retained_mean = (
            float(retained_values.mean())
            if not retained_values.empty
            else np.nan
        )
        pooled_std = float(frame[column].std()) or 0.0
        standardized_gap = (
            abs(churn_mean - retained_mean) / pooled_std if pooled_std else 0.0
        )
        records.append({
            "feature": column,
            "churn_mean": churn_mean,
            "retained_mean": retained_mean,
            "churn_median": (
                float(churn_values.median()) if not churn_values.empty else np.nan
            ),
            "retained_median": (
                float(retained_values.median())
                if not retained_values.empty
                else np.nan
            ),
            "standardized_mean_gap": standardized_gap,
            "churn_count": len(churn_values),
            "retained_count": len(retained_values),
        })
        if len(churn_values) >= 10 and len(retained_values) >= 10:
            direction = "higher" if churn_mean > retained_mean else "lower"
            candidates.append({
                "kind": "numeric",
                "feature": column,
                "description": (
                    f"{column}: churners average {churn_mean:.2f} versus "
                    f"{retained_mean:.2f} for retained customers "
                    f"({direction} among churners; standardized gap "
                    f"{standardized_gap:.2f})."
                ),
                "rate_gap": standardized_gap,
                "support": len(churn_values) + len(retained_values),
            })
        sample = frame[[column]].copy()
        sample["churn_status"] = np.where(binary_target, "Churn", "Retained")
        figure, axis = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=sample, x="churn_status", y=column, showfliers=False, ax=axis)
        axis.set(title=f"{column} by churn status", xlabel="", ylabel=column)
        figure.tight_layout()
        figure.savefig(output_dir / f"distribution_{_safe_name(column)}.png", dpi=140)
        plt.close(figure)
    return pd.DataFrame(records), candidates


def run_eda(
    frame: pd.DataFrame,
    target: str,
    positive_label: str,
    output_dir: str | Path,
    excluded_columns: list[str] | None = None,
) -> list[str]:
    """Write EDA CSVs, plots, and a concise, dataset-specific Markdown report."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} was not found.")
    total_rows, total_columns = frame.shape
    missing = (frame.isna().mean() * 100).rename("missing_percent")
    quality = pd.DataFrame({
        "dtype": frame.dtypes.astype(str),
        "missing_count": frame.isna().sum(),
        "missing_percent": missing,
        "unique_values": frame.nunique(dropna=True),
    })
    quality.to_csv(destination / "data_quality.csv", index_label="column")
    duplicate_count = int(frame.duplicated().sum())
    target_counts = frame[target].astype(str).value_counts()
    churn_rate = float((frame[target].astype(str) == positive_label).mean())

    excluded = set(excluded_columns or [])
    segment_table, categorical_candidates = _categorical_analysis(
        frame, target, positive_label, destination, excluded
    )
    numeric_table, numeric_candidates = _numeric_analysis(
        frame, target, positive_label, destination, excluded
    )
    segment_table.to_csv(destination / "categorical_churn_rates.csv", index=False)
    numeric_table.to_csv(destination / "numeric_churn_differences.csv", index=False)

    figure, axis = plt.subplots(figsize=(6, 4))
    target_counts.plot(kind="bar", color=["#4472C4", "#ED7D31"], ax=axis)
    axis.set(title="Target class balance", xlabel=target, ylabel="Customers")
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    figure.savefig(destination / "target_class_balance.png", dpi=140)
    plt.close(figure)

    candidates = sorted(
        categorical_candidates + numeric_candidates,
        key=lambda item: (item["rate_gap"], item["support"]),
        reverse=True,
    )
    insights = [item["description"] for item in candidates[:5]]
    if not insights:
        insights = [
            "Insufficient segment support to produce stable comparative insights."
        ]
    missing_lines = [
        f"- `{column}`: {int(row['missing_count'])} "
        f"({row['missing_percent']:.2f}%)"
        for column, row in quality.iterrows()
        if row["missing_count"] > 0
    ]
    if not missing_lines:
        missing_lines = ["- No missing values detected after input normalization."]
    dtypes = quality["dtype"].value_counts().to_dict()
    balance_lines = [
        f"- `{label}`: {int(count):,} ({count / total_rows:.1%})"
        for label, count in target_counts.items()
    ]
    report_lines = [
        "# Customer churn EDA",
        "",
        "## Dataset overview",
        f"- Rows: {total_rows:,}; columns: {total_columns:,}.",
        f"- Duplicate rows: {duplicate_count:,}.",
        f"- Data types: {dtypes}.",
        f"- Churn rate (`{positive_label}`): {churn_rate:.2%}.",
        "- Target class balance:",
        *balance_lines,
        "",
        "## Missing values",
        *missing_lines,
        "",
        "## Top data-backed observations",
        *[f"{index}. {insight}" for index, insight in enumerate(insights, start=1)],
        "",
        "Segment and numeric detail tables are saved beside this report; plots cover "
        "the target balance, categorical segment rates, and numeric distributions.",
        "",
        "These are descriptive associations, not causal findings. Validate "
        "interventions with controlled experiments.",
    ]
    (destination / "eda_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    LOGGER.info("EDA artifacts written to %s", destination)
    return insights
