"""Interactive Streamlit dashboard for customer churn analysis and scoring."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.predict import predict_frame

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
EDA_DIR = OUTPUTS / "eda"
EVALUATION_DIR = OUTPUTS / "evaluation"
MODEL_PATH = OUTPUTS / "model.joblib"
SCORES_PATH = OUTPUTS / "scored_predictions.csv"

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv(path: str, modified_time: float) -> pd.DataFrame:
    """Load a local CSV and refresh the cache when its modification time changes."""
    del modified_time
    return pd.read_csv(path, low_memory=False)


@st.cache_resource(show_spinner=False)
def load_model(path: str, modified_time: float) -> dict:
    """Load the locally generated model bundle once per dashboard session."""
    del modified_time
    return joblib.load(path)


def artifact_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return load_csv(str(path), path.stat().st_mtime)


def metric_value(metrics: pd.DataFrame | None, column: str) -> float | None:
    if metrics is None or metrics.empty or column not in metrics:
        return None
    return float(metrics.iloc[0][column])


st.markdown(
    """
    <style>
    @keyframes aurora-drift {
        0% {transform: translate3d(-4%, -2%, 0) rotate(0deg);}
        50% {transform: translate3d(3%, 2%, 0) rotate(5deg);}
        100% {transform: translate3d(-4%, -2%, 0) rotate(0deg);}
    }
    @keyframes radar-sweep {
        from {transform: translate(-50%, -50%) rotate(0deg);}
        to {transform: translate(-50%, -50%) rotate(360deg);}
    }
    @keyframes signal-pulse {
        0%, 100% {box-shadow: 0 0 0 0 rgba(53, 231, 255, .48);}
        50% {box-shadow: 0 0 0 9px rgba(53, 231, 255, 0);}
    }
    @keyframes card-arrive {
        from {opacity: 0; transform: translateY(14px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 2%, rgba(33, 99, 196, .2), transparent 34%),
            radial-gradient(
                ellipse at 90% 15%,
                rgba(103, 47, 183, .16),
                transparent 32%
            ),
            linear-gradient(145deg, #080d1b 0%, #0b1224 48%, #09101d 100%);
        background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        z-index: 0;
        top: -25vh;
        left: -18vw;
        width: 70vw;
        height: 65vh;
        pointer-events: none;
        background: radial-gradient(
            ellipse,
            rgba(0, 211, 255, .12),
            rgba(107, 78, 255, .04) 42%,
            transparent 72%
        );
        filter: blur(35px);
        animation: aurora-drift 20s ease-in-out infinite;
    }
    [data-testid="stMain"], [data-testid="stSidebar"] {
        position: relative;
        z-index: 1;
    }
    [data-testid="stSidebar"] {
        background: rgba(8, 14, 29, .72);
        border-right: 1px solid rgba(105, 163, 255, .16);
        backdrop-filter: blur(18px);
    }
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1440px;}
    .hero-shell {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 2rem;
        min-height: 250px;
        margin: 0 0 1.5rem;
        padding: clamp(1.6rem, 4vw, 3rem);
        overflow: hidden;
        isolation: isolate;
        border: 1px solid rgba(119, 181, 255, .26);
        border-radius: 26px;
        background:
            linear-gradient(115deg, rgba(16, 29, 56, .96), rgba(13, 19, 42, .78)),
            radial-gradient(
                circle at 85% 50%,
                rgba(29, 220, 255, .24),
                transparent 36%
            );
        box-shadow:
            0 22px 70px rgba(0, 0, 0, .28),
            inset 0 1px 0 rgba(255, 255, 255, .08);
    }
    .hero-shell::before {
        content: "";
        position: absolute;
        z-index: -1;
        inset: -50%;
        background: conic-gradient(
            from 25deg at 70% 50%,
            transparent 0deg,
            rgba(46, 218, 255, .1) 60deg,
            rgba(142, 83, 255, .13) 135deg,
            transparent 220deg
        );
        filter: blur(30px);
        animation: aurora-drift 24s ease-in-out infinite;
    }
    .hero-copy {max-width: 760px;}
    .hero-eyebrow {
        display: flex;
        align-items: center;
        gap: .65rem;
        margin-bottom: 1rem;
        color: #86dfff;
        font-size: .73rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .signal-pulse {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #35e7ff;
        box-shadow: 0 0 14px rgba(53, 231, 255, .9);
        animation: signal-pulse 2s ease-out infinite;
    }
    .hero-title {
        margin: 0;
        color: #f2f7ff;
        font-size: clamp(2rem, 3.5vw, 3.2rem);
        font-weight: 780;
        letter-spacing: -.055em;
        line-height: 1.02;
        text-shadow: 0 0 35px rgba(71, 165, 255, .18);
    }
    .hero-title span {
        color: #55e5ff;
        background: linear-gradient(95deg, #73e8ff, #87a7ff 55%, #c3a0ff);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-description {
        max-width: 630px;
        margin: 1rem 0 0;
        color: #aab9d4;
        font-size: 1rem;
        line-height: 1.6;
    }
    .hero-radar {
        position: relative;
        flex: 0 0 clamp(120px, 14vw, 175px);
        aspect-ratio: 1;
        overflow: hidden;
        border: 1px solid rgba(112, 218, 255, .36);
        border-radius: 50%;
        background:
            radial-gradient(circle, rgba(44, 219, 255, .18) 0 2%, transparent 3%),
            repeating-radial-gradient(
                circle, transparent 0 22%, rgba(105, 185, 255, .12) 22.4% 22.9%
            ),
            linear-gradient(145deg, rgba(18, 68, 109, .4), rgba(42, 24, 86, .28));
        box-shadow:
            0 0 24px rgba(39, 208, 255, .16),
            inset 0 0 28px rgba(50, 180, 255, .12);
    }
    .hero-radar::before {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 100%;
        height: 1px;
        background: rgba(122, 215, 255, .3);
        box-shadow: 0 0 12px rgba(44, 219, 255, .6);
    }
    .hero-radar::after {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 50%;
        height: 50%;
        transform-origin: 0 0;
        background: conic-gradient(
            from 0deg, rgba(36, 231, 255, .32), transparent 62deg, transparent 360deg
        );
        animation: radar-sweep 7s linear infinite;
    }
    .radar-core {
        position: absolute;
        z-index: 2;
        top: 50%;
        left: 50%;
        width: 9px;
        height: 9px;
        transform: translate(-50%, -50%);
        border-radius: 50%;
        background: #c1fbff;
        box-shadow: 0 0 9px 4px rgba(45, 224, 255, .65);
    }
    [data-testid="stMetric"] {
        background: linear-gradient(
            145deg, rgba(24, 40, 69, .76), rgba(14, 22, 42, .72)
        );
        border: 1px solid rgba(111, 178, 255, .23);
        padding: 10px 12px;
        border-radius: 14px;
        box-shadow:
            0 8px 26px rgba(0, 0, 0, .18),
            inset 0 1px 0 rgba(255, 255, 255, .055);
        backdrop-filter: blur(12px);
        transition: transform .24s ease, box-shadow .24s ease, border-color .24s ease;
        animation: card-arrive .55s ease both;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(68, 223, 255, .62);
        box-shadow: 0 12px 34px rgba(0, 180, 255, .13);
    }
    [data-testid="stMetricValue"] {font-size: clamp(1.2rem, 2vw, 1.7rem);}
    [data-testid="stTabs"] button[role="tab"] {
        border: 1px solid transparent;
        border-radius: 11px;
        transition: color .2s ease, background .2s ease, box-shadow .2s ease;
    }
    [data-testid="stTabs"] button[role="tab"]:hover {
        color: #75e8ff;
        background: rgba(64, 158, 255, .09);
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #d7f8ff;
        border-color: rgba(66, 220, 255, .32);
        background: linear-gradient(
            110deg, rgba(23, 137, 190, .22), rgba(104, 81, 202, .2)
        );
        box-shadow: 0 0 20px rgba(34, 196, 255, .1);
    }
    .stButton > button, .stDownloadButton > button {
        border: 1px solid rgba(93, 197, 255, .36);
        border-radius: 11px;
        background: linear-gradient(
            110deg, rgba(21, 107, 157, .32), rgba(93, 69, 162, .27)
        );
        transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        border-color: rgba(89, 226, 255, .8);
        box-shadow: 0 7px 22px rgba(33, 186, 255, .17);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(107, 160, 230, .2);
        border-radius: 12px;
        overflow: hidden;
    }
    @media (max-width: 700px) {
        .hero-shell {min-height: 0; padding: 1.5rem;}
        .hero-radar {display: none;}
        .hero-eyebrow {
            font-size: .63rem;
            letter-spacing: .1em;
            white-space: normal;
        }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero-shell">
      <div class="hero-copy">
        <div class="hero-eyebrow">
          <span class="signal-pulse"></span>
          RETENTION ANALYTICS
          <span aria-hidden="true">/</span>
          MODEL INTELLIGENCE
        </div>
        <h1 class="hero-title">
          Customer Churn<br><span>Intelligence</span>
        </h1>
        <p class="hero-description">
          See risk earlier. Understand the signals. Focus retention effort where
          it can make a difference.
        </p>
      </div>
      <div class="hero-radar" aria-hidden="true"><div class="radar-core"></div></div>
    </section>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file() or not SCORES_PATH.is_file():
    st.error("Model and score artifacts are missing. Run the analysis pipeline first:")
    st.code(
        "python main.py --data data\\WA_Fn-UseC_-Telco-Customer-Churn.csv "
        "--target Churn --positive-label Yes"
    )
    st.stop()

bundle = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
scores = artifact_csv(SCORES_PATH)
comparison = artifact_csv(EVALUATION_DIR / "model_comparison.csv")
metrics = artifact_csv(EVALUATION_DIR / "test_metrics.csv")
quality = artifact_csv(EDA_DIR / "data_quality.csv")
segments = artifact_csv(EDA_DIR / "categorical_churn_rates.csv")
importance = artifact_csv(EVALUATION_DIR / "feature_importance.csv")
local_explanations = artifact_csv(
    EVALUATION_DIR / "high_risk_customer_explanations.csv"
)

with st.sidebar:
    st.header("Dashboard controls")
    st.caption(f"Selected model: **{bundle.get('model_name', 'Unknown')}**")
    st.caption(f"Decision threshold: **{bundle.get('threshold', 0.5):.3f}**")
    if st.button("Refresh dashboard data", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.divider()
    st.markdown(
        "Use **Customer risk** to filter the current scored list or upload a "
        "new customer CSV for in-memory scoring."
    )

if scores is None:
    st.error(f"Scored customer file not found: {SCORES_PATH}")
    st.stop()

overview_tab, drivers_tab, model_tab, risk_tab, actions_tab = st.tabs(
    [
        "Overview",
        "Churn drivers",
        "Model performance",
        "Customer risk",
        "Retention playbook",
    ]
)

with overview_tab:
    total_customers = len(scores)
    high_risk_count = int((scores["risk_tier"] == "High").sum())
    roc_auc = metric_value(metrics, "roc_auc")
    pr_auc = metric_value(metrics, "pr_auc")
    first, second, third, fourth = st.columns(4)
    first.metric("Customers", f"{total_customers:,}")
    second.metric(
        "High risk",
        f"{high_risk_count:,}",
        f"{high_risk_count / max(total_customers, 1):.1%} of scored list",
    )
    third.metric(
        "Test ROC-AUC",
        f"{roc_auc:.3f}" if roc_auc is not None else "N/A",
    )
    fourth.metric("Test PR-AUC", f"{pr_auc:.3f}" if pr_auc is not None else "N/A")

    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Risk tier distribution")
        risk_counts = (
            scores["risk_tier"]
            .value_counts()
            .reindex(["High", "Medium", "Low"], fill_value=0)
            .rename_axis("Risk tier")
            .to_frame("Customers")
        )
        st.bar_chart(risk_counts)
    with right:
        st.subheader("Holdout model snapshot")
        if metrics is not None and not metrics.empty:
            display_metrics = metrics.iloc[0].to_dict()
            display_metrics.pop("test_rows", None)
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Metric": key.replace("_", " ").title(), "Value": value}
                        for key, value in display_metrics.items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info(
                "Test metrics are unavailable. Rerun the pipeline to generate them."
            )

    if pr_auc is not None:
        st.caption(
            f"Holdout PR-AUC: {pr_auc:.3f}. The F1 threshold was selected from "
            "out-of-fold training predictions, not the test set."
        )
    st.info(
        "Risk tiers prioritize review; they are not guarantees of churn. "
        "Validate retention actions with a randomized holdout."
    )

with drivers_tab:
    st.subheader("What is associated with churn?")
    report_path = EDA_DIR / "eda_report.md"
    if report_path.is_file():
        report_text = report_path.read_text(encoding="utf-8")
        observations = report_text.split("## Top data-backed observations", 1)
        if len(observations) == 2:
            insight_text = observations[1].split("\n\n", 1)[0].strip()
            st.markdown(insight_text)
        with st.expander("Dataset quality and full EDA summary"):
            st.markdown(report_text)
    else:
        st.warning("EDA report is unavailable. Rerun the pipeline to create it.")

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("#### Global model drivers")
        if importance is not None and not importance.empty:
            top_features = importance.head(15).sort_values("importance")
            st.bar_chart(top_features, x="feature", y="importance")
            st.caption(
                "Global importance describes model associations; it does not "
                "establish causal effects."
            )
        else:
            st.info("Feature importance output is unavailable.")
    with right:
        st.markdown("#### Churn rate by customer segment")
        if segments is not None and not segments.empty:
            features = sorted(segments["feature"].dropna().unique().tolist())
            selected_feature = st.selectbox("Segment feature", features)
            segment_data = segments[segments["feature"] == selected_feature].copy()
            segment_data = segment_data.sort_values("churn_rate", ascending=False)
            st.bar_chart(segment_data, x="segment", y="churn_rate")
            st.dataframe(
                segment_data[["segment", "customers", "churners", "churn_rate"]],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("Categorical segment analysis is unavailable.")

with model_tab:
    st.subheader("Cross-validated model comparison")
    if comparison is not None and not comparison.empty:
        st.dataframe(comparison, hide_index=True, width="stretch")
        chart_metrics = [
            column for column in
            ("test_roc_auc", "test_pr_auc", "test_precision", "test_recall", "test_f1")
            if column in comparison
        ]
        if chart_metrics:
            st.bar_chart(comparison.set_index("model")[chart_metrics])
    else:
        st.warning("Model comparison output is unavailable.")

    confusion_path = EVALUATION_DIR / "model_evaluation.png"
    if confusion_path.is_file():
        st.subheader("Selected model: ROC, precision-recall, and confusion matrix")
        st.image(str(confusion_path), width="stretch")
    st.caption(
        "Thresholds are tuned using training out-of-fold predictions. "
        "Holdout metrics are for evaluation, not threshold selection."
    )

with risk_tab:
    st.subheader("Customer risk explorer")
    risk_tiers = sorted(scores["risk_tier"].dropna().unique().tolist())
    selected_tiers = st.multiselect(
        "Risk tiers",
        options=risk_tiers,
        default=risk_tiers,
    )
    minimum_probability = st.slider(
        "Minimum churn probability",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )
    identifier_columns = [
        column for column in scores.columns
        if column not in {"churn_probability", "predicted_churn", "risk_tier"}
    ]
    search_text = st.text_input("Search customer ID", placeholder="Type an ID")
    filtered = scores[
        scores["risk_tier"].isin(selected_tiers)
        & (scores["churn_probability"] >= minimum_probability)
    ]
    if search_text and identifier_columns:
        matching_id = filtered[identifier_columns].astype(str).apply(
            lambda column: column.str.contains(
                search_text, case=False, regex=False, na=False
            )
        ).any(axis=1)
        filtered = filtered[matching_id]
    filtered = filtered.sort_values("churn_probability", ascending=False)
    st.caption(f"Showing {len(filtered):,} of {total_customers:,} customers")
    st.dataframe(filtered, hide_index=True, width="stretch")
    st.download_button(
        "Download filtered risk list",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customer_churn_scores.csv",
        mime="text/csv",
    )

    if local_explanations is not None and not local_explanations.empty:
        id_columns = [
            column for column in local_explanations.columns
            if column not in {
                "risk_rank", "test_row_position", "churn_probability",
                "feature", "shap_value",
            }
        ]
        if id_columns:
            selected_id_column = id_columns[0]
            local_ids = (
                local_explanations[selected_id_column].drop_duplicates().tolist()
            )
            selected_id = st.selectbox("Explain a high-risk customer", local_ids)
            explanation = local_explanations[
                local_explanations[selected_id_column] == selected_id
            ].sort_values("shap_value")
            st.markdown(
                f"#### Local SHAP drivers for `{selected_id}` "
                f"(risk {explanation['churn_probability'].iloc[0]:.1%})"
            )
            st.bar_chart(explanation, x="feature", y="shap_value")

    st.divider()
    st.markdown("#### Score a new customer CSV")
    st.caption("Upload a feature CSV; scoring is performed locally in this session.")
    uploaded_file = st.file_uploader(
        "Upload CSV for scoring",
        type=["csv"],
        key="customer_score_upload",
    )
    if uploaded_file is not None:
        if st.session_state.get("score_upload_name") != uploaded_file.name:
            st.session_state.pop("uploaded_scores", None)
            st.session_state["score_upload_name"] = uploaded_file.name
        try:
            upload_frame = pd.read_csv(uploaded_file, low_memory=False)
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not parse uploaded CSV: {exc}")
        else:
            if st.button("Score uploaded customers", type="primary"):
                try:
                    st.session_state["uploaded_scores"] = predict_frame(
                        upload_frame, bundle
                    )
                except (KeyError, ValueError) as exc:
                    st.error(f"Could not score uploaded records: {exc}")
            uploaded_scores = st.session_state.get("uploaded_scores")
            if uploaded_scores is not None:
                st.dataframe(
                    uploaded_scores,
                    hide_index=True,
                    width="stretch",
                )
                st.download_button(
                    "Download uploaded customer scores",
                    data=uploaded_scores.to_csv(index=False).encode("utf-8"),
                    file_name="uploaded_customer_churn_scores.csv",
                    mime="text/csv",
                    key="download_uploaded_scores",
                )

with actions_tab:
    st.subheader("Retention playbook")
    playbook_path = OUTPUTS / "retention_playbook.md"
    if playbook_path.is_file():
        playbook = playbook_path.read_text(encoding="utf-8")
        st.markdown(playbook)
    else:
        st.warning(
            "Retention playbook is unavailable. Rerun the pipeline to create it."
        )
