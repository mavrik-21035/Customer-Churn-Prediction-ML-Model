# Customer Churn Analysis & Prediction

End-to-end, schema-adaptive binary churn analysis for the supplied Telco CSV.
The checked-in dataset has 7,043 rows, 21 columns, a `Churn` target (`Yes` is
the positive class), and a `customerID` identifier. The pipeline detects ID
columns, converts numeric-looking text (including blank `TotalCharges`), and
fits all imputers/encoders inside stratified training folds.

## Setup and run

From the project directory in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --data data\WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn --positive-label Yes
```

In VS Code, select `.venv\Scripts\python.exe` as the workspace interpreter if
the Python extension has not selected it automatically.

For another dataset, supply its CSV, binary target, positive class, and any
known leakage columns:

```powershell
python main.py --data data\customer_churn.csv --target Churn --positive-label 1 --drop-cols customer_id signup_date
```

Use `--no-auto-detect-ids` to disable identifier inference. Only identifiers
are inferred: specify any outcome-leaking fields explicitly with `--drop-cols`.
Run a previously saved model on new records (unseen categories are ignored by
the fitted encoder):

```powershell
python -m src.predict --model outputs\model.joblib --data data\new_customers.csv --output outputs\new_customer_scores.csv
```

Launch the interactive dashboard after running the pipeline:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).
The dashboard includes EDA and segment exploration, model metrics, SHAP
explanations, risk-tier filtering/downloads, and local scoring for uploaded CSVs.
Its dark theme, animated aurora/radar hero, glowing metric cards, and hover
transitions are defined in `app.py` and `.streamlit/config.toml`; motion effects
respect the browser's reduced-motion preference.

## Outputs

- `outputs/eda/`: schema/data-quality summary, target and segment churn rates,
  numeric comparisons, plots, and a five-observation EDA report.
- `outputs/evaluation/`: cross-validated model comparison, untouched test
  metrics, confusion matrix, ROC/PR/confusion plots, global feature importance,
  and SHAP explanations (or a logged feature-importance fallback). Local
  high-risk explanations include the available identifier columns.
- `outputs/scored_predictions.csv`: record IDs (or generated `record_id`),
  churn probability, predicted target label, and `High`/`Medium`/`Low` risk tier.
- `outputs/model.joblib`: fitted preprocessing/model pipeline and inference
  metadata.
- `outputs/retention_playbook.md` and `outputs/run_summary.md`.

The four model families (balanced Logistic Regression, Random Forest, XGBoost,
and LightGBM) are tuned with stratified cross-validation on the training split;
the highest mean CV ROC-AUC model is selected. The comparison CSV includes
ROC-AUC, PR-AUC, precision, recall, F1, thresholds, and confusion-matrix counts
for every model on the same stratified 20% holdout. Each model's F1-maximizing
threshold is chosen from out-of-fold training probabilities. Risk bands use
fixed probability cutoffs of 0.33 and 0.67; they are operational bands, not
calibrated business cost thresholds. No resampling occurs outside training folds.

The scored file is a prioritization aid, not a causal explanation or guarantee
that a customer will churn. Its rows include records used during model fitting;
use held-out metrics for performance claims and randomized holdouts to validate
retention actions. Set `--drop-cols` for any leakage fields that cannot be
identified from column names.
