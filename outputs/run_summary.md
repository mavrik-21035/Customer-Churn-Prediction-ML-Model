# Churn modeling run

- Dataset: `data\WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Target: `Churn`; positive class: `Yes`.
- Dropped identifier/leakage columns: ['customerID']
- Selected model: **Logistic Regression**.
- OOF F1 threshold (training data only): **0.5608**.
- Held-out test metrics: `{'roc_auc': 0.8454338784262059, 'pr_auc': 0.6521333765273201, 'precision': 0.5444015444015444, 'recall': 0.7540106951871658, 'f1': 0.6322869955156951, 'threshold': 0.5607682088498318, 'test_rows': 1409}`.
- Top global feature driver: **numeric__tenure**.
- Model artifact: `outputs\model.joblib`.

See `eda/eda_report.md`, `evaluation/model_comparison.csv`, `evaluation/test_metrics.csv`, and `retention_playbook.md` for details.
