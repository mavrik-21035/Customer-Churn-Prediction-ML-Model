# Customer churn EDA

## Dataset overview
- Rows: 7,043; columns: 21.
- Duplicate rows: 0.
- Data types: {'string': 17, 'int64': 2, 'float64': 1, 'Float64': 1}.
- Churn rate (`Yes`): 26.54%.
- Target class balance:
- `No`: 5,174 (73.5%)
- `Yes`: 1,869 (26.5%)

## Missing values
- `TotalCharges`: 11 (0.16%)

## Top data-backed observations
1. tenure: churners average 17.98 versus 37.57 for retained customers (lower among churners; standardized gap 0.80).
2. TotalCharges: churners average 1531.80 versus 2555.34 for retained customers (lower among churners; standardized gap 0.45).
3. MonthlyCharges: churners average 74.44 versus 61.27 for retained customers (higher among churners; standardized gap 0.44).
4. Contract: Month-to-month has 42.7% churn (n=3875) versus Two year at 2.8% (n=1695).
5. InternetService: Fiber optic has 41.9% churn (n=3096) versus No at 7.4% (n=1526).

Segment and numeric detail tables are saved beside this report; plots cover the target balance, categorical segment rates, and numeric distributions.

These are descriptive associations, not causal findings. Validate interventions with controlled experiments.
