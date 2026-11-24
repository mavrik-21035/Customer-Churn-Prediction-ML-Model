# Retention playbook

Observed dataset churn rate: **26.54%**.

Prioritize the following evidence-backed hypotheses. Validate them with controlled retention experiments before scaling.

## 1. tenure: churners average 17.98 versus 37.57 for retained customers (lower among churners; standardized gap 0.80).

- **Who to target:** Customers in the higher-risk segment described above; use the scored list and refresh it before each campaign.
- **Action to test:** Offer early-life onboarding and a clear, no-pressure review of contract options before renewal; test term incentives against margin.
- **Next step:** Run a randomized holdout test; track incremental retention, contact cost, and margin over a defined follow-up window.

## 2. TotalCharges: churners average 1531.80 versus 2555.34 for retained customers (lower among churners; standardized gap 0.45).

- **Who to target:** Customers in the higher-risk segment described above; use the scored list and refresh it before each campaign.
- **Action to test:** Offer a bill review and right-sized plan comparison, explain recent charges, and test targeted offers against a no-discount holdout.
- **Next step:** Run a randomized holdout test; track incremental retention, contact cost, and margin over a defined follow-up window.

## 3. MonthlyCharges: churners average 74.44 versus 61.27 for retained customers (higher among churners; standardized gap 0.44).

- **Who to target:** Customers in the higher-risk segment described above; use the scored list and refresh it before each campaign.
- **Action to test:** Offer a bill review and right-sized plan comparison, explain recent charges, and test targeted offers against a no-discount holdout.
- **Next step:** Run a randomized holdout test; track incremental retention, contact cost, and margin over a defined follow-up window.

## 4. Contract: Month-to-month has 42.7% churn (n=3875) versus Two year at 2.8% (n=1695).

- **Who to target:** Customers in the higher-risk segment described above; use the scored list and refresh it before each campaign.
- **Action to test:** Offer early-life onboarding and a clear, no-pressure review of contract options before renewal; test term incentives against margin.
- **Next step:** Run a randomized holdout test; track incremental retention, contact cost, and margin over a defined follow-up window.

## 5. InternetService: Fiber optic has 41.9% churn (n=3096) versus No at 7.4% (n=1526).

- **Who to target:** Customers in the higher-risk segment described above; use the scored list and refresh it before each campaign.
- **Action to test:** Run a needs-based service review for the segment and test a relevant service-quality or plan-fit intervention before offering a discount.
- **Next step:** Run a randomized holdout test; track incremental retention, contact cost, and margin over a defined follow-up window.

## Operational use

- Start with the High tier, then test Medium-tier customers when contact capacity allows. Tier boundaries are descriptive probability bands, not guarantees of churn.
- Compare interventions against a no-contact holdout and monitor calibration and subgroup performance after deployment.
- Investigate missing data and shifts in customer mix before acting on changes in predicted risk.
