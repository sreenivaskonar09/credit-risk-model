# Credit Scoring Model Development & SR 11-7 Validation

**Rutgers Business School | MQF Program | Fall 2025**

An end-to-end credit risk modeling framework built from the model developer's seat: a WoE/IV logistic regression PD scorecard, an XGBoost challenger, a TTC-to-PIT calibration overlay, macroeconomic PD stress testing, rating transition matrices, and an independent SR 11-7 validation report with severity-graded findings.

All data is synthetic and seeded (`seed=42`), so every number below reproduces exactly.

---

## Project Structure

```
credit-risk-model/
├── main.py                          # 10-step pipeline entry point
├── requirements.txt
├── README.md
├── utils/
│   └── data_generator.py            # Synthetic obligors, rating history, macro scenarios
├── models/
│   ├── pd_model.py                  # WoE/IV encoder, PD scorecard, TTC→PIT, challenger
│   └── transition_matrix.py         # Cohort + duration TM estimation and tests
└── validation/
    └── model_validator.py           # SR 11-7 FindingsLog + ModelValidator
```

---

## Setup & Run

```bash
git clone https://github.com/sreenivaskonar09/credit-risk-model.git
cd credit-risk-model
pip install -r requirements.txt
python main.py
```

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `utils/data_generator.py` | 5,000 performing obligors (AAA–CCC), 10-year rating history, 3 macro scenarios |
| 2 | `models/pd_model.py` | WoE/IV feature analysis across 7 predictors |
| 3 | `models/pd_model.py` | Logistic regression PD scorecard on WoE-transformed features |
| 4 | `models/pd_model.py` + `validation/` | Discrimination (AUC, Gini, KS), calibration (Brier, Hosmer-Lemeshow), PSI |
| 5 | `models/pd_model.py` | XGBoost challenger with SHAP feature importances |
| 6 | `models/pd_model.py` | TTC-to-PIT calibration overlay (Vasicek single-factor Z-shift) |
| 7 | `models/pd_model.py` | Stressed PD under unemployment, GDP and HPI scenario paths |
| 8 | `models/transition_matrix.py` | 8×8 cohort + generator TM, cumulative default rates, Moody's benchmark |
| 9 | `models/transition_matrix.py` + `validation/` | Monotonicity, diagonal dominance, stationarity (χ² homogeneity) |
| 10 | `validation/model_validator.py` | Consolidated SR 11-7 report with finding severity levels |

---

## Results

### PD Scorecard (test set: 1,000 obligors, 2.30% default rate)

| Metric | Value | Threshold |
|--------|------:|-----------|
| AUC | 0.912 | ≥ 0.70 |
| Gini | 0.824 | ≥ 0.40 |
| KS | 0.729 | ≥ 0.30 |
| Brier score | 0.020 | < 0.05 |
| Hosmer-Lemeshow p-value | 0.374 | > 0.05 |
| PSI (train vs test) | 0.003 | < 0.10 stable |

**Challenger:** XGBoost AUC 0.883 (−0.029 vs. champion). The interpretable scorecard is retained.

### TTC-to-PIT Calibration Overlay

```
PD_PIT = Φ( [Φ⁻¹(PD_TTC) − √ρ · Z] / √(1 − ρ) ),   ρ = 0.12
```

Z is a credit-cycle index built from standardised gaps of unemployment, GDP growth and HPI
against long-run U.S. averages (Z < 0 = downturn).

| Conditions | Z | Avg TTC PD | Avg PIT PD |
|------------|--:|-----------:|-----------:|
| Base (2024) | +0.24 | 2.36% | 1.68% |
| Adverse | −1.27 | 2.36% | 4.37% |
| Severely Adverse | −2.57 | 2.36% | 8.76% |

Current Base conditions are benign (unemployment ≈ 4.1% vs. 5.8% long-run), so PIT PDs sit below TTC.

### Macroeconomic Stressed PD

`multiplier = exp(β_u·ΔU + β_g·ΔGDP + β_h·ΔHPI)`, with shocks measured against Base quarter by quarter.

| Scenario | Peak quarter | ΔU (pp) | ΔGDP (pp) | ΔHPI (pp) | Peak avg PD | Multiplier |
|----------|-------------|--------:|----------:|----------:|------------:|-----------:|
| Base | 2024Q1 | 0.0 | 0.0 | 0.0 | 2.36% | 1.00× |
| Adverse | 2024Q4 | +3.6 | −3.9 | −8.4 | 4.53% | 1.93× |
| Severely Adverse | 2024Q4 | +6.8 | −7.4 | −16.4 | 8.15% | 3.46× |

Scenario monotonicity (Base ≤ Adverse ≤ Severe): **PASS**

### Transition Matrix

Monotone default column (PASS), stationary across sub-periods (χ² p = 0.19, PASS), average diagonal 84.3% (just below the 85% threshold).

### SR 11-7 Validation Report

```
Overall Opinion   : Pass with Observations
Highest Severity  : Minor
Total Findings    : 3

[Minor] Transition Matrix — Diagonal Dominance
  Issue: Avg diagonal 0.8431 below 85% threshold
[Minor] Transition Matrix — Moody's Benchmark (A)
  Issue: 5Y CDR deviates 68.0% from Moody's published rate
[Minor] Transition Matrix — Moody's Benchmark (BBB)
  Issue: 5Y CDR deviates 67.9% from Moody's published rate
```

---

## Key Concepts & Formulas

| Metric | Formula | Threshold |
|--------|---------|-----------|
| WoE | `ln[P(D in bin) / P(ND in bin)]` | — |
| IV | `Σ (P_D − P_ND) × WoE` | ≥ 0.10 medium, ≥ 0.30 strong |
| Gini | `2 × AUC − 1` | ≥ 0.40 |
| KS | `max(CDF_D − CDF_ND)` | ≥ 0.30 |
| Brier | `mean[(PD − y)²]` | < 0.05 |
| PSI | `Σ (Act% − Exp%) × ln(Act% / Exp%)` | < 0.10 stable |

**Transition matrix:** cohort `TM[i,j] = n(i→j) / n(i→·)`; duration `TM = expm(Q)`; multi-year `CDR(h) = (TM^h)[·, D]`.

### SR 11-7 Finding Severity

| Severity | Meaning |
|----------|---------|
| Observation | Minor issue; monitor |
| Minor | Investigate; remediate within 6 months |
| Moderate | Significant; conditional model approval |
| High | Model cannot be used until remediated |

---

## Limitations

- Data is synthetic. Features are generated from obligor ratings, so IVs (> 1.0 for five features) and discrimination are much stronger than in real portfolios.
- Macro sensitivities (β) and long-run macro anchors are illustrative, not estimated.
- The duration-method TM approximates the generator from annual observations rather than exact transition timestamps.

---

## Author

Sreenivas Konar
Master of Quantitative Finance, Rutgers Business School
[GitHub](https://github.com/sreenivaskonar09)
