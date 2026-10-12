"""
Credit Risk Modeling Project — Main Pipeline
============================================
Rutgers Business School | MQF Program | Fall 2025

10-step pipeline covering:
  1. Data generation
  2. WoE / IV feature analysis
  3. PD Scorecard (logistic regression on WoE features)
  4. Model validation (discriminatory power + calibration)
  5. Challenger model (XGBoost + SHAP)
  6. TTC-to-PIT calibration overlay (Vasicek single-factor Z-shift)
  7. Macroeconomic stressed PD (unemployment, GDP, HPI scenarios)
  8. Transition matrix estimation (cohort + duration methods)
  9. Transition matrix validation
 10. SR 11-7 consolidated report

Run:
    pip install -r requirements.txt
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Internal modules
from utils.data_generator import (
    generate_obligor_data,
    generate_rating_history,
    generate_macro_scenarios,
)
from models.pd_model import (
    PDScorecard,
    build_challenger,
    credit_cycle_index,
    ttc_to_pit,
)
from models.transition_matrix import (
    estimate_cohort,
    estimate_duration,
    check_monotonicity,
    check_diagonal_dominance,
    stationarity_test,
    cumulative_default_rates,
    benchmark_vs_moodys,
    tm_to_dataframe,
)
from validation.model_validator import (
    ModelValidator,
    assess_iv,
    compute_psi,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "debt_to_income",
    "payment_history",
    "credit_utilization",
    "years_in_bureau",
    "num_inquiries",
    "loan_to_value",
    "industry",
]


# ---------------------------------------------------------------------------
# Step 1 — Data Generation
# ---------------------------------------------------------------------------

def step_data():
    print("=" * 60)
    print("STEP 1: Data Generation")
    print("=" * 60)
    df = generate_obligor_data(n=5000, seed=42)
    history = generate_rating_history(n_obligors=1000, n_years=10, seed=42)
    macro   = generate_macro_scenarios()

    print(f"  Obligors generated       : {len(df):,}")
    print(f"  Default rate             : {df['default_flag'].mean():.2%}")
    print(f"  Rating observations      : {len(history):,}")
    print(f"  Macro scenarios          : {list(macro.keys())}")
    print(f"  Quarters per scenario    : {len(next(iter(macro.values())))}")
    return df, history, macro


# ---------------------------------------------------------------------------
# Step 2 — WoE / IV Feature Analysis
# ---------------------------------------------------------------------------

def step_woe_iv(df, scorecard):
    print("\n" + "=" * 60)
    print("STEP 2: WoE / IV Feature Analysis")
    print("=" * 60)
    iv_df = scorecard.iv_summary()
    print(iv_df.to_string(index=False))
    return iv_df


# ---------------------------------------------------------------------------
# Step 3 — PD Scorecard
# ---------------------------------------------------------------------------

def step_pd_scorecard(df):
    print("\n" + "=" * 60)
    print("STEP 3: PD Scorecard — Fit")
    print("=" * 60)
    scorecard = PDScorecard(feature_cols=FEATURE_COLS, test_size=0.20, seed=42)
    scorecard.fit(df, target="default_flag")
    print("  WoE encoder fit complete")
    print("  Logistic regression trained on WoE features")
    print()
    coef_df = scorecard.scorecard_coefficients()
    print(coef_df.to_string(index=False))
    return scorecard


# ---------------------------------------------------------------------------
# Step 4 — PD Model Validation
# ---------------------------------------------------------------------------

def step_pd_validation(scorecard, iv_df, validator):
    print("\n" + "=" * 60)
    print("STEP 4: PD Model Validation (SR 11-7 Outcomes Analysis)")
    print("=" * 60)
    metrics = scorecard.validate()
    print(f"  AUC          : {metrics['auc']}")
    print(f"  Gini         : {metrics['gini']}")
    print(f"  KS           : {metrics['ks']}")
    print(f"  Brier Score  : {metrics['brier_score']}")
    print(f"  H-L p-value  : {metrics['hl_pvalue']}")
    print(f"  Test size    : {metrics['n_test']} obs | Default rate: {metrics['default_rate_test']:.2%}")

    # PSI — compare train vs test score distributions
    train_scores = scorecard.predict_proba(
        scorecard.X_train[FEATURE_COLS] if hasattr(scorecard.X_train, 'columns')
        else pd.DataFrame(scorecard.X_train, columns=FEATURE_COLS)
    )
    test_scores = scorecard.predict_proba(
        scorecard.X_test[FEATURE_COLS] if hasattr(scorecard.X_test, 'columns')
        else pd.DataFrame(scorecard.X_test, columns=FEATURE_COLS)
    )
    psi = compute_psi(train_scores, test_scores)
    print(f"  PSI          : {psi} ({'stable' if psi < 0.10 else 'minor shift' if psi < 0.25 else 'MAJOR SHIFT'})")

    iv_dict = dict(zip(iv_df["feature"], iv_df["IV"]))
    findings = validator.validate_pd_model(metrics, iv_dict, psi=psi)
    return metrics, psi


# ---------------------------------------------------------------------------
# Step 5 — Challenger Model
# ---------------------------------------------------------------------------

def step_challenger(df, validator, champion_auc):
    print("\n" + "=" * 60)
    print("STEP 5: Challenger Model — XGBoost")
    print("=" * 60)
    result = build_challenger(df, FEATURE_COLS)
    if "error" in result:
        print(f"  Challenger skipped: {result['error']}")
        return result

    print(f"  Challenger AUC   : {result['auc']}")
    print(f"  Challenger Gini  : {result['gini']}")
    print(f"  AUC gap vs champion : {result['auc'] - champion_auc:+.4f}")
    print()
    print("  SHAP Feature Importances:")
    print(result["feature_importances"].to_string(index=False))

    # Add challenger comparison to SR 11-7 findings
    validator.validate_challenger(champion_auc, result["auc"])
    return result


# ---------------------------------------------------------------------------
# Step 6 — TTC-to-PIT Calibration Overlay
# ---------------------------------------------------------------------------

def step_ttc_to_pit(df, scorecard, macro):
    print("\n" + "=" * 60)
    print("STEP 6: TTC-to-PIT Calibration Overlay")
    print("=" * 60)
    # Defaults in the development sample are drawn from long-run rating
    # default rates, so scorecard PDs are treated as through-the-cycle (TTC).
    ttc_pd = scorecard.predict_proba(df[FEATURE_COLS])
    print(f"  Portfolio avg TTC PD : {ttc_pd.mean():.2%}  ({len(df):,} obligors)")
    print("  PIT_PD = Φ([Φ⁻¹(TTC_PD) − √ρ·Z] / √(1−ρ)),  ρ = 0.12")
    print()

    rows = []
    for name, macro_df in macro.items():
        z = credit_cycle_index(macro_df)
        pit_pd = ttc_to_pit(ttc_pd, z)
        rows.append({
            "conditions":   name,
            "Z":            round(z, 3),
            "avg_TTC_PD_%": round(ttc_pd.mean() * 100, 2),
            "avg_PIT_PD_%": round(pit_pd.mean() * 100, 2),
            "change_bps":   round((pit_pd.mean() - ttc_pd.mean()) * 1e4, 1),
        })
    pit_df = pd.DataFrame(rows)
    print(pit_df.to_string(index=False))
    return ttc_pd, pit_df


# ---------------------------------------------------------------------------
# Step 7 — Macroeconomic Stressed PD
# ---------------------------------------------------------------------------

def step_stressed_pd(df, scorecard, macro, base_pd):
    print("\n" + "=" * 60)
    print("STEP 7: Macroeconomic Stressed PD")
    print("=" * 60)
    print("  multiplier = exp(β_u·ΔU + β_g·ΔGDP + β_h·ΔHPI), shocks vs. Base by quarter")
    print()

    base = macro["Base"]
    rows = []
    for name, macro_df in macro.items():
        d_u   = macro_df["unemployment"].values - base["unemployment"].values
        d_gdp = macro_df["gdp_growth"].values   - base["gdp_growth"].values
        d_hpi = macro_df["hpi_change"].values   - base["hpi_change"].values
        avg_pd = np.array([
            scorecard.stressed_pd(df, du, dg, dh, base_pd=base_pd).mean()
            for du, dg, dh in zip(d_u, d_gdp, d_hpi)
        ])
        q = int(avg_pd.argmax())
        rows.append({
            "scenario":      name,
            "peak_quarter":  macro_df["quarter"].iloc[q],
            "dU_pp":         round(d_u[q], 2),
            "dGDP_pp":       round(d_gdp[q], 2),
            "dHPI_pp":       round(d_hpi[q], 2),
            "peak_avg_PD_%": round(avg_pd[q] * 100, 2),
            "multiplier":    round(avg_pd[q] / base_pd.mean(), 2),
        })
    stress_df = pd.DataFrame(rows)
    print(stress_df.to_string(index=False))

    peaks = stress_df["peak_avg_PD_%"].values
    monotone = bool(np.all(np.diff(peaks) >= 0))
    print(f"\n  Scenario monotonicity (Base ≤ Adverse ≤ Severe): {'PASS' if monotone else 'FAIL'}")
    return stress_df


# ---------------------------------------------------------------------------
# Step 8 — Transition Matrix
# ---------------------------------------------------------------------------

def step_transition_matrix(history):
    print("\n" + "=" * 60)
    print("STEP 8: Transition Matrix Estimation")
    print("=" * 60)
    tm_cohort   = estimate_cohort(history)
    tm_duration = estimate_duration(history)

    print("\n  Cohort Method — 1-Year Transition Matrix (%):")
    print(tm_to_dataframe(tm_cohort).to_string())

    cdrs = cumulative_default_rates(tm_cohort, horizons=[1, 3, 5, 7, 10])
    print("\n  Cumulative Default Rates (%):")
    print(cdrs.to_string())

    benchmark = benchmark_vs_moodys(cdrs)
    print("\n  Benchmark vs Moody's (5Y CDR):")
    print(benchmark.to_string(index=False))

    return tm_cohort, tm_duration, cdrs, benchmark


# ---------------------------------------------------------------------------
# Step 9 — Transition Matrix Validation
# ---------------------------------------------------------------------------

def step_tm_validation(history, tm_cohort, benchmark, validator):
    print("\n" + "=" * 60)
    print("STEP 9: Transition Matrix Validation")
    print("=" * 60)

    mono = check_monotonicity(tm_cohort)
    diag = check_diagonal_dominance(tm_cohort)
    stat = stationarity_test(history)

    print(f"  Monotonicity    : {'PASS' if mono['passed'] else 'FAIL'} — {mono.get('violations', [])}")
    print(f"  Diag dominance  : {'PASS' if diag['passed'] else 'FAIL'} — avg {diag['avg_diagonal']:.4f}")
    print(f"  Stationarity    : {'PASS' if stat['passed'] else 'FAIL'} — p={stat['p_value']}")

    validator.validate_transition_matrix(mono, diag, stat, benchmark)
    return mono, diag, stat


# ---------------------------------------------------------------------------
# Step 10 — SR 11-7 Consolidated Report
# ---------------------------------------------------------------------------

def step_sr117_report(validator):
    print("\n" + "=" * 60)
    print("STEP 10: SR 11-7 Consolidated Validation Report")
    print("=" * 60)
    validator.print_report()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    validator = ModelValidator()

    # Step 1
    df, history, macro = step_data()

    # Step 3 (scorecard must be fit before step 2 can call iv_summary)
    scorecard = step_pd_scorecard(df)

    # Step 2
    iv_df = step_woe_iv(df, scorecard)

    # Step 4
    metrics, psi = step_pd_validation(scorecard, iv_df, validator)

    # Step 5
    challenger = step_challenger(df, validator, champion_auc=metrics["auc"])

    # Step 6
    ttc_pd, pit_df = step_ttc_to_pit(df, scorecard, macro)

    # Step 7
    stress_df = step_stressed_pd(df, scorecard, macro, base_pd=ttc_pd)

    # Step 8
    tm_cohort, tm_duration, cdrs, benchmark = step_transition_matrix(history)

    # Step 9
    mono, diag, stat = step_tm_validation(history, tm_cohort, benchmark, validator)

    # Step 10
    step_sr117_report(validator)

    print("\nPipeline complete.")
