"""
SR 11-7 Model Validation Framework.

Implements the three pillars of SR 11-7:
  1. Conceptual Soundness
  2. Ongoing Monitoring
  3. Outcomes Analysis

Finding severity levels: None → Observation → Minor → Moderate → High
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Severity Levels
# ---------------------------------------------------------------------------

SEVERITY_ORDER = ["None", "Observation", "Minor", "Moderate", "High"]


def _severity_rank(s: str) -> int:
    return SEVERITY_ORDER.index(s) if s in SEVERITY_ORDER else 0


# ---------------------------------------------------------------------------
# Findings Log
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    component:   str
    description: str
    severity:    str
    remediation: str


class FindingsLog:
    """Accumulates model validation findings."""

    def __init__(self):
        self._findings: List[Finding] = []

    def add(
        self,
        component:   str,
        description: str,
        severity:    str,
        remediation: str,
    ) -> None:
        if severity not in SEVERITY_ORDER or severity == "None":
            return
        self._findings.append(Finding(component, description, severity, remediation))

    def to_dataframe(self) -> pd.DataFrame:
        if not self._findings:
            return pd.DataFrame(
                columns=["component", "description", "severity", "remediation"]
            )
        return pd.DataFrame([
            {
                "component":   f.component,
                "description": f.description,
                "severity":    f.severity,
                "remediation": f.remediation,
            }
            for f in self._findings
        ])

    def summary(self) -> Dict[str, int]:
        counts = {s: 0 for s in SEVERITY_ORDER if s != "None"}
        for f in self._findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts

    def highest_severity(self) -> str:
        if not self._findings:
            return "None"
        return max(self._findings, key=lambda f: _severity_rank(f.severity)).severity


# ---------------------------------------------------------------------------
# IV Assessment
# ---------------------------------------------------------------------------

IV_THRESHOLDS = {
    "Useless": (0.00, 0.02),
    "Weak":    (0.02, 0.10),
    "Medium":  (0.10, 0.30),
    "Strong":  (0.30, float("inf")),
}


def assess_iv(iv_dict: Dict[str, float]) -> pd.DataFrame:
    """
    Assess Information Value for each feature and give a recommendation.

    Parameters
    ----------
    iv_dict : dict
        {feature_name: IV_value}

    Returns
    -------
    pd.DataFrame
        feature | IV | strength | recommendation
    """
    rows = []
    for feat, iv in sorted(iv_dict.items(), key=lambda x: -x[1]):
        strength = next(
            k for k, (lo, hi) in IV_THRESHOLDS.items() if lo <= iv < hi
        )
        if strength == "Useless":
            rec = "Remove from model — no predictive power"
        elif strength == "Weak":
            rec = "Consider removing or combining with other features"
        elif strength == "Medium":
            rec = "Retain — acceptable predictive power"
        else:
            rec = "Retain — strong predictor; check for data leakage"
        rows.append({
            "feature":        feat,
            "IV":             round(iv, 4),
            "strength":       strength,
            "recommendation": rec,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def compute_psi(
    expected: np.ndarray,
    actual:   np.ndarray,
    bins:     int = 10,
) -> float:
    """
    Population Stability Index (PSI).

    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
    Thresholds: <0.10 stable | 0.10–0.25 minor shift | >0.25 major shift

    Parameters
    ----------
    expected : array-like
        Score distribution at development time.
    actual : array-like
        Score distribution at monitoring time.
    bins : int
        Number of quantile bins.

    Returns
    -------
    float
        PSI value.
    """
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] = np.inf

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual,   bins=breakpoints)[0]

    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)

    # Avoid log(0) — replace zeros with small value
    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(psi, 4))


# ---------------------------------------------------------------------------
# Model Validator
# ---------------------------------------------------------------------------

class ModelValidator:
    """
    SR 11-7 Model Validation across three pillars:
      1. Conceptual Soundness  (WoE/IV, model selection, challenger)
      2. Ongoing Monitoring    (PSI, performance drift)
      3. Outcomes Analysis     (Gini, KS, AUC, Brier, HL, TM checks)
    """

    def __init__(self):
        self.log = FindingsLog()
        self._pd_metrics:  Optional[Dict] = None
        self._tm_checks:   Optional[Dict] = None

    # ------------------------------------------------------------------
    # Pillar 3 — PD Model Outcomes Analysis
    # ------------------------------------------------------------------

    def validate_pd_model(
        self,
        metrics:  Dict,
        iv_dict:  Dict[str, float],
        psi:      Optional[float] = None,
        challenger_auc: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Validate PD scorecard against SR 11-7 quantitative benchmarks.

        Parameters
        ----------
        metrics : dict
            Output of PDScorecard.validate().
        iv_dict : dict
            {feature: IV} from WoEEncoder.iv_summary().
        psi : float, optional
            Population Stability Index for score distribution.
        challenger_auc : float, optional
            AUC of the XGBoost challenger model.

        Returns
        -------
        pd.DataFrame
            Summary of findings from this pillar.
        """
        self._pd_metrics = metrics

        # --- Discriminatory Power ---
        gini = metrics.get("gini", 0)
        if gini < 0.20:
            self.log.add(
                "PD Model — Gini",
                f"Gini coefficient {gini:.4f} is below 0.20 threshold",
                "High",
                "Revisit feature selection; consider non-linear models",
            )
        elif gini < 0.40:
            self.log.add(
                "PD Model — Gini",
                f"Gini coefficient {gini:.4f} is below recommended 0.40",
                "Moderate",
                "Investigate additional predictive features; review WoE binning",
            )

        ks = metrics.get("ks", 0)
        if ks < 0.20:
            self.log.add(
                "PD Model — KS",
                f"KS statistic {ks:.4f} below 0.20 floor",
                "High",
                "Model has poor separation; review variable selection",
            )
        elif ks < 0.30:
            self.log.add(
                "PD Model — KS",
                f"KS statistic {ks:.4f} below recommended 0.30",
                "Minor",
                "Monitor over time; consider additional feature engineering",
            )

        # --- Calibration ---
        hl_p = metrics.get("hl_pvalue", 1.0)
        if hl_p < 0.01:
            self.log.add(
                "PD Model — Hosmer-Lemeshow",
                f"H-L p-value {hl_p:.4f} indicates poor calibration (p < 0.01)",
                "Moderate",
                "Low expected default counts in low-PD deciles inflate H-L; apply "
                "Pluto-Tasche upper-bound PDs to low-default grades, then Platt / "
                "isotonic recalibration if miscalibration persists",
            )
        elif hl_p < 0.05:
            self.log.add(
                "PD Model — Hosmer-Lemeshow",
                f"H-L p-value {hl_p:.4f} marginal calibration (0.01 < p < 0.05)",
                "Minor",
                "Monitor calibration; consider recalibration if drift continues",
            )

        # --- Predictive Power (IV) ---
        useless_feats = [f for f, iv in iv_dict.items() if iv < 0.02]
        if useless_feats:
            self.log.add(
                "PD Model — IV Assessment",
                f"Features with IV < 0.02 (useless): {useless_feats}",
                "Minor",
                "Remove or re-engineer useless features",
            )

        # --- Population Stability ---
        if psi is not None:
            if psi > 0.25:
                self.log.add(
                    "PD Model — PSI",
                    f"PSI {psi:.4f} indicates major population shift (> 0.25)",
                    "High",
                    "Immediate model recalibration / redevelopment required",
                )
            elif psi > 0.10:
                self.log.add(
                    "PD Model — PSI",
                    f"PSI {psi:.4f} indicates minor population shift (0.10–0.25)",
                    "Observation",
                    "Increase monitoring frequency; trigger if PSI > 0.25",
                )

        # --- Challenger Benchmark (SR 11-7 Pillar 1: Conceptual Soundness) ---
        if challenger_auc is not None:
            self.validate_challenger(metrics.get("auc", 0), challenger_auc)

        return self.log.to_dataframe()

    def validate_challenger(self, champion_auc: float, challenger_auc: float) -> None:
        """Raise a finding if the challenger materially outperforms the champion."""
        gap = challenger_auc - champion_auc
        if gap > 0.05:
            self.log.add(
                "PD Model — Challenger",
                f"XGBoost challenger AUC ({challenger_auc:.4f}) exceeds "
                f"champion by {gap:.4f} (> 0.05 threshold)",
                "Moderate",
                "Initiate model replacement process for challenger model",
            )

    # ------------------------------------------------------------------
    # Pillar 1/3 — Transition Matrix Validation
    # ------------------------------------------------------------------

    def validate_transition_matrix(
        self,
        mono_result:  Dict,
        diag_result:  Dict,
        stat_result:  Dict,
        benchmark_df: pd.DataFrame,
    ) -> None:
        """Raise findings for transition matrix checks."""
        self._tm_checks = {
            "monotonicity": mono_result,
            "diagonal":     diag_result,
            "stationarity": stat_result,
        }

        if not mono_result["passed"]:
            self.log.add(
                "Transition Matrix — Monotonicity",
                f"Default column not monotone. Violations: {mono_result['violations']}",
                "Moderate",
                "Review data quality; check for vintage effects or rating inflation",
            )

        if not diag_result["passed"]:
            avg_d = diag_result["avg_diagonal"]
            self.log.add(
                "Transition Matrix — Diagonal Dominance",
                f"Avg diagonal {avg_d:.4f} below 85% threshold",
                "Minor",
                "Investigate data aggregation methodology; consider monthly cohorts",
            )

        if not stat_result["passed"]:
            self.log.add(
                "Transition Matrix — Stationarity",
                f"Chi-square p-value {stat_result['p_value']:.4f} rejects stationarity",
                "Minor",
                "Adopt non-stationary TM or time-varying generator approach",
            )

        # Benchmark deviations
        for _, row in benchmark_df.iterrows():
            dev = row.get("deviation_%", 0)
            if pd.isna(dev):
                continue
            if abs(dev) > 100:
                self.log.add(
                    f"Transition Matrix — Moody's Benchmark ({row['rating']})",
                    f"5Y CDR deviates {dev:.1f}% from Moody's published rate",
                    "Moderate",
                    "Investigate data limitations; consider overlaying Moody's CDRs",
                )
            elif abs(dev) > 50:
                self.log.add(
                    f"Transition Matrix — Moody's Benchmark ({row['rating']})",
                    f"5Y CDR deviates {dev:.1f}% from Moody's published rate",
                    "Minor",
                    "Document and monitor; add to model limitations",
                )

    # ------------------------------------------------------------------
    # Report Generation
    # ------------------------------------------------------------------

    def generate_report(self) -> Dict:
        """
        Generate the consolidated SR 11-7 validation opinion.

        Overall opinion logic:
          - Any High finding      → Conditional Fail
          - Any Moderate finding  → Conditional Pass
          - Only Minor/Observation → Pass with Observations
          - No findings           → Pass
        """
        findings_df = self.log.to_dataframe()
        summary     = self.log.summary()
        highest     = self.log.highest_severity()

        if highest == "High":
            opinion = "Conditional Fail"
        elif highest == "Moderate":
            opinion = "Conditional Pass"
        elif highest in ("Minor", "Observation"):
            opinion = "Pass with Observations"
        else:
            opinion = "Pass"

        return {
            "overall_opinion":   opinion,
            "highest_severity":  highest,
            "finding_counts":    summary,
            "findings_table":    findings_df,
            "total_findings":    len(findings_df),
        }

    def print_report(self) -> None:
        """Print formatted SR 11-7 validation report to console."""
        report = self.generate_report()
        print("\n" + "=" * 70)
        print("SR 11-7 MODEL VALIDATION REPORT")
        print("=" * 70)
        print(f"  Overall Opinion   : {report['overall_opinion']}")
        print(f"  Highest Severity  : {report['highest_severity']}")
        print(f"  Total Findings    : {report['total_findings']}")
        print()
        counts = report["finding_counts"]
        for severity in ["High", "Moderate", "Minor", "Observation"]:
            n = counts.get(severity, 0)
            if n > 0:
                print(f"  {severity:<15}: {n}")
        print()
        if not report["findings_table"].empty:
            print("FINDINGS DETAIL")
            print("-" * 70)
            for _, row in report["findings_table"].iterrows():
                print(f"  [{row['severity']}] {row['component']}")
                print(f"    Issue      : {row['description']}")
                print(f"    Remediation: {row['remediation']}")
                print()
        print("=" * 70)
