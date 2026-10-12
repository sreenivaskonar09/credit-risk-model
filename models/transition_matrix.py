"""
Credit rating transition matrix estimation.
Supports cohort method and duration (generator) method.
Includes monotonicity, diagonal dominance, stationarity tests,
cumulative default rate computation, and Moody's benchmark comparison.
"""

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import chi2
from typing import List, Dict, Tuple

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
N = len(RATINGS)

# Published Moody's 5-year cumulative default rates (illustrative, %)
MOODYS_CDR = {
    "AAA": 0.10, "AA": 0.40, "A": 0.90,
    "BBB": 2.50, "BB": 9.00, "B": 20.00,
    "CCC": 45.00,
}


def estimate_cohort(history: pd.DataFrame) -> np.ndarray:
    """
    Cohort method: count transitions and normalise by row total.

    Parameters
    ----------
    history : pd.DataFrame
        Columns: start_rating, end_rating (year-over-year observations)

    Returns
    -------
    np.ndarray
        8×8 row-stochastic transition matrix.
    """
    count = np.zeros((N, N))
    for _, row in history.iterrows():
        i = RATINGS.index(row["start_rating"])
        j = RATINGS.index(row["end_rating"])
        count[i, j] += 1

    # Normalise; rows with zero counts keep diagonal = 1
    tm = np.zeros((N, N))
    for i in range(N):
        row_sum = count[i].sum()
        if row_sum == 0:
            tm[i, i] = 1.0
        else:
            tm[i] = count[i] / row_sum
    return tm


def estimate_duration(history: pd.DataFrame) -> np.ndarray:
    """
    Duration (generator / intensity) method.
    Estimates generator matrix Q from transition counts and sojourn times,
    then computes 1-year TM = expm(Q).

    Parameters
    ----------
    history : pd.DataFrame
        Columns: start_rating, end_rating, year

    Returns
    -------
    np.ndarray
        8×8 row-stochastic transition matrix via matrix exponentiation.
    """
    # Approximate: use cohort count matrix as basis for Q
    count = np.zeros((N, N))
    sojourn = np.zeros(N)    # total time spent in each state

    for _, row in history.iterrows():
        i = RATINGS.index(row["start_rating"])
        j = RATINGS.index(row["end_rating"])
        count[i, j] += 1
        sojourn[i] += 1.0   # each observation = 1 year

    # Build generator: Q[i,j] = count[i,j] / sojourn[i] for i≠j
    # Q[i,i] = -Σ_{j≠i} Q[i,j]
    Q = np.zeros((N, N))
    for i in range(N - 1):   # D is absorbing, skip
        if sojourn[i] == 0:
            continue
        for j in range(N):
            if i != j:
                Q[i, j] = count[i, j] / sojourn[i]
        Q[i, i] = -Q[i, :].sum()

    # 1-year transition matrix
    tm = expm(Q)
    # Enforce non-negativity and row-stochastic (numerical fix)
    tm = np.clip(tm, 0, 1)
    tm = tm / tm.sum(axis=1, keepdims=True)
    return tm


def check_monotonicity(tm: np.ndarray) -> Dict:
    """
    Check that the default column (column 7) is monotonically non-decreasing
    from AAA to CCC (rows 0–6).

    Returns
    -------
    dict with keys: passed (bool), violations (list), default_probs (list)
    """
    default_col = tm[:N-1, N-1]   # AAA through CCC default prob
    violations = []
    for i in range(len(default_col) - 1):
        if default_col[i] > default_col[i + 1] + 1e-6:
            violations.append(
                f"{RATINGS[i]} ({default_col[i]:.4f}) > "
                f"{RATINGS[i+1]} ({default_col[i+1]:.4f})"
            )
    return {
        "passed":        len(violations) == 0,
        "violations":    violations,
        "default_probs": {RATINGS[i]: round(default_col[i], 6) for i in range(len(default_col))},
    }


def check_diagonal_dominance(tm: np.ndarray, threshold: float = 0.85) -> Dict:
    """
    Check that the average diagonal (stay rate) is at or above threshold.

    Returns
    -------
    dict with keys: passed (bool), avg_diagonal (float),
                    diagonal_by_rating (dict), threshold (float)
    """
    diag = np.diag(tm)[:N-1]
    avg_diag = diag.mean()
    return {
        "passed":              avg_diag >= threshold,
        "avg_diagonal":        round(avg_diag, 4),
        "diagonal_by_rating":  {RATINGS[i]: round(diag[i], 4) for i in range(len(diag))},
        "threshold":           threshold,
    }


def stationarity_test(history: pd.DataFrame) -> Dict:
    """
    Test stationarity of the transition matrix across two sub-periods
    using a chi-square test of homogeneity on transition counts
    (one 2×K contingency table per starting rating, summed).

    Returns
    -------
    dict with keys: chi2_stat, p_value, passed (p > 0.05), periods
    """
    years = sorted(history["year"].unique())
    mid = years[len(years) // 2]
    period1 = history[history["year"] <= mid]
    period2 = history[history["year"] > mid]

    def _counts(h: pd.DataFrame) -> np.ndarray:
        c = np.zeros((N, N))
        for s, e in zip(h["start_rating"], h["end_rating"]):
            c[RATINGS.index(s), RATINGS.index(e)] += 1
        return c

    c1, c2 = _counts(period1), _counts(period2)

    chi2_stat, dof = 0.0, 0
    for i in range(N - 1):            # D is absorbing
        table = np.vstack([c1[i], c2[i]])
        table = table[:, table.sum(axis=0) > 0]   # drop unobserved destinations
        if table.shape[1] < 2 or (table.sum(axis=1) == 0).any():
            continue
        expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0) / table.sum()
        chi2_stat += ((table - expected) ** 2 / expected).sum()
        dof += table.shape[1] - 1

    p_value = 1 - chi2.cdf(chi2_stat, df=dof) if dof else 1.0

    return {
        "chi2_stat": round(chi2_stat, 4),
        "p_value":   round(p_value, 4),
        "passed":    p_value > 0.05,
        "periods":   [f"Y1–Y{mid}", f"Y{mid+1}–Y{years[-1]}"],
    }


def cumulative_default_rates(
    tm: np.ndarray,
    horizons: List[int] = [1, 3, 5, 7, 10]
) -> pd.DataFrame:
    """
    Compute multi-year cumulative default rates via matrix exponentiation.
    CDR(h, grade) = (TM^h)[grade, D]

    Parameters
    ----------
    tm : np.ndarray
        1-year transition matrix.
    horizons : list of int
        Forecast horizons in years.

    Returns
    -------
    pd.DataFrame
        Rows = non-default ratings, columns = horizon years (as %).
    """
    results = {}
    tm_h = np.eye(N)
    prev_h = 0
    for h in sorted(horizons):
        steps = h - prev_h
        for _ in range(steps):
            tm_h = tm_h @ tm
        results[f"{h}Y"] = {
            RATINGS[i]: round(tm_h[i, N - 1] * 100, 4)
            for i in range(N - 1)
        }
        prev_h = h

    return pd.DataFrame(results)


def benchmark_vs_moodys(internal_cdr: pd.DataFrame) -> pd.DataFrame:
    """
    Compare internal 5-year CDRs against Moody's published benchmarks.

    Parameters
    ----------
    internal_cdr : pd.DataFrame
        Output of cumulative_default_rates(); must contain "5Y" column.

    Returns
    -------
    pd.DataFrame
        Rating, Internal 5Y CDR, Moody's CDR, Deviation %.
    """
    rows = []
    for rating, moody_val in MOODYS_CDR.items():
        internal_val = internal_cdr.loc[rating, "5Y"] if rating in internal_cdr.index else np.nan
        deviation = (internal_val - moody_val) / moody_val * 100 if moody_val else np.nan
        rows.append({
            "rating":           rating,
            "internal_5Y_CDR":  internal_val,
            "moodys_5Y_CDR":    moody_val,
            "deviation_%":      round(deviation, 1) if not np.isnan(deviation) else np.nan,
        })
    return pd.DataFrame(rows)


def tm_to_dataframe(tm: np.ndarray) -> pd.DataFrame:
    """Return transition matrix as a labelled DataFrame (percentages)."""
    df = pd.DataFrame(
        (tm * 100).round(3),
        index=RATINGS,
        columns=RATINGS,
    )
    return df
