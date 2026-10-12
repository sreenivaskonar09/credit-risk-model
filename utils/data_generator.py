"""
Synthetic data generator for credit risk modeling project.
Generates obligor data, rating histories, and macro scenarios.
"""

import numpy as np
import pandas as pd
from typing import Optional


RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
RATING_DEFAULTS = {
    "AAA": 0.0001, "AA": 0.0005, "A": 0.001,
    "BBB": 0.003,  "BB": 0.01,   "B": 0.05,
    "CCC": 0.20,   "D": 1.0
}


def generate_obligor_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic obligor-level data for PD scorecard modeling.

    Parameters
    ----------
    n : int
        Number of obligors to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with obligor features and default flag.
    """
    rng = np.random.default_rng(seed)

    # Rating distribution (skewed toward investment grade). Performing obligors
    # only: already-defaulted (D) names are excluded from a PD development sample.
    rating_probs = np.array([0.05, 0.12, 0.18, 0.25, 0.20, 0.12, 0.06])
    ratings = rng.choice(RATINGS[:-1], size=n, p=rating_probs / rating_probs.sum())

    # Numeric features correlated with rating
    rating_idx = np.array([RATINGS.index(r) for r in ratings])  # 0=AAA ... 6=CCC
    scaled = rating_idx / 7.0  # 0 to 1

    debt_to_income = np.clip(scaled * 0.8 + rng.normal(0, 0.10, n), 0.05, 1.5)
    payment_history = np.clip(1 - scaled * 0.9 + rng.normal(0, 0.05, n), 0, 1)
    credit_utilization = np.clip(scaled * 0.85 + rng.normal(0, 0.08, n), 0, 1)
    years_in_bureau = np.clip(
        rng.normal(10, 5, n) - rating_idx * 0.5, 0.5, 40
    ).astype(int)
    num_inquiries = np.clip(
        rng.poisson(2, n) + rating_idx, 0, 20
    ).astype(int)
    loan_to_value = np.clip(scaled * 0.7 + rng.normal(0, 0.12, n), 0.05, 1.2)

    industries = ["Finance", "Technology", "Healthcare", "Energy",
                  "Consumer", "Industrials", "Real Estate", "Utilities"]
    industry = rng.choice(industries, size=n)

    # Default flag driven by rating-based PD with noise
    base_pd = np.array([RATING_DEFAULTS[r] for r in ratings])
    noise = rng.uniform(0.8, 1.2, n)
    true_pd = np.clip(base_pd * noise, 0, 1)
    default_flag = (rng.uniform(0, 1, n) < true_pd).astype(int)

    df = pd.DataFrame({
        "obligor_id": [f"OBL{i:05d}" for i in range(n)],
        "rating": ratings,
        "debt_to_income": debt_to_income.round(4),
        "payment_history": payment_history.round(4),
        "credit_utilization": credit_utilization.round(4),
        "years_in_bureau": years_in_bureau,
        "num_inquiries": num_inquiries,
        "loan_to_value": loan_to_value.round(4),
        "industry": industry,
        "default_flag": default_flag,
    })

    return df


def generate_rating_history(
    n_obligors: int = 1000,
    n_years: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic rating transition history for transition matrix estimation.

    Parameters
    ----------
    n_obligors : int
        Number of obligors to track.
    n_years : int
        Number of years of history.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame: obligor_id, start_rating, end_rating, year.
    """
    rng = np.random.default_rng(seed)

    # Transition probability matrix (simplified, row-stochastic)
    # Rows: from_rating, Cols: to_rating (AAA AA A BBB BB B CCC D)
    TM_PRIOR = np.array([
        # AAA   AA      A     BBB    BB     B    CCC     D
        [0.920, 0.065, 0.010, 0.004, 0.001, 0.000, 0.000, 0.000],  # AAA
        [0.010, 0.900, 0.072, 0.013, 0.003, 0.001, 0.001, 0.000],  # AA
        [0.001, 0.025, 0.895, 0.060, 0.012, 0.004, 0.002, 0.001],  # A
        [0.000, 0.003, 0.040, 0.880, 0.055, 0.015, 0.004, 0.003],  # BBB
        [0.000, 0.001, 0.007, 0.060, 0.840, 0.065, 0.017, 0.010],  # BB
        [0.000, 0.001, 0.003, 0.010, 0.080, 0.810, 0.046, 0.050],  # B
        [0.000, 0.000, 0.002, 0.005, 0.020, 0.120, 0.653, 0.200],  # CCC
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],  # D (absorbing)
    ])

    records = []
    rating_idx = [0, 1, 2, 3, 4, 5, 6, 7]

    # Initial rating distribution for obligors
    init_probs = [0.05, 0.12, 0.20, 0.28, 0.18, 0.10, 0.05, 0.02]
    current_ratings = rng.choice(rating_idx, size=n_obligors, p=init_probs)

    for year in range(1, n_years + 1):
        for i in range(n_obligors):
            from_r = current_ratings[i]
            if from_r == 7:  # D is absorbing
                continue
            to_r = rng.choice(rating_idx, p=TM_PRIOR[from_r])
            records.append({
                "obligor_id": f"OBL{i:04d}",
                "year": year,
                "start_rating": RATINGS[from_r],
                "end_rating": RATINGS[to_r],
            })
            current_ratings[i] = to_r

    return pd.DataFrame(records)


def generate_macro_scenarios() -> dict:
    """
    Generate quarterly macro-economic paths for Base, Adverse, and Severely Adverse
    stress scenarios (9 quarters, Q1 2024 – Q1 2026).

    Returns
    -------
    dict
        Keys: 'Base', 'Adverse', 'Severely Adverse'.
        Values: pd.DataFrame with columns [quarter, unemployment, gdp_growth,
                hpi_change, credit_spread].
    """
    quarters = [
        "2024Q1", "2024Q2", "2024Q3", "2024Q4",
        "2025Q1", "2025Q2", "2025Q3", "2025Q4",
        "2026Q1",
    ]
    n = len(quarters)

    scenarios = {
        "Base": pd.DataFrame({
            "quarter": quarters,
            "unemployment": [4.0, 4.1, 4.1, 4.2, 4.2, 4.1, 4.0, 3.9, 3.9],
            "gdp_growth": [2.5, 2.4, 2.3, 2.4, 2.5, 2.6, 2.5, 2.5, 2.6],
            "hpi_change": [3.0, 2.8, 2.5, 2.4, 2.5, 2.7, 2.8, 3.0, 3.1],
            "credit_spread": [1.50, 1.55, 1.55, 1.60, 1.58, 1.55, 1.52, 1.50, 1.48],
        }),
        "Adverse": pd.DataFrame({
            "quarter": quarters,
            "unemployment": [5.5, 6.5, 7.2, 7.8, 8.0, 7.8, 7.5, 7.0, 6.8],
            "gdp_growth": [0.5, -1.0, -2.0, -1.5, -0.5, 0.5, 1.0, 1.5, 2.0],
            "hpi_change": [-2.0, -5.0, -7.0, -6.0, -4.0, -2.0, -0.5, 1.0, 2.0],
            "credit_spread": [2.50, 3.20, 3.80, 4.00, 3.80, 3.50, 3.20, 2.90, 2.70],
        }),
        "Severely Adverse": pd.DataFrame({
            "quarter": quarters,
            "unemployment": [7.0, 8.5, 10.0, 11.0, 11.5, 11.0, 10.5, 10.0, 9.5],
            "gdp_growth": [-1.0, -4.0, -6.0, -5.0, -2.5, -0.5, 1.0, 2.0, 2.5],
            "hpi_change": [-5.0, -10.0, -15.0, -14.0, -10.0, -6.0, -3.0, 0.0, 1.5],
            "credit_spread": [4.00, 5.50, 6.50, 7.00, 6.50, 5.80, 5.00, 4.20, 3.80],
        }),
    }
    return scenarios
