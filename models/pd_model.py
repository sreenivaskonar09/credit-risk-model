"""
PD Scorecard: WoE/IV encoding + Logistic Regression scorecard.
Includes challenger XGBoost model for SR 11-7 benchmarking.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp, chi2_contingency
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Weight of Evidence / Information Value
# ---------------------------------------------------------------------------

class WoEEncoder:
    """
    Bins numeric and categorical features, computes Weight of Evidence (WoE)
    and Information Value (IV) per feature.

    WoE(bin i) = ln[ P(Defaults in bin i) / P(Non-defaults in bin i) ]
    IV = Σ (P_D_i - P_ND_i) × WoE_i
    """

    IV_THRESHOLDS = {
        "Useless": (0.00, 0.02),
        "Weak":    (0.02, 0.10),
        "Medium":  (0.10, 0.30),
        "Strong":  (0.30, float("inf")),
    }

    def __init__(self, n_bins: int = 10, min_bin_size: float = 0.05):
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self._woe_maps: Dict[str, Dict] = {}
        self._iv: Dict[str, float] = {}
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        self.feature_names_ = list(X.columns)
        for col in X.columns:
            if X[col].dtype == "object" or X[col].nunique() <= 10:
                woe_map, iv = self._categorical_woe(X[col], y)
            else:
                woe_map, iv = self._numeric_woe(X[col], y)
            self._woe_maps[col] = woe_map
            self._iv[col] = iv
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in self.feature_names_:
            if col not in X.columns:
                out[col] = 0.0
                continue
            woe_map = self._woe_maps[col]
            if isinstance(list(woe_map.keys())[0], str):
                out[col] = X[col].astype(str).map(woe_map).fillna(0.0)
            else:
                # Interval mapping for numeric
                out[col] = X[col].apply(
                    lambda v: self._lookup_interval(v, woe_map)
                )
        return out.astype(float)

    def iv_summary(self) -> pd.DataFrame:
        rows = []
        for feat, iv in sorted(self._iv.items(), key=lambda x: -x[1]):
            strength = next(
                k for k, (lo, hi) in self.IV_THRESHOLDS.items()
                if lo <= iv < hi
            )
            rows.append({"feature": feat, "IV": round(iv, 4), "strength": strength})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _categorical_woe(
        self, series: pd.Series, y: pd.Series
    ) -> Tuple[Dict, float]:
        total_d = y.sum()
        total_nd = len(y) - total_d
        woe_map = {}
        iv = 0.0
        for val, grp in pd.concat([series.astype(str), y], axis=1).groupby(series.astype(str).name):
            d = grp.iloc[:, 1].sum()
            nd = len(grp) - d
            p_d  = max(d, 0.5) / total_d
            p_nd = max(nd, 0.5) / total_nd
            woe = np.log(p_d / p_nd)
            woe_map[str(val)] = woe
            iv += (p_d - p_nd) * woe
        return woe_map, iv

    def _numeric_woe(
        self, series: pd.Series, y: pd.Series
    ) -> Tuple[Dict, float]:
        total_d = y.sum()
        total_nd = len(y) - total_d
        try:
            bins = pd.qcut(series, q=self.n_bins, duplicates="drop")
        except Exception:
            bins = pd.cut(series, bins=self.n_bins)
        woe_map = {}
        iv = 0.0
        tmp = pd.concat([bins, y], axis=1)
        tmp.columns = ["bin", "target"]
        for interval, grp in tmp.groupby("bin", observed=True):
            d = grp["target"].sum()
            nd = len(grp) - d
            p_d  = max(d, 0.5) / total_d
            p_nd = max(nd, 0.5) / total_nd
            woe = np.log(p_d / p_nd)
            woe_map[interval] = woe
            iv += (p_d - p_nd) * woe
        return woe_map, iv

    def _lookup_interval(self, value, woe_map: Dict):
        for interval, woe in woe_map.items():
            try:
                if value in interval:
                    return woe
            except TypeError:
                pass
        # Fallback: nearest interval midpoint
        return 0.0


# ---------------------------------------------------------------------------
# PD Scorecard
# ---------------------------------------------------------------------------

class PDScorecard:
    """
    Logistic regression PD scorecard trained on WoE-transformed features.

    Steps:
      1. Encode categoricals with LabelEncoder
      2. Fit WoEEncoder on training data
      3. Transform features to WoE space
      4. Fit LogisticRegression
    """

    def __init__(self, feature_cols: List[str], test_size: float = 0.2,
                 seed: int = 42):
        self.feature_cols = feature_cols
        self.test_size = test_size
        self.seed = seed
        self.woe_encoder = WoEEncoder(n_bins=10)
        self.model = LogisticRegression(
            max_iter=1000, C=0.5, solver="lbfgs", random_state=seed
        )
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test:  Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series]    = None
        self.y_test:  Optional[pd.Series]    = None

    def fit(self, df: pd.DataFrame, target: str = "default_flag") -> "PDScorecard":
        X = df[self.feature_cols].copy()
        y = df[target]

        # Encode object columns
        for col in X.select_dtypes("object").columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self._label_encoders[col] = le

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y
        )
        self.X_train, self.X_test = X_tr, X_te
        self.y_train, self.y_test = y_tr, y_te

        # WoE fit on training data
        self.woe_encoder.fit(X_tr, y_tr)
        X_woe = self.woe_encoder.transform(X_tr).astype(float)
        self.model.fit(X_woe, y_tr)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = X.copy()
        for col, le in self._label_encoders.items():
            # Skip columns already label-encoded (e.g. self.X_train / self.X_test)
            if col in X_enc.columns and not pd.api.types.is_numeric_dtype(X_enc[col]):
                X_enc[col] = le.transform(X_enc[col].astype(str))
        X_woe = self.woe_encoder.transform(X_enc).astype(float)
        return self.model.predict_proba(X_woe)[:, 1]

    def validate(self) -> Dict:
        """
        Compute discriminatory and calibration metrics on held-out test set.

        Returns
        -------
        dict with keys: auc, gini, ks, brier_score, hl_pvalue
        """
        X_test_enc = self.X_test.copy()
        for col in X_test_enc.select_dtypes("object").columns:
            le = LabelEncoder()
            X_test_enc[col] = le.fit_transform(X_test_enc[col].astype(str))

        X_woe = self.woe_encoder.transform(X_test_enc)
        X_woe = X_woe.astype(float)  # Critical: ensure float dtype before predict

        y_pred_proba = self.model.predict_proba(X_woe)[:, 1]
        y_true = self.y_test.values

        # AUC / Gini
        auc = roc_auc_score(y_true, y_pred_proba)
        gini = 2 * auc - 1

        # KS statistic
        defaults    = y_pred_proba[y_true == 1]
        non_defaults = y_pred_proba[y_true == 0]
        ks_stat, _ = ks_2samp(defaults, non_defaults)

        # Brier score
        brier = brier_score_loss(y_true, y_pred_proba)

        # Hosmer-Lemeshow test (10 deciles)
        hl_pvalue = self._hosmer_lemeshow(y_true, y_pred_proba)

        return {
            "auc":         round(auc, 4),
            "gini":        round(gini, 4),
            "ks":          round(ks_stat, 4),
            "brier_score": round(brier, 4),
            "hl_pvalue":   round(hl_pvalue, 4),
            "n_test":      len(y_true),
            "default_rate_test": round(y_true.mean(), 4),
        }

    def stressed_pd(
        self, df: pd.DataFrame,
        delta_unemployment: float = 3.0,
        delta_gdp: float = -4.0,
        delta_hpi: float = -10.0,
        base_pd: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Macro-adjusted PD using sensitivity coefficients.
        multiplier = exp(β_u·Δu + β_g·Δg + β_h·Δh)
        Pass base_pd to reuse precomputed scorecard PDs.
        """
        if base_pd is None:
            base_pd = self.predict_proba(df[self.feature_cols])
        beta_u, beta_g, beta_h = 0.12, -0.04, -0.008
        multiplier = np.exp(
            beta_u * delta_unemployment
            + beta_g * delta_gdp
            + beta_h * delta_hpi
        )
        return np.clip(base_pd * multiplier, 0, 1)

    def ttc_pd_table(self) -> pd.DataFrame:
        """Return through-the-cycle PD estimates by rating bucket."""
        ttc = {
            "AAA": 0.0001, "AA": 0.0005, "A":   0.0010,
            "BBB": 0.0030, "BB": 0.0100, "B":   0.0500,
            "CCC": 0.2000,
        }
        rows = []
        for rating, pd_val in ttc.items():
            rows.append({
                "rating":   rating,
                "ttc_pd":   pd_val,
                "ttc_pd_%": f"{pd_val*100:.2f}%",
            })
        return pd.DataFrame(rows)

    def scorecard_coefficients(self) -> pd.DataFrame:
        """Return scorecard coefficients for interpretability / SR 11-7."""
        coefs = self.model.coef_[0]
        rows = []
        for feat, coef in zip(self.feature_cols, coefs):
            rows.append({
                "feature":     feat,
                "coefficient": round(coef, 4),
                "direction":   "increases PD" if coef > 0 else "decreases PD",
            })
        rows.append({
            "feature":     "intercept",
            "coefficient": round(self.model.intercept_[0], 4),
            "direction":   "baseline log-odds",
        })
        return pd.DataFrame(rows)

    def iv_summary(self) -> pd.DataFrame:
        return self.woe_encoder.iv_summary()

    # ------------------------------------------------------------------
    def _hosmer_lemeshow(self, y_true: np.ndarray, y_pred: np.ndarray,
                          n_bins: int = 10) -> float:
        df = pd.DataFrame({"y": y_true, "p": y_pred})
        df["decile"] = pd.qcut(df["p"], q=n_bins, duplicates="drop", labels=False)
        chi2 = 0.0
        for _, grp in df.groupby("decile"):
            n = len(grp)
            obs_d   = grp["y"].sum()
            exp_d   = grp["p"].sum()
            obs_nd  = n - obs_d
            exp_nd  = n - exp_d
            chi2 += (obs_d - exp_d) ** 2 / max(exp_d, 1e-9)
            chi2 += (obs_nd - exp_nd) ** 2 / max(exp_nd, 1e-9)
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=n_bins - 2)
        return p_value


# ---------------------------------------------------------------------------
# TTC-to-PIT Calibration Overlay
# ---------------------------------------------------------------------------

# Long-run U.S. macro anchors (approx. 1990–2023) used to measure the cycle
LONG_RUN_MACRO = {
    "unemployment": {"mean": 5.8, "std": 1.7},   # %
    "gdp_growth":   {"mean": 2.5, "std": 2.0},   # % annualised
    "hpi_change":   {"mean": 4.0, "std": 6.0},   # % YoY
}


def credit_cycle_index(macro_df: pd.DataFrame) -> float:
    """
    Credit-cycle index Z from current macro conditions (Z < 0 = downturn).

    Z = mean of standardised gaps, signed so that higher unemployment and
    lower GDP / HPI growth push Z negative:
        Z = ⅓·[ −(U−Ū)/σ_U + (G−Ḡ)/σ_G + (H−H̄)/σ_H ]
    Uses the average of the first four quarters of the supplied path.
    """
    cur = macro_df.head(4)[list(LONG_RUN_MACRO)].mean()
    gaps = [
        -(cur["unemployment"] - LONG_RUN_MACRO["unemployment"]["mean"]) / LONG_RUN_MACRO["unemployment"]["std"],
        (cur["gdp_growth"] - LONG_RUN_MACRO["gdp_growth"]["mean"]) / LONG_RUN_MACRO["gdp_growth"]["std"],
        (cur["hpi_change"] - LONG_RUN_MACRO["hpi_change"]["mean"]) / LONG_RUN_MACRO["hpi_change"]["std"],
    ]
    return float(np.mean(gaps))


def ttc_to_pit(ttc_pd: np.ndarray, z: float, rho: float = 0.12) -> np.ndarray:
    """
    Convert through-the-cycle PDs to point-in-time PDs with the Basel
    single-factor (Vasicek) model:

        PD_PIT = Φ( [Φ⁻¹(PD_TTC) − √ρ · Z] / √(1 − ρ) )

    ρ = asset correlation (0.12 = Basel corporate floor).
    Z < 0 (downturn) raises PDs; Z > 0 (benign) lowers them.
    """
    from scipy.stats import norm
    p = np.clip(ttc_pd, 1e-6, 1 - 1e-6)
    return norm.cdf((norm.ppf(p) - np.sqrt(rho) * z) / np.sqrt(1 - rho))


# ---------------------------------------------------------------------------
# Challenger Model
# ---------------------------------------------------------------------------

def build_challenger(df: pd.DataFrame, feature_cols: List[str],
                     target: str = "default_flag", seed: int = 42) -> Dict:
    """
    Build XGBoost challenger model for SR 11-7 benchmarking.

    Returns dict with model, AUC, feature importances.
    """
    try:
        from xgboost import XGBClassifier
        import shap
    except ImportError:
        return {"error": "xgboost and shap required"}

    X = df[feature_cols].copy()
    y = df[target]

    # Encode categoricals
    for col in X.select_dtypes("object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_pred)
    gini = 2 * auc - 1

    # SHAP feature importances
    try:
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_te)
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        fi_df = pd.DataFrame({
            "feature":          feature_cols,
            "mean_abs_shap":    mean_abs_shap.round(4),
        }).sort_values("mean_abs_shap", ascending=False)
    except Exception:
        fi_df = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": clf.feature_importances_,
        }).sort_values("mean_abs_shap", ascending=False)

    return {
        "model":               clf,
        "auc":                 round(auc, 4),
        "gini":                round(gini, 4),
        "feature_importances": fi_df,
    }
