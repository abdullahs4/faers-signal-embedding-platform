"""
Core pharmacovigilance disproportionality statistics.

Implements the formulas from the FAERS Signal Embedding Platform spec
(sections 8, 9, 10, 11, 12, 13, 14):

  - ROR / PRR / reporting fraction from a 2x2 contingency table
  - log-scale 95% confidence interval for ROR
  - temporal features (persistence, trend, volatility, recent acceleration)
  - comparator robustness
  - therapeutic-class anomaly
  - demographic heterogeneity
  - signal strength / uncertainty

These are pure functions with no I/O so they can be unit-tested and reused
by both the one-time batch build (db_build.py) and the live/interactive
pairwise comparator in the dashboard (dashboard.py).
"""

from __future__ import annotations

from math import exp, log, sqrt
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 2x2 contingency table statistics (spec section 8)
# ---------------------------------------------------------------------------

def apply_continuity(a: float, b: float, c: float, d: float, continuity: float = 0.5):
    """Apply a continuity correction only when at least one cell is zero,
    exactly as specified in the document's ror_from_counts reference code."""
    if min(a, b, c, d) == 0:
        return a + continuity, b + continuity, c + continuity, d + continuity
    return a, b, c, d


def ror_from_counts(a: float, b: float, c: float, d: float, continuity: float = 0.5):
    """Reporting Odds Ratio and its log-scale 95% CI.

    ROR = (a*d) / (b*c)
    log(ROR) +/- 1.96 * sqrt(1/a + 1/b + 1/c + 1/d)
    """
    a, b, c, d = apply_continuity(a, b, c, d, continuity)
    if b == 0 or c == 0:
        return float("nan"), float("nan"), float("nan")
    ror = (a * d) / (b * c)
    se = sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo = exp(log(ror) - 1.96 * se)
    hi = exp(log(ror) + 1.96 * se)
    return ror, lo, hi


def se_log_ror(a: float, b: float, c: float, d: float, continuity: float = 0.5) -> float:
    a, b, c, d = apply_continuity(a, b, c, d, continuity)
    return sqrt(1 / a + 1 / b + 1 / c + 1 / d)


def prr_from_counts(a: float, b: float, c: float, d: float, continuity: float = 0.5) -> float:
    """Proportional Reporting Ratio: [a/(a+b)] / [c/(c+d)]."""
    a, b, c, d = apply_continuity(a, b, c, d, continuity)
    if (a + b) == 0 or (c + d) == 0 or c == 0:
        return float("nan")
    return (a / (a + b)) / (c / (c + d))


def rf_from_counts(a: float, b: float) -> float:
    """Reporting fraction: a / (a + b)."""
    if (a + b) == 0:
        return float("nan")
    return a / (a + b)


def signal_strength(ror: float, se_log: float) -> float:
    """S_raw = log(ROR) / (1 + SE_logROR)  (spec section 14.1)."""
    if ror is None or np.isnan(ror) or ror <= 0:
        return float("nan")
    return log(ror) / (1 + se_log)


def is_signal(ror_low: float, n_reports: float, min_count: int = 3, threshold: float = 1.0) -> bool:
    """Minimum eligibility + signal rule used throughout the spec:
    N >= min_count and the lower CI bound exceeds the threshold (default 1.0)."""
    if ror_low is None or np.isnan(ror_low):
        return False
    return (n_reports >= min_count) and (ror_low > threshold)


# ---------------------------------------------------------------------------
# Temporal features (spec section 9.1)
# ---------------------------------------------------------------------------

def temporal_features(year_df: pd.DataFrame, signal_threshold: float = 1.0, min_count: int = 3) -> dict:
    """year_df must have columns: year, ror, ror_low, n_reports.

    Returns persistence, trend, volatility, recent_acceleration -- following
    the reference implementation in the spec (section 9.1) as closely as
    possible.
    """
    x = year_df.loc[year_df["n_reports"] >= min_count].copy()
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["ror"])
    x = x.sort_values("year")

    if len(x) < 3:
        return {
            "persistence": float("nan"),
            "trend": float("nan"),
            "volatility": float("nan"),
            "recent_acceleration": float("nan"),
        }

    from sklearn.linear_model import LinearRegression

    x["log_ror"] = np.log(x["ror"])
    persistence = float((x["ror_low"] > signal_threshold).mean())

    model = LinearRegression().fit(x[["year"]], x["log_ror"])
    trend = float(model.coef_[0])
    volatility = float(x["log_ror"].std(ddof=1)) if len(x) > 1 else float("nan")

    split = max(2, len(x) // 2)
    early = x.iloc[:-split] if len(x) > split else x.iloc[:0]
    recent = x.iloc[-split:]

    def slope(z):
        if len(z) < 2:
            return float("nan")
        m = LinearRegression().fit(z[["year"]], z["log_ror"])
        return float(m.coef_[0])

    acceleration = slope(recent) - slope(early) if len(early) >= 2 else float("nan")

    return {
        "persistence": persistence,
        "trend": trend,
        "volatility": volatility,
        "recent_acceleration": acceleration,
    }


# ---------------------------------------------------------------------------
# Comparator robustness (spec section 11)
# ---------------------------------------------------------------------------

def comparator_robustness(ror_lows: list[float]) -> float:
    """R = (# valid comparators with positive lower CI bound) / (# valid comparators)."""
    valid = [x for x in ror_lows if x is not None and not np.isnan(x)]
    if not valid:
        return float("nan")
    return sum(1 for x in valid if x > 1) / len(valid)


# ---------------------------------------------------------------------------
# Therapeutic-class anomaly (spec section 12)
# ---------------------------------------------------------------------------

def class_anomaly(log_ror_drug: float, class_log_rors: list[float]) -> float:
    """C_AO = (log(ROR_AO) - mean_class{log(ROR_O)}) / SD_class{log(ROR_O)}."""
    peers = [x for x in class_log_rors if x is not None and not np.isnan(x)]
    if len(peers) < 2 or log_ror_drug is None or np.isnan(log_ror_drug):
        return float("nan")
    mean = float(np.mean(peers))
    sd = float(np.std(peers, ddof=1))
    if sd == 0:
        return float("nan")
    return (log_ror_drug - mean) / sd


# ---------------------------------------------------------------------------
# Demographic heterogeneity (spec section 10)
# ---------------------------------------------------------------------------

def demographic_heterogeneity(log_rors: list[float]) -> float:
    """H = SD{log(ROR_male), log(ROR_female), log(ROR_<65), log(ROR_>=65)}."""
    valid = [x for x in log_rors if x is not None and not np.isnan(x)]
    if len(valid) < 2:
        return float("nan")
    return float(np.std(valid, ddof=1))


def percentile_score(s: pd.Series) -> pd.Series:
    """DisplayScore_j = 100 * empirical_percentile(feature_j)  (spec section 19.1)."""
    return s.rank(pct=True) * 100
