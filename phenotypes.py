"""
Analyst-facing layer on top of the eight-dimensional signal embedding:

  1. DEFAULT_PAIRWISE  -- a recommended pairwise comparator (spec C3) for
     every medication, chosen on pharmacological grounds (closest
     therapeutic alternative, or the pair that isolates one ingredient).
  2. Display scores    -- 0-100 empirical percentile per dimension
     (spec section 19.1).
  3. Signal phenotypes -- interpretable, mutually exclusive categories with
     thresholds calibrated on the empirical distribution of the computed
     features (the "signal profile -> suggested analytic action" table in
     spec section 25 / the illustrative cluster table in section 16.2).
     These are rule-based and transparent; they are the human-readable
     baseline that HDBSCAN clusters will later be compared against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Recommended pairwise comparator per medication
# ---------------------------------------------------------------------------
# (comparator, rationale). Strength notes: "isolates" pairs differ by one
# ingredient/isomer and are the cleanest; "same-class" pairs are standard
# active comparators; "cross-class" pairs share an indication only and are
# weaker -- flagged so the analyst knows.
DEFAULT_PAIRWISE: dict[str, tuple[str, str]] = {
    # penicillins / beta-lactams
    "amoxicillin": ("clavulanate / amoxicillin", "isolates the clavulanate component (classic for cholestatic hepatitis)"),
    "clavulanate / amoxicillin": ("amoxicillin", "isolates the clavulanate component"),
    "ampicillin": ("amoxicillin", "same-class aminopenicillin"),
    "cloxacillin": ("cephalexin", "alternative anti-staphylococcal agent for skin/soft-tissue infection"),
    "penicillin G": ("penicillin V", "same molecule, parenteral vs oral"),
    "penicillin V": ("amoxicillin", "oral alternative for streptococcal pharyngitis"),
    "cephalexin": ("amoxicillin", "oral beta-lactam alternative with overlapping indications"),
    # macrolide
    "azithromycin": ("amoxicillin", "cross-class: first-line alternatives for community respiratory infection (QT signal contrast)"),
    # benzodiazepines
    "lorazepam": ("diazepam", "same class"),
    "diazepam": ("lorazepam", "same class"),
    # urinary anti-infectives / sulfonamide
    "sulfamethoxazole / trimethoprim": ("nitrofurantoin", "cross-class: uncomplicated UTI alternatives"),
    "nitrofurantoin": ("sulfamethoxazole / trimethoprim", "cross-class: uncomplicated UTI alternatives"),
    "fosfomycin": ("nitrofurantoin", "cross-class: uncomplicated UTI alternatives"),
    "methenamine": ("nitrofurantoin", "cross-class: UTI prophylaxis alternatives"),
    # statins
    "atorvastatin": ("rosuvastatin", "the two high-intensity statins"),
    "rosuvastatin": ("atorvastatin", "the two high-intensity statins"),
    "simvastatin": ("atorvastatin", "lipophilic statins, different CYP3A4 exposure"),
    "pravastatin": ("simvastatin", "hydrophilic vs lipophilic statin (myopathy contrast)"),
    "lovastatin": ("simvastatin", "structurally closest statin"),
    "fluvastatin": ("pravastatin", "low-potency statins"),
    # antidepressant (only one in set)
    "paroxetine": ("lorazepam", "cross-class, WEAK: overlapping anxiety-disorder population only"),
    # fluoroquinolones
    "ciprofloxacin": ("levofloxacin", "same class"),
    "levofloxacin": ("moxifloxacin", "respiratory fluoroquinolones"),
    "moxifloxacin": ("levofloxacin", "respiratory fluoroquinolones"),
    "norfloxacin": ("ciprofloxacin", "urinary fluoroquinolones"),
    # ACE inhibitors
    "ramipril": ("lisinopril", "two most-used ACE inhibitors"),
    "lisinopril": ("ramipril", "two most-used ACE inhibitors"),
    "enalapril": ("lisinopril", "same class"),
    "perindopril": ("ramipril", "same class, similar market"),
    "quinapril": ("lisinopril", "same class"),
    "fosinopril": ("lisinopril", "same class"),
    "trandolapril": ("ramipril", "same class"),
    "cilazapril": ("ramipril", "same class (Canadian market)"),
    "lisinopril / hydrochlorothiazide": ("lisinopril", "isolates the hydrochlorothiazide component"),
    # proton pump inhibitors
    "omeprazole": ("esomeprazole", "racemate vs S-isomer"),
    "esomeprazole": ("omeprazole", "S-isomer vs racemate"),
    "pantoprazole": ("omeprazole", "same class"),
    "rabeprazole": ("pantoprazole", "same class"),
    # ARBs (cross-class ACEi comparison is the classic for angioedema/cough)
    "losartan": ("lisinopril", "cross-class: ARB vs ACE inhibitor (angioedema / cough contrast)"),
    "telmisartan": ("losartan", "same class"),
    "candesartan": ("losartan", "same class"),
    # calcium channel blockers
    "amlodipine": ("diltiazem", "dihydropyridine vs non-dihydropyridine CCB"),
    "diltiazem": ("amlodipine", "non-dihydropyridine vs dihydropyridine CCB"),
    # beta blockers
    "metoprolol": ("bisoprolol", "same class"),
    "bisoprolol": ("metoprolol", "same class"),
    # loop diuretic (only one in set)
    "furosemide": ("ramipril", "cross-class, WEAK: shared heart-failure population only"),
    # antidiabetics
    "metformin": ("empagliflozin", "cross-class: oral type-2-diabetes alternatives"),
    "empagliflozin": ("dapagliflozin", "same class"),
    "dapagliflozin": ("empagliflozin", "same class"),
    "canagliflozin": ("empagliflozin", "same class (amputation signal was canagliflozin-specific)"),
    # anticoagulants
    "apixaban": ("rivaroxaban", "same class (bleeding contrast)"),
    "rivaroxaban": ("apixaban", "same class (bleeding contrast)"),
    # GLP-1 agonists
    "semaglutide": ("liraglutide", "same class"),
    "liraglutide": ("semaglutide", "same class"),
    "dulaglutide": ("semaglutide", "same class"),
}


def default_pair(drug: str) -> str | None:
    return DEFAULT_PAIRWISE.get(drug, (None, None))[0]


# ---------------------------------------------------------------------------
# 2. Display scores (spec 19.1): 0-100 empirical percentile per dimension
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "signal_strength", "comparator_robustness", "persistence", "trend",
    "volatility", "demographic_heterogeneity", "class_anomaly", "uncertainty",
]

FEATURE_LABELS = {
    "signal_strength": "Strength (S)",
    "comparator_robustness": "Robustness (R)",
    "persistence": "Persistence (P)",
    "trend": "Trend (T)",
    "volatility": "Volatility (V)",
    "demographic_heterogeneity": "Heterogeneity (H)",
    "class_anomaly": "Class anomaly (C)",
    "uncertainty": "Uncertainty (U)",
}


def add_display_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile ranks are computed across the eligible pairs that have the
    dimension defined. Direction is left as-is (higher volatility /
    uncertainty score = more volatile / more uncertain), per spec 19.1."""
    for col in FEATURE_COLS:
        df[col + "_score"] = df[col].rank(pct=True) * 100
    return df


# ---------------------------------------------------------------------------
# 3. Signal phenotypes -- thresholds calibrated on the computed distributions
# ---------------------------------------------------------------------------
# Quantiles quoted are from the 47k non-fragile pairs with ror_low > 1 in
# the current build (see README). Re-check them if the input data changes.
THRESHOLDS = {
    "uncertainty_high": 0.45,     # ~p75 of SE(ln ROR) among signals; CI spans >2.4x
    "strength_mid": 1.15,         # ~p50 of signal_strength among signals
    "persistence_high": 0.75,     # p50-p75 (median is 0.80)
    "persistence_low": 0.60,      # below the median run of eligible years
    "trend_up": 0.15,             # ~p80 of yearly ln(ROR) slope
    "trend_down": -0.15,          # ~p20
    "acceleration_up": 0.40,      # ~p75 of recent minus historical slope
    "class_anomaly_high": 2.0,    # ~p75 of the (clipped) class z-score
    "heterogeneity_high": 1.30,   # ~p75 of SD of subgroup ln(ROR)
    "min_reports_mature": 10,     # below this, a series can't earn a 'mature' label
}

PHENOTYPES = [
    # (label, priority order, one-line meaning, suggested action)
    ("No disproportionality",       "Lower CI bound of the curated-universe ROR is <= 1", "None; reported at or below background"),
    ("Sparse / unstable",           "Zero-cell correction applied or SE(ln ROR) is high", "De-prioritise; needs more evidence"),
    ("Emerging",                    "Steep upward ln(ROR) trend with short or accelerating history", "Monitor; investigate recent temporal context"),
    ("Waning",                      "Downward ln(ROR) trend, signal no longer persistent", "Historical signal; check for labelling / market changes"),
    ("Persistent robust",           "High persistence, positive in every comparator, at least median strength", "Priority for further assessment"),
    ("Drug-specific (class-anomalous)", "Elevated vs. same-class peers, high class z-score", "Potentially drug-specific reporting behaviour"),
    ("Class-wide pattern",          "Elevated vs. universe but NOT vs. same-class peers", "Likely a class-level reporting pattern"),
    ("Demographically heterogeneous", "Large spread of ROR across sex / age strata", "Examine subgroup reporting before interpreting"),
    ("Moderate signal",             "Disproportionate but none of the distinguishing patterns above", "Routine follow-up"),
]


def assign_phenotypes(df: pd.DataFrame, t: dict = THRESHOLDS) -> pd.DataFrame:
    """Adds 'phenotype' (mutually exclusive, first matching rule wins) and
    boolean tag columns so that no secondary pattern is lost.

    Expects columns: ror_low_global, continuity_applied, uncertainty,
    signal_strength, comparator_robustness, persistence, trend,
    recent_acceleration, class_anomaly, ror_low_class,
    demographic_heterogeneity, n_reports.
    """
    d = df
    sig = d["ror_low_global"] > 1

    # tags (independent of the exclusive label)
    d["tag_fragile"] = d["continuity_applied"].fillna(False).astype(bool) | (d["uncertainty"] >= t["uncertainty_high"])
    d["tag_emerging"] = sig & (d["trend"] >= t["trend_up"]) & (
        (d["persistence"] < t["persistence_low"]) | (d["recent_acceleration"] >= t["acceleration_up"])
    )
    d["tag_waning"] = sig & (d["trend"] <= t["trend_down"]) & (d["persistence"] < t["persistence_low"])
    d["tag_persistent_robust"] = sig & (d["persistence"] >= t["persistence_high"]) & \
        (d["comparator_robustness"] >= 0.999) & (d["signal_strength"] >= t["strength_mid"]) & \
        (d["n_reports"] >= t["min_reports_mature"])
    d["tag_class_anomalous"] = sig & (d["class_anomaly"] >= t["class_anomaly_high"]) & (d["ror_low_class"] > 1)
    d["tag_class_wide"] = sig & d["ror_low_class"].notna() & (d["ror_low_class"] <= 1)
    d["tag_heterogeneous"] = sig & (d["demographic_heterogeneity"] >= t["heterogeneity_high"])

    for c in [c for c in d.columns if c.startswith("tag_")]:
        d[c] = d[c].fillna(False).astype(bool)

    label = pd.Series("Moderate signal", index=d.index, dtype="object")
    label[d["tag_heterogeneous"]] = "Demographically heterogeneous"
    label[d["tag_class_wide"]] = "Class-wide pattern"
    label[d["tag_class_anomalous"]] = "Drug-specific (class-anomalous)"
    label[d["tag_persistent_robust"]] = "Persistent robust"
    label[d["tag_waning"]] = "Waning"
    label[d["tag_emerging"]] = "Emerging"
    label[d["tag_fragile"]] = "Sparse / unstable"
    label[~sig] = "No disproportionality"
    d["phenotype"] = label
    return d
