"""
Export the precomputed signal results for the selected medications x ALL
outcomes to flat files, so they can be shared / analysed outside the
dashboard.

Reads faers_signals.duckdb (built by db_build.py) and writes to ./results/:

  all_drugs_global_signals.csv   drug x outcome x comparator, all years pooled
  all_drugs_annual_signals.csv   drug x outcome x comparator x year (long)
  all_drugs_subgroup_signals.csv drug x outcome x (sex | age) subgroup
  all_drugs_signal_features.csv  drug x outcome eight-dimensional features
  by_drug/<drug>.csv             one wide table per medication: every
                                 outcome as a row, pooled ROR/PRR/CI, and
                                 curated-universe ROR + report count for
                                 every year 2004-2025 as columns
  README_results.txt             column dictionary + analysis assumptions

Run:  python export_results.py                 # all 55 medications
      python export_results.py --drug ramipril --drug lisinopril
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import phenotypes

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "faers_signals.duckdb"
OUT_DIR = APP_DIR / "results"
MIN_REPORTS = 3

# Study classes exactly as grouped in the project's medication list
# (Anthony's table). The data's own ATC-4 class is kept alongside this in
# the 'atc4_class' column; 'study_class' is the grouping the team uses.
# Note: the source table listed simvastatin under "Antidepressant" -- that
# is a copy/paste slip; it is a statin and is grouped here accordingly.
STUDY_CLASS = {
    "Penicillin antibiotic class": [
        "amoxicillin", "clavulanate / amoxicillin", "ampicillin",
        "cloxacillin", "penicillin G", "penicillin V",
    ],
    "Macrolides": ["azithromycin"],
    "Benzodiazepine": ["lorazepam", "diazepam"],
    "Sulphonamide": ["sulfamethoxazole / trimethoprim"],
    "Cephalosporins": ["cephalexin"],
    "Statin": ["rosuvastatin", "fluvastatin", "lovastatin", "pravastatin",
               "atorvastatin", "simvastatin"],
    "Antidepressant": ["paroxetine"],
    "Fluoroquinolones": ["ciprofloxacin", "levofloxacin", "moxifloxacin", "norfloxacin"],
    "ACE Inhibitor": ["ramipril", "perindopril", "cilazapril", "enalapril", "fosinopril",
                      "lisinopril / hydrochlorothiazide", "lisinopril", "quinapril",
                      "trandolapril"],
    "Proton pump inhibitors": ["omeprazole", "pantoprazole", "rabeprazole", "esomeprazole"],
    "Angiotensin II receptor blockers": ["telmisartan", "losartan", "candesartan"],
    "Calcium channel blocker": ["amlodipine", "diltiazem"],
    "Beta blockers": ["metoprolol", "bisoprolol"],
    "Loop diuretics": ["furosemide"],
    "Biguanides": ["metformin"],
    "Urinary anti-infectives": ["nitrofurantoin", "fosfomycin", "methenamine"],
    "Antidiabetic - SGLT2 inhibitors": ["empagliflozin", "dapagliflozin", "canagliflozin"],
    "Anticoagulant": ["apixaban", "rivaroxaban"],
    "GLP-1 receptor agonist": ["semaglutide", "dulaglutide", "liraglutide"],
}
DRUG_TO_STUDY_CLASS = {d: c for c, ds in STUDY_CLASS.items() for d in ds}


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def add_signal_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["signal"] = (df["ror_low"] > 1.0) & (df["n_reports"] >= MIN_REPORTS)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", action="append", help="restrict to these drugs (repeatable)")
    args = ap.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"{DB_PATH} not found -- run db_build.py first.")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "by_drug").mkdir(exist_ok=True)

    all_drugs = [r[0] for r in con.execute("SELECT DISTINCT drug FROM exposure ORDER BY drug").fetchall()]
    drugs = args.drug or all_drugs
    unknown = [d for d in drugs if d not in all_drugs]
    if unknown:
        raise SystemExit(f"Unknown drug(s): {unknown}. Available: {all_drugs}")
    drug_list_sql = ", ".join("'" + d.replace("'", "''") + "'" for d in drugs)

    atc = con.execute(
        "SELECT DISTINCT drug, drug_class AS atc4_class FROM exposure"
    ).fetchdf().drop_duplicates("drug")
    n_drug = con.execute(
        "SELECT drug, COUNT(DISTINCT report_id) AS n_reports_drug FROM exposure GROUP BY drug"
    ).fetchdf()

    def decorate(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(atc, on="drug", how="left").merge(n_drug, on="drug", how="left")
        df["study_class"] = df["drug"].map(DRUG_TO_STUDY_CLASS)
        front = ["study_class", "drug", "atc4_class", "n_reports_drug", "pt"]
        rest = [c for c in df.columns if c not in front]
        return df[front + rest]

    # ---- long tables --------------------------------------------------------
    print("Exporting pooled (all-years) signals...")
    g = con.execute(f"""
        SELECT drug, pt, comparator, a, b, c, d, n_reports, ror, ror_low, ror_high,
               prr, rf, continuity_applied
        FROM signals_global WHERE drug IN ({drug_list_sql})
        ORDER BY drug, comparator, ror DESC
    """).fetchdf()
    g = decorate(add_signal_flag(g))
    g.to_csv(OUT_DIR / "all_drugs_global_signals.csv", index=False)
    print(f"  {len(g):,} rows")

    print("Exporting annual signals (long)...")
    a = con.execute(f"""
        SELECT drug, pt, comparator, year, a, b, c, d, n_reports, ror, ror_low, ror_high,
               prr, rf, continuity_applied
        FROM signals_annual WHERE drug IN ({drug_list_sql})
        ORDER BY drug, pt, comparator, year
    """).fetchdf()
    a = decorate(add_signal_flag(a))
    a.to_csv(OUT_DIR / "all_drugs_annual_signals.csv", index=False)
    print(f"  {len(a):,} rows")

    print("Exporting subgroup signals...")
    s = con.execute(f"""
        SELECT drug, pt, subgroup_type, subgroup_value, a, b, c, d, n_reports,
               ror, ror_low, ror_high, prr, rf, continuity_applied
        FROM signals_subgroup WHERE drug IN ({drug_list_sql})
        ORDER BY drug, pt, subgroup_type, subgroup_value
    """).fetchdf()
    s = decorate(add_signal_flag(s))
    s.to_csv(OUT_DIR / "all_drugs_subgroup_signals.csv", index=False)
    print(f"  {len(s):,} rows")

    print("Exporting eight-dimensional signal features...")
    f = con.execute(f"""
        SELECT * FROM signal_features WHERE drug IN ({drug_list_sql})
        ORDER BY drug, signal_strength DESC NULLS LAST
    """).fetchdf()
    f = decorate(f)
    f.to_csv(OUT_DIR / "all_drugs_signal_features.csv", index=False)
    print(f"  {len(f):,} rows")

    # ---- ML layer (present only after ml_build.py) ------------------------
    has_ml = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'signal_embeddings'"
    ).fetchone()[0] > 0
    n_emb = n_nb = n_cl = 0
    if has_ml:
        print("Exporting embeddings / clusters / anomalies / neighbours...")
        e = con.execute(f"""
            SELECT drug, pt, n_reports, phenotype, ror_global, ror_low_global,
                   {", ".join(phenotypes.FEATURE_COLS)},
                   class_anomaly_imputed, heterogeneity_imputed,
                   umap_1, umap_2, cluster, cluster_probability, anomaly_raw, anomaly_percentile
            FROM signal_embeddings WHERE drug IN ({drug_list_sql})
            ORDER BY anomaly_percentile DESC
        """).fetchdf()
        e = decorate(e)
        e.to_csv(OUT_DIR / "all_drugs_embeddings_clusters_anomalies.csv", index=False)
        n_emb = len(e)
        cl = con.execute("SELECT * FROM cluster_profiles ORDER BY cluster").fetchdf()
        cl.to_csv(OUT_DIR / "cluster_profiles.csv", index=False)
        n_cl = int((cl["cluster"] >= 0).sum())
        nb = con.execute(f"""
            SELECT n.drug, n.pt, n.rank, n.neighbor_drug, n.neighbor_pt, n.similarity,
                   f.phenotype AS neighbor_phenotype, f.ror_global AS neighbor_ror
            FROM signal_neighbors n
            JOIN signal_features f ON f.drug = n.neighbor_drug AND f.pt = n.neighbor_pt
            WHERE n.drug IN ({drug_list_sql}) ORDER BY n.drug, n.pt, n.rank
        """).fetchdf()
        nb.to_csv(OUT_DIR / "all_drugs_similar_signals.csv", index=False)
        n_nb = len(nb)
        print(f"  {n_emb:,} embedded pairs, {n_cl} clusters, {n_nb:,} neighbour rows")

    # ---- one wide table per drug -------------------------------------------
    print("Writing one wide ROR-by-year table per medication...")
    years = sorted(a["year"].dropna().astype(int).unique().tolist())
    cur_g = g[g["comparator"] == "curated_universe"]
    cls_g = g[g["comparator"] == "same_class"][["drug", "pt", "ror", "ror_low", "ror_high"]].rename(
        columns={"ror": "ror_same_class", "ror_low": "ror_low_same_class", "ror_high": "ror_high_same_class"}
    )
    cur_a = a[a["comparator"] == "curated_universe"]

    for drug in drugs:
        base = cur_g[cur_g["drug"] == drug][[
            "study_class", "drug", "atc4_class", "n_reports_drug", "pt",
            "n_reports", "ror", "ror_low", "ror_high", "prr", "rf", "signal", "continuity_applied",
        ]].rename(columns={"n_reports": "n_reports_pair"})
        base = base.merge(cls_g[cls_g["drug"] == drug].drop(columns="drug"), on="pt", how="left")

        ann = cur_a[cur_a["drug"] == drug]
        ror_w = ann.pivot_table(index="pt", columns="year", values="ror", aggfunc="first")
        ror_w.columns = [f"ROR_{int(y)}" for y in ror_w.columns]
        n_w = ann.pivot_table(index="pt", columns="year", values="n_reports", aggfunc="first")
        n_w.columns = [f"n_{int(y)}" for y in n_w.columns]
        # keep a full, consistent 2004-2025 year grid in every file
        ror_w = ror_w.reindex(columns=[f"ROR_{y}" for y in years])
        n_w = n_w.reindex(columns=[f"n_{y}" for y in years])

        feat_cols = [
            "pt", "phenotype", "comparator_drug", "ror_pairwise", "ror_low_pairwise",
            "signal_strength", "comparator_robustness", "n_comparators", "persistence", "trend",
            "volatility", "recent_acceleration", "demographic_heterogeneity",
            "class_anomaly", "uncertainty",
        ]
        feat_cols += [c for c in f.columns if c.endswith("_score") or c.startswith("tag_")]
        feat = f[f["drug"] == drug][[c for c in feat_cols if c in f.columns]]
        if has_ml:
            feat = feat.merge(
                e[e["drug"] == drug][["pt", "cluster", "cluster_probability", "anomaly_percentile", "umap_1", "umap_2"]],
                on="pt", how="left",
            )

        wide = (base.merge(ror_w, left_on="pt", right_index=True, how="left")
                    .merge(n_w, left_on="pt", right_index=True, how="left")
                    .merge(feat, on="pt", how="left"))
        # Rank by the LOWER confidence bound rather than the point estimate:
        # sorting on raw ROR puts 3-report zero-cell outcomes with ROR in the
        # thousands at the top; ror_low is the conventional conservative
        # ranking and floats well-supported signals up instead.
        wide = wide.sort_values(["ror_low", "n_reports_pair"], ascending=[False, False])
        for col in n_w.columns:
            wide[col] = wide[col].astype("Int64")
        wide.to_csv(OUT_DIR / "by_drug" / f"{safe_name(drug)}.csv", index=False)
    print(f"  {len(drugs)} files in results/by_drug/")

    # ---- README ---------------------------------------------------------------
    readme = f"""FAERS signal results -- selected medications x all outcomes
==========================================================

Source: FAERS_4_SSA_2004-2025_drug_rows.csv (openFDA FAERS extract, reports
received 2004-2025; 3,999,131 distinct reports / 8,452,042 drug-mention rows).

Files
-----
all_drugs_global_signals.csv    {len(g):,} rows  drug x outcome x comparator, all years pooled
all_drugs_annual_signals.csv    {len(a):,} rows  ...x year (long format)
all_drugs_subgroup_signals.csv  {len(s):,} rows  drug x outcome x (sex | age <65 / >=65)
all_drugs_signal_features.csv   {len(f):,} rows  eight-dimensional features per drug x outcome
by_drug/<drug>.csv              {len(drugs)} files one wide table per medication (see below)
all_drugs_embeddings_clusters_anomalies.csv  {n_emb:,} rows  ML layer (below){"" if has_ml else "  -- NOT PRESENT, run ml_build.py"}
cluster_profiles.csv            {n_cl} HDBSCAN clusters (+ noise), median profile + majority phenotype
all_drugs_similar_signals.csv   {n_nb:,} rows  top-15 nearest neighbours per embedded pair

ML layer (ml_build.py; spec sections 14.2, 15-18)
--------------------------------------------------
Population: disproportionate pairs (lower CI > 1) with >= 3 eligible
years, so the four longitudinal dimensions exist. class_anomaly is
imputed to 0 where the class has < 4 drugs (class_anomaly_imputed flags
it); demographic_heterogeneity to the median where missing.
Scaling: RobustScaler on the 8 dimensions.
umap_1, umap_2         UMAP(n_neighbors=25, min_dist=0.1, euclidean, seed 42)
cluster, cluster_probability
                       HDBSCAN(min_cluster_size=400, min_samples=20, leaf) run
                       on the 2-D UMAP coordinates. In the raw 8-D space
                       HDBSCAN only separates on the near-discrete robustness
                       dimension (2 clusters, 31-43% noise); on the UMAP
                       manifold six clusters emerge with distinct profiles.
                       -1 = noise (not assigned).
anomaly_raw, anomaly_percentile
                       IsolationForest(500 trees, seed 42) on the 8-D scaled
                       space; percentile 100 = most unusual COMBINATION of
                       dimensions. Note that mass-reporting artefacts (one
                       product with hundreds of near-identical reports) rank
                       highly here too -- which is useful QA, but read the
                       n_reports / ROR columns before treating a top anomaly
                       as a safety signal.
similarity             cosine similarity in the scaled 8-D space (behavioural
                       similarity, not clinical similarity).

Unit of analysis / counting rules
---------------------------------
* Every count is COUNT(DISTINCT Safetyreport ID), never rows. A report spans
  ~2.1 rows in the source file, so row counts would inflate everything.
* "Exposed to drug A" = the report lists A with Role_Code PS (primary
  suspect) or SS (secondary suspect). Concomitant (C), interacting (I) and
  blank-role mentions are NOT counted as exposure. Roughly two thirds of
  mentions in the file are concomitant, so this restriction matters a lot;
  change SUSPECT_ROLES in db_build.py and rebuild to test sensitivity.
* Outcome = one MedDRA Preferred Term. The Reactions column is split on
  " / " (space-slash-space) so multi-term reports contribute one row per PT.
  Terms are upper-cased before matching.
* Only drug x outcome pairs with >= {MIN_REPORTS} distinct reports are included.
* Universe for the curated-universe comparator = the {len(all_drugs)} medications in
  this extract (suspect-role reports), NOT all of FAERS. Same-class
  comparator = other medications in the same ATC-4 class (first ';'-
  separated class only; 'clavulanate / amoxicillin' has no ATC-4 class in
  the source and therefore has no same-class results).

2x2 table (per drug A, outcome O, comparator reference set D)
-------------------------------------------------------------
a = reports with A and O          b = reports with A, without O
c = reports in D with O           d = reports in D, without O
ROR = ad / bc ; 95% CI = exp(ln ROR +/- 1.96 * sqrt(1/a+1/b+1/c+1/d))
PRR = [a/(a+b)] / [c/(c+d)] ;  rf (reporting fraction) = a/(a+b)
continuity_applied = True when any cell was 0 and +0.5 was added to all
four cells. Treat those rows as fragile.
signal = (ror_low > 1) AND (n_reports >= {MIN_REPORTS})   -- the pre-specified rule

Columns common to all files
---------------------------
study_class      class grouping from the project medication list
drug             standardized ingredient (RxNorm) as it appears in the source
atc4_class       WHO ATC level-4 class from the source (first class listed)
n_reports_drug   distinct suspect-role reports for this drug, all years
pt               MedDRA Preferred Term (outcome)
comparator       curated_universe | same_class
n_reports        = a, distinct reports with this drug AND this outcome

by_drug/<drug>.csv (one row per outcome, sorted by ror_low descending --
the conventional conservative ranking; sorting on ror itself would rank
sparse zero-cell outcomes first)
----------------------------------------
n_reports_pair, ror, ror_low, ror_high, prr, rf, signal, continuity_applied
    curated-universe comparator, all years pooled
ror_same_class, ror_low_same_class, ror_high_same_class
    same ATC-4 class comparator, all years pooled
ROR_2004 ... ROR_2025      curated-universe ROR in that calendar year
                           (blank = the pair had no reports that year)
n_2004 ... n_2025          distinct reports with drug AND outcome that year
signal_strength ... uncertainty
    the eight embedding dimensions (see FAERS_Signal_Embedding_Platform.docx
    sections 9-14): persistence, trend, volatility, recent_acceleration,
    comparator_robustness (share of curated / same-class / pairwise
    comparators with lower CI > 1; n_comparators says how many were
    available), class_anomaly (z vs same-class peers, needs >= 3 peers,
    clipped to +/-5), demographic_heterogeneity, uncertainty (SE of ln ROR),
    signal_strength = ln(ROR)/(1+SE).
*_score
    0-100 empirical percentile of each dimension across all eligible pairs
    (spec 19.1) -- the values plotted on the dashboard's fingerprint radar.
comparator_drug, ror_pairwise, ror_low_pairwise
    the recommended pairwise comparator for this drug (phenotypes.py,
    DEFAULT_PAIRWISE, with the pharmacological rationale) and the ROR
    against it, all years pooled.
phenotype (+ tag_* booleans)
    interpretable signal category, first matching rule wins; tag_* keep
    every secondary pattern. Thresholds are calibrated on the empirical
    distribution of the computed features (phenotypes.THRESHOLDS):
      No disproportionality           lower CI of universe ROR <= 1
      Sparse / unstable               zero-cell correction or SE(ln ROR) >= 0.45
      Emerging                        trend >= 0.15/yr and (persistence < 0.6 or acceleration >= 0.4)
      Waning                          trend <= -0.15/yr and persistence < 0.6
      Persistent robust               persistence >= 0.75, every comparator > 1, strength >= 1.15, N >= 10
      Drug-specific (class-anomalous) class z >= 2 and elevated vs same-class peers
      Class-wide pattern              elevated vs universe but NOT vs same-class peers
      Demographically heterogeneous   SD of subgroup ln(ROR) >= 1.3
      Moderate signal                 disproportionate, none of the above

Interpretation
--------------
These are measures of disproportionate REPORTING in a spontaneous-report
database with no exposure denominator. They are hypothesis-generating and
do not estimate incidence, relative risk, or causation.
"""
    (OUT_DIR / "README_results.txt").write_text(readme)
    print(f"Done. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
