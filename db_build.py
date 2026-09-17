"""
One-time (re-runnable) batch build for the FAERS Signal Embedding Platform.

Reads the raw FAERS CSV (drug-row level, one row per report x drug, with a
'/'-delimited multi-valued Reactions column) and precomputes everything the
interactive dashboard needs, following the architecture recommendation in
the spec (section 23.1):

    "Do not calculate all 2x2 tables interactively from raw reports each
    time the user clicks. Precompute a signal feature table [...] The
    interactive application should query precomputed results and only
    recalculate when the analyst changes a custom comparator or subgroup
    definition."

Output: a single DuckDB file (faers_signals.duckdb) containing:

  exposure        one row per (report_id, drug) -- suspect-drug exposures
  events          one row per (report_id, pt)   -- exploded reactions
  report_demo     one row per report_id          -- sex / age for subgroups
  eligible_pairs  drug x pt combinations meeting the minimum-report rule
  signals_annual  drug x pt x comparator x year ROR/PRR/CI  (multi-year ROR)
  signals_global  drug x pt x comparator, all years pooled
  signals_subgroup drug x pt x (sex|age) subgroup ROR
  signal_features drug x pt: the eight-dimensional interpretable embedding
                  inputs (signal_strength, comparator_robustness,
                  persistence, trend, volatility, demographic_heterogeneity,
                  class_anomaly, uncertainty) -- ready for a later
                  UMAP/HDBSCAN/Isolation Forest phase, not computed here.

The dashboard (dashboard.py) only ever reads from this file plus the small
'exposure'/'events' tables for live pairwise-comparator queries -- it never
re-scans the multi-GB source CSV.

Run:  python db_build.py
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
CSV_PATH = APP_DIR.parent / "FAERS_4_SSA_2004-2025_drug_rows.csv"
FINAL_DB_PATH = APP_DIR / "faers_signals.duckdb"

# The database is built in local scratch space, not directly on the
# connected-folder mount: DuckDB's spill-to-disk temp files could not be
# removed on that mount (IOException: Operation not permitted), which is a
# limitation of the mount rather than of the data. The finished, compact
# .duckdb file is copied onto the mounted folder as the last build step so
# dashboard.py can read it from the project folder as usual.
BUILD_DIR = Path.home() / ".faers_build"
DB_PATH = BUILD_DIR / "faers_signals.duckdb"

ID_COL = "Safetyreport ID"
YEAR_COL = "report_year"
DRUG_COL = "Generic Name (Cleaned)"
CLASS_COL = "ATC-4 Class"
REACTIONS_COL = "Reactions"
ROLE_COL = "Role_Code"
SEX_COL = "Patient Sex"
AGE_COL = "Patient Age"

# Primary Suspect + Secondary Suspect = standard "suspect drug" definition
# used in disproportionality analysis; concomitant ('C') and interacting
# ('I') mentions are excluded from the exposure definition by default.
SUSPECT_ROLES = ("PS", "SS")

MIN_REPORTS = 3          # spec section 6: minimum-support rule
AGE_CUTPOINT = 65         # spec section 10
NULL_STRINGS = ["None", "NA", "N/A", "NULL", "UNK", "UNKNOWN", ""]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Step 0: connect
# ---------------------------------------------------------------------------

def get_connection(reset: bool = False) -> duckdb.DuckDBPyConnection:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find source CSV at {CSV_PATH}. Edit CSV_PATH in "
            f"db_build.py if the file lives somewhere else."
        )
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if reset and DB_PATH.exists():
        DB_PATH.unlink()
    con = duckdb.connect(str(DB_PATH))
    con.execute("PRAGMA threads=4")
    tmp_dir = BUILD_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{str(tmp_dir)}'")
    return con


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    r = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()[0]
    return r > 0


# ---------------------------------------------------------------------------
# Step 1: raw ingestion + normalized base tables (spec sections 2-6)
# ---------------------------------------------------------------------------

def build_base_tables(con: duckdb.DuckDBPyConnection) -> None:
    nullstr_sql = ", ".join(f"'{s}'" for s in NULL_STRINGS)
    csv_posix = str(CSV_PATH).replace("'", "''")

    # 'raw' is materialized as a TABLE (not a view) so the ~10GB CSV is
    # scanned and parsed exactly once. Every step after this reads the
    # compact, typed, columnar copy instead of re-parsing the source file.
    if table_exists(con, "raw"):
        log("'raw' already built, skipping CSV scan.")
    else:
        log("Scanning raw CSV once (streams from disk; does not load 9+GB into RAM)...")
        con.execute(f"""
            CREATE TABLE raw AS
            SELECT
                "{ID_COL}"            AS report_id,
                TRY_CAST({YEAR_COL} AS INTEGER)      AS year,
                "{DRUG_COL}"          AS drug,
                "{CLASS_COL}"         AS drug_class,
                "{ROLE_COL}"          AS role_code,
                "{REACTIONS_COL}"     AS reactions_raw,
                TRY_CAST("{SEX_COL}" AS INTEGER)     AS sex,
                TRY_CAST("{AGE_COL}" AS DOUBLE)      AS age
            FROM read_csv('{csv_posix}', sample_size=-1, nullstr=[{nullstr_sql}])
        """)
        n = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
        log(f"raw: {n:,} rows materialized")

    # One row per (report_id, drug): suspect-role exposures only, deduplicated.
    # MAX(...) collapses the row-level duplication seen in the raw extract
    # (the same report/drug pair can repeat many times with different
    # dosage/date detail rows) while keeping a representative sex/age/class.
    if table_exists(con, "exposure"):
        log("'exposure' already built, skipping.")
    else:
        log("Building 'exposure' (suspect-drug reports)...")
        # The source 'ATC-4 Class' field is inconsistently multi-valued
        # (e.g. "Penicillins with extended spectrum" for one drug vs.
        # "Penicillins with extended spectrum;Antibiotics" for another
        # drug that should be in the very same class). Taking only the
        # first ';'-separated segment recovers the intended class grouping
        # without any false merges observed in this dataset -- verified by
        # inspecting every distinct (drug, drug_class) pair in the extract.
        con.execute(f"""
            CREATE TABLE exposure AS
            SELECT
                report_id,
                MAX(year) AS year,
                drug,
                MAX(trim(split_part(drug_class, ';', 1))) AS drug_class,
                MAX(sex) AS sex,
                MAX(age) AS age
            FROM raw
            WHERE drug IS NOT NULL
              AND role_code IN ({", ".join(f"'{r}'" for r in SUSPECT_ROLES)})
            GROUP BY report_id, drug
        """)
        n_exp = con.execute("SELECT COUNT(*) FROM exposure").fetchone()[0]
        n_rep = con.execute("SELECT COUNT(DISTINCT report_id) FROM exposure").fetchone()[0]
        log(f"exposure: {n_exp:,} drug-report rows / {n_rep:,} distinct reports")

    # report-level demographics (used for subgroup universes; sex/age are
    # patient attributes, not drug-role attributes, so pooled across all
    # rows for the report regardless of suspect/concomitant status).
    if table_exists(con, "report_demo"):
        log("'report_demo' already built, skipping.")
    else:
        log("Building 'report_demo'...")
        con.execute("""
            CREATE TABLE report_demo AS
            SELECT report_id, MAX(year) AS year, MAX(sex) AS sex, MAX(age) AS age
            FROM raw
            GROUP BY report_id
        """)

    # Exploded reactions (spec section 5/6): one row per (report_id, PT),
    # restricted to reports that are part of the suspect-drug universe so
    # that "universe" totals stay internally consistent with 'exposure'.
    if table_exists(con, "events"):
        log("'events' already built, skipping.")
    else:
        log("Exploding Reactions into one row per (report, PT)...")
        con.execute("""
            CREATE TABLE events AS
            SELECT DISTINCT
                r.report_id,
                e.year,
                upper(trim(pt)) AS pt
            FROM raw r
            JOIN (SELECT DISTINCT report_id, year FROM exposure) e
              ON e.report_id = r.report_id
            , UNNEST(string_split(r.reactions_raw, ' / ')) AS t(pt)
            WHERE r.reactions_raw IS NOT NULL
        """)
        n_evt = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        n_pt = con.execute("SELECT COUNT(DISTINCT pt) FROM events").fetchone()[0]
        log(f"events:   {n_evt:,} report-PT rows / {n_pt:,} distinct PTs")


# ---------------------------------------------------------------------------
# Step 2: minimum-support eligible Drug x Outcome pairs (spec section 6)
# ---------------------------------------------------------------------------

def build_eligible_pairs(con: duckdb.DuckDBPyConnection) -> None:
    if table_exists(con, "eligible_pairs"):
        log("'eligible_pairs' already built, skipping.")
        return
    log(f"Finding eligible Drug x Outcome pairs (N >= {MIN_REPORTS})...")
    con.execute(f"""
        CREATE OR REPLACE TABLE eligible_pairs AS
        SELECT e.drug, ev.pt, COUNT(DISTINCT e.report_id) AS n_reports
        FROM exposure e
        JOIN events ev ON ev.report_id = e.report_id
        GROUP BY e.drug, ev.pt
        HAVING COUNT(DISTINCT e.report_id) >= {MIN_REPORTS}
    """)
    n = con.execute("SELECT COUNT(*) FROM eligible_pairs").fetchone()[0]
    log(f"eligible_pairs: {n:,} Drug x Outcome combinations")


# ---------------------------------------------------------------------------
# Step 3: curated-universe comparator counts (spec section 7, C1)
#         reference = "all other curated medications" = the whole dataset,
#         which is the standard whole-database 2x2 table:
#           a = drug & event         b = drug & !event
#           c = !drug & event        d = !drug & !event
# ---------------------------------------------------------------------------

def _finish_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Given a/b/c/d columns, add ror/ror_low/ror_high/prr/rf/n_reports plus
    a 'continuity_applied' flag: True whenever any of a/b/c/d was zero and a
    +0.5 continuity correction had to be used. Those rows are mathematically
    valid but statistically fragile (typically an event reported for only
    one curated drug) -- the flag lets the dashboard warn about them
    instead of presenting a five-digit ROR as if it were a precise
    estimate (spec sections 8.2 and 21.1 both call this out explicitly)."""
    if df.empty:
        for col in ["ror", "ror_low", "ror_high", "prr", "rf", "n_reports", "continuity_applied"]:
            df[col] = pd.Series(dtype="float64")
        return df
    out = df.apply(
        lambda r: pd.Series(stats.ror_from_counts(r.a, r.b, r.c, r.d)),
        axis=1,
    )
    out.columns = ["ror", "ror_low", "ror_high"]
    df = pd.concat([df, out], axis=1)
    df["prr"] = df.apply(lambda r: stats.prr_from_counts(r.a, r.b, r.c, r.d), axis=1)
    df["rf"] = df.apply(lambda r: stats.rf_from_counts(r.a, r.b), axis=1)
    df["n_reports"] = df["a"]
    df["continuity_applied"] = (df[["a", "b", "c", "d"]] == 0).any(axis=1)
    return df


def build_curated_universe(con: duckdb.DuckDBPyConnection) -> None:
    if table_exists(con, "signals_global_curated") and table_exists(con, "signals_annual_curated"):
        log("Curated-universe signals already built, skipping.")
        return
    log("Computing curated-universe comparator (global, all years pooled)...")

    n_total = con.execute("SELECT COUNT(DISTINCT report_id) FROM exposure").fetchone()[0]
    n_drug = con.execute(
        "SELECT drug, COUNT(DISTINCT report_id) n FROM exposure GROUP BY drug"
    ).fetchdf().set_index("drug")["n"]
    n_event = con.execute(
        "SELECT pt, COUNT(DISTINCT report_id) n FROM events GROUP BY pt"
    ).fetchdf().set_index("pt")["n"]
    n_pair = con.execute("""
        SELECT e.drug, ev.pt, COUNT(DISTINCT e.report_id) n
        FROM exposure e JOIN events ev ON ev.report_id = e.report_id
        GROUP BY e.drug, ev.pt
    """).fetchdf()

    elig = con.execute("SELECT drug, pt FROM eligible_pairs").fetchdf()
    df = elig.merge(n_pair, on=["drug", "pt"], how="left")
    df["n"] = df["n"].fillna(0)
    df["a"] = df["n"]
    df["b"] = df["drug"].map(n_drug) - df["a"]
    df["c"] = df["pt"].map(n_event) - df["a"]
    df["d"] = n_total - df["a"] - df["b"] - df["c"]
    df = df[["drug", "pt", "a", "b", "c", "d"]]
    df = _finish_counts(df)
    df["comparator"] = "curated_universe"
    con.register("signals_global_curated_df", df)
    con.execute("""
        CREATE OR REPLACE TABLE signals_global_curated AS
        SELECT * FROM signals_global_curated_df
    """)

    log("Computing curated-universe comparator (annual)...")
    n_total_y = con.execute(
        "SELECT year, COUNT(DISTINCT report_id) n FROM exposure GROUP BY year"
    ).fetchdf().set_index("year")["n"]
    n_drug_y = con.execute(
        "SELECT drug, year, COUNT(DISTINCT report_id) n FROM exposure GROUP BY drug, year"
    ).fetchdf()
    n_event_y = con.execute(
        "SELECT pt, year, COUNT(DISTINCT report_id) n FROM events GROUP BY pt, year"
    ).fetchdf()
    n_pair_y = con.execute("""
        SELECT e.drug, ev.pt, e.year, COUNT(DISTINCT e.report_id) n
        FROM exposure e JOIN events ev
          ON ev.report_id = e.report_id AND ev.year = e.year
        GROUP BY e.drug, ev.pt, e.year
    """).fetchdf()

    dfa = n_pair_y.merge(elig, on=["drug", "pt"], how="inner")
    dfa = dfa.merge(n_drug_y.rename(columns={"n": "n_drug"}), on=["drug", "year"], how="left")
    dfa = dfa.merge(n_event_y.rename(columns={"n": "n_event"}), on=["pt", "year"], how="left")
    dfa["n_total"] = dfa["year"].map(n_total_y)
    dfa["a"] = dfa["n"]
    dfa["b"] = dfa["n_drug"] - dfa["a"]
    dfa["c"] = dfa["n_event"] - dfa["a"]
    dfa["d"] = dfa["n_total"] - dfa["a"] - dfa["b"] - dfa["c"]
    dfa = dfa[["drug", "pt", "year", "a", "b", "c", "d"]]
    dfa = _finish_counts(dfa)
    dfa["comparator"] = "curated_universe"
    con.register("signals_annual_curated_df", dfa)
    con.execute("""
        CREATE OR REPLACE TABLE signals_annual_curated AS
        SELECT * FROM signals_annual_curated_df
    """)
    log(f"signals_annual_curated: {len(dfa):,} rows")


# ---------------------------------------------------------------------------
# Step 4: same-therapeutic-class comparator (spec section 7, C2)
#         reference = other drugs in the same ATC-4 class, excluding A
# ---------------------------------------------------------------------------

def build_same_class(con: duckdb.DuckDBPyConnection) -> None:
    if table_exists(con, "signals_global_class") and table_exists(con, "signals_annual_class"):
        log("Same-class signals already built, skipping.")
        return
    log("Computing same-therapeutic-class comparator (global)...")

    drug_class = con.execute(
        "SELECT DISTINCT drug, drug_class FROM exposure WHERE drug_class IS NOT NULL"
    ).fetchdf()

    # Reports from OTHER drugs in the same class (excludes the target drug's
    # own reports even under polypharmacy overlap, via the d2.drug != d1.drug
    # join condition).
    class_total = con.execute("""
        SELECT d1.drug AS drug, COUNT(DISTINCT d2.report_id) AS n
        FROM (SELECT DISTINCT drug, drug_class FROM exposure) d1
        JOIN exposure d2
          ON d2.drug_class = d1.drug_class AND d2.drug != d1.drug
        GROUP BY d1.drug
    """).fetchdf().set_index("drug")["n"]

    class_event = con.execute("""
        SELECT d1.drug AS drug, ev.pt AS pt, COUNT(DISTINCT ev.report_id) AS n
        FROM (SELECT DISTINCT drug, drug_class FROM exposure) d1
        JOIN exposure d2 ON d2.drug_class = d1.drug_class AND d2.drug != d1.drug
        JOIN events ev ON ev.report_id = d2.report_id
        GROUP BY d1.drug, ev.pt
    """).fetchdf()

    elig = con.execute("SELECT drug, pt, n_reports AS a FROM eligible_pairs").fetchdf()
    df = elig.merge(class_event, on=["drug", "pt"], how="left")
    df["n"] = df["n"].fillna(0)
    df["c"] = df["n"]
    df["b"] = df["drug"].map(
        con.execute("SELECT drug, COUNT(DISTINCT report_id) n FROM exposure GROUP BY drug")
        .fetchdf().set_index("drug")["n"]
    ) - df["a"]
    df["d"] = df["drug"].map(class_total) - df["c"]
    df = df[["drug", "pt", "a", "b", "c", "d"]]
    df = _finish_counts(df)
    df["comparator"] = "same_class"
    con.register("signals_global_class_df", df)
    con.execute("""
        CREATE OR REPLACE TABLE signals_global_class AS
        SELECT * FROM signals_global_class_df
    """)

    log("Computing same-therapeutic-class comparator (annual)...")
    class_total_y = con.execute("""
        SELECT d1.drug AS drug, d2.year AS year, COUNT(DISTINCT d2.report_id) AS n
        FROM (SELECT DISTINCT drug, drug_class FROM exposure) d1
        JOIN exposure d2 ON d2.drug_class = d1.drug_class AND d2.drug != d1.drug
        GROUP BY d1.drug, d2.year
    """).fetchdf()
    class_event_y = con.execute("""
        SELECT d1.drug AS drug, ev.pt AS pt, ev.year AS year, COUNT(DISTINCT ev.report_id) AS n
        FROM (SELECT DISTINCT drug, drug_class FROM exposure) d1
        JOIN exposure d2 ON d2.drug_class = d1.drug_class AND d2.drug != d1.drug
        JOIN events ev ON ev.report_id = d2.report_id AND ev.year = d2.year
        GROUP BY d1.drug, ev.pt, ev.year
    """).fetchdf()
    n_pair_y = con.execute("""
        SELECT e.drug, ev.pt, e.year, COUNT(DISTINCT e.report_id) n
        FROM exposure e JOIN events ev
          ON ev.report_id = e.report_id AND ev.year = e.year
        GROUP BY e.drug, ev.pt, e.year
    """).fetchdf()
    n_drug_y = con.execute(
        "SELECT drug, year, COUNT(DISTINCT report_id) n FROM exposure GROUP BY drug, year"
    ).fetchdf()

    elig2 = con.execute("SELECT drug, pt FROM eligible_pairs").fetchdf()
    dfa = n_pair_y.merge(elig2, on=["drug", "pt"], how="inner").rename(columns={"n": "a"})
    dfa = dfa.merge(class_event_y.rename(columns={"n": "c"}), on=["drug", "pt", "year"], how="left")
    dfa["c"] = dfa["c"].fillna(0)
    dfa = dfa.merge(n_drug_y.rename(columns={"n": "n_drug"}), on=["drug", "year"], how="left")
    dfa["b"] = dfa["n_drug"] - dfa["a"]
    dfa = dfa.merge(class_total_y.rename(columns={"n": "n_class_total"}), on=["drug", "year"], how="left")
    dfa["d"] = dfa["n_class_total"] - dfa["c"]
    dfa = dfa[["drug", "pt", "year", "a", "b", "c", "d"]].dropna(subset=["d"])
    dfa = _finish_counts(dfa)
    dfa["comparator"] = "same_class"
    con.register("signals_annual_class_df", dfa)
    con.execute("""
        CREATE OR REPLACE TABLE signals_annual_class AS
        SELECT * FROM signals_annual_class_df
    """)
    log(f"signals_annual_class: {len(dfa):,} rows")


# ---------------------------------------------------------------------------
# Step 4b: recommended pairwise comparator (spec C3) for every drug
#          reference = ONE analyst-chosen second drug (phenotypes.DEFAULT_PAIRWISE)
#          a = A & O, b = A & !O, c = B & O, d = B & !O
# ---------------------------------------------------------------------------

def build_pairwise_defaults(con: duckdb.DuckDBPyConnection) -> None:
    if table_exists(con, "signals_global_pairwise") and table_exists(con, "signals_annual_pairwise"):
        log("Default pairwise signals already built, skipping.")
        return
    import phenotypes
    log("Computing recommended pairwise comparator for each drug...")

    pairs = pd.DataFrame(
        [(a, b) for a, (b, _) in phenotypes.DEFAULT_PAIRWISE.items()],
        columns=["drug", "comparator_drug"],
    )
    known = {r[0] for r in con.execute("SELECT DISTINCT drug FROM exposure").fetchall()}
    pairs = pairs[pairs["drug"].isin(known) & pairs["comparator_drug"].isin(known)]
    con.register("pairs_df", pairs)

    elig = con.execute("SELECT drug, pt, n_reports AS a FROM eligible_pairs").fetchdf()
    n_drug = con.execute(
        "SELECT drug, COUNT(DISTINCT report_id) n FROM exposure GROUP BY drug"
    ).fetchdf().set_index("drug")["n"]

    # global: c = comparator-drug reports with O, for every outcome eligible for the target
    comp_event = con.execute("""
        SELECT p.drug, p.comparator_drug, ev.pt, COUNT(DISTINCT e.report_id) AS c
        FROM pairs_df p
        JOIN exposure e ON e.drug = p.comparator_drug
        JOIN events ev ON ev.report_id = e.report_id
        GROUP BY p.drug, p.comparator_drug, ev.pt
    """).fetchdf()
    df = elig.merge(pairs, on="drug", how="inner")
    df = df.merge(comp_event, on=["drug", "comparator_drug", "pt"], how="left")
    df["c"] = df["c"].fillna(0)
    df["b"] = df["drug"].map(n_drug) - df["a"]
    df["d"] = df["comparator_drug"].map(n_drug) - df["c"]
    df = _finish_counts(df[["drug", "comparator_drug", "pt", "a", "b", "c", "d"]])
    df["comparator"] = "pairwise"
    con.register("signals_global_pairwise_df", df)
    con.execute("CREATE OR REPLACE TABLE signals_global_pairwise AS SELECT * FROM signals_global_pairwise_df")

    # annual
    n_drug_y = con.execute(
        "SELECT drug, year, COUNT(DISTINCT report_id) n FROM exposure GROUP BY drug, year"
    ).fetchdf()
    n_pair_y = con.execute("""
        SELECT e.drug, ev.pt, e.year, COUNT(DISTINCT e.report_id) a
        FROM exposure e JOIN events ev ON ev.report_id = e.report_id AND ev.year = e.year
        GROUP BY e.drug, ev.pt, e.year
    """).fetchdf()
    comp_event_y = con.execute("""
        SELECT p.drug, p.comparator_drug, ev.pt, e.year, COUNT(DISTINCT e.report_id) AS c
        FROM pairs_df p
        JOIN exposure e ON e.drug = p.comparator_drug
        JOIN events ev ON ev.report_id = e.report_id AND ev.year = e.year
        GROUP BY p.drug, p.comparator_drug, ev.pt, e.year
    """).fetchdf()
    dfa = n_pair_y.merge(elig[["drug", "pt"]], on=["drug", "pt"], how="inner").merge(pairs, on="drug", how="inner")
    dfa = dfa.merge(comp_event_y, on=["drug", "comparator_drug", "pt", "year"], how="left")
    dfa["c"] = dfa["c"].fillna(0)
    dfa = dfa.merge(n_drug_y.rename(columns={"n": "n_a"}), on=["drug", "year"], how="left")
    dfa = dfa.merge(n_drug_y.rename(columns={"drug": "comparator_drug", "n": "n_b"}), on=["comparator_drug", "year"], how="left")
    dfa["b"] = dfa["n_a"] - dfa["a"]
    dfa["d"] = dfa["n_b"].fillna(0) - dfa["c"]
    dfa = _finish_counts(dfa[["drug", "comparator_drug", "pt", "year", "a", "b", "c", "d"]])
    dfa["comparator"] = "pairwise"
    con.register("signals_annual_pairwise_df", dfa)
    con.execute("CREATE OR REPLACE TABLE signals_annual_pairwise AS SELECT * FROM signals_annual_pairwise_df")
    log(f"signals_global_pairwise: {len(df):,} rows; signals_annual_pairwise: {len(dfa):,} rows")


def combine_annual_and_global(con: duckdb.DuckDBPyConnection) -> None:
    if table_exists(con, "signals_annual") and table_exists(con, "signals_global"):
        log("'signals_annual'/'signals_global' already built, skipping.")
        return
    # UNION ALL BY NAME fills 'comparator_drug' with NULL for the two
    # population-based comparators.
    con.execute("""
        CREATE OR REPLACE TABLE signals_annual AS
        SELECT * FROM signals_annual_curated
        UNION ALL BY NAME SELECT * FROM signals_annual_class
        UNION ALL BY NAME SELECT * FROM signals_annual_pairwise
    """)
    con.execute("""
        CREATE OR REPLACE TABLE signals_global AS
        SELECT * FROM signals_global_curated
        UNION ALL BY NAME SELECT * FROM signals_global_class
        UNION ALL BY NAME SELECT * FROM signals_global_pairwise
    """)


# ---------------------------------------------------------------------------
# Step 5: demographic subgroups (spec section 10) -- curated-universe basis
# ---------------------------------------------------------------------------

def build_subgroups(con: duckdb.DuckDBPyConnection) -> None:
    if table_exists(con, "signals_subgroup"):
        log("'signals_subgroup' already built, skipping.")
        return
    log("Computing demographic subgroup signals (sex, age)...")
    elig = con.execute("SELECT drug, pt FROM eligible_pairs").fetchdf()

    subgroup_defs = [
        ("sex", "male", "rd.sex = 1"),
        ("sex", "female", "rd.sex = 2"),
        ("age", "under_65", f"rd.age < {AGE_CUTPOINT}"),
        ("age", "65_plus", f"rd.age >= {AGE_CUTPOINT}"),
    ]

    frames = []
    for sub_type, sub_val, cond in subgroup_defs:
        n_total = con.execute(f"""
            SELECT COUNT(DISTINCT e.report_id) FROM exposure e
            JOIN report_demo rd ON rd.report_id = e.report_id
            WHERE {cond}
        """).fetchone()[0]
        n_drug = con.execute(f"""
            SELECT e.drug, COUNT(DISTINCT e.report_id) n FROM exposure e
            JOIN report_demo rd ON rd.report_id = e.report_id
            WHERE {cond}
            GROUP BY e.drug
        """).fetchdf().set_index("drug")["n"]
        n_event = con.execute(f"""
            SELECT ev.pt, COUNT(DISTINCT ev.report_id) n FROM events ev
            JOIN report_demo rd ON rd.report_id = ev.report_id
            WHERE {cond}
            GROUP BY ev.pt
        """).fetchdf().set_index("pt")["n"]
        n_pair = con.execute(f"""
            SELECT e.drug, ev.pt, COUNT(DISTINCT e.report_id) n
            FROM exposure e
            JOIN events ev ON ev.report_id = e.report_id
            JOIN report_demo rd ON rd.report_id = e.report_id
            WHERE {cond}
            GROUP BY e.drug, ev.pt
        """).fetchdf()

        df = elig.merge(n_pair, on=["drug", "pt"], how="left")
        df["n"] = df["n"].fillna(0)
        df["a"] = df["n"]
        df["b"] = df["drug"].map(n_drug).fillna(0) - df["a"]
        df["c"] = df["pt"].map(n_event).fillna(0) - df["a"]
        df["d"] = n_total - df["a"] - df["b"] - df["c"]
        df = df[["drug", "pt", "a", "b", "c", "d"]]
        df = _finish_counts(df)
        df["subgroup_type"] = sub_type
        df["subgroup_value"] = sub_val
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    con.register("signals_subgroup_df", out)
    con.execute("CREATE OR REPLACE TABLE signals_subgroup AS SELECT * FROM signals_subgroup_df")
    log(f"signals_subgroup: {len(out):,} rows")


# ---------------------------------------------------------------------------
# Step 6: merge into the eight-dimensional signal_features table
#         (spec section 14: Z_AO = [S, R, P, T, V, H, C, U])
# ---------------------------------------------------------------------------

def _leave_one_out_zscore(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.Series:
    """Vectorized leave-one-out z-score of value_col within each group in
    group_cols: for every row, compare it to the mean/SD of its *peers*
    (same group, excluding itself) rather than the whole group including
    itself. Uses closed-form leave-one-out algebra (group sums / sums of
    squares) so it runs in O(n) instead of looping per row.
    """
    g = df.groupby(group_cols)[value_col]
    n = g.transform("count")
    s1 = g.transform("sum")
    s2 = g.transform(lambda v: (v * v).sum())

    n_loo = n - 1
    mean_loo = (s1 - df[value_col]) / n_loo
    sumsq_loo = s2 - df[value_col] ** 2
    var_loo = (sumsq_loo - n_loo * mean_loo ** 2) / (n_loo - 1)
    var_loo = var_loo.clip(lower=0)
    sd_loo = np.sqrt(var_loo)

    z = (df[value_col] - mean_loo) / sd_loo
    # With only two peers the SD is defined but meaningless (z-scores in the
    # thousands appeared in the first build). Require >= 3 peers -- i.e. the
    # class has >= 4 drugs reporting this outcome -- and clip to +/-5 so the
    # dimension stays on a sane scale for scaling/ML.
    z[(n_loo < 3) | (sd_loo == 0)] = np.nan
    return z.clip(-5, 5)


def _ols_slope(years: np.ndarray, y: np.ndarray) -> float:
    n = len(years)
    if n < 2:
        return float("nan")
    sx, sy = years.sum(), y.sum()
    sxy, sxx = (years * y).sum(), (years * years).sum()
    denom = n * sxx - sx * sx
    if denom == 0:
        return float("nan")
    return (n * sxy - sx * sy) / denom


def _temporal_features_fast(annual_curated: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Same definitions as stats.temporal_features (spec section 9.1), but
    computed with a light per-group numpy loop instead of sklearn, which is
    the difference between finishing in seconds vs. not finishing at all
    across >100k Drug x Outcome pairs."""
    x = annual_curated[annual_curated["n_reports"] >= min_count].copy()
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["ror"])
    x = x[x["ror"] > 0]
    x["log_ror"] = np.log(x["ror"])

    records = []
    for (drug, pt), g in x.groupby(["drug", "pt"], sort=False):
        if len(g) < 3:
            continue
        g = g.sort_values("year")
        years = g["year"].to_numpy(dtype=float)
        logror = g["log_ror"].to_numpy(dtype=float)

        persistence = float((g["ror_low"] > 1.0).mean())
        trend = _ols_slope(years, logror)
        volatility = float(np.std(logror, ddof=1)) if len(logror) > 1 else float("nan")

        split = max(2, len(g) // 2)
        early = g.iloc[:-split] if len(g) > split else g.iloc[:0]
        recent = g.iloc[-split:]
        slope_recent = _ols_slope(recent["year"].to_numpy(dtype=float), recent["log_ror"].to_numpy(dtype=float))
        slope_early = _ols_slope(early["year"].to_numpy(dtype=float), early["log_ror"].to_numpy(dtype=float)) if len(early) >= 2 else float("nan")
        acceleration = slope_recent - slope_early if not np.isnan(slope_early) else float("nan")

        records.append({
            "drug": drug, "pt": pt,
            "persistence": persistence, "trend": trend,
            "volatility": volatility, "recent_acceleration": acceleration,
        })
    return pd.DataFrame(records)


def build_signal_features(con: duckdb.DuckDBPyConnection) -> None:
    if table_exists(con, "signal_features"):
        log("'signal_features' already built, skipping.")
        return
    log("Assembling eight-dimensional signal_features table (vectorized)...")

    import phenotypes

    curated = con.execute("SELECT * FROM signals_global_curated").fetchdf()
    curated = curated[(curated["ror"] > 0) & curated["ror"].notna()].copy()
    class_g = con.execute("SELECT drug, pt, ror_low FROM signals_global_class").fetchdf()
    pair_g = con.execute(
        "SELECT drug, pt, comparator_drug, ror_low, ror AS ror_pairwise FROM signals_global_pairwise"
    ).fetchdf()
    drug_class = con.execute(
        "SELECT DISTINCT drug, drug_class FROM exposure WHERE drug_class IS NOT NULL"
    ).fetchdf()
    annual_curated = con.execute("SELECT drug, pt, year, ror, ror_low, n_reports FROM signals_annual_curated").fetchdf()
    subgroup = con.execute("SELECT drug, pt, ror FROM signals_subgroup").fetchdf()

    # --- signal strength / uncertainty (spec 14.1, 13) --------------------
    curated["se_log_ror"] = (np.log(curated["ror_high"]) - np.log(curated["ror_low"])) / (2 * 1.96)
    curated["signal_strength"] = np.log(curated["ror"]) / (1 + curated["se_log_ror"])

    # --- comparator robustness (spec 11): curated + same-class + pairwise --
    curated = curated.merge(class_g.rename(columns={"ror_low": "ror_low_class"}), on=["drug", "pt"], how="left")
    curated = curated.merge(pair_g.rename(columns={"ror_low": "ror_low_pairwise"}), on=["drug", "pt"], how="left")
    lows = curated[["ror_low", "ror_low_class", "ror_low_pairwise"]]
    n_valid = lows.notna().sum(axis=1)
    n_pos = (lows > 1).sum(axis=1)
    curated["comparator_robustness"] = n_pos / n_valid
    curated["n_comparators"] = n_valid

    # --- therapeutic-class anomaly (spec 12): leave-one-out z-score of
    #     log(ROR) among OTHER drugs in the SAME ATC-4 class, same outcome --
    curated = curated.merge(drug_class, on="drug", how="left")
    curated["log_ror"] = np.log(curated["ror"])
    curated["class_anomaly"] = np.nan
    has_class = curated["drug_class"].notna()
    curated.loc[has_class, "class_anomaly"] = _leave_one_out_zscore(
        curated.loc[has_class], ["drug_class", "pt"], "log_ror"
    )

    # --- temporal features (spec 9.1) --------------------------------------
    temporal = _temporal_features_fast(annual_curated, MIN_REPORTS)

    # --- demographic heterogeneity (spec 10): SD of log(ROR) across the
    #     4 subgroup estimates for this pair -------------------------------
    sub = subgroup[(subgroup["ror"] > 0) & subgroup["ror"].notna()].copy()
    sub["log_ror"] = np.log(sub["ror"])
    het = sub.groupby(["drug", "pt"])["log_ror"].agg(
        demographic_heterogeneity=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan
    ).reset_index()

    out = curated.merge(temporal, on=["drug", "pt"], how="left")
    out = out.merge(het, on=["drug", "pt"], how="left")
    out = out.rename(columns={
        "ror": "ror_global", "ror_low": "ror_low_global", "ror_high": "ror_high_global",
        "se_log_ror": "uncertainty",
    })
    out = out[[
        "drug", "pt", "n_reports", "signal_strength", "comparator_robustness", "n_comparators",
        "persistence", "trend", "volatility", "recent_acceleration",
        "demographic_heterogeneity", "class_anomaly", "uncertainty",
        "ror_global", "ror_low_global", "ror_high_global", "continuity_applied",
        "ror_low_class", "comparator_drug", "ror_pairwise", "ror_low_pairwise",
    ]]

    # --- display scores + interpretable phenotypes (spec 19, 16.2, 25) ----
    out = phenotypes.add_display_scores(out)
    out = phenotypes.assign_phenotypes(out)

    con.register("signal_features_df", out)
    con.execute("CREATE OR REPLACE TABLE signal_features AS SELECT * FROM signal_features_df")
    log(f"signal_features: {len(out):,} Drug x Outcome pairs")
    log("phenotype mix:\n" + out["phenotype"].value_counts().to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import sys
    reset = "--reset" in sys.argv
    t0 = time.time()
    con = get_connection(reset=reset)
    if "--rebuild-features" in sys.argv:
        # Cheap re-derivation layer: keeps raw/exposure/events/curated/class
        # tables, recomputes pairwise + unions + features/phenotypes only.
        for t in ["signals_global_pairwise", "signals_annual_pairwise",
                  "signals_annual", "signals_global", "signal_features"]:
            con.execute(f"DROP TABLE IF EXISTS {t}")
        log("Dropped derived tables for --rebuild-features.")
    build_base_tables(con)
    build_eligible_pairs(con)
    build_curated_universe(con)
    build_same_class(con)
    build_pairwise_defaults(con)
    combine_annual_and_global(con)
    build_subgroups(con)
    build_signal_features(con)

    # Helpful indexes for the interactive dashboard.
    con.execute("CREATE INDEX IF NOT EXISTS idx_exposure_drug ON exposure(drug)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_exposure_report ON exposure(report_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_events_report ON events(report_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_events_pt ON events(pt)")

    con.close()

    import shutil
    log(f"Copying finished database onto the project folder ({FINAL_DB_PATH})...")
    shutil.copy2(DB_PATH, FINAL_DB_PATH)

    log(f"Done in {time.time() - t0:.1f}s. Database written to {FINAL_DB_PATH}")


if __name__ == "__main__":
    main()
