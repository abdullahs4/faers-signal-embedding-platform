"""
FAERS Signal Embedding Platform -- interactive Streamlit dashboard.

Implements the "main signal screen" and "comparator-sensitive time series"
views from spec sections 20.1 and 20.3:

  - Drug selector, outcome selector, comparator selector, subgroup selector
  - Reporting Odds Ratio computed for every available year (multi-year ROR)
    with 95% confidence intervals, overlaid across the curated-universe,
    same-therapeutic-class, and an optional analyst-chosen pairwise
    comparator
  - Temporal features: persistence, trend, volatility, recent acceleration
  - Demographic heterogeneity (sex, age <65 / >=65)
  - Therapeutic-class anomaly and comparator robustness
  - A Drug x Outcome overview table for the selected drug (a simplified,
    single-row slice of the "matrix" view in spec section 20.5)

The dashboard ONLY reads the small precomputed DuckDB file produced by
db_build.py (run that first). The only queries that touch the underlying
exposure/events tables live are the analyst-chosen pairwise comparator and
the outcome-search box, both of which are cheap indexed lookups on top of a
database that is megabytes, not gigabytes, in size.

Run:  streamlit run dashboard.py
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
import urllib.request

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import stats
import phenotypes

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "faers_signals.duckdb"
# The precomputed database is >100MB, so it can't be committed to git as a
# single file. It is split into <100MB chunks tracked in data/ and
# reassembled here on first launch. Sizes and hashes are checked after every
# download -- a previous deploy silently produced a truncated/corrupted file
# (DuckDB "dictionary string index out of range"), most likely from two
# concurrent script runs racing on the same temp file during cold start.
DB_PARTS = [
    (
        "https://raw.githubusercontent.com/abdullahs4/faers-signal-embedding-platform/main/data/faers_signals.duckdb.part-00",
        94371840,
        "f374d069f5dac4b1e9cadd27364bb9c175750f8578b47ed81e047572aceccb7b",
    ),
    (
        "https://raw.githubusercontent.com/abdullahs4/faers-signal-embedding-platform/main/data/faers_signals.duckdb.part-01",
        94371840,
        "05a6ee015afc6ad8d4027257baa41c1bb3a1bbef526a55696b9a27fb0e9e6f29",
    ),
    (
        "https://raw.githubusercontent.com/abdullahs4/faers-signal-embedding-platform/main/data/faers_signals.duckdb.part-02",
        49557504,
        "06076fba3d7522f7e1b74e40123ebafb08c0813756f6c325706078f81a6fb4c6",
    ),
]
DB_EXPECTED_SIZE = sum(size for _, size, _ in DB_PARTS)
MIN_REPORTS_DEFAULT = 3
AGE_CUTPOINT = 65

st.set_page_config(page_title="FAERS Signal Embedding Platform", layout="wide")


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def _download_db() -> None:
    """Download and reassemble the chunked database, verifying every part's
    size and SHA-256 before use, retrying a few times, and writing to a
    unique per-attempt temp file so two racing script runs can never
    corrupt each other's output."""
    last_exc: Exception | None = None
    for attempt in range(3):
        tmp_path = APP_DIR / f".faers_signals.duckdb.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            with open(tmp_path, "wb") as out:
                for url, expected_size, expected_sha256 in DB_PARTS:
                    with urllib.request.urlopen(url, timeout=120) as resp:
                        data = resp.read()
                    if len(data) != expected_size:
                        raise IOError(
                            f"{url} downloaded {len(data)} bytes, expected {expected_size}"
                        )
                    actual_sha256 = hashlib.sha256(data).hexdigest()
                    if actual_sha256 != expected_sha256:
                        raise IOError(
                            f"{url} sha256 {actual_sha256} != expected {expected_sha256}"
                        )
                    out.write(data)
            actual_size = tmp_path.stat().st_size
            if actual_size != DB_EXPECTED_SIZE:
                raise IOError(f"assembled size {actual_size} != expected {DB_EXPECTED_SIZE}")
            os.replace(tmp_path, DB_PATH)
            return
        except Exception as exc:
            last_exc = exc
            tmp_path.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download signal database after 3 attempts: {last_exc}")


@st.cache_resource
def get_connection() -> duckdb.DuckDBPyConnection:
    # A file left over from a previous, interrupted run (crash, race, bad
    # deploy) is indistinguishable from a good one by existence alone, so
    # size is checked too; anything else self-heals by re-downloading
    # rather than persisting a corrupted database across deploys.
    if not DB_PATH.exists() or DB_PATH.stat().st_size != DB_EXPECTED_SIZE:
        with st.spinner(
            "First launch: downloading the precomputed signal database "
            "(~220 MB, one-time)..."
        ):
            try:
                DB_PATH.unlink(missing_ok=True)
                _download_db()
            except Exception as exc:
                st.error(
                    f"Could not download the signal database.\n\n{exc}\n\n"
                    "Alternatively, run `python db_build.py` locally first to "
                    "precompute the signal tables from the raw CSV."
                )
                st.stop()
    return duckdb.connect(str(DB_PATH), read_only=True)


@st.cache_data
def get_drug_list(_con) -> pd.DataFrame:
    return _con.execute("""
        SELECT drug, COUNT(DISTINCT report_id) AS n_reports
        FROM exposure GROUP BY drug ORDER BY n_reports DESC
    """).fetchdf()


@st.cache_data
def get_outcomes_for_drug(_con, drug: str) -> pd.DataFrame:
    return _con.execute("""
        SELECT pt, n_reports FROM eligible_pairs
        WHERE drug = ? ORDER BY n_reports DESC
    """, [drug]).fetchdf()


@st.cache_data
def get_global_signals(_con, drug: str, pt: str) -> pd.DataFrame:
    return _con.execute("""
        SELECT * FROM signals_global WHERE drug = ? AND pt = ?
    """, [drug, pt]).fetchdf()


@st.cache_data
def get_annual_signals(_con, drug: str, pt: str) -> pd.DataFrame:
    return _con.execute("""
        SELECT * FROM signals_annual WHERE drug = ? AND pt = ? ORDER BY year
    """, [drug, pt]).fetchdf()


@st.cache_data
def get_subgroup_signals(_con, drug: str, pt: str) -> pd.DataFrame:
    return _con.execute("""
        SELECT * FROM signals_subgroup WHERE drug = ? AND pt = ?
    """, [drug, pt]).fetchdf()


@st.cache_data
def get_signal_features(_con, drug: str, pt: str) -> pd.DataFrame:
    return _con.execute("""
        SELECT * FROM signal_features WHERE drug = ? AND pt = ?
    """, [drug, pt]).fetchdf()


@st.cache_data
def get_drug_overview(_con, drug: str) -> pd.DataFrame:
    """All eligible outcomes for this drug, curated-universe ROR ranked --
    a single-row slice of the Drug x Outcome matrix (spec section 20.5)."""
    return _con.execute("""
        SELECT g.pt, g.n_reports, g.ror, g.ror_low, g.ror_high, g.prr, g.rf,
               g.continuity_applied, f.phenotype,
               c.ror AS ror_same_class, p.ror AS ror_pairwise
        FROM signals_global g
        LEFT JOIN signal_features f ON f.drug = g.drug AND f.pt = g.pt
        LEFT JOIN signals_global_class c ON c.drug = g.drug AND c.pt = g.pt
        LEFT JOIN signals_global_pairwise p ON p.drug = g.drug AND p.pt = g.pt
        WHERE g.drug = ? AND g.comparator = 'curated_universe'
        ORDER BY g.ror_low DESC
    """, [drug]).fetchdf()


@st.cache_data
def get_phenotype_mix(_con, drug: str) -> pd.DataFrame:
    return _con.execute("""
        SELECT phenotype, COUNT(*) AS n FROM signal_features
        WHERE drug = ? GROUP BY phenotype ORDER BY n DESC
    """, [drug]).fetchdf()


def pairwise_annual(_con, drug_a: str, drug_b: str, pt: str) -> pd.DataFrame:
    """Live pairwise comparator (spec C3): reference set D = drug_b only.
    Computed on demand against the (small) precomputed exposure/events
    tables -- this is the one case the spec says should recalculate
    interactively rather than being pre-baked for every possible pair."""
    q = """
        WITH exp_ab AS (
            SELECT report_id, year, drug FROM exposure WHERE drug IN (?, ?)
        ),
        evt AS (
            SELECT DISTINCT report_id, year FROM events WHERE pt = ?
        ),
        per_year AS (
            SELECT e.drug, e.year,
                   COUNT(DISTINCT e.report_id) AS n_drug,
                   COUNT(DISTINCT CASE WHEN v.report_id IS NOT NULL THEN e.report_id END) AS n_drug_event
            FROM exp_ab e
            LEFT JOIN evt v ON v.report_id = e.report_id AND v.year = e.year
            GROUP BY e.drug, e.year
        )
        SELECT * FROM per_year ORDER BY year
    """
    raw = _con.execute(q, [drug_a, drug_b, pt]).fetchdf()
    a_df = raw[raw["drug"] == drug_a].set_index("year")
    b_df = raw[raw["drug"] == drug_b].set_index("year")
    years = sorted(set(a_df.index) | set(b_df.index))
    rows = []
    for y in years:
        a = a_df["n_drug_event"].get(y, 0)
        n_a = a_df["n_drug"].get(y, 0)
        c = b_df["n_drug_event"].get(y, 0)
        n_b = b_df["n_drug"].get(y, 0)
        b = n_a - a
        d = n_b - c
        ror, lo, hi = stats.ror_from_counts(a, b, c, d)
        rows.append({"year": y, "a": a, "b": b, "c": c, "d": d,
                      "ror": ror, "ror_low": lo, "ror_high": hi, "n_reports": a})
    return pd.DataFrame(rows)


def pairwise_global(_con, drug_a: str, drug_b: str, pt: str) -> tuple[float, float, float, float]:
    q = """
        SELECT
            SUM(CASE WHEN e.drug = ? THEN 1 ELSE 0 END) AS n_a,
            SUM(CASE WHEN e.drug = ? THEN 1 ELSE 0 END) AS n_b,
            SUM(CASE WHEN e.drug = ? AND v.report_id IS NOT NULL THEN 1 ELSE 0 END) AS a,
            SUM(CASE WHEN e.drug = ? AND v.report_id IS NOT NULL THEN 1 ELSE 0 END) AS c
        FROM (SELECT DISTINCT report_id, drug FROM exposure WHERE drug IN (?, ?)) e
        LEFT JOIN (SELECT DISTINCT report_id FROM events WHERE pt = ?) v
          ON v.report_id = e.report_id
    """
    n_a, n_b, a, c = _con.execute(q, [drug_a, drug_b, drug_a, drug_b, drug_a, drug_b, pt]).fetchone()
    n_a, n_b, a, c = (n_a or 0), (n_b or 0), (a or 0), (c or 0)
    b = n_a - a
    d = n_b - c
    return stats.ror_from_counts(a, b, c, d) + (a,)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

con = get_connection()

st.title("FAERS Signal Embedding Platform")
st.caption(
    "Disproportionality signal explorer (ROR / PRR) across the curated "
    "medication universe, therapeutic-class, and pairwise comparators. "
    "Findings are hypothesis-generating measures of *reporting* "
    "disproportionality, not estimates of clinical risk."
)

drugs = get_drug_list(con)

with st.sidebar:
    st.header("Signal selection")
    drug = st.selectbox("Medication", drugs["drug"].tolist())

    outcomes = get_outcomes_for_drug(con, drug)
    if outcomes.empty:
        st.warning("No outcomes meet the minimum report threshold for this drug.")
        st.stop()
    outcome_labels = [f"{r.pt}  (n={r.n_reports})" for r in outcomes.itertuples()]
    outcome_idx = st.selectbox(
        "Adverse event (MedDRA PT)", range(len(outcome_labels)),
        format_func=lambda i: outcome_labels[i],
    )
    pt = outcomes.iloc[outcome_idx]["pt"]

    min_count = st.slider("Minimum reports/year to plot a point", 1, 20, MIN_REPORTS_DEFAULT)

    st.divider()
    st.subheader("Pairwise comparator")
    other_drugs = ["(none)"] + [d for d in drugs["drug"].tolist() if d != drug]
    rec_pair, rec_reason = phenotypes.DEFAULT_PAIRWISE.get(drug, (None, None))
    default_idx = other_drugs.index(rec_pair) if rec_pair in other_drugs else 0
    pairwise_drug = st.selectbox("Compare directly against", other_drugs, index=default_idx)
    if rec_pair:
        st.caption(f"Recommended pair: **{rec_pair}** -- {rec_reason}")

tab_profile, tab_landscape, tab_similar, tab_clusters = st.tabs(
    ["Signal profile", "Embedding landscape", "Similar signals", "Clusters & anomalies"]
)

with tab_profile:
    # ---- Global summary cards -------------------------------------------------

    globals_df = get_global_signals(con, drug, pt)
    curated_g = globals_df[globals_df["comparator"] == "curated_universe"].iloc[0]
    class_g_rows = globals_df[globals_df["comparator"] == "same_class"]
    class_g = class_g_rows.iloc[0] if not class_g_rows.empty else None

    st.subheader(f"{drug} -> {pt}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROR (curated universe)", f"{curated_g['ror']:.2f}",
              help=f"95% CI: {curated_g['ror_low']:.2f} - {curated_g['ror_high']:.2f}")
    c2.metric("PRR (curated universe)", f"{curated_g['prr']:.2f}")
    c3.metric("Reporting fraction", f"{curated_g['rf']:.3f}")
    c4.metric("Reports (this pair)", f"{int(curated_g['n_reports']):,}")
    signal_flag = stats.is_signal(curated_g["ror_low"], curated_g["n_reports"], min_count=MIN_REPORTS_DEFAULT)
    c5.metric("Signal (lower CI > 1)", "YES" if signal_flag else "no")

    if class_g is not None:
        st.caption(
            f"Same-class comparator ROR: {class_g['ror']:.2f} "
            f"(95% CI {class_g['ror_low']:.2f} - {class_g['ror_high']:.2f})"
        )

    feat = get_signal_features(con, drug, pt)
    if not feat.empty:
        frow = feat.iloc[0]
        if pd.notna(frow.get("ror_pairwise")) and pairwise_drug == frow.get("comparator_drug"):
            st.caption(
                f"Pairwise comparator ROR vs {frow['comparator_drug']}: {frow['ror_pairwise']:.2f} "
                f"(lower CI {frow['ror_low_pairwise']:.2f})"
            )

        # ---- Signal phenotype -----------------------------------------------
        meaning = {p[0]: (p[1], p[2]) for p in phenotypes.PHENOTYPES}
        ph_label = frow["phenotype"]
        ph_meaning, ph_action = meaning.get(ph_label, ("", ""))
        tags = [c.replace("tag_", "").replace("_", " ") for c in feat.columns
                if c.startswith("tag_") and bool(frow[c])]
        st.markdown(f"#### Signal phenotype: **{ph_label}**")
        st.caption(f"{ph_meaning}. Suggested action: {ph_action}."
                   + (f"  Also flagged: {', '.join(tags)}." if tags else ""))

    if bool(curated_g.get("continuity_applied", False)):
        st.warning(
            "One or more contingency-table cells were zero for this pair (most often "
            "because this event is reported for very few curated drugs). A +0.5 "
            "continuity correction was applied so ROR/PRR remain computable, but "
            "treat this estimate as statistically fragile rather than precise -- "
            "see spec sections 8.2 / 21.1 on zero-cell sensitivity."
        )

    # ---- Multi-year ROR trend --------------------------------------------------

    st.markdown("### ROR across years")
    annual = get_annual_signals(con, drug, pt)
    annual = annual[annual["n_reports"] >= min_count]

    fig = go.Figure()
    for comparator, name, color in [
        ("curated_universe", "Curated universe", "#4C78A8"),
        ("same_class", "Same therapeutic class", "#F58518"),
    ]:
        d = annual[annual["comparator"] == comparator].sort_values("year")
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["year"], y=d["ror"], mode="lines+markers", name=name,
            line=dict(color=color),
            error_y=dict(
                type="data",
                symmetric=False,
                array=d["ror_high"] - d["ror"],
                arrayminus=d["ror"] - d["ror_low"],
            ),
        ))

    if pairwise_drug != "(none)":
        pre = annual[(annual["comparator"] == "pairwise") & (annual.get("comparator_drug") == pairwise_drug)]
        pw = pre if not pre.empty else pairwise_annual(con, drug, pairwise_drug, pt)
        pw = pw[pw["n_reports"] >= min_count]
        if not pw.empty:
            fig.add_trace(go.Scatter(
                x=pw["year"], y=pw["ror"], mode="lines+markers",
                name=f"Pairwise vs {pairwise_drug}", line=dict(color="#54A24B", dash="dot"),
            ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="ROR = 1")
    fig.update_layout(
        xaxis_title="Year", yaxis_title="Reporting Odds Ratio",
        yaxis_type="log", height=450, legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, width='stretch')

    with st.expander("Show underlying annual counts and statistics"):
        st.dataframe(
            annual[["comparator", "year", "a", "b", "c", "d", "n_reports", "ror", "ror_low", "ror_high", "prr", "rf"]]
            .sort_values(["comparator", "year"]),
            width='stretch',
        )

    # ---- Temporal features ------------------------------------------------

    st.markdown("### Temporal behaviour")
    year_df = annual[annual["comparator"] == "curated_universe"][["year", "ror", "ror_low", "n_reports"]]
    temporal = stats.temporal_features(year_df, min_count=MIN_REPORTS_DEFAULT)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Persistence", f"{temporal['persistence']:.2f}" if not np.isnan(temporal["persistence"]) else "n/a",
              help="Fraction of eligible years where the lower CI bound exceeds 1")
    t2.metric("Trend (slope of log ROR/yr)", f"{temporal['trend']:.3f}" if not np.isnan(temporal["trend"]) else "n/a")
    t3.metric("Volatility (SD of log ROR)", f"{temporal['volatility']:.3f}" if not np.isnan(temporal["volatility"]) else "n/a")
    t4.metric("Recent acceleration", f"{temporal['recent_acceleration']:.3f}" if not np.isnan(temporal["recent_acceleration"]) else "n/a")

    # ---- Demographic heterogeneity -----------------------------------------

    st.markdown("### Demographic heterogeneity")
    sub = get_subgroup_signals(con, drug, pt)
    if not sub.empty:
        sub_plot = sub.dropna(subset=["ror"])
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=sub_plot["subgroup_value"], y=sub_plot["ror"],
            error_y=dict(
                type="data", symmetric=False,
                array=sub_plot["ror_high"] - sub_plot["ror"],
                arrayminus=sub_plot["ror"] - sub_plot["ror_low"],
            ),
            marker_color="#B279A2",
        ))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="gray")
        fig2.update_layout(height=350, yaxis_title="ROR", xaxis_title="Subgroup")
        st.plotly_chart(fig2, width='stretch')
    else:
        st.caption("Not enough subgroup data to display sex/age heterogeneity for this pair.")

    # ---- Class anomaly + comparator robustness -----------------------------

    # ---- Eight-dimensional fingerprint (spec 14 / 20.2) ---------------------

    st.markdown("### Signal fingerprint (eight-dimensional embedding)")
    if not feat.empty:
        f = feat.iloc[0]
        left, right = st.columns([1, 1])

        score_cols = [c + "_score" for c in phenotypes.FEATURE_COLS]
        labels = [phenotypes.FEATURE_LABELS[c] for c in phenotypes.FEATURE_COLS]
        vals = [f[c] if pd.notna(f[c]) else 0 for c in score_cols]
        radar = go.Figure(go.Scatterpolar(
            r=vals + vals[:1], theta=labels + labels[:1], fill="toself",
            line=dict(color="#4C78A8"),
            hovertemplate="%{theta}: %{r:.0f}th percentile<extra></extra>",
        ))
        radar.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100], showticklabels=True, ticks="")),
            showlegend=False, height=380, margin=dict(t=30, b=30),
        )
        left.plotly_chart(radar, width="stretch")
        left.caption("Each axis is the empirical percentile (0-100) of that dimension across all "
                     "eligible Drug x Outcome pairs. Missing dimensions plot at 0.")

        with right:
            r1, r2 = st.columns(2)
            r1.metric("Signal strength (S)", f"{f['signal_strength']:.2f}" if pd.notna(f["signal_strength"]) else "n/a",
                      help="ln(ROR) / (1 + SE)")
            r2.metric("Comparator robustness (R)", f"{f['comparator_robustness']:.2f}" if pd.notna(f["comparator_robustness"]) else "n/a",
                      help=f"Share of the {int(f['n_comparators'])} available comparators (curated, same-class, pairwise) with lower CI > 1")
            r3, r4 = st.columns(2)
            r3.metric("Class anomaly (C)", f"{f['class_anomaly']:.2f}" if pd.notna(f["class_anomaly"]) else "n/a",
                      help="z-score of ln(ROR) vs other drugs in the same ATC-4 class for this outcome (needs >= 3 class peers)")
            r4.metric("Uncertainty (U)", f"{f['uncertainty']:.3f}" if pd.notna(f["uncertainty"]) else "n/a",
                      help="Standard error of ln(ROR)")
            r5, r6 = st.columns(2)
            r5.metric("Demographic heterogeneity (H)", f"{f['demographic_heterogeneity']:.2f}" if pd.notna(f["demographic_heterogeneity"]) else "n/a")
            r6.metric("Volatility (V)", f"{f['volatility']:.2f}" if pd.notna(f["volatility"]) else "n/a")

    # ---- Phenotype mix for this drug --------------------------------------

    st.markdown(f"### Signal phenotypes across all {drug} outcomes")
    mix = get_phenotype_mix(con, drug)
    order = [p[0] for p in phenotypes.PHENOTYPES]
    mix["phenotype"] = pd.Categorical(mix["phenotype"], categories=order, ordered=True)
    mix = mix.sort_values("phenotype")
    bar = go.Figure(go.Bar(x=mix["n"], y=mix["phenotype"].astype(str), orientation="h",
                           marker_color="#72B7B2", hovertemplate="%{y}: %{x} outcomes<extra></extra>"))
    bar.update_layout(height=320, margin=dict(t=10, b=10), xaxis_title="Number of outcomes",
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(bar, width="stretch")

    # ---- Drug x Outcome overview for the selected drug ---------------------

    st.markdown(f"### All eligible outcomes for {drug} (curated-universe ROR)")
    overview = get_drug_overview(con, drug)
    ph_filter = st.multiselect("Filter by phenotype", order, default=[])
    if ph_filter:
        overview = overview[overview["phenotype"].isin(ph_filter)]
    st.caption("Sorted by the lower CI bound (conservative ranking). Rows flagged 'fragile' had a "
               "zero contingency-table cell before continuity correction -- treat their ROR as unstable.")
    st.dataframe(
        overview.rename(columns={
            "pt": "Adverse event", "n_reports": "N", "phenotype": "Phenotype", "ror": "ROR",
            "ror_low": "ROR low", "ror_high": "ROR high", "ror_same_class": "ROR same-class",
            "ror_pairwise": f"ROR vs {rec_pair}" if rec_pair else "ROR pairwise",
            "prr": "PRR", "rf": "Reporting fraction", "continuity_applied": "Fragile (zero-cell)",
        }),
        width='stretch', height=350,
    )

    st.download_button(
        "Download this drug's full signal table (CSV)",
        overview.to_csv(index=False),
        file_name=f"{drug}_signals.csv",
        mime="text/csv",
    )

    st.caption(
        "Reporting Odds Ratios measure disproportionate reporting in FAERS, a "
        "spontaneous-report database with no reliable exposure denominator. "
        "They do not estimate incidence or clinical risk, and associations "
        "are hypothesis-generating."
    )


# ===========================================================================
# ML layer (spec sections 15-18) -- reads tables written by ml_build.py
# ===========================================================================

@st.cache_data
def ml_available(_con) -> bool:
    n = _con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'signal_embeddings'"
    ).fetchone()[0]
    return n > 0


@st.cache_data
def get_embeddings(_con) -> pd.DataFrame:
    df = _con.execute("""
        SELECT e.*, x.drug_class
        FROM signal_embeddings e
        LEFT JOIN (SELECT DISTINCT drug, drug_class FROM exposure) x USING (drug)
    """).fetchdf()
    df["cluster_label"] = df["cluster"].map(lambda c: "noise" if c == -1 else f"cluster {c}")
    return df


@st.cache_data
def get_cluster_profiles(_con) -> pd.DataFrame:
    return _con.execute("SELECT * FROM cluster_profiles ORDER BY cluster").fetchdf()


@st.cache_data
def get_neighbors(_con, drug: str, pt: str) -> pd.DataFrame:
    return _con.execute("""
        SELECT n.rank, n.neighbor_drug AS drug, n.neighbor_pt AS pt, n.similarity,
               f.phenotype, f.ror_global, f.n_reports, e.cluster, e.anomaly_percentile,
               x.drug_class
        FROM signal_neighbors n
        JOIN signal_features f ON f.drug = n.neighbor_drug AND f.pt = n.neighbor_pt
        JOIN signal_embeddings e ON e.drug = n.neighbor_drug AND e.pt = n.neighbor_pt
        LEFT JOIN (SELECT DISTINCT drug, drug_class FROM exposure) x ON x.drug = n.neighbor_drug
        WHERE n.drug = ? AND n.pt = ?
        ORDER BY n.rank
    """, [drug, pt]).fetchdf()


ML_NOTE = ("Run `python ml_build.py` once to compute the UMAP / HDBSCAN / Isolation Forest / "
           "nearest-neighbour layer; this tab then activates.")

if not ml_available(con):
    for t in (tab_landscape, tab_similar, tab_clusters):
        with t:
            st.info(ML_NOTE)
else:
    emb = get_embeddings(con)
    prof = get_cluster_profiles(con)
    sel = emb[(emb["drug"] == drug) & (emb["pt"] == pt)]
    in_model = not sel.empty
    cluster_names = {int(r.cluster): (f"cluster {int(r.cluster)}: {r.majority_phenotype}"
                                      if r.cluster >= 0 else "noise") for r in prof.itertuples()}

    # ---- Tab: embedding landscape (spec 20.4) --------------------------
    with tab_landscape:
        st.markdown("### Embedding landscape (UMAP of the eight-dimensional signal space)")
        st.caption(f"{len(emb):,} disproportionate Drug x Outcome pairs with a usable annual series. "
                   "Each point is one signal; proximity means similar multidimensional behaviour, "
                   "not similar clinical meaning.")
        c1, c2, c3 = st.columns([1, 1, 2])
        color_by = c1.selectbox("Colour by", ["cluster", "phenotype", "anomaly_percentile",
                                              "drug_class", "signal_strength_pct", "trend", "persistence"])
        only_drug = c2.checkbox(f"Highlight only {drug}", value=False)
        emb_plot = emb.copy()
        emb_plot["signal_strength_pct"] = emb_plot["signal_strength"].rank(pct=True) * 100
        hover = ("<b>%{customdata[0]}</b> -> %{customdata[1]}<br>phenotype: %{customdata[2]}"
                 "<br>ROR %{customdata[3]:.1f}  N=%{customdata[4]}<br>%{customdata[5]}"
                 "<br>anomaly pct %{customdata[6]:.0f}<extra></extra>")
        cd = emb_plot[["drug", "pt", "phenotype", "ror_global", "n_reports", "cluster_label", "anomaly_percentile"]].to_numpy()

        fig_u = go.Figure()
        if color_by in ("cluster", "phenotype", "drug_class"):
            key = "cluster_label" if color_by == "cluster" else color_by
            for i, (lvl, g) in enumerate(emb_plot.groupby(key, dropna=False)):
                fig_u.add_trace(go.Scattergl(
                    x=g["umap_1"], y=g["umap_2"], mode="markers", name=str(lvl),
                    marker=dict(size=4, opacity=0.25 if only_drug else 0.6),
                    customdata=g[["drug", "pt", "phenotype", "ror_global", "n_reports", "cluster_label", "anomaly_percentile"]].to_numpy(),
                    hovertemplate=hover,
                ))
        else:
            fig_u.add_trace(go.Scattergl(
                x=emb_plot["umap_1"], y=emb_plot["umap_2"], mode="markers", name="signals",
                marker=dict(size=4, opacity=0.25 if only_drug else 0.7, color=emb_plot[color_by],
                            colorscale="Viridis", showscale=True, colorbar=dict(title=color_by)),
                customdata=cd, hovertemplate=hover,
            ))
        if only_drug:
            g = emb_plot[emb_plot["drug"] == drug]
            fig_u.add_trace(go.Scattergl(
                x=g["umap_1"], y=g["umap_2"], mode="markers", name=drug,
                marker=dict(size=7, color="#E45756", line=dict(width=0.5, color="white")),
                customdata=g[["drug", "pt", "phenotype", "ror_global", "n_reports", "cluster_label", "anomaly_percentile"]].to_numpy(),
                hovertemplate=hover,
            ))
        if in_model:
            fig_u.add_trace(go.Scatter(
                x=sel["umap_1"], y=sel["umap_2"], mode="markers", name=f"selected: {drug} -> {pt}",
                marker=dict(size=18, symbol="star", color="#F58518", line=dict(width=1.5, color="black")),
                hoverinfo="name",
            ))
        fig_u.update_layout(height=620, xaxis_title="UMAP 1", yaxis_title="UMAP 2",
                            legend=dict(orientation="h", y=-0.12), margin=dict(t=20))
        st.plotly_chart(fig_u, width="stretch")
        if not in_model:
            st.warning(f"{drug} -> {pt} is not in the modelled set (it needs a lower CI > 1 and at "
                       "least 3 eligible years), so it has no position on this map.")

    # ---- Tab: similar signals (spec 18 / 20.6) --------------------------
    with tab_similar:
        st.markdown(f"### Signals most similar to **{drug} -> {pt}**")
        if not in_model:
            st.warning("This pair is not in the modelled set, so it has no neighbours. "
                       "Pick a pair with lower CI > 1 and a multi-year history.")
        else:
            srow = sel.iloc[0]
            st.caption(f"Cluster: {cluster_names.get(int(srow['cluster']), srow['cluster'])} "
                       f"(membership probability {srow['cluster_probability']:.2f}) - "
                       f"Isolation-Forest anomaly percentile: {srow['anomaly_percentile']:.0f}")
            mode = st.radio("Retrieval mode", ["Unrestricted", "Exclude same drug", "Same outcome only",
                                               "Same class only", "Different class only"], horizontal=True)
            nb = get_neighbors(con, drug, pt)
            my_class = srow["drug_class"]
            if mode == "Exclude same drug":
                nb = nb[nb["drug"] != drug]
            elif mode == "Same outcome only":
                nb = nb[nb["pt"] == pt]
            elif mode == "Same class only":
                nb = nb[nb["drug_class"] == my_class]
            elif mode == "Different class only":
                nb = nb[nb["drug_class"] != my_class]
            if nb.empty:
                st.info("No neighbours match this mode within the top-15. (Neighbour lists are "
                        "precomputed to K=15; raise K_NEIGHBORS in ml_build.py for deeper retrieval.)")
            else:
                nb["cluster"] = nb["cluster"].map(lambda c: cluster_names.get(int(c), c))
                st.dataframe(
                    nb.rename(columns={"rank": "Rank", "drug": "Drug", "pt": "Adverse event",
                                       "similarity": "Cosine similarity", "phenotype": "Phenotype",
                                       "ror_global": "ROR", "n_reports": "N", "cluster": "Cluster",
                                       "anomaly_percentile": "Anomaly pct", "drug_class": "ATC-4 class"}),
                    width="stretch", hide_index=True,
                )
            st.caption("Similarity is cosine similarity between robust-scaled eight-dimensional "
                       "embeddings; it says two signals *behave* alike across strength, robustness, "
                       "time, subgroups and class context -- not that they share a mechanism.")

    # ---- Tab: clusters & anomalies (spec 16.2 / 17) ---------------------
    with tab_clusters:
        st.markdown("### HDBSCAN cluster profiles")
        st.caption("Median of each dimension per cluster, plus the majority rule-based phenotype. "
                   "Clusters are not auto-named (spec 16.2): read the profile, then label.")
        show = prof.copy()
        show["cluster"] = show["cluster"].map(lambda c: "noise (-1)" if c == -1 else str(c))
        med_cols = [c for c in show.columns if c.startswith("median_")]
        show = show[["cluster", "n_pairs", "n_drugs", "majority_phenotype", "majority_phenotype_share", "top_drugs"] + med_cols]
        show.columns = [c.replace("median_", "med ").replace("_", " ") for c in show.columns]
        st.dataframe(show.round(2), width="stretch", hide_index=True)

        st.markdown("### Phenotype x cluster agreement")
        ct = pd.crosstab(emb["cluster_label"], emb["phenotype"])
        st.dataframe(ct, width="stretch")
        st.caption("Where a cluster concentrates one phenotype, the unsupervised structure is "
                   "recovering the rule; where a cluster mixes phenotypes, the embedding is "
                   "grouping signals the rules keep apart -- those cells are worth reading.")

        st.markdown("### Highest-anomaly signals (Isolation Forest)")
        c1, c2 = st.columns([1, 3])
        top_n = c1.slider("Show top", 10, 100, 25, step=5)
        scope = c2.radio("Scope", ["All drugs", f"Only {drug}"], horizontal=True)
        an = emb if scope == "All drugs" else emb[emb["drug"] == drug]
        an = an.sort_values("anomaly_percentile", ascending=False).head(top_n)
        an = an[["drug", "pt", "anomaly_percentile", "phenotype", "cluster_label", "ror_global", "n_reports",
                 "signal_strength", "comparator_robustness", "persistence", "trend", "class_anomaly",
                 "demographic_heterogeneity"]]
        st.dataframe(an.round(2).rename(columns={"drug": "Drug", "pt": "Adverse event",
                                                 "anomaly_percentile": "Anomaly pct", "cluster_label": "Cluster",
                                                 "ror_global": "ROR", "n_reports": "N"}),
                     width="stretch", hide_index=True)
        st.caption("A high anomaly percentile means an unusual *combination* of dimensions, not the "
                   "largest ROR. Interpretation stays hypothesis-generating (spec section 25).")
