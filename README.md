# FAERS Signal Embedding Platform -- ROR Dashboard (Phase 2 + partial Phase 5)

This is a working implementation of the statistical engine and main signal
screen described in `FAERS_Signal_Embedding_Platform.docx`, scoped to what
was asked for now: for any selected medication and adverse event, compute
the Reporting Odds Ratio across every year of data, plus the other signal
dimensions the spec recommends building alongside it.

## What it computes

For any Drug x Outcome (MedDRA PT) pair with at least 3 reports:

- **ROR and PRR**, annually and pooled across all years, with log-scale 95%
  confidence intervals (spec section 8)
- Under **three comparator contexts** (spec section 7): curated medication
  universe (all other drugs in the dataset), same ATC-4 therapeutic class,
  and an analyst-chosen pairwise comparator (computed live in the
  dashboard)
- **Temporal features**: persistence, trend, volatility, recent
  acceleration (spec section 9)
- **Demographic heterogeneity**: ROR by sex and by age (<65 / >=65) (spec
  section 10)
- **Therapeutic-class anomaly**: how unusual the drug's ROR is relative to
  its class peers for the same outcome (spec section 12)
- **Comparator robustness**: whether the signal survives across comparator
  contexts (spec section 11)

These are exactly the eight interpretable dimensions (`signal_features`
table) the spec proposes feeding into UMAP / HDBSCAN / Isolation Forest /
nearest-neighbour retrieval in a later phase (spec sections 14-18) -- that
ML layer is not built yet; see "Next steps" below.

## Data assumptions (please sanity-check with Anthony)

- Reports are counted at the **Safetyreport ID** level. The raw extract has
  heavy row-level duplication per report (up to >1000 rows for a single ID
  in some cases); the build always uses `COUNT(DISTINCT report_id)`, never
  raw row counts, so this duplication does not inflate signal counts. It is
  not full FAERS case-version deduplication (no CASEID/PRIMARYID/date
  fields are present in this extract to do that properly) -- worth flagging
  as a documented limitation in the methods section.
- "Exposed to drug A" = drug rows where `Role_Code` is `PS` (primary
  suspect) or `SS` (secondary suspect). Concomitant (`C`) and interacting
  (`I`) mentions are excluded from the exposure definition. This is the
  standard convention in disproportionality analysis; change
  `SUSPECT_ROLES` in `db_build.py` to test sensitivity to this choice (spec
  section 21.1 calls for exactly this kind of sensitivity check).
- The `Reactions` column is split on `" / "` to recover individual MedDRA
  PT-level events per report.
- The curated universe = the 55 medications present in this extract -- it
  is labelled "curated-universe comparator," not "all FAERS," per the
  spec's explicit warning in section 7.
- The raw `ATC-4 Class` field is inconsistently multi-valued (e.g.
  `Penicillins with extended spectrum` for one drug vs. `Penicillins with
  extended spectrum;Antibiotics` for a drug that should share the same
  class). The build takes only the first `;`-separated segment, which
  collapses this into 23 therapeutic classes (was 27 before normalizing;
  verified by inspecting every distinct drug/class pair -- no false merges
  observed). Worth a second look if the source data changes.
- Every ROR/PRR row carries a `continuity_applied` flag: True whenever a
  contingency-table cell was zero and the +0.5 continuity correction (spec
  8.2) had to be used. This happens for ~0.7% of curated-universe pairs but
  ~33% of same-class pairs (the smaller comparator population makes
  zero-cells far more common) -- the dashboard surfaces this as a "fragile"
  warning rather than presenting a five-digit ROR as a precise estimate.
- FAERS `Reactions` terms include non-clinical/administrative codes (e.g.
  "Drug ineffective," "Off label use," and occasional miscoded obstetric
  terms). These are legitimate MedDRA PTs in the source data and are not
  filtered out, but they are not "adverse events" in the clinical sense --
  worth a stoplist if Anthony wants outcome selection restricted to true
  clinical reactions.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Build the signal database (run once, re-run after any config change)

```bash
python db_build.py
```

This streams the ~10GB CSV with DuckDB (it is never loaded fully into
memory) and writes a small `faers_signals.duckdb` file with all precomputed
signal tables. Expect this to take a few minutes. Re-run it any time you
change `MIN_REPORTS`, `SUSPECT_ROLES`, `AGE_CUTPOINT`, or the source file.

## 1b. Build the ML layer (run once after step 1; ~1 minute)

```bash
python ml_build.py
```

RobustScaler -> UMAP -> HDBSCAN -> Isolation Forest -> nearest neighbours
(spec sections 14.2, 15-18) on the disproportionate pairs with a usable
annual series (~20k). Adds `signal_embeddings`, `signal_neighbors`,
`cluster_profiles` to the database and activates the three ML tabs in the
dashboard. Checkpointed under `~/.faers_build/ml_ckpt`; `--reset` recomputes.

## 1c. Export flat files

```bash
python export_results.py
```

Writes everything (statistics, phenotypes, embeddings, clusters,
neighbours, one wide table per drug) to `results/`.

## 2. Launch the dashboard

```bash
streamlit run dashboard.py
```

Pick a medication and an adverse event in the sidebar. The main panel
shows: summary ROR/PRR cards, a multi-year ROR trend chart (with
comparator overlay and an optional live pairwise comparator), temporal
behaviour metrics, demographic heterogeneity, class anomaly / comparator
robustness, and a ranked table of every eligible outcome for that drug
(downloadable as CSV).

## Files

- `stats.py` -- pure statistical functions (ROR, PRR, CI, temporal
  features, robustness, class anomaly, heterogeneity). No I/O; safe to
  unit-test directly, and reused by both the batch build and the
  dashboard's live pairwise comparator.
- `db_build.py` -- one-time batch build. Reads the raw CSV once and writes
  every precomputed table the dashboard needs, per the spec's own
  recommendation (section 23.1) not to recompute 2x2 tables on every click.
- `dashboard.py` -- the Streamlit app.

- `phenotypes.py` -- recommended pairwise comparator per drug
  (`DEFAULT_PAIRWISE`, with rationale), 0-100 display scores, and the
  rule-based signal phenotypes with their calibrated thresholds.
- `ml_build.py` -- Phase 4: scaling, UMAP, HDBSCAN, Isolation Forest,
  nearest neighbours. Parameters and the population choice are documented
  at the top of the file.
- `export_results.py` -- flat-file export; `results/README_results.txt`
  is the full column dictionary.

## Status against the spec roadmap

| Phase | Status |
|---|---|
| 1 Data harmonization | done (with the documented simplifications above) |
| 2 Statistical engine | done: ROR/PRR/CI, 3 comparators, annual series, subgroups |
| 3 Embedding | done: 8 dimensions, robust scaling, display scores, phenotypes |
| 4 ML | done: UMAP, HDBSCAN, Isolation Forest, k-NN retrieval |
| 5 Visual analytics | done: profile, fingerprint, comparator time series, landscape, similar-signal panel, cluster/anomaly tables. Not built: the full Drug x Outcome matrix heat-map view |
| 6 Validation | **not started** -- threshold sensitivity, bootstrap stability, UMAP seed/parameter stability, ARI for HDBSCAN, temporal hold-forward. This is where the paper's evidence comes from. |
| 7 Manuscript | -- |
