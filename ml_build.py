"""
Phase 4 -- unsupervised discovery on the eight-dimensional signal embedding
(spec sections 14.2 and 15-18).

Reads signal_features from faers_signals.duckdb and adds:

  signal_embeddings   one row per modelled Drug x Outcome pair:
                        the 8 robust-scaled dimensions (x_*),
                        umap_1 / umap_2                    (UMAP, section 15)
                        cluster, cluster_probability       (HDBSCAN, section 16)
                        anomaly_raw, anomaly_percentile    (Isolation Forest, section 17)
  signal_neighbors    top-K nearest neighbours per pair, cosine similarity
                      in the scaled 8-D space (section 18); the dashboard
                      filters these into the spec's retrieval modes
  cluster_profiles    median of every dimension per cluster + the majority
                      rule-based phenotype, for post-hoc labelling (16.2)

Which pairs are modelled
------------------------
A pair enters the model when its longitudinal dimensions exist (>= 3
eligible years, so persistence / trend / volatility are defined). Two
dimensions are imputed rather than dropping the pair, because a strict
complete-case rule would leave only the four classes with >= 4 drugs:
  class_anomaly             missing -> 0   ("no evidence of deviation from
                                            class"); flag class_anomaly_imputed
  demographic_heterogeneity missing -> median; flag heterogeneity_imputed
The flags are stored so this choice can be tested in sensitivity analyses.

Stages are checkpointed to the local build directory so the script can be
re-run and resumes where it stopped.

Run:  python ml_build.py            (after db_build.py)
      python ml_build.py --reset    (recompute everything)
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import phenotypes

APP_DIR = Path(__file__).resolve().parent
FINAL_DB_PATH = APP_DIR / "faers_signals.duckdb"
BUILD_DIR = Path.home() / ".faers_build"
DB_PATH = BUILD_DIR / "faers_signals.duckdb"
CKPT = BUILD_DIR / "ml_ckpt"

RANDOM_STATE = 42
# Only disproportionate pairs (lower CI > 1) are modelled: the ~16k "no
# disproportionality" pairs form one dense featureless blob at the centre
# of the space and swamp every density-based method (first pass: 2
# clusters, 96% noise). The landscape/anomaly/neighbour questions in the
# spec are all questions about signals, so this is the natural population.
SIGNAL_ONLY = True
UMAP_PARAMS = dict(n_neighbors=25, min_dist=0.10, n_components=2, metric="euclidean")
# HDBSCAN is run on the 2-D UMAP coordinates rather than the raw 8-D
# space. Three of the eight dimensions are near-discrete (robustness takes
# 5 values, persistence is a fraction of <= 22 years, class anomaly is 0
# for the 78% of pairs whose class has < 4 drugs), which puts most of the
# density on a few hyperplanes; in 8-D HDBSCAN then finds two blobs split
# purely on robustness. On the UMAP manifold, leaf selection recovers six
# clusters that map cleanly onto distinct feature profiles (see
# cluster_profiles). Parameters were chosen on a grid of
# min_cluster_size x {eom, leaf}; the README records the alternatives.
HDBSCAN_ON = "umap"        # "umap" | "x"
HDBSCAN_PARAMS = dict(min_cluster_size=400, min_samples=20, cluster_selection_method="leaf")
ISO_PARAMS = dict(n_estimators=500, contamination="auto")
K_NEIGHBORS = 15

FEATURES = phenotypes.FEATURE_COLS  # S R P T V H C U


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------

def load_model_table(con) -> pd.DataFrame:
    df = con.execute(f"""
        SELECT drug, pt, n_reports, phenotype, ror_global, ror_low_global,
               {", ".join(FEATURES)}
        FROM signal_features
        WHERE persistence IS NOT NULL AND trend IS NOT NULL AND volatility IS NOT NULL
          AND signal_strength IS NOT NULL AND uncertainty IS NOT NULL
          {"AND ror_low_global > 1" if SIGNAL_ONLY else ""}
    """).fetchdf()
    df["class_anomaly_imputed"] = df["class_anomaly"].isna()
    df["class_anomaly"] = df["class_anomaly"].fillna(0.0)
    df["heterogeneity_imputed"] = df["demographic_heterogeneity"].isna()
    df["demographic_heterogeneity"] = df["demographic_heterogeneity"].fillna(
        df["demographic_heterogeneity"].median()
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES).reset_index(drop=True)
    return df


def stage_scale(df: pd.DataFrame) -> np.ndarray:
    p = CKPT / "X_scaled.npy"
    if p.exists():
        log("scaled matrix: checkpoint found")
        return np.load(p)
    from sklearn.preprocessing import RobustScaler
    log("RobustScaler on 8 dimensions (spec 14.2)...")
    X = RobustScaler().fit_transform(df[FEATURES].to_numpy(dtype=float))
    np.save(p, X)
    return X


def stage_umap(X: np.ndarray) -> np.ndarray:
    p = CKPT / "umap.npy"
    if p.exists():
        log("UMAP: checkpoint found")
        return np.load(p)
    import umap
    log(f"UMAP on {X.shape[0]:,} x {X.shape[1]} (this is the slow step)...")
    coords = umap.UMAP(random_state=RANDOM_STATE, **UMAP_PARAMS).fit_transform(X)
    np.save(p, coords)
    return coords


def stage_hdbscan(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = CKPT / "hdbscan.npz"
    if p.exists():
        log("HDBSCAN: checkpoint found")
        z = np.load(p)
        return z["labels"], z["prob"]
    from sklearn.cluster import HDBSCAN
    log("HDBSCAN...")
    model = HDBSCAN(**HDBSCAN_PARAMS).fit(X)
    np.savez(p, labels=model.labels_, prob=model.probabilities_)
    return model.labels_, model.probabilities_


def stage_isoforest(X: np.ndarray) -> np.ndarray:
    p = CKPT / "iso.npy"
    if p.exists():
        log("Isolation Forest: checkpoint found")
        return np.load(p)
    from sklearn.ensemble import IsolationForest
    log("Isolation Forest...")
    iso = IsolationForest(random_state=RANDOM_STATE, **ISO_PARAMS).fit(X)
    raw = -iso.decision_function(X)          # larger = more anomalous
    np.save(p, raw)
    return raw


def stage_neighbors(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = CKPT / "nn.npz"
    if p.exists():
        log("Nearest neighbours: checkpoint found")
        z = np.load(p)
        return z["idx"], z["sim"]
    from sklearn.neighbors import NearestNeighbors
    log(f"Nearest neighbours (cosine, K={K_NEIGHBORS})...")
    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    np.savez(p, idx=idx[:, 1:], sim=1 - dist[:, 1:])
    return idx[:, 1:], 1 - dist[:, 1:]


# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    reset = "--reset" in sys.argv
    if reset and CKPT.exists():
        shutil.rmtree(CKPT)
    CKPT.mkdir(parents=True, exist_ok=True)

    if not DB_PATH.exists():
        # first run on a machine where only the project-folder copy exists
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FINAL_DB_PATH, DB_PATH)
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"PRAGMA temp_directory='{BUILD_DIR / 'tmp'}'")

    df_path = CKPT / "model_table.parquet"
    if df_path.exists():
        df = pd.read_parquet(df_path)
        log(f"model table: checkpoint found ({len(df):,} pairs)")
    else:
        df = load_model_table(con)
        df.to_parquet(df_path)
        log(f"model table: {len(df):,} pairs with longitudinal features "
            f"({df['class_anomaly_imputed'].mean():.0%} class_anomaly imputed, "
            f"{df['heterogeneity_imputed'].mean():.0%} heterogeneity imputed)")

    X = stage_scale(df)
    coords = stage_umap(X)
    labels, prob = stage_hdbscan(coords if HDBSCAN_ON == "umap" else X)
    anomaly = stage_isoforest(X)
    nn_idx, nn_sim = stage_neighbors(X)

    # ---- assemble embeddings table ----------------------------------------
    emb = df.copy()
    for j, c in enumerate(FEATURES):
        emb[f"x_{c}"] = X[:, j]
    emb["umap_1"], emb["umap_2"] = coords[:, 0], coords[:, 1]
    emb["cluster"] = labels.astype(int)
    emb["cluster_probability"] = prob
    emb["anomaly_raw"] = anomaly
    emb["anomaly_percentile"] = pd.Series(anomaly).rank(pct=True).to_numpy() * 100
    emb["row_id"] = np.arange(len(emb))

    con.register("emb_df", emb)
    con.execute("CREATE OR REPLACE TABLE signal_embeddings AS SELECT * FROM emb_df")

    # ---- neighbours (long) -------------------------------------------------
    n, k = nn_idx.shape
    nb = pd.DataFrame({
        "row_id": np.repeat(np.arange(n), k),
        "rank": np.tile(np.arange(1, k + 1), n),
        "neighbor_row_id": nn_idx.ravel(),
        "similarity": nn_sim.ravel(),
    })
    key = emb[["row_id", "drug", "pt"]]
    nb = nb.merge(key, on="row_id").merge(
        key.rename(columns={"row_id": "neighbor_row_id", "drug": "neighbor_drug", "pt": "neighbor_pt"}),
        on="neighbor_row_id",
    )
    con.register("nb_df", nb)
    con.execute("CREATE OR REPLACE TABLE signal_neighbors AS SELECT * FROM nb_df")

    # ---- cluster profiles (post-hoc, spec 16.2) ----------------------------
    prof = emb.groupby("cluster").agg(
        n_pairs=("row_id", "size"),
        n_drugs=("drug", "nunique"),
        **{f"median_{c}": (c, "median") for c in FEATURES},
        median_anomaly_pct=("anomaly_percentile", "median"),
        majority_phenotype=("phenotype", lambda s: s.value_counts().index[0]),
        majority_phenotype_share=("phenotype", lambda s: s.value_counts(normalize=True).iloc[0]),
        top_drugs=("drug", lambda s: ", ".join(s.value_counts().index[:3])),
    ).reset_index()
    con.register("prof_df", prof)
    con.execute("CREATE OR REPLACE TABLE cluster_profiles AS SELECT * FROM prof_df")

    meta = dict(
        n_modelled=int(len(emb)), n_clusters=int((prof["cluster"] >= 0).sum()),
        noise_share=float((emb["cluster"] == -1).mean()),
        signal_only=SIGNAL_ONLY, umap=UMAP_PARAMS, hdbscan=HDBSCAN_PARAMS,
        hdbscan_on=HDBSCAN_ON, iso=ISO_PARAMS, k=K_NEIGHBORS,
        random_state=RANDOM_STATE, features=FEATURES,
    )
    con.execute("CREATE OR REPLACE TABLE ml_meta AS SELECT ? AS json", [json.dumps(meta)])
    con.close()

    log(f"clusters: {meta['n_clusters']}  noise share: {meta['noise_share']:.1%}")
    log("cluster profiles:\n" + prof[["cluster", "n_pairs", "n_drugs", "majority_phenotype",
                                       "majority_phenotype_share"]].to_string(index=False))
    log(f"Copying database onto the project folder ({FINAL_DB_PATH})...")
    shutil.copy2(DB_PATH, FINAL_DB_PATH)
    log(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
