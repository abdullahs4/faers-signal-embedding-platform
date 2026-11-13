"""
Event families and drug reporting profiles -- a second unsupervised view
that complements the eight-dimensional pair embedding built by ml_build.py.

Why a second view
-----------------
The pair embedding (ml_build.py) places each Drug x Outcome pair by how the
signal *behaves*: strength, robustness, persistence, trend, volatility,
heterogeneity, class anomaly, uncertainty. Those features vary smoothly, so
the space is one continuum: k-means silhouettes stay at 0.2-0.3 for three or
more groups and HDBSCAN on the 8-D features finds no dense groups. The
HDBSCAN groups shown in the landscape are therefore regions of a continuum,
useful for navigation but not discovered signal types.

This module asks a different question: which adverse events are reported
disproportionately for the same medications? Each event is described by its
profile across the 54 medications (natural log of the curated-universe ROR
where the pair is disproportionate, 0 otherwise). Events with similar
profiles form families, and those families turn out to be clinically
recognisable reaction groups (bleeding for the anticoagulants; ketoacidosis
and amputation for the SGLT2 inhibitors; myopathy for the statins; tendon
disorders for the fluoroquinolones; ...). The same matrix read the other way
gives each medication's reporting profile, and clustering those recovers most
therapeutic classes without being told them.

Population: the disproportionate pairs modelled in signal_embeddings (lower
CI > 1 and >= 3 eligible years), so this view and the pair embedding cover
the same signals.

What is stored and what is computed live
----------------------------------------
Only the slow, seed-dependent part is precomputed: the UMAP + HDBSCAN family
label and display position of every event (data/event_families.csv), plus
validation statistics (data/families_meta.json). Family profiles, drug
similarities and the drug dendrogram are deterministic and take milliseconds,
so the dashboard derives them from the database with the functions below.
The event file has no names: rows follow the sorted event list, and the
dashboard checks the stored SHA-256 of that list before using it.

Run:  python family_build.py      (after ml_build.py; ~2 minutes)
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import phenotypes

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "faers_signals.duckdb"
OUT_DIR = APP_DIR / "data"
EVENTS_FILE = OUT_DIR / "event_families.csv"
META_FILE = OUT_DIR / "families_meta.json"

RANDOM_STATE = 0
ROR_CAP = 1000.0              # cap before log so a few zero-cell RORs do not dominate
EVENT_MIN_REPORTS = 5         # pair enters the event matrix with >= 5 reports
DRUG_MIN_REPORTS = 10         # pair enters the drug matrix with >= 10 reports
UMAP_CLUSTER = dict(n_neighbors=30, min_dist=0.10, n_components=2, metric="cosine")
UMAP_DISPLAY = dict(n_neighbors=100, min_dist=0.60, n_components=2, metric="cosine")
HDBSCAN_PARAMS = dict(min_cluster_size=25, min_samples=5)
STABILITY_SEEDS = [1, 2, 3]
N_PERMUTATIONS = 500
DRUG_GROUPS = 12
COVERAGE_DOMINANT = 0.6       # a class "dominates" a family if its drugs report >= 60% of the events
COVERAGE_MARGIN = 0.25        # ... and lead the next class by >= 25 points
DISPLAY_CORE_PCT = 85         # display layout: points inside this radius percentile are unchanged
DISPLAY_OUTER = 0.45          # ... the farthest point is placed at (1 + this) x that radius
DISPLAY_GRID = 1000           # display coordinates are stored as integers 0..DISPLAY_GRID


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Shared by the build and the dashboard
# ---------------------------------------------------------------------------

def prepare_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """pairs: drug, pt, n_reports, ror_global (the signal_embeddings rows)."""
    out = pairs[["drug", "pt", "n_reports", "ror_global"]].copy()
    out["v"] = np.log(out["ror_global"].clip(upper=ROR_CAP))
    return out


def event_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    """Event x drug matrix of log ROR (0 where not disproportionate); rows sorted by event."""
    p = pairs[pairs["n_reports"] >= EVENT_MIN_REPORTS]
    return p.pivot_table(index="pt", columns="drug", values="v", fill_value=0.0).sort_index()


def drug_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    """Drug x event matrix over events disproportionate for at least two drugs."""
    p = pairs[pairs["n_reports"] >= DRUG_MIN_REPORTS]
    M = p.pivot_table(index="drug", columns="pt", values="v", fill_value=0.0).sort_index()
    return M.loc[:, (M > 0).sum(axis=0) >= 2]


def event_order_hash(events) -> str:
    return hashlib.sha256("\n".join(events).encode("utf-8")).hexdigest()


def l2_rows(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def family_tables(pairs: pd.DataFrame, M: pd.DataFrame, labels: np.ndarray,
                  cls: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-event attributes and per-family profiles for given family labels."""
    present = M > 0
    classes = M.columns.map(cls)
    by_class_any = present.T.groupby(classes).max().T
    dominant: dict[int, str] = {}
    for k in np.unique(labels):
        if k == -1:
            continue
        cov = by_class_any[labels == k].mean().sort_values(ascending=False)
        lead, second = cov.iloc[0], (cov.iloc[1] if len(cov) > 1 else 0.0)
        dominant[int(k)] = cov.index[0] if (lead >= COVERAGE_DOMINANT and lead - second >= COVERAGE_MARGIN) else "Mixed"

    p = pairs[pairs["n_reports"] >= EVENT_MIN_REPORTS]
    totals = p.groupby("pt")["n_reports"].sum().reindex(M.index)
    events = pd.DataFrame({
        "pt": M.index, "family": labels.astype(int),
        "n_drugs": present.sum(axis=1).to_numpy(), "total_reports": totals.to_numpy(),
    })
    events["dominant_class"] = events["family"].map(lambda k: dominant.get(int(k), "Unassigned"))

    rows = []
    for k in sorted(dominant):
        idx = labels == k
        cov = present[idx].mean().sort_values(ascending=False)
        top_drugs = cov[cov >= 0.5] if (cov >= 0.5).any() else cov.head(3)
        top_events = events.loc[idx].sort_values("total_reports", ascending=False)["pt"].head(6)
        rows.append(dict(
            family=k, n_events=int(idx.sum()), dominant_class=dominant[k],
            drugs=", ".join(f"{d} ({v:.0%})" for d, v in top_drugs.head(6).items()),
            mean_drugs_per_event=float(present[idx].sum(axis=1).mean()),
            top_events=", ".join(t.title() for t in top_events),
            total_reports=int(events.loc[idx, "total_reports"].sum()),
        ))
    return events, pd.DataFrame(rows)


def drug_tables(M: pd.DataFrame, cls: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drug profiles (class, dendrogram order, group, nearest drug) and cosine similarities."""
    from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

    X = l2_rows(M.to_numpy())
    S = X @ X.T
    Z = linkage(X, method="average", metric="cosine")
    groups = fcluster(Z, DRUG_GROUPS, criterion="maxclust")
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    drugs = list(M.index)
    off = S.copy()
    np.fill_diagonal(off, -np.inf)
    nn = off.argmax(axis=1)
    prof = pd.DataFrame({
        "drug": drugs, "drug_class": [cls[d] for d in drugs], "group": groups.astype(int),
        "order": [leaves.index(i) for i in range(len(drugs))],
        "nearest_drug": [drugs[j] for j in nn],
        "nearest_similarity": [float(S[i, j]) for i, j in enumerate(nn)],
    })
    sim = pd.DataFrame(S, index=drugs, columns=drugs)
    return prof, sim


# ---------------------------------------------------------------------------
# Build only
# ---------------------------------------------------------------------------

def purity(labels: np.ndarray, dominant: np.ndarray) -> float:
    ks = [k for k in np.unique(labels) if k != -1]
    sizes = np.array([(labels == k).sum() for k in ks])
    pur = np.array([pd.Series(dominant[labels == k]).value_counts(normalize=True).iloc[0] for k in ks])
    return float(np.average(pur, weights=sizes))


def compress_outliers(xy: np.ndarray) -> np.ndarray:
    """Display only: pull far-off islands towards the centre so the main
    structure fills the plot. Points beyond the DISPLAY_CORE_PCT radius keep
    their direction and order of distance; the outermost lands at
    (1 + DISPLAY_OUTER) x that radius. Distances between islands carry no
    meaning in UMAP, so nothing interpretable is lost."""
    c = np.median(xy, axis=0)
    v = xy - c
    r = np.linalg.norm(v, axis=1)
    r0 = np.percentile(r, DISPLAY_CORE_PCT)
    if r.max() <= r0:
        return xy
    rn = np.where(r > r0, r0 + (r - r0) * (DISPLAY_OUTER * r0 / (r.max() - r0)), r)
    return c + v * (rn / np.maximum(r, 1e-12))[:, None]


def to_grid(xy: np.ndarray) -> np.ndarray:
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    scale = DISPLAY_GRID / (hi - lo).max()
    return np.rint((xy - lo) * scale).astype(int)


def main() -> None:
    import umap
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import adjusted_rand_score

    t0 = time.time()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    pairs = prepare_pairs(con.execute(
        "SELECT drug, pt, n_reports, ror_global FROM signal_embeddings").fetchdf())
    cls = {d: phenotypes.corrected_class(d, c) for d, c in
           con.execute("SELECT DISTINCT drug, drug_class FROM exposure").fetchall()}
    con.close()
    log(f"{len(pairs):,} modelled pairs from signal_embeddings")

    # ---- event families -----------------------------------------------------
    M = event_matrix(pairs)
    X = l2_rows(M.to_numpy())
    log(f"event matrix: {M.shape[0]:,} events x {M.shape[1]} drugs; UMAP (clustering layout)...")
    emb = umap.UMAP(random_state=RANDOM_STATE, **UMAP_CLUSTER).fit_transform(X)
    labels = HDBSCAN(**HDBSCAN_PARAMS).fit_predict(emb)
    log(f"HDBSCAN: {labels.max() + 1} families, noise {np.mean(labels == -1):.1%}; UMAP (display layout)...")
    disp = to_grid(compress_outliers(umap.UMAP(random_state=RANDOM_STATE, **UMAP_DISPLAY).fit_transform(X)))

    classes = M.columns.map(cls)
    event_dominant = M.T.groupby(classes).sum().T.idxmax(axis=1).to_numpy()
    obs = purity(labels, event_dominant)
    rng = np.random.default_rng(RANDOM_STATE)
    null = np.array([purity(rng.permutation(labels), event_dominant) for _ in range(N_PERMUTATIONS)])
    aris = []
    for seed in STABILITY_SEEDS:
        l2 = HDBSCAN(**HDBSCAN_PARAMS).fit_predict(umap.UMAP(random_state=seed, **UMAP_CLUSTER).fit_transform(X))
        both = (labels != -1) & (l2 != -1)
        aris.append(float(adjusted_rand_score(labels[both], l2[both])))
    log(f"purity {obs:.2f} (permuted mean {null.mean():.2f}, max {null.max():.2f}); seed ARI {aris}")

    # ---- drug profiles ------------------------------------------------------
    D = drug_matrix(pairs)
    prof, _ = drug_tables(D, cls)
    klass = prof["drug_class"].to_numpy()
    nn = prof["nearest_drug"].map({d: i for i, d in enumerate(prof["drug"])}).to_numpy()
    nn_same = float(np.mean(klass == klass[nn]))
    dnull = np.array([np.mean(q == q[nn]) for q in (rng.permutation(klass) for _ in range(N_PERMUTATIONS))])
    ari = float(adjusted_rand_score(klass, prof["group"]))
    log(f"drug matrix: {D.shape[0]} drugs x {D.shape[1]:,} events; nearest drug in same class "
        f"{nn_same:.0%} (permuted mean {dnull.mean():.1%}); ARI vs class {ari:.2f}")

    # ---- write --------------------------------------------------------------
    OUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"x": disp[:, 0], "y": disp[:, 1], "family": labels.astype(int)}).to_csv(
        EVENTS_FILE, index=False)
    meta = dict(
        event_order_sha256=event_order_hash(M.index),
        events=dict(n_events=int(M.shape[0]), n_drugs=int(M.shape[1]),
                    n_families=int(labels.max() + 1), noise_share=float(np.mean(labels == -1)),
                    purity=obs, purity_permuted_mean=float(null.mean()),
                    purity_permuted_max=float(null.max()), seed_ari=aris),
        drugs=dict(n_drugs=int(D.shape[0]), n_events=int(D.shape[1]), groups=DRUG_GROUPS,
                   nn_same_class=nn_same, nn_same_class_permuted_mean=float(dnull.mean()),
                   nn_same_class_permuted_max=float(dnull.max()), ari_vs_class=ari),
        params=dict(ror_cap=ROR_CAP, event_min_reports=EVENT_MIN_REPORTS,
                    drug_min_reports=DRUG_MIN_REPORTS, umap_cluster=UMAP_CLUSTER,
                    umap_display=UMAP_DISPLAY, display_core_pct=DISPLAY_CORE_PCT,
                    display_outer=DISPLAY_OUTER, display_grid=DISPLAY_GRID,
                    hdbscan=HDBSCAN_PARAMS, stability_seeds=STABILITY_SEEDS,
                    permutations=N_PERMUTATIONS, drug_groups=DRUG_GROUPS,
                    coverage_dominant=COVERAGE_DOMINANT, coverage_margin=COVERAGE_MARGIN,
                    random_state=RANDOM_STATE),
    )
    META_FILE.write_text(json.dumps(meta, indent=2))
    log(f"Wrote {EVENTS_FILE.name} and {META_FILE.name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
