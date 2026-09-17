# Deploying to Streamlit Community Cloud

The app's code lives in this repo. The precomputed signal database
(`faers_signals.duckdb`, ~220 MB) is NOT committed as a single file --
GitHub blocks files over 100 MB in a normal repo. Instead it is split into
three <100MB chunks tracked in `data/` (`faers_signals.duckdb.part-00/01/02`),
and `dashboard.py` downloads and reassembles them into the app's working
directory the first time it starts (see `get_connection()` in dashboard.py,
which fetches each part from raw.githubusercontent.com). Subsequent reruns
in the same container reuse the reassembled file; if the Streamlit Cloud
container restarts, it re-downloads once (~10-20s).

## Steps

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click "Create app" -> "Yup, I have an app" (or "New app").
3. Repository: `abdullahs4/faers-signal-embedding-platform`, branch `main`,
   main file path: `dashboard.py`.
4. Deploy. The first load takes a bit longer while it downloads and
   reassembles the database -- subsequent loads are fast.

## Updating the database later

If Anthony's pipeline (`db_build.py` -> `ml_build.py`) produces a new
`faers_signals.duckdb`, split it the same way and replace the parts in
`data/`:

```bash
split -b 90m -a 2 -d faers_signals.duckdb data/faers_signals.duckdb.part-
git add data/ && git commit -m "Update signal database" && git push
```

Streamlit Cloud redeploys automatically on push to `main`. Since the
filenames don't change, also bump anything caching the old file (or just
delete the old `faers_signals.duckdb` in the deployed container via the
app's "Manage app" -> reboot).
