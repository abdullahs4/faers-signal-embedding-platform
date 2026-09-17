# Deploying to Streamlit Community Cloud

The app's code lives in this repo, but the precomputed signal database
(`faers_signals.duckdb`, ~220 MB) is NOT committed to git -- GitHub blocks
files over 100 MB in a normal repo. Instead it is attached to this repo's
"data-v1" GitHub Release, and `dashboard.py` downloads it automatically to
the app's working directory the first time it starts (see `get_connection()`
in dashboard.py). Subsequent reruns in the same session reuse the cached
file; if the Streamlit Cloud container restarts, it re-downloads once.

## Steps

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click "Create app" -> "Yup, I have an app" (or "New app").
3. Repository: `abdullahs4/faers-signal-embedding-platform`, branch `main`,
   main file path: `dashboard.py`.
4. Deploy. The first load will take a bit longer while it downloads the
   database from the Release asset -- subsequent loads are fast.

## Updating the database later

If Anthony's pipeline (`db_build.py` -> `ml_build.py`) produces a new
`faers_signals.duckdb`, upload the new file to a new GitHub Release (e.g.
tag `data-v2`) and update `DB_DOWNLOAD_URL` in `dashboard.py` to point at it,
then push -- Streamlit Cloud redeploys automatically on push to `main`.
