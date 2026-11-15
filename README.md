# Harvest Atlas — NASA-powered crop-rotation decision lab

Pick a field on the map, set your priorities and soil context, and get a field
diagnosis plus three AI rotation scenarios compared against your
current-practice baseline. Every number is traceable to a source; missing data is
reported as a gap rather than guessed.

## Data sources

- **NASA POWER** — temperature, rainfall, humidity, root-zone wetness, solar
  radiation, wind, growing degree days (no API key).
- **NASA GIBS** — MODIS Terra map overlays.
- **OpenLandMap SoilDB** (0–30 cm, 120 m) via OpenGeoHub TiTiler — sand, clay,
  silt, pH, organic carbon, bulk density (CC-BY 4.0).
- **NASA AppEEARS** (optional) — MODIS NDVI, land-surface temperature and SMAP
  soil moisture point samples; needs a free Earthdata Login.
- **OpenStreetMap Nominatim** — place names.
- **NASA SRTM** — terrain context for field zones.

## Run locally

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env      # then paste your Groq key
python main.py
```

Open the printed URL (default <http://127.0.0.1:5000>). The terminal also hosts a
Groq chat helper; `/help` lists its commands.

## Deploy

See [DEPLOY.md](DEPLOY.md). Short version: push to GitHub, create a Render
Blueprint from `render.yaml`, and set `GROQ_API_KEY` (plus optionally
`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`) in the dashboard.

## Layout

```
main.py            Flask app + NASA data pipeline + AI diagnosis
wsgi.py            production entry point (wsgi:app)
templates2/        the app page actually served at /
templates/         original page, kept locally for reference (not deployed)
render.yaml        Render Blueprint
Procfile           gunicorn start command
Dockerfile         portable image
```
