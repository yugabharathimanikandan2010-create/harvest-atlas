#!/usr/bin/env python3
"""
Harvest Atlas (main): NASA-powered crop-rotation decision support.

This version serves its own page from the `templates2` folder. The NASA POWER
pipeline provides temperature, rainfall, humidity, root-zone wetness, solar
radiation, wind and growing degree days for the selected field.

What main adds on top of that data:
    - measured 0-30 cm soil texture, pH, organic carbon and bulk density from
      OpenLandMap SoilDB (sampled through OpenGeoHub TiTiler)
    - optional NASA AppEEARS point samples (MODIS NDVI, land-surface temperature
      and SMAP soil moisture) fetched in the background
    - a field vulnerability diagnosis (heat, water, erosion, volatility, soil)
    - farmer priorities on a 0-5 scale, turned into transparent score weights
    - local soil context (texture, organic matter, irrigation, field size,
      current rotation)
    - three AI rotation scenarios compared against the current-practice baseline
    - per-scenario scores, evidence, trade-offs, confidence and data gaps

Setup:
    pip install -r requirements.txt
    PowerShell:  $env:GROQ_API_KEY="your_key_here"
    CMD:         set GROQ_API_KEY=your_key_here
    (or put it in a .env file next to this script)

    Optional satellite samples: create a free Earthdata Login account and set
    EARTHDATA_USERNAME and EARTHDATA_PASSWORD in .env. Without them the app still
    works and reports satellite coverage as a gap.

Run:
    python main.py

Then open the printed map URL (default http://127.0.0.1:5000), tap a field or
use GPS, set your priorities and soil context, and press "Diagnose this field".
The place name and NASA agro-climatology are logged in the terminal, and the
same data feeds the diagnosis and the scenario comparison in the browser.

Commands while chatting:
    /help          show commands
    /model [name]  show or change the model
    /clear         forget the conversation so far
    /exit          quit (Ctrl+C / Ctrl+D also work)

Data sources:
    NASA POWER (https://power.larc.nasa.gov) - free, no API key needed.
    NASA GIBS (https://gibs.earthdata.nasa.gov) - MODIS Terra map overlays.
    OpenLandMap SoilDB (https://openlandmap.org) via OpenGeoHub TiTiler - global
        0-30 cm soil properties (CC-BY 4.0), no API key needed.
    NASA AppEEARS (https://appeears.earthdatacloud.nasa.gov) - optional MODIS and
        SMAP point samples; requires free Earthdata Login credentials.
    OpenStreetMap Nominatim - reverse geocoding for place names.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import base64
import json
import math
import os
import re
import socket
import sys
import textwrap
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Never crash on characters the Windows console cannot encode.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass

try:
    import groq
except ImportError:
    sys.exit("Missing dependency 'groq'. Install it with: pip install groq")

try:
    from flask import Flask, jsonify, render_template, request
except ImportError:
    sys.exit("Missing dependency 'flask'. Install it with: pip install -r requirements.txt")

DEFAULT_MODEL = "openai/gpt-oss-120b"
SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."
RECOMMEND_SYSTEM_PROMPT = (
    "You are an experienced agronomist and farm decision-support analyst. You help "
    "a farmer compare crop-rotation scenarios using NASA POWER agro-climatology "
    "(temperature, rainfall, humidity, root-zone soil moisture, solar radiation, "
    "growing degree days) plus the farmer's priorities and local soil context. "
    "You are honest about trade-offs, uncertainty and missing data. You never "
    "promise yields. Reply with a single JSON object and nothing else."
)
EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}

# Looked up defensively so the script keeps working across SDK versions.
AUTH_ERROR = getattr(groq, "AuthenticationError", ())
RATE_LIMIT_ERROR = getattr(groq, "RateLimitError", ())
CONNECTION_ERROR = getattr(groq, "APIConnectionError", ())
API_ERROR = getattr(groq, "APIError", Exception)

# --- Web map + app state ----------------------------------------------------

MAP_HOST = "127.0.0.1"
MAP_PORT = 5000

# Actual port once the server starts (moves to the next free port if 5000 is taken).
map_port = MAP_PORT

# Latest location picked on the web map, e.g.
# {"lat": 13.0827, "lng": 80.2707, "accuracy": 25.0}
# (accuracy is in meters for GPS picks, or None for manual clicks)
# None until the user presses "Select" or their browser shares GPS.
selected_location = None

# Polished location + NASA POWER data for the selected spot, e.g.
# {"location": {...}, "nasa": {...}, "errors": []}
environment_data = None

# AI crop-rotation advice generated from environment_data, e.g.
# {"summary": ..., "rotations": [...], "caveats": ..., "model": ...}
crop_recommendations = None

# Groq client and active model. The terminal builds the client in main(); the
# web app can also build it on demand from the environment, so /recommend works
# no matter how the server was started.
groq_client = None
current_model = (os.environ.get("GROQ_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL

# Extra Groq keys let one rejected or rate-limited key fall through to the next
# instead of failing the whole recommendation. Keys are read from GROQ_API_KEY,
# GROQ_API_KEY_2, GROQ_API_KEY_3 and a comma-separated GROQ_API_KEYS.
_groq_clients: dict = {}
_groq_key_index = 0

_location_lock = threading.Lock()
_groq_lock = threading.Lock()

_environment_cache: dict = {}
_ENVIRONMENT_TTL = 600  # seconds

# main keeps its own themed template folder so the original page stays untouched.
app = Flask(__name__, template_folder="templates2")


# --- NASA POWER + reverse geocoding ----------------------------------------

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

# Nominatim requires a descriptive User-Agent.
USER_AGENT = "nasa-farm-crop-assistant/0.1 (local demo)"

WINDOW_DAYS = 30
POWER_PARAMS = [
    "T2M",               # average air temperature at 2 m (C)
    "T2M_MAX",           # max air temperature (C)
    "T2M_MIN",           # min air temperature (C)
    "PRECTOTCORR",       # corrected precipitation (mm/day)
    "RH2M",              # relative humidity at 2 m (%)
    "WS2M",              # wind speed at 2 m (m/s)
    "ALLSKY_SFC_SW_DWN", # solar radiation at surface (MJ/m2/day)
    "GWETROOT",          # root-zone soil wetness (0-1 fraction)
]

_geocode_cache: dict = {}
_cache_lock = threading.Lock()
_last_geocode_at = 0.0


def _get_json(url: str, timeout: float = 30.0) -> dict:
    request_ = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request_, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def reverse_geocode(lat: float, lng: float) -> dict:
    """Turn coordinates into a human-readable place name (cached)."""
    global _last_geocode_at

    key = (round(lat, 4), round(lng, 4))
    with _cache_lock:
        if key in _geocode_cache:
            return _geocode_cache[key]

    # Nominatim asks for at most one request per second.
    wait = 1.1 - (time.monotonic() - _last_geocode_at)
    if wait > 0:
        time.sleep(wait)

    query = urllib.parse.urlencode(
        {
            "format": "jsonv2",
            "lat": f"{lat:.6f}",
            "lon": f"{lng:.6f}",
            "zoom": 10,
            "addressdetails": 1,
        }
    )
    data = _get_json(f"{NOMINATIM_URL}?{query}")
    _last_geocode_at = time.monotonic()

    address = data.get("address") or {}
    locality = (
        address.get("village")
        or address.get("town")
        or address.get("city")
        or address.get("suburb")
        or address.get("hamlet")
    )
    parts = [
        part
        for part in (locality, address.get("county"), address.get("state"), address.get("country"))
        if part
    ]
    # Drop side-by-side duplicates (e.g. "Chennai, Chennai, Tamil Nadu").
    unique_parts = []
    for part in parts:
        if not unique_parts or unique_parts[-1].lower() != part.lower():
            unique_parts.append(part)

    result = {
        "name": ", ".join(unique_parts)
        or data.get("display_name")
        or f"{lat:.4f}, {lng:.4f}",
        "display_name": data.get("display_name"),
        "country": address.get("country"),
    }

    with _cache_lock:
        _geocode_cache[key] = result
    return result


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _iso(day: str) -> str:
    """Convert a POWER date key (YYYYMMDD) to ISO (YYYY-MM-DD)."""
    return f"{day[:4]}-{day[4:6]}-{day[6:]}"


def fetch_power(lat: float, lng: float, days: int = WINDOW_DAYS) -> dict:
    """Fetch daily agro-climatology for the last `days` days from NASA POWER."""
    end = dt.date.today() - dt.timedelta(days=2)  # POWER data lags a little
    start = end - dt.timedelta(days=days - 1)

    query = urllib.parse.urlencode(
        {
            "parameters": ",".join(POWER_PARAMS),
            "community": "AG",
            "latitude": f"{lat:.4f}",
            "longitude": f"{lng:.4f}",
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON",
        }
    )
    data = _get_json(f"{POWER_URL}?{query}", timeout=45)
    parameters = data["properties"]["parameter"]

    series: dict[str, dict[str, float]] = {}
    for name in POWER_PARAMS:
        raw = parameters.get(name) or {}
        series[name] = {
            day: float(value)
            for day, value in raw.items()
            if isinstance(value, (int, float)) and value > -900  # -999 = fill
        }

    dates = sorted({day for values in series.values() for day in values})
    if not dates:
        return {
            "source": "NASA POWER (agroclimatology, community=AG)",
            "period": None,
            "temperature_c": {"avg": None, "min": None, "max": None},
            "rainfall_mm": {"last_7_days": None, "last_30_days": None},
            "humidity_pct": None,
            "soil_moisture_root_zone": None,
            "solar_radiation_mj_m2_day": None,
            "solar_radiation_max_mj_m2_day": None,
            "wind_speed_ms": None,
            "wind_speed_max_ms": None,
            "growing_degree_days_30d": None,
            "series": {
                "dates": [],
                "temp_c": [],
                "temp_max_c": [],
                "temp_min_c": [],
                "rain_mm": [],
                "humidity_pct": [],
                "wind_ms": [],
                "solar_mj": [],
                "root_wetness": [],
            },
        }

    recent = dates[-7:]

    def values(name: str, subset: list[str]) -> list[float]:
        return [series[name][day] for day in subset if day in series[name]]

    def aligned(name: str) -> list[float | None]:
        return [series[name].get(day) for day in dates]

    temp = values("T2M", recent)
    temp_min = values("T2M_MIN", recent)
    temp_max = values("T2M_MAX", recent)
    rain_recent = values("PRECTOTCORR", recent)
    rain_all = values("PRECTOTCORR", dates)
    humidity = values("RH2M", recent)
    wind = values("WS2M", recent)
    solar = values("ALLSKY_SFC_SW_DWN", recent)
    soil = values("GWETROOT", recent)

    # Growing degree days (base 10 C) over the full window.
    gdd = 0.0
    for day in dates:
        t_max = series["T2M_MAX"].get(day)
        t_min = series["T2M_MIN"].get(day)
        if t_max is not None and t_min is not None:
            gdd += max(0.0, (t_max + t_min) / 2.0 - 10.0)

    return {
        "source": "NASA POWER (agroclimatology, community=AG)",
        "period": {
            "start": _iso(dates[0]),
            "end": _iso(dates[-1]),
            "days": len(dates),
        },
        "temperature_c": {
            "avg": _round(_mean(temp), 1),
            "min": _round(min(temp_min), 1) if temp_min else None,
            "max": _round(max(temp_max), 1) if temp_max else None,
        },
        "rainfall_mm": {
            "last_7_days": _round(sum(rain_recent), 1) if rain_recent else None,
            "last_30_days": _round(sum(rain_all), 1) if rain_all else None,
        },
        "humidity_pct": _round(_mean(humidity), 0),
        "soil_moisture_root_zone": _round(_mean(soil), 2),
        "solar_radiation_mj_m2_day": _round(_mean(solar), 1),
        "solar_radiation_max_mj_m2_day": _round(max(solar), 1) if solar else None,
        "wind_speed_ms": _round(_mean(wind), 1),
        "wind_speed_max_ms": _round(max(wind), 1) if wind else None,
        "growing_degree_days_30d": _round(gdd, 0),
        "series": {
            "dates": [_iso(day) for day in dates],
            "temp_c": aligned("T2M"),
            "temp_max_c": aligned("T2M_MAX"),
            "temp_min_c": aligned("T2M_MIN"),
            "rain_mm": aligned("PRECTOTCORR"),
            "humidity_pct": aligned("RH2M"),
            "wind_ms": aligned("WS2M"),
            "solar_mj": aligned("ALLSKY_SFC_SW_DWN"),
            "root_wetness": aligned("GWETROOT"),
        },
    }


# --- Soil properties (OpenLandMap SoilDB via OpenGeoHub TiTiler) ------------
#
# SoilGrids' REST endpoint (rest.isric.org) was unreachable from this network,
# so the app samples the OpenLandMap SoilDB 2020-2022 global mosaics instead.
# These are 120 m, 0-30 cm Cloud-Optimized GeoTIFFs read at the selected point
# by OpenGeoHub's TiTiler deployment. Values are real predictions from that
# dataset and are never invented: if the service is down, `available` is false
# and the gap is reported instead of guessed.

SOIL_COG_BASE = "https://ceph.opengeohub.org/arco/soildb"
TITILER_BASE = "https://titiler.opengeohub.org"

# key -> (COG filename, scale factor, raw nodata, unit, label)
SOIL_LAYERS = {
    "sand_pct": (
        "soildbtxt_sand_m_120m_bg0..30cm_20200101_20221231_go01_epsg4326_v1.0.0_srl04.tif",
        1.0,
        255.0,
        "%",
        "Sand",
    ),
    "clay_pct": (
        "soildbtxt_clay_m_120m_bg0..30cm_20200101_20221231_go01_epsg4326_v1.0.0_srl04.tif",
        1.0,
        255.0,
        "%",
        "Clay",
    ),
    "silt_pct": (
        "soildbtxt_silt_m_120m_bg0..30cm_20200101_20221231_go01_epsg4326_v1.0.0_srl04.tif",
        1.0,
        255.0,
        "%",
        "Silt",
    ),
    "ph": (
        "soildbpc_phh2o_m_120m_bg0..30cm_20200101_20221231_go01_epsg4326_v1.2.0_srl04.tif",
        0.1,
        255.0,
        "pH",
        "pH in water",
    ),
    "organic_carbon_g_kg": (
        "soildbpc_ccwpml_m_120m_bg0..30cm_20200101_20221231_go01_epsg4326_v1.2.0_srl04.tif",
        0.1,
        32767.0,
        "g/kg",
        "Organic carbon",
    ),
    "bulk_density_g_cm3": (
        "soildbpc_bdcore_m_120m_bg0..30cm_20200101_20221231_go01_epsg4326_v1.0.0_srl04.tif",
        0.01,
        32767.0,
        "g/cm3",
        "Bulk density",
    ),
}

SOIL_SOURCE = (
    "OpenLandMap SoilDB 2020-2022 (0-30 cm, 120 m) via OpenGeoHub TiTiler"
)
SOIL_LICENSE = "CC-BY 4.0 (OpenLandMap / OpenGeoHub)"


def _soil_point(
    lat: float, lng: float, filename: str, scale: float, nodata: float
) -> float | None:
    """Read one raw band value at a point and apply the layer scale factor."""
    cog = f"{SOIL_COG_BASE}/{filename}"
    query = urllib.parse.urlencode({"url": cog})
    url = f"{TITILER_BASE}/cog/point/{lng:.6f},{lat:.6f}?{query}"
    data = _get_json(url, timeout=25)
    values = data.get("values") or []
    if not values or values[0] is None:
        return None
    raw = float(values[0])
    if nodata is not None and abs(raw - nodata) < 1e-6:
        return None
    return raw * scale


def _usda_texture_class(sand: float, clay: float) -> str | None:
    """Approximate USDA-NRCS texture class from sand and clay percentages."""
    if sand is None or clay is None:
        return None
    silt = 100.0 - sand - clay
    if min(sand, clay, silt) < -0.5:
        return None

    if clay >= 40:
        if sand > 45:
            return "sandy clay"
        if silt > 40:
            return "silty clay"
        return "clay"
    if clay >= 35:
        if sand > 45:
            return "sandy clay"
        if silt > 40:
            return "silty clay"
        return "clay loam"
    if clay >= 27:
        if sand > 45:
            return "sandy clay loam"
        if silt > 40:
            return "silty clay loam"
        return "clay loam"
    if clay >= 20:
        if sand > 45:
            return "sandy clay loam"
        if silt >= 50:
            return "silt loam"
        return "loam"
    if silt >= 80:
        return "silt"
    if silt >= 50:
        return "silt loam"
    if sand >= 85:
        return "sand"
    if sand >= 70:
        return "loamy sand" if clay < 15 else "sandy loam"
    if sand >= 43:
        return "sandy loam"
    if clay >= 7 and silt >= 28:
        return "loam"
    return "sandy loam"


def _organic_matter_class(percent: float | None) -> str | None:
    if percent is None:
        return None
    if percent < 1.0:
        return "low"
    if percent < 2.0:
        return "medium"
    return "high"


def fetch_soil(lat: float, lng: float) -> dict:
    """Sample OpenLandMap soil properties at a point in parallel; never raises."""
    result = {
        "available": False,
        "source": SOIL_SOURCE,
        "license": SOIL_LICENSE,
        "depth_cm": "0-30",
        "resolution_m": 120,
        "errors": [],
    }
    values: dict[str, float | None] = {}

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(_soil_point, lat, lng, filename, scale, nodata): key
            for key, (filename, scale, nodata, _unit, _label) in SOIL_LAYERS.items()
        }
        try:
            for future in as_completed(futures, timeout=50):
                key = futures[future]
                try:
                    values[key] = future.result()
                except Exception as exc:  # noqa: BLE001 - per-layer best effort
                    result["errors"].append(f"{key}: {exc}")
        except TimeoutError:
            result["errors"].append("soil point queries timed out")

    sand = values.get("sand_pct")
    clay = values.get("clay_pct")
    silt = values.get("silt_pct")
    if silt is None and sand is not None and clay is not None:
        silt = max(0.0, 100.0 - sand - clay)

    ph = values.get("ph")
    soc = values.get("organic_carbon_g_kg")
    bulk = values.get("bulk_density_g_cm3")
    # Soil organic carbon (g/kg) -> organic matter percent, using the common
    # Van Bemmelen factor 1.724 (OM = SOC x 1.724, and percent = g/kg / 10).
    organic_matter = soc * 0.1724 if soc is not None else None

    result.update(
        {
            "sand_pct": _round(sand, 1),
            "clay_pct": _round(clay, 1),
            "silt_pct": _round(silt, 1),
            "texture_class": _usda_texture_class(sand, clay),
            "ph": _round(ph, 1),
            "organic_carbon_g_kg": _round(soc, 1),
            "organic_matter_pct": _round(organic_matter, 2),
            "organic_matter_class": _organic_matter_class(organic_matter),
            "bulk_density_g_cm3": _round(bulk, 2),
        }
    )
    result["available"] = any(
        value is not None for value in (sand, clay, silt, ph, soc, bulk)
    )
    if not result["available"] and not result["errors"]:
        result["errors"].append("no soil values were returned for this point")
    return result


# --- AppEEARS satellite point samples ---------------------------------------
#
# AppEEARS is asynchronous and needs a free Earthdata Login account. It is
# optional: without EARTHDATA_USERNAME / EARTHDATA_PASSWORD the app reports the
# gap and keeps working instead of inventing satellite values. The task runs in
# a background thread so /select never waits on NASA's queue.

APPEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"
APPEARS_WINDOW_DAYS = 60
APPEARS_LAYERS = [
    {
        "key": "ndvi",
        "product": "MOD13Q1.061",
        "layer": "_250m_16_days_NDVI",
        "label": "MODIS NDVI (250 m, 16-day)",
        "unit": "",
        "scale": 0.0001,
        "fill": -3000.0,
        "range": (-1.0, 2.0),
    },
    {
        "key": "lst_day_k",
        "product": "MOD11A2.061",
        "layer": "LST_Day_1km",
        "label": "MODIS daytime land-surface temperature (1 km, 8-day)",
        "unit": "K",
        "scale": 0.02,
        "fill": 0.0,
        "range": (200.0, 400.0),
    },
    {
        "key": "sm_surface",
        "product": "SPL4SMGP.008",
        "layer": "Geophysical_Data_sm_surface",
        "label": "SMAP surface soil moisture (9 km, daily)",
        "unit": "m3/m3",
        "scale": 1.0,
        "fill": -9999.0,
        "range": (0.0, 1.5),
    },
    {
        "key": "sm_rootzone",
        "product": "SPL4SMGP.008",
        "layer": "Geophysical_Data_sm_rootzone",
        "label": "SMAP root-zone soil moisture (9 km, daily)",
        "unit": "m3/m3",
        "scale": 1.0,
        "fill": -9999.0,
        "range": (0.0, 1.5),
    },
]

_satellite_lock = threading.Lock()
_satellite_generation = 0
_satellite_state = {
    "status": "idle",  # idle | queued | running | done | error | unavailable
    "point": None,
    "task_id": None,
    "message": "No satellite task has been started yet.",
    "readings": None,
    "errors": [],
    "submitted_at": None,
    "updated_at": None,
}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _satellite_snapshot() -> dict:
    with _satellite_lock:
        return json.loads(json.dumps(_satellite_state))


def _set_satellite_state(**changes) -> dict:
    with _satellite_lock:
        _satellite_state.update(changes)
        _satellite_state["updated_at"] = _now_iso()
        return json.loads(json.dumps(_satellite_state))


def _update_satellite_if_current(generation: int, **changes) -> dict | None:
    """Update state only if this background run is still the newest one."""
    with _satellite_lock:
        if _satellite_generation != generation:
            return None
        _satellite_state.update(changes)
        _satellite_state["updated_at"] = _now_iso()
        return json.loads(json.dumps(_satellite_state))


def _satellite_age_seconds(state: dict) -> float | None:
    stamp = state.get("updated_at")
    if not stamp:
        return None
    try:
        when = dt.datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return (dt.datetime.now(dt.timezone.utc) - when).total_seconds()


def _appeears_credentials() -> tuple[str, str]:
    return (
        os.environ.get("EARTHDATA_USERNAME", "").strip(),
        os.environ.get("EARTHDATA_PASSWORD", "").strip(),
    )


class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the Earthdata bearer token to the S3 download host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_request = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_request is not None:
            old_host = urllib.parse.urlparse(req.full_url).netloc
            new_host = urllib.parse.urlparse(newurl).netloc
            if old_host != new_host:
                new_request.remove_header("Authorization")
        return new_request


_APPEARS_DOWNLOAD_OPENER = urllib.request.build_opener(_DropAuthOnRedirect)


def _appeears_request(
    url: str, token: str | None = None, payload: dict | None = None, timeout: float = 45.0
) -> dict:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request_ = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request_, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body.strip() else {}


def _appeears_login() -> str:
    """Exchange Earthdata Login credentials for an AppEEARS bearer token.

    AppEEARS uses HTTP Basic auth plus a client_credentials grant, not a JSON
    body: POST /login with Basic auth and form data grant_type=client_credentials.
    """
    username, password = _appeears_credentials()
    if not username or not password:
        raise RuntimeError(
            "Earthdata credentials are not configured. Add EARTHDATA_USERNAME and "
            "EARTHDATA_PASSWORD to .env to enable AppEEARS satellite samples."
        )

    basic = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    form = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("ascii")
    request_ = urllib.request.Request(
        f"{APPEARS_BASE}/login",
        data=form,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        },
        method="POST",
    )
    with urllib.request.urlopen(request_, timeout=45) as response:
        body = response.read().decode("utf-8")

    data = json.loads(body) if body.strip() else {}
    token = data.get("token")
    if not token:
        message = data.get("message") if isinstance(data, dict) else None
        raise RuntimeError(
            "AppEEARS login did not return a token"
            + (f": {message}" if message else ".")
        )
    return str(token)


def _appeears_submit(lat: float, lng: float, token: str) -> str:
    end = dt.date.today() - dt.timedelta(days=2)
    start = end - dt.timedelta(days=APPEARS_WINDOW_DAYS - 1)
    safe_point = f"{lat:.4f}_{lng:.4f}".replace("-", "m").replace(".", "p")
    task_name = (
        f"harvest_atlas_{safe_point}_"
        f"{dt.datetime.now(dt.timezone.utc):%Y%m%d%H%M%S}"
    )
    payload = {
        "task_type": "point",
        "task_name": task_name,
        "params": {
            "dates": [
                {
                    "startDate": start.strftime("%m-%d-%Y"),
                    "endDate": end.strftime("%m-%d-%Y"),
                }
            ],
            "layers": [
                {"product": item["product"], "layer": item["layer"]}
                for item in APPEARS_LAYERS
            ],
            "coordinates": [
                {
                    "latitude": round(lat, 4),
                    "longitude": round(lng, 4),
                    "id": "field",
                }
            ],
        },
    }
    data = _appeears_request(f"{APPEARS_BASE}/task", token=token, payload=payload)
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError(f"AppEEARS did not return a task id: {data}")
    return str(task_id)


def _appeears_progress(task_id: str, token: str) -> int | None:
    """Return AppEEARS' own percent-complete summary for a task, if available."""
    try:
        data = _appeears_request(f"{APPEARS_BASE}/status/{task_id}", token=token)
    except Exception:  # noqa: BLE001 - progress is a nicety, never fatal
        return None
    entries = data if isinstance(data, list) else [data]
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("task_id") != task_id:
            continue
        summary = (entry.get("progress") or {}).get("summary")
        if isinstance(summary, (int, float)):
            return int(summary)
    return None


def _appeears_wait(
    task_id: str,
    token: str,
    timeout: float = 3600.0,
    interval: float = 20.0,
    on_progress=None,
) -> tuple[str, dict]:
    """Poll until AppEEARS finishes. NASA's queue can take a while for points."""
    deadline = time.monotonic() + timeout
    polls = 0
    while time.monotonic() < deadline:
        data = _appeears_request(f"{APPEARS_BASE}/task/{task_id}", token=token)
        status = str(data.get("status") or "").lower()
        if status in ("done", "failed", "error", "cancelled"):
            return status, data
        polls += 1
        if on_progress is not None and status in ("queued", "processing"):
            percent = _appeears_progress(task_id, token) if polls % 2 == 0 else None
            try:
                on_progress(status, percent)
            except Exception:  # noqa: BLE001 - a status callback must not break the wait
                pass
        time.sleep(interval)
    raise TimeoutError(
        "AppEEARS did not finish within an hour. NASA may still complete the "
        "task; press Diagnose again later to reuse it."
    )


def _appeears_value(raw, spec: dict) -> float | None:
    """Parse one CSV cell, dropping fill values and fixing raw-scale integers."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in ("na", "nan", "none", "null", "no data"):
        return None
    try:
        value = float(text)
    except ValueError:
        return None

    fill = float(spec.get("fill", -9999.0))
    if abs(value - fill) < 1e-6:
        return None

    # AppEEARS normally applies the product scale factor already. If a value is
    # clearly still in raw integer space, apply the documented scale once.
    low, high = spec.get("range", (None, None))
    if low is not None and not (low <= value <= high):
        scaled = value * float(spec.get("scale", 1.0))
        if low <= scaled <= high:
            value = scaled
        else:
            return None
    return value


def _appeears_date(raw) -> str | None:
    """Normalise an AppEEARS Date cell to YYYY-MM-DD."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(text[:19], fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _parse_appeears_csv(text: str) -> dict:
    """Turn an AppEEARS point-sample CSV into per-layer summary readings."""
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    fieldnames = reader.fieldnames or []
    readings: dict[str, dict] = {}

    date_column = next(
        (name for name in fieldnames if name.strip().lower() in ("date", "observation_date")),
        None,
    )

    for spec in APPEARS_LAYERS:
        # Only the data column, never the QA companions AppEEARS adds.
        columns = [
            name
            for name in fieldnames
            if spec["layer"] in name
            and not any(
                skip in name.lower()
                for skip in ("quality", "_qc", "bitmask", "description")
            )
        ]
        values: list[float] = []
        latest = None
        series: list[dict] = []
        for row in rows:
            row_value = None
            for column in columns:
                parsed = _appeears_value(row.get(column), spec)
                if parsed is not None:
                    row_value = parsed
                    values.append(parsed)
            if row_value is not None:
                latest = row_value
                if date_column:
                    day = _appeears_date(row.get(date_column))
                    if day:
                        series.append({"date": day, "value": round(row_value, 4)})
        if values:
            series.sort(key=lambda point: point["date"])
            readings[spec["key"]] = {
                "label": spec["label"],
                "unit": spec["unit"],
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "latest": round(latest, 4) if latest is not None else None,
                "samples": len(values),
                "series": series[-120:],
            }
    return readings


def _appeears_readings(task_id: str, token: str) -> dict:
    """Download every product CSV in the bundle and merge their readings.

    AppEEARS returns one CSV per requested product, so a MODIS + SMAP request
    yields several files. Reading only one of them silently loses layers.
    """
    bundle = _appeears_request(f"{APPEARS_BASE}/bundle/{task_id}", token=token)
    files = bundle.get("files") or []
    csv_files = [
        item
        for item in files
        if str(item.get("file_name", "")).lower().endswith(".csv")
    ]
    if not csv_files:
        raise RuntimeError("AppEEARS bundle did not contain a CSV point sample")

    # Skip any QA companion files, then read the rest.
    csv_files.sort(key=lambda item: "qa" in str(item.get("file_name", "")).lower())

    readings: dict[str, dict] = {}
    failures: list[str] = []
    for item in csv_files:
        url = f"{APPEARS_BASE}/bundle/{task_id}/{item['file_id']}"
        request_ = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Authorization": f"Bearer {token}"},
        )
        try:
            with _APPEARS_DOWNLOAD_OPENER.open(request_, timeout=120) as response:
                text = response.read().decode("utf-8-sig", errors="replace")
            parsed = _parse_appeears_csv(text)
        except Exception as exc:  # noqa: BLE001 - one bad file must not lose the rest
            failures.append(f"{item.get('file_name')}: {exc}")
            continue
        for key, value in parsed.items():
            readings.setdefault(key, value)

    if not readings and failures:
        raise RuntimeError("AppEEARS CSVs could not be read: " + "; ".join(failures))
    return readings


def _run_satellite_task(lat: float, lng: float, generation: int) -> None:
    def update(**changes) -> bool:
        return _update_satellite_if_current(generation, **changes) is not None

    try:
        if not update(status="running", message="Signing in to NASA Earthdata..."):
            return
        token = _appeears_login()

        if not update(message="Submitting the AppEEARS point task..."):
            return
        task_id = _appeears_submit(lat, lng, token)
        if not update(
            task_id=task_id,
            message="AppEEARS task queued. Waiting for NASA to process it...",
        ):
            return

        def progress(state: str, percent: int | None) -> None:
            if percent is None:
                text = f"NASA AppEEARS task is {state}. Waiting for it to finish..."
            else:
                text = f"NASA AppEEARS task is {state} ({percent}% complete)."
            update(message=text)

        status, task = _appeears_wait(task_id, token, on_progress=progress)
        if status != "done":
            raise RuntimeError(f"AppEEARS task ended with status '{status}'")

        if not update(message="Downloading the AppEEARS point sample..."):
            return
        readings = _appeears_readings(task_id, token)
        update(
            status="done",
            readings=readings,
            message=(
                "Satellite point sample ready."
                if readings
                else "AppEEARS finished but returned no usable values for this point."
            ),
            errors=[],
        )
    except Exception as exc:  # noqa: BLE001 - surface the failure, keep the app up
        _update_satellite_if_current(
            generation, status="error", message=str(exc), errors=[str(exc)]
        )


def start_satellite_task(lat: float, lng: float) -> dict:
    """Start (or reuse) a background AppEEARS task for this point."""
    global _satellite_generation

    point = {"lat": round(lat, 4), "lng": round(lng, 4)}
    username, password = _appeears_credentials()
    if not username or not password:
        return _set_satellite_state(
            status="unavailable",
            point=point,
            task_id=None,
            readings=None,
            errors=[],
            message=(
                "AppEEARS is optional and not configured. Add EARTHDATA_USERNAME and "
                "EARTHDATA_PASSWORD to .env to sample MODIS and SMAP here."
            ),
        )

    current = _satellite_snapshot()
    same_point = current.get("point") == point
    status = current.get("status")
    if same_point and status in ("queued", "running"):
        return current
    if same_point and status == "done":
        age = _satellite_age_seconds(current)
        if age is not None and age < 6 * 3600:
            return current

    with _satellite_lock:
        _satellite_generation += 1
        generation = _satellite_generation

    _set_satellite_state(
        status="queued",
        point=point,
        task_id=None,
        readings=None,
        errors=[],
        submitted_at=_now_iso(),
        message="Starting the AppEEARS satellite sample...",
    )
    threading.Thread(
        target=_run_satellite_task,
        args=(lat, lng, generation),
        daemon=True,
    ).start()
    return _satellite_snapshot()


def satellite_status(lat: float | None = None, lng: float | None = None) -> dict:
    """Return the latest AppEEARS task state, optionally checked against a point."""
    snapshot = _satellite_snapshot()
    if lat is not None and lng is not None:
        snapshot["matches_selection"] = snapshot.get("point") == {
            "lat": round(lat, 4),
            "lng": round(lng, 4),
        }
    return snapshot


def collect_environment(lat: float, lng: float, accuracy: float | None = None) -> dict:
    """Reverse geocode + NASA POWER + soil in parallel; never raises."""
    errors: list[str] = []
    location = {
        "name": f"{lat:.4f}, {lng:.4f}",
        "lat": lat,
        "lng": lng,
        "accuracy": accuracy,
    }
    nasa = None
    climatology = None
    soil = None
    terrain = None

    with ThreadPoolExecutor(max_workers=5) as pool:
        geo_future = pool.submit(reverse_geocode, lat, lng)
        power_future = pool.submit(fetch_power, lat, lng)
        climate_history_future = pool.submit(fetch_power, lat, lng, 365)
        soil_future = pool.submit(fetch_soil, lat, lng)
        terrain_future = pool.submit(fetch_terrain, lat, lng)

        try:
            geo = geo_future.result(timeout=30)
            location["name"] = geo["name"]
            location["display_name"] = geo.get("display_name")
            location["country"] = geo.get("country")
        except Exception as exc:  # noqa: BLE001 - best effort
            errors.append(f"reverse geocoding failed: {exc}")

        try:
            nasa = power_future.result(timeout=60)
        except Exception as exc:  # noqa: BLE001 - best effort
            errors.append(f"NASA POWER failed: {exc}")

        try:
            climatology = _seasonal_history(climate_history_future.result(timeout=60))
        except Exception as exc:  # noqa: BLE001 - best effort
            climatology = {
                "available": False,
                "note": f"Seasonal history unavailable: {exc}",
            }

        try:
            soil = soil_future.result(timeout=70)
        except Exception as exc:  # noqa: BLE001 - best effort
            errors.append(f"soil lookup failed: {exc}")

        try:
            terrain = terrain_future.result(timeout=30)
        except Exception as exc:  # noqa: BLE001 - best effort
            errors.append(f"terrain lookup failed: {exc}")

    if soil is None:
        soil = {
            "available": False,
            "source": SOIL_SOURCE,
            "license": SOIL_LICENSE,
            "depth_cm": "0-30",
            "resolution_m": 120,
            "errors": ["soil lookup did not complete"],
        }
    elif not soil.get("available"):
        errors.append("soil data unavailable for this point")

    if terrain is None:
        terrain = {
            "available": False,
            "source": TERRAIN_SOURCE,
            "resolution_m": 90,
            "errors": ["terrain lookup did not complete"],
        }
    elif not terrain.get("available"):
        errors.append("terrain lookup unavailable for this point")

    return {
        "location": location,
        "nasa": nasa,
        "climatology": climatology,
        "soil": soil,
        "terrain": terrain,
        "errors": errors,
    }


def get_environment(lat: float, lng: float, accuracy: float | None = None) -> dict:
    """collect_environment() with a short cache so repeat picks stay fast."""
    key = (round(lat, 4), round(lng, 4))
    now = time.monotonic()

    with _cache_lock:
        cached = _environment_cache.get(key)
        if cached is not None and now - cached[0] < _ENVIRONMENT_TTL:
            stored = cached[1]
            location = dict(stored["location"], accuracy=accuracy)
            return {
                "location": location,
                "nasa": stored["nasa"],
                "climatology": stored.get("climatology"),
                "soil": stored.get("soil"),
                "terrain": stored.get("terrain"),
                "errors": list(stored["errors"]),
            }

    environment = collect_environment(lat, lng, accuracy)
    with _cache_lock:
        _environment_cache[key] = (now, environment)
    return environment


def _fmt(value: float | None, digits: int = 1, unit: str = "") -> str:
    if value is None:
        return "n/a"
    text = f"{value:.{digits}f}"
    return f"{text} {unit}".strip() if unit else text


_UNICODE_FIXES = {
    "\u202f": " ",
    "\u00a0": " ",
    "\u2009": " ",
    "\u2013": "-",
    "\u2014": "-",
    "\u2011": "-",
    "\u2010": "-",
    "\u2248": "~",
    "\u00d7": "x",
    "\u00b2": "2",
    "\u00b3": "3",
    "\u207b": "-",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2022": "-",
    "\u2026": "...",
    "\u2192": "->",
    "\u2190": "<-",
}


def _safe_text(text: str) -> str:
    """Normalize fancy unicode so any terminal can print it."""
    text = (
        text.replace("\u00b0C", " deg C")
        .replace("\u00b0F", " deg F")
        .replace("\u00b0", " deg ")
    )
    for bad, good in _UNICODE_FIXES.items():
        text = text.replace(bad, good)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _soil_label(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.20:
        return "dry"
    if value < 0.40:
        return "slightly dry"
    if value < 0.60:
        return "adequate"
    return "moist"


def log_environment(environment: dict) -> None:
    """Print a clean, farmer-friendly summary to the terminal."""
    location = environment.get("location", {})
    nasa = environment.get("nasa")

    line = "=" * 62
    print(f"\n{line}")
    print("  SELECTED LOCATION")
    print(line)
    print(f"  {_safe_text(location.get('name', 'unknown'))}")

    coords = f"  Lat {location.get('lat', 0):.6f} | Lng {location.get('lng', 0):.6f}"
    accuracy = location.get("accuracy")
    if accuracy is not None:
        coords += f" | GPS accuracy ~{accuracy:.0f} m"
    print(coords)

    print()
    if not nasa:
        print("  NASA POWER AGRO-CLIMATOLOGY: data unavailable right now")
    else:
        period = nasa.get("period")
        window = (
            f" ({period['start']} -> {period['end']}, {period['days']} days)"
            if period
            else ""
        )
        print(f"  NASA POWER AGRO-CLIMATOLOGY{window}")
        print(f"  {'-' * 58}")

        temp = nasa["temperature_c"]
        print(
            f"  Temperature     : {_fmt(temp['avg'], 1, 'C')} avg"
            f"  (min {_fmt(temp['min'], 1, 'C')}, max {_fmt(temp['max'], 1, 'C')})"
        )

        rain = nasa["rainfall_mm"]
        print(
            f"  Rainfall        : {_fmt(rain['last_7_days'], 1, 'mm')} last 7 days"
            f" | {_fmt(rain['last_30_days'], 1, 'mm')} last 30 days"
        )

        print(f"  Humidity        : {_fmt(nasa['humidity_pct'], 0, '%')}")

        soil = nasa["soil_moisture_root_zone"]
        print(
            f"  Soil moisture   : {_fmt(soil, 2)}"
            f" (root zone - {_soil_label(soil)})"
        )

        print(
            f"  Solar radiation : {_fmt(nasa['solar_radiation_mj_m2_day'], 1, 'MJ/m2/day')}"
        )
        print(f"  Wind speed      : {_fmt(nasa['wind_speed_ms'], 1, 'm/s')}")
        print(
            f"  Growing degree  : {_fmt(nasa['growing_degree_days_30d'], 0, 'GDD')}"
            " (base 10 C)"
        )

    print()
    field_soil = environment.get("soil") or {}
    if field_soil.get("available"):
        print("  SOIL (OpenLandMap SoilDB, 0-30 cm, 120 m)")
        print(f"  {'-' * 58}")
        print(
            f"  Texture         : {field_soil.get('texture_class') or 'n/a'}"
            f"  (sand {_fmt(field_soil.get('sand_pct'), 0, '%')},"
            f" clay {_fmt(field_soil.get('clay_pct'), 0, '%')},"
            f" silt {_fmt(field_soil.get('silt_pct'), 0, '%')})"
        )
        print(
            f"  pH in water     : {_fmt(field_soil.get('ph'), 1)}"
            f" | Organic matter: {_fmt(field_soil.get('organic_matter_pct'), 2, '%')}"
            f" ({field_soil.get('organic_matter_class') or 'n/a'})"
        )
        print(
            f"  Bulk density    : {_fmt(field_soil.get('bulk_density_g_cm3'), 2, 'g/cm3')}"
            f" | Organic carbon: {_fmt(field_soil.get('organic_carbon_g_kg'), 1, 'g/kg')}"
        )
    else:
        print("  SOIL (OpenLandMap SoilDB): data unavailable right now")

    satellite = environment.get("satellite") or {}
    if satellite:
        print(
            f"  SATELLITE       : {satellite.get('status', 'idle')}"
            f" - {_safe_text(str(satellite.get('message') or ''))}"
        )

    for error in environment.get("errors") or []:
        print(f"  ! {_safe_text(str(error))}")

    print(f"{line}\n", flush=True)


# --- AI crop-rotation advice ------------------------------------------------


def _extract_json(text: str) -> dict:
    """Parse JSON from a model reply, tolerating code fences or extra prose."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError("AI reply was not valid JSON")


def _ascii_text(value: str) -> str:
    """Force AI prose to plain ASCII so terminals and files stay predictable."""
    text = str(value)
    for bad, good in _UNICODE_FIXES.items():
        text = text.replace(bad, good)
    normalized = unicodedata.normalize("NFKD", text)
    cleaned = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", cleaned).strip()


def _ascii_tree(value):
    """Recursively sanitize every string in the model's JSON reply."""
    if isinstance(value, str):
        return _ascii_text(value)
    if isinstance(value, list):
        return [_ascii_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _ascii_tree(item) for key, item in value.items()}
    return value


def _nasa_for_prompt(nasa: dict | None) -> dict:
    if not nasa:
        return {"note": "NASA POWER data unavailable for this point"}
    return {
        "window_days": (nasa.get("period") or {}).get("days"),
        "temperature_c": nasa.get("temperature_c"),
        "rainfall_mm": nasa.get("rainfall_mm"),
        "humidity_pct": nasa.get("humidity_pct"),
        "soil_moisture_root_zone_0_1": nasa.get("soil_moisture_root_zone"),
        "solar_radiation_mj_m2_day": nasa.get("solar_radiation_mj_m2_day"),
        "wind_speed_ms": nasa.get("wind_speed_ms"),
        "growing_degree_days_base10c_30d": nasa.get("growing_degree_days_30d"),
    }


# Farmer priorities map to the six scenario score dimensions. The browser sends
# importance on a 0-5 scale; the server turns that into weights and computes one
# transparent overall score per scenario.
PRIORITY_KEYS = {
    "water": "water_security",
    "soil": "soil_health",
    "climate": "climate_resilience",
    "income": "income_stability",
    "cost": "cost_efficiency",
    "labour": "labour_efficiency",
}

PRIORITY_LABELS = {
    "water": "Water security",
    "soil": "Soil health",
    "climate": "Climate resilience",
    "income": "Income stability",
    "cost": "Input cost",
    "labour": "Labour and time",
}

DEFAULT_PRIORITIES = {
    "water": 4,
    "soil": 3,
    "climate": 4,
    "income": 3,
    "cost": 3,
    "labour": 2,
}

SCORE_KEYS = list(PRIORITY_KEYS.values())

# Missions that informed the original concept but are NOT queried by this app.
# They are shown separately so the UI never implies data it did not fetch.
MISSION_NOT_QUERIED = ["ECOSTRESS", "GPM", "Landsat", "GRACE-FO"]


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _concern(score: float) -> str:
    """Turn a 0-100 safety score into a plain concern level."""
    if score >= 70:
        return "low"
    if score >= 45:
        return "moderate"
    return "high"


def _clean_priorities(priorities: dict | None) -> dict:
    cleaned = {}
    for key in PRIORITY_KEYS:
        raw = (priorities or {}).get(key, DEFAULT_PRIORITIES[key])
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float(DEFAULT_PRIORITIES[key])
        cleaned[key] = _clamp(value, 0.0, 5.0)
    return cleaned


def _clean_context(context: dict | None) -> dict:
    """Keep farmer-supplied context short, typed and safe to echo back."""
    context = context if isinstance(context, dict) else {}

    def as_text(key: str, limit: int) -> str:
        value = context.get(key)
        return str(value).strip()[:limit] if value is not None else ""

    def as_number(key: str, low: float, high: float) -> float | None:
        try:
            value = float(context.get(key))
        except (TypeError, ValueError):
            return None
        if not (low <= value <= high):
            return None
        return value

    try:
        size = float(context.get("field_size_ha"))
        if size <= 0 or size > 100000:
            size = None
    except (TypeError, ValueError):
        size = None

    return {
        "soil_texture": as_text("soil_texture", 40),
        "organic_matter": as_text("organic_matter", 20),
        "irrigation": as_text("irrigation", 40),
        "irrigation_status": as_text("irrigation_status", 40),
        "field_size_ha": size,
        "current_rotation": as_text("current_rotation", 120),
        "polygon_area_ha": as_number("polygon_area_ha", 0.0001, 100000),
        "soil_test_ph": as_number("soil_test_ph", 0.0, 14.0),
        "soil_test_om": as_number("soil_test_om", 0.0, 25.0),
        "soil_test_n": as_number("soil_test_n", 0.0, 5000.0),
        "soil_test_p": as_number("soil_test_p", 0.0, 1000.0),
        "soil_test_k": as_number("soil_test_k", 0.0, 5000.0),
    }


def _priority_weights(priorities: dict | None) -> dict:
    """Normalize 0-5 importance values into score weights that sum to 1."""
    raw = _clean_priorities(priorities)
    total = sum(raw.values())
    if total <= 0:
        raw = dict(DEFAULT_PRIORITIES)
        total = sum(raw.values())
    return {PRIORITY_KEYS[key]: value / total for key, value in raw.items()}


def _weighted_overall(scores: dict, weights: dict) -> int:
    total = 0.0
    for dimension in SCORE_KEYS:
        total += float(scores.get(dimension) or 0.0) * float(weights.get(dimension) or 0.0)
    return int(round(_clamp(total)))


def _water_need_from_score(score: float) -> str:
    if score >= 70:
        return "low"
    if score >= 45:
        return "medium"
    return "high"


def _neutral_scores() -> dict:
    return {
        "water_security": 50,
        "soil_health": 50,
        "climate_resilience": 50,
        "income_stability": 50,
        "cost_efficiency": 50,
        "labour_efficiency": 50,
    }


# --- Derived analytics (deterministic; computed from the saved NASA window) --

ROOT_DEPTH_BY_TEXTURE = {
    "sand": 60,
    "loamy sand": 70,
    "sandy loam": 80,
    "loam": 100,
    "silt loam": 110,
    "silt": 100,
    "sandy clay loam": 90,
    "clay loam": 95,
    "silty clay loam": 100,
    "sandy clay": 85,
    "silty clay": 90,
    "clay": 85,
}

AWC_BY_TEXTURE = {
    "sand": 0.07,
    "loamy sand": 0.09,
    "sandy loam": 0.12,
    "loam": 0.17,
    "silt loam": 0.20,
    "silt": 0.18,
    "sandy clay loam": 0.15,
    "clay loam": 0.18,
    "silty clay loam": 0.19,
    "sandy clay": 0.14,
    "silty clay": 0.16,
    "clay": 0.16,
}

TERRAIN_SOURCE = "Copernicus DEM GLO-90 via Open-Meteo (SRTM-era)"


def _texture_key(texture: str | None) -> str:
    text = str(texture or "").lower()
    for key in (
        "sandy clay loam",
        "silty clay loam",
        "sandy clay",
        "silty clay",
        "clay loam",
        "sandy loam",
        "silt loam",
        "loamy sand",
        "sand",
        "silt",
        "loam",
        "clay",
    ):
        if key in text:
            return key
    return "loam"


def _extraterrestrial_radiation(lat_deg: float, iso_day: str) -> float | None:
    """FAO-56 extraterrestrial radiation Ra in MJ/m2/day."""
    try:
        day = dt.date.fromisoformat(iso_day).timetuple().tm_yday
    except (TypeError, ValueError):
        return None
    phi = math.radians(max(-66.0, min(66.0, lat_deg)))
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * day / 365.0)
    decl = 0.409 * math.sin(2.0 * math.pi * day / 365.0 - 1.39)
    x = max(-1.0, min(1.0, -math.tan(phi) * math.tan(decl)))
    ws = math.acos(x)
    return (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(phi) * math.sin(decl)
        + math.cos(phi) * math.cos(decl) * math.sin(ws)
    )


def _hargreaves_et0(
    lat: float,
    iso_day: str,
    tmax: float | None,
    tmin: float | None,
    tmean: float | None,
) -> float | None:
    """Reference evapotranspiration estimate in mm/day (Hargreaves)."""
    if tmax is None or tmin is None or tmean is None:
        return None
    ra = _extraterrestrial_radiation(lat, iso_day)
    if ra is None:
        return None
    spread = max(0.0, float(tmax) - float(tmin))
    # FAO-56 Hargreaves expects Ra in mm/day, so convert MJ/m2/day with 0.408.
    return max(0.0, 0.0023 * 0.408 * (float(tmean) + 17.8) * math.sqrt(spread) * ra)


def _climate_analytics(nasa: dict | None, lat: float) -> dict:
    """Dry spells, wind, sunlight, rainfall trend and a simple water balance."""
    if not nasa:
        return {
            "available": False,
            "note": "NASA POWER is unavailable, so no climate analytics were computed.",
        }
    series = nasa.get("series") or {}
    dates = series.get("dates") or []
    if not dates:
        return {
            "available": False,
            "note": "NASA POWER returned no daily values for this point.",
        }

    def at(key: str, index: int):
        values = series.get(key) or []
        value = values[index] if index < len(values) else None
        return value if isinstance(value, (int, float)) else None

    days = []
    for index, day in enumerate(dates):
        entry = {
            "date": day,
            "rain_mm": at("rain_mm", index),
            "tmax": at("temp_max_c", index),
            "tmin": at("temp_min_c", index),
            "tmean": at("temp_c", index),
            "wind_ms": at("wind_ms", index),
            "solar_mj": at("solar_mj", index),
        }
        entry["et0_mm"] = _hargreaves_et0(
            lat, day, entry["tmax"], entry["tmin"], entry["tmean"]
        )
        days.append(entry)

    rain_values = [d["rain_mm"] for d in days if d["rain_mm"] is not None]
    et0_values = [d["et0_mm"] for d in days if d["et0_mm"] is not None]
    tmax_values = [d["tmax"] for d in days if d["tmax"] is not None]
    wind_values = [d["wind_ms"] for d in days if d["wind_ms"] is not None]
    solar_values = [d["solar_mj"] for d in days if d["solar_mj"] is not None]

    rain_total = round(sum(rain_values), 1) if rain_values else None
    et0_total = round(sum(et0_values), 1) if et0_values else None
    balance = (
        round(rain_total - et0_total, 1)
        if rain_total is not None and et0_total is not None
        else None
    )

    # A dry day has under 1 mm of rain. Track the longest run and the current run.
    dry_flags = [d["rain_mm"] is not None and d["rain_mm"] < 1.0 for d in days]
    longest = current = 0
    for flag in dry_flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    current_dry = 0
    for flag in reversed(dry_flags):
        if not flag:
            break
        current_dry += 1

    weeks = []
    start = max(0, len(days) - 28)
    for index in range(start, len(days), 7):
        block = days[index : index + 7]
        if not block:
            continue
        mm = sum(d["rain_mm"] for d in block if d["rain_mm"] is not None)
        weeks.append(
            {
                "start": block[0]["date"],
                "end": block[-1]["date"],
                "rain_mm": round(mm, 1),
            }
        )

    last7 = [d["rain_mm"] for d in days[-7:] if d["rain_mm"] is not None]
    prev7 = [d["rain_mm"] for d in days[-14:-7] if d["rain_mm"] is not None]
    last7_total = round(sum(last7), 1) if last7 else None
    prev7_total = round(sum(prev7), 1) if prev7 else None
    rain_change = (
        round(last7_total - prev7_total, 1)
        if last7_total is not None and prev7_total is not None
        else None
    )

    def balance_label(value: float | None) -> str:
        if value is None:
            return "unknown"
        if value >= 0:
            return "surplus"
        if value >= -60:
            return "slight deficit"
        if value >= -150:
            return "deficit"
        return "large deficit"

    def dry_label(value: int) -> str:
        if value >= 14:
            return "long dry spell"
        if value >= 7:
            return "moderate dry spell"
        return "short dry spell"

    return {
        "available": True,
        "rain_total_mm": rain_total,
        "et0_total_mm": et0_total,
        "water_balance_mm": balance,
        "water_balance_label": balance_label(balance),
        "longest_dry_spell_days": longest,
        "current_dry_spell_days": current_dry,
        "dry_spell_label": dry_label(longest),
        "dry_days": sum(1 for flag in dry_flags if flag),
        "heat_days_35c": sum(1 for value in tmax_values if value >= 35.0),
        "sunny_days": sum(1 for value in solar_values if value >= 18.0),
        "wind_max_ms": round(max(wind_values), 1) if wind_values else None,
        "solar_max_mj": round(max(solar_values), 1) if solar_values else None,
        "last7_rain_mm": last7_total,
        "prev7_rain_mm": prev7_total,
        "rain_change_mm": rain_change,
        "rain_trend": (
            "wetter than the previous week"
            if rain_change is not None and rain_change > 5
            else "drier than the previous week"
            if rain_change is not None and rain_change < -5
            else "similar to the previous week"
            if rain_change is not None
            else "unknown"
        ),
        "weeks": weeks,
        "daily": [
            {
                "date": d["date"],
                "rain_mm": d["rain_mm"],
                "et0_mm": round(d["et0_mm"], 2) if d["et0_mm"] is not None else None,
                "wind_ms": d["wind_ms"],
                "solar_mj": d["solar_mj"],
                "temp_max_c": d["tmax"],
            }
            for d in days
        ],
        "method": (
            "Daily values from NASA POWER. Reference evapotranspiration is estimated "
            "with the Hargreaves equation from measured temperature and latitude. It is "
            "an estimate, not a lysimeter or soil-probe reading."
        ),
    }


def _moisture_analytics(nasa: dict | None, satellite_data: dict | None) -> dict:
    """Combine POWER root-zone wetness with the SMAP surface/root-zone samples."""
    nasa = nasa or {}
    readings = (satellite_data or {}).get("readings") or {}
    series = nasa.get("series") or {}
    dates = series.get("dates") or []
    root = series.get("root_wetness") or []

    points = []
    for index, day in enumerate(dates):
        value = root[index] if index < len(root) else None
        if isinstance(value, (int, float)):
            points.append({"date": day, "value": round(float(value), 3)})

    values = [point["value"] for point in points]
    smap_surface = (readings.get("sm_surface") or {}).get("mean")
    smap_root = (readings.get("sm_rootzone") or {}).get("mean")
    gradient = (
        round(float(smap_root) - float(smap_surface), 3)
        if smap_root is not None and smap_surface is not None
        else None
    )

    trend = None
    if len(values) >= 14:
        trend = round(sum(values[-7:]) / 7 - sum(values[:7]) / 7, 3)
    elif len(values) >= 4:
        half = len(values) // 2
        trend = round(
            sum(values[half:]) / len(values[half:])
            - sum(values[:half]) / len(values[:half]),
            3,
        )

    mean = round(sum(values) / len(values), 3) if values else None

    return {
        "available": bool(values) or smap_surface is not None or smap_root is not None,
        "root_wetness": {
            "mean": mean,
            "min": round(min(values), 3) if values else None,
            "max": round(max(values), 3) if values else None,
            "latest": values[-1] if values else None,
            "trend": trend,
            "dry_days": sum(1 for value in values if value < 0.30),
            "wet_days": sum(1 for value in values if value > 0.60),
            "label": _soil_label(mean),
        },
        "smap_surface": smap_surface,
        "smap_rootzone": smap_root,
        "gradient": gradient,
        "gradient_note": (
            "Root-zone moisture is higher than the surface layer, so recent rain is moving down."
            if gradient is not None and gradient > 0.02
            else "The surface is wetter than the root zone, so water is not reaching depth."
            if gradient is not None and gradient < -0.02
            else "Surface and root-zone moisture are close."
            if gradient is not None
            else "SMAP surface and root-zone samples are not both available."
        ),
        "daily": points,
        "note": (
            "POWER root-zone wetness is a modelled 0-1 fraction for the top metre. "
            "SMAP is a 9 km satellite estimate. Neither is a probe in your field."
        ),
    }


def _soil_nourishment(soil_data: dict | None, context: dict | None) -> dict:
    """Ground-truth soil test values first, then modelled OpenLandMap values."""
    soil_data = soil_data or {}
    context = context or {}
    measured_om = soil_data.get("organic_matter_pct")
    measured_ph = soil_data.get("ph")
    test_om = context.get("soil_test_om")
    test_ph = context.get("soil_test_ph")
    nitrogen = context.get("soil_test_n")
    phosphorus = context.get("soil_test_p")
    potassium = context.get("soil_test_k")

    organic_matter = test_om if test_om is not None else measured_om
    ph = test_ph if test_ph is not None else measured_ph
    has_test = any(
        value is not None
        for value in (test_om, test_ph, nitrogen, phosphorus, potassium)
    )

    items = []

    def add(key: str, label: str, value: str, status: str, note: str) -> None:
        items.append(
            {"key": key, "label": label, "value": value, "status": status, "note": note}
        )

    if organic_matter is None:
        add(
            "organic_matter",
            "Organic matter",
            "not measured",
            "unknown",
            "Add a soil test or wait for the OpenLandMap sample.",
        )
    else:
        value = float(organic_matter)
        status = "low" if value < 1.0 else "medium" if value < 2.0 else "high"
        source = "your soil test" if test_om is not None else "OpenLandMap 0-30 cm"
        add(
            "organic_matter",
            "Organic matter",
            f"{value:.2f} %",
            status,
            f"Source: {source}. Below 1 % is low, 1-2 % is medium, above 2 % is high.",
        )

    if ph is None:
        add("ph", "pH in water", "not measured", "unknown", "A pH test is cheap and worth doing.")
    else:
        value = float(ph)
        if value < 5.5:
            status = "low"
        elif value < 6.5:
            status = "slightly low"
        elif value <= 7.5:
            status = "good"
        elif value <= 8.5:
            status = "slightly high"
        else:
            status = "high"
        source = "your soil test" if test_ph is not None else "OpenLandMap 0-30 cm"
        add(
            "ph",
            "pH in water",
            f"{value:.1f}",
            status,
            f"Source: {source}. Most crops prefer 6.5 to 7.5.",
        )

    if nitrogen is None:
        om_status = "unknown"
        for item in items:
            if item["key"] == "organic_matter":
                om_status = item["status"]
        estimated = {
            "low": "likely low",
            "medium": "likely medium",
            "high": "likely high",
        }.get(om_status, "not tested")
        add(
            "nitrogen",
            "Nitrogen (N)",
            estimated,
            om_status if om_status != "unknown" else "unknown",
            "Estimated from organic matter only. A soil test is the only measured value.",
        )
    else:
        value = float(nitrogen)
        status = "low" if value < 280 else "medium" if value < 560 else "high"
        add("nitrogen", "Nitrogen (N)", f"{value:.0f} kg/ha", status, "Source: your soil test.")

    for key, label, raw, low_cut, high_cut, unit in (
        ("phosphorus", "Phosphorus (P)", phosphorus, 10.0, 20.0, "ppm"),
        ("potassium", "Potassium (K)", potassium, 120.0, 200.0, "ppm"),
    ):
        if raw is None:
            add(
                key,
                label,
                "not tested",
                "unknown",
                "OpenLandMap does not map plant-available P or K. A soil test is needed.",
            )
        else:
            value = float(raw)
            status = "low" if value < low_cut else "medium" if value < high_cut else "high"
            add(key, label, f"{value:.0f} {unit}", status, "Source: your soil test.")

    score = 50.0
    if organic_matter is not None:
        score += 14 if float(organic_matter) >= 2.0 else 4 if float(organic_matter) >= 1.0 else -14
    if ph is not None:
        value = float(ph)
        if 6.5 <= value <= 7.5:
            score += 10
        elif 6.0 <= value <= 8.0:
            score += 3
        else:
            score -= 8
    for raw, low_cut, high_cut in (
        (nitrogen, 280.0, 560.0),
        (phosphorus, 10.0, 20.0),
        (potassium, 120.0, 200.0),
    ):
        if raw is not None:
            value = float(raw)
            score += 8 if value >= high_cut else -8 if value < low_cut else 0

    return {
        "available": any(item["status"] != "unknown" for item in items),
        "score": int(round(_clamp(score))),
        "items": items,
        "source": (
            "Your soil test results override the modelled OpenLandMap values."
            if has_test
            else "Modelled OpenLandMap 0-30 cm values; N, P and K are not mapped."
        ),
        "note": (
            "Nutrients are not measured by satellite. P and K need a laboratory soil test. "
            "Treat the score as a planning signal, not a fertiliser prescription."
        ),
    }


def _root_zone_profile(soil_data: dict | None, moisture: dict | None) -> dict:
    """Estimate effective root depth and plant-available water from texture."""
    soil_data = soil_data or {}
    moisture = moisture or {}
    texture = soil_data.get("texture_class") or ""
    key = _texture_key(texture)
    depth = ROOT_DEPTH_BY_TEXTURE.get(key, 100)
    awc_fraction = AWC_BY_TEXTURE.get(key, 0.17)
    awc_mm = round(depth * awc_fraction * 10)
    return {
        "texture": texture or "unclassified",
        "depth_cm": depth,
        "awc_fraction": awc_fraction,
        "awc_mm": awc_mm,
        "gradient": moisture.get("gradient"),
        "basis": (
            "Estimated from the USDA texture class. A soil pit or root-depth probe is "
            "the only way to measure it."
        ),
        "note": (
            "Effective root zone is the depth where most crop roots take up water. "
            f"At {depth} cm with {awc_fraction:.2f} plant-available water per cm, "
            f"the topsoil can hold about {awc_mm} mm of plant-available water."
        ),
    }


def _seasonal_history(climatology: dict | None) -> dict:
    """Summarise the last 12 months of POWER data into seasonal windows.

    This is real measured history (POWER daily reanalysis), not a forecast. It
    gives the crop calendar frost dates and the months when soil is most likely
    to sit exposed to wind, heat and little rain.
    """
    if not climatology:
        return {"available": False, "note": "Seasonal history was not requested."}

    series = climatology.get("series") or {}
    dates = series.get("dates") or []
    rain = series.get("rain_mm") or []
    tmax = series.get("temp_max_c") or []
    tmin = series.get("temp_min_c") or []
    if not dates:
        return {
            "available": False,
            "note": "NASA POWER returned no daily values for the seasonal history.",
        }

    months: dict[int, dict] = {}
    for index, day in enumerate(dates):
        try:
            month = int(str(day)[5:7])
        except (TypeError, ValueError):
            continue
        entry = months.setdefault(
            month,
            {"month": month, "rain_mm": 0.0, "tmax": [], "tmin": [], "days": 0},
        )
        if index < len(rain) and isinstance(rain[index], (int, float)):
            entry["rain_mm"] += float(rain[index])
        if index < len(tmax) and isinstance(tmax[index], (int, float)):
            entry["tmax"].append(float(tmax[index]))
        if index < len(tmin) and isinstance(tmin[index], (int, float)):
            entry["tmin"].append(float(tmin[index]))
        entry["days"] += 1

    month_rows = []
    for month in range(1, 13):
        entry = months.get(month)
        if not entry or entry["days"] < 15:
            continue  # skip partial months at the ends of the window
        month_rows.append(
            {
                "month": month,
                "label": _month_label(month),
                "rain_mm": round(entry["rain_mm"], 1),
                "tmax_c": round(sum(entry["tmax"]) / len(entry["tmax"]), 1)
                if entry["tmax"]
                else None,
                "tmin_c": round(min(entry["tmin"]), 1) if entry["tmin"] else None,
                "days": entry["days"],
            }
        )

    if not month_rows:
        return {
            "available": False,
            "note": "Not enough complete months to build a seasonal history.",
        }

    wettest = max(month_rows, key=lambda row: row["rain_mm"])
    driest = min(month_rows, key=lambda row: row["rain_mm"])
    frost_months = [
        row["label"]
        for row in month_rows
        if row["tmin_c"] is not None and row["tmin_c"] < 2.0
    ]
    frost_dates = [
        dates[index]
        for index in range(len(dates))
        if index < len(tmin)
        and isinstance(tmin[index], (int, float))
        and float(tmin[index]) < 2.0
    ]

    # A dry-hot run is the window when bare soil is most exposed.
    runs: list[list[dict]] = []
    current: list[dict] = []
    for row in month_rows:
        dry_hot = row["rain_mm"] < 25 and (row["tmax_c"] or 0) >= 30
        if dry_hot:
            current.append(row)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    exposure_windows = []
    for run in runs:
        mean_tmax = [row["tmax_c"] for row in run if row["tmax_c"] is not None]
        exposure_windows.append(
            {
                "start_month": run[0]["month"],
                "end_month": run[-1]["month"],
                "label": (
                    run[0]["label"]
                    if len(run) == 1
                    else f"{run[0]['label']} to {run[-1]['label']}"
                ),
                "rain_mm": round(sum(row["rain_mm"] for row in run), 1),
                "mean_tmax_c": round(sum(mean_tmax) / len(mean_tmax), 1)
                if mean_tmax
                else None,
                "note": (
                    "Low rain with high heat. Bare soil in this window is exposed "
                    "to wind, crusting and organic-matter loss."
                ),
            }
        )

    return {
        "available": True,
        "months": month_rows,
        "wettest_month": wettest,
        "driest_month": driest,
        "frost_months": frost_months,
        "first_frost": frost_dates[0] if frost_dates else None,
        "last_frost": frost_dates[-1] if frost_dates else None,
        "exposure_windows": exposure_windows,
        "source": "NASA POWER daily values, last 12 months",
        "note": (
            "Monthly rain and temperature come from the last 12 months of NASA "
            "POWER. Frost dates are the first and last nights below 2 C in that "
            "window, not a long-term climate normal."
        ),
    }


def _bare_field_timeline(
    satellite_data: dict | None, climatology: dict | None = None
) -> dict:
    """Flag low-NDVI windows and seasonal dry-hot exposure windows."""
    readings = (satellite_data or {}).get("readings") or {}
    ndvi = readings.get("ndvi") or {}
    series = ndvi.get("series") or []
    low = [
        point
        for point in series
        if point.get("value") is not None and float(point["value"]) < 0.25
    ]
    seasonal = climatology or {}
    seasonal_windows = seasonal.get("exposure_windows") or []
    observed = bool(series)
    return {
        "available": observed or bool(seasonal.get("available")),
        "observed_available": observed,
        "series": series,
        "low_cover_samples": low,
        "low_cover_count": len(low),
        "seasonal_available": bool(seasonal.get("available")),
        "seasonal_windows": seasonal_windows,
        "seasonal_months": seasonal.get("months") or [],
        "frost_months": seasonal.get("frost_months") or [],
        "first_frost": seasonal.get("first_frost"),
        "last_frost": seasonal.get("last_frost"),
        "note": (
            "NDVI below 0.25 usually means bare soil or very little green cover. "
            "MODIS composites are 16-day, 250 m samples, so this flags a window, "
            "not an exact day. Cloud cover can blank an overpass, and the seasonal "
            "windows come from the last 12 months of NASA POWER, not multi-year imagery."
        ),
    }


def fetch_terrain(lat: float, lng: float) -> dict:
    """Sample a 3x3 elevation grid and estimate slope; never raises."""
    result = {
        "available": False,
        "source": TERRAIN_SOURCE,
        "resolution_m": 90,
        "errors": [],
    }
    step = 0.002  # roughly 220 m, a little larger than the 90 m DEM cell
    offsets = [
        (-step, -step),
        (-step, 0.0),
        (-step, step),
        (0.0, -step),
        (0.0, 0.0),
        (0.0, step),
        (step, -step),
        (step, 0.0),
        (step, step),
    ]
    lats = ",".join(f"{lat + dlat:.5f}" for dlat, _ in offsets)
    lngs = ",".join(f"{lng + dlng:.5f}" for _, dlng in offsets)
    url = (
        "https://api.open-meteo.com/v1/elevation"
        f"?latitude={lats}&longitude={lngs}"
    )
    try:
        data = _get_json(url, timeout=20)
        elevations = [float(value) for value in (data.get("elevation") or [])]
        if len(elevations) < 9:
            raise ValueError("elevation API returned too few points")
    except Exception as exc:  # noqa: BLE001 - terrain is optional
        result["errors"].append(str(exc))
        return result

    west = (elevations[0] + elevations[3] + elevations[6]) / 3.0
    east = (elevations[2] + elevations[5] + elevations[8]) / 3.0
    south = (elevations[0] + elevations[1] + elevations[2]) / 3.0
    north = (elevations[6] + elevations[7] + elevations[8]) / 3.0

    dy = 2.0 * step * 111320.0
    dx = 2.0 * step * 111320.0 * max(0.05, math.cos(math.radians(lat)))
    rise = math.hypot((east - west) / dx, (north - south) / dy)
    slope_pct = rise * 100.0
    if slope_pct < 2:
        slope_class = "flat"
    elif slope_pct < 5:
        slope_class = "gentle"
    elif slope_pct < 10:
        slope_class = "moderate"
    elif slope_pct < 20:
        slope_class = "steep"
    else:
        slope_class = "very steep"

    result.update(
        {
            "available": True,
            "elevation_m": round(elevations[4], 0),
            "relief_m": round(max(elevations) - min(elevations), 0),
            "slope_pct": round(slope_pct, 1),
            "slope_degrees": round(math.degrees(math.atan(rise)), 1),
            "slope_class": slope_class,
            "note": (
                "Slope is estimated from a 3x3 elevation grid about 220 m apart. "
                "It shows the surrounding terrain, not a surveyed field gradient."
            ),
        }
    )
    return result


def _grid_row(
    mission: str,
    variable: str,
    resolution: str,
    value: str,
    source_type: str,
    status: str,
    note: str = "",
) -> dict:
    return {
        "mission": mission,
        "variable": variable,
        "resolution": resolution,
        "value": value,
        "source_type": source_type,
        "status": status,
        "note": note,
    }


def _data_grid(
    nasa: dict | None,
    soil_data: dict | None,
    satellite_data: dict | None,
    terrain_data: dict | None,
    context: dict | None,
) -> dict:
    """List every layer in one honest grid: value, native scale and status."""
    nasa = nasa or {}
    soil_data = soil_data or {}
    satellite_data = satellite_data or {}
    terrain_data = terrain_data or {}
    context = context or {}
    readings = satellite_data.get("readings") or {}
    sat_status = str(satellite_data.get("status") or "idle")
    pending = sat_status in ("queued", "running")
    rows: list[dict] = []

    def power_row(variable: str, value: str, ok: bool) -> None:
        rows.append(
            _grid_row(
                "NASA POWER",
                variable,
                "~50 km, daily",
                value if ok else "No value",
                "Reanalysis model",
                "ok" if ok else "gap",
                "Area average, not an on-farm sensor.",
            )
        )

    temp = nasa.get("temperature_c") or {}
    rain = nasa.get("rainfall_mm") or {}
    power_row(
        "Air temperature",
        f"{_fmt(temp.get('avg'), 1, 'C')} avg / {_fmt(temp.get('max'), 1, 'C')} peak",
        temp.get("avg") is not None,
    )
    power_row(
        "Rainfall",
        f"{_fmt(rain.get('last_30_days'), 1, 'mm')} in 30 days",
        rain.get("last_30_days") is not None,
    )
    power_row(
        "Wind speed",
        f"{_fmt(nasa.get('wind_speed_ms'), 1, 'm/s')} mean / "
        f"{_fmt(nasa.get('wind_speed_max_ms'), 1, 'm/s')} peak",
        nasa.get("wind_speed_ms") is not None,
    )
    power_row(
        "Solar radiation",
        f"{_fmt(nasa.get('solar_radiation_mj_m2_day'), 1, 'MJ/m2/day')} mean / "
        f"{_fmt(nasa.get('solar_radiation_max_mj_m2_day'), 1, 'MJ/m2/day')} peak",
        nasa.get("solar_radiation_mj_m2_day") is not None,
    )
    power_row(
        "Root-zone wetness",
        _fmt(nasa.get("soil_moisture_root_zone"), 2),
        nasa.get("soil_moisture_root_zone") is not None,
    )

    def satellite_row(mission: str, variable: str, key: str, unit: str, resolution: str) -> None:
        item = readings.get(key) or {}
        has_value = item.get("mean") is not None
        if has_value:
            value = f"{_fmt(item.get('mean'), 3, unit)} mean"
            status = "ok"
            note = f"{item.get('samples', 0)} clear-sky sample(s) in 60 days."
        elif pending:
            value = "Processing"
            status = "pending"
            note = "AppEEARS task is still running."
        else:
            value = "No value"
            status = "gap"
            note = "No clear-sky sample returned for this point."
        rows.append(
            _grid_row(
                mission, variable, resolution, value, "Satellite", status, note
            )
        )

    satellite_row("MODIS MOD13Q1", "NDVI (green cover)", "ndvi", "", "250 m, 16-day")
    satellite_row(
        "MODIS MOD11A2", "Daytime land-surface temperature", "lst_day_k", "K", "1 km, 8-day"
    )
    satellite_row(
        "SMAP SPL4SMGP", "Surface soil moisture", "sm_surface", "m3/m3", "9 km, daily"
    )
    satellite_row(
        "SMAP SPL4SMGP", "Root-zone soil moisture", "sm_rootzone", "m3/m3", "9 km, daily"
    )

    if soil_data.get("available"):
        rows.append(
            _grid_row(
                "OpenLandMap SoilDB",
                "Texture, pH, organic matter",
                "120 m, 0-30 cm",
                (
                    f"{soil_data.get('texture_class') or 'texture n/a'}, "
                    f"pH {_fmt(soil_data.get('ph'), 1)}, "
                    f"OM {_fmt(soil_data.get('organic_matter_pct'), 2, '%')}"
                ),
                "Modelled soil map",
                "ok",
                "Prediction, not a laboratory sample.",
            )
        )
    else:
        rows.append(
            _grid_row(
                "OpenLandMap SoilDB",
                "Texture, pH, organic matter",
                "120 m, 0-30 cm",
                "No value",
                "Modelled soil map",
                "gap",
                "Service did not return a value for this point.",
            )
        )

    if terrain_data.get("available"):
        rows.append(
            _grid_row(
                "Copernicus DEM",
                "Elevation and slope",
                "90 m",
                f"{_fmt(terrain_data.get('elevation_m'), 0, 'm')}, "
                f"{_fmt(terrain_data.get('slope_pct'), 1, '%')} slope",
                "Satellite DEM",
                "ok",
                "SRTM-era elevation model, not a surveyed field gradient.",
            )
        )
    else:
        rows.append(
            _grid_row(
                "Copernicus DEM",
                "Elevation and slope",
                "90 m",
                "No value",
                "Satellite DEM",
                "gap",
                "Elevation service did not respond.",
            )
        )

    has_test = any(
        context.get(key) is not None
        for key in ("soil_test_ph", "soil_test_om", "soil_test_n", "soil_test_p", "soil_test_k")
    )
    if has_test:
        parts = []
        for key, label in (
            ("soil_test_ph", "pH"),
            ("soil_test_om", "OM"),
            ("soil_test_n", "N"),
            ("soil_test_p", "P"),
            ("soil_test_k", "K"),
        ):
            if context.get(key) is not None:
                parts.append(f"{label} {context[key]}")
        rows.append(
            _grid_row(
                "Farmer soil test",
                "pH, organic matter, NPK",
                "Point sample",
                ", ".join(parts),
                "Ground sample",
                "ok",
                "Your laboratory result, used ahead of the modelled values.",
            )
        )
    else:
        rows.append(
            _grid_row(
                "Farmer soil test",
                "pH, organic matter, NPK",
                "Point sample",
                "Not supplied",
                "Ground sample",
                "input",
                "Optional: add your soil test to calibrate the model.",
            )
        )

    irrigation = str(context.get("irrigation_status") or context.get("irrigation") or "").strip()
    rows.append(
        _grid_row(
            "Farmer context",
            "Irrigation status",
            "Field",
            irrigation or "Not supplied",
            "Ground input",
            "ok" if irrigation else "input",
            "Used to judge whether a dry spell is a yield risk.",
        )
    )

    for mission, variable, note in (
        ("ECOSTRESS", "Evapotranspiration and canopy heat", "Not queried by this app."),
        ("GPM IMERG", "Satellite rainfall", "POWER rainfall is used instead."),
        ("Landsat", "30 m vegetation and moisture", "MODIS NDVI is used instead."),
        ("GRACE-FO", "Groundwater storage trend", "Not queried by this app."),
    ):
        rows.append(
            _grid_row(
                mission,
                variable,
                "Native mission scale",
                "Not queried",
                "Satellite",
                "not_queried",
                note,
            )
        )

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    return {
        "available": True,
        "rows": rows,
        "ok_count": ok_count,
        "row_count": len(rows),
        "note": (
            "Every layer is sampled at the selected point or polygon and keeps its "
            "native resolution. They are listed on one screen, not physically "
            "resampled onto a single raster, so a 250 m MODIS pixel and a 9 km SMAP "
            "pixel are not the same footprint."
        ),
    }


def _point_in_polygon(lat: float, lng: float, polygon: list[dict]) -> bool:
    inside = False
    count = len(polygon)
    j = count - 1
    for i in range(count):
        yi, xi = polygon[i]["lat"], polygon[i]["lng"]
        yj, xj = polygon[j]["lat"], polygon[j]["lng"]
        if (yi > lat) != (yj > lat):
            cross = (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi
            if lng < cross:
                inside = not inside
        j = i
    return inside


def _zone_sample_points(
    lat: float, lng: float, polygon: list[dict] | None
) -> list[dict]:
    """Pick up to 9 terrain sample points inside the field, or around the pin."""
    cleaned = []
    for point in polygon or []:
        try:
            cleaned.append({"lat": float(point["lat"]), "lng": float(point["lng"])})
        except (KeyError, TypeError, ValueError):
            continue
    if len(cleaned) >= 3:
        min_lat = min(p["lat"] for p in cleaned)
        max_lat = max(p["lat"] for p in cleaned)
        min_lng = min(p["lng"] for p in cleaned)
        max_lng = max(p["lng"] for p in cleaned)
        candidates = []
        for lat_fraction in (0.2, 0.5, 0.8):
            for lng_fraction in (0.2, 0.5, 0.8):
                point = {
                    "lat": min_lat + (max_lat - min_lat) * lat_fraction,
                    "lng": min_lng + (max_lng - min_lng) * lng_fraction,
                }
                if _point_in_polygon(point["lat"], point["lng"], cleaned):
                    candidates.append(point)
        if len(candidates) < 3:
            candidates = [
                {
                    "lat": sum(p["lat"] for p in cleaned) / len(cleaned),
                    "lng": sum(p["lng"] for p in cleaned) / len(cleaned),
                }
            ]
        unique = []
        seen = set()
        for point in candidates:
            key = (round(point["lat"], 6), round(point["lng"], 6))
            if key not in seen:
                seen.add(key)
                unique.append(point)
        return unique[:9]

    step = 0.0015  # about 165 m
    return [
        {"lat": lat, "lng": lng},
        {"lat": lat + step, "lng": lng},
        {"lat": lat - step, "lng": lng},
        {"lat": lat, "lng": lng + step},
        {"lat": lat, "lng": lng - step},
    ]


def build_hotspots(
    lat: float,
    lng: float,
    polygon: list[dict] | None = None,
    diagnostics: dict | None = None,
) -> dict:
    """Derive field-zone hotspots from a 90 m elevation model.

    Climate and soil values are area signals, so the zone differences shown
    here come from terrain: slope, aspect and cold-air pooling. They are
    relative planning signals, not separate weather stations.
    """
    diagnostics = diagnostics or {}
    climate = diagnostics.get("climate") or {}
    soil = diagnostics.get("soil") or {}
    context = diagnostics.get("context") or {}
    points = _zone_sample_points(lat, lng, polygon)
    if not points:
        return {
            "available": False,
            "note": "No sample points could be placed inside this field.",
            "zones": [],
        }

    step = 0.0015
    offsets = [(0.0, 0.0), (-step, 0.0), (step, 0.0), (0.0, -step), (0.0, step)]
    lats: list[str] = []
    lngs: list[str] = []
    for point in points:
        for dlat, dlng in offsets:
            lats.append(f"{point['lat'] + dlat:.5f}")
            lngs.append(f"{point['lng'] + dlng:.5f}")

    url = (
        "https://api.open-meteo.com/v1/elevation"
        f"?latitude={','.join(lats)}&longitude={','.join(lngs)}"
    )
    try:
        data = _get_json(url, timeout=35)
        elevations = [float(value) for value in (data.get("elevation") or [])]
        if len(elevations) < len(lats):
            raise ValueError("elevation API returned too few points")
    except Exception as exc:  # noqa: BLE001 - zones are optional
        return {
            "available": False,
            "note": f"Terrain sampling failed for these zones: {exc}",
            "zones": [],
        }

    rain = float(climate.get("rain_total_mm") or 0.0)
    summary = diagnostics.get("nasa_summary") or {}
    tmax = float(summary.get("tmax_c") or 0.0)
    tmin = float(summary.get("tmin_c") or 0.0)
    wind = float(summary.get("wind_speed_ms") or 0.0)
    texture = str(soil.get("texture_class") or context.get("soil_texture") or "").lower()
    organic = str(soil.get("organic_matter_class") or context.get("organic_matter") or "").lower()

    center_elevations = [elevations[index * 5] for index in range(len(points))]
    mean_elevation = sum(center_elevations) / len(center_elevations)
    dy = 2.0 * step * 111320.0
    dx = 2.0 * step * 111320.0 * max(0.05, math.cos(math.radians(lat)))

    zones = []
    for index, point in enumerate(points):
        base = index * 5
        center = elevations[base]
        south, north = elevations[base + 1], elevations[base + 2]
        west, east = elevations[base + 3], elevations[base + 4]
        east_grad = (east - west) / dx
        north_grad = (north - south) / dy
        slope_pct = math.hypot(east_grad, north_grad) * 100.0
        aspect = (math.degrees(math.atan2(-east_grad, -north_grad)) + 360.0) % 360.0
        southness = max(0.0, -math.cos(math.radians(aspect)))
        northness = max(0.0, math.cos(math.radians(aspect)))

        erosion = 92.0
        erosion -= max(0.0, slope_pct - 1.0) * 2.8
        erosion -= max(0.0, rain - 70.0) * 0.16
        erosion -= max(0.0, wind - 3.5) * 5.0
        if "sand" in texture:
            erosion -= 10.0
        if organic == "low":
            erosion -= 8.0
        erosion = _clamp(erosion)

        frost = 100.0 - max(0.0, 4.0 - tmin) * 14.0
        frost -= max(0.0, mean_elevation - center) * 1.6
        frost = _clamp(frost)

        moisture = 65.0 + (center - mean_elevation) * 0.7
        moisture -= slope_pct * 1.1
        moisture -= southness * 9.0
        moisture += min(rain / 150.0, 1.0) * 22.0
        if "sand" in texture:
            moisture -= 8.0
        moisture = _clamp(moisture)

        heat = 100.0 - max(0.0, tmax - 31.0) * 6.0
        heat -= southness * 8.0
        heat += northness * 4.0
        heat = _clamp(heat)

        risks = {
            "erosion": int(round(erosion)),
            "frost": int(round(frost)),
            "moisture": int(round(moisture)),
            "heat": int(round(heat)),
        }
        dominant_key = min(risks, key=risks.get)
        labels = {
            "erosion": "runoff erosion",
            "frost": "frost",
            "moisture": "moisture deficit",
            "heat": "canopy heat",
        }
        zones.append(
            {
                "lat": round(point["lat"], 5),
                "lng": round(point["lng"], 5),
                "elevation_m": round(center, 0),
                "relative_elevation_m": round(center - mean_elevation, 1),
                "slope_pct": round(slope_pct, 1),
                "aspect_deg": round(aspect, 0),
                "risks": risks,
                "dominant": dominant_key,
                "dominant_label": labels[dominant_key],
                "level": _concern(float(risks[dominant_key])),
            }
        )

    counts: dict[str, int] = {}
    for zone in zones:
        counts[zone["dominant"]] = counts.get(zone["dominant"], 0) + 1
    parts = [
        f"{count} of {len(zones)} zones most exposed to {label}"
        for key, count in counts.items()
        for label in [next(z["dominant_label"] for z in zones if z["dominant"] == key)]
    ]
    return {
        "available": True,
        "zones": zones,
        "zone_count": len(zones),
        "summary": "; ".join(parts) + ".",
        "source": TERRAIN_SOURCE,
        "resolution_m": 90,
        "note": (
            "Zone differences come from a 90 m elevation model (slope, aspect and "
            "low-lying cold-air pooling). NASA POWER rain and temperature and the "
            "OpenLandMap soil values are area signals, so they are the same for "
            "every zone. Treat this as a relative hotspot map, not a weather station "
            "in each corner of the field."
        ),
    }


def build_diagnostics(
    nasa: dict | None,
    context: dict | None = None,
    soil_data: dict | None = None,
    satellite_data: dict | None = None,
    terrain_data: dict | None = None,
    climatology_data: dict | None = None,
    lat: float | None = None,
) -> dict:
    """Derive transparent field-vulnerability indicators from the saved data.

    This does not fetch or alter any NASA values. It only reads the NASA POWER
    aggregates that /select already saved, the OpenLandMap soil sample and the
    AppEEARS satellite status, then combines them with the farmer's context.
    """
    context = context or {}
    climatology_data = climatology_data or {}
    soil_data = soil_data or {}
    satellite_data = satellite_data or {}
    terrain_data = terrain_data or {}
    readings = satellite_data.get("readings") or {}

    # Deterministic analytics shared by the panel, the AI prompt and the stress test.
    climate = _climate_analytics(nasa, float(lat) if lat is not None else 0.0)
    moisture = _moisture_analytics(nasa, satellite_data)
    nourishment = _soil_nourishment(soil_data, context)
    root_zone = _root_zone_profile(soil_data, moisture)
    bare_field = _bare_field_timeline(satellite_data, climatology_data)
    data_grid = _data_grid(nasa, soil_data, satellite_data, terrain_data, context)

    # Only tag a mission on an indicator when its reading is actually present.
    has_lst = "lst_day_k" in readings
    has_ndvi = "ndvi" in readings
    has_smap = "sm_surface" in readings or "sm_rootzone" in readings
    has_soil_map = bool(soil_data.get("available"))

    def _sources(*extra: str) -> list[str]:
        used = ["NASA POWER"]
        for name in extra:
            if name and name not in used:
                used.append(name)
        return used

    if not nasa:
        return {
            "available": False,
            "headline": "NASA agro-climatology is not available for this point yet.",
            "period": None,
            "indicators": [],
            "scores": _neutral_scores(),
            "coverage": {
                "available": [],
                "missing": [
                    "NASA POWER window (request failed)",
                    "Field boundary",
                    "Soil organic matter",
                    "Groundwater storage trend",
                ],
                "note": "Select the field again to retry the NASA lookup.",
            },
            "missions": [],
            "missions_not_queried": MISSION_NOT_QUERIED,
            "soil": soil_data,
            "satellite": satellite_data,
            "terrain": terrain_data,
            "climate": climate,
            "moisture": moisture,
            "nourishment": nourishment,
            "root_zone": root_zone,
            "bare_field": bare_field,
            "climatology": climatology_data,
            "data_grid": data_grid,
            "context": context,
        }

    temp = nasa.get("temperature_c") or {}
    rain = nasa.get("rainfall_mm") or {}
    tavg = temp.get("avg")
    tmax = temp.get("max")
    tmin = temp.get("min")
    r7 = rain.get("last_7_days")
    r30 = rain.get("last_30_days")
    humidity = nasa.get("humidity_pct")
    soil = nasa.get("soil_moisture_root_zone")
    wind = nasa.get("wind_speed_ms")
    solar = nasa.get("solar_radiation_mj_m2_day")
    gdd = nasa.get("growing_degree_days_30d")

    # Each score is a safety score: 100 is comfortable, 0 is severe pressure.
    heat = 100.0
    if tmax is not None:
        heat -= max(0.0, tmax - 31.0) * 6.0
    if tavg is not None:
        heat -= max(0.0, tavg - 26.0) * 5.0
    if solar is not None and solar > 24:
        heat -= min((solar - 24.0) * 1.5, 6.0)
    heat = _clamp(heat)

    water = 55.0
    if soil is not None:
        water = soil * 100.0 * 0.68
        if r30 is not None:
            water += min(r30 / 150.0, 1.0) * 32.0
    elif r30 is not None:
        water = min(r30 / 150.0, 1.0) * 100.0
    if humidity is not None and humidity < 40:
        water -= 8.0
    if humidity is not None and humidity > 75:
        water += 4.0
    water = _clamp(water)

    texture = str(context.get("soil_texture") or "").strip()
    organic = str(context.get("organic_matter") or "").strip().lower()
    irrigation = str(context.get("irrigation") or "").strip()

    # Measured OpenLandMap values take precedence over farmer estimates, and the
    # evidence text says which one was used.
    measured_texture = str(soil_data.get("texture_class") or "").strip()
    measured_organic = str(soil_data.get("organic_matter_class") or "").strip().lower()
    soil_measured = bool(measured_texture or measured_organic)
    texture = measured_texture or texture
    organic = measured_organic or organic
    texture_lower = texture.lower()

    # Ground-truth soil test values override the modelled classes where supplied.
    test_ph = context.get("soil_test_ph")
    test_om = context.get("soil_test_om")
    test_n = context.get("soil_test_n")
    test_p = context.get("soil_test_p")
    test_k = context.get("soil_test_k")
    has_soil_test = any(
        value is not None
        for value in (test_ph, test_om, test_n, test_p, test_k)
    )

    if test_om is not None:
        organic = "low" if float(test_om) < 1.0 else "medium" if float(test_om) < 2.0 else "high"

    # Slope adds erosion pressure on top of rain and wind.
    slope_pct = terrain_data.get("slope_pct") if terrain_data.get("available") else None
    slope_class = terrain_data.get("slope_class") if terrain_data.get("available") else None

    erosion = 90.0
    if r30 is not None:
        erosion -= max(0.0, r30 - 70.0) * 0.16
    if wind is not None:
        erosion -= max(0.0, wind - 3.5) * 7.0
    if slope_pct is not None:
        erosion -= max(0.0, float(slope_pct) - 2.0) * 2.4
    if "sand" in texture_lower:
        erosion -= 10.0
    elif "clay" in texture_lower:
        erosion -= 3.0
    if organic == "low":
        erosion -= 8.0
    elif organic == "high":
        erosion += 5.0
    erosion = _clamp(erosion)

    volatility = 85.0
    if r30 is not None and r7 is not None and r30 > 0:
        weekly_pace = r7 * 30.0 / 7.0
        swing = abs(weekly_pace - r30) / max(r30, 20.0)
        volatility -= min(swing, 1.5) * 22.0
    if tmax is not None and tmin is not None:
        volatility -= max(0.0, (tmax - tmin) - 12.0) * 2.2
    if gdd is not None:
        volatility -= min(max(0.0, gdd - 500.0) / 25.0, 12.0)
    volatility = _clamp(volatility)

    soil_score = (soil * 100.0) if soil is not None else 55.0
    if organic == "low":
        soil_score -= 14.0
    elif organic == "high":
        soil_score += 8.0
    if "sand" in texture_lower:
        soil_score -= 6.0
    elif "clay" in texture_lower:
        soil_score += 2.0
    if test_ph is not None:
        ph_value = float(test_ph)
        if 6.5 <= ph_value <= 7.5:
            soil_score += 6.0
        elif ph_value < 5.5 or ph_value > 8.5:
            soil_score -= 8.0
    if test_n is not None and float(test_n) < 280:
        soil_score -= 4.0
    if test_p is not None and float(test_p) < 10:
        soil_score -= 3.0
    if test_k is not None and float(test_k) < 120:
        soil_score -= 3.0
    soil_score = _clamp(soil_score)

    # Frost risk: a cold-night score that only bites below about 4 C.
    frost = 100.0
    if tmin is not None:
        frost -= max(0.0, 4.0 - tmin) * 14.0
    frost = _clamp(frost)

    # Chronic moisture deficit from the measured water balance, if computed.
    balance = climate.get("water_balance_mm") if climate.get("available") else None
    deficit = 100.0
    if balance is not None:
        deficit = _clamp(70.0 + float(balance) * 0.35)
    elif soil is not None:
        deficit = _clamp(soil * 100.0)
    deficit = _clamp(deficit)

    dry_spell = climate.get("longest_dry_spell_days") if climate.get("available") else None
    current_dry = climate.get("current_dry_spell_days") if climate.get("available") else None
    rain_trend = climate.get("rain_trend") if climate.get("available") else None
    balance_label = climate.get("water_balance_label") if climate.get("available") else None

    indicators = [
        {
            "key": "heat",
            "label": "Canopy heat stress",
            "score": int(round(heat)),
            "level": _concern(heat),
            "value": f"{_fmt(tavg, 1, 'C')} avg / {_fmt(tmax, 1, 'C')} peak",
            "evidence": (
                "Crops slow down when daytime peaks pass roughly 32 C. "
                f"This window averaged {_fmt(tavg, 1, 'C')} with a {_fmt(tmax, 1, 'C')} peak. "
                f"Sunlight peaked at {_fmt(climate.get('solar_max_mj'), 1, 'MJ/m2/day')} and "
                f"wind at {_fmt(climate.get('wind_max_ms'), 1, 'm/s')}."
            ),
            "missions": _sources("MODIS" if has_lst else ""),
        },
        {
            "key": "frost",
            "label": "Frost risk",
            "score": int(round(frost)),
            "level": _concern(frost),
            "value": f"coldest night {_fmt(tmin, 1, 'C')}",
            "evidence": (
                f"The coldest night in the window was {_fmt(tmin, 1, 'C')}. "
                "A reading near or below 4 C is where sensitive crops and flowers "
                "start to take damage. This is a 30-day snapshot, not a frost forecast."
            ),
            "missions": _sources(),
        },
        {
            "key": "water",
            "label": "Water availability",
            "score": int(round(water)),
            "level": _concern(water),
            "value": f"root-zone {_fmt(soil, 2)}",
            "evidence": (
                f"Root-zone wetness is {_fmt(soil, 2)} on a 0 dry to 1 wet scale, "
                f"with {_fmt(r30, 0, 'mm')} of rain in 30 days. "
                f"Humidity averaged {_fmt(humidity, 0, '%')} and the last week was "
                f"{rain_trend or 'not comparable'}."
            ),
            "missions": _sources("SMAP" if has_smap else ""),
        },
        {
            "key": "deficit",
            "label": "Chronic moisture deficit",
            "score": int(round(deficit)),
            "level": _concern(deficit),
            "value": f"{_fmt(balance, 0, 'mm')} rain minus ET0",
            "evidence": (
                f"Rain over the window was {_fmt(climate.get('rain_total_mm'), 0, 'mm')} "
                f"against an estimated {_fmt(climate.get('et0_total_mm'), 0, 'mm')} of "
                f"reference evapotranspiration, a {balance_label or 'unknown'} balance. "
                f"The longest dry run was {dry_spell if dry_spell is not None else 'unknown'} "
                f"days, with {current_dry if current_dry is not None else 'unknown'} dry days at the end."
            ),
            "missions": _sources(),
        },
        {
            "key": "erosion",
            "label": "Soil erosion pressure",
            "score": int(round(erosion)),
            "level": _concern(erosion),
            "value": f"{_fmt(r30, 0, 'mm')} rain / {_fmt(wind, 1, 'm/s')} wind",
            "evidence": (
                f"Seasonal rain of {_fmt(r30, 0, 'mm')} and wind of "
                f"{_fmt(wind, 1, 'm/s')} act on {texture or 'an unrecorded'} soil "
                f"with {organic or 'unrecorded'} organic matter. "
                + (
                    f"The surrounding terrain is {slope_class} at about "
                    f"{_fmt(slope_pct, 1, '%')} slope, which raises runoff risk."
                    if slope_pct is not None
                    else "Slope could not be estimated for this point."
                )
            ),
            "missions": _sources(
                "MODIS" if has_ndvi else "",
                "OpenLandMap" if has_soil_map else "",
            ),
        },
        {
            "key": "volatility",
            "label": "Rainfall volatility",
            "score": int(round(volatility)),
            "level": _concern(volatility),
            "value": f"{_fmt(r7, 0, 'mm')} in 7 d",
            "evidence": (
                f"The last 7 days brought {_fmt(r7, 0, 'mm')} against "
                f"{_fmt(r30, 0, 'mm')} across 30 days. A wide gap means the season "
                "is arriving in bursts rather than steady rain. "
                f"The longest dry spell was {dry_spell if dry_spell is not None else 'unknown'} days."
            ),
            "missions": _sources(),
        },
        {
            "key": "soil",
            "label": "Soil condition signal",
            "score": int(round(soil_score)),
            "level": _concern(soil_score),
            "value": f"organic matter {organic or 'not given'}",
            "evidence": (
                f"Root-zone moisture contributes {_fmt(soil, 2)} to this signal. "
                + (
                    "Your soil test results were used for pH, organic matter and NPK. "
                    if has_soil_test
                    else ""
                )
                + (
                    f"OpenLandMap maps this point as {texture or 'unclassified'} "
                    f"with {_fmt(soil_data.get('organic_matter_pct'), 2, '%')} organic "
                    f"matter, pH {_fmt(soil_data.get('ph'), 1)} (0-30 cm, 120 m)."
                    if soil_measured
                    else f"Texture is {texture or 'not given'} and organic matter is "
                    f"{organic or 'not given'}; both are farmer inputs, not satellite readings."
                )
            ),
            "missions": _sources(
                "SMAP" if has_smap else "",
                "OpenLandMap" if has_soil_map else "",
            ),
        },
    ]

    weakest = sorted(indicators, key=lambda item: item["score"])[:2]
    headline = (
        f"{weakest[0]['label']} is the tightest constraint, "
        f"followed by {weakest[1]['label'].lower()}."
    )

    scores = {
        "water_security": int(round(water)),
        "soil_health": int(round(_clamp(erosion * 0.55 + soil_score * 0.45))),
        "climate_resilience": int(round(_clamp(heat * 0.5 + volatility * 0.5))),
        "income_stability": 55,
        "cost_efficiency": 62,
        "labour_efficiency": 66,
    }

    available = [
        "Temperature and growing degree days",
        "Rainfall totals and 7-day pace",
        "Humidity and wind",
        "Root-zone wetness",
        "Solar radiation",
        "Dry spells and a rainfall-minus-ET0 water balance",
    ]
    if terrain_data.get("available"):
        available.append(
            f"Terrain slope and elevation ({TERRAIN_SOURCE}, 3x3 grid)"
        )
    if has_soil_test:
        available.append("Your soil test results (pH, organic matter, NPK)")

    missing = [
        "Field boundary and crop cover (draw a polygon; Landsat is not queried separately)",
        "Groundwater storage trend (GRACE-FO is not queried separately)",
        "Actual yields and household income history",
    ]
    if not terrain_data.get("available"):
        missing.append("Slope and elevation (terrain lookup unavailable for this point)")
    missing.append(
        "Sub-surface soil moisture below the root zone and compaction (not measured)"
    )

    if soil_data.get("available"):
        available.append(
            "Soil texture, pH, organic carbon and bulk density "
            "(OpenLandMap 0-30 cm, 120 m)"
        )
        missing.append(
            "Soil depth, compaction and plant-available nutrients (not measured)"
        )
    else:
        missing.append(
            "Soil texture and organic matter (OpenLandMap unavailable; using your inputs only)"
        )

    if readings:
        labels = ", ".join(
            str(item.get("label") or key) for key, item in readings.items()
        )
        available.append(f"AppEEARS satellite point samples: {labels}")
        missing.append(
            "Cloud-free daily imagery between satellite overpasses; MODIS "
            "composites can hide a cloudy week"
        )
    elif satellite_data.get("status") in ("queued", "running"):
        missing.append("MODIS/SMAP satellite samples (AppEEARS task still processing)")
    elif satellite_data.get("status") == "unavailable":
        missing.append(
            "MODIS/SMAP satellite samples (AppEEARS not configured; add Earthdata credentials)"
        )
    else:
        missing.append(
            "MODIS NDVI / land-surface temperature and SMAP soil moisture "
            "(AppEEARS unavailable for this point)"
        )

    sources = ["NASA POWER agro-climatology"]
    if soil_data.get("available"):
        sources.append("OpenLandMap SoilDB 2020-2022 (0-30 cm)")
    if readings:
        sources.append("NASA AppEEARS MODIS/SMAP point sample")

    used_missions: list[str] = []
    for indicator in indicators:
        for name in indicator.get("missions") or []:
            if name not in used_missions:
                used_missions.append(name)

    return {
        "available": True,
        "headline": headline,
        "period": nasa.get("period"),
        "indicators": indicators,
        "scores": scores,
        "coverage": {
            "available": available,
            "missing": missing,
            "note": (
                "Readings come from "
                + ", ".join(sources)
                + ". Active chips are sources queried for this field. Muted chips "
                "are missions from the original concept that this prototype does "
                "not query, so no value is invented for them."
            ),
        },
        "missions": used_missions,
        "missions_not_queried": MISSION_NOT_QUERIED,
        "soil": soil_data,
        "satellite": satellite_data,
        "terrain": terrain_data,
        "climate": climate,
        "moisture": moisture,
        "nourishment": nourishment,
        "root_zone": root_zone,
        "bare_field": bare_field,
        "climatology": climatology_data,
        "data_grid": data_grid,
        "context": context,
        "nasa_summary": {
            "tmax_c": tmax,
            "tmin_c": tmin,
            "wind_speed_ms": wind,
        },
    }


def build_baseline_scenario(context: dict, diagnostics: dict, weights: dict) -> dict:
    """The 'do nothing different' benchmark every scenario is compared against."""
    context = context or {}
    scores = dict(diagnostics.get("scores") or _neutral_scores())
    current = str(context.get("current_rotation") or "").strip() or "Current practice"
    indicators = diagnostics.get("indicators") or []
    weakest = sorted(indicators, key=lambda item: item["score"])[:2]

    reasons = [item["evidence"] for item in weakest]
    tradeoffs = [
        "No new seed, input or planning cost",
        "Your existing labour calendar stays the same",
        "Today's water, soil and heat exposure continues",
        "Soil organic matter and erosion risk do not improve",
    ]

    scenario = {
        "name": "Continue current rotation",
        "rotation": current,
        "is_baseline": True,
        "season_plan": "Keep your existing crop calendar.",
        "why": (
            "This is the benchmark, not an AI recommendation. It shows what the "
            "field is likely to deliver if nothing about the rotation, soil cover "
            "or water plan changes."
        ),
        "steps": [],
        "benefits": [
            "No change to this season's plan",
            "You already know the crops, labour and input needs",
        ],
        "tradeoffs": tradeoffs,
        "soil_health_gain": (
            "No soil-health gain. The same crops, the same soil cover and the same "
            "residue handling continue, so measured organic matter, texture and pH "
            "stay on their current path."
        ),
        "weather_resilience": (
            "No added weather buffer. The same planting window and the same crops "
            "face the measured heat peak, rain bursts and dry spells without a "
            "second season or a tolerant crop."
        ),
        "water_need": _water_need_from_score(scores["water_security"]),
        "risk": (
            "The vulnerabilities in the diagnosis stay in place, and a poor rain "
            "month hits the field with no buffer."
        ),
        "scores": scores,
        "exposure_window": "Same as your current calendar",
        "confidence": "medium",
        "confidence_reason": (
            "Estimated from this field's recent NASA signals and your soil inputs, "
            "not from an AI-generated scenario."
        ),
        "evidence": reasons
        or ["No NASA evidence is available for this benchmark yet."],
        "data_gaps": (diagnostics.get("coverage") or {}).get("missing", [])[:3],
        "tradeoff_scores": _clean_tradeoffs(
            {},
            scores,
            _water_need_from_score(scores["water_security"]),
        ),
        "calendar": _clean_calendar([], current),
        "risk_alerts": _clean_risk_alerts(
            [],
            current,
            _water_need_from_score(scores["water_security"]),
            _clean_tradeoffs(
                {},
                scores,
                _water_need_from_score(scores["water_security"]),
            ),
            scores,
        ),
        "overall": _weighted_overall(scores, weights),
    }
    return scenario


def _water_level(text: str) -> str:
    """Reduce a water-need phrase to low / medium / high."""
    lowered = str(text or "medium").lower()
    found = [(lowered.find(level), level) for level in ("low", "medium", "high")]
    found = [(position, level) for position, level in found if position != -1]
    return min(found)[1] if found else "medium"


def _string_list(value, limit: int = 5) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]


def _clean_scores(raw, fallback: dict) -> dict:
    scores = {}
    raw = raw if isinstance(raw, dict) else {}
    for dimension in SCORE_KEYS:
        try:
            value = float(raw.get(dimension))
        except (TypeError, ValueError):
            value = float(fallback.get(dimension, 50))
        scores[dimension] = int(round(_clamp(value)))
    return scores


TRADEOFF_KEYS = (
    "water_use",
    "organic_matter",
    "erosion_control",
    "forage_value",
    "market_viability",
)

TRADEOFF_LABELS = {
    "water_use": "Water use efficiency",
    "organic_matter": "Organic matter build-up",
    "erosion_control": "Erosion control",
    "forage_value": "Forage and residue value",
    "market_viability": "Local market viability",
}

CALENDAR_ROLES = ("main", "cover", "legume", "fallow")
MONTH_NAMES = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def _clean_tradeoffs(raw, scores: dict, water_need: str) -> dict:
    """Rate the five trade-off dimensions 0-100; higher is always better."""
    raw = raw if isinstance(raw, dict) else {}
    water_base = {"low": 82, "medium": 58, "high": 34}.get(water_need, 58)
    fallback = {
        "water_use": water_base,
        "organic_matter": scores.get("soil_health", 50),
        "erosion_control": scores.get("soil_health", 50),
        "forage_value": 50,
        "market_viability": scores.get("income_stability", 50),
    }
    result = {}
    for key in TRADEOFF_KEYS:
        try:
            value = float(raw.get(key))
        except (TypeError, ValueError):
            value = float(fallback[key])
        result[key] = int(round(_clamp(value)))
    return result


def _clean_calendar(raw, rotation: str) -> list[dict]:
    """Normalise a 3-year crop calendar; fall back to one main crop in year 1."""
    entries = []
    if isinstance(raw, list):
        for item in raw[:14]:
            if not isinstance(item, dict):
                continue
            crop = str(item.get("crop") or "").strip()[:60]
            if not crop:
                continue
            try:
                start = int(item.get("start_month"))
                end = int(item.get("end_month"))
                year = int(item.get("year") or 1)
            except (TypeError, ValueError):
                continue
            if not (1 <= start <= 12 and 1 <= end <= 12):
                continue
            year = max(1, min(3, year))
            role = str(item.get("role") or "main").strip().lower()
            if role not in CALENDAR_ROLES:
                role = "main"
            entries.append(
                {
                    "crop": crop,
                    "start_month": start,
                    "end_month": end,
                    "role": role,
                    "year": year,
                }
            )
    if entries:
        return entries

    first = str(rotation or "Main crop").split("->")[0].strip()[:60] or "Main crop"
    return [{"crop": first, "start_month": 6, "end_month": 10, "role": "main", "year": 1}]


def _clean_risk_alerts(
    raw,
    rotation: str,
    water_need: str,
    tradeoffs: dict,
    scores: dict,
) -> list[dict]:
    """Normalise agronomic risk alerts, with an honest deterministic fallback."""
    alerts = []
    if isinstance(raw, list):
        for item in raw[:4]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("type") or item.get("label") or "").strip()[:60]
            note = str(item.get("note") or item.get("detail") or "").strip()[:240]
            if not label or not note:
                continue
            severity = _water_level(item.get("severity") or "medium")
            if severity not in ("low", "medium", "high"):
                severity = "medium"
            alerts.append({"type": label, "severity": severity, "note": note})
    if alerts:
        return alerts

    crop_names = [
        part.strip().lower()
        for part in str(rotation or "").replace("->", ",").split(",")
        if part.strip()
    ]
    repeated = len(crop_names) != len(set(crop_names))
    if repeated:
        alerts.append(
            {
                "type": "Disease carryover",
                "severity": "high",
                "note": (
                    "The same crop appears twice in the rotation. Pests and soil-borne "
                    "diseases can build up when a host crop returns too soon."
                ),
            }
        )
    if str(water_need).lower() == "high":
        alerts.append(
            {
                "type": "Water demand",
                "severity": "high",
                "note": (
                    "This rotation needs a lot of water. A long dry spell or a weak "
                    "irrigation supply can cut yield."
                ),
            }
        )
    market = int(tradeoffs.get("market_viability", 50))
    if market < 45:
        alerts.append(
            {
                "type": "Limited buyers",
                "severity": "high",
                "note": (
                    "Local demand for one or more crops may be thin. Line up a buyer "
                    "or a storage plan before planting."
                ),
            }
        )
    elif market < 60:
        alerts.append(
            {
                "type": "Market access",
                "severity": "medium",
                "note": (
                    "Check the local price and buyer before planting. A good harvest "
                    "is only useful if it can be sold."
                ),
            }
        )
    if int(tradeoffs.get("organic_matter", 50)) < 45:
        alerts.append(
            {
                "type": "Slow soil recovery",
                "severity": "medium",
                "note": (
                    "Organic matter may build slowly. Keep residues on the field and "
                    "avoid burning or deep tillage."
                ),
            }
        )
    if int(scores.get("cost_efficiency", 60)) < 45:
        alerts.append(
            {
                "type": "Upfront seed cost",
                "severity": "medium",
                "note": (
                    "Seed and input costs are high for this rotation. Check whether "
                    "the extra cost is covered by the expected income."
                ),
            }
        )
    if not alerts:
        alerts.append(
            {
                "type": "Local verification",
                "severity": "low",
                "note": (
                    "No major downside was detected from the measured data, but confirm "
                    "seed availability and market demand locally."
                ),
            }
        )
    return alerts[:4]


def _month_label(month: int) -> str:
    return MONTH_NAMES[max(1, min(12, int(month))) - 1]


def _normalize_recommendations(
    data: dict,
    model: str,
    weights: dict,
    diagnostics: dict,
    context: dict,
    priorities: dict,
) -> dict:
    scenarios = data.get("scenarios") or data.get("rotations")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("AI reply did not contain any crop scenarios")

    fallback_scores = diagnostics.get("scores") or _neutral_scores()
    cleaned = []
    for index, item in enumerate(scenarios[:3], 1):
        if not isinstance(item, dict):
            continue
        benefits = item.get("benefits")
        steps = item.get("steps")
        scores = _clean_scores(item.get("scores"), fallback_scores)
        water_need = _water_level(item.get("water_need"))
        cleaned.append(
            {
                "name": str(item.get("name") or f"Scenario {index}"),
                "rotation": str(item.get("rotation") or "Rotation"),
                "is_baseline": False,
                "season_plan": str(item.get("season_plan") or ""),
                "why": str(item.get("why") or ""),
                "steps": [
                    re.sub(
                        r"^\s*step\s*\d+\s*[:.\-]\s*",
                        "",
                        str(step),
                        flags=re.IGNORECASE,
                    ).strip()
                    for step in steps
                ][:5]
                if isinstance(steps, list)
                else [],
                "benefits": _string_list(benefits, 4),
                "tradeoffs": _string_list(item.get("tradeoffs"), 4),
                "soil_health_gain": str(item.get("soil_health_gain") or ""),
                "weather_resilience": str(item.get("weather_resilience") or ""),
                "water_need": water_need,
                "risk": str(item.get("risk") or ""),
                "scores": scores,
                "tradeoff_scores": _clean_tradeoffs(
                    item.get("tradeoff_scores"), scores, water_need
                ),
                "calendar": _clean_calendar(item.get("calendar"), item.get("rotation") or ""),
                "risk_alerts": _clean_risk_alerts(
                    item.get("risk_alerts"),
                    str(item.get("rotation") or ""),
                    water_need,
                    _clean_tradeoffs(item.get("tradeoff_scores"), scores, water_need),
                    scores,
                ),
                "exposure_window": str(item.get("exposure_window") or ""),
                "confidence": _water_level(item.get("confidence") or "medium")
                if _water_level(item.get("confidence") or "medium")
                in ("low", "medium", "high")
                else "medium",
                "confidence_reason": str(item.get("confidence_reason") or ""),
                "evidence": _string_list(item.get("evidence"), 3),
                "data_gaps": _string_list(item.get("data_gaps"), 3),
                "overall": _weighted_overall(scores, weights),
            }
        )

    if not cleaned:
        raise ValueError("AI reply had no usable crop scenarios")

    # Rank the improved scenarios best-first; the baseline stays pinned first.
    cleaned.sort(key=lambda scenario: scenario["overall"], reverse=True)

    baseline = build_baseline_scenario(context, diagnostics, weights)
    all_scenarios = [baseline] + cleaned
    ranked = sorted(all_scenarios, key=lambda scenario: scenario["overall"], reverse=True)
    for position, scenario in enumerate(ranked, 1):
        scenario["rank"] = position

    uncertainty = data.get("uncertainty")
    if not isinstance(uncertainty, dict):
        uncertainty = {}
    gaps = _string_list(uncertainty.get("data_gaps"), 5) or list(
        (diagnostics.get("coverage") or {}).get("missing", [])
    )[:4]
    assumptions = _string_list(uncertainty.get("assumptions"), 4) or [
        "NASA POWER values are a 30-day area signal, not an on-farm sensor reading.",
        "Scores compare scenarios against each other; they are not guaranteed yields.",
        "Local prices, labour availability and market access are not modelled.",
    ]

    return {
        "summary": str(data.get("summary") or ""),
        "scenarios": all_scenarios,
        "caveats": str(data.get("caveats") or ""),
        "uncertainty": {
            "summary": str(
                uncertainty.get("summary")
                or "These are planning scenarios, not forecasts. Check them against "
                "your own records and a local agronomist."
            ),
            "data_gaps": gaps,
            "assumptions": assumptions,
        },
        "priorities": priorities,
        "weights": weights,
        "context": context,
        "diagnostics": diagnostics,
        "model": model,
    }


def generate_crop_rotations(
    model: str,
    environment: dict,
    priorities: dict | None = None,
    context: dict | None = None,
    diagnostics: dict | None = None,
) -> dict:
    """Ask Groq for 3 improved rotation scenarios plus trade-off reasoning."""
    location = environment.get("location") or {}
    nasa = _nasa_for_prompt(environment.get("nasa"))
    context = context or {}
    priorities = _clean_priorities(priorities)
    weights = _priority_weights(priorities)
    diagnostics = diagnostics or build_diagnostics(
        environment.get("nasa"),
        context,
        environment.get("soil"),
        environment.get("satellite"),
        environment.get("terrain"),
        environment.get("climatology"),
        location.get("lat"),
    )

    soil_data = environment.get("soil") or {}
    satellite_data = environment.get("satellite") or {}
    readings = satellite_data.get("readings") or {}

    priority_lines = "\n".join(
        f"- {PRIORITY_LABELS[key]}: {int(value)} out of 5"
        for key, value in priorities.items()
    )
    context_lines = "\n".join(
        [
            f"- Soil texture: {context.get('soil_texture') or 'not given'}",
            f"- Organic matter: {context.get('organic_matter') or 'not given'}",
            f"- Irrigation: {context.get('irrigation') or 'not given'}",
            f"- Field size: {context.get('field_size_ha') or 'not given'} hectares",
            f"- Current rotation: {context.get('current_rotation') or 'not given'}",
        ]
    )
    if soil_data.get("available"):
        soil_lines = "\n".join(
            [
                f"- Source: {soil_data.get('source')}",
                f"- Texture class: {soil_data.get('texture_class') or 'not returned'}",
                f"- Sand / clay / silt: {_fmt(soil_data.get('sand_pct'), 1, '%')} / "
                f"{_fmt(soil_data.get('clay_pct'), 1, '%')} / "
                f"{_fmt(soil_data.get('silt_pct'), 1, '%')}",
                f"- pH in water: {_fmt(soil_data.get('ph'), 1)}",
                f"- Organic matter: {_fmt(soil_data.get('organic_matter_pct'), 2, '%')} "
                f"({soil_data.get('organic_matter_class') or 'n/a'})",
                f"- Organic carbon: {_fmt(soil_data.get('organic_carbon_g_kg'), 1, 'g/kg')}",
                f"- Bulk density: {_fmt(soil_data.get('bulk_density_g_cm3'), 2, 'g/cm3')}",
            ]
        )
    else:
        soil_lines = "- Not available for this point; do not invent soil values."

    if readings:
        satellite_lines = "\n".join(
            f"- {item.get('label') or key}: mean {_fmt(item.get('mean'), 3)} "
            f"{item.get('unit') or ''}, min {_fmt(item.get('min'), 3)}, "
            f"max {_fmt(item.get('max'), 3)}, samples {item.get('samples')}"
            for key, item in readings.items()
        )
    else:
        satellite_lines = (
            f"- No satellite readings yet (AppEEARS status: "
            f"{satellite_data.get('status') or 'idle'}). Do not invent satellite values."
        )

    indicator_lines = "\n".join(
        f"- {item['label']}: {item['level']} concern (safety {item['score']}/100). "
        f"{item['evidence']}"
        for item in diagnostics.get("indicators", [])
    )
    available = ", ".join(
        (diagnostics.get("coverage") or {}).get("available", [])
    ) or "none"
    missing = "; ".join(
        (diagnostics.get("coverage") or {}).get("missing", [])
    ) or "none recorded"

    climate = diagnostics.get("climate") or {}
    moisture = diagnostics.get("moisture") or {}
    terrain = diagnostics.get("terrain") or {}
    nourishment = diagnostics.get("nourishment") or {}
    root_zone = diagnostics.get("root_zone") or {}

    if climate.get("available"):
        climate_lines = "\n".join(
            [
                f"- Rain total: {_fmt(climate.get('rain_total_mm'), 1, 'mm')}",
                f"- Estimated reference ET0: {_fmt(climate.get('et0_total_mm'), 1, 'mm')}",
                f"- Rain minus ET0: {_fmt(climate.get('water_balance_mm'), 1, 'mm')} "
                f"({climate.get('water_balance_label')})",
                f"- Longest dry spell: {climate.get('longest_dry_spell_days')} days; "
                f"current dry run: {climate.get('current_dry_spell_days')} days",
                f"- Rain trend: {climate.get('rain_trend')} "
                f"({_fmt(climate.get('rain_change_mm'), 1, 'mm')} week on week)",
                f"- Days above 35 C: {climate.get('heat_days_35c')}; "
                f"sunny days: {climate.get('sunny_days')}",
            ]
        )
    else:
        climate_lines = "- Climate analytics unavailable; do not invent values."

    if moisture.get("available"):
        moisture_lines = "\n".join(
            [
                f"- POWER root-zone wetness mean: "
                f"{_fmt((moisture.get('root_wetness') or {}).get('mean'), 3)} "
                f"({(moisture.get('root_wetness') or {}).get('label')})",
                f"- SMAP surface moisture: {_fmt(moisture.get('smap_surface'), 3)}; "
                f"root zone: {_fmt(moisture.get('smap_rootzone'), 3)}",
                f"- Surface-to-root gradient: {_fmt(moisture.get('gradient'), 3)} "
                f"({moisture.get('gradient_note')})",
            ]
        )
    else:
        moisture_lines = "- Moisture analytics unavailable; do not invent values."

    if terrain.get("available"):
        terrain_lines = "\n".join(
            [
                f"- Elevation: {_fmt(terrain.get('elevation_m'), 0, 'm')}; "
                f"relief: {_fmt(terrain.get('relief_m'), 0, 'm')}",
                f"- Slope: {_fmt(terrain.get('slope_pct'), 1, '%')} "
                f"({terrain.get('slope_class')})",
                f"- Source: {terrain.get('source')}. {terrain.get('note')}",
            ]
        )
    else:
        terrain_lines = "- Terrain unavailable; do not invent a slope."

    nourishment_items = "; ".join(
        f"{item.get('label')}: {item.get('value')} ({item.get('status')})"
        for item in nourishment.get("items", [])
    )
    root_lines = "\n".join(
        [
            f"- Effective root depth: {_fmt(root_zone.get('depth_cm'), 0, 'cm')} "
            f"({root_zone.get('texture')})",
            f"- Plant-available water capacity: "
            f"{_fmt(root_zone.get('awc_mm'), 0, 'mm')}",
            f"- Soil nourishment: {nourishment_items or 'no measured values'}",
        ]
    )

    grid = diagnostics.get("data_grid") or {}
    grid_lines = "\n".join(
        f"- {row.get('mission')} | {row.get('variable')} | {row.get('resolution')} | "
        f"{row.get('value')} | {row.get('status')}"
        for row in (grid.get("rows") or [])
    ) or "- No harmonized grid rows were built."

    seasonal = diagnostics.get("climatology") or {}
    if seasonal.get("available"):
        seasonal_lines = "\n".join(
            [
                f"- Wettest month: {seasonal.get('wettest_month', {}).get('label')} "
                f"({_fmt(seasonal.get('wettest_month', {}).get('rain_mm'), 0, 'mm')})",
                f"- Driest month: {seasonal.get('driest_month', {}).get('label')} "
                f"({_fmt(seasonal.get('driest_month', {}).get('rain_mm'), 0, 'mm')})",
                f"- Frost months (min below 2 C): "
                f"{', '.join(seasonal.get('frost_months') or []) or 'none in the last 12 months'}",
                f"- First and last frost in the window: "
                f"{seasonal.get('first_frost') or 'none'} to "
                f"{seasonal.get('last_frost') or 'none'}",
                "- Dry-hot exposure windows: "
                + (
                    "; ".join(
                        f"{item.get('label')} ({_fmt(item.get('rain_mm'), 0, 'mm')} rain)"
                        for item in seasonal.get("exposure_windows") or []
                    )
                    or "none"
                ),
            ]
        )
    else:
        seasonal_lines = "- Seasonal history unavailable; do not invent frost dates."

    prompt = (
        f"Farm location: {location.get('name', 'unknown')} "
        f"(lat {location.get('lat', 0):.4f}, lng {location.get('lng', 0):.4f})\n\n"
        "NASA POWER agro-climatology for this farm:\n"
        f"{json.dumps(nasa, indent=2)}\n\n"
        "Farmer priorities, importance from 0 to 5:\n"
        f"{priority_lines}\n\n"
        "Local context supplied by the farmer:\n"
        f"{context_lines}\n\n"
        "Measured soil at this point (OpenLandMap SoilDB, 0-30 cm, 120 m, CC-BY):\n"
        f"{soil_lines}\n\n"
        "Satellite point samples (NASA AppEEARS, last 60 days):\n"
        f"{satellite_lines}\n\n"
        "Deterministic climate analytics (rainfall, ET0, dry spells):\n"
        f"{climate_lines}\n\n"
        "Moisture analytics (POWER + SMAP):\n"
        f"{moisture_lines}\n\n"
        "Terrain and slope:\n"
        f"{terrain_lines}\n\n"
        "Soil nourishment and root zone:\n"
        f"{root_lines}\n\n"
        "Harmonized data grid (native resolution and status):\n"
        f"{grid_lines}\n\n"
        "Seasonal history from the last 12 months of NASA POWER:\n"
        f"{seasonal_lines}\n\n"
        "Server-side field diagnosis built from the same data:\n"
        f"{indicator_lines}\n"
        f"Signals available: {available}\n"
        f"Known data gaps: {missing}\n\n"
        "Design exactly 3 different one-year rotation scenarios that improve on "
        "the farmer's current practice. Do NOT include the current practice as "
        "one of your three; the app adds that baseline itself.\n\n"
        "Hard design rules for every scenario:\n"
        "- Improve soil health. Every rotation must include at least one legume, "
        "green manure or cover crop, and must keep the soil covered through the "
        "rainiest month. Say which measured soil problem it fixes (organic "
        "matter, texture, pH or erosion).\n"
        "- Strengthen weather resilience. Match crops to the measured heat peak, "
        "rain bursts, dry spells and root-zone moisture. Use at least two "
        "different growing seasons or planting windows so one bad month cannot "
        "ruin the year.\n"
        "- Match the water supply. Never recommend a water-heavy crop unless the "
        "measured rainfall and root-zone moisture can support it.\n"
        "- Use the measured soil and satellite values where present and cite "
        "them in evidence. When a source is missing, say so instead of "
        "inventing a value.\n"
        "- If a scenario cannot improve soil health or weather resilience, say "
        "so honestly in tradeoffs and give it a lower soil_health or "
        "climate_resilience score.\n\n"
        "Write for a farmer with no scientific training:\n"
        "- Use simple, everyday English and short sentences.\n"
        "- Explain numbers in plain words, for example 'about 268 mm of rain in "
        "the last month, which is a lot'.\n"
        "- Name real, locally appropriate crops and roughly which months to grow them.\n"
        "- Be honest about trade-offs. Say clearly when a scenario uses more water, "
        "costs more, needs more labour, or carries more market risk.\n"
        "- Avoid jargon and special unicode characters (plain ASCII only).\n\n"
        "Scoring rules for the six dimensions (0 to 100, higher is better):\n"
        "- water_security: how safe the scenario is under this field's water supply.\n"
        "- soil_health: expected effect on organic matter, erosion and soil structure. "
        "Judge it against the MEASURED organic matter, texture and pH above. A rotation "
        "with a legume or cover crop that targets the measured soil problem must score "
        "higher than one without. If soil data is missing, say the score is an estimate.\n"
        "- climate_resilience: ability to absorb heat, dry spells and rain bursts. "
        "Judge it against the MEASURED heat peak, 7-day rain bursts, dry spells and "
        "root-zone moisture above. A rotation with two or more staggered seasons, or "
        "a drought-tolerant crop in the driest window, must score higher than a "
        "single-season rotation.\n"
        "- income_stability: how steady and diversified the farm income is.\n"
        "- cost_efficiency: how well it uses money spent on seed, fertiliser and fuel.\n"
        "- labour_efficiency: how well it fits available labour and time.\n"
        "Scores are relative comparisons, not measured yields. They must reflect "
        "the priorities and local context, and must differ between scenarios.\n\n"
        "Trade-off scorecards (0 to 100, higher is always better):\n"
        "- water_use: water-use efficiency, so a thirsty crop scores lower.\n"
        "- organic_matter: expected build-up of soil organic matter.\n"
        "- erosion_control: how well the rotation holds soil on this slope and rainfall.\n"
        "- forage_value: useful fodder, grazing or saleable residue.\n"
        "- market_viability: how easily the farmer can sell it locally.\n"
        "Base these on the measured rain, ET0, slope, soil texture and organic matter.\n\n"
        "Multi-year calendar:\n"
        "- Show a 3-year sequence, not a single year. Use year 1 for the first year, "
        "2 for the second and 3 for the third.\n"
        "- Give 2 to 4 entries per year: main crops, cover crops, legumes and fallow.\n"
        "- start_month and end_month are whole numbers from 1 (January) to 12 (December).\n"
        "- A season that crosses New Year can use a start greater than the end.\n"
        "- role must be exactly one of: main, cover, legume, fallow.\n"
        "- year must be 1, 2 or 3.\n"
        "- Align planting and cover-crop windows with the measured frost months and "
        "the dry-hot exposure windows above. Do not invent a frost date that is not listed.\n\n"
        "Agronomic risk alerts:\n"
        "- Give 2 to 4 honest downsides for each scenario, such as disease carryover, "
        "high upfront seed cost, limited regional buyers, water demand or slow soil recovery.\n"
        "- type is a short label, severity is low, medium or high, and note explains the risk "
        "and what the farmer can do.\n\n"
        "Return one JSON object only, with this exact shape:\n"
        "{\n"
        '  "summary": "3 short sentences: the main climate constraint, how the scenarios respond, and what the farmer should weigh up",\n'
        '  "scenarios": [\n'
        "    {\n"
        '      "name": "Short memorable scenario name, e.g. Water-saver rotation",\n'
        '      "rotation": "Crop A -> Crop B -> Crop C",\n'
        '      "season_plan": "Simple months for each crop, e.g. Rice: June to October. Groundnut: November to February.",\n'
        '      "why": "3 to 4 short sentences in simple English explaining why this fits this place, using the NASA numbers in plain words.",\n'
        '      "steps": ["prepare the field ...", "sow the seeds ...", "care for the crop ...", "harvest ..."],\n'
        '      "benefits": ["short benefit in simple words", "short benefit in simple words"],\n'
        '      "tradeoffs": ["honest cost, water or labour trade-off", "what the farmer gives up"],\n'
        '      "soil_health_gain": "1 to 2 short sentences naming the measured soil problem this rotation fixes and how (legume, cover crop, residue, pH correction).",\n'
        '      "weather_resilience": "1 to 2 short sentences naming the measured heat, rain burst or dry spell this rotation buffers, and which season or crop provides the buffer.",\n'
        '      "water_need": "low",\n'
        '      "risk": "1 to 2 short sentences about what can go wrong and what the farmer can do.",\n'
        '      "scores": {"water_security": 70, "soil_health": 75, "climate_resilience": 68, "income_stability": 60, "cost_efficiency": 65, "labour_efficiency": 62},\n'
        '      "tradeoff_scores": {"water_use": 70, "organic_matter": 75, "erosion_control": 68, "forage_value": 55, "market_viability": 60},\n'
        '      "calendar": [{"crop": "Rice", "start_month": 6, "end_month": 10, "role": "main", "year": 1}, {"crop": "Groundnut", "start_month": 11, "end_month": 2, "role": "main", "year": 1}, {"crop": "Cowpea cover", "start_month": 3, "end_month": 5, "role": "cover", "year": 2}],\n'
        '      "risk_alerts": [{"type": "disease carryover", "severity": "medium", "note": "short explanation and what to do"}],\n'
        '      "exposure_window": "The weeks or season when this scenario is most exposed, in plain words",\n'
        '      "confidence": "medium",\n'
        '      "confidence_reason": "One sentence explaining why confidence is high, medium or low.",\n'
        '      "evidence": ["NASA number or diagnosis point that supports this scenario"],\n'
        '      "data_gaps": ["what this scenario still cannot know"]\n'
        "    }\n"
        "  ],\n"
        '  "caveats": "one short sentence telling the farmer to confirm with a local agronomist",\n'
        '  "uncertainty": {\n'
        '    "summary": "2 short sentences about what these scenarios cannot promise",\n'
        '    "data_gaps": ["gap 1", "gap 2"],\n'
        '    "assumptions": ["assumption 1", "assumption 2"]\n'
        "  }\n"
        "}\n\n"
        "Keep each rotation to 2-3 crops. water_need must be exactly one word: "
        "low, medium or high. confidence must be exactly one word: low, medium or "
        "high. Give 3 to 5 steps per scenario. Return exactly 3 scenarios."
    )

    messages = [
        {"role": "system", "content": RECOMMEND_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    def request_scenarios(conversation: list[dict], temperature: float = 0.4) -> dict:
        """One JSON-mode request, with a plain-mode and a repair retry."""
        try:
            completion = groq_completion(
                conversation,
                model,
                temperature=temperature,
                max_tokens=3500,
                json_mode=True,
            )
        except API_ERROR as exc:
            message = str(exc).lower()
            if "json" not in message and "response_format" not in message:
                raise
            completion = groq_completion(
                conversation, model, temperature=0.3, max_tokens=3500
            )

        text = completion.choices[0].message.content or ""
        try:
            return _ascii_tree(_extract_json(text))
        except (ValueError, json.JSONDecodeError):
            # The model occasionally emits a stray quote. Ask once for a clean copy.
            repair_messages = conversation + [
                {"role": "assistant", "content": text[:8000]},
                {
                    "role": "user",
                    "content": (
                        "That reply was not valid JSON. Return the same crop scenarios "
                        "again as one valid JSON object only. Escape every double quote "
                        "inside strings, use no line breaks inside strings, and add no "
                        "text outside the JSON."
                    ),
                },
            ]
            completion = groq_completion(
                repair_messages, model, temperature=0.2, max_tokens=3500
            )
            text = completion.choices[0].message.content or ""
            return _ascii_tree(_extract_json(text))

    def scenario_key(item: dict) -> str:
        return re.sub(
            r"[^a-z0-9]+",
            "",
            f"{item.get('rotation') or ''}{item.get('name') or ''}".lower(),
        )

    payload = request_scenarios(messages)
    collected: list[dict] = []
    seen: set[str] = set()

    def add_scenarios(raw) -> int:
        added = 0
        if not isinstance(raw, list):
            return added
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = scenario_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            collected.append(item)
            added += 1
        return added

    add_scenarios(payload.get("scenarios") or payload.get("rotations"))

    # Models often return one or two scenarios even when asked for three. Ask for
    # the missing rotations one at a time, each with a different design theme so
    # the follow-up replies do not simply echo the rotations already collected.
    themes = [
        "a legume-heavy rotation that rebuilds nitrogen and organic matter",
        "a drought-tolerant cereal rotation with a soil-covering crop",
        "a market-oriented rotation with one cash crop and a soil-building cover",
        "a two-season rotation that keeps living cover through the wettest month",
        "a low-input rotation that protects the soil surface from erosion",
    ]
    attempts = 0
    while len(collected) < 3 and attempts < len(themes):
        theme = themes[attempts]
        attempts += 1
        existing = "\n".join(
            f"- {item.get('name') or 'Scenario'}: {item.get('rotation') or ''}"
            for item in collected
        ) or "- none yet"
        compact = [
            {
                "name": item.get("name"),
                "rotation": item.get("rotation"),
                "season_plan": item.get("season_plan"),
            }
            for item in collected
        ]
        follow_up = messages + [
            {
                "role": "assistant",
                "content": json.dumps(
                    {"scenarios": compact}, ensure_ascii=True
                )[:6000],
            },
            {
                "role": "user",
                "content": (
                    f"The farmer needs three scenarios in total and {len(collected)} "
                    f"are already chosen. Add exactly ONE new scenario built around "
                    f"{theme}.\n\n"
                    f"Rotations already chosen (do not repeat these crops or this "
                    f"season plan):\n{existing}\n\n"
                    "Return one JSON object with a \"scenarios\" array containing "
                    "exactly one scenario in the same shape as before, including "
                    "scores, tradeoff_scores, a 3-year calendar, risk_alerts, "
                    "evidence and confidence. Choose different crop species and a "
                    "different planting window from the rotations listed above. "
                    "You may omit the top-level summary, caveats and uncertainty "
                    "fields in this reply."
                ),
            },
        ]
        try:
            extra = request_scenarios(follow_up, temperature=0.65)
        except (ValueError, json.JSONDecodeError):
            continue
        except API_ERROR as exc:
            # A transient provider error should not abandon the remaining slots.
            print(f"\n[AI] Scenario top-up failed: {exc}", flush=True)
            continue
        if add_scenarios(extra.get("scenarios") or extra.get("rotations")) == 0:
            continue

    if not collected:
        raise ValueError("AI reply did not contain any crop scenarios")

    if len(collected) < 3:
        print(
            f"\n[AI] Only {len(collected)} distinct rotation scenario(s) were "
            "returned after retries.",
            flush=True,
        )

    payload["scenarios"] = collected
    return _normalize_recommendations(
        payload, model, weights, diagnostics, context, priorities
    )


def log_recommendations(recommendations: dict, environment: dict) -> None:
    """Print the scenario comparison as a clean, farmer-friendly terminal block."""
    line = "=" * 62
    place = _safe_text(
        (environment.get("location") or {}).get("name", "selected location")
    )

    print(f"\n{line}")
    print("  ROTATION SCENARIOS (NASA data + your priorities)")
    print(f"  {place}")
    print(line)

    summary = _safe_text(recommendations.get("summary") or "")
    if summary:
        print(textwrap.fill(summary, 74, initial_indent="  ", subsequent_indent="  "))

    for index, scenario in enumerate(recommendations.get("scenarios", []), 1):
        tag = "BASELINE" if scenario.get("is_baseline") else f"SCENARIO {index - 1}"
        print()
        print(
            f"  {tag}  |  score {scenario.get('overall', 0)}/100  |  "
            f"{_safe_text(scenario.get('name', 'Scenario'))}"
        )
        print(f"     Rotation: {_safe_text(scenario.get('rotation', ''))}")
        season = _safe_text(scenario.get("season_plan", ""))
        if season:
            print(
                textwrap.fill(
                    f"Season  : {season}",
                    76,
                    initial_indent="     ",
                    subsequent_indent="       ",
                )
            )
        why = _safe_text(scenario.get("why", ""))
        if why:
            print(
                textwrap.fill(
                    f"Why     : {why}",
                    76,
                    initial_indent="     ",
                    subsequent_indent="       ",
                )
            )
        if scenario.get("tradeoffs"):
            tradeoffs = _safe_text("; ".join(scenario["tradeoffs"]))
            print(
                textwrap.fill(
                    f"Trade   : {tradeoffs}",
                    76,
                    initial_indent="     ",
                    subsequent_indent="       ",
                )
            )
        soil_gain = _safe_text(scenario.get("soil_health_gain", ""))
        if soil_gain:
            print(
                textwrap.fill(
                    f"Soil    : {soil_gain}",
                    76,
                    initial_indent="     ",
                    subsequent_indent="       ",
                )
            )
        weather_gain = _safe_text(scenario.get("weather_resilience", ""))
        if weather_gain:
            print(
                textwrap.fill(
                    f"Weather : {weather_gain}",
                    76,
                    initial_indent="     ",
                    subsequent_indent="       ",
                )
            )
        steps = scenario.get("steps") or []
        if steps:
            print("     Steps   :")
            for step_number, step in enumerate(steps, 1):
                print(
                    textwrap.fill(
                        f"{step_number}) {_safe_text(step)}",
                        76,
                        initial_indent="       ",
                        subsequent_indent="          ",
                    )
                )
        print(
            f"     Water   : {_safe_text(scenario.get('water_need', 'n/a'))}"
            f"  |  Confidence: {_safe_text(scenario.get('confidence', 'medium'))}"
        )
        risk = _safe_text(scenario.get("risk", ""))
        if risk:
            print(
                textwrap.fill(
                    f"Watch   : {risk}",
                    76,
                    initial_indent="     ",
                    subsequent_indent="       ",
                )
            )

    uncertainty = recommendations.get("uncertainty") or {}
    if uncertainty.get("data_gaps"):
        print()
        print("  DATA GAPS")
        for gap in uncertainty["data_gaps"]:
            print(textwrap.fill(_safe_text(str(gap)), 74, initial_indent="  - ", subsequent_indent="    "))

    caveats = _safe_text(recommendations.get("caveats") or "")
    if caveats:
        print()
        print(textwrap.fill(caveats, 74, initial_indent="  ", subsequent_indent="  "))

    print(f"{line}\n", flush=True)


# --- Stress test and disease analysis ---------------------------------------


STRESS_WATER_MULTIPLIER = {"low": 0.82, "medium": 1.0, "high": 1.28}


def build_stress_test(
    diagnostics: dict,
    scenarios: list[dict],
    rainfall_change_pct: float = 0.0,
    temp_change_c: float = 1.0,
) -> dict:
    """Stress each rotation with deterministic rainfall and temperature shifts."""
    climate = diagnostics.get("climate") or {}
    if not climate.get("available"):
        return {
            "available": False,
            "note": (
                "NASA POWER daily values are not available for this field, so the "
                "water balance cannot be projected."
            ),
            "rainfall_change_pct": rainfall_change_pct,
            "temp_change_c": temp_change_c,
            "scenarios": [],
        }

    rain_base = float(climate.get("rain_total_mm") or 0.0)
    et0_base = float(climate.get("et0_total_mm") or 0.0)
    rain_stressed = rain_base * (1.0 + rainfall_change_pct / 100.0)
    et0_stressed = et0_base * (1.0 + 0.05 * temp_change_c)
    base_balance = rain_base - et0_base

    results = []
    for scenario in scenarios:
        water_need = scenario.get("water_need") or "medium"
        multiplier = STRESS_WATER_MULTIPLIER.get(water_need, 1.0)
        et0_scenario = et0_stressed * multiplier
        balance = rain_stressed - et0_scenario
        change = balance - base_balance

        if balance < -150:
            risk = "high"
            label = "large deficit"
            note = "This rotation would run a large water deficit in this scenario."
        elif balance < -60:
            risk = "moderate"
            label = "deficit"
            note = "This rotation would need extra water or a shorter season."
        elif balance < 0:
            risk = "low"
            label = "slight deficit"
            note = "This rotation is close to balanced but still short of water."
        else:
            risk = "low"
            label = "surplus"
            note = "Rain would cover the estimated crop water use in this scenario."

        results.append(
            {
                "name": scenario.get("name") or "Scenario",
                "rotation": scenario.get("rotation") or "",
                "is_baseline": bool(scenario.get("is_baseline")),
                "water_need": water_need,
                "water_multiplier": multiplier,
                "rain_mm": round(rain_stressed, 1),
                "et0_mm": round(et0_scenario, 1),
                "balance_mm": round(balance, 1),
                "balance_change_mm": round(change, 1),
                "label": label,
                "risk": risk,
                "groundwater_flag": balance < -100,
                "note": note,
                "water_security_stressed": int(
                    round(_clamp((scenario.get("scores") or {}).get("water_security", 55) + change * 0.12))
                ),
            }
        )

    results.sort(key=lambda item: item["balance_mm"], reverse=True)
    depleting = [item["name"] for item in results if item["groundwater_flag"]]

    return {
        "available": True,
        "rainfall_change_pct": round(rainfall_change_pct, 1),
        "temp_change_c": round(temp_change_c, 1),
        "base": {
            "rain_mm": round(rain_base, 1),
            "et0_mm": round(et0_base, 1),
            "balance_mm": round(base_balance, 1),
        },
        "scenarios": results,
        "groundwater_risk": depleting,
        "note": (
            "This is a deterministic what-if, not a forecast. Rain is scaled evenly "
            "across the window and reference ET0 rises about 5 percent per degree C. "
            "Crop water use is estimated from each scenario's water_need."
        ),
    }


DISEASE_SYSTEM_PROMPT = (
    "You are a careful plant-health adviser for smallholder farmers. You are not a "
    "laboratory. You look at the farmer's description of visible plant changes, the "
    "local weather measured by NASA POWER and the field context, then list the most "
    "likely problems and practical steps. You must be honest about uncertainty, never "
    "claim a confirmed diagnosis, and always tell the farmer to confirm with a local "
    "extension officer before spending money on chemicals. Return JSON only."
)


def diagnose_plant_problem(
    model: str,
    environment: dict,
    diagnostics: dict,
    context: dict,
    description: str,
    crop: str = "",
) -> dict:
    """Ask Groq for a structured, uncertainty-aware plant-health assessment."""
    location = environment.get("location") or {}
    climate = diagnostics.get("climate") or {}
    moisture = diagnostics.get("moisture") or {}
    readings = (environment.get("satellite") or {}).get("readings") or {}

    weather_lines = "\n".join(
        [
            f"- Place: {location.get('name') or 'unknown'}",
            f"- Rain in window: {_fmt(climate.get('rain_total_mm'), 1, 'mm')}",
            f"- Estimated ET0: {_fmt(climate.get('et0_total_mm'), 1, 'mm')}",
            f"- Longest dry spell: {climate.get('longest_dry_spell_days')} days",
            f"- Days above 35 C: {climate.get('heat_days_35c')}",
            f"- Root-zone wetness: "
            f"{_fmt((moisture.get('root_wetness') or {}).get('mean'), 3)}",
            f"- Soil texture: {context.get('soil_texture') or 'not given'}",
            f"- Organic matter: {context.get('organic_matter') or 'not given'}",
            f"- Irrigation: {context.get('irrigation_status') or context.get('irrigation') or 'not given'}",
            f"- Current crop: {crop or context.get('current_rotation') or 'not given'}",
        ]
    )
    satellite_lines = "\n".join(
        f"- {item.get('label') or key}: mean {_fmt(item.get('mean'), 3)} "
        f"{item.get('unit') or ''}"
        for key, item in readings.items()
    ) or "- No satellite point samples are available."

    prompt = (
        "A farmer described the following change in the plant:\n"
        f"{description}\n\n"
        "Measured field conditions:\n"
        f"{weather_lines}\n\n"
        "Satellite point samples:\n"
        f"{satellite_lines}\n\n"
        "Give 2 to 4 plausible problems, best match first. Separate what is likely "
        "from what is only possible. If the description points to a nutrient "
        "deficiency, water stress, pest damage or a spray burn rather than a "
        "pathogen, say so. Do not recommend a specific chemical dose; a local "
        "agronomist must confirm the product, dose and legality.\n\n"
        "Return one JSON object only:\n"
        "{\n"
        '  "summary": "2 short sentences in plain English",\n'
        '  "possible_diseases": [\n'
        "    {\n"
        '      "name": "name of disease or disorder",\n'
        '      "confidence": "low",\n'
        '      "why": "why the described signs and weather fit",\n'
        '      "symptoms_matched": ["sign the farmer described"]\n'
        "    }\n"
        "  ],\n"
        '  "likely_cause": "one short sentence",\n'
        '  "immediate_actions": ["safe step the farmer can take today"],\n'
        '  "treatment_options": [\n'
        '    {"type": "cultural", "action": "what to do", "note": "why and any caution"}\n'
        "  ],\n"
        '  "prevention_next_season": ["rotation, spacing, seed or residue step"],\n'
        '  "monitor": "what to check over the next 7 days",\n'
        '  "when_to_call_expert": "the sign that means call an extension officer now",\n'
        '  "safety_note": "this is not a laboratory diagnosis; confirm before spraying"\n'
        "}\n\n"
        "confidence must be exactly one word: low, medium or high. type must be "
        "one of: cultural, biological, chemical. Use plain ASCII only."
    )

    completion = groq_completion(
        [
            {"role": "system", "content": DISEASE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model,
        temperature=0.3,
        max_tokens=1600,
        json_mode=True,
    )
    text = completion.choices[0].message.content or ""
    payload = _ascii_tree(_extract_json(text))
    return _normalize_disease(payload, description, crop)


def _normalize_disease(payload: dict, description: str, crop: str) -> dict:
    """Clamp the AI reply into a predictable, safe shape."""
    payload = payload if isinstance(payload, dict) else {}

    diseases = []
    raw_diseases = payload.get("possible_diseases")
    if isinstance(raw_diseases, list):
        for item in raw_diseases[:4]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()[:120]
            if not name:
                continue
            confidence = str(item.get("confidence") or "low").strip().lower()
            if confidence not in ("low", "medium", "high"):
                confidence = "low"
            diseases.append(
                {
                    "name": name,
                    "confidence": confidence,
                    "why": str(item.get("why") or "").strip()[:600],
                    "symptoms_matched": _string_list(item.get("symptoms_matched"), 5),
                }
            )

    if not diseases:
        raise ValueError("AI reply did not contain a usable plant-health assessment")

    treatments = []
    raw_treatments = payload.get("treatment_options")
    if isinstance(raw_treatments, list):
        for item in raw_treatments[:6]:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action") or "").strip()[:400]
            if not action:
                continue
            kind = str(item.get("type") or "cultural").strip().lower()
            if kind not in ("cultural", "biological", "chemical"):
                kind = "cultural"
            treatments.append(
                {"type": kind, "action": action, "note": str(item.get("note") or "").strip()[:400]}
            )

    return {
        "crop": crop or "not specified",
        "description": description,
        "summary": str(payload.get("summary") or "").strip(),
        "possible_diseases": diseases,
        "likely_cause": str(payload.get("likely_cause") or "").strip(),
        "immediate_actions": _string_list(payload.get("immediate_actions"), 6),
        "treatment_options": treatments,
        "prevention_next_season": _string_list(payload.get("prevention_next_season"), 6),
        "monitor": str(payload.get("monitor") or "").strip(),
        "when_to_call_expert": str(payload.get("when_to_call_expert") or "").strip(),
        "safety_note": (
            "This is not a laboratory diagnosis. Confirm the problem with a local "
            "agronomist or extension officer before buying or applying any chemical."
        ),
    }


# --- Web routes -------------------------------------------------------------


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/select")
def select_location():
    """Receive a picked location from the browser, log it, store it."""
    global selected_location, environment_data

    data = request.get_json(silent=True) or {}
    try:
        lat = float(data["lat"])
        lng = float(data["lng"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Body must be JSON with numeric 'lat' and 'lng'."), 400

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return jsonify(error="Coordinates out of range."), 400

    # Optional GPS accuracy in meters, sent by the browser's geolocation API.
    accuracy = data.get("accuracy")
    try:
        accuracy = float(accuracy) if accuracy is not None else None
    except (TypeError, ValueError):
        accuracy = None
    if accuracy is not None and accuracy < 0:
        accuracy = None

    location = {"lat": lat, "lng": lng, "accuracy": accuracy}

    with _location_lock:
        selected_location = location

    suffix = f" (GPS accuracy ~{accuracy:.0f} m)" if accuracy is not None else ""
    print(
        f"\n[Map] Location selected: lat={lat:.6f}, lng={lng:.6f}{suffix}",
        flush=True,
    )

    # Reverse geocode + NASA POWER data (best effort; never blocks a selection).
    try:
        environment = get_environment(lat, lng, accuracy)
    except Exception as exc:  # noqa: BLE001 - keep the map usable
        environment = {
            "location": dict(location, name=f"{lat:.4f}, {lng:.4f}"),
            "nasa": None,
            "soil": None,
            "errors": [f"environment lookup failed: {exc}"],
        }

    # Start the optional AppEEARS satellite sample in the background; this never
    # blocks the map selection and degrades to a reported gap without credentials.
    environment["satellite"] = start_satellite_task(lat, lng)

    with _location_lock:
        environment_data = environment

    log_environment(environment)

    field_context = _clean_context(data.get("context"))
    diagnostics = build_diagnostics(
        environment.get("nasa"),
        field_context,
        environment.get("soil"),
        environment.get("satellite"),
        environment.get("terrain"),
        environment.get("climatology"),
        lat,
    )
    print(
        textwrap.fill(
            _safe_text(f"[Field] Diagnosis: {diagnostics['headline']}"),
            74,
            initial_indent="  ",
            subsequent_indent="  ",
        ),
        flush=True,
    )

    return jsonify(
        ok=True,
        location=location,
        place=environment["location"].get("name"),
        nasa=environment.get("nasa") is not None,
        environment=environment,
        diagnostics=diagnostics,
        field_context=field_context,
        soil=environment.get("soil"),
        satellite=environment.get("satellite"),
    )


@app.post("/zones")
def field_zones():
    """Sample terrain zones inside the drawn field and flag relative hotspots."""
    data = request.get_json(silent=True) or {}
    try:
        lat = float(data["lat"])
        lng = float(data["lng"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Body must be JSON with numeric 'lat' and 'lng'."), 400

    polygon = data.get("points") if isinstance(data.get("points"), list) else []

    with _location_lock:
        environment = environment_data

    field_context = _clean_context(data.get("context"))
    diagnostics = None
    if environment:
        diagnostics = build_diagnostics(
            environment.get("nasa"),
            field_context,
            environment.get("soil"),
            environment.get("satellite"),
            environment.get("terrain"),
            environment.get("climatology"),
            lat,
        )

    result = build_hotspots(lat, lng, polygon, diagnostics)
    result["point"] = {"lat": lat, "lng": lng}
    print(
        f"\n[Zones] {result.get('summary') or result.get('note') or 'no zones'}",
        flush=True,
    )
    return jsonify(ok=True, **result)


@app.post("/recommend")
def recommend_crops():
    """Compare 3 AI rotation scenarios against the current-practice baseline."""
    global crop_recommendations

    with _location_lock:
        environment = environment_data

    if not environment:
        return jsonify(error="Select a field on the map first."), 409

    if not _groq_api_keys():
        return (
            jsonify(
                error=(
                    "Groq is not configured. Add GROQ_API_KEY (and optionally "
                    "GROQ_API_KEY_2 or GROQ_API_KEYS) to .env and restart main.py, "
                    "then try the comparison again."
                )
            ),
            503,
        )

    data = request.get_json(silent=True) or {}
    priorities = _clean_priorities(data.get("priorities"))
    field_context = _clean_context(data.get("context"))
    satellite = satellite_status(
        (environment.get("location") or {}).get("lat"),
        (environment.get("location") or {}).get("lng"),
    )
    diagnostics = build_diagnostics(
        environment.get("nasa"),
        field_context,
        environment.get("soil"),
        satellite,
        environment.get("terrain"),
        environment.get("climatology"),
        (environment.get("location") or {}).get("lat"),
    )

    try:
        recommendations = generate_crop_rotations(
            current_model,
            environment,
            priorities,
            field_context,
            diagnostics,
        )
    except ValueError as exc:
        print(f"\n[AI] Bad reply: {exc}", flush=True)
        return jsonify(error=str(exc)), 502
    except API_ERROR as exc:
        print(f"\n[AI] Groq error: {exc}", flush=True)
        return jsonify(error=f"AI request failed: {exc}"), 502
    except Exception as exc:  # noqa: BLE001 - report instead of crashing the route
        print(f"\n[AI] Unexpected error: {exc}", flush=True)
        return jsonify(error=f"AI request failed: {exc}"), 502

    with _location_lock:
        crop_recommendations = recommendations

    log_recommendations(recommendations, environment)

    return jsonify(
        ok=True,
        recommendations=recommendations,
        context=environment,
        diagnostics=diagnostics,
        field_context=field_context,
        priorities=priorities,
    )


@app.get("/satellite")
def satellite_state():
    """Live status of the background AppEEARS task for the selected point."""
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    return jsonify(ok=True, satellite=satellite_status(lat, lng))


@app.post("/stress")
def stress_test():
    """Deterministic rainfall/temperature what-if for the compared rotations."""
    global crop_recommendations

    with _location_lock:
        environment = environment_data
        cached = crop_recommendations

    if not environment:
        return jsonify(error="Select a field on the map first."), 409

    data = request.get_json(silent=True) or {}
    try:
        rainfall_change = float(data.get("rainfall_change_pct", 0.0))
    except (TypeError, ValueError):
        rainfall_change = 0.0
    try:
        temp_change = float(data.get("temp_change_c", 1.0))
    except (TypeError, ValueError):
        temp_change = 1.0
    rainfall_change = max(-20.0, min(20.0, rainfall_change))
    temp_change = max(0.0, min(3.0, temp_change))

    field_context = _clean_context(data.get("context"))
    diagnostics = build_diagnostics(
        environment.get("nasa"),
        field_context,
        environment.get("soil"),
        environment.get("satellite"),
        environment.get("terrain"),
        environment.get("climatology"),
        (environment.get("location") or {}).get("lat"),
    )

    scenarios = []
    if cached and cached.get("scenarios"):
        scenarios = list(cached["scenarios"])
    else:
        scenarios = [
            build_baseline_scenario(
                field_context,
                diagnostics,
                _priority_weights(_clean_priorities(data.get("priorities"))),
            )
        ]

    stress = build_stress_test(diagnostics, scenarios, rainfall_change, temp_change)
    return jsonify(ok=True, stress=stress, diagnostics=diagnostics)


@app.post("/disease")
def disease_analysis():
    """Turn a farmer's description of plant changes into a cautious assessment."""
    with _location_lock:
        environment = environment_data

    if not environment:
        return jsonify(error="Select a field on the map first."), 409

    data = request.get_json(silent=True) or {}
    description = str(data.get("description") or "").strip()[:1500]
    if len(description) < 10:
        return (
            jsonify(
                error=(
                    "Describe what changed on the plant in a few words, for example "
                    "'yellow spots on lower leaves with a brown edge'."
                )
            ),
            400,
        )

    if not _groq_api_keys():
        return (
            jsonify(
                error=(
                    "Groq is not configured. Add GROQ_API_KEY (and optionally "
                    "GROQ_API_KEY_2 or GROQ_API_KEYS) to .env and restart main.py, "
                    "then try again."
                )
            ),
            503,
        )

    field_context = _clean_context(data.get("context"))
    crop = str(data.get("crop") or "").strip()[:80]
    diagnostics = build_diagnostics(
        environment.get("nasa"),
        field_context,
        environment.get("soil"),
        environment.get("satellite"),
        environment.get("terrain"),
        environment.get("climatology"),
        (environment.get("location") or {}).get("lat"),
    )

    try:
        assessment = diagnose_plant_problem(
            current_model,
            environment,
            diagnostics,
            field_context,
            description,
            crop,
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 502
    except API_ERROR as exc:
        return jsonify(error=f"AI request failed: {exc}"), 502
    except Exception as exc:  # noqa: BLE001 - report instead of crashing the route
        return jsonify(error=f"AI request failed: {exc}"), 502

    print(f"\n[Plant] Assessment for {crop or 'crop'}: {assessment['possible_diseases'][0]['name']}", flush=True)
    return jsonify(ok=True, assessment=assessment)


def find_free_port(host: str, preferred: int, attempts: int = 20) -> int:
    """Return the first bindable port at or after `preferred`."""
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    return preferred


def start_map_server(port: int) -> None:
    """Run the Flask map server (called from a background thread)."""
    app.run(
        host=MAP_HOST,
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


# --- Groq chat --------------------------------------------------------------


def _groq_api_keys() -> list[str]:
    """Every configured Groq key, in fallback order, without duplicates."""
    keys: list[str] = []
    for name in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"):
        value = os.environ.get(name, "").strip()
        if value and value not in keys:
            keys.append(value)
    for value in re.split(r"[,\s]+", os.environ.get("GROQ_API_KEYS", "").strip()):
        value = value.strip()
        if value and value not in keys:
            keys.append(value)
    return keys


def _groq_client_for_key(api_key: str) -> "groq.Groq":
    with _groq_lock:
        client = _groq_clients.get(api_key)
        if client is None:
            client = groq.Groq(api_key=api_key)
            _groq_clients[api_key] = client
        return client


def _client_from_env() -> "groq.Groq | None":
    """Build a Groq client from the first configured key (web-safe)."""
    keys = _groq_api_keys()
    if not keys:
        return None
    return _groq_client_for_key(keys[0])


def get_groq_client() -> "groq.Groq | None":
    """Return a Groq client for callers that only need a configuration check."""
    global groq_client
    if groq_client is not None:
        return groq_client
    return _client_from_env()


def groq_completion(
    messages: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    json_mode: bool = False,
):
    """Call Groq and rotate to the next key when one key is rejected or limited.

    Authentication failures, rate limits, connection errors and server-side
    errors move to the next configured key. A plain 4xx (other than 429) is a
    request problem, so it is raised instead of burning every key on it.
    """
    global _groq_key_index

    keys = _groq_api_keys()
    if not keys:
        raise ValueError(
            "Groq is not configured. Add GROQ_API_KEY (and optionally "
            "GROQ_API_KEY_2, GROQ_API_KEY_3 or GROQ_API_KEYS) to .env, restart "
            "main.py, then try again."
        )

    with _groq_lock:
        start = _groq_key_index % len(keys)

    last_error: Exception | None = None
    for offset in range(len(keys)):
        index = (start + offset) % len(keys)
        client = _groq_client_for_key(keys[index])
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            completion = client.chat.completions.create(**kwargs)
        except (AUTH_ERROR, RATE_LIMIT_ERROR, CONNECTION_ERROR) as exc:
            last_error = exc
            print(
                f"\n[AI] Key {index + 1}/{len(keys)} failed "
                f"({type(exc).__name__}); trying the next key.",
                flush=True,
            )
            continue
        except API_ERROR as exc:
            status = getattr(exc, "status_code", None)
            if isinstance(status, int) and 400 <= status < 500 and status != 429:
                raise
            last_error = exc
            print(
                f"\n[AI] Key {index + 1}/{len(keys)} failed ({exc}); "
                "trying the next key.",
                flush=True,
            )
            continue

        with _groq_lock:
            _groq_key_index = index
        return completion

    if last_error is not None:
        raise last_error
    raise RuntimeError("Every configured Groq API key failed.")


def build_client() -> "groq.Groq":
    """Create the Groq client for the terminal, asking for a key if needed."""
    client = _client_from_env()
    if client is not None:
        return client

    try:
        api_key = input("GROQ_API_KEY is not set. Paste your API key: ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nNo API key provided.")
    if not api_key:
        sys.exit("No API key provided. Create one at https://console.groq.com/keys")
    # Store it so the web routes (which use the key pool) can see the same key.
    os.environ["GROQ_API_KEY"] = api_key
    return groq.Groq(api_key=api_key)


def stream_reply(client: "groq.Groq", model: str, messages: list[dict]) -> str:
    """Stream a reply to the terminal and return the full text."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        stream=True,
    )

    parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content
        if piece:
            parts.append(piece)
            print(_safe_text(piece), end="", flush=True)
    print()
    return "".join(parts)


def show_help(model: str) -> None:
    print(
        "Commands:\n"
        "  /help          show this help\n"
        "  /model [name]  show or switch the model\n"
        "  /clear         forget the conversation so far\n"
        "  /exit          quit\n"
        f"\nCurrent model: {model}\n"
    )


def main() -> None:
    global map_port, groq_client, current_model

    client = build_client()
    groq_client = client
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    current_model = model
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    map_port = find_free_port(MAP_HOST, MAP_PORT)
    threading.Thread(
        target=start_map_server, args=(map_port,), name="map-server", daemon=True
    ).start()
    print("\n  Harvest Atlas  |  NASA crop-rotation decision lab")
    print(f"  Map running at http://{MAP_HOST}:{map_port}")
    print('  Pick a field, set priorities, then press "Diagnose this field".')
    print("  NASA data is logged here; the browser compares 3 scenarios with")
    print("  your current-practice baseline.\n")

    print(f"Groq terminal chat (model: {model})")
    print("Type /help for commands, /exit to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("Bye.")
            return

        if user_input.startswith("/"):
            command, _, argument = user_input.partition(" ")
            command = command.lower()
            argument = argument.strip()

            if command == "/help":
                show_help(model)
            elif command == "/clear":
                messages = [messages[0]]
                print("Conversation cleared.\n")
            elif command == "/model":
                if argument:
                    model = argument
                    current_model = argument
                    print(f"Switched to model: {model}\n")
                else:
                    print(f"Current model: {model}\n")
            else:
                print(f"Unknown command: {command}. Type /help for options.\n")
            continue

        messages.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)
        try:
            reply = stream_reply(client, model, messages)
        except AUTH_ERROR:
            messages.pop()
            print("\nAuthentication failed. Check that GROQ_API_KEY is valid.\n")
            continue
        except RATE_LIMIT_ERROR:
            messages.pop()
            print("\nRate limit reached. Wait a moment and try again.\n")
            continue
        except CONNECTION_ERROR:
            messages.pop()
            print("\nCould not reach Groq. Check your network connection.\n")
            continue
        except API_ERROR as exc:
            messages.pop()
            print(f"\nGroq API error: {exc}\n")
            continue

        if not reply:
            # Keep the history clean if the model returned nothing.
            messages.pop()
            print("No response received. Try again.\n")
            continue

        messages.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
