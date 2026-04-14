# analyse_course_v3.py
# Streamlit app — prédiction de course
# NOUVEAUTÉS v3 :
#   1) Recalibration des références vers "conditions idéales" : section dédiée + expliquée
#   2) Interface Simple / Expert avec explications pédagogiques (⬆️⬇️ pour chaque param)
#   3) Intervalle de confiance sur la prédiction (±)
#   4) Résumé visuel des facteurs météo de la journée de course
#   5) Synthèse lisible en haut des résultats (pas juste un tableau brut)
#
# Dépendances : streamlit, gpxpy, fitparse, pandas, numpy, pydeck, matplotlib, requests

import streamlit as st
import math
import gpxpy
from fitparse import FitFile
from datetime import datetime, timedelta, date, time
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import requests

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Prédiction Course v3", layout="wide")
TZ_NAME_DEFAULT = "Europe/Paris"

# ══════════════════════════════════════════════════════════════
# STYLES CSS légers
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
.param-box {
    background: #f8f9fa;
    border-left: 4px solid #1f77b4;
    border-radius: 4px;
    padding: 8px 12px;
    margin-bottom: 8px;
    font-size: 0.88rem;
}
.param-up   { color: #d62728; font-weight: 600; }
.param-down { color: #2ca02c; font-weight: 600; }
.highlight-box {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
}
.result-metric {
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


def param_help(text_up: str, text_down: str, note: str = ""):
    """Affiche une aide pédagogique pour un paramètre."""
    note_html = f"<br><em>{note}</em>" if note else ""
    st.markdown(
        f'<div class="param-box">'
        f'<span class="param-up">⬆️ Augmenter</span> : {text_up}<br>'
        f'<span class="param-down">⬇️ Diminuer</span> : {text_down}'
        f'{note_html}</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
# MODÈLES PHYSIQUES
# ══════════════════════════════════════════════════════════════

def wbgt_simplified(T_c: float, RH: float) -> float:
    """WBGT approximé (Stull 2011) — combine chaleur + humidité."""
    try:
        RH_c = max(0.0, min(100.0, float(RH)))
        T = float(T_c)
        Tw = (T * math.atan(0.151977 * (RH_c + 8.313659) ** 0.5)
              + math.atan(T + RH_c)
              - math.atan(RH_c - 1.676331)
              + 0.00391838 * RH_c ** 1.5 * math.atan(0.023101 * RH_c)
              - 4.686035)
        Tg = T + 2.0
        return 0.7 * Tw + 0.2 * Tg + 0.1 * T
    except Exception:
        return float(T_c)


def effective_temp(T_c: float, RH: float, use_wbgt: bool) -> float:
    return wbgt_simplified(T_c, RH) if use_wbgt else float(T_c)


def altitude_vo2_multiplier(altitude_m: float, altitude_ref_m: float = 0.0) -> float:
    """Pénalité hypoxie : ~1 % par 100 m au-dessus de 1500 m (relatif à l'altitude d'entraînement)."""
    alt = max(0.0, float(altitude_m))
    alt_ref = max(0.0, float(altitude_ref_m))
    effective_alt = max(0.0, alt - max(1500.0, alt_ref))
    penalty = min(0.25, 0.01 * (effective_alt / 100.0))
    return 1.0 + penalty


def minetti_cost(grade_fraction: float) -> float:
    """Coût métabolique J/kg/m — Minetti et al. (2002, J. Exp. Biol.)"""
    g = max(-0.45, min(0.45, float(grade_fraction)))
    c = (155.4*g**5 - 30.4*g**4 - 43.3*g**3 + 46.3*g**2 + 19.5*g + 3.6)
    return max(0.1, float(c))


def minetti_multiplier(grade_pct: float) -> float:
    flat = minetti_cost(0.0)  # 3.6
    ratio = minetti_cost(float(grade_pct) / 100.0) / flat
    return float(max(0.92, min(1.35, ratio)))


def grade_multiplier_heuristic(grade_pct, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down):
    try:
        g = float(grade_pct) / 100.0
        g0u = max(1e-6, float(g0_up) / 100.0)
        g0d = max(1e-6, float(g0_down) / 100.0)
        if g >= 0:
            g_eff = math.tanh(g / g0u) * g0u
            mult = 1.0 + float(k_up) * g_eff
        else:
            g_eff = math.tanh((-g) / g0d) * g0d
            bonus = min(float(k_down) * g_eff, abs(float(down_cap)))
            mult = 1.0 - bonus
        mult = min(mult, 1.0 + float(max_up))
        mult = max(mult, 1.0 + float(max_down))
        return max(0.01, float(mult))
    except Exception:
        return 1.0


def combined_grade_multiplier(grade_pct, use_minetti, minetti_weight,
                               k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down):
    if not use_minetti:
        return grade_multiplier_heuristic(grade_pct, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down)
    m_min = minetti_multiplier(grade_pct)
    m_heu = grade_multiplier_heuristic(grade_pct, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down)
    w = max(0.0, min(1.0, float(minetti_weight)))
    return w * m_min + (1.0 - w) * m_heu


def temp_multiplier(temp_eff, opt_temp, cold_quad, hot_quad, max_penalty):
    if temp_eff is None:
        return 1.0
    d = float(temp_eff) - float(opt_temp)
    pen = hot_quad * d**2 if d >= 0 else cold_quad * (-d)**2
    return 1.0 + min(float(max_penalty), float(pen))


def wind_components(wind_speed_ms, wind_dir_from_deg, course_bearing_deg):
    if wind_speed_ms is None or wind_dir_from_deg is None:
        return 0.0, 0.0
    ws = float(wind_speed_ms)
    if ws <= 0:
        return 0.0, 0.0
    wind_to = (float(wind_dir_from_deg) + 180.0) % 360.0
    delta = math.radians((wind_to - course_bearing_deg + 540.0) % 360.0 - 180.0)
    along = ws * math.cos(delta)
    return float(max(0.0, -along)), float(max(0.0, along))  # head, tail


def wind_multiplier(head_ms, tail_ms, pace_s_per_km, drag_coeff, tail_credit, cap_head, cap_tail):
    pace = max(150.0, float(pace_s_per_km))
    v_run = 1000.0 / pace
    w_along = float(head_ms) - float(tail_ms)
    v_rel = max(0.0, v_run + w_along)
    base = max(1e-9, v_run ** 2)
    extra = (v_rel**2 - v_run**2) / base
    if extra < 0:
        extra = float(tail_credit) * extra
    mult = 1.0 + float(drag_coeff) * extra
    return float(max(1.0 + cap_tail, min(1.0 + cap_head, mult)))


def wind_gate(grade_pct, g1=2.0, g2=8.0, min_gate=0.25):
    g = max(0.0, float(grade_pct))
    if g <= g1:
        return 1.0
    if g >= g2:
        return float(min_gate)
    return float(1.0 - (g - g1) / (g2 - g1) * (1.0 - min_gate))


def cap_combined(mult_total, grade_pct, base_cap, extra_per_pct, max_cap):
    g = max(0.0, float(grade_pct))
    cap = min(float(max_cap), float(base_cap) + float(extra_per_pct) * g)
    return min(float(mult_total), 1.0 + cap)


def fatigue_multiplier(d_plus_cum, dist_cum, d_plus_total, dist_total, rate_pct, mode):
    if rate_pct <= 0:
        return 1.0
    rate = rate_pct / 100.0
    prog_dist  = min(1.0, dist_cum  / max(1.0, dist_total))
    prog_dplus = min(1.0, d_plus_cum / max(1.0, d_plus_total))
    dplus_ratio = d_plus_total / max(1.0, dist_total)
    w_dplus = min(0.8, dplus_ratio * 10.0)
    if mode == "distance":
        prog = prog_dist
    elif mode == "d_plus":
        prog = prog_dplus
    else:
        prog = w_dplus * prog_dplus + (1.0 - w_dplus) * prog_dist
    k = 2.0
    factor = (math.exp(k * prog) - 1.0) / (math.exp(k) - 1.0)
    return 1.0 + rate * factor


# ══════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════

def safe_float(val, default=0.0):
    try:
        if val is None:
            return float(default)
        if isinstance(val, str):
            s = val.strip()
            if s in ("", "nan", "none"):
                return float(default)
            return float(s.replace(",", "."))
        if isinstance(val, (float, int, np.number)):
            if np.isnan(val) or np.isinf(val):
                return float(default)
            return float(val)
        return float(val)
    except Exception:
        return float(default)


def hms_to_seconds(hms: str) -> int:
    if hms is None:
        return 0
    try:
        parts = [int(p) for p in str(hms).strip().split(":")]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            a, b = parts
            h, m, s = (0, a, b) if a >= 10 else (a, b, 0)
        elif len(parts) == 1:
            h, m, s = 0, 0, parts[0]
        else:
            return 0
        if not (0 <= m <= 59 and 0 <= s <= 59):
            return 0
        return max(0, h * 3600 + m * 60 + s)
    except Exception:
        return 0


def seconds_to_hms(s: float) -> str:
    s = int(round(s))
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"


def hms_to_timedelta(hms: str) -> timedelta:
    return timedelta(seconds=hms_to_seconds(hms))


def pace_str(secs_per_km: float) -> str:
    if secs_per_km is None or secs_per_km <= 0 or not math.isfinite(secs_per_km):
        return "0:00"
    t = int(round(float(secs_per_km)))
    return f"{t//60}:{t%60:02d}"


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def compute_dplus_dminus(elevs):
    arr = np.array([safe_float(e, np.nan) for e in elevs], dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0, 0.0
    diffs = np.diff(arr)
    return float(np.sum(np.clip(diffs, 0, None))), float(-np.sum(np.clip(diffs, None, 0)))


class SimplePoint:
    def __init__(self, lat, lon, elev=0.0, time=None):
        self.latitude = float(lat)
        self.longitude = float(lon)
        self.elevation = float(elev) if elev is not None else 0.0
        self.time = time

    def distance_3d(self, other):
        h = haversine_m(self.latitude, self.longitude, other.latitude, other.longitude)
        v = self.elevation - other.elevation
        return math.sqrt(h*h + v*v)


# ══════════════════════════════════════════════════════════════
# MÉTÉO
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_weather_minutely(lat, lon, dt_local_naive, tz_name=TZ_NAME_DEFAULT):
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               "&hourly=temperature_2m,relativehumidity_2m,wind_speed_10m,wind_direction_10m"
               f"&timezone={tz_name}")
        data = requests.get(url, timeout=20).json()
        if "hourly" not in data:
            return None
        times = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
        temps = data["hourly"]["temperature_2m"]
        winds = data["hourly"]["wind_speed_10m"]
        hums  = data["hourly"]["relativehumidity_2m"]
        wdirs = data["hourly"]["wind_direction_10m"]
        dt = dt_local_naive
        for i in range(len(times) - 1):
            if times[i] <= dt <= times[i+1]:
                r = (dt - times[i]).total_seconds() / max(1.0, (times[i+1]-times[i]).total_seconds())
                a1, a2 = float(wdirs[i]) % 360, float(wdirs[i+1]) % 360
                da = (a2 - a1 + 540.0) % 360.0 - 180.0
                return {
                    "temp": temps[i] + r*(temps[i+1]-temps[i]),
                    "wind": winds[i] + r*(winds[i+1]-winds[i]),
                    "humidity": hums[i] + r*(hums[i+1]-hums[i]),
                    "wind_dir": (a1 + r*da) % 360.0,
                }
        idx = min(range(len(times)), key=lambda i: abs(times[i]-dt))
        return {"temp": float(temps[idx]), "wind": float(winds[idx]),
                "humidity": float(hums[idx]), "wind_dir": float(wdirs[idx])}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_weather_archive_day(lat, lon, date_obj, tz_name=TZ_NAME_DEFAULT):
    try:
        ds = date_obj.strftime("%Y-%m-%d")
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               f"&start_date={ds}&end_date={ds}"
               "&hourly=temperature_2m,relativehumidity_2m,wind_speed_10m,wind_direction_10m"
               f"&timezone={tz_name}")
        data = requests.get(url, timeout=20).json()
        if "hourly" not in data:
            return None
        return ([datetime.fromisoformat(t) for t in data["hourly"]["time"]],
                data["hourly"]["temperature_2m"],
                data["hourly"]["wind_speed_10m"],
                data["hourly"]["relativehumidity_2m"],
                data["hourly"]["wind_direction_10m"])
    except Exception:
        return None


def get_avg_weather(lat, lon, start_dt, end_dt, tz_name=TZ_NAME_DEFAULT):
    if start_dt is None or end_dt is None:
        return None, None, None
    if (end_dt - start_dt).total_seconds() < 300:
        start_dt -= timedelta(minutes=2)
        end_dt   += timedelta(minutes=2)
    res = get_weather_archive_day(lat, lon, start_dt.date(), tz_name=tz_name)
    if not res:
        return None, None, None
    times, temps, winds, hums, _ = res
    selT = [T for t, T in zip(times, temps) if start_dt <= t <= end_dt]
    selW = [W for t, W in zip(times, winds) if start_dt <= t <= end_dt]
    selH = [H for t, H in zip(times, hums)  if start_dt <= t <= end_dt]
    if not selT:
        idx = min(range(len(times)), key=lambda i: abs(times[i]-start_dt))
        return float(temps[idx]), float(winds[idx]), float(hums[idx])
    return float(np.mean(selT)), float(np.mean(selW)), float(np.mean(selH))


# ══════════════════════════════════════════════════════════════
# DEM
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Correction altimétrique DEM...")
def fetch_dem_elevations(lats: tuple, lons: tuple, dataset: str = "srtm30m") -> list:
    try:
        locs = "|".join(f"{la},{lo}" for la, lo in zip(lats, lons))
        data = requests.get(f"https://api.opentopodata.org/v1/{dataset}?locations={locs}", timeout=30).json()
        if data.get("status") != "OK":
            return [None] * len(lats)
        return [r.get("elevation") for r in data["results"]]
    except Exception:
        return [None] * len(lats)


def correct_elevations_dem(points, max_points=100, dataset="srtm30m"):
    n = len(points)
    if n < 2:
        return np.array([getattr(p, "elevation", 0.0) or 0.0 for p in points])
    step = max(1, n // max_points)
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    lats = tuple(points[i].latitude for i in indices)
    lons = tuple(points[i].longitude for i in indices)
    dem = fetch_dem_elevations(lats, lons, dataset=dataset)
    cum_all = [0.0]
    for i in range(1, n):
        cum_all.append(cum_all[-1] + haversine_m(
            points[i-1].latitude, points[i-1].longitude,
            points[i].latitude, points[i].longitude))
    cum_sub = [cum_all[i] for i in indices]
    valid = [(d, e) for d, e in zip(cum_sub, dem) if e is not None]
    if len(valid) < 2:
        return np.array([getattr(p, "elevation", 0.0) or 0.0 for p in points])
    return np.interp(cum_all, [v[0] for v in valid], [v[1] for v in valid])


# ══════════════════════════════════════════════════════════════
# ANALYSE FC
# ══════════════════════════════════════════════════════════════

def analyze_hr(hr_records: list) -> dict:
    hrs = [h for h in hr_records if h is not None and 50 <= h <= 220]
    if len(hrs) < 10:
        return {"hr_max": None, "hr_avg": None, "hr_drift": None, "reliability": "inconnue"}
    arr = np.array(hrs, dtype=float)
    n = len(arr)
    hr_max = float(np.percentile(arr, 95))
    hr_avg = float(np.mean(arr))
    q1, q3 = int(n*0.25), int(n*0.75)
    drift = float(np.mean(arr[q3:])) - float(np.mean(arr[:q1]))
    reliability = "haute" if drift < 5 else ("moyenne" if drift < 12 else "basse (dérive cardiaque forte)")
    return {
        "hr_max": round(hr_max), "hr_avg": round(hr_avg),
        "hr_drift": round(drift, 1),
        "hr_threshold_est": round(hr_max * 0.88),
        "reliability": reliability
    }


# ══════════════════════════════════════════════════════════════
# PARSING FIT / TCX / GPX
# ══════════════════════════════════════════════════════════════

def parse_gpx_points(file):
    try:
        file.seek(0)
        gpx = gpxpy.parse(file)
        pts = [p for track in gpx.tracks for seg in track.segments for p in seg.points]
        return gpx, pts
    except Exception as e:
        st.error(f"Erreur GPX : {e}")
        return None, []


def parse_fit(file, tz_name=TZ_NAME_DEFAULT):
    try:
        file.seek(0)
        fit = FitFile(file)
        fit.parse()
        records, times_pts, hr_records = [], [], []
        start_global = elapsed_global = None
        for msg in fit.get_messages("session"):
            vals = {d.name: d.value for d in msg}
            if isinstance(vals.get("start_time"), datetime):
                start_global = vals["start_time"].replace(tzinfo=None)
            if isinstance(vals.get("total_elapsed_time"), (int, float)):
                elapsed_global = float(vals["total_elapsed_time"])
        for msg in fit.get_messages("record"):
            vals = {d.name: d.value for d in msg}
            lat_r = vals.get("position_lat")
            lon_r = vals.get("position_long")
            if lat_r is None or lon_r is None:
                continue
            lat = lat_r * (180 / 2**31)
            lon = lon_r * (180 / 2**31)
            ts = vals.get("timestamp")
            dt = ts.replace(tzinfo=None) if isinstance(ts, datetime) else None
            alt = (vals.get("enhanced_altitude") or vals.get("altitude") or
                   vals.get("baro_altitude") or vals.get("gps_altitude") or 0.0)
            dist = float(vals.get("distance") or 0.0)
            hr = vals.get("heart_rate")
            hr_records.append(int(hr) if hr is not None else None)
            records.append((lat, lon, float(alt), dist))
            times_pts.append(dt)
        if not records:
            return None
        df = pd.DataFrame(records, columns=["lat", "lon", "elev", "dist"])
        valid_t = [t for t in times_pts if t is not None]
        if len(valid_t) >= 2:
            start_dt, end_dt = min(valid_t), max(valid_t)
        elif start_global and elapsed_global:
            start_dt = start_global
            end_dt = start_global + timedelta(seconds=elapsed_global)
        else:
            start_dt = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=1)
            end_dt = start_dt + timedelta(minutes=5)
        avgT, avgW, avgH = get_avg_weather(records[0][0], records[0][1], start_dt, end_dt, tz_name)
        elev_arr = df["elev"].values
        dup = float(np.sum(np.clip(np.diff(elev_arr), 0, None))) if elev_arr.size >= 2 else 0.0
        ddn = float(-np.sum(np.clip(np.diff(elev_arr), None, 0))) if elev_arr.size >= 2 else 0.0
        return {
            "points": [{"lat": r[0], "lon": r[1], "elev": r[2], "dist": r[3], "time": t}
                       for r, t in zip(records, times_pts)],
            "distance": float(df["dist"].max()),
            "D_up": dup, "D_down": ddn,
            "duration_hms": seconds_to_hms((end_dt - start_dt).total_seconds()),
            "avg_temp": avgT, "avg_wind": avgW, "avg_humidity": avgH,
            "hr_analysis": analyze_hr(hr_records),
        }
    except Exception as e:
        st.error(f"Erreur FIT : {e}")
        return None


def parse_tcx(file, tz_name=TZ_NAME_DEFAULT):
    try:
        file.seek(0)
        root = ET.parse(file).getroot()
    except Exception:
        return None
    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    pts, times, elevs = [], [], []
    for tp in root.findall(".//tcx:Trackpoint", ns):
        lat = tp.find("tcx:Position/tcx:LatitudeDegrees", ns)
        lon = tp.find("tcx:Position/tcx:LongitudeDegrees", ns)
        if lat is None or lon is None:
            continue
        ele = tp.find("tcx:AltitudeMeters", ns)
        tim = tp.find("tcx:Time", ns)
        elev = float(ele.text) if ele is not None else 0.0
        try:
            t = datetime.fromisoformat(tim.text.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            t = None
        pts.append(SimplePoint(float(lat.text), float(lon.text), elev, t))
        times.append(t)
        elevs.append(elev)
    if len(pts) < 2:
        return None
    vt = [t for t in times if t is not None]
    start_dt = vt[0] if vt else datetime.now() - timedelta(days=1)
    end_dt = vt[-1] if len(vt) > 1 else start_dt + timedelta(minutes=5)
    avgT, avgW, avgH = get_avg_weather(pts[0].latitude, pts[0].longitude, start_dt, end_dt, tz_name)
    total = sum(pts[i].distance_3d(pts[i-1]) for i in range(1, len(pts)))
    dup, ddn = compute_dplus_dminus(elevs)
    return {
        "points": pts, "distance": round(total),
        "D_up": round(dup, 1), "D_down": round(ddn, 1),
        "duration_hms": seconds_to_hms((end_dt - start_dt).total_seconds()),
        "avg_temp": avgT, "avg_wind": avgW, "avg_humidity": avgH, "hr_analysis": None
    }


def extract_segment(points, start_td, end_td):
    def get_t(p):
        return p.get("time") if isinstance(p, dict) else getattr(p, "time", None)
    ts = [get_t(p) for p in points if get_t(p) is not None]
    if len(ts) < 2:
        return points
    t0 = min(ts)
    seg = [p for p in points if get_t(p) is not None
           and t0 + start_td <= get_t(p) <= t0 + end_td + timedelta(seconds=1)]
    return seg if len(seg) >= 2 else points


# ══════════════════════════════════════════════════════════════
# MODÈLE LOG-LOG (Riegel)
# ══════════════════════════════════════════════════════════════

def fit_loglog(refs):
    X, Y = [], []
    for r in refs:
        d_m = safe_float(r.get("distance", 0))
        t = r.get("temps")
        secs = float(t) if isinstance(t, (int, float, np.number)) else hms_to_seconds(str(t))
        if d_m <= 0 or secs <= 0:
            continue
        X.append(math.log(d_m / 1000.0))
        Y.append(math.log(secs))
    if len(X) >= 2:
        K, loga = np.polyfit(X, Y, 1)
        K = float(max(0.85, min(1.25, K)))
        a = math.exp(float(loga))
        return (a if 0 < a < 1e7 else 240.0), K
    elif len(X) == 1:
        return math.exp(Y[0]) / (math.exp(X[0])), 1.0
    return 240.0, 1.0


def predict_flat(dist_m, a, K):
    return float(a) * ((dist_m / 1000.0) ** float(K))


def crossval_loo(refs_prepared):
    """Leave-One-Out : pour chaque ref, prédit avec les autres."""
    n = len(refs_prepared)
    if n < 3:
        return None
    rows = []
    for i in range(n):
        train = [r for j, r in enumerate(refs_prepared) if j != i]
        test = refs_prepared[i]
        a_cv, K_cv = fit_loglog(train)
        pred_s = predict_flat(test["distance"], a_cv, K_cv)
        actual_s = float(test["temps"])
        rows.append({
            "Réf": i+1,
            "Distance (km)": round(test["distance"]/1000.0, 2),
            "Temps réel": seconds_to_hms(actual_s),
            "Temps prédit": seconds_to_hms(pred_s),
            "Erreur (s)": round(pred_s - actual_s, 0),
            "Erreur (%)": round((pred_s - actual_s) / actual_s * 100.0, 2) if actual_s > 0 else 0,
        })
    df_cv = pd.DataFrame(rows)
    mae = float(np.mean(np.abs(df_cv["Erreur (s)"].values)))
    mape = float(np.mean(np.abs(df_cv["Erreur (%)"].values)))
    return df_cv, mae, mape


# ══════════════════════════════════════════════════════════════
# ★ RECALIBRATION VERS CONDITIONS IDÉALES ★
# (Section centrale — expliquée en détail dans l'UI)
# ══════════════════════════════════════════════════════════════

def elev_factor_global(D_up_m, D_down_m, dist_m,
                        k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down):
    dist = max(1e-6, float(dist_m))
    g_up = float(D_up_m) / dist
    g_dn = float(D_down_m) / dist
    g0u = max(1e-6, float(g0_up) / 100.0)
    g0d = max(1e-6, float(g0_down) / 100.0)
    up_term   = float(k_up) * math.tanh(g_up / g0u) * g0u
    down_bonus= min(float(k_down) * math.tanh(g_dn / g0d) * g0d, abs(float(down_cap)))
    mult = 1.0 + up_term - down_bonus
    mult = min(mult, 1.0 + float(max_up))
    mult = max(mult, 1.0 + float(max_down))
    return max(0.01, float(mult))


def recalibrate_ref_to_ideal(
    ref: dict,
    # Température idéale cible
    opt_temp: float,
    use_wbgt: bool,
    cold_quad: float, hot_quad: float, temp_max_penalty: float,
    # Pente
    k_up: float, k_down: float, down_cap: float,
    g0_up: float, g0_down: float, max_up: float, max_down: float,
    # Atténuation (damping) de la correction
    elev_ref_power: float,   # 0 = pas de correction pente, 1 = correction totale
    temp_ref_power: float,   # 0 = pas de correction temp,  1 = correction totale
) -> float:
    """
    Convertit le temps d'une référence vers un temps "conditions parfaites" :
    plat + température optimale + humidité neutre.

    Formule :
      temps_idéal = temps_brut / (facteur_pente ^ elev_ref_power)
                                / (facteur_temp  ^ temp_ref_power)

    elev_ref_power et temp_ref_power < 1 car les courses réelles
    ne se courent jamais exactement comme le modèle le prédit
    (terrain technique, signalisation, profil non uniforme...).
    """
    secs = hms_to_seconds(ref.get("temps")) if ref.get("temps") is not None else 0
    D_up  = safe_float(ref.get("D_up", 0.0))
    D_down= safe_float(ref.get("D_down", 0.0))
    dist  = max(1.0, safe_float(ref.get("distance", 1000.0)))

    # 1) Correction pente
    f_elev = elev_factor_global(D_up, D_down, dist, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down)
    secs_no_elev = secs / (f_elev ** float(elev_ref_power))

    # 2) Correction température (via WBGT si activé)
    temp_real = ref.get("avg_temp")
    hum_real  = safe_float(ref.get("avg_humidity", 50.0), 50.0)
    if temp_real is not None:
        temp_eff = effective_temp(temp_real, hum_real, use_wbgt)
        f_temp = temp_multiplier(temp_eff, opt_temp, cold_quad, hot_quad, temp_max_penalty)
        secs_no_temp = secs_no_elev / (max(0.01, f_temp) ** float(temp_ref_power))
    else:
        secs_no_temp = secs_no_elev

    return max(0.0, float(secs_no_temp))


def prepare_refs(refs_input, use_recalibrated, opt_temp, use_wbgt,
                 cold_quad, hot_quad, temp_max_penalty,
                 k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down,
                 elev_ref_power, temp_ref_power):
    out = []
    for r in refs_input:
        d = safe_float(r.get("distance", 0.0))
        raw_t = r.get("duration_hms_file") or r.get("temps", "0:00:00")
        if use_recalibrated:
            secs = recalibrate_ref_to_ideal(
                ref={**r, "temps": raw_t},
                opt_temp=opt_temp, use_wbgt=use_wbgt,
                cold_quad=cold_quad, hot_quad=hot_quad, temp_max_penalty=temp_max_penalty,
                k_up=k_up, k_down=k_down, down_cap=down_cap,
                g0_up=g0_up, g0_down=g0_down, max_up=max_up, max_down=max_down,
                elev_ref_power=elev_ref_power, temp_ref_power=temp_ref_power,
            )
        else:
            secs = float(hms_to_seconds(raw_t))
        out.append({"distance": float(d), "temps": float(secs)})
    return out


# ══════════════════════════════════════════════════════════════
# PACING ULTRA
# ══════════════════════════════════════════════════════════════

def apply_ultra_pacing(t_raw, d_end_m, seg_len_m, total_corr_m, amp_pct):
    if len(t_raw) == 0 or amp_pct <= 0:
        return t_raw
    total_corr_m = max(1e-9, float(total_corr_m))
    d_mid = np.asarray(d_end_m) - 0.5 * np.asarray(seg_len_m)
    prog = np.clip(d_mid / total_corr_m, 0.0, 1.0)
    A = amp_pct / 100.0
    mult = 1.0 + A * (2.0 * prog - 1.0)
    t_adj = np.asarray(t_raw) * mult
    s_raw = np.sum(t_raw)
    s_adj = np.sum(t_adj)
    if s_raw > 0 and s_adj > 0:
        t_adj *= s_raw / s_adj
    return t_adj


# ══════════════════════════════════════════════════════════════
# PRÉDICTION PRINCIPALE
# ══════════════════════════════════════════════════════════════

def run_prediction(
    distance_cible_km, refs_input, points, date_course, heure_course,
    # recalibration
    use_recalibrated, opt_temp, use_wbgt,
    cold_quad, hot_quad, temp_max_penalty, temp_power,
    elev_ref_power, temp_ref_power,
    # pente
    apply_grade, use_minetti, minetti_weight,
    k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down,
    elev_smooth_window, grade_power,
    # altitude physio
    apply_altitude, altitude_ref_m,
    # vent
    apply_wind, wind_mode, wind_smooth_km,
    drag_coeff, tail_credit, wind_cap_head, wind_cap_tail, wind_power,
    wind_gate_g1, wind_gate_g2, wind_gate_min,
    # cap cumul
    base_cap, extra_per_pct, max_cap,
    # fatigue
    apply_fatigue, fatigue_rate, fatigue_mode,
    # ultra pacing
    apply_ultra, ultra_amp,
    # objectif
    objective_hms,
    # affichage
    show_smooth_pace, smooth_window_km,
    # DEM
    dem_elevations,
    tz_name=TZ_NAME_DEFAULT,
):
    if not points or len(points) < 2:
        raise ValueError("GPX invalide ou trop court.")

    # ── altitudes ──
    if dem_elevations is not None and len(dem_elevations) == len(points):
        elev_arr = np.array([e if e is not None else 0.0 for e in dem_elevations], dtype=float)
    else:
        elev_arr = np.array([getattr(p, "elevation", 0.0) or 0.0 for p in points], dtype=float)

    # ── distances cumulées ──
    total_m = 0.0
    cum = [0.0]
    for i in range(1, len(points)):
        total_m += haversine_m(points[i-1].latitude, points[i-1].longitude,
                               points[i].latitude, points[i].longitude)
        cum.append(total_m)
    dist_gpx_km = total_m / 1000.0
    if not distance_cible_km:
        distance_cible_km = dist_gpx_km
    fac = distance_cible_km / max(dist_gpx_km, 1e-9)
    total_corr = total_m * fac
    dists_corr = np.array(cum, dtype=float) * fac

    # ── interpolation altitude ──
    if elev_arr.size != dists_corr.size:
        xs = np.linspace(0, total_m, elev_arr.size)
        elev_arr = np.interp(np.linspace(0, total_m, dists_corr.size), xs, elev_arr)

    # ── lissage altitude ──
    w = int(elev_smooth_window)
    if w % 2 == 0: w += 1
    if w >= 3 and elev_arr.size >= w:
        elev_s = np.convolve(elev_arr, np.ones(w)/w, mode="same")
    else:
        elev_s = elev_arr

    diffs_el = np.diff(elev_s)
    d_plus_total = float(np.sum(np.clip(diffs_el, 0, None)))
    avg_alt = float(np.mean(elev_s))

    # ── fit log-log ──
    refs_fit = prepare_refs(
        refs_input=refs_input, use_recalibrated=use_recalibrated,
        opt_temp=opt_temp, use_wbgt=use_wbgt,
        cold_quad=cold_quad, hot_quad=hot_quad, temp_max_penalty=temp_max_penalty,
        k_up=k_up, k_down=k_down, down_cap=down_cap,
        g0_up=g0_up, g0_down=g0_down, max_up=max_up, max_down=max_down,
        elev_ref_power=elev_ref_power, temp_ref_power=temp_ref_power,
    )
    a, K = fit_loglog(refs_fit)
    if objective_hms:
        obj_s = hms_to_seconds(objective_hms)
        d_km = distance_cible_km
        a = obj_s / (d_km ** K) if d_km > 0 else a
    base_total_s = predict_flat(int(distance_cible_km * 1000), a, K)
    base_s_per_km = base_total_s / max(distance_cible_km, 1e-9)

    # ── altitude physio ──
    alt_mult = altitude_vo2_multiplier(avg_alt, altitude_ref_m) if apply_altitude else 1.0

    # ── segments ──
    km_marks = [i * 1000 for i in range(1, int(total_corr // 1000) + 1)]
    last = total_corr - int(total_corr // 1000) * 1000
    if last > 1e-6:
        km_marks.append(total_corr)

    lats_arr = np.array([p.latitude for p in points], dtype=float)
    lons_arr = np.array([p.longitude for p in points], dtype=float)
    dt_dep = datetime.combine(date_course, heure_course)

    pre = []
    cum_t = 0.0
    cum_dp = 0.0
    cum_dist = 0.0

    for i, d in enumerate(km_marks):
        seg_len = 1000.0
        if i == len(km_marks) - 1 and last > 1e-6:
            seg_len = d - (km_marks[-2] if len(km_marks) >= 2 else 0)
        e_cur = float(np.interp(d, dists_corr, elev_s))
        e_prv = float(np.interp(max(d - seg_len, 0), dists_corr, elev_s)) if i > 0 else e_cur
        grade = (e_cur - e_prv) / max(1e-6, seg_len) * 100.0
        seg_dp = max(0.0, e_cur - e_prv)
        cum_dp   += seg_dp
        cum_dist += seg_len

        t_flat = base_s_per_km * (seg_len / 1000.0)

        # pente
        if apply_grade:
            gm = combined_grade_multiplier(grade, use_minetti, minetti_weight,
                                           k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down)
            t1 = t_flat * (gm ** grade_power)
        else:
            gm = 1.0
            t1 = t_flat

        # altitude physio
        t2 = t1 * alt_mult

        # fatigue
        if apply_fatigue and fatigue_rate > 0:
            fm = fatigue_multiplier(cum_dp, cum_dist, d_plus_total, total_corr, fatigue_rate, fatigue_mode)
        else:
            fm = 1.0
        t3 = t2 * fm

        # météo (passage milieu segment)
        passage_dt = dt_dep + timedelta(seconds=cum_t + t3/2.0)
        lat_s = float(np.interp(d, dists_corr, lats_arr))
        lon_s = float(np.interp(d, dists_corr, lons_arr))
        lat0  = float(np.interp(max(d-seg_len,0), dists_corr, lats_arr))
        lon0  = float(np.interp(max(d-seg_len,0), dists_corr, lons_arr))
        cap   = bearing_deg(lat0, lon0, lat_s, lon_s)

        meteo = get_weather_minutely(lat_s, lon_s, passage_dt, tz_name)
        temp_raw  = meteo["temp"] if meteo else None
        wind_raw  = meteo["wind"] if meteo else None
        hum_raw   = meteo["humidity"] if meteo else None
        wdir_raw  = meteo.get("wind_dir") if meteo else None

        temp_eff_val = None
        if temp_raw is not None and hum_raw is not None:
            temp_eff_val = effective_temp(temp_raw, hum_raw, use_wbgt)

        if temp_eff_val is not None:
            tm = temp_multiplier(temp_eff_val, opt_temp, cold_quad, hot_quad, temp_max_penalty)
            t4 = t3 * (tm ** temp_power)
        else:
            tm = 1.0
            t4 = t3

        pace_local = (t4 / seg_len) * 1000.0 if seg_len > 0 else t4
        head, tail = wind_components(wind_raw, wdir_raw, cap)

        pre.append({
            "idx": i, "d": d, "seg_len": seg_len, "grade": grade, "grade_mult": gm,
            "seg_dp": seg_dp, "cum_dp": cum_dp, "fat_mult": fm, "alt_mult": alt_mult,
            "temp_raw": temp_raw, "temp_eff": temp_eff_val, "hum": hum_raw,
            "wind": wind_raw, "wdir": wdir_raw, "cap": cap,
            "head": head, "tail": tail, "temp_mult": tm,
            "t_flat": t_flat, "t_no_wind": t4, "pace_no_wind": pace_local,
        })
        cum_t += t4

    df_pre = pd.DataFrame(pre)

    # ── vent ──
    if apply_wind and not df_pre.empty:
        if wind_mode == "Global":
            hg = float(np.median(df_pre["head"]))
            tg = float(np.median(df_pre["tail"]))
            pg = float(np.median(df_pre["pace_no_wind"]))
            wm_raw = wind_multiplier(hg, tg, pg, drag_coeff, tail_credit, wind_cap_head, wind_cap_tail)
            df_pre["wind_mult_raw"] = wm_raw
        else:
            w_s = int(max(1, wind_smooth_km)); w_s += (1 if w_s%2==0 else 0)
            hs = pd.Series(df_pre["head"]).rolling(w_s, center=True, min_periods=1).median()
            ts_ = pd.Series(df_pre["tail"]).rolling(w_s, center=True, min_periods=1).median()
            wms = [wind_multiplier(h, t, p, drag_coeff, tail_credit, wind_cap_head, wind_cap_tail)
                   for h, t, p in zip(hs, ts_, df_pre["pace_no_wind"])]
            df_pre["wind_mult_raw"] = wms
            df_pre["head_s"] = hs.values
            df_pre["tail_s"] = ts_.values
    else:
        df_pre["wind_mult_raw"] = 1.0

    # ── wind gate + cap ──
    t_raw = []
    wm_adj_list = []
    for _, row in df_pre.iterrows():
        wm = float(row["wind_mult_raw"])
        g  = float(row["grade"])
        gate = wind_gate(g, wind_gate_g1, wind_gate_g2, wind_gate_min)
        wm_gated = 1.0 + gate * (wm - 1.0)
        t_w = float(row["t_no_wind"]) * (wm_gated ** wind_power)
        mt = t_w / max(1e-9, float(row["t_flat"]))
        mt = cap_combined(mt, g, base_cap, extra_per_pct, max_cap)
        t_raw.append(float(row["t_flat"]) * mt)
        wm_adj_list.append(wm_gated)
    df_pre["wind_mult_adj"] = wm_adj_list
    t_raw = np.array(t_raw, dtype=float)

    # ── ultra pacing ──
    if apply_ultra and ultra_amp > 0:
        t_raw = apply_ultra_pacing(t_raw, df_pre["d"].values, df_pre["seg_len"].values, total_corr, ultra_amp)

    # ── objectif scale ──
    if objective_hms:
        s_obj = hms_to_seconds(objective_hms)
        s_sum = float(np.sum(t_raw))
        t_raw = t_raw * (s_obj / s_sum) if s_sum > 0 else t_raw

    # ── table résultats ──
    rows = []
    cum_t2 = 0.0
    for i in range(len(df_pre)):
        seg = df_pre.iloc[i]
        ts = float(t_raw[i])
        cum_t2 += ts
        pace_val = (ts / float(seg["seg_len"])) * 1000.0 if seg["seg_len"] > 0 else ts
        rows.append({
            "Km": (int(seg["idx"])+1) if seg["seg_len"] >= 999 else f"{int(seg['idx'])+1} ({seg['seg_len']:.0f}m)",
            "Pente (%)": round(float(seg["grade"]), 2),
            "Mult Pente": round(float(seg["grade_mult"]), 4),
            "D+ seg (m)": round(float(seg["seg_dp"]), 1),
            "D+ cum (m)": round(float(seg["cum_dp"]), 1),
            "Mult Fatigue": round(float(seg["fat_mult"]), 4),
            "Mult Altitude": round(float(seg["alt_mult"]), 4),
            "Temp GPS (°C)": round(float(seg["temp_raw"]), 1) if seg["temp_raw"] is not None else None,
            "Temp eff/WBGT (°C)": round(float(seg["temp_eff"]), 1) if seg["temp_eff"] is not None else None,
            "Mult Temp": round(float(seg["temp_mult"]), 4),
            "Vent (m/s)": round(float(seg["wind"]), 1) if seg["wind"] is not None else None,
            "Headwind (m/s)": round(float(seg.get("head_s", seg["head"])), 2),
            "Tailwind (m/s)": round(float(seg.get("tail_s", seg["tail"])), 2),
            "Mult Vent": round(float(seg["wind_mult_adj"]), 4),
            "Humidité (%)": round(float(seg["hum"]), 1) if seg["hum"] is not None else None,
            "Temps seg (s)": round(ts, 1),
            "Allure (min/km)": pace_str(pace_val),
            "Temps cumulé": seconds_to_hms(cum_t2),
        })

    df_out = pd.DataFrame(rows)

    if show_smooth_pace and not df_out.empty:
        w_p = int(max(1, smooth_window_km)); w_p += (1 if w_p%2==0 else 0)
        s_p = pd.Series(df_out["Temps seg (s)"].astype(float)).rolling(w_p, center=True, min_periods=1).median()
        df_out["Allure lissée (min/km)"] = s_p.apply(pace_str)

    total_s = float(np.sum(t_raw))
    # Intervalle de confiance ±5% (simpliste — améliorable avec résidus cross-val)
    ci_low  = total_s * 0.95
    ci_high = total_s * 1.05

    return {
        "df": df_out, "total_s": total_s, "total_human": seconds_to_hms(total_s),
        "ci_low": seconds_to_hms(ci_low), "ci_high": seconds_to_hms(ci_high),
        "dist_gpx_km": dist_gpx_km, "K": K, "avg_alt": avg_alt, "d_plus_total": d_plus_total,
        "refs_fit": refs_fit, "pre_df": df_pre,
    }


# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# UI PRINCIPALE
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════

st.title("🏃 Prédiction de course — Coach & Athlète")
st.caption("v3 — WBGT · Minetti · DEM · Recalibration des références · Interface pédagogique")

# Mode Simple / Expert
col_mode1, col_mode2 = st.columns([2, 3])
with col_mode1:
    mode = st.radio("Mode d'interface", ["🟢 Simple (recommandé)", "🔵 Expert (tous les curseurs)"],
                    horizontal=True)
EXPERT = "Expert" in mode

# ─────────────────────────────────────────────────────────────
# SECTION 1 — GPX
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("1️⃣  Parcours GPX")

gpx_file = st.file_uploader("📂 Importer le fichier GPX de la course cible", type=["gpx"])
points = None
dem_elevations = None

if gpx_file:
    _gpx, points = parse_gpx_points(gpx_file)
    if points:
        tot_tmp = sum(haversine_m(points[i-1].latitude, points[i-1].longitude,
                                   points[i].latitude, points[i].longitude)
                      for i in range(1, len(points)))
        dup_tmp, ddn_tmp = compute_dplus_dminus([getattr(p, "elevation", 0.0) or 0.0 for p in points])
        avg_alt_tmp = np.mean([getattr(p, "elevation", 0.0) or 0.0 for p in points])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Distance GPX", f"{tot_tmp/1000:.2f} km")
        c2.metric("D+ GPS", f"{dup_tmp:.0f} m")
        c3.metric("D- GPS", f"{ddn_tmp:.0f} m")
        c4.metric("Alt. moy.", f"{avg_alt_tmp:.0f} m")

with st.expander("🏔️ Correction altimétrique DEM (optionnel — recommandé en montagne)"):
    st.info(
        "Le GPS vertical a une précision de ±5-15 m, ce qui peut *inventer* 200-400 m de D+ sur un marathon. "
        "Le DEM (modèle numérique de terrain) donne l'altitude réelle à ±1 m."
    )
    use_dem = st.checkbox("Activer la correction DEM", value=False)
    dem_dataset = "srtm30m"
    if use_dem:
        dem_dataset = st.selectbox("Dataset", ["srtm30m (global, 30m)", "eudem25m (Europe, 25m — plus précis)", "mapzen (global fusion)"],
                                    index=0).split()[0]
        if gpx_file and points and st.button("🔄 Télécharger et corriger l'altitude"):
            with st.spinner("Correction DEM en cours..."):
                dem_elevations = list(correct_elevations_dem(points, max_points=100, dataset=dem_dataset))
                st.session_state["dem_elevations"] = dem_elevations
                dup_dem, ddn_dem = compute_dplus_dminus([e or 0.0 for e in dem_elevations])
                st.success(f"DEM OK — D+ DEM: **{dup_dem:.0f} m** (vs GPS: {dup_tmp:.0f} m) | D- DEM: **{ddn_dem:.0f} m**")
    if "dem_elevations" in st.session_state:
        dem_elevations = st.session_state["dem_elevations"]


# ─────────────────────────────────────────────────────────────
# SECTION 2 — RÉFÉRENCES
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("2️⃣  Courses de référence")
st.info(
    "Les références servent à **calibrer le modèle** sur l'athlète. "
    "Plus les distances sont variées (5 km, semi, marathon...), meilleure est la précision. "
    "Minimum conseillé : **3 références** sur les 12 derniers mois."
)

if "n_refs" not in st.session_state:
    st.session_state.n_refs = 3
cc1, cc2 = st.columns(2)
with cc1:
    if st.button("➕ Ajouter une référence") and st.session_state.n_refs < 6:
        st.session_state.n_refs += 1
with cc2:
    if st.button("➖ Retirer") and st.session_state.n_refs > 1:
        st.session_state.n_refs -= 1

refs_raw = []
for i in range(1, st.session_state.n_refs + 1):
    with st.expander(f"📌 Référence {i}", expanded=(i <= 2)):
        use_file = st.checkbox(f"Importer depuis fichier FIT/TCX (Garmin, Polar...)", key=f"use_file_{i}")
        c1, c2, c3, c4 = st.columns(4)
        dist = c1.number_input("Distance (m)", value=float(st.session_state.get(f"dist_{i}", 5000*i)), key=f"dist_{i}")
        temps = c2.text_input("Temps (h:mm:ss)", value=str(st.session_state.get(f"temps_{i}", "0:40:00")), key=f"temps_{i}")
        dup  = c3.number_input("D+ (m)", value=float(st.session_state.get(f"dup_{i}", 0.0)), key=f"dup_{i}")
        ddn  = c4.number_input("D- (m)", value=float(st.session_state.get(f"ddn_{i}", 0.0)), key=f"ddn_{i}")
        file_in = st.file_uploader(f"Fichier FIT/TCX", type=["fit", "tcx"], key=f"fileref_{i}") if use_file else None

        dur_hms_file = avg_temp_ref = avg_wind_ref = avg_hum_ref = hr_ref = None
        fname = file_in.name.lower() if file_in else ""
        fit_data = tcx_data = None

        if file_in:
            if fname.endswith(".fit"):
                fit_data = parse_fit(file_in)
                if fit_data:
                    dist, dup, ddn = fit_data["distance"], fit_data["D_up"], fit_data["D_down"]
                    dur_hms_file = fit_data["duration_hms"]
                    avg_temp_ref, avg_wind_ref, avg_hum_ref = fit_data["avg_temp"], fit_data["avg_wind"], fit_data["avg_humidity"]
                    hr_ref = fit_data.get("hr_analysis")
            elif fname.endswith(".tcx"):
                tcx_data = parse_tcx(file_in)
                if tcx_data:
                    dist, dup, ddn = tcx_data["distance"], tcx_data["D_up"], tcx_data["D_down"]
                    dur_hms_file = tcx_data["duration_hms"]
                    avg_temp_ref, avg_wind_ref, avg_hum_ref = tcx_data["avg_temp"], tcx_data["avg_wind"], tcx_data["avg_humidity"]

            # Segment optionnel
            cs, ce = st.columns(2)
            sh = cs.text_input("Début segment (hh:mm:ss)", "00:00:00", key=f"start_{i}")
            eh = ce.text_input("Fin segment (hh:mm:ss)",  "23:59:59", key=f"end_{i}")
            start_td, end_td = hms_to_timedelta(sh), hms_to_timedelta(eh)
            if start_td.total_seconds() > 0 or end_td.total_seconds() < 86399:
                pts_src = None
                if fit_data and "points" in fit_data: pts_src = fit_data["points"]
                elif tcx_data and "points" in tcx_data: pts_src = tcx_data["points"]
                if pts_src:
                    seg = extract_segment(pts_src, start_td, end_td)
                    seg_dist = 0.0; seg_elevs = []; seg_times = []
                    for j in range(1, len(seg)):
                        p1, p2 = seg[j-1], seg[j]
                        la1, lo1 = (p1["lat"], p1["lon"]) if isinstance(p1, dict) else (p1.latitude, p1.longitude)
                        la2, lo2 = (p2["lat"], p2["lon"]) if isinstance(p2, dict) else (p2.latitude, p2.longitude)
                        e2 = p2.get("elev", 0) if isinstance(p2, dict) else p2.elevation
                        t2 = p2.get("time") if isinstance(p2, dict) else p2.time
                        seg_dist += haversine_m(la1, lo1, la2, lo2)
                        seg_elevs.append(e2)
                        if t2: seg_times.append(t2)
                    dup, ddn = compute_dplus_dminus(seg_elevs)
                    if len(seg_times) >= 2:
                        dur_hms_file = seconds_to_hms((seg_times[-1]-seg_times[0]).total_seconds())
                    dist = round(seg_dist)
        else:
            if EXPERT:
                cs2, ce2 = st.columns(2)
                avg_temp_ref = cs2.number_input(f"Temp moy. course (°C)", value=15.0, key=f"avgT_{i}")
                avg_hum_ref  = ce2.number_input(f"Humidité moy. (%)", value=60.0, key=f"avgH_{i}")
            else:
                avg_temp_ref = None
                avg_hum_ref = None

        temps_eff = dur_hms_file if dur_hms_file else temps
        secs_brut = hms_to_seconds(temps_eff)
        dist_km = safe_float(dist, 1.0) / 1000.0
        if secs_brut > 0 and dist_km > 0:
            pace_val = pace_str(secs_brut / dist_km)
            st.caption(f"📍 {dist:.0f} m · {temps_eff} · **{pace_val}/km**"
                       + (f" · D+ {dup:.0f}m" if dup > 0 else "")
                       + (f" · Temp GPS: {avg_temp_ref:.0f}°C" if avg_temp_ref else "")
                       + (f" · FC fiabilité: {hr_ref.get('reliability')}" if hr_ref else ""))
        if hr_ref and hr_ref.get("hr_max"):
            st.caption(f"💓 FC max {hr_ref['hr_max']} bpm · dérive {hr_ref['hr_drift']} bpm · seuil estimé ~{hr_ref['hr_threshold_est']} bpm")

        refs_raw.append({
            "distance": float(dist), "temps": str(temps_eff),
            "D_up": float(dup), "D_down": float(ddn),
            "duration_hms_file": dur_hms_file,
            "avg_temp": avg_temp_ref, "avg_humidity": avg_hum_ref, "avg_wind": avg_wind_ref,
            "hr_analysis": hr_ref,
        })


# ─────────────────────────────────────────────────────────────
# ★ SECTION 3 — RECALIBRATION ★ (section bien visible)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("3️⃣  Recalibration des références vers les conditions idéales")

st.markdown("""
<div class="highlight-box">
<strong>Pourquoi recalibrer ?</strong><br>
Une course réalisée par 30°C et 80% d'humidité vaut <em>physiologiquement mieux</em>
qu'un temps identique par 12°C et temps sec. Sans correction, le modèle croit que
l'athlète est plus lent qu'il ne l'est vraiment.<br><br>
La recalibration <em>restitue</em> chaque référence à ce qu'aurait été le résultat
dans des conditions parfaites (plat, température optimale, humidité neutre),
avant de construire le modèle de performance.
</div>
""", unsafe_allow_html=True)

use_recalibrated = st.checkbox(
    "✅ Recalibrer les références vers les conditions idéales (fortement recommandé)",
    value=True
)

if use_recalibrated:
    st.success("Les références seront normalisées avant le fit : une référence faite par forte chaleur "
               "sera revue à la hausse (l'athlète aurait couru plus vite dans de meilleures conditions).")
else:
    st.warning("Les références brutes sont utilisées. Le modèle peut sous-estimer la performance "
               "si certaines références ont été faites dans des conditions difficiles.")

with st.expander("⚙️ Paramètres de recalibration (cliquer pour voir)"):
    st.markdown("**Température de référence « idéale »**")
    opt_temp = st.slider(
        "Température optimale de course (°C)", 5.0, 20.0, 12.0, 0.5,
        help="Température à laquelle la performance est à son maximum. "
             "Consensus scientifique : entre 8°C et 15°C selon les athlètes."
    )
    param_help(
        "L'athlète est considéré comme avantagé par des températures plus basses",
        "L'athlète est considéré comme optimal à des températures plus élevées",
        "12°C est une bonne valeur par défaut pour un coureur standard"
    )

    use_wbgt = st.checkbox(
        "Utiliser le WBGT (température ressentie chaleur+humidité) — recommandé",
        value=True,
        help="Le WBGT (Wet Bulb Globe Temperature) combine température et humidité. "
             "À 28°C et 85% d'humidité, le WBGT est ~27°C, soit une pénalité bien supérieure "
             "à 28°C par temps sec."
    )

    col_ep1, col_ep2 = st.columns(2)
    with col_ep1:
        elev_ref_power = st.slider(
            "Force de correction de la pente des références",
            0.0, 1.0, 0.60, 0.05,
            help="0 = on ne corrige pas du tout la pente des références. 1 = correction totale."
        )
        param_help(
            "La correction de pente est plus agressive → les refs en montagne sont plus fortement normalisées",
            "La correction est plus prudente → les refs gardent une partie de leur temps 'réel'",
            "0.5-0.7 recommandé : les courses réelles ont toujours du hors-piste, signalisation, etc."
        )
    with col_ep2:
        temp_ref_power = st.slider(
            "Force de correction de la température des références",
            0.0, 1.0, 0.85, 0.05,
            help="0 = on ignore la météo des références. 1 = correction totale."
        )
        param_help(
            "La correction météo est plus agressive → une ref par 30°C sera fortement améliorée",
            "La correction est plus prudente → la météo a moins d'influence sur la normalisation",
            "0.8-0.9 recommandé pour la température (l'impact météo est bien documenté)"
        )

# Tableau de recalibration
st.subheader("📋 Résumé de la recalibration")

# Pour afficher le tableau, on a besoin des paramètres pente — on utilise des valeurs par défaut si pas encore saisies
# On les récupérera plus tard dans les paramètres avancés. Ici on affiche un aperçu avec valeurs par défaut.
_k_up_prev = st.session_state.get("k_up_val", 12.0)
_k_down_prev = st.session_state.get("k_down_val", 5.0)
_g0u_prev = st.session_state.get("g0_up_val", 3.0)
_g0d_prev = st.session_state.get("g0_down_val", 2.5)

calib_rows = []
for r in refs_raw:
    t_brut = hms_to_seconds(r.get("duration_hms_file") or r.get("temps", ""))
    dist_km = safe_float(r.get("distance", 1.0)) / 1000.0
    avg_t = r.get("avg_temp")
    avg_h = safe_float(r.get("avg_humidity", 50.0), 50.0)
    wbgt_val = wbgt_simplified(avg_t, avg_h) if avg_t is not None and use_wbgt else None
    temp_eff_ref = wbgt_val if wbgt_val is not None else avg_t

    t_ideal = recalibrate_ref_to_ideal(
        ref={**r, "temps": r.get("duration_hms_file") or r.get("temps", "0:00:00")},
        opt_temp=opt_temp, use_wbgt=use_wbgt,
        cold_quad=0.0012, hot_quad=0.0016, temp_max_penalty=0.10,
        k_up=_k_up_prev, k_down=_k_down_prev, down_cap=-0.08,
        g0_up=_g0u_prev, g0_down=_g0d_prev, max_up=0.30, max_down=-0.06,
        elev_ref_power=elev_ref_power, temp_ref_power=temp_ref_power,
    ) if use_recalibrated else float(t_brut)

    gain_s = t_brut - t_ideal
    calib_rows.append({
        "Distance": f"{safe_float(r['distance'])/1000:.1f} km",
        "Temps brut": seconds_to_hms(t_brut),
        "Allure brute": pace_str(t_brut / dist_km) if dist_km > 0 else "-",
        "D+": f"{r['D_up']:.0f} m",
        "Temp GPS": f"{avg_t:.0f}°C" if avg_t is not None else "?",
        "WBGT": f"{wbgt_val:.1f}°C" if wbgt_val is not None else "-",
        "Temps recalibré": seconds_to_hms(t_ideal) if use_recalibrated else "—",
        "Allure recalibrée": pace_str(t_ideal / dist_km) if (use_recalibrated and dist_km > 0) else "—",
        "Gain correction": f"-{seconds_to_hms(gain_s)}" if gain_s > 0 else (f"+{seconds_to_hms(-gain_s)}" if gain_s < 0 else "0"),
    })

st.dataframe(pd.DataFrame(calib_rows), use_container_width=True)
if use_recalibrated:
    st.caption("💡 La colonne 'Gain correction' représente le temps que le modèle estime avoir été *perdu* à cause des conditions (météo + pente). "
               "Ce temps est restitué à l'athlète pour le calcul du niveau de performance.")


# ─────────────────────────────────────────────────────────────
# SECTION 4 — PARAMÈTRES AVANCÉS
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("4️⃣  Paramètres du modèle")

# ─── Température ───
with st.expander("🌡️ Température & Humidité" + (" ✅" if True else ""), expanded=False):
    apply_temp_section = True  # toujours actif
    cold_quad = 0.0012
    hot_quad  = 0.0016
    temp_max_penalty = 0.10
    temp_power = 1.0
    if EXPERT:
        st.markdown("**Modèle de pénalité thermique (quadratique)**")
        col1, col2 = st.columns(2)
        with col1:
            cold_quad = st.number_input("Sensibilité au froid (coeff quad.)", value=0.0012, step=0.0002, format="%.4f")
            param_help("Plus sensible au froid → les courses froides coûtent plus cher",
                       "Moins sensible → le froid a peu d'impact", "0.0012 = valeur standard")
        with col2:
            hot_quad = st.number_input("Sensibilité à la chaleur (coeff quad.)", value=0.0016, step=0.0002, format="%.4f")
            param_help("Plus sensible → une course chaude est plus pénalisée",
                       "Moins sensible → la chaleur a moins d'impact", "0.0016 = légèrement plus que le froid")
        temp_max_penalty = st.slider("Pénalité maximale température (%)", 0.00, 0.20, 0.10, 0.01)
        param_help("Cap à +20% → la chaleur extrême (ex: Sahara) ne peut jamais dépasser +20%",
                   "Cap plus faible → on sous-estime l'impact des conditions très chaudes",
                   "0.10 (10%) est raisonnable pour des conditions jusqu'à 35°C WBGT")
        temp_power = st.slider("Damping température (puissance)", 0.2, 1.2, 1.0, 0.05)
        param_help("Puissance > 1 → effet amplifié (exponentiel)",
                   "Puissance < 1 → effet amorti", "1.0 = linéaire")

    # Démo interactive
    if use_wbgt:
        st.markdown("**Aperçu : impact WBGT sur l'allure**")
        ex_t = st.slider("Température exemple (°C)", -5, 40, 20, 1, key="demo_temp")
        ex_h = st.slider("Humidité exemple (%)", 10, 100, 60, 5, key="demo_hum")
        ex_wbgt = wbgt_simplified(ex_t, ex_h)
        ex_mult = temp_multiplier(ex_wbgt, opt_temp, cold_quad, hot_quad, temp_max_penalty)
        pen_pct = (ex_mult - 1.0) * 100.0
        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("WBGT", f"{ex_wbgt:.1f}°C")
        col_d2.metric("Multiplicateur", f"{ex_mult:.3f}")
        col_d3.metric("Pénalité", f"+{pen_pct:.1f}%" if pen_pct > 0 else f"{pen_pct:.1f}%")


# ─── Altitude physiologique ───
with st.expander("🏔️ Altitude physiologique (hypoxie)"):
    apply_altitude = st.checkbox("Appliquer la pénalité d'altitude (VO2 réduite au-dessus de 1500 m)", value=True)
    altitude_ref_m = 0.0
    if apply_altitude:
        altitude_ref_m = st.number_input(
            "Altitude habituelle d'entraînement de l'athlète (m)",
            value=0.0, step=100.0,
            help="Si l'athlète s'entraîne à 800 m, la pénalité est relative à 800 m. "
                 "Un athlète kényan entraîné à 2400 m sera peu affecté par une course à 1800 m."
        )
        param_help(
            "Augmenter l'altitude de référence → réduit la pénalité (l'athlète est acclimaté)",
            "Altitude 0 → pénalité maximale pour toute course au-dessus de 1500 m",
            "Formule : ~1% de pénalité par 100 m au-dessus de max(1500m, altitude_ref)"
        )
        if points and len(points) > 0:
            avg_alt_gx = np.mean([getattr(p, "elevation", 0.0) or 0.0 for p in points])
            alt_mult_preview = altitude_vo2_multiplier(avg_alt_gx, altitude_ref_m)
            st.caption(f"→ Altitude moy. du parcours : {avg_alt_gx:.0f} m | Multiplicateur prévu : **{alt_mult_preview:.3f}** ({(alt_mult_preview-1)*100:.1f}%)")


# ─── Pente ───
with st.expander("🎢 Modèle de pente"):
    apply_grade = st.checkbox("Prendre en compte la pente", value=True)
    use_minetti = st.checkbox(
        "Modèle Minetti (base physiologique — Minetti et al. 2002)",
        value=True,
        help="Polynôme du 5e degré ajusté sur des mesures réelles de VO2 à différentes pentes. "
             "Plus précis que le modèle heuristique, surtout pour les pentes > 10%."
    )
    minetti_weight = 0.6
    if use_minetti:
        minetti_weight = st.slider(
            "Part de Minetti dans le calcul (vs heuristique)", 0.0, 1.0, 0.6, 0.1,
            help="0 = 100% heuristique (ancien modèle) | 1 = 100% Minetti | 0.6 = mélange recommandé"
        )
        param_help(
            "Plus de Minetti → modèle physiquement fondé, mieux calibré sur pentes extrêmes",
            "Plus d'heuristique → modèle plus 'manuel', tu peux régler finement k_up/k_down",
            "0.6 est un bon compromis terrain réel / base scientifique"
        )

    elev_smooth_window = 11
    grade_power = 0.85
    k_up, k_down, down_cap = 12.0, 5.0, -0.08
    g0_up, g0_down, max_up, max_down = 3.0, 2.5, 0.30, -0.06

    if EXPERT:
        elev_smooth_window = st.slider("Lissage altitude (fenêtre en points GPS)", 1, 51, 11, 2)
        param_help("Lissage fort → moins sensible aux pics GPS parasites (D+ réduit)",
                   "Lissage faible → les vraies variations de pente sont conservées",
                   "11-21 recommandé pour sentiers montagne, 5-11 pour route")
        grade_power = st.slider("Amortissement de l'effet pente (puissance)", 0.2, 1.0, 0.85, 0.05)
        param_help("Puissance proche de 1 → l'effet pente est fort et direct",
                   "Puissance faible → l'effet pente est amorti (utile si le modèle sur-prédit en montagne)")
        c1, c2, c3 = st.columns(3)
        with c1:
            k_up = st.number_input("Sensibilité montée (k_up)", value=12.0, step=0.5)
            param_help("Monte l'impact du D+ sur le temps", "Réduit l'impact du D+",
                       "12 est une valeur typique pour un coureur de montagne")
        with c2:
            k_down = st.number_input("Sensibilité descente (k_down)", value=5.0, step=0.5)
            param_help("Le gain en descente est plus fort", "Le gain en descente est réduit",
                       "La descente raide peut devenir coûteuse (fein musculaire) — Minetti le capture bien")
        with c3:
            down_cap = st.number_input("Cap bonus descente", value=-0.08, step=0.01, format="%.2f")
            param_help("Plafond du gain en descente moins élevé (ex: -0.05 = max -5%)",
                       "Plafond plus élevé → forte descente = grand gain",
                       "-0.08 = max 8% de gain en descente légère")
        # Stocker pour la recalibration
        st.session_state["k_up_val"] = k_up
        st.session_state["k_down_val"] = k_down
        st.session_state["g0_up_val"] = g0_up
        st.session_state["g0_down_val"] = g0_down


# ─── Vent ───
with st.expander("💨 Vent"):
    apply_wind = st.checkbox("Appliquer l'effet du vent", value=True)
    wind_mode = "Lissé"
    wind_smooth_km = 5
    drag_coeff, tail_credit = 0.012, 0.35
    wind_cap_head, wind_cap_tail = 0.10, -0.04
    wind_power = 1.0
    wind_gate_g1, wind_gate_g2, wind_gate_min = 2.0, 8.0, 0.25

    if apply_wind:
        st.info("Le vent de face ralentit davantage qu'un vent de dos n'accélère (asymétrie aérodynamique). "
                "En montée, l'effet du vent est automatiquement réduit (wind gate).")
        if EXPERT:
            wind_mode = st.selectbox("Mode calcul vent", ["Lissé (segment par segment)", "Global (unique)"],
                                      index=0).split()[0]
            wind_smooth_km = st.slider("Lissage vent sur N km", 1, 11, 5, 2)
            param_help("Lissage fort → le vent a un effet régulier (ignorer les rafales)",
                       "Lissage faible → le vent est pris km par km (plus réaliste si direction variable)")
            c1, c2 = st.columns(2)
            drag_coeff  = c1.number_input("Coefficient aérodynamique (drag_coeff)", value=0.012, step=0.002, format="%.3f")
            tail_credit = c2.slider("Crédit vent arrière (fraction bénéfice)", 0.0, 0.8, 0.35, 0.05)
            param_help("Crédit plus élevé → le vent de dos aide davantage",
                       "Crédit faible → le vent de dos aide peu (réaliste, car on ne peut pas 'surfer' le vent)",
                       "0.35 = 35% du bénéfice théorique (valeur de la littérature running)")
            wind_cap_head = st.slider("Pénalité max vent de face (%)", 0.00, 0.20, 0.10, 0.01)
            wind_cap_tail = st.slider("Gain max vent de dos (%)", -0.10, 0.00, -0.04, 0.01)


# ─── Anti-cumul ───
base_cap, extra_per_pct, max_cap = 0.08, 0.004, 0.18
if EXPERT:
    with st.expander("🧱 Plafond anti-accumulation"):
        st.info("Empêche qu'un seul kilomètre accuse plus de X% de ralentissement (évite les résultats absurdes).")
        c1, c2, c3 = st.columns(3)
        base_cap = c1.slider("Plafond de base (%)", 0.02, 0.20, 0.08, 0.01)
        extra_per_pct = c2.slider("Extra par % de pente", 0.000, 0.020, 0.004, 0.001)
        max_cap = c3.slider("Plafond absolu (%)", 0.05, 0.40, 0.18, 0.01)
        param_help("Plafond plus élevé → les km de montagne extrême peuvent être très lents",
                   "Plafond bas → le modèle lisse les extrêmes (plus stable)",
                   "Utile pour les courses avec sections très raides (>20%)")


# ─── Fatigue ───
with st.expander("🔋 Fatigue en course"):
    st.markdown(
        "La fatigue modélise le *ralentissement progressif* de l'athlète au fil des km. "
        "Elle est basée sur le D+ cumulé (montagne) et/ou la distance (plat)."
    )
    apply_fatigue = st.checkbox("Activer la fatigue", value=False)
    fatigue_rate, fatigue_mode = 0.0, "mixte"
    if apply_fatigue:
        fatigue_rate = st.slider(
            "Ralentissement total en fin de course (%)",
            0.0, 30.0, 8.0, 0.5,
            help="Ex: 8% → l'athlète est 8% plus lent sur le dernier km que si la course était plate et fraîche."
        )
        param_help(
            "Ralentissement plus fort → la fin de course est nettement plus lente (typique marathon/ultra)",
            "Ralentissement faible → l'athlète gère bien son allure (athlète expérimenté ou course courte)",
            "8-12% pour un marathon bien géré | 15-25% pour un ultra-trail"
        )
        fatigue_mode = st.selectbox(
            "Type de fatigue",
            ["mixte (recommandé)", "distance (plat)", "d_plus (montagne)"],
            help="mixte : pondère D+ et distance selon la montagnité du parcours"
        ).split()[0]


# ─── Ultra pacing ───
with st.expander("⚡ Stratégie de pacing Ultra"):
    st.markdown("Option avancée : départ plus vite / fin plus lente. Le temps total reste inchangé.")
    apply_ultra = st.checkbox("Activer le pacing ultra (positive split)", value=False)
    ultra_amp = 0.0
    if apply_ultra:
        ultra_amp = st.slider("Amplitude (%)", 0.0, 40.0, 10.0, 0.5,
                               help="10% → au km 1 : -10% sur le temps prédit | au dernier km : +10%")


# ─── Affichage ───
show_smooth_pace = True
smooth_window_km = 3
with st.expander("📉 Options d'affichage"):
    show_smooth_pace = st.checkbox("Afficher l'allure lissée (médiane glissante)", value=True)
    smooth_window_km = st.slider("Fenêtre de lissage (km)", 1, 9, 3, 2) if show_smooth_pace else 3


# ─────────────────────────────────────────────────────────────
# SECTION 5 — COURSE
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("5️⃣  Paramètres de la course cible")

c1, c2 = st.columns(2)
date_course   = c1.date_input("📅 Date de course", value=date.today())
heure_course  = c2.time_input("⏰ Heure de départ", value=time(9, 0))

colf1, colf2 = st.columns(2)
with colf1:
    force_dist = st.checkbox("Forcer la distance (si GPX != distance officielle)", value=False)
    dist_forcee = st.number_input("Distance (km)", value=42.195, format="%.3f") if force_dist else None
with colf2:
    force_temps = st.checkbox("Travailler à partir d'un objectif de temps", value=False)
    temps_objectif = st.text_input("Temps objectif (h:mm:ss)", value="3:30:00") if force_temps else None

st.markdown("---")


# ─────────────────────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────
with st.expander("🔬 Cross-validation (fiabilité du modèle)"):
    st.info(
        "La cross-validation Leave-One-Out teste la précision du modèle : "
        "pour chaque référence, elle l'exclut et prédit son temps avec les autres. "
        "Un MAPE < 3% = excellent | < 7% = correct | > 7% = revoir les références."
    )
    if st.button("Lancer la cross-validation"):
        refs_cv = prepare_refs(
            refs_raw, use_recalibrated, opt_temp, use_wbgt,
            cold_quad, hot_quad, temp_max_penalty,
            k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down,
            elev_ref_power, temp_ref_power
        )
        cv = crossval_loo(refs_cv)
        if cv is None:
            st.warning("Au moins 3 références nécessaires pour la cross-validation.")
        else:
            df_cv, mae, mape = cv
            st.dataframe(df_cv, use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("Erreur absolue moyenne", f"{seconds_to_hms(mae)} ({mae:.0f}s)")
            c2.metric("Erreur relative moyenne (MAPE)", f"{mape:.2f} %")
            if mape < 3:
                st.success("✅ Modèle bien calibré — les références sont cohérentes entre elles.")
            elif mape < 7:
                st.warning("⚠️ Calibration acceptable — ajouter des références améliorera la précision.")
            else:
                st.error("❌ Calibration faible — vérifier les références (format des temps, distances, D+/D-).")


# ─────────────────────────────────────────────────────────────
# CALCUL
# ─────────────────────────────────────────────────────────────
st.header("6️⃣  Calcul & Résultats")

if st.button("▶️ Calculer la prédiction", type="primary"):
    if not gpx_file or points is None:
        st.error("⚠️ Importe un fichier GPX d'abord (section 1).")
    elif not any(safe_float(r.get("distance", 0)) > 0 and hms_to_seconds(r.get("temps", "0")) > 0 for r in refs_raw):
        st.error("⚠️ Renseigne au moins une référence valide (distance + temps).")
    else:
        with st.spinner("Calcul en cours (récupération météo + prédiction)..."):
            try:
                res = run_prediction(
                    distance_cible_km=dist_forcee if force_dist else None,
                    refs_input=refs_raw, points=points,
                    date_course=date_course, heure_course=heure_course,
                    use_recalibrated=use_recalibrated, opt_temp=opt_temp, use_wbgt=use_wbgt,
                    cold_quad=cold_quad, hot_quad=hot_quad, temp_max_penalty=temp_max_penalty, temp_power=temp_power,
                    elev_ref_power=elev_ref_power, temp_ref_power=temp_ref_power,
                    apply_grade=apply_grade, use_minetti=use_minetti, minetti_weight=minetti_weight,
                    k_up=k_up, k_down=k_down, down_cap=down_cap,
                    g0_up=g0_up, g0_down=g0_down, max_up=max_up, max_down=max_down,
                    elev_smooth_window=elev_smooth_window, grade_power=grade_power,
                    apply_altitude=apply_altitude, altitude_ref_m=altitude_ref_m,
                    apply_wind=apply_wind, wind_mode="Lissé", wind_smooth_km=wind_smooth_km,
                    drag_coeff=drag_coeff, tail_credit=tail_credit,
                    wind_cap_head=wind_cap_head, wind_cap_tail=wind_cap_tail, wind_power=wind_power,
                    wind_gate_g1=wind_gate_g1, wind_gate_g2=wind_gate_g2, wind_gate_min=wind_gate_min,
                    base_cap=base_cap, extra_per_pct=extra_per_pct, max_cap=max_cap,
                    apply_fatigue=apply_fatigue, fatigue_rate=fatigue_rate, fatigue_mode=fatigue_mode,
                    apply_ultra=apply_ultra, ultra_amp=ultra_amp,
                    objective_hms=temps_objectif if force_temps else None,
                    show_smooth_pace=show_smooth_pace, smooth_window_km=smooth_window_km,
                    dem_elevations=dem_elevations,
                )
                st.session_state["res"] = res
            except Exception as e:
                import traceback
                st.error(f"Erreur : {e}")
                st.code(traceback.format_exc())

if "res" in st.session_state:
    res = st.session_state["res"]
    st.markdown("---")
    st.subheader("🎯 Prédiction")

    # Métriques principales
    avg_pace_s = res["total_s"] / max(res["dist_gpx_km"], 1e-6)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("⏱ Temps prédit", res["total_human"])
    c2.metric("📊 Allure moy.", pace_str(avg_pace_s) + "/km")
    c3.metric("Fourchette basse (−5%)", res["ci_low"])
    c4.metric("Fourchette haute (+5%)", res["ci_high"])
    c5.metric("K Riegel", f"{res['K']:.3f}")

    st.caption(
        f"Distance GPX : {res['dist_gpx_km']:.3f} km | "
        f"D+ total (lissé) : {res['d_plus_total']:.0f} m | "
        f"Altitude moy. : {res['avg_alt']:.0f} m"
    )

    # Graphiques
    df_out = res["df"]
    if not df_out.empty:
        tab1, tab2, tab3 = st.tabs(["📈 Allure par km", "🔎 Facteurs de ralentissement", "📋 Tableau détaillé"])

        with tab1:
            fig, ax = plt.subplots(figsize=(12, 4))
            pv = []
            for v in df_out["Allure (min/km)"].values:
                try:
                    parts = str(v).split(":")
                    pv.append(int(parts[0]) + int(parts[1])/60.0)
                except Exception:
                    pv.append(float("nan"))
            x = list(range(1, len(pv)+1))
            ax.plot(x, pv, lw=1.5, alpha=0.35, color="steelblue", label="Allure brute")
            if "Allure lissée (min/km)" in df_out.columns:
                ps = []
                for v in df_out["Allure lissée (min/km)"].values:
                    try:
                        parts = str(v).split(":")
                        ps.append(int(parts[0]) + int(parts[1])/60.0)
                    except Exception:
                        ps.append(float("nan"))
                ax.plot(x, ps, lw=2.5, color="firebrick", label="Allure lissée")
            ax.invert_yaxis()
            ax.set_xlabel("Kilomètre")
            ax.set_ylabel("Allure (min/km)")
            ax.set_title("Allure prévisionnelle km par km")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with tab2:
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            x = list(range(1, len(df_out)+1))
            ax2.plot(x, df_out["Mult Pente"].values, label="Pente (Minetti+heu)", lw=2)
            if "Mult Temp" in df_out.columns:
                ax2.plot(x, df_out["Mult Temp"].values, label="Température/WBGT", lw=2)
            if "Mult Vent" in df_out.columns:
                ax2.plot(x, df_out["Mult Vent"].values, label="Vent", lw=2)
            if "Mult Fatigue" in df_out.columns:
                ax2.plot(x, df_out["Mult Fatigue"].values, label="Fatigue", lw=2, ls=":")
            if "Mult Altitude" in df_out.columns:
                ax2.plot(x, df_out["Mult Altitude"].values, label="Altitude physio", lw=1.5, ls="--")
            ax2.axhline(1.0, color="gray", lw=0.8)
            ax2.set_xlabel("Kilomètre")
            ax2.set_ylabel("Multiplicateur (1.0 = neutre)")
            ax2.set_title("Décomposition des facteurs de ralentissement km par km")
            ax2.legend()
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)
            plt.close(fig2)

        with tab3:
            st.dataframe(df_out, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# CARTE & PROFIL
# ─────────────────────────────────────────────────────────────
if gpx_file and points:
    with st.expander("🗺️ Carte & Profil d'altitude", expanded=False):
        try:
            lats_m = [p.latitude for p in points]
            lons_m = [p.longitude for p in points]
            view = pdk.ViewState(latitude=float(np.mean(lats_m)), longitude=float(np.mean(lons_m)), zoom=13, pitch=0)
            deck = pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                initial_view_state=view,
                layers=[pdk.Layer("PathLayer",
                                  data=[{"path": [[lon, lat] for lat, lon in zip(lats_m, lons_m)]}],
                                  get_path="path", get_color=[220, 50, 50], width_min_pixels=4)],
            )
            st.pydeck_chart(deck, use_container_width=True)

            # Profil
            cum_d = [0.0]
            for i in range(1, len(points)):
                cum_d.append(cum_d[-1] + haversine_m(
                    points[i-1].latitude, points[i-1].longitude,
                    points[i].latitude, points[i].longitude))
            x_km = np.array(cum_d) / 1000.0
            y_gps = np.array([getattr(p, "elevation", 0.0) or 0.0 for p in points])
            w = int(elev_smooth_window); w += (1 if w%2==0 else 0)
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            if w >= 3 and y_gps.size >= w:
                y_s = np.convolve(y_gps, np.ones(w)/w, mode="same")
                ax3.plot(x_km, y_s, lw=2, label="GPS lissé", color="steelblue")
                ax3.plot(x_km, y_gps, lw=1, alpha=0.2, color="gray", label="GPS brut")
            else:
                ax3.plot(x_km, y_gps, lw=2, label="GPS", color="steelblue")
            if dem_elevations is not None and len(dem_elevations) == len(points):
                y_dem = np.array([e if e is not None else 0.0 for e in dem_elevations])
                ax3.plot(x_km, y_dem, lw=2, ls="--", label="DEM corrigé", color="forestgreen")
            ax3.fill_between(x_km, ax3.get_ylim()[0] if ax3.get_ylim()[0] > 0 else 0, y_gps, alpha=0.08, color="steelblue")
            ax3.set_xlabel("Distance (km)")
            ax3.set_ylabel("Altitude (m)")
            ax3.set_title("Profil d'altitude")
            ax3.legend()
            ax3.grid(alpha=0.3)
            st.pyplot(fig3)
            plt.close(fig3)
        except Exception as e:
            st.error(f"Impossible d'afficher la carte : {e}")


# ══════════════════════════════════════════════════════════════
# ★ SECTION "MES IDÉES D'AMÉLIORATION" ★
# (Directement intégrées dans l'app pour que le coach les ait sous la main)
# ══════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("💡 Pistes d'amélioration futures (note de l'auteur du modèle)"):
    st.markdown("""
### Ce que ce modèle ne fait pas encore — et comment aller plus loin

**1. Intervalle de confiance probabiliste (pas juste ±5% fixe)**

L'incertitude réelle dépend de la qualité des références et de la distance à prédire.
Avec les résidus de la cross-validation, on pourrait calculer un vrai IC 80-90%.
Exemple : si MAPE = 3%, l'IC réel est ±5-6 min sur un marathon, pas ±10 min.

---

**2. Puissance de course (Stryd) — le gold standard**

Le modèle actuel utilise le *temps* et la *distance* des références.
Mais le temps dépend de la météo, du terrain, de la fatigue du jour.
La *puissance de course* (capteur Stryd) mesure le travail mécanique réel,
indépendamment de la pente et du vent. C'est la variable la plus stable pour modéliser
le potentiel d'un athlète.

→ Avec un fichier FIT contenant la puissance, on pourrait construire un modèle
`temps = f(puissance_critique, distance)` bien plus précis que Riegel.

---

**3. Charge d'entraînement (CTL/ATL/TSB)**

L'état de forme de l'athlète le jour de la course dépend de son historique
des 42 derniers jours (CTL = Chronic Training Load) et de sa fatigue récente (ATL).
La "forme" TSB = CTL - ATL prédit si l'athlète arrive frais ou fatigué.

→ Intégration possible avec l'API Garmin Connect, Strava, ou TrainingPeaks.
Un TSB de +10 peut représenter 1-2% de gain de performance.

---

**4. Type de surface et technicité du terrain**

Trail technique ≠ route ≠ piste. Les multiplicateurs de pente sont très différents
sur un sentier caillouteux vs un chemin forestier vs une route asphaltée.
Le GPX seul ne donne pas cette information.

→ Piste d'amélioration : croisement avec OpenStreetMap pour identifier le type de surface,
  ou champ manuel "technicité" (0-5) qui ajuste les multiplicateurs.

---

**5. Aérodynamisme individuel (CdA)**

Le coefficient aérodynamique varie fortement selon la morphologie de l'athlète
(grand/petit, courbure, kit de course). Un athlète élancé en combinaison est ~15%
moins exposé au vent qu'un athlète large en t-shirt.

→ Paramètre "gabarit athlète" (XS/S/M/L/XL) qui ajuste drag_coeff automatiquement.

---

**6. Hydratation et sodium**

La dégradation de performance par déshydratation est non linéaire :
- 1% de masse corporelle perdue → -2% de performance
- 2% → -4 à 6%
- 4% → risque médical

→ En combinant la chaleur (WBGT), la durée estimée et le taux de sudation de l'athlète,
  on pourrait prédire la dégradation par déshydratation et recommander un plan d'hydratation.

---

**7. Modèle de récupération intra-course (ultra-trail)**

Sur les ultras, les passages en ravitaillement (stops) et la marche en côte modifient
l'allure de façon non continue. Une modélisation par tronçons (sous-tracé GPX)
avec des règles métier "marche si pente > X%" serait plus fidèle.

---
*Ce modèle est un outil d'aide à la décision — validez toujours avec le ressenti de l'athlète.*
    """)
