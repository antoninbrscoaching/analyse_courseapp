# analyse_course_realiste.py
# Streamlit app — prédiction de course (pente + météo) avec impacts météo réalistes et stables.
#
# Objectifs par rapport à ton code actuel :
# - Vent moins “vélo-like” : dépend de la vitesse de course, impact faible, cap bas, lissé et/ou global.
# - Météo moins bruitée km/km : lissage (rolling median) sur headwind / tailwind et (option) application globale.
# - Température : courbe douce + caps (pas d’explosions).
# - Pente : garde ton modèle non-linéaire + cap + lissage altimétrique.
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

# -------------------------
# CONFIG UI
# -------------------------
st.set_page_config(page_title="Prédiction course route (réaliste)", layout="wide")
st.title("🏃‍♂️ Analyse & Prédiction de course — Version réaliste (météo stabilisée)")

TZ_NAME_DEFAULT = "Europe/Paris"


# ============================================================
# MÉTÉO (Open-Meteo)
# ============================================================

@st.cache_data(show_spinner=False)
def get_weather_openmeteo_minutely(lat, lon, dt_local_naive, tz_name=TZ_NAME_DEFAULT):
    """
    Forecast Open-Meteo (horaire) interpolé à la minute.
    dt_local_naive : datetime naive supposé dans tz_name.
    wind_direction_10m: direction FROM (°).
    """
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,relativehumidity_2m,wind_speed_10m,wind_direction_10m"
            f"&timezone={tz_name}"
        )
        r = requests.get(url, timeout=20)
        data = r.json()
        if "hourly" not in data:
            return None

        times = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
        temps = data["hourly"]["temperature_2m"]
        winds = data["hourly"]["wind_speed_10m"]
        hums = data["hourly"]["relativehumidity_2m"]
        wdirs = data["hourly"]["wind_direction_10m"]

        dt = dt_local_naive

        before = None
        after = None
        for i in range(len(times) - 1):
            if times[i] <= dt <= times[i + 1]:
                before = (times[i], temps[i], winds[i], hums[i], wdirs[i])
                after = (times[i + 1], temps[i + 1], winds[i + 1], hums[i + 1], wdirs[i + 1])
                break

        if before is None:
            idx = min(range(len(times)), key=lambda i: abs(times[i] - dt))
            return {
                "temp": float(temps[idx]),
                "wind": float(winds[idx]),
                "humidity": float(hums[idx]),
                "wind_dir": float(wdirs[idx]),
            }

        t1, temp1, wind1, hum1, dir1 = before
        t2, temp2, wind2, hum2, dir2 = after
        ratio = (dt - t1).total_seconds() / max(1.0, (t2 - t1).total_seconds())

        temp_interp = temp1 + ratio * (temp2 - temp1)
        wind_interp = wind1 + ratio * (wind2 - wind1)
        hum_interp = hum1 + ratio * (hum2 - hum1)

        # interpolation circulaire (direction)
        a1 = float(dir1) % 360.0
        a2 = float(dir2) % 360.0
        delta = (a2 - a1 + 540.0) % 360.0 - 180.0
        dir_interp = (a1 + ratio * delta) % 360.0

        return {
            "temp": float(temp_interp),
            "wind": float(wind_interp),
            "humidity": float(hum_interp),
            "wind_dir": float(dir_interp),
        }

    except Exception as e:
        st.error(f"Erreur météo minute : {e}")
        return None


@st.cache_data(show_spinner=False)
def get_weather_openmeteo_day(lat, lon, date_obj, tz_name=TZ_NAME_DEFAULT):
    """
    Archive météo (jour complet).
    Retourne times, temps, winds, hums, wdirs.
    """
    try:
        date_str = date_obj.strftime("%Y-%m-%d")
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={date_str}&end_date={date_str}"
            "&hourly=temperature_2m,relativehumidity_2m,wind_speed_10m,wind_direction_10m"
            f"&timezone={tz_name}"
        )
        r = requests.get(url, timeout=20)
        data = r.json()
        if "hourly" not in data:
            return None
        times = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
        temps = data["hourly"]["temperature_2m"]
        winds = data["hourly"]["wind_speed_10m"]
        hums = data["hourly"]["relativehumidity_2m"]
        wdirs = data["hourly"]["wind_direction_10m"]
        return times, temps, winds, hums, wdirs
    except Exception:
        return None


def get_avg_weather_for_period(lat, lon, start_dt, end_dt, tz_name=TZ_NAME_DEFAULT):
    """
    Météo moyenne robuste sur un intervalle.
    """
    if start_dt is None or end_dt is None:
        return None, None, None

    if (end_dt - start_dt).total_seconds() < 300:
        start_dt -= timedelta(minutes=2)
        end_dt += timedelta(minutes=2)

    meteo_day = get_weather_openmeteo_day(lat, lon, start_dt.date(), tz_name=tz_name)
    if not meteo_day:
        return None, None, None

    times, temps, winds, hums, _wdirs = meteo_day

    selT = [T for t, T in zip(times, temps) if start_dt <= t <= end_dt]
    selW = [W for t, W in zip(times, winds) if start_dt <= t <= end_dt]
    selH = [H for t, H in zip(times, hums) if start_dt <= t <= end_dt]

    if not selT:
        idx = min(range(len(times)), key=lambda i: abs(times[i] - start_dt))
        return float(temps[idx]), float(winds[idx]), float(hums[idx])

    return float(np.mean(selT)), float(np.mean(selW)), float(np.mean(selH))


# -------------------------
# UTILITAIRES
# -------------------------
def safe_float(val, default=0.0):
    try:
        if val is None:
            return float(default)
        if isinstance(val, str):
            s = val.strip()
            if s == "" or s.lower() in ("nan", "none"):
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
    """
    Accepte:
      - hh:mm:ss
      - mm:ss
      - hh:mm (heuristique)
      - ss
    Heuristique:
      - si 2 champs et le 1er >= 10 => mm:ss (ex: 18:30)
      - sinon => hh:mm (ex: 1:40 = 1h40)
    """
    if hms is None:
        return 0
    try:
        parts = str(hms).strip().split(":")
        parts = [int(p) for p in parts]

        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            a, b = parts
            if a >= 10:
                h, m, s = 0, a, b
            else:
                h, m, s = a, b, 0
        elif len(parts) == 1:
            h, m, s = 0, 0, parts[0]
        else:
            return 0

        if m < 0 or s < 0 or m > 59 or s > 59 or h < 0:
            return 0
        return max(0, h * 3600 + m * 60 + s)
    except Exception:
        return 0


def seconds_to_hms(seconds: float) -> str:
    try:
        seconds = int(round(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    except Exception:
        return "0:00:00"


def hms_to_timedelta(hms: str) -> timedelta:
    if hms is None:
        return timedelta(seconds=0)
    try:
        parts = str(hms).strip().split(":")
        parts = [int(p) for p in parts]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0
            m, s = parts
        elif len(parts) == 1:
            h = 0
            m = 0
            s = parts[0]
        else:
            return timedelta(seconds=0)
        return timedelta(hours=h, minutes=m, seconds=s)
    except Exception:
        return timedelta(seconds=0)


def pace_seconds_to_str_per_km(seconds_per_km: float) -> str:
    if seconds_per_km <= 0 or math.isnan(seconds_per_km) or math.isinf(seconds_per_km):
        return "0:00"
    m = int(seconds_per_km // 60)
    s = int(round(seconds_per_km % 60))
    return f"{m}:{s:02d}"


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0


def smallest_angle_diff_deg(a, b) -> float:
    return (b - a + 540.0) % 360.0 - 180.0


class SimplePoint:
    def __init__(self, lat, lon, elev=0.0, time=None):
        self.latitude = float(lat)
        self.longitude = float(lon)
        self.elevation = float(elev) if elev is not None else 0.0
        self.time = time

    def distance_3d(self, other):
        horiz = haversine_m(self.latitude, self.longitude, other.latitude, other.longitude)
        vert = (self.elevation - other.elevation)
        return math.sqrt(horiz * horiz + vert * vert)


def compute_dplus_dminus(elevs):
    try:
        arr = np.array([safe_float(e, np.nan) for e in elevs], dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size < 2:
            return 0.0, 0.0
        diffs = np.diff(arr)
        dup = float(np.sum(np.clip(diffs, a_min=0, a_max=None)))
        ddn = float(-np.sum(np.clip(diffs, a_min=None, a_max=0)))
        return dup, ddn
    except Exception:
        return 0.0, 0.0


# ============================================================
# MODÈLES (réalistes) : Temp + Pente + Vent
# ============================================================

def temp_multiplier_realistic(
    temp_c: float,
    opt_temp: float = 12.0,
    cold_quad: float = 0.0012,
    hot_quad: float = 0.0016,
    max_penalty: float = 0.08
) -> float:
    """
    Modèle doux, plafonné.
    - pénalité ~ quadratique, asymétrique (chaleur souvent plus coûteuse)
    - cap max à +8% (par défaut)
    """
    try:
        if temp_c is None:
            return 1.0
        t = float(temp_c)
        d = t - float(opt_temp)
        if d >= 0:
            pen = hot_quad * (d ** 2)
        else:
            pen = cold_quad * ((-d) ** 2)
        pen = min(float(max_penalty), float(pen))
        return 1.0 + pen
    except Exception:
        return 1.0


def grade_multiplier_nonlinear_capped(
    grade_pct,
    k_up=12.0,
    k_down=6.0,
    down_cap=-0.08,
    g0_up=3.0,
    g0_down=2.5,
    max_up=0.30,
    max_down=-0.06
):
    """
    Ton modèle de pente (bon) :
    - saturation progressive (tanh)
    - cap final
    """
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


def wind_components_along_course(wind_speed_ms, wind_dir_from_deg, course_bearing_deg_):
    """
    Retourne (headwind_ms, tailwind_ms, crosswind_ms) en prenant la direction "FROM".
    """
    if wind_speed_ms is None or wind_dir_from_deg is None or course_bearing_deg_ is None:
        return 0.0, 0.0, 0.0

    ws = float(wind_speed_ms)
    if ws <= 0:
        return 0.0, 0.0, 0.0

    wind_to = (float(wind_dir_from_deg) + 180.0) % 360.0
    delta = math.radians(smallest_angle_diff_deg(course_bearing_deg_, wind_to))

    along = ws * math.cos(delta)   # >0 tailwind, <0 headwind
    cross = ws * math.sin(delta)   # latéral

    tail = max(0.0, along)
    head = max(0.0, -along)
    cross = abs(float(cross))
    return float(head), float(tail), float(cross)


def wind_multiplier_realistic(
    head_ms: float,
    tail_ms: float,
    pace_s_per_km: float,
    drag_coeff: float = 0.012,
    tail_credit: float = 0.35,
    cap_head: float = 0.10,
    cap_tail: float = -0.04
) -> float:
    """
    Vent réaliste pour la course à pied :
    - dépend de la vitesse de course (v_run)
    - impact basé sur surplus de "résistance" ~ (v_rel^2 - v_run^2) / v_run^2
    - tailwind : bénéfice partiel (tail_credit < 1)
    - caps faibles (par défaut +10% / -4%)

    drag_coeff : règle l'amplitude globale (0.008–0.015 typique).
    """
    try:
        pace = max(120.0, float(pace_s_per_km))  # évite vitesses irréalistes, 2:00/km min
        v_run = 1000.0 / pace  # m/s

        # On calcule en "équivalent headwind" : head positif, tail négatif
        w_along = float(head_ms) - float(tail_ms)  # >0 défavorable
        # vitesse relative air
        v_rel = max(0.0, v_run + w_along)

        # Surplus relatif
        base = max(1e-9, v_run ** 2)
        extra = (v_rel ** 2 - v_run ** 2) / base

        # tailwind : on ne "rend" qu'une fraction du gain
        if extra < 0:
            extra = float(tail_credit) * extra

        mult = 1.0 + float(drag_coeff) * float(extra)
        mult = min(1.0 + float(cap_head), mult)
        mult = max(1.0 + float(cap_tail), mult)
        return float(mult)
    except Exception:
        return 1.0


# ============================================================
# FIT / TCX / GPX
# ============================================================

def parse_gpx_points(file):
    try:
        file.seek(0)
        gpx = gpxpy.parse(file)
        points = []
        for track in gpx.tracks:
            for segment in track.segments:
                for p in segment.points:
                    points.append(p)
        return gpx, points
    except Exception as e:
        st.error(f"Erreur parsing GPX : {e}")
        return None, []


def gpx_to_df(points):
    return pd.DataFrame([{
        "lat": p.latitude,
        "lon": p.longitude,
        "elev": (p.elevation or 0.0),
        "time": getattr(p, "time", None)
    } for p in points])


def extract_segment_from_points(points, start_td, end_td):
    if not points or len(points) < 2:
        return points

    def get_time(p):
        if isinstance(p, dict):
            return p.get("time", None)
        return getattr(p, "time", None)

    times = [get_time(p) for p in points if get_time(p) is not None]
    if len(times) < 2:
        return points

    t0 = min(times)
    start_dt = t0 + start_td
    end_dt = t0 + end_td + timedelta(seconds=1)

    seg = [p for p in points if (get_time(p) is not None and start_dt <= get_time(p) <= end_dt)]
    return seg if len(seg) >= 2 else points


def parse_fit(file, tz_name=TZ_NAME_DEFAULT):
    try:
        file.seek(0)
        fit = FitFile(file)
        fit.parse()

        records = []
        times_points = []

        start_global = None
        elapsed_global = None

        for msg in fit.get_messages("session"):
            vals = {d.name: d.value for d in msg}
            if isinstance(vals.get("start_time"), datetime):
                start_global = vals["start_time"].replace(tzinfo=None)
            if isinstance(vals.get("total_elapsed_time"), (int, float)):
                elapsed_global = float(vals["total_elapsed_time"])

        for msg in fit.get_messages("record"):
            vals = {d.name: d.value for d in msg}
            lat_raw = vals.get("position_lat")
            lon_raw = vals.get("position_long")
            ts = vals.get("timestamp")

            if lat_raw is None or lon_raw is None:
                continue

            lat = lat_raw * (180 / 2**31)
            lon = lon_raw * (180 / 2**31)

            dt_local = ts.replace(tzinfo=None) if isinstance(ts, datetime) else None
            records.append((lat, lon, vals.get("altitude", 0.0), vals.get("distance", 0.0)))
            times_points.append(dt_local)

        if not records:
            return None

        df = pd.DataFrame(records, columns=["lat", "lon", "elev", "dist"])
        valid_times = [t for t in times_points if t is not None]

        if len(valid_times) >= 2:
            start_dt = min(valid_times)
            end_dt = max(valid_times)
        else:
            start_dt = start_global
            if start_global and elapsed_global:
                end_dt = start_global + timedelta(seconds=elapsed_global)
            elif start_global:
                end_dt = start_global + timedelta(minutes=5)
            else:
                start_dt = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=1)
                end_dt = start_dt + timedelta(minutes=5)

        avgT, avgW, avgH = get_avg_weather_for_period(records[0][0], records[0][1], start_dt, end_dt, tz_name=tz_name)

        elev_arr = df["elev"].astype(float).values
        dup = float(np.sum(np.clip(np.diff(elev_arr), a_min=0, a_max=None))) if elev_arr.size >= 2 else 0.0
        ddn = float(-np.sum(np.clip(np.diff(elev_arr), a_min=None, a_max=0))) if elev_arr.size >= 2 else 0.0

        return {
            "points": [{"lat": r[0], "lon": r[1], "elev": r[2], "dist": r[3], "time": t}
                       for (r, t) in zip(records, times_points)],
            "distance": float(df["dist"].max()) if not df.empty else 0.0,
            "D_up": dup,
            "D_down": ddn,
            "duration_hms": seconds_to_hms((end_dt - start_dt).total_seconds()),
            "avg_temp": avgT,
            "avg_wind": avgW,
            "avg_humidity": avgH
        }
    except Exception as e:
        st.error(f"Erreur FIT : {e}")
        return None


def parse_tcx(file, tz_name=TZ_NAME_DEFAULT):
    try:
        file.seek(0)
        tree = ET.parse(file)
        root = tree.getroot()
    except Exception:
        return None

    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    tps = root.findall(".//tcx:Trackpoint", ns)

    pts, times, elevs = [], [], []
    for tp in tps:
        lat = tp.find("tcx:Position/tcx:LatitudeDegrees", ns)
        lon = tp.find("tcx:Position/tcx:LongitudeDegrees", ns)
        tim = tp.find("tcx:Time", ns)
        ele = tp.find("tcx:AltitudeMeters", ns)
        if lat is None or lon is None:
            continue

        lat = float(lat.text)
        lon = float(lon.text)
        elev = float(ele.text) if ele is not None else 0.0

        try:
            t = datetime.fromisoformat(tim.text.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            t = None

        pts.append(SimplePoint(lat, lon, elev, t))
        times.append(t)
        elevs.append(elev)

    if len(pts) < 2:
        return None

    valid_times = [t for t in times if t is not None]
    if len(valid_times) >= 2:
        start_dt, end_dt = valid_times[0], valid_times[-1]
    elif len(valid_times) == 1:
        start_dt = valid_times[0]
        end_dt = start_dt + timedelta(minutes=5)
    else:
        start_dt = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=1)
        end_dt = start_dt + timedelta(minutes=5)

    avgT, avgW, avgH = get_avg_weather_for_period(pts[0].latitude, pts[0].longitude, start_dt, end_dt, tz_name=tz_name)

    total = sum(pts[i].distance_3d(pts[i - 1]) for i in range(1, len(pts)))
    dup, ddn = compute_dplus_dminus(elevs)

    return {
        "points": pts,
        "distance": round(total),
        "D_up": round(dup, 1),
        "D_down": round(ddn, 1),
        "duration_hms": seconds_to_hms((end_dt - start_dt).total_seconds()),
        "avg_temp": avgT,
        "avg_wind": avgW,
        "avg_humidity": avgH
    }


# ============================================================
# FIT DE PERF (log-log)
# ============================================================

def fit_loglog_model(refs):
    X, Y = [], []
    for r in refs:
        d_m = r.get("distance", None)
        t_raw = r.get("temps")
        if d_m is None or d_m <= 0:
            continue
        secs = float(t_raw) if isinstance(t_raw, (int, float, np.number)) else hms_to_seconds(str(t_raw))
        if secs <= 0:
            continue
        d_km = float(d_m) / 1000.0
        X.append(math.log(max(1e-6, d_km)))
        Y.append(math.log(max(1e-6, secs)))

    if len(X) >= 2:
        coeffs = np.polyfit(X, Y, 1)
        K = float(coeffs[0])
        K = max(0.85, min(1.25, K))
        loga = float(coeffs[1])
        a = math.exp(loga)
        if not (0 < a < 1e7):
            a = 240.0
        return a, K
    elif len(X) == 1:
        d_km = math.exp(X[0])
        secs = math.exp(Y[0])
        a = secs / max(1e-6, d_km)
        return a, 1.0
    else:
        return 240.0, 1.0


def predict_time_flat(distance_m, a, K):
    d_km = float(distance_m) / 1000.0
    return float(a) * (d_km ** float(K))


def override_with_objective(distance_m, objective_time_hms, K):
    objective_seconds = hms_to_seconds(objective_time_hms)
    d_km = float(distance_m) / 1000.0
    if d_km <= 0:
        return None
    return float(objective_seconds) / (d_km ** float(K))


# ============================================================
# NORMALISATION RÉFÉRENCES (simple, stable)
# ============================================================

def elev_factor_from_dplus_dminus(
    D_up_m: float,
    D_down_m: float,
    distance_m: float,
    grade_k_up: float,
    grade_k_down: float,
    grade_down_cap: float,
    g0_up_pct: float,
    g0_down_pct: float,
    max_up: float,
    max_down: float
) -> float:
    dist = max(1e-6, float(distance_m))
    dup = max(0.0, float(D_up_m))
    ddn = max(0.0, float(D_down_m))

    g_up = dup / dist
    g_down = ddn / dist

    g0u = max(1e-6, float(g0_up_pct) / 100.0)
    g0d = max(1e-6, float(g0_down_pct) / 100.0)

    g_eff_up = math.tanh(g_up / g0u) * g0u
    up_term = float(grade_k_up) * g_eff_up

    g_eff_down = math.tanh(g_down / g0d) * g0d
    down_bonus = min(float(grade_k_down) * g_eff_down, abs(float(grade_down_cap)))

    mult = 1.0 + up_term - down_bonus
    mult = min(mult, 1.0 + float(max_up))
    mult = max(mult, 1.0 + float(max_down))
    return max(0.01, float(mult))


def recalibrate_ref_to_ideal(
    ref,
    opt_temp: float,
    # pente refs
    grade_k_up: float,
    grade_k_down: float,
    grade_down_cap: float,
    g0_up_pct: float,
    g0_down_pct: float,
    max_grade_up: float,
    max_grade_down: float,
    # damping refs
    elev_ref_power: float = 0.60,
    temp_ref_power: float = 0.85,
    # temp modèle réaliste
    cold_quad: float = 0.0012,
    hot_quad: float = 0.0016,
    temp_max_penalty: float = 0.08
):
    secs = hms_to_seconds(ref.get("temps")) if ref.get("temps") is not None else 0
    D_up = safe_float(ref.get("D_up", 0.0))
    D_down = safe_float(ref.get("D_down", 0.0))
    seg_len = safe_float(ref.get("distance", 1000.0))
    seg_len = seg_len if seg_len > 0 else 1000.0

    factor_elev = elev_factor_from_dplus_dminus(
        D_up_m=D_up,
        D_down_m=D_down,
        distance_m=seg_len,
        grade_k_up=grade_k_up,
        grade_k_down=grade_k_down,
        grade_down_cap=grade_down_cap,
        g0_up_pct=g0_up_pct,
        g0_down_pct=g0_down_pct,
        max_up=max_grade_up,
        max_down=max_grade_down
    )

    # damping pente refs (évite "plat idéal" trop optimiste)
    factor_elev = max(0.01, float(factor_elev))
    secs_no_elev = secs / (factor_elev ** float(elev_ref_power))

    # temp refs
    temp_real = ref.get("avg_temp")
    if temp_real is not None:
        mult_real = temp_multiplier_realistic(
            temp_real, opt_temp=opt_temp,
            cold_quad=cold_quad, hot_quad=hot_quad,
            max_penalty=temp_max_penalty
        )
        secs_no_temp = secs_no_elev / (max(0.01, float(mult_real)) ** float(temp_ref_power))
    else:
        secs_no_temp = secs_no_elev

    # remettre à T° opt (mult = 1.0 car à l'opt, pénalité = 0)
    return max(0.0, float(secs_no_temp))


def prepare_refs_for_fit(
    refs_input,
    ideal_refs: bool,
    opt_temp: float,
    grade_k_up: float,
    grade_k_down: float,
    grade_down_cap: float,
    g0_up_pct: float,
    g0_down_pct: float,
    max_grade_up: float,
    max_grade_down: float,
    elev_ref_power: float,
    temp_ref_power: float,
    cold_quad: float,
    hot_quad: float,
    temp_max_penalty: float
):
    prepared = []
    for r in refs_input:
        d = safe_float(r.get("distance", 0.0))
        file_dur = r.get("duration_hms_file")
        raw_t = file_dur if file_dur else r.get("temps", "0:00:00")

        ref_for_calib = {
            "distance": d,
            "temps": raw_t,
            "D_up": r.get("D_up", 0.0),
            "D_down": r.get("D_down", 0.0),
            "avg_temp": r.get("avg_temp"),
        }

        if ideal_refs:
            secs_recal = recalibrate_ref_to_ideal(
                ref_for_calib,
                opt_temp=opt_temp,
                grade_k_up=grade_k_up,
                grade_k_down=grade_k_down,
                grade_down_cap=grade_down_cap,
                g0_up_pct=g0_up_pct,
                g0_down_pct=g0_down_pct,
                max_grade_up=max_grade_up,
                max_grade_down=max_grade_down,
                elev_ref_power=elev_ref_power,
                temp_ref_power=temp_ref_power,
                cold_quad=cold_quad,
                hot_quad=hot_quad,
                temp_max_penalty=temp_max_penalty
            )
        else:
            secs_recal = hms_to_seconds(raw_t)

        prepared.append({"distance": float(d), "temps": float(secs_recal)})
    return prepared


# ============================================================
# PRÉDICTION PRINCIPALE
# ============================================================

def run_prediction_df(
    distance_cible_km,
    refs_input,
    points,
    date_course_local,
    heure_course_local,
    ideal_refs=True,

    # pente
    apply_grade=True,
    grade_k_up=12.0,
    grade_k_down=5.0,
    grade_down_cap=-0.08,
    g0_up_pct=3.0,
    g0_down_pct=2.5,
    max_grade_up=0.30,
    max_grade_down=-0.06,
    elev_smooth_window=11,
    grade_power=0.85,

    # temp
    apply_temp=True,
    opt_temp=12.0,
    cold_quad=0.0012,
    hot_quad=0.0016,
    temp_max_penalty=0.08,
    temp_power=1.0,

    # vent
    apply_wind=True,
    wind_mode="Lissé (km/km)",
    wind_smooth_window_km=5,
    drag_coeff=0.012,
    tail_credit=0.35,
    wind_cap_head=0.10,
    wind_cap_tail=-0.04,
    wind_power=1.0,

    # refs damping
    elev_ref_power=0.60,
    temp_ref_power=0.85,

    # fatigue
    apply_fatigue=False,
    fatigue_rate=0.0,

    # objectif
    objective_time_hms=None,

    tz_name=TZ_NAME_DEFAULT,
    show_smoothed_pace=True,
    smooth_pace_window_km=3
):
    if not points or len(points) < 2:
        raise ValueError("GPX invalide ou trop court.")

    # distances cumulées (3D)
    total_m = 0.0
    cum = [0.0]
    for i in range(1, len(points)):
        d = SimplePoint(
            points[i - 1].latitude, points[i - 1].longitude, getattr(points[i - 1], "elevation", 0.0)
        ).distance_3d(
            SimplePoint(points[i].latitude, points[i].longitude, getattr(points[i], "elevation", 0.0))
        )
        total_m += d
        cum.append(total_m)

    distance_gpx_km = total_m / 1000.0
    if distance_cible_km is None or distance_cible_km <= 0:
        distance_cible_km = distance_gpx_km

    facteur_dist = distance_cible_km / max(distance_gpx_km, 1e-9)
    total_corr = total_m * facteur_dist
    dists_corr = np.asarray([d * facteur_dist for d in cum], dtype=float)

    # altitude
    elev_list = np.asarray([getattr(p, "elevation", 0.0) or 0.0 for p in points], dtype=float)
    if elev_list.size != dists_corr.size:
        xs = np.linspace(0, total_m, elev_list.size)
        new_x = np.linspace(0, total_m, dists_corr.size)
        elev_list = np.interp(new_x, xs, elev_list)

    # lissage altitude
    if elev_smooth_window and elev_smooth_window >= 3 and elev_list.size >= elev_smooth_window:
        w = int(elev_smooth_window)
        if w % 2 == 0:
            w += 1
        kernel = np.ones(w) / w
        elev_smooth = np.convolve(elev_list, kernel, mode="same")
    else:
        elev_smooth = elev_list

    # refs -> fit
    refs_for_fit = prepare_refs_for_fit(
        refs_input=refs_input,
        ideal_refs=ideal_refs,
        opt_temp=opt_temp,
        grade_k_up=grade_k_up,
        grade_k_down=grade_k_down,
        grade_down_cap=grade_down_cap,
        g0_up_pct=g0_up_pct,
        g0_down_pct=g0_down_pct,
        max_grade_up=max_grade_up,
        max_grade_down=max_grade_down,
        elev_ref_power=elev_ref_power,
        temp_ref_power=temp_ref_power,
        cold_quad=cold_quad,
        hot_quad=hot_quad,
        temp_max_penalty=temp_max_penalty
    )

    a, K = fit_loglog_model(refs_for_fit)

    a_override = override_with_objective(int(distance_cible_km * 1000), objective_time_hms, K) if objective_time_hms else None
    baseline_a = a_override if a_override is not None else a

    base_flat_total = predict_time_flat(int(distance_cible_km * 1000), baseline_a, K)
    base_s_per_km_flat = base_flat_total / max(distance_cible_km, 1e-9)

    # segments (1km + dernier)
    km_marks = [i * 1000 for i in range(1, int(total_corr // 1000) + 1)]
    last_seg = total_corr - (int(total_corr // 1000) * 1000)
    if last_seg > 1e-6:
        km_marks.append(total_corr)

    df_points = pd.DataFrame([{
        "lat": p.latitude,
        "lon": p.longitude,
        "elev": getattr(p, "elevation", 0.0),
        "time": getattr(p, "time", None),
    } for p in points])

    dt_depart = datetime.combine(date_course_local, heure_course_local)

    # 1) passe 1 : calc météo + pente + temps "sans vent" pour estimer allure locale
    pre = []
    cum_time_tmp = 0.0

    for i, d in enumerate(km_marks):
        seg_length_m = 1000.0
        if i == len(km_marks) - 1 and last_seg > 1e-6:
            seg_length_m = (d - km_marks[-2]) if len(km_marks) >= 2 else d

        e_cur = float(np.interp(d, dists_corr, elev_smooth))
        e_prev_d = max(d - seg_length_m, 0.0)
        e_prev = float(np.interp(e_prev_d, dists_corr, elev_smooth)) if i > 0 else e_cur
        delta_e = e_cur - e_prev
        grade_pct = (delta_e / max(1e-6, seg_length_m)) * 100.0

        # temps plat segment
        t_flat = base_s_per_km_flat * (seg_length_m / 1000.0)

        # pente
        if apply_grade:
            g_mult = grade_multiplier_nonlinear_capped(
                grade_pct,
                k_up=grade_k_up,
                k_down=grade_k_down,
                down_cap=grade_down_cap,
                g0_up=g0_up_pct,
                g0_down=g0_down_pct,
                max_up=max_grade_up,
                max_down=max_grade_down
            )
            t_after_grade = t_flat * (g_mult ** float(grade_power))
        else:
            g_mult = 1.0
            t_after_grade = t_flat

        # fatigue
        if apply_fatigue and fatigue_rate > 0 and total_corr > 0:
            progression = d / total_corr
            t_after_fatigue = t_after_grade * (1.0 + (fatigue_rate / 100.0) * progression)
        else:
            t_after_fatigue = t_after_grade

        # passage au milieu segment pour météo
        passage_dt = dt_depart + timedelta(seconds=cum_time_tmp + t_after_fatigue / 2.0)

        lat_seg = float(np.interp(d, dists_corr, df_points["lat"].values))
        lon_seg = float(np.interp(d, dists_corr, df_points["lon"].values))

        d_start = max(d - seg_length_m, 0.0)
        lat_start = float(np.interp(d_start, dists_corr, df_points["lat"].values))
        lon_start = float(np.interp(d_start, dists_corr, df_points["lon"].values))
        course_bearing = bearing_deg(lat_start, lon_start, lat_seg, lon_seg)

        meteo = get_weather_openmeteo_minutely(lat_seg, lon_seg, passage_dt, tz_name=tz_name)
        temp_here = meteo["temp"] if meteo else None
        wind_here = meteo["wind"] if meteo else None
        hum_here = meteo["humidity"] if meteo else None
        wind_dir_here = meteo.get("wind_dir") if meteo else None

        # temp
        if apply_temp and temp_here is not None:
            temp_mult = temp_multiplier_realistic(
                temp_here, opt_temp=opt_temp,
                cold_quad=cold_quad, hot_quad=hot_quad,
                max_penalty=temp_max_penalty
            )
            t_after_temp = t_after_fatigue * (float(temp_mult) ** float(temp_power))
        else:
            temp_mult = 1.0
            t_after_temp = t_after_fatigue

        # allure locale estimée (avant vent)
        pace_s_per_km_local = (t_after_temp / seg_length_m) * 1000.0 if seg_length_m > 0 else t_after_temp

        head_ms, tail_ms, cross_ms = wind_components_along_course(wind_here, wind_dir_here, course_bearing)

        pre.append({
            "idx": i,
            "d": float(d),
            "seg_length_m": float(seg_length_m),
            "grade_pct": float(grade_pct),
            "grade_mult": float(g_mult),
            "temp": temp_here,
            "humidity": hum_here,
            "wind": wind_here,
            "wind_dir": wind_dir_here,
            "course_bearing": float(course_bearing),
            "head_ms": float(head_ms),
            "tail_ms": float(tail_ms),
            "cross_ms": float(cross_ms),
            "temp_mult": float(temp_mult),
            "t_no_wind": float(t_after_temp),
            "pace_no_wind": float(pace_s_per_km_local),
        })

        cum_time_tmp += float(t_after_temp)

    pre_df = pd.DataFrame(pre)

    # 2) Vent : choisir mode
    #    - Global : calc un multiplicateur unique basé sur head/tail "moyen orienté"
    #    - Lissé : rolling median sur head/tail
    if apply_wind and not pre_df.empty:
        if wind_mode == "Global (un seul effet sur la course)":
            head_g = float(np.median(pre_df["head_ms"].values))
            tail_g = float(np.median(pre_df["tail_ms"].values))
            pace_g = float(np.median(pre_df["pace_no_wind"].values))
            global_wind_mult = wind_multiplier_realistic(
                head_ms=head_g,
                tail_ms=tail_g,
                pace_s_per_km=pace_g,
                drag_coeff=drag_coeff,
                tail_credit=tail_credit,
                cap_head=wind_cap_head,
                cap_tail=wind_cap_tail
            )
            pre_df["wind_mult"] = float(global_wind_mult)
        else:
            # Lissé km/km
            w = int(max(1, wind_smooth_window_km))
            if w % 2 == 0:
                w += 1
            head_s = pd.Series(pre_df["head_ms"]).rolling(window=w, center=True, min_periods=1).median()
            tail_s = pd.Series(pre_df["tail_ms"]).rolling(window=w, center=True, min_periods=1).median()

            wind_mults = []
            for hm, tm, pace in zip(head_s.values, tail_s.values, pre_df["pace_no_wind"].values):
                wind_mults.append(
                    wind_multiplier_realistic(
                        head_ms=float(hm),
                        tail_ms=float(tm),
                        pace_s_per_km=float(pace),
                        drag_coeff=drag_coeff,
                        tail_credit=tail_credit,
                        cap_head=wind_cap_head,
                        cap_tail=wind_cap_tail
                    )
                )
            pre_df["wind_mult"] = np.array(wind_mults, dtype=float)
            pre_df["head_ms_smooth"] = head_s.values
            pre_df["tail_ms_smooth"] = tail_s.values
    else:
        pre_df["wind_mult"] = 1.0

    # 3) Calcule temps final (avec vent)
    t_raw = pre_df["t_no_wind"].values * (pre_df["wind_mult"].values ** float(wind_power))

    # scale si objectif
    if objective_time_hms:
        objective_seconds = hms_to_seconds(objective_time_hms)
        sum_raw = float(np.sum(t_raw))
        scale = (objective_seconds / sum_raw) if sum_raw > 0 else 1.0
    else:
        scale = 1.0

    # table finale
    results = []
    cum_time = 0.0
    for i in range(len(pre_df)):
        seg = pre_df.iloc[i]
        t_seg = float(t_raw[i]) * float(scale)
        cum_time += t_seg
        pace_per_km = (t_seg / float(seg["seg_length_m"])) * 1000.0 if seg["seg_length_m"] > 0 else t_seg

        km_label = (seg["idx"] + 1) if seg["seg_length_m"] >= 1000 - 1e-6 else f"{int(seg['idx']+1)} ({seg['seg_length_m']:.0f}m)"

        # affichage head/tail : si lissé, afficher lissé
        head_disp = seg.get("head_ms_smooth", seg["head_ms"])
        tail_disp = seg.get("tail_ms_smooth", seg["tail_ms"])

        results.append({
            "Km": km_label,
            "Pente (%)": round(float(seg["grade_pct"]), 2),
            "Mult Pente": round(float(seg["grade_mult"]), 4),

            "Temp (°C)": round(float(seg["temp"]), 1) if seg["temp"] is not None else None,
            "Mult Temp": round(float(seg["temp_mult"]), 4),

            "Vent (m/s)": round(float(seg["wind"]), 1) if seg["wind"] is not None else None,
            "Dir vent (° FROM)": round(float(seg["wind_dir"]), 0) if seg["wind_dir"] is not None else None,
            "Cap seg (°)": round(float(seg["course_bearing"]), 0),

            "Headwind (m/s)": round(float(head_disp), 2),
            "Tailwind (m/s)": round(float(tail_disp), 2),
            "Mult Vent": round(float(seg["wind_mult"]), 4),

            "Humidité (%)": round(float(seg["humidity"]), 1) if seg["humidity"] is not None else None,

            "Temps segment (s)": round(t_seg, 1),
            "Allure (min/km)": pace_seconds_to_str_per_km(pace_per_km),
            "Temps cumulé": seconds_to_hms(cum_time),
        })

    df = pd.DataFrame(results)

    # allure lissée (affichage)
    if show_smoothed_pace and not df.empty:
        try:
            pace_s = df["Temps segment (s)"].astype(float).values
            w = int(max(1, smooth_pace_window_km))
            if w % 2 == 0:
                w += 1
            s = pd.Series(pace_s).rolling(window=w, center=True, min_periods=1).median()
            df["Allure lissée (min/km)"] = s.apply(pace_seconds_to_str_per_km)
        except Exception:
            pass

    total_seconds = float(np.sum(t_raw)) * float(scale)

    return {
        "df": df,
        "total_seconds": float(total_seconds),
        "total_human": seconds_to_hms(total_seconds),
        "distance_gpx_km": float(distance_gpx_km),
        "base_flat_total": float(base_flat_total),
        "a": float(baseline_a),
        "K": float(K),
    }


# ============================================================
# UI
# ============================================================

st.header("1️⃣ Parcours GPX")
gpx_file = st.file_uploader("📂 Importer un fichier GPX", type=["gpx"])
points = None
if gpx_file:
    gpx, points = parse_gpx_points(gpx_file)
    if points:
        total_m_tmp = sum(
            SimplePoint(points[i - 1].latitude, points[i - 1].longitude, getattr(points[i - 1], "elevation", 0.0)).distance_3d(
                SimplePoint(points[i].latitude, points[i].longitude, getattr(points[i], "elevation", 0.0))
            )
            for i in range(1, len(points))
        )
        st.session_state["gpx_original_distance_km"] = total_m_tmp / 1000.0
    else:
        st.session_state["gpx_original_distance_km"] = None


st.header("2️⃣ Courses de référence (manuel ou FIT/TCX)")
if "n_refs" not in st.session_state:
    st.session_state.n_refs = 3

cols = st.columns([1, 1])
with cols[0]:
    if st.button("➕ Ajouter (max 6)") and st.session_state.n_refs < 6:
        st.session_state.n_refs += 1
with cols[1]:
    if st.button("➖ Retirer") and st.session_state.n_refs > 1:
        st.session_state.n_refs -= 1

refs_raw = []
for i in range(1, st.session_state.n_refs + 1):
    st.markdown(f"#### Référence {i}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        use_file = st.checkbox("Importer (FIT/TCX) ?", key=f"use_file_{i}")

    default_dist = st.session_state.get(f"dist_{i}", 5000 * i)
    default_temps = st.session_state.get(f"temps_{i}", "0:40:00")
    default_dup = st.session_state.get(f"dup_{i}", 0.0)
    default_ddn = st.session_state.get(f"ddn_{i}", 0.0)

    with c2:
        dist = st.number_input(f"Dist {i} (m)", value=float(default_dist), key=f"dist_{i}")
    with c3:
        temps = st.text_input(f"Temps {i} (h:mm:ss)", value=str(default_temps), key=f"temps_{i}")
    with c4:
        dup = st.number_input(f"D+ {i}", value=float(default_dup), key=f"dup_{i}")
    with c5:
        ddn = st.number_input(f"D- {i}", value=float(default_ddn), key=f"ddn_{i}")

    with c6:
        file_in = st.file_uploader(f"FIT/TCX {i}", type=["fit", "tcx"], key=f"fileref_{i}") if use_file else None

    col_s, col_e = st.columns(2)
    start_hms = col_s.text_input(f"Début réf {i} (hh:mm:ss)", value="00:00:00", key=f"start_{i}")
    end_hms = col_e.text_input(f"Fin réf {i} (hh:mm:ss)", value="23:59:59", key=f"end_{i}")

    start_td = hms_to_timedelta(start_hms)
    end_td = hms_to_timedelta(end_hms)

    duration_hms_file = None
    avg_temp_ref = None
    avg_wind_ref = None
    avg_hum_ref = None
    fit_data = None
    tcx_data = None

    filename = file_in.name.lower() if file_in else ""

    if file_in:
        if filename.endswith(".fit"):
            fit_data = parse_fit(file_in)
            if fit_data:
                dist = fit_data["distance"]
                dup = fit_data["D_up"]
                ddn = fit_data["D_down"]
                duration_hms_file = fit_data["duration_hms"]
                avg_temp_ref = fit_data["avg_temp"]
                avg_wind_ref = fit_data["avg_wind"]
                avg_hum_ref = fit_data["avg_humidity"]

        elif filename.endswith(".tcx"):
            tcx_data = parse_tcx(file_in)
            if tcx_data:
                dist = tcx_data["distance"]
                dup = tcx_data["D_up"]
                ddn = tcx_data["D_down"]
                duration_hms_file = tcx_data["duration_hms"]
                avg_temp_ref = tcx_data["avg_temp"]
                avg_wind_ref = tcx_data["avg_wind"]
                avg_hum_ref = tcx_data["avg_humidity"]

        # découpe segment
        if (start_td.total_seconds() > 0 or end_td.total_seconds() < 86399):
            pts = None
            if filename.endswith(".fit") and fit_data and "points" in fit_data:
                pts = fit_data["points"]
            elif filename.endswith(".tcx") and tcx_data and "points" in tcx_data:
                pts = tcx_data["points"]

            if pts:
                seg = extract_segment_from_points(pts, start_td, end_td)

                dist_seg = 0.0
                elevs_seg = []
                times_seg = []

                for j in range(1, len(seg)):
                    p1, p2 = seg[j - 1], seg[j]
                    if isinstance(p1, dict):
                        lat1, lon1, elev1, t1 = p1["lat"], p1["lon"], p1["elev"], p1.get("time")
                    else:
                        lat1, lon1, elev1, t1 = p1.latitude, p1.longitude, p1.elevation, p1.time

                    if isinstance(p2, dict):
                        lat2, lon2, elev2, t2 = p2["lat"], p2["lon"], p2["elev"], p2.get("time")
                    else:
                        lat2, lon2, elev2, t2 = p2.latitude, p2.longitude, p2.elevation, p2.time

                    dist_seg += haversine_m(lat1, lon1, lat2, lon2)
                    elevs_seg.append(elev2)
                    if t2 is not None:
                        times_seg.append(t2)

                dup, ddn = compute_dplus_dminus(elevs_seg)

                if len(times_seg) >= 2:
                    duration_hms_file = seconds_to_hms((times_seg[-1] - times_seg[0]).total_seconds())

                dist = round(dist_seg)

    temps_effectif = duration_hms_file if duration_hms_file else temps

    refs_raw.append({
        "distance": float(dist),
        "temps": str(temps_effectif),
        "D_up": float(dup),
        "D_down": float(ddn),
        "duration_hms_file": duration_hms_file,
        "avg_temp": avg_temp_ref,
        "avg_wind": avg_wind_ref,
        "avg_humidity": avg_hum_ref,
        "start_td": start_td,
        "end_td": end_td,
        "start_hms": start_hms,
        "end_hms": end_hms,
    })

st.subheader("🧪 Contrôle : allure implicite des références")
for idx, r in enumerate(refs_raw, start=1):
    secs = hms_to_seconds(r["temps"])
    dist_km = float(r["distance"]) / 1000.0 if r["distance"] else 0.0
    if secs > 0 and dist_km > 0:
        pace = secs / dist_km
        st.write(f"Réf {idx} — {r['distance']:.0f} m en {r['temps']} → {pace_seconds_to_str_per_km(pace)}/km")
        if pace < 150:
            st.warning("⚠️ Allure extrêmement rapide → vérifie le format du temps (ex: 1:40 = 1h40 ou 1min40).")

st.subheader("⏱️ Récap références (raw)")
for idx, r in enumerate(refs_raw, start=1):
    st.write(
        f"Réf {idx} — Dist: {r['distance']:.0f} m | Temps: {r['temps']} | "
        f"D+ {r['D_up']:.0f} m / D- {r['D_down']:.0f} m | "
        f"Dur file: {r.get('duration_hms_file')} | "
        f"Temp moy: {r.get('avg_temp')}°C | "
        f"Intervalle: {r.get('start_hms','0:00:00')} → {r.get('end_hms','fin')}"
    )


# -------------------------
# Paramètres modèle
# -------------------------
st.header("3️⃣ Paramètres modèle (réalistes)")

st.subheader("⛰️ Références : normalisation (recommandée)")
ideal_refs = st.checkbox("Normaliser les références vers conditions idéales (plat + T° opt)", value=True)

colR1, colR2 = st.columns(2)
with colR1:
    elev_ref_power = st.slider("Atténuation pente refs (0=off, 1=full)", 0.0, 1.0, 0.60, 0.05)
with colR2:
    temp_ref_power = st.slider("Atténuation température refs (0=off, 1=full)", 0.0, 1.0, 0.85, 0.05)

st.subheader("🎢 Pente (GPX)")
apply_grade = st.checkbox("Prendre en compte la pente", value=True)
colg1, colg2, colg3 = st.columns(3)
with colg1:
    grade_k_up = st.number_input("Sensibilité montée", value=12.0, step=0.5)
with colg2:
    grade_k_down = st.number_input("Sensibilité descente", value=5.0, step=0.5)
with colg3:
    grade_down_cap = st.number_input("Cap bonus descente (ex -0.08 = -8%)", value=-0.08, step=0.01, format="%.2f")

elev_smooth_window = st.slider("Lissage altitude (fenêtre)", 1, 51, 11, 2)

st.subheader("🧯 Anti-extrêmes pente")
colp1, colp2, colp3, colp4 = st.columns(4)
with colp1:
    g0_up_pct = st.number_input("Saturation montée g0 (%)", value=3.0, step=0.5)
with colp2:
    g0_down_pct = st.number_input("Saturation descente g0 (%)", value=2.5, step=0.5)
with colp3:
    max_grade_up = st.number_input("Cap pente montée (+)", value=0.30, step=0.05, format="%.2f")
with colp4:
    max_grade_down = st.number_input("Cap pente descente (-)", value=-0.06, step=0.02, format="%.2f")

grade_power = st.slider("Damping pente (puissance)", 0.2, 1.0, 0.85, 0.05)

st.subheader("🌡️ Température (réaliste, douce, plafonnée)")
apply_temp = st.checkbox("Appliquer température", value=True)
colt1, colt2, colt3 = st.columns(3)
with colt1:
    opt_temp = st.number_input("Temp optimale (°C)", value=12.0, step=0.5)
with colt2:
    cold_quad = st.number_input("Froid (quad)", value=0.0012, step=0.0002, format="%.4f")
with colt3:
    hot_quad = st.number_input("Chaud (quad)", value=0.0016, step=0.0002, format="%.4f")
temp_max_penalty = st.slider("Cap pénalité temp", 0.00, 0.15, 0.08, 0.01)
temp_power = st.slider("Damping temp (puissance)", 0.2, 1.2, 1.0, 0.05)

st.subheader("💨 Vent (réaliste : dépend de l’allure, caps faibles)")
apply_wind = st.checkbox("Appliquer le vent", value=True)
wind_mode = st.selectbox("Mode vent", ["Lissé (km/km)", "Global (un seul effet sur la course)"], index=0)
colw1, colw2, colw3 = st.columns(3)
with colw1:
    wind_smooth_window_km = st.slider("Lissage vent (km)", 1, 11, 5, 2)
with colw2:
    drag_coeff = st.number_input("drag_coeff (amplitude)", value=0.012, step=0.002, format="%.3f")
with colw3:
    tail_credit = st.slider("Crédit tailwind (fraction)", 0.0, 0.8, 0.35, 0.05)

colw4, colw5, colw6 = st.columns(3)
with colw4:
    wind_cap_head = st.slider("Cap pénalité vent (+)", 0.00, 0.20, 0.10, 0.01)
with colw5:
    wind_cap_tail = st.slider("Cap bonus vent (-)", -0.10, 0.00, -0.04, 0.01)
with colw6:
    wind_power = st.slider("Damping vent (puissance)", 0.2, 1.2, 1.0, 0.05)

st.subheader("📉 Affichage allure lissée")
show_smoothed_pace = st.checkbox("Afficher allure lissée (médiane)", value=True)
smooth_pace_window_km = st.slider("Fenêtre lissage allure (km)", 1, 9, 3, 2) if show_smoothed_pace else 3

st.subheader("📅 Course")
col1, col2 = st.columns(2)
with col1:
    date_course = st.date_input("Date", value=date.today())
with col2:
    heure_course = st.time_input("Heure départ", value=time(9, 0))

st.info("Météo : Open-Meteo forecast (horaire) interpolé + vent orienté (head/tail) stabilisé (lissé ou global).")

# Fatigue
st.header("3️⃣ bis. Fatigue (option)")
fatigue_active = st.checkbox("Activer fatigue ?", value=False)
fatigue_rate = st.slider("Régression finale (%)", 0.0, 30.0, 5.0, 0.5) if fatigue_active else 0.0

st.markdown("---")
st.header("4️⃣ Calcul & Comparaison")

colf1, colf2 = st.columns(2)
with colf1:
    force_distance_checkbox = st.checkbox("Forcer distance ?", value=False)
    if "dist_forced" not in st.session_state:
        st.session_state["dist_forced"] = 42.195
    distance_forced_km = st.number_input("Distance forcée (km)", value=float(st.session_state["dist_forced"]),
                                         format="%.3f", key="dist_forced") if force_distance_checkbox else None
with colf2:
    force_time_checkbox = st.checkbox("Forcer temps objectif ?", value=False)
    if "time_forced" not in st.session_state:
        st.session_state["time_forced"] = "3:30:00"
    time_forced_hms = st.text_input("Temps objectif (h:mm:ss)", value=str(st.session_state["time_forced"]),
                                    key="time_forced") if force_time_checkbox else None

if st.button("▶️ Calculer prédiction"):
    if not gpx_file or points is None:
        st.error("Importe un fichier GPX d'abord.")
    else:
        try:
            dist_target = distance_forced_km if (force_distance_checkbox and distance_forced_km) else None
            res = run_prediction_df(
                distance_cible_km=dist_target,
                refs_input=refs_raw,
                points=points,
                date_course_local=date_course,
                heure_course_local=heure_course,
                ideal_refs=ideal_refs,

                apply_grade=apply_grade,
                grade_k_up=grade_k_up,
                grade_k_down=grade_k_down,
                grade_down_cap=grade_down_cap,
                g0_up_pct=g0_up_pct,
                g0_down_pct=g0_down_pct,
                max_grade_up=max_grade_up,
                max_grade_down=max_grade_down,
                elev_smooth_window=elev_smooth_window,
                grade_power=grade_power,

                apply_temp=apply_temp,
                opt_temp=opt_temp,
                cold_quad=cold_quad,
                hot_quad=hot_quad,
                temp_max_penalty=temp_max_penalty,
                temp_power=temp_power,

                apply_wind=apply_wind,
                wind_mode=wind_mode,
                wind_smooth_window_km=wind_smooth_window_km,
                drag_coeff=drag_coeff,
                tail_credit=tail_credit,
                wind_cap_head=wind_cap_head,
                wind_cap_tail=wind_cap_tail,
                wind_power=wind_power,

                elev_ref_power=elev_ref_power,
                temp_ref_power=temp_ref_power,

                apply_fatigue=fatigue_active,
                fatigue_rate=fatigue_rate,

                objective_time_hms=time_forced_hms if force_time_checkbox else None,

                show_smoothed_pace=show_smoothed_pace,
                smooth_pace_window_km=smooth_pace_window_km
            )
            st.session_state["res"] = res
            st.success("Prédiction calculée ✅")
        except Exception as e:
            st.error(f"Erreur : {e}")

if "res" in st.session_state:
    res = st.session_state["res"]
    st.subheader("📈 Résultat")
    avg_pace = res["total_seconds"] / max(res["distance_gpx_km"], 1e-6)
    st.write(f"Distance GPX détectée : {res['distance_gpx_km']:.3f} km")
    st.write(f"Temps total : {res['total_human']} ({pace_seconds_to_str_per_km(avg_pace)}/km)")
    st.dataframe(res["df"], use_container_width=True)

# -------------------------
# CARTE & PROFIL
# -------------------------
if gpx_file and points:
    try:
        df_points = gpx_to_df(points)

        st.subheader("🗺️ Carte & Profil (GPX)")

        view = pdk.ViewState(
            latitude=float(df_points.lat.mean()),
            longitude=float(df_points.lon.mean()),
            zoom=13,
            pitch=0
        )
        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": df_points[["lon", "lat"]].values.tolist(), "name": "Parcours"}],
            get_path="path",
            get_color=[255, 0, 0],
            width_min_pixels=4
        )
        deck = pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=view,
            layers=[path_layer],
            tooltip={"text": "{name}"}
        )
        st.pydeck_chart(deck, use_container_width=True)

        st.subheader("📊 Profil d'altitude")
        plt.figure(figsize=(10, 4))

        total_m = 0.0
        cumdists = [0.0]
        for i in range(1, len(points)):
            d_ = SimplePoint(points[i - 1].latitude, points[i - 1].longitude, getattr(points[i - 1], "elevation", 0.0)).distance_3d(
                SimplePoint(points[i].latitude, points[i].longitude, getattr(points[i], "elevation", 0.0))
            )
            total_m += d_
            cumdists.append(total_m)

        x_km = np.array(cumdists) / 1000.0
        y_elev = np.array([p.elevation or 0.0 for p in points], dtype=float)

        w = int(elev_smooth_window)
        if w >= 3 and y_elev.size >= w:
            if w % 2 == 0:
                w += 1
            kernel = np.ones(w) / w
            y_s = np.convolve(y_elev, kernel, mode="same")
            plt.plot(x_km, y_s, lw=2, label="Altitude (lissée)")
            plt.plot(x_km, y_elev, lw=1, alpha=0.35, label="Altitude brute")
            plt.legend()
        else:
            plt.plot(x_km, y_elev, lw=2)

        plt.xlabel("Distance (km)")
        plt.ylabel("Altitude (m)")
        plt.title("Profil d'altitude du parcours")
        plt.grid(alpha=0.3)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Impossible d'afficher la carte/profil : {e}")
