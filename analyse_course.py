# analyse_course_refactor.py
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
# CONFIG
# -------------------------
st.set_page_config(page_title="Prédiction course route (refactor)", layout="wide")
st.title("🏃‍♂️ Analyse & Prédiction de course — Refactorisé")

# ============================================================
# MÉTÉO
#   - Open-Meteo forecast : Jour J (segment par segment)
#   - Open-Meteo archive  : historique pour les références FIT/TCX
# ============================================================

OW_API_KEY = st.secrets.get("openweather", {}).get("api_key", "")

@st.cache_data(show_spinner=False)
def get_weather_openmeteo_minutely(lat, lon, dt_local_naive, tz_name="Europe/Paris"):
    """
    Météo future : interpolation à la minute à partir d'un forecast horaire Open-Meteo.
    dt_local_naive : datetime naive supposé dans tz_name.

    Ajout: wind_direction_10m (degrés, direction "FROM" = d'où vient le vent).
    Interpolation circulaire correcte sur 0..360.
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
        hums  = data["hourly"]["relativehumidity_2m"]
        wdirs = data["hourly"]["wind_direction_10m"]  # degrés "FROM"

        dt = dt_local_naive

        before = None
        after = None
        for i in range(len(times) - 1):
            if times[i] <= dt <= times[i + 1]:
                before = (times[i], temps[i], winds[i], hums[i], wdirs[i])
                after  = (times[i + 1], temps[i + 1], winds[i + 1], hums[i + 1], wdirs[i + 1])
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
        hum_interp  = hum1  + ratio * (hum2  - hum1)

        # interpolation circulaire (0..360) sur le plus court chemin
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
def get_weather_openmeteo_day(lat, lon, date_obj, tz_name="Europe/Paris"):
    """
    Archive météo (jour complet).
    Retourne times, temps, winds, hums, wdirs (times en tz_name, naive).
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
        hums  = data["hourly"]["relativehumidity_2m"]
        wdirs = data["hourly"]["wind_direction_10m"]
        return times, temps, winds, hums, wdirs
    except Exception:
        return None


def get_avg_weather_for_period(lat, lon, start_dt, end_dt, tz_name="Europe/Paris"):
    """
    Météo moyenne robuste sur un intervalle.
    (Garde la même signature/retour que ton code: temp, wind, humidity.)
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
    selH = [H for t, H in zip(times, hums)  if start_dt <= t <= end_dt]

    if not selT:
        idx = min(range(len(times)), key=lambda i: abs(times[i] - start_dt))
        return float(temps[idx]), float(winds[idx]), float(hums[idx])

    return float(np.mean(selT)), float(np.mean(selW)), float(np.mean(selH))


# -------------------------
# UTILITAIRES
# -------------------------
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


# -------- bearing + angle helpers (vent orienté) --------
def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """
    Bearing (cap) en degrés 0..360, de (lat1,lon1) vers (lat2,lon2).
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0


def smallest_angle_diff_deg(a, b) -> float:
    """
    Différence angulaire signée (b-a) dans [-180, +180].
    """
    return (b - a + 540.0) % 360.0 - 180.0


def wind_multiplier_along_course(
    wind_speed_ms: float,
    wind_dir_from_deg: float,
    course_bearing_deg_: float,
    k_head: float = 0.025,   # pénalité par m/s de headwind "plein"
    k_tail: float = 0.010,   # bonus par m/s de tailwind "plein"
    cap_head: float = 0.25,  # max +25%
    cap_tail: float = -0.08  # max -8%
):
    """
    Open-Meteo wind_dir_* = direction "FROM" (d'où vient le vent).
    On convertit en direction "TO" (où va le vent) = +180°.
    On projette le vent sur l'axe du déplacement (course_bearing).

    Retour: (mult, headwind_ms, tailwind_ms)
    """
    if wind_speed_ms is None or wind_dir_from_deg is None or course_bearing_deg_ is None:
        return 1.0, 0.0, 0.0

    ws = float(wind_speed_ms)
    if ws <= 0:
        return 1.0, 0.0, 0.0

    wind_to = (float(wind_dir_from_deg) + 180.0) % 360.0
    delta = math.radians(smallest_angle_diff_deg(course_bearing_deg_, wind_to))
    along = ws * math.cos(delta)  # >0 tailwind, <0 headwind

    tail = max(0.0, along)
    head = max(0.0, -along)

    mult = 1.0 + float(k_head) * head - float(k_tail) * tail
    mult = min(1.0 + float(cap_head), mult)
    mult = max(1.0 + float(cap_tail), mult)
    return float(mult), float(head), float(tail)
# ------------------------------------------------------------


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


# -------------------------
# Modèle & facteurs
# -------------------------
def temp_multiplier_nonlin(temp, opt_temp=12.0, k_hot=0.002, k_cold=0.002):
    try:
        if temp is None:
            return 1.0
        diff = float(temp) - float(opt_temp)
        if diff > 0:
            mult = 1.0 + float(k_hot) * diff
        else:
            mult = 1.0 + float(k_cold) * (-diff)
        return max(0.1, mult)
    except Exception:
        return 1.0


def grade_multiplier(grade_pct, k_up=12.0, k_down=6.0, down_cap=-0.10):
    """
    Mult basé sur la PENTE (%) et non le D+ brut.
    - grade_pct : (delta_elev / distance) * 100
    """
    try:
        g = float(grade_pct) / 100.0
        if g >= 0:
            return max(0.01, 1.0 + k_up * g)
        else:
            bonus = -k_down * g  # g négatif => bonus positif
            bonus = min(bonus, abs(down_cap))  # cap
            return max(0.01, 1.0 - bonus)
    except Exception:
        return 1.0


# -------- NEW: facteur pente refs cohérent avec grade_multiplier --------
def elev_factor_from_dplus_dminus(
    D_up_m: float,
    D_down_m: float,
    distance_m: float,
    grade_k_up: float,
    grade_k_down: float,
    grade_down_cap: float
) -> float:
    """
    Approximation cohérente avec grade_multiplier, pour normaliser une ref
    quand on n'a que D+/D- global :
      factor ≈ 1 + grade_k_up*(D+/dist) - min(grade_k_down*(D-/dist), abs(down_cap))

    => même sémantique de paramètres entre refs et course.
    """
    dist = max(1e-6, float(distance_m))
    dup = max(0.0, float(D_up_m))
    ddn = max(0.0, float(D_down_m))

    up_term = float(grade_k_up) * (dup / dist)

    down_bonus = float(grade_k_down) * (ddn / dist)
    max_bonus = abs(float(grade_down_cap))
    down_bonus = min(down_bonus, max_bonus)

    factor = 1.0 + up_term - down_bonus
    return max(0.01, float(factor))
# ----------------------------------------------------------------------


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


def recalibrate_ref_to_ideal(
    ref,
    k_up, k_down,  # conservés pour compat (mode legacy)
    k_temp_hot, k_temp_cold, opt_temp,
    # NEW: unification pente refs/course
    use_unified_grade_for_refs: bool = True,
    grade_k_up: float = 12.0,
    grade_k_down: float = 6.0,
    grade_down_cap: float = -0.10
):
    secs = hms_to_seconds(ref.get("temps")) if ref.get("temps") is not None else 0

    D_up = safe_float(ref.get("D_up", 0.0))
    D_down = safe_float(ref.get("D_down", 0.0))
    seg_len = safe_float(ref.get("distance", 1000.0))
    seg_len = seg_len if seg_len > 0 else 1000.0

    # pente refs
    if use_unified_grade_for_refs:
        factor_elev = elev_factor_from_dplus_dminus(
            D_up_m=D_up,
            D_down_m=D_down,
            distance_m=seg_len,
            grade_k_up=grade_k_up,
            grade_k_down=grade_k_down,
            grade_down_cap=grade_down_cap
        )
    else:
        # ancien mode (legacy)
        up_factor = (k_up - 1.0) * (D_up / seg_len)
        down_factor = (1.0 - k_down) * (D_down / seg_len)
        factor_elev = 1.0 + up_factor + down_factor
        if factor_elev == 0:
            factor_elev = 1.0

    secs_no_elev = secs / factor_elev

    # température
    temp_real = ref.get("avg_temp")
    if temp_real is not None:
        mult_real = temp_multiplier_nonlin(temp_real, opt_temp=opt_temp, k_hot=k_temp_hot, k_cold=k_temp_cold)
        secs_no_temp = secs_no_elev / mult_real if mult_real != 0 else secs_no_elev
    else:
        secs_no_temp = secs_no_elev

    mult_opt = temp_multiplier_nonlin(opt_temp, opt_temp=opt_temp, k_hot=k_temp_hot, k_cold=k_temp_cold)
    secs_ideal = secs_no_temp * mult_opt
    return max(0.0, secs_ideal)


def recalibrate_ref_using_current(
    ref,
    k_up, k_down,  # conservés pour compat
    k_temp_hot, k_temp_cold, opt_temp,
    assumed_temp=None,
    # NEW: unification pente refs/course
    use_unified_grade_for_refs: bool = True,
    grade_k_up: float = 12.0,
    grade_k_down: float = 6.0,
    grade_down_cap: float = -0.10
):
    secs = hms_to_seconds(ref.get("temps")) if ref.get("temps") is not None else 0
    D_up = safe_float(ref.get("D_up", 0.0))
    D_down = safe_float(ref.get("D_down", 0.0))
    seg_len = safe_float(ref.get("distance", 1000.0))
    seg_len = seg_len if seg_len > 0 else 1000.0

    if use_unified_grade_for_refs:
        factor_elev = elev_factor_from_dplus_dminus(
            D_up_m=D_up,
            D_down_m=D_down,
            distance_m=seg_len,
            grade_k_up=grade_k_up,
            grade_k_down=grade_k_down,
            grade_down_cap=grade_down_cap
        )
    else:
        up_factor = (k_up - 1.0) * (D_up / seg_len)
        down_factor = (1.0 - k_down) * (D_down / seg_len)
        factor_elev = 1.0 + up_factor + down_factor
        if factor_elev == 0:
            factor_elev = 1.0

    secs_no_elev = secs / factor_elev
    if assumed_temp is None:
        return max(0.0, secs_no_elev)

    mult_temp = temp_multiplier_nonlin(assumed_temp, opt_temp=opt_temp, k_hot=k_temp_hot, k_cold=k_temp_cold)
    if mult_temp == 0:
        mult_temp = 1.0
    return max(0.0, secs_no_elev / mult_temp)


def prepare_refs_for_fit(
    refs_input,
    k_up, k_down,
    k_temp_hot, k_temp_cold, opt_temp,
    ideal_refs=False,
    # NEW: unification pente refs/course
    use_unified_grade_for_refs=True,
    grade_k_up=12.0,
    grade_k_down=6.0,
    grade_down_cap=-0.10
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
                k_up, k_down,
                k_temp_hot, k_temp_cold, opt_temp,
                use_unified_grade_for_refs=use_unified_grade_for_refs,
                grade_k_up=grade_k_up,
                grade_k_down=grade_k_down,
                grade_down_cap=grade_down_cap
            )
        else:
            secs_recal = recalibrate_ref_using_current(
                ref_for_calib,
                k_up, k_down,
                k_temp_hot, k_temp_cold, opt_temp,
                assumed_temp=None,
                use_unified_grade_for_refs=use_unified_grade_for_refs,
                grade_k_up=grade_k_up,
                grade_k_down=grade_k_down,
                grade_down_cap=grade_down_cap
            )

        prepared.append({"distance": float(d), "temps": float(secs_recal)})
    return prepared


# -------------------------
# Parsers GPX / FIT / TCX
# -------------------------
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


def parse_fit(file, tz_name="Europe/Paris"):
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

            dt_local = None
            if isinstance(ts, datetime):
                dt_local = ts.replace(tzinfo=None)

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


def parse_tcx(file, tz_name="Europe/Paris"):
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


# -------------------------
# PRÉDICTION PRINCIPALE (pente + météo + vent orienté + fatigue)
# -------------------------
def run_prediction_df(
    distance_cible_km,
    refs_input,
    points,
    date_course_local,
    heure_course_local,
    ideal_refs=False,
    apply_grade=True,
    apply_temp=True,
    apply_fatigue=True,
    objective_time_hms=None,
    # paramètres recalibrage refs (legacy)
    k_up=1.040, k_down=0.996,
    # paramètres météo température
    k_temp_hot=0.002, k_temp_cold=0.002, opt_temp=12.0,
    # paramètres pente (%)
    grade_k_up=12.0, grade_k_down=6.0, grade_down_cap=-0.10,
    # NEW: unification pente refs/course
    use_unified_grade_for_refs=True,
    # vent orienté
    apply_wind=True,
    k_wind_head=0.025,
    k_wind_tail=0.010,
    wind_cap_head=0.25,
    wind_cap_tail=-0.08,
    # fatigue
    fatigue_rate=0.0,
    tz_name="Europe/Paris",
    # lissage altitude
    elev_smooth_window=11,
):
    if not points or len(points) < 2:
        raise ValueError("GPX invalide ou trop court.")

    # distances cumulées (3D)
    total_m = 0.0
    cum = [0.0]
    for i in range(1, len(points)):
        d = SimplePoint(points[i - 1].latitude, points[i - 1].longitude, getattr(points[i - 1], "elevation", 0.0)).distance_3d(
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

    # lissage altitude pour pente
    if elev_smooth_window and elev_smooth_window >= 3 and elev_list.size >= elev_smooth_window:
        w = int(elev_smooth_window)
        if w % 2 == 0:
            w += 1
        kernel = np.ones(w) / w
        elev_smooth = np.convolve(elev_list, kernel, mode="same")
    else:
        elev_smooth = elev_list

    # refs fit (NEW: unification pente)
    refs_for_fit = prepare_refs_for_fit(
        refs_input,
        k_up=k_up, k_down=k_down,
        k_temp_hot=k_temp_hot, k_temp_cold=k_temp_cold, opt_temp=opt_temp,
        ideal_refs=ideal_refs,
        use_unified_grade_for_refs=use_unified_grade_for_refs,
        grade_k_up=grade_k_up,
        grade_k_down=grade_k_down,
        grade_down_cap=grade_down_cap
    )

    a, K = fit_loglog_model(refs_for_fit)

    a_override = None
    if objective_time_hms:
        a_override = override_with_objective(int(distance_cible_km * 1000), objective_time_hms, K)

    baseline_a = a_override if a_override is not None else a

    distance_cible_m = int(distance_cible_km * 1000)
    base_flat_total = predict_time_flat(distance_cible_m, baseline_a, K)
    base_s_per_km_flat = base_flat_total / max(distance_cible_km, 1e-9)

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

    segment_infos = []
    cum_time_temp = 0.0
    dt_depart = datetime.combine(date_course_local, heure_course_local)

    for i, d in enumerate(km_marks):
        # segment length
        seg_length_m = 1000.0
        if i == len(km_marks) - 1 and last_seg > 1e-6:
            seg_length_m = (d - km_marks[-2]) if len(km_marks) >= 2 else d

        # altitudes (lissées pour pente)
        e_cur = float(np.interp(d, dists_corr, elev_smooth))
        e_prev_d = max(d - seg_length_m, 0.0)
        e_prev = float(np.interp(e_prev_d, dists_corr, elev_smooth)) if i > 0 else e_cur

        delta_e = e_cur - e_prev
        grade_pct = (delta_e / max(1e-6, seg_length_m)) * 100.0

        # D+/D- (pour affichage)
        d_up = max(0.0, delta_e)
        d_down = max(0.0, -delta_e)

        # temps plat théorique
        t_flat = base_s_per_km_flat * (seg_length_m / 1000.0)

        # pente (%)
        if apply_grade:
            g_mult = grade_multiplier(grade_pct, k_up=grade_k_up, k_down=grade_k_down, down_cap=grade_down_cap)
            t_after_grade = t_flat * g_mult
        else:
            g_mult = 1.0
            t_after_grade = t_flat

        # fatigue linéaire
        if apply_fatigue and fatigue_rate > 0 and total_corr > 0:
            progression = d / total_corr
            t_after_fatigue = t_after_grade * (1.0 + (fatigue_rate / 100.0) * progression)
        else:
            t_after_fatigue = t_after_grade

        # passage au milieu du segment
        passage_dt = dt_depart + timedelta(seconds=cum_time_temp + t_after_fatigue / 2.0)

        # lat/lon fin segment
        lat_seg = float(np.interp(d, dists_corr, df_points["lat"].values))
        lon_seg = float(np.interp(d, dists_corr, df_points["lon"].values))

        # lat/lon début segment (pour bearing)
        d_start = max(d - seg_length_m, 0.0)
        lat_start = float(np.interp(d_start, dists_corr, df_points["lat"].values))
        lon_start = float(np.interp(d_start, dists_corr, df_points["lon"].values))
        course_bearing = bearing_deg(lat_start, lon_start, lat_seg, lon_seg)

        meteo = get_weather_openmeteo_minutely(lat_seg, lon_seg, passage_dt, tz_name=tz_name)
        temp_here = meteo["temp"] if meteo else None
        wind_here = meteo["wind"] if meteo else None
        hum_here = meteo["humidity"] if meteo else None
        wind_dir_here = meteo.get("wind_dir") if meteo else None

        # température
        if apply_temp and temp_here is not None:
            temp_mult = temp_multiplier_nonlin(temp_here, opt_temp=opt_temp, k_hot=k_temp_hot, k_cold=k_temp_cold)
            t_after_temp = t_after_fatigue * temp_mult
        else:
            temp_mult = 1.0
            t_after_temp = t_after_fatigue

        # vent orienté
        if apply_wind and wind_here is not None and wind_dir_here is not None:
            wind_mult, head_ms, tail_ms = wind_multiplier_along_course(
                wind_speed_ms=wind_here,
                wind_dir_from_deg=wind_dir_here,
                course_bearing_deg_=course_bearing,
                k_head=k_wind_head,
                k_tail=k_wind_tail,
                cap_head=wind_cap_head,
                cap_tail=wind_cap_tail
            )
            t_after_wind = t_after_temp * wind_mult
        else:
            wind_mult, head_ms, tail_ms = 1.0, 0.0, 0.0
            t_after_wind = t_after_temp

        segment_infos.append({
            "idx": i,
            "d": float(d),
            "seg_length_m": float(seg_length_m),
            "grade_pct": float(grade_pct),
            "grade_mult": float(g_mult),
            "d_up": float(d_up),
            "d_down": float(d_down),

            "temp": temp_here,
            "wind": wind_here,
            "humidity": hum_here,

            "wind_dir": wind_dir_here,
            "course_bearing": float(course_bearing),
            "wind_mult": float(wind_mult),
            "head_ms": float(head_ms),
            "tail_ms": float(tail_ms),

            "temp_mult": float(temp_mult),
            "t_raw": float(t_after_wind),
        })

        cum_time_temp += float(t_after_wind)

    # scale si objectif temps
    if objective_time_hms:
        objective_seconds = hms_to_seconds(objective_time_hms)
        sum_raw = sum(s["t_raw"] for s in segment_infos)
        scale = (objective_seconds / sum_raw) if sum_raw > 0 else 1.0
    else:
        scale = 1.0

    # dataframe résultats
    results = []
    cum_time = 0.0
    for seg in segment_infos:
        t_seg = seg["t_raw"] * scale
        cum_time += t_seg
        pace_per_km = (t_seg / seg["seg_length_m"]) * 1000.0 if seg["seg_length_m"] > 0 else t_seg

        km_label = (seg["idx"] + 1) if seg["seg_length_m"] >= 1000 - 1e-6 else f"{seg['idx']+1} ({seg['seg_length_m']:.0f}m)"

        results.append({
            "Km": km_label,
            "Pente (%)": round(seg["grade_pct"], 2),
            "Mult Pente": round(seg["grade_mult"], 4),
            "D+ (m)": round(seg["d_up"], 1),
            "D- (m)": round(seg["d_down"], 1),

            "Temp (°C)": round(seg["temp"], 1) if seg["temp"] is not None else None,
            "Vent (m/s)": round(seg["wind"], 1) if seg["wind"] is not None else None,
            "Dir vent (° FROM)": round(seg["wind_dir"], 0) if seg["wind_dir"] is not None else None,
            "Cap seg (°)": round(seg["course_bearing"], 0) if seg.get("course_bearing") is not None else None,
            "Headwind (m/s)": round(seg["head_ms"], 2) if seg.get("head_ms") is not None else None,
            "Tailwind (m/s)": round(seg["tail_ms"], 2) if seg.get("tail_ms") is not None else None,
            "Humidité (%)": round(seg["humidity"], 1) if seg["humidity"] is not None else None,

            "Mult Temp": round(seg["temp_mult"], 4),
            "Mult Vent": round(seg["wind_mult"], 4),
            "Temps segment (s)": round(t_seg, 1),
            "Allure (min/km)": pace_seconds_to_str_per_km(pace_per_km),
            "Temps cumulé": seconds_to_hms(cum_time),
        })

    df = pd.DataFrame(results)
    total_seconds = sum(s["t_raw"] for s in segment_infos) * scale

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
                avg_hum_ref  = fit_data["avg_humidity"]

        elif filename.endswith(".tcx"):
            tcx_data = parse_tcx(file_in)
            if tcx_data:
                dist = tcx_data["distance"]
                dup = tcx_data["D_up"]
                ddn = tcx_data["D_down"]
                duration_hms_file = tcx_data["duration_hms"]
                avg_temp_ref = tcx_data["avg_temp"]
                avg_wind_ref = tcx_data["avg_wind"]
                avg_hum_ref  = tcx_data["avg_humidity"]

        # découpe
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
st.header("3️⃣ Paramètres modèle")

c1, c2 = st.columns(2)
with c1:
    use_elev_coeff = st.checkbox("Activer correction D+/D- pour normaliser les références (plat)", value=True)
    if use_elev_coeff:
        k_up = st.number_input("k_up (refs legacy)", value=1.040, format="%.3f", step=0.001)
        k_down = st.number_input("k_down (refs legacy)", value=0.996, format="%.3f", step=0.001)
    else:
        k_up, k_down = 1.0, 1.0

with c2:
    use_temp_coeff = st.checkbox("Activer température (réfs + course)", value=True)
    if use_temp_coeff:
        k_temp_hot = st.number_input("k_temp_hot", value=0.002, format="%.4f", step=0.0005)
        k_temp_cold = st.number_input("k_temp_cold", value=0.002, format="%.4f", step=0.0005)
        opt_temp = st.number_input("Temp optimale (°C)", value=12.0, format="%.1f", step=0.5)
    else:
        k_temp_hot, k_temp_cold, opt_temp = 0.0, 0.0, 12.0

st.subheader("🎢 Pente (course objectif)")
apply_grade = st.checkbox("Prendre en compte la pente (%) du GPX (recommandé)", value=True)
colg1, colg2, colg3 = st.columns(3)
with colg1:
    grade_k_up = st.number_input("Sensibilité montée (pente)", value=12.0, format="%.1f", step=0.5)
with colg2:
    grade_k_down = st.number_input("Sensibilité descente (pente)", value=6.0, format="%.1f", step=0.5)
with colg3:
    grade_down_cap = st.number_input("Cap bonus descente (ex -0.10 = -10%)", value=-0.10, format="%.2f", step=0.01)

elev_smooth_window = st.slider("Lissage altitude (impact pente) - fenêtre", 1, 51, 11, 2)

# NEW: cohérence pente refs/course
st.subheader("⛰️ Cohérence pente Réfs ↔ Course")
use_unified_grade_for_refs = st.checkbox(
    "Utiliser le même modèle de pente (grade%) pour normaliser les références",
    value=True
)
st.caption("Si décoché: utilise l'ancien mode legacy k_up/k_down basé sur D+/D- (moins cohérent).")

# Vent orienté
st.subheader("💨 Vent (orienté par le GPX)")
apply_wind = st.checkbox("Prendre en compte le vent (head/tail selon orientation)", value=True)
colw1, colw2, colw3, colw4 = st.columns(4)
with colw1:
    k_wind_head = st.number_input("k headwind (par m/s)", value=0.025, format="%.3f", step=0.005)
with colw2:
    k_wind_tail = st.number_input("k tailwind (par m/s)", value=0.010, format="%.3f", step=0.005)
with colw3:
    wind_cap_head = st.number_input("Cap pénalité (+)", value=0.25, format="%.2f", step=0.05)
with colw4:
    wind_cap_tail = st.number_input("Cap bonus (-)", value=-0.08, format="%.2f", step=0.02)

col1, col2 = st.columns(2)
with col1:
    date_course = st.date_input("Date de la course (Jour J)", value=date.today())
with col2:
    heure_course = st.time_input("Heure de départ (Jour J)", value=time(9, 0))

st.info("Météo: Open-Meteo forecast par segment (température, vent, direction du vent, humidité).")

# Recalibrage refs
st.subheader("⏱️ Références recalibrées (plat & T° opt)")
refs_calibrated = []
for r in refs_raw:
    t_brut = hms_to_seconds(r["temps"])
    t_ideal = recalibrate_ref_to_ideal(
        ref=r,
        k_up=k_up, k_down=k_down,
        k_temp_hot=k_temp_hot, k_temp_cold=k_temp_cold,
        opt_temp=opt_temp,
        use_unified_grade_for_refs=(use_unified_grade_for_refs and apply_grade),
        grade_k_up=grade_k_up,
        grade_k_down=grade_k_down,
        grade_down_cap=grade_down_cap
    )
    refs_calibrated.append({
        "distance": r["distance"],
        "D_up": r["D_up"],
        "D_down": r["D_down"],
        "temps_brut": t_brut,
        "temps_ideal": t_ideal,
        "temp_moy": r.get("avg_temp"),
        "vent_moy": r.get("avg_wind"),
        "hum_moy": r.get("avg_humidity"),
    })

df_refs = pd.DataFrame([{
    "Distance (m)": r["distance"],
    "D+ (m)": r["D_up"],
    "D- (m)": r["D_down"],
    "Temps brut": seconds_to_hms(r["temps_brut"]),
    "Temps conditions idéales": seconds_to_hms(r["temps_ideal"]),
    "Temp moy (°C)": r["temp_moy"],
    "Vent moy (m/s)": r["vent_moy"],
    "Hum moy (%)": r["hum_moy"],
} for r in refs_calibrated])
st.dataframe(df_refs, use_container_width=True)

# Fatigue
st.header("3️⃣ bis. Fatigue linéaire")
fatigue_active = st.checkbox("Activer fatigue ?", value=False)
fatigue_rate = st.slider("Régression finale (%)", 0.0, 30.0, 5.0, 0.5) if fatigue_active else 0.0

# Fit en conditions idéales ?
st.markdown("---")
ideal_refs = st.checkbox("Utiliser les références recalibrées (conditions idéales) pour le fit ?", value=True)

# -------------------------
# Calculs
# -------------------------
st.header("4️⃣ Calcul & Comparaison")

if st.button("▶️ Calculer prédiction (BASE, d'après références)"):
    if not gpx_file or points is None:
        st.error("Importe un fichier GPX d'abord.")
    else:
        try:
            res_base = run_prediction_df(
                distance_cible_km=None,
                refs_input=refs_raw,
                points=points,
                date_course_local=date_course,
                heure_course_local=heure_course,
                ideal_refs=ideal_refs,
                apply_grade=apply_grade,
                apply_temp=use_temp_coeff,
                apply_wind=apply_wind,
                apply_fatigue=fatigue_active,
                objective_time_hms=None,
                k_up=k_up, k_down=k_down,
                k_temp_hot=k_temp_hot, k_temp_cold=k_temp_cold, opt_temp=opt_temp,
                grade_k_up=grade_k_up, grade_k_down=grade_k_down, grade_down_cap=grade_down_cap,
                use_unified_grade_for_refs=(use_unified_grade_for_refs and apply_grade),
                k_wind_head=k_wind_head, k_wind_tail=k_wind_tail,
                wind_cap_head=wind_cap_head, wind_cap_tail=wind_cap_tail,
                fatigue_rate=fatigue_rate,
                elev_smooth_window=elev_smooth_window,
            )
            st.session_state["res_base"] = res_base
            st.success(f"Base calculée — distance GPX détectée: {res_base['distance_gpx_km']:.3f} km")
        except Exception as e:
            st.error(f"Erreur lors du calcul base : {e}")

st.markdown("---")
st.markdown("**Forcer distance et/ou temps objectif (tableau 'FORCÉ' distinct)**")
colf1, colf2 = st.columns(2)

with colf1:
    force_distance_checkbox = st.checkbox("Forcer distance ?", value=False)
    if "dist_forced" not in st.session_state:
        st.session_state["dist_forced"] = 42.195
    distance_forced_km = st.number_input(
        "Distance forcée (km)",
        value=float(st.session_state["dist_forced"]),
        format="%.3f",
        key="dist_forced",
    ) if force_distance_checkbox else None

with colf2:
    force_time_checkbox = st.checkbox("Forcer temps objectif ?", value=False)
    if "time_forced" not in st.session_state:
        st.session_state["time_forced"] = "3:30:00"
    time_forced_hms = st.text_input(
        "Temps objectif (h:mm:ss)",
        value=str(st.session_state["time_forced"]),
        key="time_forced",
    ) if force_time_checkbox else None

if st.button("📊 Calculer prédiction finale (FORCÉ si activé)"):
    if not gpx_file or points is None:
        st.error("Importe un fichier GPX d'abord.")
    else:
        dist_target = distance_forced_km if (force_distance_checkbox and distance_forced_km) else None
        try:
            res_forced = run_prediction_df(
                distance_cible_km=dist_target,
                refs_input=refs_raw,
                points=points,
                date_course_local=date_course,
                heure_course_local=heure_course,
                ideal_refs=ideal_refs,
                apply_grade=apply_grade,
                apply_temp=use_temp_coeff,
                apply_wind=apply_wind,
                apply_fatigue=fatigue_active,
                objective_time_hms=time_forced_hms if force_time_checkbox else None,
                k_up=k_up, k_down=k_down,
                k_temp_hot=k_temp_hot, k_temp_cold=k_temp_cold, opt_temp=opt_temp,
                grade_k_up=grade_k_up, grade_k_down=grade_k_down, grade_down_cap=grade_down_cap,
                use_unified_grade_for_refs=(use_unified_grade_for_refs and apply_grade),
                k_wind_head=k_wind_head, k_wind_tail=k_wind_tail,
                wind_cap_head=wind_cap_head, wind_cap_tail=wind_cap_tail,
                fatigue_rate=fatigue_rate,
                elev_smooth_window=elev_smooth_window,
            )
            st.session_state["res_forced"] = res_forced
            st.success("Prédiction forcée calculée ✅")
        except Exception as e:
            st.error(f"Erreur lors du calcul forcé : {e}")

# display
if "res_base" in st.session_state or "res_forced" in st.session_state:
    base = st.session_state.get("res_base", None)
    forced = st.session_state.get("res_forced", None)

    left, right = st.columns(2)

    with left:
        st.subheader("📈 Base")
        if base:
            avg_pace_base = base["total_seconds"] / max(base["distance_gpx_km"], 1e-6)
            st.write(f"Distance GPX: {base['distance_gpx_km']:.3f} km")
            st.write(f"Temps total: {base['total_human']} ({pace_seconds_to_str_per_km(avg_pace_base)}/km)")
            st.dataframe(base["df"], use_container_width=True)
        else:
            st.info("Clique sur 'Calculer prédiction (BASE)'.")

    with right:
        st.subheader("🎯 Forcé")
        if forced:
            dist_display = float(distance_forced_km) if (force_distance_checkbox and distance_forced_km) else float(forced["distance_gpx_km"])
            avg_pace_forced = forced["total_seconds"] / max(dist_display, 1e-6)
            st.write(f"Distance cible: {dist_display:.3f} km")
            st.write(f"Temps total: {forced['total_human']} ({pace_seconds_to_str_per_km(avg_pace_forced)}/km)")
            st.dataframe(forced["df"], use_container_width=True)
        else:
            st.info("Clique sur 'Calculer prédiction finale (FORCÉ)'.")

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

        if elev_smooth_window and elev_smooth_window >= 3 and y_elev.size >= elev_smooth_window:
            w = int(elev_smooth_window)
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
