# analyse_course_v2.py
# Streamlit app — prédiction de course améliorée
# NOUVEAUTÉS v2 :
#   1) WBGT / indice de chaleur (température ressentie = temp + humidité)
#   2) Effet altitude physiologique sur VO2 (> 1500m)
#   3) Correction altimétrique via Open-Topo-Data (DEM) — qualité GPS corrigée
#   4) Extraction FC depuis FIT (HR zones, drift cardiaque)
#   5) Modèle de pente Minetti et al. (2002) — base physique réelle
#   6) Fatigue basée sur D+ cumulé (exponentielle) plutôt que distance linéaire
#   7) Cross-validation leave-one-out des références
#   8) Tableau de fiabilité des références (HR drift, météo, terrain)
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
st.set_page_config(page_title="Prédiction course v2 (physio + météo avancée)", layout="wide")
st.title("🏃‍♂️ Analyse & Prédiction de course — v2 (modèles physiologiques)")
TZ_NAME_DEFAULT = "Europe/Paris"

# ============================================================
# NOUVEAUTÉ 1 : WBGT / Indice de chaleur
# ============================================================
def heat_index_celsius(T_c: float, RH: float) -> float:
    """
    Indice de chaleur (Steadman 1979, approximation NWS).
    Valide pour T > 27°C et RH > 40%. En dessous, retourne T_c.
    RH en % (0-100).
    """
    if T_c < 26.0 or RH < 40.0:
        return float(T_c)
    T_f = T_c * 9.0 / 5.0 + 32.0
    HI = (
        -42.379
        + 2.04901523 * T_f
        + 10.14333127 * RH
        - 0.22475541 * T_f * RH
        - 0.00683783 * T_f ** 2
        - 0.05481717 * RH ** 2
        + 0.00122874 * T_f ** 2 * RH
        + 0.00085282 * T_f * RH ** 2
        - 0.00000199 * T_f ** 2 * RH ** 2
    )
    return (HI - 32.0) * 5.0 / 9.0


def wbgt_simplified(T_c: float, RH: float) -> float:
    """
    Approximation WBGT (Wet Bulb Globe Temperature) en extérieur, soleil modéré.
    Formule : WBGT ≈ 0.7 * Tw + 0.2 * Tg + 0.1 * T_c
    Approximation simplifiée :
      Tw (température bulbe humide) ≈ T_c * atan(0.151977*(RH+8.313659)^0.5) + ...
    On utilise la formule de Stull (2011) pour Tw.
    """
    try:
        RH_c = max(0.0, min(100.0, float(RH)))
        T = float(T_c)
        # Formule de Stull (2011) — précise pour -20°C à 50°C, RH 5-99%
        Tw = (T * math.atan(0.151977 * (RH_c + 8.313659) ** 0.5)
              + math.atan(T + RH_c)
              - math.atan(RH_c - 1.676331)
              + 0.00391838 * RH_c ** 1.5 * math.atan(0.023101 * RH_c)
              - 4.686035)
        # Globe temperature Tg ≈ T + 2 (soleil modéré plein air)
        Tg = T + 2.0
        return 0.7 * Tw + 0.2 * Tg + 0.1 * T
    except Exception:
        return float(T_c)


def effective_temp_for_performance(T_c: float, RH: float, use_wbgt: bool = True) -> float:
    """
    Retourne la température "effective" à utiliser dans le modèle de pénalité.
    Si use_wbgt=True : WBGT (recommandé)
    Sinon : indice de chaleur classique
    """
    if use_wbgt:
        return wbgt_simplified(T_c, RH)
    else:
        return heat_index_celsius(T_c, RH)


# ============================================================
# NOUVEAUTÉ 2 : Effet altitude physiologique
# ============================================================
def altitude_vo2_multiplier(altitude_m: float, altitude_ref_m: float = 0.0) -> float:
    """
    Réduit la performance selon l'altitude (effet hypoxie).
    - En dessous de 1500m : effet négligeable (retourne 1.0)
    - Au-dessus : ~1% de dégradation par 100m (à partir de 1500m)
    - Cap à -25% (altitude extrême)
    Formule basée sur Chapman et al. (1998) et Wehrlin & Hallén (2006).
    """
    alt = max(0.0, float(altitude_m))
    alt_ref = max(0.0, float(altitude_ref_m))
    effective_alt = max(0.0, alt - max(1500.0, alt_ref))
    penalty = 0.01 * (effective_alt / 100.0)  # 1% par 100m au-dessus de 1500m
    penalty = min(0.25, penalty)
    return 1.0 + penalty  # multiplicateur (>1 = plus lent)


# ============================================================
# NOUVEAUTÉ 3 : DEM Open-Topo-Data (correction altitude GPS)
# ============================================================
@st.cache_data(show_spinner="Correction altimétrique DEM en cours...")
def fetch_dem_elevations(lats: tuple, lons: tuple, dataset: str = "srtm30m") -> list:
    """
    Récupère les altitudes corrigées via Open-Topo-Data (gratuit, max 100 pts/requête).
    dataset: 'srtm30m' (global, 30m) ou 'eudem25m' (Europe, 25m, plus précis)
    Retourne une liste d'altitudes (None si erreur).
    """
    try:
        locations = "|".join(f"{lat},{lon}" for lat, lon in zip(lats, lons))
        url = f"https://api.opentopodata.org/v1/{dataset}?locations={locations}"
        r = requests.get(url, timeout=30)
        data = r.json()
        if data.get("status") != "OK":
            return [None] * len(lats)
        return [res.get("elevation") for res in data["results"]]
    except Exception:
        return [None] * len(lats)


def correct_gpx_elevations_with_dem(points, max_points: int = 100, dataset: str = "srtm30m"):
    """
    Sous-échantillonne le tracé, récupère les altitudes DEM, interpole sur tous les points.
    Retourne un array numpy d'altitudes corrigées.
    """
    n = len(points)
    if n < 2:
        return np.array([getattr(p, "elevation", 0.0) or 0.0 for p in points])

    # Sous-échantillonnage
    step = max(1, n // max_points)
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    lats = tuple(points[i].latitude for i in indices)
    lons = tuple(points[i].longitude for i in indices)

    dem_elevs = fetch_dem_elevations(lats, lons, dataset=dataset)

    # Distances cumulées pour interpolation
    cum_all = [0.0]
    for i in range(1, n):
        d = haversine_m(points[i-1].latitude, points[i-1].longitude,
                        points[i].latitude, points[i].longitude)
        cum_all.append(cum_all[-1] + d)

    cum_sub = [cum_all[i] for i in indices]
    valid = [(d, e) for d, e in zip(cum_sub, dem_elevs) if e is not None]

    if len(valid) < 2:
        return np.array([getattr(p, "elevation", 0.0) or 0.0 for p in points])

    xs = [v[0] for v in valid]
    ys = [v[1] for v in valid]
    corrected = np.interp(cum_all, xs, ys)
    return corrected


# ============================================================
# MÉTÉO (Open-Meteo) — inchangée + utilisation WBGT
# ============================================================
@st.cache_data(show_spinner=False)
def get_weather_openmeteo_minutely(lat, lon, dt_local_naive, tz_name=TZ_NAME_DEFAULT):
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
        wdirs = data["hourly"]["wind_direction_10m"]
        dt = dt_local_naive
        before, after = None, None
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


def get_avg_weather_for_period(lat, lon, start_dt, end_dt, tz_name=TZ_NAME_DEFAULT):
    if start_dt is None or end_dt is None:
        return None, None, None
    if (end_dt - start_dt).total_seconds() < 300:
        start_dt -= timedelta(minutes=2)
        end_dt   += timedelta(minutes=2)
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
            h = 0; m = 0; s = parts[0]
        else:
            return timedelta(seconds=0)
        return timedelta(hours=h, minutes=m, seconds=s)
    except Exception:
        return timedelta(seconds=0)


def pace_seconds_to_str_per_km(seconds_per_km: float) -> str:
    if seconds_per_km is None or seconds_per_km <= 0 or math.isnan(seconds_per_km) or math.isinf(seconds_per_km):
        return "0:00"
    total = int(round(float(seconds_per_km)))
    m = total // 60
    s = total % 60
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
# NOUVEAUTÉ 5 : Modèle de pente Minetti et al. (2002)
# ============================================================
def minetti_cost_J_per_kg_per_m(grade_fraction: float) -> float:
    """
    Coût métabolique en J/kg/m en fonction de la pente (Minetti et al., 2002).
    Valide pour pentes entre -0.45 et +0.45 (±45%).
    Polynôme du 5e degré ajusté sur données expérimentales.
    """
    g = max(-0.45, min(0.45, float(grade_fraction)))
    c = (155.4 * g**5
         - 30.4 * g**4
         - 43.3 * g**3
         + 46.3 * g**2
         + 19.5 * g
         + 3.6)
    return max(0.1, float(c))


def minetti_multiplier(grade_pct: float) -> float:
    """
    Ratio coût sur pente / coût sur plat.
    Le coût sur plat = Minetti à g=0 = 3.6 J/kg/m.
    Retourne un multiplicateur >1 en montée, <1 en légère descente, >1 en forte descente.
    """
    flat_cost = minetti_cost_J_per_kg_per_m(0.0)  # 3.6
    grade_cost = minetti_cost_J_per_kg_per_m(float(grade_pct) / 100.0)
    mult = grade_cost / max(1e-9, flat_cost)
    # Caps physiologiques : cap montée à +35%, cap descente légère à -8%
    mult = min(1.35, max(0.92, mult))
    return float(mult)


# ============================================================
# MODÈLES EXISTANTS (pente heuristique, temp, vent)
# ============================================================
def grade_multiplier_nonlinear_capped(
    grade_pct,
    k_up=12.0, k_down=6.0, down_cap=-0.08, g0_up=3.0, g0_down=2.5,
    max_up=0.30, max_down=-0.06
):
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


def combined_grade_multiplier(grade_pct: float, use_minetti: bool, minetti_weight: float,
                               k_up=12.0, k_down=6.0, down_cap=-0.08, g0_up=3.0, g0_down=2.5,
                               max_up=0.30, max_down=-0.06) -> float:
    """
    Combinaison pondérée Minetti (physique) + heuristique (terrain réel).
    minetti_weight: 0.0 = 100% heuristique, 1.0 = 100% Minetti
    """
    if not use_minetti:
        return grade_multiplier_nonlinear_capped(grade_pct, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down)
    m_min = minetti_multiplier(grade_pct)
    m_heu = grade_multiplier_nonlinear_capped(grade_pct, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down)
    w = max(0.0, min(1.0, float(minetti_weight)))
    return w * m_min + (1.0 - w) * m_heu


def temp_multiplier_realistic(temp_c, opt_temp=12.0, cold_quad=0.0012, hot_quad=0.0016, max_penalty=0.08):
    """
    Modèle doux, plafonné. Prend la température effective (WBGT si activé).
    """
    try:
        if temp_c is None:
            return 1.0
        t = float(temp_c)
        d = t - float(opt_temp)
        pen = hot_quad * (d ** 2) if d >= 0 else cold_quad * ((-d) ** 2)
        pen = min(float(max_penalty), float(pen))
        return 1.0 + pen
    except Exception:
        return 1.0


def wind_components_along_course(wind_speed_ms, wind_dir_from_deg, course_bearing_deg_):
    if wind_speed_ms is None or wind_dir_from_deg is None or course_bearing_deg_ is None:
        return 0.0, 0.0, 0.0
    ws = float(wind_speed_ms)
    if ws <= 0:
        return 0.0, 0.0, 0.0
    wind_to = (float(wind_dir_from_deg) + 180.0) % 360.0
    delta = math.radians(smallest_angle_diff_deg(course_bearing_deg_, wind_to))
    along = ws * math.cos(delta)
    cross = ws * math.sin(delta)
    tail = max(0.0, along)
    head = max(0.0, -along)
    cross = abs(float(cross))
    return float(head), float(tail), float(cross)


def wind_multiplier_realistic(head_ms, tail_ms, pace_s_per_km, drag_coeff=0.012,
                               tail_credit=0.35, cap_head=0.10, cap_tail=-0.04):
    try:
        pace = max(150.0, float(pace_s_per_km))
        v_run = 1000.0 / pace
        w_along = float(head_ms) - float(tail_ms)
        v_rel = max(0.0, v_run + w_along)
        base = max(1e-9, v_run ** 2)
        extra = (v_rel ** 2 - v_run ** 2) / base
        if extra < 0:
            extra = float(tail_credit) * extra
        mult = 1.0 + float(drag_coeff) * float(extra)
        mult = min(1.0 + float(cap_head), mult)
        mult = max(1.0 + float(cap_tail), mult)
        return float(mult)
    except Exception:
        return 1.0


def wind_gate_from_grade(grade_pct: float, g1: float = 2.0, g2: float = 8.0, min_gate: float = 0.25) -> float:
    g = max(0.0, float(grade_pct))
    if g <= g1:
        return 1.0
    if g >= g2:
        return float(min_gate)
    x = (g - g1) / (g2 - g1)
    return float(1.0 - x * (1.0 - min_gate))


def cap_combined_multiplier(mult_total, grade_pct, base_cap=0.08, extra_per_pct=0.004, max_cap=0.18):
    g = max(0.0, float(grade_pct))
    cap = min(float(max_cap), float(base_cap) + float(extra_per_pct) * g)
    return min(float(mult_total), 1.0 + cap)


# ============================================================
# NOUVEAUTÉ 6 : Fatigue basée sur D+ cumulé (exponentielle)
# ============================================================
def fatigue_multiplier_cumulative(
    d_plus_cum: float,
    distance_cum: float,
    d_plus_total: float,
    distance_total: float,
    fatigue_rate_pct: float,
    fatigue_mode: str = "mixte"
) -> float:
    """
    Multiplicateur de fatigue basé sur la progression combinée distance+D+.
    fatigue_mode:
      - 'distance' : linéaire sur distance (ancien modèle)
      - 'd_plus'   : exponentiel sur D+ cumulé (montagne)
      - 'mixte'    : moyenne pondérée (recommandé)
    fatigue_rate_pct : ralentissement total en % à la fin (ex: 8.0 => +8% en fin de course)
    """
    if fatigue_rate_pct <= 0:
        return 1.0
    rate = float(fatigue_rate_pct) / 100.0
    dist_tot = max(1.0, float(distance_total))
    dplus_tot = max(1.0, float(d_plus_total))

    prog_dist = min(1.0, float(distance_cum) / dist_tot)
    prog_dplus = min(1.0, float(d_plus_cum) / dplus_tot) if dplus_tot > 0 else prog_dist

    if fatigue_mode == "distance":
        prog = prog_dist
    elif fatigue_mode == "d_plus":
        prog = prog_dplus
    else:  # mixte
        # Poids D+ plus fort si le parcours est montagneux (D+ > 5% de la distance)
        dplus_ratio = dplus_tot / dist_tot
        w_dplus = min(0.8, dplus_ratio * 10.0)  # 0% si plat, max 80% si très montagneux
        prog = w_dplus * prog_dplus + (1.0 - w_dplus) * prog_dist

    # Courbe exponentielle : fatigues d'abord lentes puis accélèrent
    # f(p) = e^(k*p) - 1 normalisé pour que f(1) = rate
    k = 2.0  # courbure fixe (plus élevé = fatigue qui frappe plus brutalement)
    factor = (math.exp(k * prog) - 1.0) / (math.exp(k) - 1.0)
    return 1.0 + rate * factor


# ============================================================
# NOUVEAUTÉ 4 : Extraction FC depuis FIT (qualité + drift)
# ============================================================
def analyze_hr_from_fit(hr_records: list, duration_s: float) -> dict:
    """
    Analyse les données FC :
    - FC max observée
    - FC moyenne
    - Drift cardiaque (augmentation en fin de course vs début)
    - Estimation seuil lactique (FC seuil ≈ 85-90% FCmax estimée)
    - Score de fiabilité de la référence
    """
    if not hr_records or len(hr_records) < 10:
        return {"hr_max": None, "hr_avg": None, "hr_drift": None, "reliability": "inconnue"}

    hrs = [h for h in hr_records if h is not None and 50 <= h <= 220]
    if len(hrs) < 10:
        return {"hr_max": None, "hr_avg": None, "hr_drift": None, "reliability": "inconnue"}

    hr_arr = np.array(hrs, dtype=float)
    n = len(hr_arr)
    hr_max = float(np.percentile(hr_arr, 95))  # 95e percentile (plus robuste que max absolu)
    hr_avg = float(np.mean(hr_arr))

    # Drift cardiaque : compare 1er quart vs dernier quart
    q1 = int(n * 0.25)
    q3 = int(n * 0.75)
    hr_start = float(np.mean(hr_arr[:q1])) if q1 > 0 else hr_avg
    hr_end = float(np.mean(hr_arr[q3:])) if q3 < n else hr_avg
    hr_drift = hr_end - hr_start  # positif = dérive cardiaque (fatigue thermique / glycolytique)

    # Score de fiabilité
    # La course est fiable si elle a été courue à effort constant (drift faible)
    # et non en sprint de fin (pattern physiologique cohérent)
    if hr_drift < 5:
        reliability = "haute"
    elif hr_drift < 12:
        reliability = "moyenne"
    else:
        reliability = "basse (dérive cardiaque forte — météo chaud ou effort variable)"

    # Estimation allure seuil lactique (LT)
    # FC seuil ≈ 88% FC max (règle empirique — à affiner avec les données athlète)
    hr_threshold_est = hr_max * 0.88

    return {
        "hr_max": round(hr_max, 0),
        "hr_avg": round(hr_avg, 0),
        "hr_drift": round(hr_drift, 1),
        "hr_threshold_est": round(hr_threshold_est, 0),
        "reliability": reliability,
        "n_records": len(hrs)
    }


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
        hr_records = []  # NOUVEAUTÉ : extraction FC
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

            alt = vals.get("enhanced_altitude", None)
            if alt is None: alt = vals.get("altitude", None)
            if alt is None: alt = vals.get("baro_altitude", None)
            if alt is None: alt = vals.get("gps_altitude", None)
            if alt is None: alt = 0.0

            dist_val = vals.get("distance", None) or 0.0

            # FC
            hr = vals.get("heart_rate", None)
            hr_records.append(int(hr) if hr is not None else None)

            records.append((lat, lon, float(alt), float(dist_val)))
            times_points.append(dt_local)

        if not records:
            return None

        df = pd.DataFrame(records, columns=["lat", "lon", "elev", "dist"])
        valid_times = [t for t in times_points if t is not None]

        if len(valid_times) >= 2:
            start_dt = min(valid_times)
            end_dt = max(valid_times)
        elif start_global and elapsed_global:
            start_dt = start_global
            end_dt = start_global + timedelta(seconds=elapsed_global)
        elif start_global:
            start_dt = start_global
            end_dt = start_global + timedelta(minutes=5)
        else:
            start_dt = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=1)
            end_dt = start_dt + timedelta(minutes=5)

        avgT, avgW, avgH = get_avg_weather_for_period(records[0][0], records[0][1], start_dt, end_dt, tz_name=tz_name)
        elev_arr = df["elev"].astype(float).values
        dup = float(np.sum(np.clip(np.diff(elev_arr), a_min=0, a_max=None))) if elev_arr.size >= 2 else 0.0
        ddn = float(-np.sum(np.clip(np.diff(elev_arr), a_min=None, a_max=0))) if elev_arr.size >= 2 else 0.0

        duration_s = (end_dt - start_dt).total_seconds()
        hr_analysis = analyze_hr_from_fit(hr_records, duration_s)

        return {
            "points": [{"lat": r[0], "lon": r[1], "elev": r[2], "dist": r[3], "time": t}
                       for (r, t) in zip(records, times_points)],
            "distance": float(df["dist"].max()) if not df.empty else 0.0,
            "D_up": dup,
            "D_down": ddn,
            "duration_hms": seconds_to_hms(duration_s),
            "avg_temp": avgT,
            "avg_wind": avgW,
            "avg_humidity": avgH,
            "hr_analysis": hr_analysis,
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
        "avg_humidity": avgH,
        "hr_analysis": None
    }


# ============================================================
# MODÈLE LOG-LOG (inchangé mais avec cross-val)
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
# NOUVEAUTÉ 7 : Cross-validation leave-one-out
# ============================================================
def crossval_loo(refs_prepared):
    """
    Leave-One-Out cross-validation sur les références préparées.
    Pour chaque ref, entraîne sur les autres, prédit sur celle-ci.
    Retourne un DataFrame avec erreurs absolues et relatives.
    """
    results = []
    n = len(refs_prepared)
    if n < 3:
        return None  # Pas assez de données

    for i in range(n):
        train = [r for j, r in enumerate(refs_prepared) if j != i]
        test = refs_prepared[i]
        a_cv, K_cv = fit_loglog_model(train)
        pred_s = predict_time_flat(test["distance"], a_cv, K_cv)
        actual_s = float(test["temps"])
        error_s = pred_s - actual_s
        error_pct = (error_s / actual_s * 100.0) if actual_s > 0 else 0.0
        dist_km = test["distance"] / 1000.0
        results.append({
            "Réf": i + 1,
            "Distance (km)": round(dist_km, 2),
            "Temps réel": seconds_to_hms(actual_s),
            "Temps prédit": seconds_to_hms(pred_s),
            "Erreur (s)": round(error_s, 0),
            "Erreur (%)": round(error_pct, 2),
        })

    df_cv = pd.DataFrame(results)
    mae = float(np.mean(np.abs(df_cv["Erreur (s)"].values)))
    mape = float(np.mean(np.abs(df_cv["Erreur (%)"].values)))
    return df_cv, mae, mape


# ============================================================
# NORMALISATION RÉFÉRENCES
# ============================================================
def elev_factor_from_dplus_dminus(D_up_m, D_down_m, distance_m, grade_k_up, grade_k_down,
                                   grade_down_cap, g0_up_pct, g0_down_pct, max_up, max_down):
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


def recalibrate_ref_to_ideal(ref, opt_temp, grade_k_up, grade_k_down, grade_down_cap,
                              g0_up_pct, g0_down_pct, max_grade_up, max_grade_down,
                              elev_ref_power=0.60, temp_ref_power=0.85,
                              cold_quad=0.0012, hot_quad=0.0016, temp_max_penalty=0.08,
                              use_wbgt=True):
    secs = hms_to_seconds(ref.get("temps")) if ref.get("temps") is not None else 0
    D_up = safe_float(ref.get("D_up", 0.0))
    D_down = safe_float(ref.get("D_down", 0.0))
    seg_len = safe_float(ref.get("distance", 1000.0))
    seg_len = seg_len if seg_len > 0 else 1000.0

    factor_elev = elev_factor_from_dplus_dminus(
        D_up_m=D_up, D_down_m=D_down, distance_m=seg_len,
        grade_k_up=grade_k_up, grade_k_down=grade_k_down,
        grade_down_cap=grade_down_cap, g0_up_pct=g0_up_pct, g0_down_pct=g0_down_pct,
        max_up=max_grade_up, max_down=max_grade_down
    )
    factor_elev = max(0.01, float(factor_elev))
    secs_no_elev = secs / (factor_elev ** float(elev_ref_power))

    temp_real = ref.get("avg_temp")
    hum_real = ref.get("avg_humidity", 50.0) or 50.0

    if temp_real is not None:
        # Utiliser WBGT si option activée
        temp_eff = effective_temp_for_performance(temp_real, hum_real, use_wbgt=use_wbgt) if use_wbgt else temp_real
        mult_real = temp_multiplier_realistic(
            temp_eff, opt_temp=opt_temp, cold_quad=cold_quad, hot_quad=hot_quad, max_penalty=temp_max_penalty
        )
        secs_no_temp = secs_no_elev / (max(0.01, float(mult_real)) ** float(temp_ref_power))
    else:
        secs_no_temp = secs_no_elev

    return max(0.0, float(secs_no_temp))


def prepare_refs_for_fit(refs_input, ideal_refs, opt_temp, grade_k_up, grade_k_down, grade_down_cap,
                          g0_up_pct, g0_down_pct, max_grade_up, max_grade_down,
                          elev_ref_power, temp_ref_power, cold_quad, hot_quad, temp_max_penalty,
                          use_wbgt=True):
    prepared = []
    for r in refs_input:
        d = safe_float(r.get("distance", 0.0))
        file_dur = r.get("duration_hms_file")
        raw_t = file_dur if file_dur else r.get("temps", "0:00:00")
        if ideal_refs:
            ref_for_calib = {
                "distance": d, "temps": raw_t,
                "D_up": r.get("D_up", 0.0), "D_down": r.get("D_down", 0.0),
                "avg_temp": r.get("avg_temp"), "avg_humidity": r.get("avg_humidity", 50.0),
            }
            secs_recal = recalibrate_ref_to_ideal(
                ref_for_calib, opt_temp=opt_temp,
                grade_k_up=grade_k_up, grade_k_down=grade_k_down, grade_down_cap=grade_down_cap,
                g0_up_pct=g0_up_pct, g0_down_pct=g0_down_pct,
                max_grade_up=max_grade_up, max_grade_down=max_grade_down,
                elev_ref_power=elev_ref_power, temp_ref_power=temp_ref_power,
                cold_quad=cold_quad, hot_quad=hot_quad, temp_max_penalty=temp_max_penalty,
                use_wbgt=use_wbgt
            )
            prepared.append({"distance": float(d), "temps": float(secs_recal)})
        else:
            prepared.append({"distance": float(d), "temps": float(hms_to_seconds(raw_t))})
    return prepared


# ============================================================
# PACING ULTRA (inchangé)
# ============================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def ultra_pacing_multiplier_linear(progress_0_1: float, amp_pct: float) -> float:
    p = clamp(float(progress_0_1), 0.0, 1.0)
    A = max(0.0, float(amp_pct)) / 100.0
    return 1.0 + A * (2.0 * p - 1.0)


def apply_ultra_pacing_profile(t_seg_raw, d_end_m, seg_len_m, total_corr_m, amp_pct):
    if t_seg_raw is None or len(t_seg_raw) == 0:
        return t_seg_raw
    total_corr_m = max(1e-9, float(total_corr_m))
    seg_len_m = np.asarray(seg_len_m, dtype=float)
    d_end_m = np.asarray(d_end_m, dtype=float)
    d_mid = d_end_m - 0.5 * seg_len_m
    progress = np.clip(d_mid / total_corr_m, 0.0, 1.0)
    mult = np.array([ultra_pacing_multiplier_linear(p, amp_pct) for p in progress], dtype=float)
    t_adj = np.asarray(t_seg_raw, dtype=float) * mult
    sum_raw = float(np.sum(t_seg_raw))
    sum_adj = float(np.sum(t_adj))
    if sum_raw > 0 and sum_adj > 0:
        t_adj *= (sum_raw / sum_adj)
    return t_adj


# ============================================================
# PRÉDICTION PRINCIPALE (v2 — tous facteurs intégrés)
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
    use_minetti=True,
    minetti_weight=0.6,
    grade_k_up=12.0, grade_k_down=5.0, grade_down_cap=-0.08,
    g0_up_pct=3.0, g0_down_pct=2.5, max_grade_up=0.30, max_grade_down=-0.06,
    elev_smooth_window=11, grade_power=0.85,
    # temp
    apply_temp=True, use_wbgt=True,
    opt_temp=12.0, cold_quad=0.0012, hot_quad=0.0016, temp_max_penalty=0.08, temp_power=1.0,
    # altitude physiologique
    apply_altitude_effect=True, altitude_ref_m=0.0,
    # vent
    apply_wind=True, wind_mode="Lissé (km/km)", wind_smooth_window_km=5,
    drag_coeff=0.012, tail_credit=0.35, wind_cap_head=0.10, wind_cap_tail=-0.04, wind_power=1.0,
    wind_gate_g1=2.0, wind_gate_g2=8.0, wind_gate_min=0.25,
    # cap cumul
    combined_base_cap=0.08, combined_extra_per_pct=0.004, combined_max_cap=0.18,
    # refs damping
    elev_ref_power=0.60, temp_ref_power=0.85,
    # fatigue v2
    apply_fatigue=False, fatigue_rate=0.0, fatigue_mode="mixte",
    # ultra pacing
    apply_ultra_pacing=False, ultra_pacing_amp_pct=10.0,
    # objectif
    objective_time_hms=None,
    tz_name=TZ_NAME_DEFAULT,
    show_smoothed_pace=True, smooth_pace_window_km=3,
    # DEM
    use_dem=False, dem_dataset="srtm30m",
    # correction altitude déjà calculée (passée depuis l'UI pour ne pas re-fetcher)
    dem_elevations=None,
):
    if not points or len(points) < 2:
        raise ValueError("GPX invalide ou trop court.")

    # -------- Altitudes --------
    if use_dem and dem_elevations is not None and len(dem_elevations) == len(points):
        elev_source = np.array([e if e is not None else 0.0 for e in dem_elevations], dtype=float)
    else:
        elev_source = np.array([getattr(p, "elevation", 0.0) or 0.0 for p in points], dtype=float)

    # -------- Distances cumulées --------
    total_m = 0.0
    cum = [0.0]
    for i in range(1, len(points)):
        d = haversine_m(points[i-1].latitude, points[i-1].longitude,
                        points[i].latitude, points[i].longitude)
        total_m += d
        cum.append(total_m)

    distance_gpx_km = total_m / 1000.0
    if distance_cible_km is None or distance_cible_km <= 0:
        distance_cible_km = distance_gpx_km
    facteur_dist = distance_cible_km / max(distance_gpx_km, 1e-9)
    total_corr = total_m * facteur_dist
    dists_corr = np.asarray([d * facteur_dist for d in cum], dtype=float)

    # -------- Altitude interpolée + lissage --------
    if elev_source.size != dists_corr.size:
        xs = np.linspace(0, total_m, elev_source.size)
        elev_source = np.interp(np.linspace(0, total_m, dists_corr.size), xs, elev_source)

    if elev_smooth_window and elev_smooth_window >= 3 and elev_source.size >= elev_smooth_window:
        w = int(elev_smooth_window)
        if w % 2 == 0: w += 1
        elev_smooth = np.convolve(elev_source, np.ones(w)/w, mode="same")
    else:
        elev_smooth = elev_source

    # -------- D+ total du parcours (pour fatigue) --------
    diffs_elev = np.diff(elev_smooth)
    d_plus_total = float(np.sum(np.clip(diffs_elev, 0, None)))

    # -------- Altitude moyenne du parcours --------
    avg_altitude_m = float(np.mean(elev_smooth))

    # -------- Références -> fit --------
    refs_for_fit = prepare_refs_for_fit(
        refs_input=refs_input, ideal_refs=ideal_refs,
        opt_temp=opt_temp, grade_k_up=grade_k_up, grade_k_down=grade_k_down,
        grade_down_cap=grade_down_cap, g0_up_pct=g0_up_pct, g0_down_pct=g0_down_pct,
        max_grade_up=max_grade_up, max_grade_down=max_grade_down,
        elev_ref_power=elev_ref_power, temp_ref_power=temp_ref_power,
        cold_quad=cold_quad, hot_quad=hot_quad, temp_max_penalty=temp_max_penalty,
        use_wbgt=use_wbgt
    )

    a, K = fit_loglog_model(refs_for_fit)
    a_override = override_with_objective(int(distance_cible_km * 1000), objective_time_hms, K) if objective_time_hms else None
    baseline_a = a_override if a_override is not None else a
    base_flat_total = predict_time_flat(int(distance_cible_km * 1000), baseline_a, K)
    base_s_per_km_flat = base_flat_total / max(distance_cible_km, 1e-9)

    # -------- Segments --------
    km_marks = [i * 1000 for i in range(1, int(total_corr // 1000) + 1)]
    last_seg = total_corr - (int(total_corr // 1000) * 1000)
    if last_seg > 1e-6:
        km_marks.append(total_corr)

    df_points = pd.DataFrame([{
        "lat": p.latitude, "lon": p.longitude,
        "elev": getattr(p, "elevation", 0.0),
        "time": getattr(p, "time", None),
    } for p in points])

    dt_depart = datetime.combine(date_course_local, heure_course_local)

    # -------- Effet altitude physiologique --------
    alt_mult = altitude_vo2_multiplier(avg_altitude_m, altitude_ref_m) if apply_altitude_effect else 1.0

    # -------- Passe 1 : pente + temp + altitude physio --------
    pre = []
    cum_time_tmp = 0.0
    cum_dplus = 0.0
    cum_dist = 0.0

    for i, d in enumerate(km_marks):
        seg_length_m = 1000.0
        if i == len(km_marks) - 1 and last_seg > 1e-6:
            seg_length_m = (d - km_marks[-2]) if len(km_marks) >= 2 else d

        e_cur = float(np.interp(d, dists_corr, elev_smooth))
        e_prev_d = max(d - seg_length_m, 0.0)
        e_prev = float(np.interp(e_prev_d, dists_corr, elev_smooth)) if i > 0 else e_cur
        delta_e = e_cur - e_prev
        grade_pct = (delta_e / max(1e-6, seg_length_m)) * 100.0

        # D+ segment
        seg_dplus = max(0.0, delta_e)
        cum_dplus += seg_dplus
        cum_dist += seg_length_m

        # Temps plat segment
        t_flat = base_s_per_km_flat * (seg_length_m / 1000.0)

        # Pente (Minetti + heuristique)
        if apply_grade:
            g_mult = combined_grade_multiplier(
                grade_pct, use_minetti=use_minetti, minetti_weight=minetti_weight,
                k_up=grade_k_up, k_down=grade_k_down, down_cap=grade_down_cap,
                g0_up=g0_up_pct, g0_down=g0_down_pct, max_up=max_grade_up, max_down=max_grade_down
            )
            t_after_grade = t_flat * (g_mult ** float(grade_power))
        else:
            g_mult = 1.0
            t_after_grade = t_flat

        # Effet altitude physiologique (appliqué globalement sur le temps de base)
        t_after_altitude = t_after_grade * alt_mult if apply_altitude_effect else t_after_grade

        # Fatigue v2 (basée sur D+ cumulé + distance)
        if apply_fatigue and fatigue_rate > 0:
            fat_mult = fatigue_multiplier_cumulative(
                d_plus_cum=cum_dplus,
                distance_cum=cum_dist,
                d_plus_total=d_plus_total,
                distance_total=total_corr,
                fatigue_rate_pct=fatigue_rate,
                fatigue_mode=fatigue_mode
            )
            t_after_fatigue = t_after_altitude * fat_mult
        else:
            fat_mult = 1.0
            t_after_fatigue = t_after_altitude

        # Météo (passage milieu segment)
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
        hum_here  = meteo["humidity"] if meteo else None
        wind_dir_here = meteo.get("wind_dir") if meteo else None

        # Température effective (WBGT si activé)
        temp_eff = None
        wbgt_here = None
        if temp_here is not None and hum_here is not None:
            if use_wbgt:
                temp_eff = effective_temp_for_performance(temp_here, hum_here, use_wbgt=True)
                wbgt_here = temp_eff
            else:
                temp_eff = temp_here

        if apply_temp and temp_eff is not None:
            temp_mult = temp_multiplier_realistic(
                temp_eff, opt_temp=opt_temp, cold_quad=cold_quad, hot_quad=hot_quad, max_penalty=temp_max_penalty
            )
            t_after_temp = t_after_fatigue * (float(temp_mult) ** float(temp_power))
        else:
            temp_mult = 1.0
            t_after_temp = t_after_fatigue

        pace_s_per_km_local = (t_after_temp / seg_length_m) * 1000.0 if seg_length_m > 0 else t_after_temp
        head_ms, tail_ms, cross_ms = wind_components_along_course(wind_here, wind_dir_here, course_bearing)

        pre.append({
            "idx": i,
            "d": float(d),
            "seg_length_m": float(seg_length_m),
            "grade_pct": float(grade_pct),
            "grade_mult": float(g_mult),
            "seg_dplus": float(seg_dplus),
            "cum_dplus": float(cum_dplus),
            "fat_mult": float(fat_mult),
            "alt_mult": float(alt_mult),
            "temp": temp_here,
            "temp_eff": temp_eff,
            "wbgt": wbgt_here,
            "humidity": hum_here,
            "wind": wind_here,
            "wind_dir": wind_dir_here,
            "course_bearing": float(course_bearing),
            "head_ms": float(head_ms),
            "tail_ms": float(tail_ms),
            "cross_ms": float(cross_ms),
            "temp_mult": float(temp_mult),
            "t_flat": float(t_flat),
            "t_no_wind": float(t_after_temp),
            "pace_no_wind": float(pace_s_per_km_local),
        })
        cum_time_tmp += float(t_after_temp)

    pre_df = pd.DataFrame(pre)

    # -------- Vent --------
    if apply_wind and not pre_df.empty:
        if wind_mode == "Global (un seul effet sur la course)":
            head_g = float(np.median(pre_df["head_ms"].values))
            tail_g = float(np.median(pre_df["tail_ms"].values))
            pace_g = float(np.median(pre_df["pace_no_wind"].values))
            global_wind_mult = wind_multiplier_realistic(
                head_ms=head_g, tail_ms=tail_g, pace_s_per_km=pace_g,
                drag_coeff=drag_coeff, tail_credit=tail_credit, cap_head=wind_cap_head, cap_tail=wind_cap_tail
            )
            pre_df["wind_mult_raw"] = float(global_wind_mult)
        else:
            w = int(max(1, wind_smooth_window_km))
            if w % 2 == 0: w += 1
            head_s = pd.Series(pre_df["head_ms"]).rolling(window=w, center=True, min_periods=1).median()
            tail_s = pd.Series(pre_df["tail_ms"]).rolling(window=w, center=True, min_periods=1).median()
            wind_mults = [
                wind_multiplier_realistic(float(hm), float(tm), float(pace), drag_coeff, tail_credit, wind_cap_head, wind_cap_tail)
                for hm, tm, pace in zip(head_s.values, tail_s.values, pre_df["pace_no_wind"].values)
            ]
            pre_df["wind_mult_raw"] = np.array(wind_mults, dtype=float)
            pre_df["head_ms_smooth"] = head_s.values
            pre_df["tail_ms_smooth"] = tail_s.values
    else:
        if not pre_df.empty:
            pre_df["wind_mult_raw"] = 1.0

    # -------- Wind gate + cap cumulé --------
    wind_mult_adj, total_mult_capped, t_final_raw = [], [], []
    for _, row in pre_df.iterrows():
        wind_mult = float(row["wind_mult_raw"])
        grade_pct = float(row["grade_pct"])
        gate = wind_gate_from_grade(grade_pct, g1=wind_gate_g1, g2=wind_gate_g2, min_gate=wind_gate_min)
        wind_mult = 1.0 + gate * (wind_mult - 1.0)
        t_with_wind = float(row["t_no_wind"]) * (wind_mult ** float(wind_power))
        t_flat = max(1e-9, float(row["t_flat"]))
        mult_total = t_with_wind / t_flat
        mult_total = cap_combined_multiplier(mult_total, grade_pct=grade_pct,
                                              base_cap=combined_base_cap,
                                              extra_per_pct=combined_extra_per_pct,
                                              max_cap=combined_max_cap)
        t_seg = t_flat * mult_total
        wind_mult_adj.append(float(wind_mult))
        total_mult_capped.append(float(mult_total))
        t_final_raw.append(float(t_seg))

    pre_df["wind_mult"] = np.array(wind_mult_adj, dtype=float)
    pre_df["mult_total_capped"] = np.array(total_mult_capped, dtype=float)
    t_raw = np.array(t_final_raw, dtype=float)

    # -------- Ultra pacing --------
    if apply_ultra_pacing and not pre_df.empty and float(ultra_pacing_amp_pct) > 0:
        t_raw = apply_ultra_pacing_profile(
            t_seg_raw=t_raw,
            d_end_m=pre_df["d"].astype(float).values,
            seg_len_m=pre_df["seg_length_m"].astype(float).values,
            total_corr_m=total_corr,
            amp_pct=ultra_pacing_amp_pct
        )

    # -------- Scale objectif --------
    if objective_time_hms:
        objective_seconds = hms_to_seconds(objective_time_hms)
        sum_raw = float(np.sum(t_raw))
        scale = (objective_seconds / sum_raw) if sum_raw > 0 else 1.0
    else:
        scale = 1.0

    # -------- Table finale --------
    results = []
    cum_time = 0.0
    for i in range(len(pre_df)):
        seg = pre_df.iloc[i]
        t_seg = float(t_raw[i]) * float(scale)
        cum_time += t_seg
        pace_per_km = (t_seg / float(seg["seg_length_m"])) * 1000.0 if seg["seg_length_m"] > 0 else t_seg
        km_label = (seg["idx"] + 1) if seg["seg_length_m"] >= 1000 - 1e-6 else f"{int(seg['idx']+1)} ({seg['seg_length_m']:.0f}m)"
        head_disp = seg.get("head_ms_smooth", seg["head_ms"])
        tail_disp = seg.get("tail_ms_smooth", seg["tail_ms"])

        results.append({
            "Km": km_label,
            "Pente (%)": round(float(seg["grade_pct"]), 2),
            "Mult Pente (Minetti+heu)": round(float(seg["grade_mult"]), 4),
            "D+ seg (m)": round(float(seg["seg_dplus"]), 1),
            "D+ cum (m)": round(float(seg["cum_dplus"]), 1),
            "Mult Fatigue": round(float(seg["fat_mult"]), 4),
            "Mult Altitude physio": round(float(seg["alt_mult"]), 4),
            "Temp GPS (°C)": round(float(seg["temp"]), 1) if seg["temp"] is not None else None,
            "Temp eff/WBGT (°C)": round(float(seg["temp_eff"]), 1) if seg["temp_eff"] is not None else None,
            "Mult Temp": round(float(seg["temp_mult"]), 4),
            "Vent (m/s)": round(float(seg["wind"]), 1) if seg["wind"] is not None else None,
            "Dir vent (° FROM)": round(float(seg["wind_dir"]), 0) if seg["wind_dir"] is not None else None,
            "Cap seg (°)": round(float(seg["course_bearing"]), 0),
            "Headwind (m/s)": round(float(head_disp), 2),
            "Tailwind (m/s)": round(float(tail_disp), 2),
            "Mult Vent (gate)": round(float(seg["wind_mult"]), 4),
            "Mult total (cappé)": round(float(seg["mult_total_capped"]), 4),
            "Humidité (%)": round(float(seg["humidity"]), 1) if seg["humidity"] is not None else None,
            "Temps segment (s)": round(t_seg, 1),
            "Allure (min/km)": pace_seconds_to_str_per_km(pace_per_km),
            "Temps cumulé": seconds_to_hms(cum_time),
        })

    df = pd.DataFrame(results)

    if show_smoothed_pace and not df.empty:
        try:
            pace_s = df["Temps segment (s)"].astype(float).values
            w = int(max(1, smooth_pace_window_km))
            if w % 2 == 0: w += 1
            s = pd.Series(pace_s).rolling(window=w, center=True, min_periods=1).median()
            df["Allure lissée (min/km)"] = s.apply(lambda x: pace_seconds_to_str_per_km(x))
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
        "avg_altitude_m": float(avg_altitude_m),
        "d_plus_total": float(d_plus_total),
        "refs_for_fit": refs_for_fit,
        "pre_df": pre_df,
    }


# ============================================================
# UI
# ============================================================
st.header("1️⃣ Parcours GPX")
gpx_file = st.file_uploader("📂 Importer un fichier GPX", type=["gpx"])
points = None

if gpx_file:
    _gpx, points = parse_gpx_points(gpx_file)
    if points:
        total_m_tmp = sum(
            haversine_m(points[i-1].latitude, points[i-1].longitude,
                        points[i].latitude, points[i].longitude)
            for i in range(1, len(points))
        )
        st.session_state["gpx_original_distance_km"] = total_m_tmp / 1000.0
        dup_tmp, ddn_tmp = compute_dplus_dminus([getattr(p, "elevation", 0.0) or 0.0 for p in points])
        avg_alt_tmp = float(np.mean([getattr(p, "elevation", 0.0) or 0.0 for p in points]))
        st.info(
            f"📍 GPX chargé — {total_m_tmp/1000:.2f} km | "
            f"D+ {dup_tmp:.0f} m | D- {ddn_tmp:.0f} m | Alt. moy. {avg_alt_tmp:.0f} m"
        )
    else:
        st.session_state["gpx_original_distance_km"] = None

# -------- DEM correction --------
st.subheader("🏔️ Correction altimétrique DEM (optionnel)")
use_dem = st.checkbox(
    "Corriger l'altitude GPS via Open-Topo-Data (DEM) — recommandé pour parcours montagneux",
    value=False
)
dem_dataset = "srtm30m"
dem_elevations = None
if use_dem:
    dem_dataset = st.selectbox(
        "Dataset DEM",
        ["srtm30m", "eudem25m", "mapzen"],
        index=0,
        help="srtm30m = global 30m | eudem25m = Europe 25m (meilleur) | mapzen = global fusion"
    )
    if gpx_file and points and st.button("🔄 Télécharger altitudes DEM"):
        with st.spinner("Récupération des altitudes DEM..."):
            dem_elevations = correct_gpx_elevations_with_dem(points, max_points=100, dataset=dem_dataset)
            st.session_state["dem_elevations"] = list(dem_elevations)
            dup_dem, ddn_dem = compute_dplus_dminus(dem_elevations)
            dup_gps, ddn_gps = compute_dplus_dminus([getattr(p, "elevation", 0.0) or 0.0 for p in points])
            st.success(
                f"DEM OK — D+ DEM: **{dup_dem:.0f} m** (vs GPS: {dup_gps:.0f} m) | "
                f"D- DEM: **{ddn_dem:.0f} m** (vs GPS: {ddn_gps:.0f} m)"
            )
    elif "dem_elevations" in st.session_state and st.session_state["dem_elevations"]:
        dem_elevations = st.session_state["dem_elevations"]
        st.info("Altitudes DEM déjà chargées (cache session).")

# ============================================================
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
    end_hms   = col_e.text_input(f"Fin réf {i} (hh:mm:ss)", value="23:59:59", key=f"end_{i}")
    start_td = hms_to_timedelta(start_hms)
    end_td   = hms_to_timedelta(end_hms)

    duration_hms_file = None
    avg_temp_ref = avg_wind_ref = avg_hum_ref = None
    hr_analysis_ref = None
    fit_data = tcx_data = None
    filename = file_in.name.lower() if file_in else ""

    if file_in:
        if filename.endswith(".fit"):
            fit_data = parse_fit(file_in)
            if fit_data:
                dist = fit_data["distance"]
                dup  = fit_data["D_up"]
                ddn  = fit_data["D_down"]
                duration_hms_file = fit_data["duration_hms"]
                avg_temp_ref = fit_data["avg_temp"]
                avg_wind_ref = fit_data["avg_wind"]
                avg_hum_ref  = fit_data["avg_humidity"]
                hr_analysis_ref = fit_data.get("hr_analysis")
                if hr_analysis_ref:
                    st.info(
                        f"💓 FC — Max: {hr_analysis_ref.get('hr_max')} bpm | "
                        f"Moy: {hr_analysis_ref.get('hr_avg')} bpm | "
                        f"Seuil estimé: {hr_analysis_ref.get('hr_threshold_est')} bpm | "
                        f"Dérive: {hr_analysis_ref.get('hr_drift')} bpm | "
                        f"Fiabilité: **{hr_analysis_ref.get('reliability')}**"
                    )
        elif filename.endswith(".tcx"):
            tcx_data = parse_tcx(file_in)
            if tcx_data:
                dist = tcx_data["distance"]
                dup  = tcx_data["D_up"]
                ddn  = tcx_data["D_down"]
                duration_hms_file = tcx_data["duration_hms"]
                avg_temp_ref = tcx_data["avg_temp"]
                avg_wind_ref = tcx_data["avg_wind"]
                avg_hum_ref  = tcx_data["avg_humidity"]

        if start_td.total_seconds() > 0 or end_td.total_seconds() < 86399:
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
        "hr_analysis": hr_analysis_ref,
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
        reliability_str = ""
        if r.get("hr_analysis"):
            reliability_str = f" | Fiabilité FC: {r['hr_analysis'].get('reliability', '?')}"
        st.write(f"Réf {idx} — {r['distance']:.0f} m en {r['temps']} → {pace_seconds_to_str_per_km(pace)}/km{reliability_str}")
        if pace < 150:
            st.warning("⚠️ Allure extrêmement rapide → vérifie le format du temps.")

# ============================================================
st.header("3️⃣ Paramètres modèle v2")

st.subheader("🌡️ Température — WBGT & indice de chaleur")
use_wbgt = st.checkbox("Utiliser WBGT (température ressentie humidité+chaleur)", value=True,
                        help="Recommandé : combine température + humidité via formule de Stull (2011). "
                             "Différence majeure par temps chaud et humide.")
apply_temp = st.checkbox("Appliquer température", value=True)
colt1, colt2, colt3 = st.columns(3)
with colt1:
    opt_temp = st.number_input("Temp optimale (°C)", value=12.0, step=0.5)
with colt2:
    cold_quad = st.number_input("Froid (quad)", value=0.0012, step=0.0002, format="%.4f")
with colt3:
    hot_quad = st.number_input("Chaud (quad)", value=0.0016, step=0.0002, format="%.4f")
temp_max_penalty = st.slider("Cap pénalité temp", 0.00, 0.20, 0.10, 0.01,
                              help="Augmenter à 0.12-0.15 pour les courses par très forte chaleur.")
temp_power = st.slider("Damping temp (puissance)", 0.2, 1.2, 1.0, 0.05)

# Démo WBGT
if use_wbgt:
    with st.expander("📊 Tableau WBGT de référence (aperçu de l'impact réel)"):
        demo_data = []
        for t_demo in [10, 15, 20, 25, 30, 35]:
            for h_demo in [30, 50, 70, 90]:
                wbgt_v = wbgt_simplified(t_demo, h_demo)
                hi_v = heat_index_celsius(t_demo, h_demo)
                mult = temp_multiplier_realistic(wbgt_v, opt_temp=opt_temp, cold_quad=cold_quad,
                                                  hot_quad=hot_quad, max_penalty=temp_max_penalty)
                demo_data.append({
                    "Temp GPS (°C)": t_demo,
                    "Humidité (%)": h_demo,
                    "WBGT (°C)": round(wbgt_v, 1),
                    "Heat Index (°C)": round(hi_v, 1),
                    "Mult perf": round(mult, 3),
                    "Pénalité (%)": round((mult - 1) * 100, 1)
                })
        st.dataframe(pd.DataFrame(demo_data), use_container_width=True)

st.subheader("🏔️ Effet altitude physiologique (hypoxie)")
apply_altitude_effect = st.checkbox("Appliquer pénalité altitude (VO2 réduite > 1500m)", value=True)
altitude_ref_m = st.number_input(
    "Altitude d'entraînement habituelle de l'athlète (m)",
    value=0.0, step=100.0,
    help="Si l'athlète s'entraîne déjà en altitude, la pénalité est relative à cette altitude."
) if apply_altitude_effect else 0.0

st.subheader("🎢 Pente — Modèle Minetti (physique) + heuristique")
apply_grade = st.checkbox("Prendre en compte la pente", value=True)
use_minetti = st.checkbox(
    "Activer modèle Minetti et al. (2002) — base physiologique réelle",
    value=True,
    help="Polynôme du 5e degré ajusté sur données expérimentales humains. "
         "Capture l'asymétrie montée/descente et la saturation aux pentes extrêmes."
)
minetti_weight = st.slider(
    "Poids Minetti vs heuristique (0 = 100% heuristique, 1 = 100% Minetti)",
    0.0, 1.0, 0.6, 0.1
) if use_minetti else 0.0

colg1, colg2, colg3 = st.columns(3)
with colg1:
    grade_k_up = st.number_input("Sensibilité montée (heuristique)", value=12.0, step=0.5)
with colg2:
    grade_k_down = st.number_input("Sensibilité descente (heuristique)", value=5.0, step=0.5)
with colg3:
    grade_down_cap = st.number_input("Cap bonus descente (ex -0.08)", value=-0.08, step=0.01, format="%.2f")
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

st.subheader("💨 Vent")
apply_wind = st.checkbox("Appliquer le vent", value=True)
wind_mode = st.selectbox("Mode vent", ["Lissé (km/km)", "Global (un seul effet sur la course)"], index=0)
colw1, colw2, colw3 = st.columns(3)
with colw1:
    wind_smooth_window_km = st.slider("Lissage vent (km)", 1, 11, 7, 2)
with colw2:
    drag_coeff = st.number_input("drag_coeff", value=0.012, step=0.002, format="%.3f")
with colw3:
    tail_credit = st.slider("Crédit tailwind", 0.0, 0.8, 0.35, 0.05)
colw4, colw5, colw6 = st.columns(3)
with colw4:
    wind_cap_head = st.slider("Cap pénalité vent (+)", 0.00, 0.20, 0.10, 0.01)
with colw5:
    wind_cap_tail = st.slider("Cap bonus vent (-)", -0.10, 0.00, -0.04, 0.01)
with colw6:
    wind_power = st.slider("Damping vent", 0.2, 1.2, 1.0, 0.05)

st.subheader("🧱 Anti cumul")
colC1, colC2, colC3 = st.columns(3)
with colC1:
    combined_base_cap = st.slider("Cap base (+%)", 0.02, 0.20, 0.08, 0.01)
with colC2:
    combined_extra_per_pct = st.slider("Extra cap par % pente", 0.000, 0.020, 0.004, 0.001)
with colC3:
    combined_max_cap = st.slider("Cap max (+%)", 0.05, 0.35, 0.18, 0.01)

st.subheader("⛰️ Normalisation références")
colR1, colR2 = st.columns(2)
with colR1:
    elev_ref_power = st.slider("Atténuation pente refs", 0.0, 1.0, 0.60, 0.05)
with colR2:
    temp_ref_power = st.slider("Atténuation température refs", 0.0, 1.0, 0.85, 0.05)

st.subheader("💨 Wind gate")
colG1, colG2, colG3 = st.columns(3)
with colG1:
    wind_gate_g1 = st.number_input("Seuil début réduction g1 (%)", value=2.0, step=0.5)
with colG2:
    wind_gate_g2 = st.number_input("Seuil réduction forte g2 (%)", value=8.0, step=0.5)
with colG3:
    wind_gate_min = st.slider("Plancher impact vent", 0.0, 1.0, 0.25, 0.05)

# -------- Tableau récap refs recalibrées --------
st.subheader("⏱️ Références recalibrées — contrôle coach")
use_recalibrated_refs = st.checkbox("Utiliser les références recalibrées pour le fit", value=True)

refs_calibrated = []
for r in refs_raw:
    t_brut = hms_to_seconds(r["temps"])
    t_ideal = recalibrate_ref_to_ideal(
        ref={**r, "avg_humidity": r.get("avg_humidity", 50.0)},
        opt_temp=opt_temp,
        grade_k_up=grade_k_up, grade_k_down=grade_k_down, grade_down_cap=grade_down_cap,
        g0_up_pct=g0_up_pct, g0_down_pct=g0_down_pct,
        max_grade_up=max_grade_up, max_grade_down=max_grade_down,
        elev_ref_power=elev_ref_power, temp_ref_power=temp_ref_power,
        cold_quad=cold_quad, hot_quad=hot_quad, temp_max_penalty=temp_max_penalty,
        use_wbgt=use_wbgt
    )
    dist_km = max(1e-9, float(r["distance"]) / 1000.0)
    pace_brut = (t_brut / dist_km) if t_brut > 0 else None
    pace_ideal = (t_ideal / dist_km) if t_ideal > 0 else None
    hr_rel = r.get("hr_analysis", {}) or {}
    refs_calibrated.append({
        "Distance (m)": float(r["distance"]),
        "D+ (m)": float(r.get("D_up", 0.0)),
        "D- (m)": float(r.get("D_down", 0.0)),
        "Temp moy (°C)": r.get("avg_temp"),
        "Hum moy (%)": r.get("avg_humidity"),
        "WBGT moy (°C)": round(wbgt_simplified(r["avg_temp"], r.get("avg_humidity", 50.0) or 50.0), 1)
                          if r.get("avg_temp") and use_wbgt else None,
        "FC max": hr_rel.get("hr_max"),
        "FC dérive (bpm)": hr_rel.get("hr_drift"),
        "Fiabilité FC": hr_rel.get("reliability"),
        "Temps brut": seconds_to_hms(t_brut),
        "Allure brute": pace_seconds_to_str_per_km(pace_brut) if pace_brut else None,
        "Temps recalibré": seconds_to_hms(t_ideal),
        "Allure recalibrée": pace_seconds_to_str_per_km(pace_ideal) if pace_ideal else None,
        "Δ temps": seconds_to_hms(max(0, t_brut - t_ideal)),
    })

df_refs = pd.DataFrame(refs_calibrated)
st.dataframe(df_refs, use_container_width=True)

if use_recalibrated_refs:
    st.success("✅ Mode actif : fit performance utilise les références recalibrées.")
else:
    st.info("ℹ️ Mode actif : fit performance utilise les références brutes.")

# ============================================================
# NOUVEAUTÉ 7 : Cross-validation
# ============================================================
st.subheader("📊 Cross-validation Leave-One-Out des références")
if st.button("🔬 Lancer la cross-validation"):
    refs_for_cv = prepare_refs_for_fit(
        refs_input=refs_raw, ideal_refs=use_recalibrated_refs,
        opt_temp=opt_temp, grade_k_up=grade_k_up, grade_k_down=grade_k_down,
        grade_down_cap=grade_down_cap, g0_up_pct=g0_up_pct, g0_down_pct=g0_down_pct,
        max_grade_up=max_grade_up, max_grade_down=max_grade_down,
        elev_ref_power=elev_ref_power, temp_ref_power=temp_ref_power,
        cold_quad=cold_quad, hot_quad=hot_quad, temp_max_penalty=temp_max_penalty,
        use_wbgt=use_wbgt
    )
    cv_result = crossval_loo(refs_for_cv)
    if cv_result is None:
        st.warning("Cross-validation nécessite au moins 3 références.")
    else:
        df_cv, mae, mape = cv_result
        st.dataframe(df_cv, use_container_width=True)
        col_mae, col_mape = st.columns(2)
        col_mae.metric("MAE (erreur absolue moyenne)", f"{mae:.0f} s ({seconds_to_hms(mae)})")
        col_mape.metric("MAPE (erreur relative moyenne)", f"{mape:.2f} %")
        if mape < 3:
            st.success("✅ Modèle bien calibré (MAPE < 3%)")
        elif mape < 7:
            st.warning("⚠️ Calibration correcte (MAPE 3-7%) — ajouter des références améliorera la précision.")
        else:
            st.error("❌ Calibration faible (MAPE > 7%) — vérifier les références (temps, distances, D+/D-).")

# ============================================================
# Fatigue v2
# ============================================================
st.header("3️⃣ bis. Fatigue v2 (basée sur D+ cumulé)")
fatigue_active = st.checkbox("Activer fatigue ?", value=False)
if fatigue_active:
    fatigue_rate = st.slider("Ralentissement total en fin de course (%)", 0.0, 30.0, 8.0, 0.5)
    fatigue_mode = st.selectbox(
        "Mode fatigue",
        ["mixte", "distance", "d_plus"],
        index=0,
        help="mixte = pondère D+ et distance (recommandé) | distance = ancien modèle | d_plus = montagne"
    )
else:
    fatigue_rate = 0.0
    fatigue_mode = "mixte"

st.header("3️⃣ ter. Pacing Ultra")
ultra_pacing_active = st.checkbox("Activer pacing Ultra (départ plus vite → fin plus lente)", value=False)
ultra_pacing_amp_pct = st.slider("Amplitude pacing (%) : début -A% / fin +A%", 0.0, 40.0, 10.0, 0.5) if ultra_pacing_active else 0.0

st.subheader("📉 Allure lissée")
show_smoothed_pace = st.checkbox("Afficher allure lissée (médiane)", value=True)
smooth_pace_window_km = st.slider("Fenêtre lissage allure (km)", 1, 9, 3, 2) if show_smoothed_pace else 3

st.subheader("📅 Course")
col1, col2 = st.columns(2)
with col1:
    date_course = st.date_input("Date", value=date.today())
with col2:
    heure_course = st.time_input("Heure départ", value=time(9, 0))

st.markdown("---")
st.header("4️⃣ Calcul")
colf1, colf2 = st.columns(2)
with colf1:
    force_distance_checkbox = st.checkbox("Forcer distance ?", value=False)
    if "dist_forced" not in st.session_state:
        st.session_state["dist_forced"] = 42.195
    distance_forced_km = st.number_input(
        "Distance forcée (km)", value=float(st.session_state["dist_forced"]),
        format="%.3f", key="dist_forced"
    ) if force_distance_checkbox else None
with colf2:
    force_time_checkbox = st.checkbox("Forcer temps objectif ?", value=False)
    if "time_forced" not in st.session_state:
        st.session_state["time_forced"] = "3:30:00"
    time_forced_hms = st.text_input(
        "Temps objectif (h:mm:ss)", value=str(st.session_state["time_forced"]), key="time_forced"
    ) if force_time_checkbox else None

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
                ideal_refs=use_recalibrated_refs,
                apply_grade=apply_grade,
                use_minetti=use_minetti,
                minetti_weight=minetti_weight,
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
                use_wbgt=use_wbgt,
                opt_temp=opt_temp,
                cold_quad=cold_quad,
                hot_quad=hot_quad,
                temp_max_penalty=temp_max_penalty,
                temp_power=temp_power,
                apply_altitude_effect=apply_altitude_effect,
                altitude_ref_m=altitude_ref_m,
                apply_wind=apply_wind,
                wind_mode=wind_mode,
                wind_smooth_window_km=wind_smooth_window_km,
                drag_coeff=drag_coeff,
                tail_credit=tail_credit,
                wind_cap_head=wind_cap_head,
                wind_cap_tail=wind_cap_tail,
                wind_power=wind_power,
                wind_gate_g1=wind_gate_g1,
                wind_gate_g2=wind_gate_g2,
                wind_gate_min=wind_gate_min,
                combined_base_cap=combined_base_cap,
                combined_extra_per_pct=combined_extra_per_pct,
                combined_max_cap=combined_max_cap,
                elev_ref_power=elev_ref_power,
                temp_ref_power=temp_ref_power,
                apply_fatigue=fatigue_active,
                fatigue_rate=fatigue_rate,
                fatigue_mode=fatigue_mode,
                apply_ultra_pacing=ultra_pacing_active,
                ultra_pacing_amp_pct=ultra_pacing_amp_pct,
                objective_time_hms=time_forced_hms if force_time_checkbox else None,
                show_smoothed_pace=show_smoothed_pace,
                smooth_pace_window_km=smooth_pace_window_km,
                use_dem=use_dem,
                dem_elevations=dem_elevations,
            )
            st.session_state["res"] = res
            st.success("Prédiction calculée ✅")
        except Exception as e:
            st.error(f"Erreur : {e}")
            import traceback
            st.code(traceback.format_exc())

if "res" in st.session_state:
    res = st.session_state["res"]
    st.subheader("📈 Résultat")
    avg_pace = res["total_seconds"] / max(res["distance_gpx_km"], 1e-6)

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Distance GPX", f"{res['distance_gpx_km']:.3f} km")
    col_r2.metric("Temps prédit", res["total_human"])
    col_r3.metric("Allure moy.", pace_seconds_to_str_per_km(avg_pace) + "/km")
    col_r4.metric("Alt. moy. parcours", f"{res.get('avg_altitude_m', 0):.0f} m")

    col_r5, col_r6 = st.columns(2)
    col_r5.metric("D+ total (lissé)", f"{res.get('d_plus_total', 0):.0f} m")
    col_r6.metric("K Riegel", f"{res['K']:.3f}")

    st.dataframe(res["df"], use_container_width=True)

    # Graphique allure par km
    if not res["df"].empty:
        st.subheader("📊 Graphique allure par km")
        fig, ax = plt.subplots(figsize=(12, 4))
        pace_vals = []
        for v in res["df"]["Allure (min/km)"].values:
            try:
                parts = str(v).split(":")
                pace_vals.append(int(parts[0]) + int(parts[1]) / 60.0)
            except Exception:
                pace_vals.append(None)
        x_vals = list(range(1, len(pace_vals) + 1))
        pace_clean = [p if p is not None else float("nan") for p in pace_vals]
        ax.plot(x_vals, pace_clean, lw=1.5, alpha=0.4, color="steelblue", label="Allure brute")

        if "Allure lissée (min/km)" in res["df"].columns:
            pace_smooth = []
            for v in res["df"]["Allure lissée (min/km)"].values:
                try:
                    parts = str(v).split(":")
                    pace_smooth.append(int(parts[0]) + int(parts[1]) / 60.0)
                except Exception:
                    pace_smooth.append(None)
            pace_smooth_clean = [p if p is not None else float("nan") for p in pace_smooth]
            ax.plot(x_vals, pace_smooth_clean, lw=2.5, color="firebrick", label="Allure lissée")

        ax.invert_yaxis()
        ax.set_xlabel("Km")
        ax.set_ylabel("Allure (min/km)")
        ax.set_title("Allure prévisionnelle par kilomètre")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Graphique multiplicateurs
    if not res["df"].empty:
        st.subheader("🔎 Décomposition des multiplicateurs par km")
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        df_plot = res["df"]
        x = list(range(1, len(df_plot) + 1))
        ax2.plot(x, df_plot["Mult Pente (Minetti+heu)"].values, label="Pente", lw=2)
        ax2.plot(x, df_plot["Mult Temp"].values, label="Temp", lw=2)
        ax2.plot(x, df_plot["Mult Vent (gate)"].values, label="Vent", lw=2)
        ax2.plot(x, df_plot["Mult total (cappé)"].values, label="Total cappé", lw=2.5, ls="--", color="black")
        ax2.plot(x, df_plot["Mult Altitude physio"].values, label="Altitude physio", lw=1.5, ls=":")
        ax2.axhline(1.0, color="gray", lw=0.8, ls="-")
        ax2.set_xlabel("Km")
        ax2.set_ylabel("Multiplicateur")
        ax2.set_title("Décomposition des facteurs de ralentissement")
        ax2.legend()
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

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
            zoom=13, pitch=0
        )
        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": df_points[["lon", "lat"]].values.tolist(), "name": "Parcours"}],
            get_path="path", get_color=[255, 0, 0], width_min_pixels=4
        )
        deck = pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=view,
            layers=[path_layer],
            tooltip={"text": "{name}"}
        )
        st.pydeck_chart(deck, use_container_width=True)

        st.subheader("📊 Profil d'altitude")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        total_m_p = 0.0
        cumdists = [0.0]
        for i in range(1, len(points)):
            d_ = haversine_m(points[i-1].latitude, points[i-1].longitude,
                             points[i].latitude, points[i].longitude)
            total_m_p += d_
            cumdists.append(total_m_p)
        x_km = np.array(cumdists) / 1000.0
        y_elev_raw = np.array([getattr(p, "elevation", 0.0) or 0.0 for p in points], dtype=float)

        w = int(elev_smooth_window)
        if w >= 3 and y_elev_raw.size >= w:
            if w % 2 == 0: w += 1
            y_s = np.convolve(y_elev_raw, np.ones(w)/w, mode="same")
            ax3.plot(x_km, y_s, lw=2, label="GPS lissé")
            ax3.plot(x_km, y_elev_raw, lw=1, alpha=0.25, label="GPS brut")
        else:
            ax3.plot(x_km, y_elev_raw, lw=2, label="GPS")

        if use_dem and dem_elevations is not None and len(dem_elevations) == len(points):
            y_dem = np.array([e if e is not None else 0.0 for e in dem_elevations], dtype=float)
            ax3.plot(x_km, y_dem, lw=2, ls="--", label="DEM corrigé", color="green")

        ax3.set_xlabel("Distance (km)")
        ax3.set_ylabel("Altitude (m)")
        ax3.set_title("Profil d'altitude du parcours")
        ax3.legend()
        ax3.grid(alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)

    except Exception as e:
        st.error(f"Impossible d'afficher la carte/profil : {e}")

st.markdown("---")
st.caption(
    "v2 — Minetti (2002) grade model · WBGT Stull (2011) · "
    "Open-Topo-Data DEM · Altitude hypoxia · HR drift analysis · LOO cross-validation · "
    "Fatigue cumulative D+ · Open-Meteo weather"
)
