import streamlit as st
import math
import gpxpy
from fitparse import FitFile
import requests
from datetime import datetime, timedelta, date, time
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from io import BytesIO

# ------------------------------------------------------
# ⚙️ CONFIGURATION
# ------------------------------------------------------
st.set_page_config(page_title="Prédiction course route", layout="wide")
st.title("🏃‍♂️ Analyse & Prédiction de course (GPX + FIT + TCX + Météo + Fatigue linéaire) — Route")

# ------------------------------------------------------
# 🧩 UTILITAIRES GÉNÉRAUX
# ------------------------------------------------------
def hms_to_seconds(hms: str) -> int:
    try:
        h, m, s = map(int, hms.strip().split(":"))
        return h * 3600 + m * 60 + s
    except Exception:
        return 0

def seconds_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"

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
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# Classe minimale "point-like" pour TCX parsing
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

# ---------------- Parsing GPX/FIT/TCX ----------------
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
        st.error(f"Erreur lors du parsing GPX : {e}")
        return None, []

def gpx_to_df(points):
    return pd.DataFrame([{"lat": p.latitude, "lon": p.longitude, "elev": p.elevation or 0, "time": getattr(p, "time", None)} for p in points])

def parse_fit(file):
    try:
        file.seek(0)
        fit = FitFile(file)
        fit.parse()
        records = []
        for msg in fit.get_messages("record"):
            data = {d.name: d.value for d in msg}
            if data.get("position_lat") and data.get("position_long"):
                lat = data["position_lat"] * (180 / 2**31)
                lon = data["position_long"] * (180 / 2**31)
                elev = data.get("altitude", 0)
                dist = data.get("distance", 0)
                records.append((lat, lon, elev, dist))
        df = pd.DataFrame(records, columns=["lat", "lon", "elev", "dist"])
        if df.empty:
            return None
        dup = float(np.sum(np.diff(df["elev"]).clip(min=0)))
        ddn = float(-np.sum(np.diff(df["elev"]).clip(max=0)))
        return dict(distance=round(float(df["dist"].max())), D_up=round(dup), D_down=round(ddn))
    except Exception:
        return None

def parse_tcx(file):
    try:
        file.seek(0)
        data = file.read()
        root = ET.fromstring(data)
    except Exception as e:
        return None

    trackpoints = root.findall('.//{*}Trackpoint')
    pts, times, elevs = [], [], []
    for tp in trackpoints:
        lat_elem = tp.find('.//{*}LatitudeDegrees')
        lon_elem = tp.find('.//{*}LongitudeDegrees')
        alt_elem = tp.find('.//{*}AltitudeMeters')
        time_elem = tp.find('.//{*}Time')
        if lat_elem is None or lon_elem is None:
            continue
        lat = float(lat_elem.text)
        lon = float(lon_elem.text)
        elev = float(alt_elem.text) if alt_elem is not None and alt_elem.text else 0.0
        t = None
        if time_elem is not None and time_elem.text:
            try:
                t = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00')).replace(tzinfo=None)
                times.append(t)
            except Exception:
                t = None
        elevs.append(elev)
        pts.append(SimplePoint(lat, lon, elev, t))

    if not pts:
        return None

    total = 0.0
    dists = [0.0]
    for i in range(1, len(pts)):
        total += pts[i].distance_3d(pts[i-1])
        dists.append(total)

    dup = float(np.sum(np.diff(elevs).clip(min=0))) if elevs else 0.0
    ddn = float(-np.sum(np.diff(elevs).clip(max=0))) if elevs else 0.0
    duration_hms = None
    if len(times) >= 2:
        dur = (times[-1] - times[0]).total_seconds()
        if dur > 0:
            duration_hms = seconds_to_hms(dur)

    return {
        "points": pts,
        "distance": round(total),
        "D_up": round(dup),
        "D_down": round(ddn),
        "duration_hms": duration_hms
    }

# ------------------------------------------------------
# météo historique (open-meteo archive)
# ------------------------------------------------------
@st.cache_data(ttl=60*60)
def fetch_open_meteo_hourly(lat, lon, start_date_str, end_date_str):
    base = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": lat, "longitude": lon, "start_date": start_date_str, "end_date": end_date_str, "hourly": "temperature_2m", "timezone": "UTC"}
    try:
        r = requests.get(base, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
        times = j.get("hourly", {}).get("time", [])
        temps = j.get("hourly", {}).get("temperature_2m", [])
        out = {datetime.fromisoformat(t): float(temp) for t, temp in zip(times, temps)}
        return out
    except Exception:
        return {}

def get_temp_for_datetime(hourly_dict, target_dt):
    if not hourly_dict:
        return None
    keys = sorted(hourly_dict.keys())
    if not keys:
        return None
    if target_dt in hourly_dict:
        return hourly_dict[target_dt]
    lower, upper = None, None
    for k in keys:
        if k <= target_dt:
            lower = k
        if k > target_dt:
            upper = k
            break
    if lower is None:
        return hourly_dict[keys[0]]
    if upper is None:
        return hourly_dict[keys[-1]]
    t0, t1 = lower, upper
    v0, v1 = hourly_dict[t0], hourly_dict[t1]
    frac = (target_dt - t0).total_seconds() / (t1 - t0).total_seconds()
    return float(v0 + (v1 - v0) * frac)

# ------------------------------------------------------
# Modèle log-log & utilitaires
# ------------------------------------------------------

def fit_loglog_model(refs, k_up=1.0, k_down=1.0):
    xs, ys = [], []
    for r in refs:
        d = float(r.get("distance", 0) or 0)
        t_raw = hms_to_seconds(r.get("temps", "0:00:00") or "0:00:00")
        dup = float(r.get("D_up", 0) or 0)
        ddn = float(r.get("D_down", 0) or 0)
        if d <= 0 or t_raw <= 0:
            continue
        # --- Correction appliquée : coefficients par % de pente ---
        g_up = (dup / d) * 100.0  # pente montée en %
        g_down = (ddn / d) * 100.0  # pente descente en %
        elev_factor = (k_up ** g_up) * (k_down ** g_down)
        t_eq = t_raw / elev_factor if elev_factor > 0 else t_raw
        # ----------------------------------------------------------
        if t_eq <= 0:
            continue
        xs.append(math.log(d))
        ys.append(math.log(t_eq))
    n = len(xs)
    if n < 2:
        raise ValueError("Il faut deux références minimum valides.")
    sum_x, sum_y = sum(xs), sum(ys)
    sum_xx = sum(x*x for x in xs)
    sum_xy = sum(x*y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x**2
    if denom == 0:
        raise ValueError("Distances identiques ou colinéaires.")
    K = (n * sum_xy - sum_x * sum_y) / denom
    a = math.exp((sum_y - K * sum_x) / n)
    return a, K

def predict_time_flat(distance_m, a, K):
    return a * (distance_m ** K)

def override_with_objective(distance_obj_m, time_obj_hms, K):
    t_obj = hms_to_seconds(time_obj_hms or "0:00:00")
    if t_obj <= 0 or distance_obj_m <= 0:
        raise ValueError("Temps objectif invalide ou distance nulle.")
    return t_obj / (distance_obj_m ** K)

def temp_multiplier_nonlin(temp_c, opt_temp=12.0, k_hot=0.002, k_cold=0.002, power=1.6):
    if temp_c is None:
        return 1.0
    delta = temp_c - opt_temp
    if delta >= 0:
        return 1.0 + k_hot * (delta ** power)
    else:
        return 1.0 + k_cold * ((-delta) ** power)

def apply_elevation_gradient_route(time_flat_s, d_up_m, d_down_m, segment_length_m=1000.0, k_up=1.040, k_down=0.996):
    if segment_length_m <= 0:
        return time_flat_s
    g_up = (d_up_m / segment_length_m) * 100.0
    g_down = (d_down_m / segment_length_m) * 100.0
    delta_up = float(k_up) - 1.0
    delta_down = float(k_down) - 1.0
    uphill_factor = math.exp(delta_up * g_up)
    downhill_factor = math.exp(delta_down * g_down)
    return float(time_flat_s * uphill_factor * downhill_factor)

# ---------------- Calcul distance cumulée ----------------
def compute_total_and_cumdist(points):
    if not points or len(points) < 2:
        return 0.0, [0.0], "none", {}

    total_3d = 0.0
    dists_3d = [0.0]
    for i in range(1, len(points)):
        try:
            seg = points[i].distance_3d(points[i-1])
        except Exception:
            seg = 0.0
        total_3d += seg
        dists_3d.append(total_3d)

    total_hav = 0.0
    dists_hav = [0.0]
    for i in range(1, len(points)):
        lat1, lon1 = points[i-1].latitude, points[i-1].longitude
        lat2, lon2 = points[i].latitude, points[i].longitude
        seg = haversine_m(lat1, lon1, lat2, lon2)
        total_hav += seg
        dists_hav.append(total_hav)

    debug = {"total_3d": total_3d, "total_hav": total_hav, "n_points": len(points)}
    if total_3d <= 0 or abs(total_3d - total_hav) / max(total_hav, 1e-6) > 0.02:
        return total_hav, dists_hav, "haversine", debug
    else:
        return total_3d, dists_3d, "3d", debug

# ---------------- UI & Inputs ----------------
st.header("1️⃣ Parcours GPX")
gpx_file = st.file_uploader("📂 Importer un fichier GPX", type=["gpx"])
# if a GPX is uploaded, parse immediately to store original distance
if gpx_file:
    try:
        gpx_tmp, pts_tmp = parse_gpx_points(gpx_file)
        total_m_tmp, _, method_tmp, debug_tmp = compute_total_and_cumdist(pts_tmp)
        st.session_state["gpx_original_distance_km"] = total_m_tmp / 1000.0
    except Exception:
        st.session_state["gpx_original_distance_km"] = None

st.header("2️⃣ Courses de référence (manuel ou FIT/TCX)")
if "n_refs" not in st.session_state:
    st.session_state.n_refs = 3
cols = st.columns([1,1])
with cols[0]:
    if st.button("➕ Ajouter (max 6)") and st.session_state.n_refs < 6:
        st.session_state.n_refs += 1
with cols[1]:
    if st.button("➖ Retirer") and st.session_state.n_refs > 1:
        st.session_state.n_refs -= 1

# --- Fonction helper pour temps plat / 12°C ---
def get_flat_reference_time(ref):
    secs = hms_to_seconds(ref.get("temps", "0:00:00"))
    return seconds_to_hms(secs)

refs = []
for i in range(1, st.session_state.n_refs + 1):
    st.markdown(f"#### Référence {i}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        use_file = st.checkbox(f"Importer fichier (FIT/TCX) ?", key=f"use_file_{i}")
    with c2:
        dist = st.number_input(f"Dist {i} (m)", value=5000 * i, key=f"dist_{i}")
    with c3:
        temps = st.text_input(f"Temps {i} (h:mm:ss)", value="0:40:00", key=f"temps_{i}")
    with c4:
        dup = st.number_input(f"D+ {i}", value=0.0, key=f"dup_{i}")
    with c5:
        ddn = st.number_input(f"D- {i}", value=0.0, key=f"ddn_{i}")
    with c6:
        file_in = st.file_uploader(f"FIT/TCX {i}", type=["fit","tcx"], key=f"fileref_{i}") if use_file else None
        if file_in:
            name = getattr(file_in, "name", "") or ""
            if name.lower().endswith(".fit"):
                data_fit = parse_fit(file_in)
                if data_fit:
                    dist, dup, ddn = data_fit["distance"], data_fit["D_up"], data_fit["D_down"]
                    st.info(f"✔ FIT détecté : {dist}m | D+{dup} | D-{ddn}")
                else:
                    st.warning("Fichier FIT non exploitable.")
            elif name.lower().endswith(".tcx"):
                tcx_res = parse_tcx(file_in)
                if tcx_res:
                    dist = int(round(tcx_res["distance"]))
                    dup = int(round(tcx_res["D_up"]))
                    ddn = int(round(tcx_res["D_down"]))
                    if tcx_res.get("duration_hms"):
                        temps = tcx_res["duration_hms"]
                    st.info(f"✔ TCX détecté : {dist}m | D+{dup} | D-{ddn} | dur: {temps}")
                else:
                    st.warning("Fichier TCX non exploitable.")
            else:
                file_in.seek(0)
                data_fit = parse_fit(file_in)
                if data_fit:
                    dist, dup, ddn = data_fit["distance"], data_fit["D_up"], data_fit["D_down"]
                    st.info(f"✔ FIT détecté : {dist}m | D+{dup} | D-{ddn}")
                else:
                    file_in.seek(0)
                    tcx_res = parse_tcx(file_in)
                    if tcx_res:
                        dist = int(round(tcx_res["distance"]))
                        dup = int(round(tcx_res["D_up"]))
                        ddn = int(round(tcx_res["D_down"]))
                        if tcx_res.get("duration_hms"):
                            temps = tcx_res["duration_hms"]
                        st.info(f"✔ TCX détecté : {dist}m | D+{dup} | D-{ddn} | dur: {temps}")
                    else:
                        st.warning("Fichier non exploitable.")

# --- Affichage du temps corrigé 0% pente & 12°C ---
secs = hms_to_seconds(temps)
# si on a la température de référence historique, sinon 12°C
temp_ref = r.get("_temp_ref", 12.0)

# appliquer l’inverse des effets de pente (neutralisation)
t_corr = apply_elevation_gradient_route(
    secs,
    dup,
    ddn,
    segment_length_m=dist,
    k_up=k_up if use_elev_coeff else 1.0,
    k_down=k_down if use_elev_coeff else 1.0
)
# diviser par le facteur de température pour neutraliser l’effet
if use_temp_coeff and temp_ref is not None:
    mult_temp_ref = temp_multiplier_nonlin(temp_ref, opt_temp=opt_temp, k_hot=k_temp_hot, k_cold=k_temp_cold)
    t_corr /= mult_temp_ref

temps_flat_0pct_12C = seconds_to_hms(t_corr)
st.markdown(f"*Temps corrigé 0% pente & 12°C : {temps_flat_0pct_12C}*")

st.header("3️⃣ Paramètres modèle")
c1, c2 = st.columns(2)
with c1:
    use_elev_coeff = st.checkbox("Activer coefficients montée/descente 🎢", value=True)
    if use_elev_coeff:
        k_up = st.number_input("Coefficient montée (k_up)", value=1.040, format="%.3f", step=0.001)
        k_down = st.number_input("Coefficient descente (k_down)", value=0.996, format="%.3f", step=0.001)
    else:
        k_up = 1.0; k_down = 1.0
with c2:
    use_temp_coeff = st.checkbox("Activer coefficients température 🌡️", value=True)
    if use_temp_coeff:
        k_temp_hot = st.number_input("Sensibilité chaude (k_temp_hot)", value=0.002, format="%.3f", step=0.001)
        k_temp_cold = st.number_input("Sensibilité froide (k_temp_cold)", value=0.002, format="%.3f", step=0.001)
        opt_temp = st.number_input("Température optimale (°C)", value=12.0, format="%.1f", step=0.5)
    else:
        k_temp_hot = 0.0; k_temp_cold = 0.0; opt_temp = 12.0

col1, col2 = st.columns(2)
with col1:
    lat_input = st.number_input("Latitude (pour météo)", value=48.8566, format="%.6f")
    lon_input = st.number_input("Longitude (pour météo)", value=2.3522, format="%.6f")
    use_hist_refs = st.checkbox("Recalibrer les références avec météo historique ?", value=False)
with col2:
    date_course = st.date_input("Date de la course (Jour J)", value=date.today())
    heure_course = st.time_input("Heure de départ (Jour J)", value=time(9, 0))

if use_hist_refs:
    st.markdown("**Dates/Heures pour les références**")
    for idx, r in enumerate(refs):
        cA, cB = st.columns(2)
        with cA:
            ref_date = st.date_input(f"Date référence #{idx+1}", key=f"ref_date_{idx}", value=date.today())
        with cB:
            ref_time = st.time_input(f"Heure référence #{idx+1}", key=f"ref_time_{idx}", value=time(9, 0))
        refs[idx]["ref_datetime"] = datetime.combine(ref_date, ref_time)

st.header("3️⃣ bis. Fatigue linéaire")
fatigue_active = st.checkbox("Activer fatigue ?", value=False)
fatigue_rate = 0.0
if fatigue_active:
    fatigue_rate = st.slider("Régression finale (%)", 0.0, 30.0, 5.0, 0.5)

# ---------------- helper to fetch historic temps (cached) ----------------
@st.cache_data(ttl=60)
def fetch_historical_range(lat, lon, start_date, end_date):
    return fetch_open_meteo_hourly(lat, lon, start_date.isoformat(), end_date.isoformat())

# ---------------- run_prediction_df (retourne df + meta) ----------------
def run_prediction_df(distance_cible_km,
                      refs_input,
                      points,
                      date_course_local,
                      heure_course_local,
                      use_hist_for_refs_local=False,
                      apply_elev=True,
                      apply_temp=True,
                      apply_fatigue=True,
                      objective_time_hms=None,
                      local_k_up=1.040, local_k_down=0.996,
                      local_k_temp_hot=0.002, local_k_temp_cold=0.002, local_opt_temp=12.0,
                      local_fatigue_rate=0.0):
    """
    Calcule et renvoie DataFrame km-par-km et métadonnées.
    Si objective_time_hms est fourni, on applique d'abord les effets par segment
    puis on SCALE pour que la somme des segments = temps objectif (donc total forcé exact).
    """
    if not points or len(points) < 2:
        raise ValueError("GPX invalide ou trop court.")

    # distance/cum dists robustes
    total_m, dists, method_used, debug = compute_total_and_cumdist(points)
    distance_gpx_km = total_m / 1000.0

    # sécurité distance cible
    if not distance_cible_km or distance_cible_km <= 0:
        distance_cible_km = distance_gpx_km

    facteur_dist = distance_cible_km / max(distance_gpx_km, 1e-6)
    total_corr = total_m * facteur_dist
    dists_corr = np.asarray([d * facteur_dist for d in dists])

    elev_list = np.asarray([p.elevation or 0 for p in points])
    if len(dists_corr) != len(elev_list):
        # resample elev_list to len(dists_corr) via simple interp
        xs = np.linspace(0, total_m, len(elev_list))
        new_x = np.linspace(0, total_m, len(dists_corr))
        elev_list = np.interp(new_x, xs, elev_list)

    # préparer références (recalées si historique demandé)
    center_lat = np.mean([p.latitude for p in points])
    center_lon = np.mean([p.longitude for p in points])
    min_ref_date = None; max_ref_date = date_course_local
    if use_hist_for_refs_local:
        for r in refs_input:
            rd = r.get("ref_datetime")
            if rd:
                d = rd.date()
                if min_ref_date is None or d < min_ref_date:
                    min_ref_date = d
                if d > max_ref_date:
                    max_ref_date = d
    if min_ref_date is None:
        min_ref_date = date_course_local

    try:
        hourly_temps_cache = fetch_historical_range(center_lat, center_lon, min_ref_date, max_ref_date)
    except Exception:
        hourly_temps_cache = {}

    # recalibration météo pour refs si demandé
    refs_for_fit = []
    if use_hist_for_refs_local and hourly_temps_cache:
        for r in refs_input:
            rr = r.copy()
            rd = r.get("ref_datetime")
            if rd:
                t_ref = get_temp_for_datetime(hourly_temps_cache, rd)
                mult = temp_multiplier_nonlin(t_ref, opt_temp=local_opt_temp, k_hot=local_k_temp_hot, k_cold=local_k_temp_cold)
                secs = hms_to_seconds(r.get("temps", "0:00:00"))
                if secs > 0 and mult > 0:
                    new_secs = secs / mult
                    rr["temps"] = seconds_to_hms(new_secs)
                    rr["_temp_ref"] = t_ref
            refs_for_fit.append(rr)
    else:
        refs_for_fit = [r.copy() for r in refs_input]

    # Fit log-log
    a, K = fit_loglog_model(refs_for_fit, k_up=(local_k_up if apply_elev else 1.0), k_down=(local_k_down if apply_elev else 1.0))

    # If objective_time_hms provided we still compute 'a' from objective to get target flat pace (but we'll scale segments afterwards)
    if objective_time_hms:
        try:
            a_override = override_with_objective(int(distance_cible_km * 1000), objective_time_hms, K)
        except Exception as e:
            raise

    distance_cible_m = int(distance_cible_km * 1000)
    base_flat_total = predict_time_flat(distance_cible_m, (a_override if objective_time_hms else a), K)
    base_s_per_km_flat = base_flat_total / distance_cible_km if distance_cible_km > 0 else base_flat_total

    # km marks (entiers) + dernier segment fractionnaire si besoin
    km_marks = [i * 1000 for i in range(1, int(total_corr // 1000) + 1)]
    last_seg = total_corr - (int(total_corr // 1000) * 1000)
    if last_seg > 1e-6:
        km_marks.append(total_corr)

    # PRE-CALC: créer la liste des segments et leurs t_km pré-ajustés (avant éventuelle mise à l'échelle objective)
    segment_infos = []
    cum_time_temp = 0.0
    dt_depart = datetime.combine(date_course_local, heure_course_local)
    for i, d in enumerate(km_marks):
        e_cur = float(np.interp(d, dists_corr, elev_list))
        e_prev = float(np.interp(max(d - 1000.0, 0.0), dists_corr, elev_list)) if i > 0 else e_cur
        d_up = max(0.0, e_cur - e_prev)
        d_down = max(0.0, e_prev - e_cur)

        # temps plat sur ce segment (base flat per km) - if last seg is shorter, scale by length
        seg_length_m = 1000.0 if (i < len(km_marks) - 1 or last_seg < 1e-6) else (d - km_marks[-2] if len(km_marks) >= 2 else d)
        # If the segment is fractional, base_s_per_km_flat must be scaled to segment length:
        t_km_flat = base_s_per_km_flat * (seg_length_m / 1000.0)

        # apply elevation
        t_km_after_elev = apply_elevation_gradient_route(t_km_flat, d_up, d_down, segment_length_m=seg_length_m, k_up=local_k_up, k_down=local_k_down) if apply_elev else t_km_flat

        # apply fatigue (progression uses d / total_corr)
        if apply_fatigue and local_fatigue_rate > 0 and total_corr > 0:
            progression = d / total_corr
            t_km_after_fatigue = t_km_after_elev * (1.0 + (local_fatigue_rate / 100.0) * progression)
        else:
            t_km_after_fatigue = t_km_after_elev

        # temperature at passage (mid-segment time using cum_time_temp)
        passage_dt = dt_depart + timedelta(seconds=cum_time_temp + t_km_after_fatigue / 2.0)
        temp_at_passage = get_temp_for_datetime(hourly_temps_cache, passage_dt)
        if apply_temp and temp_at_passage is not None:
            mult_temp = temp_multiplier_nonlin(temp_at_passage, opt_temp=local_opt_temp, k_hot=local_k_temp_hot, k_cold=local_k_temp_cold)
            t_km_after_temp = t_km_after_fatigue * mult_temp
        else:
            mult_temp = 1.0
            t_km_after_temp = t_km_after_fatigue

        # append info (raw segment time before any global scaling)
        segment_infos.append({
            "idx": i,
            "d": d,
            "seg_length_m": seg_length_m,
            "d_up": d_up,
            "d_down": d_down,
            "temp": temp_at_passage,
            "temp_mult": mult_temp,
            "t_raw": t_km_after_temp
        })

        cum_time_temp += t_km_after_temp

    # If objective_time_hms provided -> scale segments so sum == objective_seconds
    if objective_time_hms:
        objective_seconds = hms_to_seconds(objective_time_hms)
        sum_raw = sum(s["t_raw"] for s in segment_infos)
        if sum_raw <= 0:
            scale = 1.0
        else:
            scale = objective_seconds / sum_raw
    else:
        scale = 1.0

    # Now build results with scaled times and cumulative
    results = []
    cum_time = 0.0
    for seg in segment_infos:
        t_km = seg["t_raw"] * scale
        cum_time += t_km
        # compute display pace per km (normalize to per 1000m)
        pace_per_km = (t_km / seg["seg_length_m"]) * 1000.0 if seg["seg_length_m"] > 0 else t_km
        results.append({
            "Km": seg["idx"] + 1 if seg["seg_length_m"] >= 1000 - 1e-6 else f"{seg['idx']+1} ({seg['seg_length_m']:.0f}m)",
            "D+ (m)": round(seg["d_up"], 1),
            "D- (m)": round(seg["d_down"], 1),
            "Temp (°C)": round(seg["temp"], 1) if seg["temp"] is not None else None,
            "Temp Mult.": round(seg["temp_mult"], 4),
            "Temps segment (s)": round(t_km, 1),
            "Allure (min/km)": pace_seconds_to_str_per_km(pace_per_km),
            "Temps cumulé": seconds_to_hms(cum_time),
        })

    df = pd.DataFrame(results)
    total_seconds = sum(s["t_raw"] for s in segment_infos) * scale
    return {
        "df": df,
        "total_seconds": total_seconds,
        "total_human": seconds_to_hms(total_seconds),
        "distance_gpx_km": distance_gpx_km,
        "method_used": method_used,
        "debug": debug,
        "base_flat_total": base_flat_total,
        "a": a, "K": K
    }

# ----------------- Interactions : bouton base et forcé (affichage côte-à-côte) -----------------
st.subheader("4️⃣ Calcul & Comparaison")

# bouton pour calcul base (références)
if st.button("▶️ Calculer prédiction (BASE, d'après références)"):
    if not gpx_file:
        st.error("Importe un fichier GPX d'abord.")
    else:
        gpx, points = parse_gpx_points(gpx_file)
        res_base = run_prediction_df(
            distance_cible_km=None,
            refs_input=refs,
            points=points,
            date_course_local=date_course,
            heure_course_local=heure_course,
            use_hist_for_refs_local=use_hist_refs,
            apply_elev=use_elev_coeff,
            apply_temp=use_temp_coeff,
            apply_fatigue=fatigue_active,
            objective_time_hms=None,
            local_k_up=k_up, local_k_down=k_down,
            local_k_temp_hot=k_temp_hot, local_k_temp_cold=k_temp_cold, local_opt_temp=opt_temp,
            local_fatigue_rate=fatigue_rate
        )
        st.session_state["res_base"] = res_base
        st.success(f"Base calculée — distance GPX détectée: {res_base['distance_gpx_km']:.3f} km (méthode: {res_base['method_used']})")

# zone for forcing options and final calculation
st.markdown("---")
st.markdown("**Forcer distance et/ou temps objectif (produit un tableau 'FORCÉ' distinct)**")
colf1, colf2 = st.columns(2)
with colf1:
    force_distance_checkbox = st.checkbox("Forcer la distance pour la prédiction finale ?", value=False)
    distance_forced_km = st.number_input("Distance forcée (km)", value=5.17, format="%.2f") if force_distance_checkbox else None
with colf2:
    force_time_checkbox = st.checkbox("Forcer un temps objectif ?", value=False)
    time_forced_hms = st.text_input("Temps objectif (h:mm:ss)", value="0:18:30") if force_time_checkbox else None

if st.button("📊 Calculer prédiction finale (FORCÉ si activé)"):
    if not gpx_file:
        st.error("Importe un fichier GPX d'abord.")
    else:
        gpx, points = parse_gpx_points(gpx_file)
        dist_target = distance_forced_km if force_distance_checkbox and distance_forced_km else None

        res_forced = run_prediction_df(
            distance_cible_km=dist_target,
            refs_input=refs,
            points=points,
            date_course_local=date_course,
            heure_course_local=heure_course,
            use_hist_for_refs_local=use_hist_refs,
            apply_elev=use_elev_coeff,
            apply_temp=use_temp_coeff,
            apply_fatigue=fatigue_active,
            objective_time_hms=time_forced_hms if force_time_checkbox else None,
            local_k_up=k_up, local_k_down=k_down,
            local_k_temp_hot=k_temp_hot, local_k_temp_cold=k_temp_cold, local_opt_temp=opt_temp,
            local_fatigue_rate=fatigue_rate
        )
        st.session_state["res_forced"] = res_forced
        st.success(f"Prédiction forcée calculée — cible: {distance_forced_km if distance_forced_km else 'GPX'} km")

# Afficher côte-à-côte si disponibles
if "res_base" in st.session_state or "res_forced" in st.session_state:
    base = st.session_state.get("res_base", None)
    forced = st.session_state.get("res_forced", None)

    left, right = st.columns(2)
    with left:
        st.subheader("📈 Base (d'après références)")
        if base:
            avg_pace_base = base["total_seconds"] / max(base["distance_gpx_km"], 1e-6)
            st.write(f"Distance GPX détectée: {base['distance_gpx_km']:.3f} km (méthode: {base['method_used']})")
            st.write(f"Temps total (base): {base['total_human']}  ({pace_seconds_to_str_per_km(avg_pace_base)} / km)")
            st.dataframe(base["df"], use_container_width=True)
        else:
            st.info("Clique sur 'Calculer prédiction (BASE)' pour générer ce tableau.")

    with right:
        st.subheader("🎯 Forcé (distance/temps forcés)")
        if forced:
            dist_display = (distance_forced_km if (force_distance_checkbox and distance_forced_km) else round(forced['distance_gpx_km'],3))
            avg_pace_forced = forced["total_seconds"] / max(float(dist_display), 1e-6)
            st.write(f"Distance cible: {dist_display} km")
            st.write(f"Temps total (forcé): {forced['total_human']}  ({pace_seconds_to_str_per_km(avg_pace_forced)} / km)")
            st.dataframe(forced["df"], use_container_width=True)
        else:
            st.info("Clique sur 'Calculer prédiction finale (FORCÉ)' pour générer ce tableau.")

# ---------------- Carte + profil (affiche using last used points if exist) ----------------
if gpx_file:
    try:
        gpx, points = parse_gpx_points(gpx_file)
        df_points = gpx_to_df(points)

        st.subheader("🗺️ Carte & Profil (GPX importé)")

        # Carte
        view = pdk.ViewState(
            latitude=df_points.lat.mean(),
            longitude=df_points.lon.mean(),
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
            tooltip={"text":"{name}"}
        )
        st.pydeck_chart(deck, use_container_width=True)

        # Profil d'altitude
        st.subheader("📊 Profil d'altitude")
        plt.figure(figsize=(10, 4))

        # Calcul distance cumulée corrigée
        total_m, cumdists, method_used, debug = compute_total_and_cumdist(points)
        x_km = np.array(cumdists) / 1000.0  # distance en km
        y_elev = np.array([p.elevation or 0 for p in points])

        plt.plot(x_km, y_elev, lw=2)
        plt.xlabel("Distance (km)")
        plt.ylabel("Altitude (m)")
        plt.title(f"Profil d'altitude du parcours (méthode {method_used})")
        plt.grid(alpha=0.3)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Impossible d'afficher la carte/profil : {e}")
