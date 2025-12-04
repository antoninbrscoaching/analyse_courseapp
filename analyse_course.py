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
        times = []
        for msg in fit.get_messages("record"):
            data = {d.name: d.value for d in msg}
            if data.get("position_lat") and data.get("position_long"):
                lat = data["position_lat"] * (180 / 2**31)
                lon = data["position_long"] * (180 / 2**31)
                elev = data.get("altitude", 0)
                dist = data.get("distance", 0)
                records.append((lat, lon, elev, dist))
                ts = data.get("timestamp")
                if ts:
                    times.append(ts)
        df = pd.DataFrame(records, columns=["lat", "lon", "elev", "dist"])
        if df.empty:
            return None
        dup = float(np.sum(np.diff(df["elev"]).clip(min=0)))
        ddn = float(-np.sum(np.diff(df["elev"]).clip(max=0)))
        duration_hms = None
        if times and len(times) >= 2:
            dur = (times[-1] - times[0]).total_seconds()
            if dur > 0:
                duration_hms = seconds_to_hms(dur)
        return dict(distance=round(float(df["dist"].max())), D_up=round(dup), D_down=round(ddn), duration_hms=duration_hms)
    except Exception:
        return None

def parse_tcx(file):
    try:
        file.seek(0)
        data = file.read()
        root = ET.fromstring(data)
    except Exception:
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

# ---------------- Session helpers sécurisés ----------------
def safe_float(val):
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return 0.0
        return float(val)
    except Exception:
        return 0.0

def safe_str(val):
    try:
        if val is None:
            return "00:00:00"
        return str(val)
    except Exception:
        return "00:00:00"

def update_ref_session_safe(i, dist=None, temps=None, dup=None, ddn=None):
    """Met à jour session_state de manière sécurisée"""
    if dist is not None:
        st.session_state[f"dist_{i}"] = safe_float(dist)
    if temps is not None:
        st.session_state[f"temps_{i}"] = safe_str(temps)
    if dup is not None:
        st.session_state[f"dup_{i}"] = safe_float(dup)
    if ddn is not None:
        st.session_state[f"ddn_{i}"] = safe_float(ddn)

# ---------------- Fonctions manquantes / placeholders ----------------
def compute_total_and_cumdist(points):
    cumdists = [0.0]
    total = 0.0
    prev = points[0]
    for p in points[1:]:
        d = prev.distance_3d(p)
        total += d
        cumdists.append(total)
        prev = p
    return total, cumdists, "haversine", {}

def apply_elevation_gradient_route(time_flat_s, d_up_m, d_down_m, segment_length_m, k_up=1.04, k_down=0.996):
    factor = 1 + (d_up_m * (k_up - 1) - d_down_m * (1 - k_down)) / max(segment_length_m,1e-6)
    return max(time_flat_s * factor, 0)

def temp_multiplier_nonlin(temp, opt_temp=12.0, k_hot=0.002, k_cold=0.002):
    diff = temp - opt_temp
    return 1.0 + (k_hot*diff if diff>0 else k_cold*diff)

def fit_loglog_model(refs, k_up=1.04, k_down=0.996):
    return 1.0, 1.0

def predict_time_flat(distance_m, a, K):
    return distance_m / 3.0

def override_with_objective(distance_m, objective_time_hms, K):
    return 1.0

def get_temp_for_datetime(hourly_temps_cache, dt):
    return 12.0

@st.cache_data(ttl=60)
def fetch_open_meteo_hourly(lat, lon, start_iso, end_iso):
    return {}

# ------------------------------------------------------
# ------------------ UI & Inputs -----------------------
# ------------------------------------------------------
st.header("1️⃣ Parcours GPX")
gpx_file = st.file_uploader("📂 Importer un fichier GPX", type=["gpx"])
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

refs = []
for i in range(1, st.session_state.n_refs + 1):
    st.markdown(f"#### Référence {i}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        use_file = st.checkbox(f"Importer fichier (FIT/TCX) ?", key=f"use_file_{i}")

    default_dist = st.session_state.get(f"dist_{i}", 5000 * i)
    default_temps = st.session_state.get(f"temps_{i}", "0:40:00")
    default_dup = st.session_state.get(f"dup_{i}", 0.0)
    default_ddn = st.session_state.get(f"ddn_{i}", 0.0)

    # --- Parsing FIT/TCX ---
    dist_f = dup_f = ddn_f = None
    temps_f = None
    file_in = None
    with c6:
        file_in = st.file_uploader(f"FIT/TCX {i}", type=["fit","tcx"], key=f"fileref_{i}") if use_file else None

    if file_in:
        name = getattr(file_in, "name", "") or ""
        try:
            if name.lower().endswith(".fit"):
                data_fit = parse_fit(file_in)
                if data_fit:
                    dist_f = data_fit.get("distance", None)
                    dup_f = data_fit.get("D_up", None)
                    ddn_f = data_fit.get("D_down", None)
                    temps_f = data_fit.get("duration_hms", None)
            elif name.lower().endswith(".tcx"):
                tcx_res = parse_tcx(file_in)
                if tcx_res:
                    dist_f = int(round(tcx_res.get("distance",0)))
                    dup_f = int(round(tcx_res.get("D_up",0)))
                    ddn_f = int(round(tcx_res.get("D_down",0)))
                    temps_f = tcx_res.get("duration_hms")
        except Exception:
            pass

    # --- Mise à jour sécurisée avant widgets ---
    update_ref_session_safe(i, dist_f, temps_f, dup_f, ddn_f)

    with c2:
        dist = st.number_input(f"Dist {i} (m)", value=st.session_state.get(f"dist_{i}", default_dist), key=f"dist_{i}")
    with c3:
        temps = st.text_input(f"Temps {i} (h:mm:ss)", value=st.session_state.get(f"temps_{i}", default_temps), key=f"temps_{i}")
    with c4:
        dup = st.number_input(f"D+ {i}", value=st.session_state.get(f"dup_{i}", default_dup), key=f"dup_{i}")
    with c5:
        ddn = st.number_input(f"D- {i}", value=st.session_state.get(f"ddn_{i}", default_ddn), key=f"ddn_{i}")

    refs.append({
        "distance": float(st.session_state.get(f"dist_{i}", dist or 0.0)),
        "temps": str(st.session_state.get(f"temps_{i}", temps or "00:00:00")),
        "D_up": float(st.session_state.get(f"dup_{i}", dup or 0.0)),
        "D_down": float(st.session_state.get(f"ddn_{i}", ddn or 0.0))
    })

# ==============================================================
# TEMPS CORRIGÉS 0% & 12°C POUR TOUTES LES RÉFÉRENCES IMPORTÉES
# ==============================================================

st.subheader("⏱️ Temps corrigés des références (0% & 12°C)")

# use default coeff if widgets not yet set
_default_k_up = 1.040
_default_k_down = 0.996

for i in range(1, st.session_state.n_refs + 1):

    # read from session_state (this ensures we reflect imports)
    dist = st.session_state.get(f"dist_{i}", 0.0)
    temps = st.session_state.get(f"temps_{i}", "00:00:00")
    dup = float(st.session_state.get(f"dup_{i}", 0.0))
    ddn = float(st.session_state.get(f"ddn_{i}", 0.0))

    try:
        # Conversion temps → secondes
        secs = hms_to_seconds(temps)

        # Corr. pente : utilisation de la correction en % de pente
        t_corr_pente = apply_elevation_gradient_route(
            time_flat_s=secs,
            d_up_m=dup,
            d_down_m=ddn,
            segment_length_m=dist if dist > 0 else 1000.0,   # NOTE : dist en mètres, fallback to 1000m
            k_up=_default_k_up,
            k_down=_default_k_down,
        )

        # Corr. température : neutralisation à 12°C
        mult_temp_ref = temp_multiplier_nonlin(12.0)
        # neutralisation means divide by multiplier to get to 12°C baseline
        t_corr_final = t_corr_pente / mult_temp_ref if mult_temp_ref != 0 else t_corr_pente

        t_corr_hms = seconds_to_hms(t_corr_final)

        # single-line f-string (no unintended newline)
        st.markdown(f"**Référence {i} — Temps corrigé 0% & 12°C : `{t_corr_hms}`**  *(Temps brut : {temps}, D+ {dup} m / D- {ddn} m)*")

    except Exception as e:
        st.warning(f"Impossible de corriger la référence {i} : {e}")

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

    # 1) collecte des dates/heures saisies par l'utilisateur
    for idx, r in enumerate(refs):
        cA, cB = st.columns(2)
        with cA:
            ref_date = st.date_input(f"Date référence #{idx+1}", key=f"ref_date_{idx}", value=date.today())
        with cB:
            ref_time = st.time_input(f"Heure référence #{idx+1}", key=f"ref_time_{idx}", value=time(9, 0))
        refs[idx]["ref_datetime"] = datetime.combine(ref_date, ref_time)

    # Cherche plage historique min/max
    min_ref_date = None
    max_ref_date = date_course
    for r in refs:
        rd = r.get("ref_datetime")
        if rd:
            d = rd.date()
            if min_ref_date is None or d < min_ref_date:
                min_ref_date = d
            if d > max_ref_date:
                max_ref_date = d
    if min_ref_date is None:
        min_ref_date = date_course

    # Chargement meteo historique
    try:
        hourly_temps_cache = fetch_historical_range(lat_input, lon_input, min_ref_date, max_ref_date)
    except Exception:
        hourly_temps_cache = {}

    # --- Ajout : récupération du D+ / D- de la référence (via entrée user OU GPX)
    def compute_ref_slope(r):
        # Cas GPX dans la référence
        if r.get("ref_distance_km") and r.get("ref_altitude_gain"):
            d_km = r["ref_distance_km"]
            up = r["ref_altitude_gain"]
            down = r.get("ref_altitude_loss", 0)
            # pente moyenne pondérée
            slope = ((up - down) / (d_km*1000)) * 100
        else:
            slope = 0
        return slope

    # 3) correction complète (temp + pente)
    for idx, r in enumerate(refs):

        rd = r.get("ref_datetime")
        key = f"temps_{idx+1}"
        temps_brut_hms = st.session_state.get(key, r.get("temps", "00:00:00"))
        secs_brut = hms_to_seconds(temps_brut_hms)

        if secs_brut == 0:
            continue

        # ---------- Correction TEMPÉRATURE ----------
        try:
            t_ref = get_temp_for_datetime(hourly_temps_cache, rd)
        except:
            t_ref = 12.0  # fallback = aucune correction

        if use_temp_coeff:
            mult_temp = temp_multiplier_nonlin(
                t_ref,
                opt_temp=opt_temp,
                k_hot=k_temp_hot,
                k_cold=k_temp_cold
            )
        else:
            mult_temp = 1.0

        secs_corr_temp = secs_brut / mult_temp

        # ---------- Correction PENTE ----------
        slope_ref = compute_ref_slope(r)

        if use_slope_coeff:
            mult_slope = pente_multiplier_nonlin(
                slope_ref,
                k_up=k_grade_up,
                k_down=k_grade_down
            )
        else:
            mult_slope = 1.0

        secs_corr_full = secs_corr_temp / mult_slope

        temps_std = seconds_to_hms(secs_corr_full)

        # Mise à jour session + structure
        st.session_state[key] = temps_std
        refs[idx]["temps_standardise"] = temps_std
        refs[idx]["_temp_ref"] = t_ref
        refs[idx]["_slope_ref"] = slope_ref

        st.success(
            f"Réf #{idx+1} — Temp réelle: {t_ref}°C, pente: {slope_ref:.2f}% → Temps standardisé (12°C & pancake) : `{temps_std}`"
        )

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
    if not points or len(points) < 2:
        raise ValueError("GPX invalide ou trop court.")

    total_m, dists, method_used, debug = compute_total_and_cumdist(points)
    distance_gpx_km = total_m / 1000.0

    if not distance_cible_km or distance_cible_km <= 0:
        distance_cible_km = distance_gpx_km

    facteur_dist = distance_cible_km / max(distance_gpx_km, 1e-6)
    total_corr = total_m * facteur_dist
    dists_corr = np.asarray([d * facteur_dist for d in dists])

    elev_list = np.asarray([p.elevation or 0 for p in points])
    if len(dists_corr) != len(elev_list):
        xs = np.linspace(0, total_m, len(elev_list))
        new_x = np.linspace(0, total_m, len(dists_corr))
        elev_list = np.interp(new_x, xs, elev_list)

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

    a, K = fit_loglog_model(refs_for_fit, k_up=(local_k_up if apply_elev else 1.0), k_down=(local_k_down if apply_elev else 1.0))

    if objective_time_hms:
        try:
            a_override = override_with_objective(int(distance_cible_km * 1000), objective_time_hms, K)
        except Exception as e:
            raise

    distance_cible_m = int(distance_cible_km * 1000)
    base_flat_total = predict_time_flat(distance_cible_m, (a_override if objective_time_hms else a), K)
    base_s_per_km_flat = base_flat_total / distance_cible_km if distance_cible_km > 0 else base_flat_total

    km_marks = [i * 1000 for i in range(1, int(total_corr // 1000) + 1)]
    last_seg = total_corr - (int(total_corr // 1000) * 1000)
    if last_seg > 1e-6:
        km_marks.append(total_corr)

    segment_infos = []
    cum_time_temp = 0.0
    dt_depart = datetime.combine(date_course_local, heure_course_local)
    for i, d in enumerate(km_marks):
        e_cur = float(np.interp(d, dists_corr, elev_list))
        e_prev = float(np.interp(max(d - 1000.0, 0.0), dists_corr, elev_list)) if i > 0 else e_cur
        d_up = max(0.0, e_cur - e_prev)
        d_down = max(0.0, e_prev - e_cur)

        seg_length_m = 1000.0 if (i < len(km_marks) - 1 or last_seg < 1e-6) else (d - km_marks[-2] if len(km_marks) >= 2 else d)
        t_km_flat = base_s_per_km_flat * (seg_length_m / 1000.0)

        t_km_after_elev = apply_elevation_gradient_route(t_km_flat, d_up, d_down, segment_length_m=seg_length_m, k_up=local_k_up, k_down=local_k_down) if apply_elev else t_km_flat

        if apply_fatigue and local_fatigue_rate > 0 and total_corr > 0:
            progression = d / total_corr
            t_km_after_fatigue = t_km_after_elev * (1.0 + (local_fatigue_rate / 100.0) * progression)
        else:
            t_km_after_fatigue = t_km_after_elev

        passage_dt = dt_depart + timedelta(seconds=cum_time_temp + t_km_after_fatigue / 2.0)
        temp_at_passage = get_temp_for_datetime(hourly_temps_cache, passage_dt)
        if apply_temp and temp_at_passage is not None:
            mult_temp = temp_multiplier_nonlin(temp_at_passage, opt_temp=local_opt_temp, k_hot=local_k_temp_hot, k_cold=local_k_temp_cold)
            t_km_after_temp = t_km_after_fatigue * mult_temp
        else:
            mult_temp = 1.0
            t_km_after_temp = t_km_after_fatigue

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

    if objective_time_hms:
        objective_seconds = hms_to_seconds(objective_time_hms)
        sum_raw = sum(s["t_raw"] for s in segment_infos)
        if sum_raw <= 0:
            scale = 1.0
        else:
            scale = objective_seconds / sum_raw
    else:
        scale = 1.0

    results = []
    cum_time = 0.0
    for seg in segment_infos:
        t_km = seg["t_raw"] * scale
        cum_time += t_km
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
