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
from io import BytesIO

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Prédiction course route (refactor)", layout="wide")
st.title("🏃‍♂️ Analyse & Prédiction de course — Refactorisé")

# -------------------------
# MÉTÉO HISTORIQUE (placeholder)
# -------------------------
def get_historical_temp(lat, lon, dt):
    """
    Placeholder : renvoie la température historique estimée pour une date/heure et une localisation.
    À remplacer par un vrai appel API météo.
    """
    # Exemple simple : température pseudo-aléatoire basée sur le jour du mois
    return 10.0 + (dt.day % 10)  # juste pour test

# -------------------------
# UTILITAIRES
# -------------------------
def hms_to_seconds(hms: str) -> int:
    if hms is None:
        return 0
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

# -------------------------
# Modèle & facteurs
# -------------------------
def temp_multiplier_nonlin(temp, opt_temp=12.0, k_hot=0.002, k_cold=0.002):
    """
    Multiplicateur simple : >opt -> 1 + k_hot*(temp-opt)
    <opt -> 1 + k_cold*(opt-temp)
    On retourne >= 0.1 (sécurité).
    """
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

def apply_elevation_gradient_route(t_flat, d_up, d_down, segment_length_m=1000.0, k_up=1.04, k_down=0.996):
    try:
        seg_len = float(segment_length_m) if segment_length_m and segment_length_m > 0 else 1000.0
        up_factor = (float(k_up) - 1.0) * (float(d_up) / seg_len)
        down_factor = (1.0 - float(k_down)) * (float(d_down) / seg_len)
        factor = 1.0 + up_factor + down_factor
        return float(t_flat) * max(0.01, factor)
    except Exception:
        return float(t_flat)

def fit_loglog_model(refs):
    """
    Fit log-log: secs = a * (distance_km ** K)
    refs: list of dicts with 'distance' (m) and 'temps' (str h:mm:ss or seconds)
    Retourne a, K
    """
    X = []
    Y = []
    for r in refs:
        d_m = r.get("distance", None)
        t_raw = r.get("temps")
        if d_m is None or d_m <= 0:
            continue
        if isinstance(t_raw, (int, float, np.number)):
            secs = float(t_raw)
        else:
            secs = hms_to_seconds(str(t_raw))
        if secs <= 0:
            continue
        d_km = float(d_m) / 1000.0
        X.append(math.log(max(1e-6, d_km)))
        Y.append(math.log(max(1e-6, secs)))
    if len(X) >= 2:
        coeffs = np.polyfit(X, Y, 1)  # Y = K * X + log(a)
        K = float(coeffs[0])
        loga = float(coeffs[1])
        a = math.exp(loga)
        if not (0 < a < 1e6):  # sanity
            a = 240.0
        if abs(K) > 5:
            K = 1.0
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
    a = float(objective_seconds) / (d_km ** float(K))
    return a

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
            if data.get("position_lat") is not None and data.get("position_long") is not None:
                lat = data["position_lat"] * (180 / 2**31)
                lon = data["position_long"] * (180 / 2**31)
                elev = data.get("altitude", 0)
                dist = data.get("distance", 0)
                records.append((lat, lon, elev, dist))
            ts = data.get("timestamp")
            if ts:
                times.append(ts)
        if not records:
            return None
        df = pd.DataFrame(records, columns=["lat", "lon", "elev", "dist"])
        dup = float(np.sum(np.diff(df["elev"]).clip(min=0))) if len(df) > 1 else 0.0
        ddn = float(-np.sum(np.diff(df["elev"]).clip(max=0))) if len(df) > 1 else 0.0
        duration_hms = None
        if times and len(times) >= 2:
            dur = (times[-1] - times[0]).total_seconds()
            if dur > 0:
                duration_hms = seconds_to_hms(dur)
        return dict(distance=round(float(df["dist"].max()) if "dist" in df.columns else 0), D_up=round(dup), D_down=round(ddn), duration_hms=duration_hms)
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
    for i in range(1, len(pts)):
        total += pts[i].distance_3d(pts[i-1])

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

# -------------------------
# Helpers safe
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

def clean_time_input(v):
    if v is None:
        return "0:00:00"
    if isinstance(v, (int, float, np.number)):
        return seconds_to_hms(float(v))
    s = str(v).strip()
    if s == "" or s.lower() in ("none", "nan"):
        return "0:00:00"
    return s

# -------------------------
# Recalibration : applique correction élévation & température
# On normalise le temps du ref pour obtenir le temps "plat & temp-opt" (temps_recal)
# -------------------------
def recalibrate_ref_to_ideal(ref, k_up, k_down, k_temp_hot, k_temp_cold, opt_temp):
    """
    ref: dict avec 'distance' (m), 'temps' (h:mm:ss or secs), 'D_up', 'D_down'
    On retire l'effet dénivelé en divisant par factor_elev,
    On considère conditions idéales pour la température (temp_mult = 1),
    donc temps_recal = secs / factor_elev
    (Si tu avais une température par référence -> on pourrait retirer aussi l'effet température)
    """
    secs = hms_to_seconds(ref.get("temps")) if ref.get("temps") is not None else 0
    D_up = safe_float(ref.get("D_up", 0.0))
    D_down = safe_float(ref.get("D_down", 0.0))
    seg_len = safe_float(ref.get("distance", 1000.0))
    seg_len = seg_len if seg_len > 0 else 1000.0
    up_factor = (k_up - 1.0) * (D_up / seg_len)
    down_factor = (1.0 - k_down) * (D_down / seg_len)
    factor_elev = 1.0 + up_factor + down_factor
    if factor_elev == 0:
        factor_elev = 1.0
    secs_no_elev = secs / factor_elev
    # conditions idéales pour la température -> mult_temp = 1
    secs_flat_ideal = secs_no_elev
    return max(0.0, secs_flat_ideal)

def recalibrate_ref_using_current(ref, k_up, k_down, k_temp_hot, k_temp_cold, opt_temp, assumed_temp=None):
    """
    Recalibre la référence en retirant l'effet de l'élévation et de la température
    en supposant qu'on connaît la température réelle (optional).
    Si assumed_temp est None => on ne retire pas l'effet température (temp_mult = 1).
    """
    secs = hms_to_seconds(ref.get("temps")) if ref.get("temps") is not None else 0
    D_up = safe_float(ref.get("D_up", 0.0))
    D_down = safe_float(ref.get("D_down", 0.0))
    seg_len = safe_float(ref.get("distance", 1000.0))
    seg_len = seg_len if seg_len > 0 else 1000.0
    up_factor = (k_up - 1.0) * (D_up / seg_len)
    down_factor = (1.0 - k_down) * (D_down / seg_len)
    factor_elev = 1.0 + up_factor + down_factor
    if factor_elev == 0:
        factor_elev = 1.0
    secs_no_elev = secs / factor_elev
    if assumed_temp is None:
        return max(0.0, secs_no_elev)
    else:
        mult_temp = temp_multiplier_nonlin(assumed_temp, opt_temp=opt_temp, k_hot=k_temp_hot, k_cold=k_temp_cold)
        if mult_temp == 0:
            mult_temp = 1.0
        return max(0.0, secs_no_elev / mult_temp)

# -------------------------
# Prépare les références AVANT fit: calcule temps_recal selon mode
# -------------------------
def prepare_refs_for_fit(refs_input, k_up, k_down, k_temp_hot, k_temp_cold, opt_temp, ideal_refs=False):
    """
    refs_input: liste dicts {distance, temps, D_up, D_down, maybe temps_fichier...}
    Si ideal_refs True => on normalise vers 0% & opt_temp (on retire élévation, temp assume opt)
    Si False => on normalise en retirant l'effet élévation (mais en gardant température inconnue)
    Retourne liste with 'distance' (m) and 'temps' (secs as float)
    """
    prepared = []
    for r in refs_input:
        # fallback distance/temps
        d = safe_float(r.get("distance", 0.0))
        # prefer duration from file if present
        file_dur = r.get("duration_hms_file")
        raw_t = file_dur if file_dur else r.get("temps", "0:00:00")
        # choose recalibration strategy
        if ideal_refs:
            secs_recal = recalibrate_ref_to_ideal({"distance": d, "temps": raw_t, "D_up": r.get("D_up", 0.0), "D_down": r.get("D_down", 0.0)},
                                                 k_up, k_down, k_temp_hot, k_temp_cold, opt_temp)
        else:
            # remove elevation effect, keep temp effect unknown (no assumed temp)
            secs_recal = recalibrate_ref_using_current({"distance": d, "temps": raw_t, "D_up": r.get("D_up", 0.0), "D_down": r.get("D_down", 0.0)},
                                                       k_up, k_down, k_temp_hot, k_temp_cold, opt_temp, assumed_temp=None)
        prepared.append({
            "distance": float(d),
            "temps": float(secs_recal)
        })
    return prepared

# -------------------------
# Calcul principal de prédiction (utilise refs recalées par prepare_refs_for_fit)
# -------------------------
def run_prediction_df(distance_cible_km,
                      refs_input,
                      points,
                      date_course_local,
                      heure_course_local,
                      ideal_refs=False,
                      apply_elev=True,
                      apply_temp=True,
                      apply_fatigue=True,
                      objective_time_hms=None,
                      k_up=1.040, k_down=0.996,
                      k_temp_hot=0.002, k_temp_cold=0.002, opt_temp=12.0,
                      fatigue_rate=0.0):
    if not points or len(points) < 2:
        raise ValueError("GPX invalide ou trop court.")

    # compute total distance and cumulative distances
    total_m = 0.0
    cum = [0.0]
    for i in range(1, len(points)):
        d = (SimplePoint(points[i-1].latitude, points[i-1].longitude, getattr(points[i-1], "elevation", 0))
             .distance_3d(SimplePoint(points[i].latitude, points[i].longitude, getattr(points[i], "elevation", 0))))
        total_m += d
        cum.append(total_m)
    distance_gpx_km = total_m / 1000.0

    if distance_cible_km is None or distance_cible_km <= 0:
        distance_cible_km = distance_gpx_km

    facteur_dist = distance_cible_km / max(distance_gpx_km, 1e-9)
    total_corr = total_m * facteur_dist
    dists_corr = np.asarray([d * facteur_dist for d in cum])

    # elevations array (resampled if needed)
    elev_list = np.asarray([getattr(p, "elevation", 0) or 0 for p in points])
    if len(dists_corr) != len(elev_list):
        xs = np.linspace(0, total_m, len(elev_list))
        new_x = np.linspace(0, total_m, len(dists_corr))
        elev_list = np.interp(new_x, xs, elev_list)

    # -------------------------
    # Affichage références recalibrées
    # -------------------------
    st.subheader("⏱️ Références recalibrées")
    df_refs = pd.DataFrame([{
        "Distance (m)": r["distance"],
        "D+ (m)": r["D_up"],
        "D- (m)": r["D_down"],
        "Temps brut": seconds_to_hms(r["temps_brut"]),
        "Temps conditions idéales": seconds_to_hms(r["temps_ideal"]),
        "Temps météo historique": seconds_to_hms(r["temps_hist"]) if r["temps_hist"] else None,
    } for r in refs_calibrated])
    st.dataframe(df_refs, use_container_width=True)

    # fit model
    a, K = fit_loglog_model(refs_for_fit)

    # override if objective_time provided
    a_override = None
    if objective_time_hms:
        a_override = override_with_objective(int(distance_cible_km * 1000), objective_time_hms, K)
    baseline_seconds_per_km = (a_override if a_override is not None else a)

    # compute base flat total and per-km seconds
    distance_cible_m = int(distance_cible_km * 1000)
    base_flat_total = predict_time_flat(distance_cible_m, baseline_seconds_per_km, K)
    base_s_per_km_flat = base_flat_total / distance_cible_km if distance_cible_km > 0 else base_flat_total

    # create km markers for segments
    km_marks = [i * 1000 for i in range(1, int(total_corr // 1000) + 1)]
    last_seg = total_corr - (int(total_corr // 1000) * 1000)
    if last_seg > 1e-6:
        km_marks.append(total_corr)

    segment_infos = []
    cum_time_temp = 0.0
    dt_depart = datetime.combine(date_course_local, heure_course_local)
    for i, d in enumerate(km_marks):
        # elevation at boundaries (resampled)
        e_cur = float(np.interp(d, dists_corr, elev_list))
        e_prev = float(np.interp(max(d - 1000.0, 0.0), dists_corr, elev_list)) if i > 0 else e_cur
        d_up = max(0.0, e_cur - e_prev)
        d_down = max(0.0, e_prev - e_cur)

        # segment length
        seg_length_m = 1000.0 if (i < len(km_marks) - 1 or last_seg < 1e-6) else (d - km_marks[-2] if len(km_marks) >= 2 else d)
        # flat time for segment (from baseline per-km)
        t_km_flat = base_s_per_km_flat * (seg_length_m / 1000.0)

        # apply elevation factor (per segment)
        t_km_after_elev = apply_elevation_gradient_route(
            t_km_flat, d_up, d_down, segment_length_m=seg_length_m, k_up=k_up, k_down=k_down
        ) if apply_elev else t_km_flat

        # apply fatigue (linear progression along distance)
        if apply_fatigue and fatigue_rate > 0 and total_corr > 0:
            progression = d / total_corr
            t_km_after_fatigue = t_km_after_elev * (1.0 + (fatigue_rate / 100.0) * progression)
        else:
            t_km_after_fatigue = t_km_after_elev

        # placeholder temperature at passage
        passage_dt = dt_depart + timedelta(seconds=cum_time_temp + t_km_after_fatigue / 2.0)
        temp_at_passage = None
        if apply_temp and temp_at_passage is not None:
            mult_temp = temp_multiplier_nonlin(
                temp_at_passage, opt_temp=opt_temp, k_hot=k_temp_hot, k_cold=k_temp_cold
            )
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

    # objective scaling
    if objective_time_hms:
        objective_seconds = hms_to_seconds(objective_time_hms)
        sum_raw = sum(s["t_raw"] for s in segment_infos)
        scale = (objective_seconds / sum_raw) if (sum_raw > 0) else 1.0
    else:
        scale = 1.0

    # build results dataframe
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
        "method_used": "3d_haversine",
        "base_flat_total": base_flat_total,
        "a": baseline_seconds_per_km,
        "K": K
    }

# -------------------------
# UI : Entrées & Références
# -------------------------
st.header("1️⃣ Parcours GPX")
gpx_file = st.file_uploader("📂 Importer un fichier GPX", type=["gpx"])
points = None
if gpx_file:
    gpx, points = parse_gpx_points(gpx_file)
    if points:
        total_m_tmp = sum(SimplePoint(points[i-1].latitude, points[i-1].longitude, getattr(points[i-1], "elevation", 0))
                        .distance_3d(SimplePoint(points[i].latitude, points[i].longitude, getattr(points[i], "elevation", 0)))
                        for i in range(1, len(points)))
        st.session_state["gpx_original_distance_km"] = total_m_tmp / 1000.0
    else:
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

# Collect raw refs (no recalculation here)
refs_raw = []
for i in range(1, st.session_state.n_refs + 1):
    st.markdown(f"#### Référence {i}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        use_file = st.checkbox(f"Importer fichier (FIT/TCX) ?", key=f"use_file_{i}")
    default_dist = st.session_state.get(f"dist_{i}", 5000 * i)
    default_temps = st.session_state.get(f"temps_{i}", "0:40:00")
    default_dup = st.session_state.get(f"dup_{i}", 0.0)
    default_ddn = st.session_state.get(f"ddn_{i}", 0.0)
    with c2:
        dist = st.number_input(f"Dist {i} (m)", value=float(st.session_state.get(f"dist_{i}", default_dist)), key=f"dist_{i}")
    with c3:
        temps = st.text_input(f"Temps {i} (h:mm:ss)", value=str(st.session_state.get(f"temps_{i}", default_temps)), key=f"temps_{i}")
    with c4:
        dup = st.number_input(f"D+ {i}", value=float(st.session_state.get(f"dup_{i}", default_dup)), key=f"dup_{i}")
    with c5:
        ddn = st.number_input(f"D- {i}", value=float(st.session_state.get(f"ddn_{i}", default_ddn)), key=f"ddn_{i}")
    with c6:
        file_in = st.file_uploader(f"FIT/TCX {i}", type=["fit","tcx"], key=f"fileref_{i}") if use_file else None

    # parse uploaded file (si présent)
    duration_hms_file = None
    if file_in:
        name = getattr(file_in, "name", "") or ""
        try:
            if name.lower().endswith(".fit"):
                data_fit = parse_fit(file_in)
                if data_fit:
                    # prefer file values when available
                    dist = data_fit.get("distance", dist)
                    dup = data_fit.get("D_up", dup)
                    ddn = data_fit.get("D_down", ddn)
                    duration_hms_file = data_fit.get("duration_hms", None)
            elif name.lower().endswith(".tcx"):
                tcx_res = parse_tcx(file_in)
                if tcx_res:
                    dist = int(round(tcx_res.get("distance", dist)))
                    dup = int(round(tcx_res.get("D_up", dup)))
                    ddn = int(round(tcx_res.get("D_down", ddn)))
                    duration_hms_file = tcx_res.get("duration_hms")
        except Exception:
            pass

    refs_raw.append({
        "distance": float(dist),
        "temps": str(temps),
        "D_up": float(dup),
        "D_down": float(ddn),
        "duration_hms_file": duration_hms_file
    })

# Show recap (these are raw inputs; they will be recalibrated at prediction time)
st.subheader("⏱️ Récap références (raw)")
for idx, r in enumerate(refs_raw, start=1):
    st.write(f"Réf {idx} — Dist: {r['distance']:.0f} m | Brut: {r['temps']} | D+ {r['D_up']:.0f} m / D- {r['D_down']:.0f} m | Dur file: {r.get('duration_hms_file')}")

# -------------------------
# Recalibrage des références (brut / idéal / météo historique)
# -------------------------
refs_calibrated = []
for r in refs_raw:
    # temps brut
    t_brut = hms_to_seconds(r['temps'])

k_up = 1.04          # facteur de pente montante
k_down = 0.996       # facteur de pente descendante
k_temp_hot = 0.002   # coefficient chaleur
k_temp_cold = 0.002  # coefficient froid
opt_temp = 12.0      # température optimale en °C

# temps sous conditions idéales
t_ideal = recalibrate_ref_to_ideal(
    ref=r,
    k_up=k_up,
    k_down=k_down,
    k_temp_hot=k_temp_hot,
    k_temp_cold=k_temp_cold,
    opt_temp=opt_temp
)

    # temps avec météo historique si activé
    t_hist = None
    if 'use_hist_refs' in locals() and use_hist_refs:
        temp_hist = get_historical_temp(
            lat_input, lon_input, datetime.combine(date_course, heure_course)
        )
        t_hist = recalibrate_ref_using_current(
            ref=r,
            k_up=k_up, k_down=k_down,
            k_temp_hot=k_temp_hot, k_temp_cold=k_temp_cold,
            opt_temp=opt_temp, assumed_temp=temp_hist
        )

    refs_calibrated.append({
        "distance": r["distance"],
        "D_up": r["D_up"],
        "D_down": r["D_down"],
        "temps_brut": t_brut,
        "temps_ideal": t_ideal,
        "temps_hist": t_hist,
        "origine": r.get("duration_hms_file", None)
    })

# -------------------------
# Affichage références recalibrées
# -------------------------
st.subheader("⏱️ Références recalibrées")
df_refs = pd.DataFrame([{
    "Distance (m)": r["distance"],
    "D+ (m)": r["D_up"],
    "D- (m)": r["D_down"],
    "Temps brut": seconds_to_hms(r["temps_brut"]),
    "Temps conditions idéales": seconds_to_hms(r["temps_ideal"]),
    "Temps météo historique": seconds_to_hms(r["temps_hist"]) if r["temps_hist"] else None,
} for r in refs_calibrated])
st.dataframe(df_refs, use_container_width=True)

# -------------------------
# Paramètres modèle UI
# -------------------------
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
        k_temp_hot = st.number_input("Sensibilité chaude (k_temp_hot)", value=0.002, format="%.4f", step=0.0005)
        k_temp_cold = st.number_input("Sensibilité froide (k_temp_cold)", value=0.002, format="%.4f", step=0.0005)
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

st.info("La récupération météo historique n'est pas implémentée dans ce squelette (placeholder).")

# FATIGUE
st.header("3️⃣ bis. Fatigue linéaire")
fatigue_active = st.checkbox("Activer fatigue ?", value=False)
fatigue_rate = 0.0
if fatigue_active:
    fatigue_rate = st.slider("Régression finale (%)", 0.0, 30.0, 5.0, 0.5)

# Option : utiliser CONDITIONS IDÉALES pour recalibrer les références
st.markdown("---")
ideal_refs = st.checkbox("🔧 Recalibrer les références en CONDITIONS IDÉALES (plat 0% & temp optimale) ?", value=True)

# -------------------------
# Calculs : Base & Forcé
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
                apply_elev=use_elev_coeff,
                apply_temp=use_temp_coeff,
                apply_fatigue=fatigue_active,
                objective_time_hms=None,
                k_up=k_up, k_down=k_down,
                k_temp_hot=k_temp_hot, k_temp_cold=k_temp_cold, opt_temp=opt_temp,
                fatigue_rate=fatigue_rate
            )
            st.session_state["res_base"] = res_base
            st.success(f"Base calculée — distance GPX détectée: {res_base['distance_gpx_km']:.3f} km")
        except Exception as e:
            st.error(f"Erreur lors du calcul base : {e}")

# Forcing area
st.markdown("---")
st.markdown("**Forcer distance et/ou temps objectif (produit un tableau 'FORCÉ' distinct)**")
colf1, colf2 = st.columns(2)
with colf1:
    force_distance_checkbox = st.checkbox("Forcer la distance pour la prédiction finale ?", value=False)
    if "dist_forced" not in st.session_state:
        st.session_state["dist_forced"] = 5.17
    distance_forced_km = st.number_input(
        "Distance forcée (km)",
        value=st.session_state["dist_forced"],
        format="%.2f",
        key="dist_forced"
    ) if force_distance_checkbox else None

with colf2:
    force_time_checkbox = st.checkbox("Forcer un temps objectif ?", value=False)
    if "time_forced" not in st.session_state:
        st.session_state["time_forced"] = "0:18:30"
    time_forced_hms = st.text_input(
        "Temps objectif (h:mm:ss)",
        value=st.session_state["time_forced"],
        key="time_forced"
    ) if force_time_checkbox else None

if st.button("📊 Calculer prédiction finale (FORCÉ si activé)"):
    if not gpx_file or points is None:
        st.error("Importe un fichier GPX d'abord.")
    else:
        dist_target = distance_forced_km if force_distance_checkbox and distance_forced_km else None
        try:
            res_forced = run_prediction_df(
                distance_cible_km=dist_target,
                refs_input=refs_raw,
                points=points,
                date_course_local=date_course,
                heure_course_local=heure_course,
                ideal_refs=ideal_refs,
                apply_elev=use_elev_coeff,
                apply_temp=use_temp_coeff,
                apply_fatigue=fatigue_active,
                objective_time_hms=time_forced_hms if force_time_checkbox else None,
                k_up=k_up, k_down=k_down,
                k_temp_hot=k_temp_hot, k_temp_cold=k_temp_cold, opt_temp=opt_temp,
                fatigue_rate=fatigue_rate
            )
            st.session_state["res_forced"] = res_forced
            st.success(f"Prédiction forcée calculée — cible: {distance_forced_km if distance_forced_km else 'GPX'} km")
        except Exception as e:
            st.error(f"Erreur lors du calcul forcé : {e}")

# display side-by-side
if "res_base" in st.session_state or "res_forced" in st.session_state:
    base = st.session_state.get("res_base", None)
    forced = st.session_state.get("res_forced", None)

    left, right = st.columns(2)
    with left:
        st.subheader("📈 Base (d'après références)")
        if base:
            avg_pace_base = base["total_seconds"] / max(base["distance_gpx_km"], 1e-6)
            st.write(f"Distance GPX détectée: {base['distance_gpx_km']:.3f} km")
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

# -------------------------
# CARTE & PROFIL (GPX)
# -------------------------
if gpx_file and points:
    try:
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

        total_m, cumdists, method_used, debug = None, None, None, None
        # reuse compute: quick compute to get cumulative distances
        total_m = 0.0
        cumdists = [0.0]
        for i in range(1, len(points)):
            d = (SimplePoint(points[i-1].latitude, points[i-1].longitude, getattr(points[i-1], "elevation", 0))
                 .distance_3d(SimplePoint(points[i].latitude, points[i].longitude, getattr(points[i], "elevation", 0))))
            total_m += d
            cumdists.append(total_m)
        x_km = np.array(cumdists) / 1000.0
        y_elev = np.array([p.elevation or 0 for p in points])

        plt.plot(x_km, y_elev, lw=2)
        plt.xlabel("Distance (km)")
        plt.ylabel("Altitude (m)")
        plt.title("Profil d'altitude du parcours")
        plt.grid(alpha=0.3)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Impossible d'afficher la carte/profil : {e}")
