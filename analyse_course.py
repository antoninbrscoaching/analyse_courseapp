# analyse_course.py (version corrigée)
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

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Prédiction course route", layout="wide")
st.title("🏃‍♂️ Analyse & Prédiction de course (GPX + FIT + TCX + Météo + Fatigue linéaire) — Route")

# -------------------------
# UTILS
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
# Fonctions manquantes ajoutées
# -------------------------
def temp_multiplier_nonlin(temp, opt_temp=12.0, k_hot=0.002, k_cold=0.002):
    """
    Simple non-linear temperature multiplier:
    - if temp > opt_temp: multiplier = 1 + k_hot * (temp - opt)
    - else: multiplier = 1 + k_cold * (opt - temp)
    Guarantees multiplier >= 0.1 (safety).
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

def compute_total_and_cumdist(points):
    """
    Retourne (total_m, cumdists_list, method_str, debug_dict)
    Méthode : simple somme des distances 3D entre points.
    """
    try:
        if not points or len(points) < 2:
            return 0.0, [0.0], "simple", {"n_pts": len(points)}
        cum = [0.0]
        total = 0.0
        for i in range(1, len(points)):
            d = (SimplePoint(points[i-1].latitude, points[i-1].longitude, getattr(points[i-1], "elevation", 0))
                 .distance_3d(SimplePoint(points[i].latitude, points[i].longitude, getattr(points[i], "elevation", 0))))
            total += d
            cum.append(total)
        return total, cum, "3d_haversine", {"n_pts": len(points)}
    except Exception as e:
        return 0.0, [0.0], "error", {"error": str(e)}

def apply_elevation_gradient_route(t_flat, d_up, d_down, segment_length_m=1000.0, k_up=1.04, k_down=0.996):
    """
    Applique un facteur de montée/descente sur le temps plat du segment.
    Approche simple : factor = 1 + (k_up-1)*(d_up/seg_len) + (1-k_down)*(d_down/seg_len)
    """
    try:
        seg_len = float(segment_length_m) if segment_length_m and segment_length_m > 0 else 1000.0
        up_factor = (float(k_up) - 1.0) * (float(d_up) / seg_len)
        down_factor = (1.0 - float(k_down)) * (float(d_down) / seg_len)
        factor = 1.0 + up_factor + down_factor
        return float(t_flat) * max(0.01, factor)
    except Exception:
        return float(t_flat)

def fit_loglog_model(refs, k_up=1.04, k_down=0.996):
    """
    Ajuste simple log-log : total_seconds = a * (distance_km ** K)
    On calcule a et K par régression linéaire sur log.
    Si pas assez de refs, valeurs par défaut raisonnables.
    Retourne (a, K) où a = seconds pour 1 km.
    """
    try:
        X = []
        Y = []
        for r in refs:
            d_m = r.get("distance", None)
            t_raw = r.get("temps")  # peut être str "h:mm:ss" ou valeur déjà en secondes
            if d_m is None or d_m <= 0:
                continue
            # obtenir seconds
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
            # ensure sane values
            if a <= 0 or math.isnan(a) or math.isinf(a):
                a = 240.0  # 4min / km as fallback
            if K <= -5 or K >= 5 or math.isnan(K) or math.isinf(K):
                K = 1.0
            return a, K
        else:
            # fallback: assume near-linear scaling (K = 1), a = mean secs for 1km from refs if any
            if len(X) == 1:
                d_km = math.exp(X[0])
                secs = math.exp(Y[0])
                # a such that secs = a * d_km^K => K=1 => a = secs / d_km
                a = secs / max(1e-6, d_km)
                return a, 1.0
            # no refs: default a=240s (4:00/km), K=1
            return 240.0, 1.0
    except Exception:
        return 240.0, 1.0

def predict_time_flat(distance_m, a, K):
    try:
        d_km = float(distance_m) / 1000.0
        return float(a) * (d_km ** float(K))
    except Exception:
        return float(a) * max(0.0, (float(distance_m) / 1000.0))

def override_with_objective(distance_m, objective_time_hms, K):
    """
    Calcule a tel que a * (distance_km ** K) = objective_seconds.
    """
    try:
        objective_seconds = hms_to_seconds(objective_time_hms)
        d_km = float(distance_m) / 1000.0
        if d_km <= 0:
            return None
        a = float(objective_seconds) / (d_km ** float(K))
        return a
    except Exception:
        return None

def get_temp_for_datetime(hourly_cache, dt):
    """
    hourly_cache placeholder: structure non imposée ici.
    On renvoie None (pas de donnée) pour garder le code fonctionnel hors-ligne.
    """
    try:
        return None
    except Exception:
        return None

@st.cache_data(ttl=60)
def fetch_open_meteo_hourly(lat, lon, start_date, end_date):
    """
    Placeholder qui renvoie {} si on ne veut pas appeler une API.
    Si tu veux la météo réelle, on peut implémenter l'appel HTTP vers Open-Meteo ici.
    """
    try:
        # Par défaut on ne fait pas de requête réseau (sécurité / offline)
        return {}
    except Exception:
        return {}

# -------------------------
# PARSERS GPX / FIT / TCX
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

# -------------------------
# SAFE / SESSION HELPERS
# -------------------------
def safe_float(val):
    try:
        if val is None:
            return 0.0
        if isinstance(val, str):
            s = val.strip()
            if s == "" or s.lower() in ("nan", "none"):
                return 0.0
            return float(s.replace(",", "."))
        if isinstance(val, (float, int, np.number)):
            if np.isnan(val) or np.isinf(val):
                return 0.0
            return float(val)
        return float(val)
    except Exception:
        return 0.0

def safe_str(val):
    try:
        if val is None:
            return "00:00:00"
        if isinstance(val, (int, float, np.number)):
            return seconds_to_hms(float(val))
        s = str(val).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return "00:00:00"
        return s
    except Exception:
        return "00:00:00"

def safe_set(key, value):
    try:
        if key is None or str(key).strip() == "":
            st.warning(f"Impossible de définir session_state : clé invalide ({key})")
            return
        if isinstance(value, (float, int, np.number)):
            if np.isnan(value) or np.isinf(value):
                value = 0.0
        if value is None:
            value = 0.0
        if isinstance(value, str) and value.strip() == "":
            value = "00:00:00"
        st.session_state[key] = value
    except Exception as e:
        st.warning(f"Impossible de définir session_state[{key}] : {e}")

def update_ref_session_safe(i, dist=None, temps=None, dup=None, ddn=None):
    if dist is not None:
        safe_set(f"dist_{i}", safe_float(dist))
    if temps is not None:
        safe_set(f"temps_{i}", safe_str(temps))
    if dup is not None:
        safe_set(f"dup_{i}", safe_float(dup))
    if ddn is not None:
        safe_set(f"ddn_{i}", safe_float(ddn))

def clean_value(v):
    try:
        if v is None:
            return None
        if isinstance(v, (int, float, np.number)):
            if np.isnan(v) or np.isinf(v):
                return 0.0
            return v
        if isinstance(v, str):
            s = v.strip()
            if s == "" or s.lower() in ("none", "nan"):
                return None
            if ":" in s:
                return s
            try:
                return float(s.replace(",", "."))
            except:
                return s
        try:
            return float(v)
        except:
            return str(v)
    except Exception:
        return None

# -------------------------
# UI & INPUTS
# -------------------------
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

    # parse uploaded file (if any) and update local vars
    dist_f = dup_f = ddn_f = None
    temps_f = None
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

    # Update session_state safely if file provided
    if any(v is not None for v in [dist_f, temps_f, dup_f, ddn_f]):
        update_ref_session_safe(i, dist_f, temps_f, dup_f, ddn_f)
        # read back to local variables (ensure consistency)
        dist = float(st.session_state.get(f"dist_{i}", dist))
        dup = float(st.session_state.get(f"dup_{i}", dup))
        ddn = float(st.session_state.get(f"ddn_{i}", ddn))
        temps = str(st.session_state.get(f"temps_{i}", temps))

    # compute recalibrated time for this reference: to 0% and 12°C baseline
    def recalibrate_time(temps_hms, D_up, D_down, distance_m, temp_ideal=12.0, k_up=1.04, k_down=0.996, k_temp_hot=0.002, k_temp_cold=0.002):
        secs = hms_to_seconds(temps_hms)
        seg_len = distance_m if distance_m and distance_m > 0 else 1000.0
        up_factor = (k_up - 1.0) * (D_up / max(seg_len, 1.0))
        down_factor = (1.0 - k_down) * (D_down / max(seg_len, 1.0))
        factor_elev = 1.0 + up_factor + down_factor
        try:
            secs_no_elev = secs / factor_elev if factor_elev != 0 else secs
        except Exception:
            secs_no_elev = secs
        # neutralize temperature by dividing by multiplier at ideal (if !=1)
        mult_temp_opt = temp_multiplier_nonlin(temp_ideal, opt_temp=temp_ideal, k_hot=k_temp_hot, k_cold=k_temp_cold)
        try:
            secs_flat_temp = secs_no_elev / mult_temp_opt if mult_temp_opt != 0 else secs_no_elev
        except Exception:
            secs_flat_temp = secs_no_elev
        return max(secs_flat_temp, 0.0)

    recal_secs = recalibrate_time(temps, dup, ddn, dist, temp_ideal=12.0, k_up=1.04, k_down=0.996, k_temp_hot=0.002, k_temp_cold=0.002)
    recal_hms = seconds_to_hms(recal_secs)

    # store in refs list (will be used by the model)
    refs.append({
        "distance": float(dist),
        "temps": str(temps),
        "D_up": float(dup),
        "D_down": float(ddn),
        "temps_recal": float(recal_secs),
        "temps_recal_hms": recal_hms
    })

    st.markdown(f"Temps brut : `{temps}`  →  Temps recalibré (0% & 12°C) : `{recal_hms}`")

# -------------------------
# Affichage des temps recalculés global (séparé)
# -------------------------
st.subheader("⏱️ Récap — Références recalibrées (0% & 12°C)")
for idx, r in enumerate(refs, start=1):
    st.write(f"Réf {idx} — Dist: {r['distance']:.0f} m | Brut: {r['temps']} | Recalibré: {r['temps_recal_hms']} | D+ {r['D_up']:.0f} m / D- {r['D_down']:.0f} m")

# -------------------------
# PARAMÈTRES MODÈLE
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
    st.info("La récupération historique n'est pas implémentée dans ce squelette par défaut (placeholder).")

# -------------------------
# FATIGUE
# -------------------------
st.header("3️⃣ bis. Fatigue linéaire")
fatigue_active = st.checkbox("Activer fatigue ?", value=False)
fatigue_rate = 0.0
if fatigue_active:
    fatigue_rate = st.slider("Régression finale (%)", 0.0, 30.0, 5.0, 0.5)

# -------------------------
# fetch_historical_range wrapper
# -------------------------
@st.cache_data(ttl=60)
def fetch_historical_range(lat, lon, start_date, end_date):
    # wrapper using placeholder fetch_open_meteo_hourly
    return fetch_open_meteo_hourly(lat, lon, start_date.isoformat() if isinstance(start_date, date) else str(start_date),
                                   end_date.isoformat() if isinstance(end_date, date) else str(end_date))

# -------------------------
# run_prediction_df: utilise temps_recal si présent
# -------------------------
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

    # hourly temps (placeholder)
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

    # Build refs_for_fit using temps_recal if present:
    refs_for_fit = []
    for r in refs_input:
        rr = r.copy()
        # prefer temps_recal (already 0% & 12C normalized)
        if r.get("temps_recal") and r.get("temps_recal") > 0:
            rr["temps"] = seconds_to_hms(r["temps_recal"])
        else:
            # fallback to raw temps
            rr["temps"] = r.get("temps", "0:00:00")
        refs_for_fit.append(rr)

    # Fit model (placeholder): returns baseline seconds per km 'a'
    a, K = fit_loglog_model(refs_for_fit, k_up=(local_k_up if apply_elev else 1.0), k_down=(local_k_down if apply_elev else 1.0))

    # objective override
    a_override = None
    if objective_time_hms:
        a_override = override_with_objective(int(distance_cible_km * 1000), objective_time_hms, K)

    distance_cible_m = int(distance_cible_km * 1000)
    baseline_seconds_per_km = (a_override if a_override is not None else a)
    base_flat_total = predict_time_flat(distance_cible_m, baseline_seconds_per_km, K)
    base_s_per_km_flat = base_flat_total / distance_cible_km if distance_cible_km > 0 else base_flat_total

    # segment markers (per km)
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

    # if objective override: compute scale to meet objective
    if objective_time_hms:
        objective_seconds = hms_to_seconds(objective_time_hms)
        sum_raw = sum(s["t_raw"] for s in segment_infos)
        scale = (objective_seconds / sum_raw) if (sum_raw > 0) else 1.0
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
        "a": baseline_seconds_per_km, "K": K
    }

# -------------------------
# INTERACTIONS : Calculs de prédiction
# -------------------------
st.subheader("4️⃣ Calcul & Comparaison")

if st.button("▶️ Calculer prédiction (BASE, d'après références)"):
    if not gpx_file:
        st.error("Importe un fichier GPX d'abord.")
    else:
        gpx, points = parse_gpx_points(gpx_file)
        try:
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
        except Exception as e:
            st.error(f"Erreur lors du calcul base : {e}")

# Forcing area
st.markdown("---")
st.markdown("**Forcer distance et/ou temps objectif (produit un tableau 'FORCÉ' distinct)**")
colf1, colf2 = st.columns(2)

# --- DISTANCE FORCÉE ---
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

# --- TEMPS OBJECTIF ---
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
    if not gpx_file:
        st.error("Importe un fichier GPX d'abord.")
    else:
        gpx, points = parse_gpx_points(gpx_file)
        dist_target = distance_forced_km if force_distance_checkbox and distance_forced_km else None
        try:
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

# -------------------------
# CARTE & PROFIL (GPX)
# -------------------------
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

        total_m, cumdists, method_used, debug = compute_total_and_cumdist(points)
        x_km = np.array(cumdists) / 1000.0
        y_elev = np.array([p.elevation or 0 for p in points])

        plt.plot(x_km, y_elev, lw=2)
        plt.xlabel("Distance (km)")
        plt.ylabel("Altitude (m)")
        plt.title(f"Profil d'altitude du parcours (méthode {method_used})")
        plt.grid(alpha=0.3)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Impossible d'afficher la carte/profil : {e}")
