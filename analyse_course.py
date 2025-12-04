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

# ---------------- Session helpers sécurisés ----------------
def safe_float(val):
    """Convertit en float, retourne 0.0 si invalide."""
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return 0.0
        return float(val)
    except Exception:
        return 0.0

def safe_str(val):
    """Convertit en string, retourne '00:00:00' si invalide."""
    try:
        if val is None or val == "":
            return "00:00:00"
        return str(val)
    except Exception:
        return "00:00:00"

def update_ref_session(i, dist=None, temps=None, dup=None, ddn=None):
    """Met à jour st.session_state de manière sécurisée pour éviter StreamlitAPIException"""
    if st.session_state is not None:
        try:
            st.session_state[f"dist_{i}"] = safe_float(dist)
        except Exception:
            st.session_state[f"dist_{i}"] = 0.0
        try:
            st.session_state[f"temps_{i}"] = safe_str(temps)
        except Exception:
            st.session_state[f"temps_{i}"] = "00:00:00"
        try:
            st.session_state[f"dup_{i}"] = safe_float(dup)
        except Exception:
            st.session_state[f"dup_{i}"] = 0.0
        try:
            st.session_state[f"ddn_{i}"] = safe_float(ddn)
        except Exception:
            st.session_state[f"ddn_{i}"] = 0.0

# ---------------- UI & Inputs pour références ----------------
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
        dist = st.number_input(f"Dist {i} (m)", value=default_dist, key=f"dist_{i}")
    with c3:
        temps = st.text_input(f"Temps {i} (h:mm:ss)", value=default_temps, key=f"temps_{i}")
    with c4:
        dup = st.number_input(f"D+ {i}", value=default_dup, key=f"dup_{i}")
    with c5:
        ddn = st.number_input(f"D- {i}", value=default_ddn, key=f"ddn_{i}")
    with c6:
        file_in = st.file_uploader(f"FIT/TCX {i}", type=["fit","tcx"], key=f"fileref_{i}") if use_file else None
        dist_f = temps_f = dup_f = ddn_f = None
        if file_in:
            name = getattr(file_in, "name", "") or ""
            try:
                if name.lower().endswith(".fit"):
                    data_fit = parse_fit(file_in)
                    if data_fit:
                        dist_f = data_fit.get("distance", 0)
                        dup_f = data_fit.get("D_up", 0)
                        ddn_f = data_fit.get("D_down", 0)
                        temps_f = data_fit.get("duration_hms") or "00:00:00"
                        st.info(f"✔ FIT détecté : {dist_f}m | D+ {dup_f} | D- {ddn_f} | dur: {temps_f}")
                    else:
                        st.warning(f"Fichier FIT Réf {i} non exploitable.")
                elif name.lower().endswith(".tcx"):
                    tcx_res = parse_tcx(file_in)
                    if tcx_res:
                        dist_f = int(round(tcx_res.get("distance",0)))
                        dup_f = int(round(tcx_res.get("D_up",0)))
                        ddn_f = int(round(tcx_res.get("D_down",0)))
                        temps_f = tcx_res.get("duration_hms") or "00:00:00"
                        st.info(f"✔ TCX détecté : {dist_f}m | D+ {dup_f} | D- {ddn_f} | dur: {temps_f}")
                    else:
                        st.warning(f"Fichier TCX Réf {i} non exploitable.")
            except Exception as e:
                st.error(f"Erreur parsing fichier Réf {i} : {e}")

            # ⚡ Mise à jour sécurisée session_state
            update_ref_session(i, dist_f, temps_f, dup_f, ddn_f)

    # Construction des refs pour calculs suivants
    refs.append({
        "distance": float(st.session_state.get(f"dist_{i}", dist or 0.0)),
        "temps": str(st.session_state.get(f"temps_{i}", temps or "00:00:00")),
        "D_up": float(st.session_state.get(f"dup_{i}", dup or 0.0)),
        "D_down": float(st.session_state.get(f"ddn_{i}", ddn or 0.0))
    })

# ==============================================================
# 3️⃣ Paramètres modèle & entrée utilisateur
# ==============================================================

st.header("3️⃣ Paramètres modèle")
c1, c2 = st.columns(2)

with c1:
    use_elev_coeff = st.checkbox("Activer coefficients montée/descente 🎢", value=True)
    if use_elev_coeff:
        k_up = st.number_input("Coefficient montée (k_up)", value=1.040, format="%.3f", step=0.001)
        k_down = st.number_input("Coefficient descente (k_down)", value=0.996, format="%.3f", step=0.001)
    else:
        k_up = 1.0
        k_down = 1.0

with c2:
    use_temp_coeff = st.checkbox("Activer coefficients température 🌡️", value=True)
    if use_temp_coeff:
        k_temp_hot = st.number_input("Sensibilité chaude (k_temp_hot)", value=0.002, format="%.3f", step=0.001)
        k_temp_cold = st.number_input("Sensibilité froide (k_temp_cold)", value=0.002, format="%.3f", step=0.001)
        opt_temp = st.number_input("Température optimale (°C)", value=12.0, format="%.1f", step=0.5)
    else:
        k_temp_hot = 0.0
        k_temp_cold = 0.0
        opt_temp = 12.0

# Coordonnées pour météo
col1, col2 = st.columns(2)
with col1:
    lat_input = st.number_input("Latitude (pour météo)", value=48.8566, format="%.6f")
    lon_input = st.number_input("Longitude (pour météo)", value=2.3522, format="%.6f")
    use_hist_refs = st.checkbox("Recalibrer les références avec météo historique ?", value=False)
with col2:
    date_course = st.date_input("Date de la course (Jour J)", value=date.today())
    heure_course = st.time_input("Heure de départ (Jour J)", value=time(9, 0))

# Fatigue linéaire
st.header("3️⃣ bis. Fatigue linéaire")
fatigue_active = st.checkbox("Activer fatigue ?", value=False)
fatigue_rate = st.slider("Régression finale (%)", 0.0, 30.0, 5.0, 0.5) if fatigue_active else 0.0

# ==============================================================
# 4️⃣ Calcul & Comparaison (BASE / FORCÉ)
# ==============================================================

st.subheader("4️⃣ Calcul & Comparaison")

# --- Calcul BASE (références) ---
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

# --- Forcer distance / temps objectif ---
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

# --- Affichage côte-à-côte BASE / FORCÉ ---
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

# ==============================================================
# 5️⃣ Carte & Profil d'altitude (GPX)
# ==============================================================

if gpx_file:
    try:
        gpx, points = parse_gpx_points(gpx_file)
        df_points = gpx_to_df(points)

        st.subheader("🗺️ Carte & Profil (GPX importé)")

        # --- Carte ---
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

        # --- Profil d'altitude ---
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

# ==============================================================
# 6️⃣ Fonction update_ref_session (mise à jour sécurisée session_state)
# ==============================================================

def safe_float(val, fallback=0.0):
    """Convertit en float de manière sécurisée."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return fallback

def update_ref_session(i, dist, temps, dup, ddn):
    """
    Met à jour les variables de session pour la référence i.
    Cette version utilise setdefault pour éviter les erreurs Streamlit.
    """
    # Initialisation sécurisée si clé absente
    if f"dist_{i}" not in st.session_state:
        st.session_state[f"dist_{i}"] = 0.0
    if f"temps_{i}" not in st.session_state:
        st.session_state[f"temps_{i}"] = "00:00:00"
    if f"dup_{i}" not in st.session_state:
        st.session_state[f"dup_{i}"] = 0.0
    if f"ddn_{i}" not in st.session_state:
        st.session_state[f"ddn_{i}"] = 0.0

    # Mise à jour sécurisée
    st.session_state[f"dist_{i}"] = safe_float(dist)
    st.session_state[f"temps_{i}"] = str(temps) if temps is not None else "00:00:00"
    st.session_state[f"dup_{i}"] = safe_float(dup)
    st.session_state[f"ddn_{i}"] = safe_float(ddn)
