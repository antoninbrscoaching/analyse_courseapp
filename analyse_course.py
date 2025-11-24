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

# ------------------------------------------------------
# ⚙️ CONFIGURATION
# ------------------------------------------------------
st.set_page_config(page_title="Prédiction course route", layout="wide")
st.title("🏃‍♂️ Analyse & Prédiction de course (GPX + FIT + Météo + Fatigue linéaire)")

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

def parse_gpx_points(file):
    gpx = gpxpy.parse(file)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                points.append(p)
    return gpx, points

def gpx_to_df(points):
    return pd.DataFrame([{"lat": p.latitude, "lon": p.longitude, "elev": p.elevation or 0} for p in points])

def parse_fit(file):
    try:
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
        dup = np.sum(np.diff(df["elev"]).clip(min=0))
        ddn = -np.sum(np.diff(df["elev"]).clip(max=0))
        return dict(distance=round(df["dist"].max()), D_up=round(dup), D_down=round(ddn))
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_weather(api_key, lat, lon):
    if not api_key:
        return None
    url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return {"data": data.get("hourly", []), "tz_offset": data.get("timezone_offset", 0)}
    except Exception:
        return None
    return None

def find_weather_entry(weather, target_dt):
    if not weather:
        return None
    entries = weather["data"]
    if not entries:
        return None
    best = min(entries, key=lambda x: abs(datetime.fromtimestamp(x["dt"]) - target_dt))
    temp = best.get("temp") or 20
    return {"temp": temp}

# ------------------------------------------------------
# 🧠 MODÈLE LOG-LOG
# ------------------------------------------------------
def fit_loglog_model(refs, k_up=1.0, k_down=1.0):
    xs, ys = [], []
    for r in refs:
        d = r["distance"]
        t_raw = hms_to_seconds(r["temps"])
        dup = r.get("D_up", 0)
        ddn = r.get("D_down", 0)
        elev_factor = (k_up ** dup) * (k_down ** ddn)
        t_eq = t_raw / elev_factor if elev_factor > 0 else t_raw
        if d > 0 and t_eq > 0:
            xs.append(math.log(d))
            ys.append(math.log(t_eq))
    if len(xs) < 2:
        raise ValueError("Il faut deux références minimum.")
    sum_x, sum_y = sum(xs), sum(ys)
    sum_xx = sum(x*x for x in xs)
    sum_xy = sum(x*y for x, y in zip(xs, ys))
    denom = len(xs) * sum_xx - sum_x**2
    if denom == 0:
        raise ValueError("Distances identiques.")
    K = (len(xs) * sum_xy - sum_x * sum_y) / denom
    a = math.exp((sum_y - K * sum_x) / len(xs))
    return a, K

def predict_time_flat(distance_m, a, K):
    return a * (distance_m ** K)

def apply_elevation_gradient_route(base_time_s, D_up_m, D_down_m, segment_length_m=1000, k_up=1.001, k_down=0.999):
    if segment_length_m <= 0:
        return base_time_s
    g_up = D_up_m / segment_length_m
    g_down = D_down_m / segment_length_m
    factor_up = k_up ** (g_up * segment_length_m)
    factor_down = k_down ** (g_down * segment_length_m)
    return base_time_s * factor_up * factor_down

def override_with_objective(distance_obj_m, time_obj_hms, K):
    t_obj = hms_to_seconds(time_obj_hms)
    return t_obj / (distance_obj_m ** K)

# ------------------------------------------------------
# 🗺️ 1. CHARGEMENT GPX
# ------------------------------------------------------
st.header("1️⃣ Parcours GPX")
gpx_file = st.file_uploader("📂 Importer un fichier GPX", type=["gpx"])

# ------------------------------------------------------
# 🧮 2. COURSES DE RÉFÉRENCE
# ------------------------------------------------------
st.header("2️⃣ Courses de référence (manuel ou FIT)")
if "n_refs" not in st.session_state:
    st.session_state.n_refs = 3

cols = st.columns([1, 1])
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
        use_fit = st.checkbox(f"FIT ?", key=f"use_fit_{i}")
    with c2:
        dist = st.number_input(f"Dist {i} (m)", value=5000*i)
    with c3:
        temps = st.text_input(f"Temps {i} (h:mm:ss)", value="0:40:00")
    with c4:
        dup = st.number_input(f"D+ {i}", value=0)
    with c5:
        ddn = st.number_input(f"D- {i}", value=0)
    with c6:
        file_fit = st.file_uploader(f"FIT {i}", type=["fit"], key=f"fit_{i}") if use_fit else None
        if file_fit:
            data_fit = parse_fit(file_fit)
            if data_fit:
                dist, dup, ddn = data_fit["distance"], data_fit["D_up"], data_fit["D_down"]
                st.info(f"✔ FIT détecté : {dist}m | D+{dup} | D-{ddn}")
    refs.append(dict(distance=dist, temps=temps, D_up=dup, D_down=ddn))

# ------------------------------------------------------
# ⚙️ 3. PARAMÈTRES
# ------------------------------------------------------
st.header("3️⃣ Paramètres modèle")
c1, c2 = st.columns(2)
with c1:
    use_elev_coeff = st.checkbox("Activer coefficients montée/descente ?", value=True)
    if use_elev_coeff:
        k_up = st.number_input("Coefficient montée (k_up)", value=1.04)
        k_down = st.number_input("Coefficient descente (k_down)", value=0.996)
    else:
        k_up = 1.0
        k_down = 1.0

with c2:
    k_temp_sup = st.number_input("k_temp_sup (>20°C)", value=1.002)
    k_temp_inf = st.number_input("k_temp_inf (<20°C)", value=0.998)

col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", value=48.8566)
    lon = st.number_input("Longitude", value=2.3522)
    API_KEY = st.text_input("Clé API OpenWeather", type="password")
with col2:
    date_course = st.date_input("Date", value=date.today())
    heure_course = st.time_input("Départ", value=time(9,0))

meteo_data = fetch_weather(API_KEY, lat, lon) if API_KEY else None

# ------------------------------------------------------
# 💤 3bis. FATIGUE
# ------------------------------------------------------
st.header("3️⃣ bis. Fatigue linéaire")
fatigue_active = st.checkbox("Activer fatigue ?")
fatigue_rate = 0
if fatigue_active:
    fatigue_rate = st.slider("Régression finale (%)", 0.0, 30.0, 5.0, 0.5)

# ------------------------------------------------------
# 🧠 4. ANALYSE
# ------------------------------------------------------
st.header("4️⃣ Analyse et prédiction route")
st.caption("1️⃣ Calcul sur la distance GPX avec facteurs montée/descente\n2️⃣ Puis éventuellement distance/temps forcés")

if "first_run_done" not in st.session_state:
    st.session_state.first_run_done = False
if "distance_gpx_km" not in st.session_state:
    st.session_state.distance_gpx_km = None

# ------------------------------------------------------
# Fonction principale run_prediction
# ------------------------------------------------------
def run_prediction(distance_cible_km, objectif_temps_forced=None, show_map=False):
    if not gpx_file:
        st.error("⚠️ Importer un fichier GPX d’abord.")
        return
    gpx, points = parse_gpx_points(gpx_file)
    df_points = gpx_to_df(points)
    if df_points.empty:
        st.error("Fichier GPX invalide.")
        return

    dists = [0]
    total = 0
    for i in range(1, len(points)):
        total += points[i].distance_3d(points[i - 1])
        dists.append(total)
    distance_gpx_km = total / 1000
    st.session_state.distance_gpx_km = distance_gpx_km

    facteur_dist = distance_cible_km / distance_gpx_km if distance_gpx_km > 0 else 1.0
    total_corr = total * facteur_dist
    dists_corr = [d * facteur_dist for d in dists]

    try:
        a, K = fit_loglog_model(refs, k_up=k_up, k_down=k_down)
    except ValueError as e:
        st.error(f"Problème lors de l’ajustement du modèle log-log : {e}")
        return
    st.info(f"📐 Exposant log-log estimé : {K:.4f}")

    if objectif_temps_forced:
        a = override_with_objective(int(distance_cible_km * 1000), objectif_temps_forced, K)
        st.success(f"🎯 Modèle recalé pour {distance_cible_km:.2f} km en {objectif_temps_forced} (sur plat)")

    distance_cible_m = int(distance_cible_km * 1000)
    base_flat_total = predict_time_flat(distance_cible_m, a, K)
    base_s_per_km_flat = base_flat_total / distance_cible_km

    km_marks = [i * 1000 for i in range(1, int(total_corr // 1000) + 1)]
    if total_corr % 1000 != 0:
        km_marks.append(total_corr)

    results = []
    cum_time = 0
    elev_list = [p.elevation or 0 for p in points]
    dt_depart = datetime.combine(date_course, heure_course)

    for i, d in enumerate(km_marks):
        e_cur = np.interp(d, dists_corr, elev_list)
        e_prev = np.interp(d - 1000, dists_corr, elev_list) if i > 0 else e_cur
        d_up = max(0, e_cur - e_prev)
        d_down = max(0, e_prev - e_cur)

        t_km = apply_elevation_gradient_route(base_s_per_km_flat, d_up, d_down, segment_length_m=1000, k_up=k_up, k_down=k_down)

        if fatigue_active and fatigue_rate > 0 and total_corr > 0:
            progression = d / total_corr
            t_km *= (1 + (fatigue_rate / 100.0) * progression)

        passage = dt_depart + timedelta(seconds=cum_time + t_km)
        w = find_weather_entry(meteo_data, passage) if meteo_data else None
        temp = w["temp"] if w else 20
        if temp > 20:
            t_km *= (k_temp_sup ** (temp - 20))
        else:
            t_km *= (k_temp_inf ** (20 - temp))

        cum_time += t_km
        results.append({
            "Km": i + 1,
            "D+ (m)": round(d_up, 1),
            "D- (m)": round(d_down, 1),
            "Temp (°C)": round(temp, 1),
            "Temps segment (s)": round(t_km, 1),
            "Allure (min/km)": f"{int(t_km // 60)}:{int(t_km % 60):02d}",
            "Temps cumulé": seconds_to_hms(cum_time),
        })

    total_time_hms = seconds_to_hms(cum_time)
    st.success(f"⏱️ Temps total prévisionnel : {total_time_hms}")

    df_results = pd.DataFrame(results)
    st.subheader("📋 Détails km par km")
    st.dataframe(df_results, use_container_width=True)

    if show_map:
        st.subheader("🗺️ Carte du parcours")
        view = pdk.ViewState(latitude=df_points.lat.mean(), longitude=df_points.lon.mean(), zoom=13, pitch=0)
        path_layer = pdk.Layer("PathLayer", data=[{"path": df_points[["lon","lat"]].values.tolist(), "name":"Parcours"}], get_path="path", get_color=[255,0,0], width_min_pixels=4)
        deck = pdk.Deck(map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json", initial_view_state=view, layers=[path_layer], tooltip={"text":"{name}"})
        st.pydeck_chart(deck, use_container_width=True)

        st.subheader("📊 Profil d’altitude")
        plt.figure(figsize=(10, 4))
        x_km = np.linspace(0, distance_cible_km, len(df_points))
        plt.plot(x_km, df_points["elev"])
        plt.xlabel("Distance (km)")
        plt.ylabel("Altitude (m)")
        plt.title("Profil d’altitude du parcours")
        plt.grid(alpha=0.3)
        st.pyplot(plt)

    return distance_gpx_km

# ------------------------------------------------------
# 4.1 Premier calcul
# ------------------------------------------------------
st.subheader("4️⃣.1 Calcul initial (distance GPX)")
if st.button("🚀 Lancer l’analyse sur la distance GPX"):
    if not gpx_file:
        st.error("⚠️ Importer un fichier GPX d’abord.")
    else:
        distance_gpx_km = run_prediction(distance_cible_km=1.0, objectif_temps_forced=None, show_map=True)
        if distance_gpx_km:
            st.session_state.first_run_done = True
            st.session_state.distance_gpx_km = distance_gpx_km

# ------------------------------------------------------
# 4.2 Ajustement distance / temps
# ------------------------------------------------------
if st.session_state.first_run_done and st.session_state.distance_gpx_km:
    st.subheader("4️⃣.2 Ajuster distance et/ou temps objectif")
    distance_gpx_km = st.session_state.distance_gpx_km

    # Distance forcée
    use_forced_distance = st.checkbox("Forcer la distance ?", value=False)
    distance_cible_km = distance_gpx_km
    if use_forced_distance:
        distance_cible_km = st.number_input("Distance forcée (km)", value=float(round(distance_gpx_km, 2)), min_value=0.5, step=0.1)

    # Temps objectif
    use_forced_time = st.checkbox("Forcer un temps objectif ?", value=False)
    objectif_temps_forced = None
    if use_forced_time:
        objectif_temps_forced = st.text_input("Temps objectif (h:mm:ss)", value="0:17:30")

    if st.button("📊 Calculer prédiction finale"):
        run_prediction(distance_cible_km=distance_cible_km, objectif_temps_forced=objectif_temps_forced, show_map=True)
