import streamlit as st
import math
import gpxpy
from fitparse import FitFile
import requests
from datetime import datetime, timedelta, date, time
import pandas as pd
import numpy as np
import pydeck as pdk

# ------------------------------------------------------
# ⚙️ CONFIGURATION
# ------------------------------------------------------
st.set_page_config(page_title="Course GPX + FIT + Météo + Satellite 3D", layout="wide")
st.title("🏃‍♂️ Analyse & Prédiction de course (GPX + FIT + Météo + Satellite 3D réaliste)")

# ------------------------------------------------------
# 🧩 UTILITAIRES
# ------------------------------------------------------
def hms_to_seconds(hms: str) -> int:
    try:
        h, m, s = map(int, hms.strip().split(":"))
        return h * 3600 + m * 60 + s
    except:
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
        df = pd.DataFrame(records, columns=["lat","lon","elev","dist"])
        if df.empty: return None
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
    except:
        return None
    return None

def find_weather_entry(weather, target_dt):
    if not weather:
        return None
    entries = weather["data"]
    best = min(entries, key=lambda x: abs(datetime.fromtimestamp(x["dt"]) - target_dt))
    temp = best.get("temp") or 20
    wind = best.get("wind_speed") or 0
    return {"temp": temp, "wind": wind}

# ------------------------------------------------------
# 🗺️ 1. CHARGEMENT GPX
# ------------------------------------------------------
st.header("1️⃣ Parcours GPX")
gpx_file = st.file_uploader("📂 Importer un fichier GPX", type=["gpx"])

# ------------------------------------------------------
# 🧮 2. COURSES DE RÉFÉRENCE
# ------------------------------------------------------
st.header("2️⃣ Courses de référence (manuel ou fichiers FIT)")
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
for i in range(1, st.session_state.n_refs+1):
    st.markdown(f"#### Référence {i}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        use_fit = st.checkbox(f"FIT ?", key=f"fit_use_{i}")
    with c2:
        dist = st.number_input(f"Dist {i} (m)", value=5000*i, step=100)
    with c3:
        time_str = st.text_input(f"Temps {i} (h:mm:ss)", value=f"0:{40+i*2}:00")
    with c4:
        dup = st.number_input(f"D+ {i}", value=0)
    with c5:
        ddn = st.number_input(f"D- {i}", value=0)
    with c6:
        fit_file = st.file_uploader(f"FIT {i}", type=["fit"], key=f"fit_{i}") if use_fit else None
        if fit_file:
            data_fit = parse_fit(fit_file)
            if data_fit:
                dist, dup, ddn = data_fit["distance"], data_fit["D_up"], data_fit["D_down"]
                st.info(f"✔ FIT détecté : {dist} m | D+ {dup} | D- {ddn}")
    refs.append(dict(distance=dist, temps=time_str, D_up=dup, D_down=ddn))

# ------------------------------------------------------
# ⚙️ 3. PARAMÈTRES
# ------------------------------------------------------
st.header("3️⃣ Paramètres de modélisation")
c1, c2 = st.columns(2)
with c1:
    k_up = st.number_input("k_montée", value=1.001)
    k_down = st.number_input("k_descente", value=0.999)
with c2:
    k_temp_sup = st.number_input("k_temp_sup (>20°C)", value=1.002)
    k_temp_inf = st.number_input("k_temp_inf (<20°C)", value=0.998)

st.markdown("### Paramètres météo et course")
col1, col2, col3 = st.columns(3)
with col1:
    lat = st.number_input("Latitude", value=48.8566)
    lon = st.number_input("Longitude", value=2.3522)
    API_KEY = st.text_input("Clé API OpenWeather", type="password")
with col2:
    date_course = st.date_input("Date de la course", value=date.today())
    heure_course = st.time_input("Heure départ", value=time(9, 0))
with col3:
    objectif_temps = st.text_input("Objectif (h:mm:ss)", value="")

# ------------------------------------------------------
# 🧠 4. ANALYSE
# ------------------------------------------------------
if st.button("🚀 Lancer l’analyse complète"):
    if not gpx_file:
        st.error("⚠️ Upload d’abord un fichier GPX.")
        st.stop()

    gpx, points = parse_gpx_points(gpx_file)
    df_points = gpx_to_df(points)

    if df_points.empty:
        st.error("Fichier GPX invalide.")
        st.stop()

    # Distance totale GPX
    dists = [0]
    total = 0
    for i in range(1, len(points)):
        total += points[i].distance_3d(points[i-1])
        dists.append(total)
    st.success(f"📏 Distance totale : {total/1000:.2f} km")

    # ------------------------------------------------------
    # 🌍 Carte Satellite + Relief 3D
    # ------------------------------------------------------
    st.subheader("🛰️ Carte Satellite 3D réaliste (relief + photo satellite)")

    view = pdk.ViewState(
        latitude=df_points.lat.mean(),
        longitude=df_points.lon.mean(),
        zoom=13,
        pitch=60,
        bearing=30,
    )

    # Relief (modèle de terrain mondial)
    terrain_layer = pdk.Layer(
        "TerrainLayer",
        data=None,
        elevation_decoder={
            "rScaler": 256,
            "gScaler": 1,
            "bScaler": 1 / 256,
            "offset": -32768,
        },
        texture="https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
        elevation_data="https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
        bounds=[df_points.lon.min()-0.02, df_points.lat.min()-0.02,
                df_points.lon.max()+0.02, df_points.lat.max()+0.02],
    )

    # Tracé GPX
    path_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": df_points[["lon","lat"]].values.tolist(), "name": "Parcours"}],
        get_path="path",
        get_color=[255, 0, 0],
        width_min_pixels=4,
    )

    # Fond satellite ESRI World Imagery
    deck = pdk.Deck(
        map_style="https://basemaps.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        initial_view_state=view,
        layers=[terrain_layer, path_layer],
        tooltip={"text": "{name}"},
    )

    st.pydeck_chart(deck, use_container_width=True)
