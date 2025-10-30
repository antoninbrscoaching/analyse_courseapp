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
st.set_page_config(page_title="Analyse course complète", layout="wide")
st.title("🏃‍♂️ Analyse & Prédiction de course (GPX + FIT + Météo + Fatigue linéaire)")

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
# 💤 3️⃣ bis. FATIGUE LINÉAIRE
# ------------------------------------------------------
st.header("3️⃣ bis. Fatigue linéaire (optionnelle)")
st.caption("L’allure régresse de manière linéaire sur toute la durée de la course (indépendamment du D+ ou de la météo).")

fatigue_active = st.checkbox("Activer la fatigue linéaire", value=False)
fatigue_rate = 0.0
if fatigue_active:
    fatigue_rate = st.slider(
        "Pourcentage de régression à la fin de la course (%)",
        min_value=0.0, max_value=30.0, step=0.5, value=5.0,
        help="Ex: 5% signifie que ton allure est 5% plus lente à la fin qu’au début, de façon linéaire."
    )

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

    # Distance cumulée
    dists = [0]
    total = 0
    for i in range(1, len(points)):
        total += points[i].distance_3d(points[i-1])
        dists.append(total)

    st.success(f"📏 Distance totale : {total/1000:.2f} km")

    # Régression log-log
    temps_sec, dists_ref = [], []
    for r in refs:
        t = hms_to_seconds(r["temps"])
        t_adj = t * (k_up ** r["D_up"]) * (k_down ** r["D_down"])
        temps_sec.append(t_adj)
        dists_ref.append(r["distance"])
    K = sum(
        math.log(temps_sec[j]/temps_sec[i]) / math.log(dists_ref[j]/dists_ref[i])
        for i in range(len(refs)) for j in range(i+1,len(refs))
    ) / max(1, len(refs)-1)
    st.info(f"Exposant log-log estimé : {K:.4f}")

    # Météo
    meteo = fetch_weather(API_KEY, lat, lon)
    dt_depart = datetime.combine(date_course, heure_course)

    # Prévision km par km
    km_marks = [i*1000 for i in range(1, int(total//1000)+1)]
    if total % 1000 != 0:
        km_marks.append(total)
    base_total = hms_to_seconds(objectif_temps) if objectif_temps else temps_sec[-1]*(total/dists_ref[-1])**K
    base_s_per_km = base_total / (total/1000)

    results = []
    cum_time = 0
    for i, d in enumerate(km_marks):
        e_cur = np.interp(d, dists, [p.elevation or 0 for p in points])
        e_prev = np.interp(d-1000, dists, [p.elevation or 0 for p in points]) if i > 0 else e_cur
        d_up = max(0, e_cur - e_prev)
        d_down = max(0, e_prev - e_cur)

        t_km = base_s_per_km * (k_up**d_up) * (k_down**d_down)

        # 👉 Appliquer une régression linéaire de fatigue
        if fatigue_active and fatigue_rate > 0:
            progression = d / total
            fatigue_mult = 1.0 + (fatigue_rate / 100.0) * progression
            t_km *= fatigue_mult
        else:
            fatigue_mult = 1.0

        # Météo (ajustement indépendant)
        passage = dt_depart + timedelta(seconds=cum_time + t_km)
        w = find_weather_entry(meteo, passage)
        temp = w["temp"] if w else 20
        if temp > 20:
            t_km *= (k_temp_sup ** (temp - 20))
        else:
            t_km *= (k_temp_inf ** (20 - temp))

        cum_time += t_km
        results.append({
            "Km": i+1,
            "D+ (m)": round(d_up,1),
            "D- (m)": round(d_down,1),
            "Temp (°C)": round(temp,1),
            "Fatigue (%)": f"{(fatigue_mult-1)*100:.2f}%",
            "Temps segment (s)": round(t_km,1),
            "Allure (min/km)": f"{int((t_km//60))}:{int(t_km%60):02d}"
        })

    total_time = seconds_to_hms(sum(r["Temps segment (s)"] for r in results))
    st.success(f"⏱️ Temps total prévisionnel : {total_time}")

    st.subheader("📋 Détails km par km")
    st.dataframe(results, use_container_width=True)

    # ------------------------------------------------------
    # 🗺️ CARTE SIMPLE 2D
    # ------------------------------------------------------
    st.subheader("🗺️ Carte du parcours")
    view = pdk.ViewState(
        latitude=df_points.lat.mean(),
        longitude=df_points.lon.mean(),
        zoom=13,
        pitch=0,
    )
    path_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": df_points[["lon","lat"]].values.tolist(), "name": "Parcours"}],
        get_path="path",
        get_color=[255, 0, 0],
        width_min_pixels=4,
    )
    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=view,
        layers=[path_layer],
        tooltip={"text": "{name}"},
    )
    st.pydeck_chart(deck, use_container_width=True)

    # ------------------------------------------------------
    # 📈 PROFIL D’ALTITUDE
    # ------------------------------------------------------
    st.subheader("📊 Profil d’altitude")

    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, total/1000, len(df_points)), df_points["elev"], color="tab:red")
    plt.xlabel("Distance (km)")
    plt.ylabel("Altitude (m)")
    plt.title("Profil d’altitude du parcours")
    plt.grid(alpha=0.3)
    st.pyplot(plt)
