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
    return pd.DataFrame(
        [{"lat": p.latitude, "lon": p.longitude, "elev": p.elevation or 0} for p in points]
    )

def parse_fit(file):
    try:
        fit = FitFile(file)
        fit.parse()
        records = []
        for msg in fit.get_messages("record"):
            data = {d.name: d.value for d in msg}
            if data.get("position_lat") and data.get("position_long"):
                lat = data["position_lat"] * (180 / 2 ** 31)
                lon = data["position_long"] * (180 / 2 ** 31)
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
    url = (
        "https://api.openweathermap.org/data/2.5/onecall"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
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
    wind = best.get("wind_speed") or 0
    return {"temp": temp, "wind": wind}

# ------------------------------------------------------
# 🧠 MODÈLE LOG-LOG
# ------------------------------------------------------
def fit_loglog_model(refs, k_up=1.0, k_down=1.0):
    xs, ys = []
    xs, ys = [], []
    for r in refs:
        d = r["distance"]
        t_raw = hms_to_seconds(r["temps"])
        dup, ddn = r.get("D_up", 0), r.get("D_down", 0)
        elev_factor = (k_up ** dup) * (k_down ** ddn)
        t_eq = t_raw / elev_factor
        if d > 0 and t_eq > 0:
            xs.append(math.log(d))
            ys.append(math.log(t_eq))

    n = len(xs)
    if n < 2:
        raise ValueError("Il faut au moins 2 références.")

    sum_x, sum_y = sum(xs), sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x ** 2

    K = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - K * sum_x) / n
    a = math.exp(intercept)
    return a, K

def predict_time_flat(distance_m, a, K):
    return a * (distance_m ** K)

def apply_elevation(time_flat_s, du, dd, k_up=1.0, k_down=1.0):
    return time_flat_s * (k_up ** du) * (k_down ** dd)

def override_with_objective(distance_m, objectif_hms, K):
    t_obj = hms_to_seconds(objectif_hms)
    return t_obj / (distance_m ** K)

# ------------------------------------------------------
# 🗺️ 1. CHARGEMENT GPX
# ------------------------------------------------------
st.header("1️⃣ Parcours GPX")
gpx_file = st.file_uploader("📂 Importer un fichier GPX", type=["gpx"])

# ------------------------------------------------------
# 🧮 2. RÉFÉRENCES
# ------------------------------------------------------
st.header("2️⃣ Références")
if "n_refs" not in st.session_state:
    st.session_state.n_refs = 3

col_add, col_rm = st.columns(2)
with col_add:
    if st.button("➕ Ajouter une référence") and st.session_state.n_refs < 6:
        st.session_state.n_refs += 1
with col_rm:
    if st.button("➖ Retirer une référence") and st.session_state.n_refs > 1:
        st.session_state.n_refs -= 1

refs = []
for i in range(1, st.session_state.n_refs + 1):
    st.subheader(f"Référence {i}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        use_fit = st.checkbox(f"FIT ?", key=f"fit_use_{i}")
    with c2:
        dist = st.number_input(f"Dist {i} (m)", value=5000 * i)
    with c3:
        t = st.text_input(f"Temps {i}", value=f"0:{40+i*2}:00")
    with c4:
        dup = st.number_input(f"D+ {i}", value=0)
    with c5:
        ddn = st.number_input(f"D- {i}", value=0)
    with c6:
        fit_file = st.file_uploader(f"FIT {i}", type=["fit"], key=f"fit_{i}") if use_fit else None
        if fit_file:
            dat = parse_fit(fit_file)
            if dat:
                dist, dup, ddn = dat["distance"], dat["D_up"], dat["D_down"]
                st.info(f"Distance={dist}m D+={dup} D-={ddn}")

    refs.append(dict(distance=dist, temps=t, D_up=dup, D_down=ddn))

# ------------------------------------------------------
# ⚙️ 3. PARAMÈTRES
# ------------------------------------------------------
st.header("3️⃣ Paramètres")

c1, c2 = st.columns(2)
with c1:
    k_up = st.number_input("k montée", value=1.001)
    k_down = st.number_input("k descente", value=0.999)
with c2:
    k_temp_sup = st.number_input("k_temp_sup (>20°C)", value=1.002)
    k_temp_inf = st.number_input("k_temp_inf (<20°C)", value=0.998)

lat = st.number_input("Latitude", value=48.8566)
lon = st.number_input("Longitude", value=2.3522)
API_KEY = st.text_input("Clé API météo", type="password")

date_course = st.date_input("Date", value=date.today())
heure_course = st.time_input("Heure", value=time(9, 0))

# ------------------------------------------------------
# FATIGUE
# ------------------------------------------------------
st.header("3️⃣ bis Fatigue linéaire")
fatigue_active = st.checkbox("Activer fatigue")
fatigue_rate = st.slider("Perte (%) fin course", 0.0, 30.0, 5.0) if fatigue_active else 0.0

# ------------------------------------------------------
# STOCKAGE (recalcul objectif)
# ------------------------------------------------------
if "rerun_with_objective" not in st.session_state:
    st.session_state.rerun_with_objective = False

if "second_distance" not in st.session_state:
    st.session_state.second_distance = None

if "second_time" not in st.session_state:
    st.session_state.second_time = None

# ------------------------------------------------------
# FONCTION DE PRÉDICTION
# ------------------------------------------------------
def run_prediction(distance_cible_km, objectif_temps_forced=None):
    gpx, points = parse_gpx_points(gpx_file)
    df_points = gpx_to_df(points)

    # distance GPX
    dists = [0]
    total = 0
    for i in range(1, len(points)):
        total += points[i].distance_3d(points[i - 1])
        dists.append(total)

    dist_gpx_km = total / 1000
    facteur = distance_cible_km / dist_gpx_km if dist_gpx_km > 0 else 1
    total_corr = total * facteur
    dists_corr = [d * facteur for d in dists]

    # modèle log-log
    a, K = fit_loglog_model(refs, k_up=k_up, k_down=k_down)
    st.info(f"K = {K:.4f}")

    if objectif_temps_forced:
        a = override_with_objective(int(distance_cible_km * 1000), objectif_temps_forced, K)
        st.success(f"🔄 Modèle recalé sur {distance_cible_km} km en {objectif_temps_forced}")

    base_flat = predict_time_flat(int(distance_cible_km * 1000), a, K)
    base_km_flat = base_flat / distance_cible_km

    km_marks = [i * 1000 for i in range(1, int(total_corr // 1000) + 1)]
    if total_corr % 1000 != 0:
        km_marks.append(total_corr)

    elev_list = [p.elevation or 0 for p in points]
    results = []
    cum = 0

    dt_depart = datetime.combine(date_course, heure_course)
    meteo = fetch_weather(API_KEY, lat, lon) if API_KEY else None

    for i, d in enumerate(km_marks):
        e_cur = np.interp(d, dists_corr, elev_list)
        e_prev = np.interp(d - 1000, dists_corr, elev_list) if i > 0 else e_cur
        du, dd = max(0, e_cur - e_prev), max(0, e_prev - e_cur)

        t_km = apply_elevation(base_km_flat, du, dd, k_up, k_down)

        # fatigue
        if fatigue_active and total_corr > 0:
            t_km *= 1 + (fatigue_rate / 100) * (d / total_corr)

        # météo
        passage = dt_depart + timedelta(seconds=cum + t_km)
        w = find_weather_entry(meteo, passage) if meteo else None
        if w:
            temp = w["temp"]
            if temp > 20:
                t_km *= k_temp_sup ** (temp - 20)
            else:
                t_km *= k_temp_inf ** (20 - temp)
        else:
            temp = 20

        cum += t_km
        results.append({
            "Km": i+1,
            "D+": round(du,1),
            "D-": round(dd,1),
            "Temp": round(temp,1),
            "Temps km": seconds_to_hms(t_km),
            "Cumul": seconds_to_hms(cum)
        })

    st.success(f"⏱ Temps total : {seconds_to_hms(cum)}")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

# ------------------------------------------------------
# 4️⃣ ANALYSE + REAJUSTEMENT
# ------------------------------------------------------
st.header("4️⃣ Analyse et prédiction")

if st.button("🚀 Lancer l’analyse complète"):
    if not gpx_file:
        st.error("Importer un GPX d'abord.")
        st.stop()

    gpx, pts = parse_gpx_points(gpx_file)
    total_gpx = sum(
        pts[i].distance_3d(pts[i-1]) for i in range(1, len(pts))
    ) / 1000

    distance_mode = st.radio(
        "Distance utilisée",
        ("GPX", "Forcer distance"),
        index=0
    )

    if distance_mode == "GPX":
        distance_cible = total_gpx
    else:
        distance_cible = st.number_input(
            "Distance forcée (km)", value=total_gpx, min_value=0.5
        )

    st.session_state.rerun_with_objective = False
    run_prediction(distance_cible)

    # Second module
    st.subheader("➕ Ajuster distance & temps pour une seconde estimation")
    st.session_state.second_distance = st.number_input(
        "Distance (km)", value=distance_cible, min_value=0.1
    )
    st.session_state.second_time = st.text_input(
        "Temps objectif (h:mm:ss)", placeholder="00:17:30"
    )

    if st.button("🔁 Recalculer avec cet objectif"):
        st.session_state.rerun_with_objective = True
        st.experimental_rerun()

if st.session_state.rerun_with_objective:
    d = st.session_state.second_distance
    t = st.session_state.second_time
    run_prediction(d, t)
