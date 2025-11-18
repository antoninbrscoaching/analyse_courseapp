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
    return pd.DataFrame(
        [{"lat": p.latitude, "lon": p.longitude, "elev": p.elevation or 0} for p in points]
    )

def parse_fit(file):
    """
    Récupère distance, D+ et D- depuis un fichier FIT.
    """
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
    """
    Appel OpenWeather OneCall (hourly). Cache 10 minutes.
    """
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
# 🧠 MODÈLE LOG-LOG (COEUR MATH)
# ------------------------------------------------------
def fit_loglog_model(refs, k_up=1.0, k_down=1.0):
    """
    Ajuste le modèle log-log T = a * D^K à partir des références.

    refs : liste de dicts :
        {
            "distance": en mètres,
            "temps": "h:mm:ss",
            "D_up": D+ en m,
            "D_down": D- en m
        }

    On ramène d'abord chaque temps à un temps "équivalent plat"
    en ENLEVANT l'effet du dénivelé via k_up / k_down.
    """
    xs, ys = [], []
    for r in refs:
        d = r["distance"]
        t_raw = hms_to_seconds(r["temps"])
        dup = r.get("D_up", 0)
        ddn = r.get("D_down", 0)

        # facteur d'effet du dénivelé (le même que tu appliques ensuite en prévision)
        elev_factor = (k_up ** dup) * (k_down ** ddn)

        # Temps équivalent plat (on enlève l'effet du D+ / D-)
        if elev_factor > 0:
            t_eq = t_raw / elev_factor
        else:
            t_eq = t_raw

        if d > 0 and t_eq > 0:
            xs.append(math.log(d))
            ys.append(math.log(t_eq))

    if len(xs) < 2:
        raise ValueError("Il faut au moins deux références valides pour ajuster le modèle log-log.")

    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))

    denom = n * sum_xx - sum_x ** 2
    if denom == 0:
        raise ValueError("Références dégénérées (distances identiques ?).")

    K = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - K * sum_x) / n
    a = math.exp(intercept)
    return a, K

def predict_time_flat(distance_m, a, K):
    """Temps prédit sur parcours plat, en secondes, à partir de T = a * D^K."""
    return a * (distance_m ** K)

def apply_elevation(time_flat_s, D_up, D_down, k_up=1.0, k_down=1.0):
    """Applique l'effet du D+ / D- local à un temps plat."""
    return time_flat_s * (k_up ** D_up) * (k_down ** D_down)

def override_with_objective(distance_obj_m, time_obj_hms, K):
    """
    Recalage du modèle pour passer exactement par (distance_obj, temps_obj).
    On garde K issu des références, on ajuste seulement a.
    """
    t_obj = hms_to_seconds(time_obj_hms)
    if distance_obj_m <= 0:
        raise ValueError("La distance objective doit être > 0.")
    a_new = t_obj / (distance_obj_m ** K)
    return a_new

# ------------------------------------------------------
# 🗺️ 1. CHARGEMENT GPX
# ------------------------------------------------------
st.header("1️⃣ Parcours GPX")
st.caption("Charge le fichier GPX de ton parcours.")
gpx_file = st.file_uploader("📂 Importer un fichier GPX", type=["gpx"])

# ------------------------------------------------------
# 🧮 2. COURSES DE RÉFÉRENCE
# ------------------------------------------------------
st.header("2️⃣ Courses de référence (manuel ou fichiers FIT)")
if "n_refs" not in st.session_state:
    st.session_state.n_refs = 3

cols = st.columns([1, 1])
with cols[0]:
    if st.button("➕ Ajouter une référence (max 6)") and st.session_state.n_refs < 6:
        st.session_state.n_refs += 1
with cols[1]:
    if st.button("➖ Retirer une référence") and st.session_state.n_refs > 1:
        st.session_state.n_refs -= 1

st.caption("Ces références servent à estimer ton profil de performance (modèle log-log).")

refs = []
for i in range(1, st.session_state.n_refs + 1):
    st.markdown(f"#### Référence {i}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        use_fit = st.checkbox(f"FIT ?", key=f"fit_use_{i}")
    with c2:
        dist = st.number_input(f"Dist {i} (m)", value=5000 * i, step=100)
    with c3:
        time_str = st.text_input(f"Temps {i} (h:mm:ss)", value=f"0:{40 + i * 2}:00")
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
    st.markdown("**Paramètres dénivelé**")
    k_up = st.number_input("k_montée", value=1.001, help="Facteur multiplicatif par mètre de D+.")
    k_down = st.number_input("k_descente", value=0.999, help="Facteur multiplicatif par mètre de D-.")
with c2:
    st.markdown("**Paramètres température**")
    k_temp_sup = st.number_input(
        "k_temp_sup (>20°C)", value=1.002, help="Impact par degré au-dessus de 20°C."
    )
    k_temp_inf = st.number_input(
        "k_temp_inf (<20°C)", value=0.998, help="Impact par degré en-dessous de 20°C."
    )

st.markdown("### Paramètres météo et objectif de course")
col1, col2, col3 = st.columns(3)

with col1:
    lat = st.number_input("Latitude", value=48.8566, help="Latitude du lieu de la course.")
    lon = st.number_input("Longitude", value=2.3522, help="Longitude du lieu de la course.")
    API_KEY = st.text_input(
        "Clé API OpenWeather",
        type="password",
        help="Clé API pour récupérer la météo (optionnel)."
    )

with col2:
    date_course = st.date_input("Date de la course", value=date.today())
    heure_course = st.time_input("Heure départ", value=time(9, 0))

with col3:
    st.markdown("**Objectif de performance (optionnel)**")
    objectif_distance_km = st.number_input(
        "Distance de l'objectif (km)",
        value=5.0,
        min_value=0.5,
        step=0.5,
        help="Ex : 5 km si tu vises un chrono spécifique sur 5 km."
    )
    objectif_temps = st.text_input(
        "Temps objectif (h:mm:ss)",
        value="",
        placeholder="Ex : 0:17:30 pour 17min30",
        help=(
            "Si renseigné, le modèle log-log sera recalé pour passer par ce point "
            "(distance objectif, temps objectif)."
        )
    )

# Pré-charger la météo à chaque rafraîchissement de l'appli
meteo_data = fetch_weather(API_KEY, lat, lon) if API_KEY else None
if API_KEY and meteo_data:
    st.caption("🌤️ Données météo chargées pour ce lieu (mise à jour auto).")
elif API_KEY and not meteo_data:
    st.warning("Impossible de récupérer la météo avec cette clé / position.")

# ------------------------------------------------------
# 💤 3️⃣ bis. FATIGUE LINÉAIRE
# ------------------------------------------------------
st.header("3️⃣ bis. Fatigue linéaire (optionnelle)")
st.caption(
    "L’allure régresse de manière linéaire sur toute la durée de la course "
    "(indépendamment du D+ ou de la météo)."
)

fatigue_active = st.checkbox("Activer la fatigue linéaire", value=False)
fatigue_rate = 0.0
if fatigue_active:
    fatigue_rate = st.slider(
        "Pourcentage de régression à la fin de la course (%)",
        min_value=0.0,
        max_value=30.0,
        step=0.5,
        value=5.0,
        help="Ex : 5% signifie que ton allure est 5% plus lente à la fin qu’au début, de façon linéaire."
    )

# ------------------------------------------------------
# 🧠 4. ANALYSE
# ------------------------------------------------------
st.header("4️⃣ Analyse et prédiction")
st.caption("Clique sur le bouton ci-dessous une fois tous les paramètres renseignés.")

if st.button("🚀 Lancer l’analyse complète"):
    if not gpx_file:
        st.error("⚠️ Upload d’abord un fichier GPX.")
        st.stop()

    gpx, points = parse_gpx_points(gpx_file)
    df_points = gpx_to_df(points)
    if df_points.empty:
        st.error("Fichier GPX invalide.")
        st.stop()

    # -----------------------------
    # Distance cumulée brute (GPX)
    # -----------------------------
    dists = [0]
    total = 0
    for i in range(1, len(points)):
        total += points[i].distance_3d(points[i - 1])
        dists.append(total)

    distance_gpx_km = total / 1000
    st.info(f"📏 Distance totale GPX : {distance_gpx_km:.2f} km")

    # -----------------------------
    # Option A : distance GPX OU distance forcée
    # -----------------------------
    distance_mode = st.radio(
        "Distance utilisée pour la prédiction",
        ("Utiliser la distance du GPX", "Forcer une distance"),
        index=0
    )

    if distance_mode == "Utiliser la distance du GPX":
        distance_cible_km = distance_gpx_km
        facteur_dist = 1.0
        st.success(f"📏 Distance utilisée : {distance_cible_km:.2f} km (GPX)")
    else:
        distance_forced_km = st.number_input(
            "Distance forcée (km)",
            value=float(round(distance_gpx_km, 2)),
            min_value=0.5,
            step=0.1,
            help="Permet de forcer une distance officielle si le GPX n'est pas précis."
        )
        distance_cible_km = distance_forced_km
        if distance_gpx_km > 0:
            facteur_dist = distance_forced_km / distance_gpx_km
        else:
            facteur_dist = 1.0
        st.success(f"📏 Distance utilisée : {distance_cible_km:.2f} km (forcée)")

    total_corr = total * facteur_dist
    dists_corr = [d * facteur_dist for d in dists]  # on étire/comprime l'axe distance

    # -----------------------------
    # Ajustement du modèle log-log
    # -----------------------------
    try:
        a, K = fit_loglog_model(refs, k_up=k_up, k_down=k_down)
    except ValueError as e:
        st.error(f"Problème lors de l’ajustement du modèle log-log : {e}")
        st.stop()

    st.info(f"📐 Exposant log-log estimé : {K:.4f}")

    # -----------------------------
    # Recalage éventuel avec un objectif
    # (ex : 17:30 sur 5 km plat)
    # -----------------------------
    if objectif_temps:
        d_obj_m = int(objectif_distance_km * 1000)
        try:
            a = override_with_objective(d_obj_m, objectif_temps, K)
            st.success(
                f"🎯 Modèle recalé pour passer par "
                f"{objectif_distance_km:.1f} km en {objectif_temps} (sur plat)."
            )
        except ValueError as e:
            st.warning(f"Objectif ignoré (problème de saisie) : {e}")

    # -----------------------------
    # Temps plat global sur la distance cible
    # -----------------------------
    distance_cible_m = int(distance_cible_km * 1000)
    base_flat_total = predict_time_flat(distance_cible_m, a, K)
    base_s_per_km_flat = base_flat_total / distance_cible_km

    dt_depart = datetime.combine(date_course, heure_course)

    # -----------------------------
    # Prévision km par km
    # -----------------------------
    km_marks = [i * 1000 for i in range(1, int(total_corr // 1000) + 1)]
    if total_corr % 1000 != 0:
        km_marks.append(total_corr)

    results = []
    cum_time = 0

    elev_list = [p.elevation or 0 for p in points]

    for i, d in enumerate(km_marks):
        # Altitude interpolée sur la distance corrigée
        e_cur = np.interp(d, dists_corr, elev_list)
        if i > 0:
            e_prev = np.interp(d - 1000, dists_corr, elev_list)
        else:
            e_prev = e_cur

        d_up = max(0, e_cur - e_prev)
        d_down = max(0, e_prev - e_cur)

        # Temps plat "moyen" sur ce km
        t_km_flat = base_s_per_km_flat

        # Application du D+ / D- local
        t_km = apply_elevation(t_km_flat, d_up, d_down, k_up=k_up, k_down=k_down)

        # 👉 Fatigue linéaire
        if fatigue_active and fatigue_rate > 0 and total_corr > 0:
            progression = d / total_corr  # de 0 à 1 sur la course
            fatigue_mult = 1.0 + (fatigue_rate / 100.0) * progression
            t_km *= fatigue_mult
        else:
            fatigue_mult = 1.0

        # Météo (ajustement indépendant)
        passage = dt_depart + timedelta(seconds=cum_time + t_km)
        w = find_weather_entry(meteo_data, passage) if meteo_data else None
        temp = w["temp"] if w else 20
        if temp > 20:
            t_km *= (k_temp_sup ** (temp - 20))
        else:
            t_km *= (k_temp_inf ** (20 - temp))

        cum_time += t_km
        results.append(
            {
                "Km": i + 1,
                "D+ (m)": round(d_up, 1),
                "D- (m)": round(d_down, 1),
                "Temp (°C)": round(temp, 1),
                "Fatigue (%)": f"{(fatigue_mult - 1) * 100:.2f}%",
                "Temps segment (s)": round(t_km, 1),
                "Allure (min/km)": f"{int(t_km // 60)}:{int(t_km % 60):02d}",
                "Temps cumulé": seconds_to_hms(cum_time),
            }
        )

    total_time = seconds_to_hms(sum(r["Temps segment (s)"] for r in results))
    st.success(f"⏱️ Temps total prévisionnel : {total_time}")

    st.subheader("📋 Détails km par km")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)

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
        data=[{"path": df_points[["lon", "lat"]].values.tolist(), "name": "Parcours"}],
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
    x_km = np.linspace(0, distance_cible_km, len(df_points))
    plt.plot(x_km, df_points["elev"])
    plt.xlabel("Distance (km)")
    plt.ylabel("Altitude (m)")
    plt.title("Profil d’altitude du parcours")
    plt.grid(alpha=0.3)
    st.pyplot(plt)
