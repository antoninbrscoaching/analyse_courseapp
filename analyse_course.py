import streamlit as st
import math
import gpxpy
import gpxpy.gpx
import requests
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import numpy as np
from fitparse import FitFile
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go

st.set_page_config(page_title="Prédiction course - GPX/FIT + météo + carte + 3D", layout="wide")
st.title("🏃‍♂️ Prédiction de course (GPX ou FIT) avec météo, distance flexible, objectif, carte & profil 3D")

# ---------------- utilitaires ----------------
def hms_to_seconds(hms: str) -> int:
    """'h:mm:ss' -> seconds"""
    h, m, s = map(int, hms.strip().split(":"))
    return h*3600 + m*60 + s

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

def gpx_cumulative_distance_and_elev(points):
    """Retourne (cum_d en m, elevs en m, total en m) + dataframe lon/lat/elev/time si disponible"""
    if not points:
        return [0.0], [0.0], 0.0, pd.DataFrame()
    cum_d = [0.0]
    elevs = [points[0].elevation if points[0].elevation is not None else 0.0]
    total = 0.0
    rows = []
    t0 = points[0].time if points[0].time else None
    for i in range(len(points)):
        if i > 0:
            d = points[i].distance_3d(points[i-1])
            total += d
            cum_d.append(total)
        e = points[i].elevation if points[i].elevation is not None else (elevs[-1] if elevs else 0.0)
        elevs.append(e) if i == 0 else None
        ts = points[i].time
        rows.append({
            "lon": points[i].longitude, "lat": points[i].latitude, "elev": e,
            "time": ts, "t_rel_s": (ts - t0).total_seconds() if (ts and t0) else None,
            "cum_d_m": total
        })
    df = pd.DataFrame(rows)
    return cum_d, elevs, total, df

def interp_elevation_at(dist_target, cum_d, elevs):
    if dist_target <= 0:
        return elevs[0]
    if dist_target >= cum_d[-1]:
        return elevs[-1]
    for i in range(1, len(cum_d)):
        if cum_d[i] >= dist_target:
            d0, d1 = cum_d[i-1], cum_d[i]
            e0, e1 = elevs[i-1], elevs[i]
            frac = (dist_target - d0) / (d1 - d0) if (d1 - d0) > 0 else 0.0
            return e0 + frac * (e1 - e0)
    return elevs[-1]

def parse_fit_points(file) -> pd.DataFrame:
    """Lit un .FIT et renvoie un DataFrame ordonné avec lon/lat/elev/time/t_rel_s/cum_d_m."""
    fit = FitFile(file)
    fit.parse()
    rows = []
    for msg in fit.get_messages("record"):
        vals = {d.name: d.value for d in msg}
        # Champs possibles : position_lat/long (semicircles), altitude (m), distance (m), timestamp (UTC)
        lat = vals.get("position_lat")
        lon = vals.get("position_long")
        if lat is None or lon is None:
            continue
        # conversion semicircles -> degrés
        lat = lat * (180 / 2**31)
        lon = lon * (180 / 2**31)
        elev = vals.get("altitude")  # déjà en mètres
        ts = vals.get("timestamp")   # datetime en UTC
        dist = vals.get("distance")  # m (cumulé si dispo)
        rows.append({"lon": lon, "lat": lat, "elev": elev, "time": ts, "dist_m": dist})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    # temps relatif
    t0 = df["time"].iloc[0]
    df["t_rel_s"] = (df["time"] - t0).dt.total_seconds()
    # distance cumulée : si non fournie on la recalcule à partir des points
    if df["dist_m"].isna().all():
        # haversine 3D approximée (sans altitude pour simplicité, suffisant ici)
        R = 6371000
        dists = [0.0]
        for i in range(1, len(df)):
            lat1, lon1 = np.radians(df.loc[i-1, "lat"]), np.radians(df.loc[i-1, "lon"])
            lat2, lon2 = np.radians(df.loc[i, "lat"]), np.radians(df.loc[i, "lon"])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
            d = R*c
            dists.append(d)
        df["cum_d_m"] = np.cumsum(dists)
    else:
        # certaines montres donnent dist_m cumulée ; on s'assure que c'est croissant
        df["cum_d_m"] = df["dist_m"].ffill().fillna(0.0)
    # élévation manquante -> ffill
    df["elev"] = df["elev"].ffill().bfill().fillna(0.0)
    return df[["lon", "lat", "elev", "time", "t_rel_s", "cum_d_m"]]

@st.cache_data(ttl=600)
def fetch_weather_forecast(api_key, lat, lon):
    """Récupère la prévision horaire (onecall / forecast)."""
    if not api_key:
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if "hourly" in data:
                return {"type": "onecall", "data": data["hourly"], "tz_offset": data.get("timezone_offset", 0)}
        url2 = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r2 = requests.get(url2, timeout=10)
        if r2.status_code == 200:
            data2 = r2.json()
            return {"type": "forecast3h", "data": data2.get("list", []), "tz_offset": 0}
    except Exception:
        return None
    return None

def find_closest_weather_entry(weather_cache, target_dt):
    if weather_cache is None:
        return None
    entries = weather_cache["data"]
    best = min(entries, key=lambda x: abs(datetime.fromtimestamp(x["dt"]) - target_dt))
    temp = best.get("temp") or best.get("main", {}).get("temp")
    wind = best.get("wind", {}).get("speed") if best.get("wind") else best.get("wind_speed") if best.get("wind_speed") else None
    if wind is None:
        wind = best.get("wind", {}).get("speed", 0)
    return {"temp": temp, "wind": wind}

# ---------------- UI ----------------
st.markdown("### 1️⃣ Source du parcours")
src = st.radio("Choisis la source de tes points", ["GPX", "FIT"], horizontal=True)

gpx_file = None
fit_file = None
if src == "GPX":
    gpx_file = st.file_uploader("Fichier GPX", type=["gpx"])
else:
    fit_file = st.file_uploader("Fichier FIT (export Nolio/Strava)", type=["fit"])

st.markdown("#### Fenêtre de temps (optionnelle, pour tronquer l'effort)")
c_from, c_to = st.columns(2)
with c_from:
    t_debut_txt = st.text_input("De (h:mm:ss)", value="0:00:00")
with c_to:
    t_fin_txt = st.text_input("À (h:mm:ss) — vide = fin", value="")

st.markdown("### 2️⃣ Courses de référence (régression log-log)")
courses = []
for i in range(1,4):
    c0, c1, c2, c3 = st.columns([1.2,1.2,1,1])
    with c0:
        dist_i = st.number_input(f"Dist {i} (m)", min_value=1, step=100, value=5000*i, key=f"dist{i}")
    with c1:
        temps_i = st.text_input(f"Temps {i} (h:mm:ss)", value=f"0:{40+i*2}:00", key=f"temps{i}")
    with c2:
        up_i = st.number_input(f"D+ {i} (m)", min_value=0, step=1, value=0, key=f"up{i}")
    with c3:
        down_i = st.number_input(f"D- {i} (m)", min_value=0, step=1, value=0, key=f"down{i}")
    courses.append({"distance": dist_i, "temps": temps_i, "D_up": up_i, "D_down": down_i})

st.markdown("### 3️⃣ Coefficients (modifiable)")
c1, c2 = st.columns(2)
with c1:
    k_up = st.number_input("k_montée (exponentiel)", min_value=1.0, max_value=2.0, step=0.00001, value=1.00100, format="%.5f")
    k_down = st.number_input("k_descente", min_value=0.90, max_value=1.0, step=0.00001, value=0.99900, format="%.5f")
with c2:
    k_temp_sup = st.number_input("k_temp_sup (>20°C)", min_value=0.90, max_value=1.1, step=0.00001, value=1.00200, format="%.5f")
    k_temp_inf = st.number_input("k_temp_inf (<20°C)", min_value=0.90, max_value=1.1, step=0.00001, value=0.99800, format="%.5f")

st.markdown("### 4️⃣ Paramètres de course")
col1, col2, col3 = st.columns(3)
with col1:
    utiliser_distance_gpx = st.checkbox("Utiliser la distance du fichier", value=True)
    distance_officielle = st.number_input("Distance manuelle (m, si non fichier)", min_value=1000, step=1, value=42195)
with col2:
    objectif_temps = st.text_input("Objectif de temps (h:mm:ss, optionnel)", value="")
with col3:
    latitude = st.number_input("Latitude (pour météo)", value=48.8566)
    longitude = st.number_input("Longitude (pour météo)", value=2.3522)
    API_KEY = st.text_input("Clé API OpenWeather", type="password")
date_course = st.date_input("Date de la course")
heure_course = st.time_input("Heure de départ")

# bouton
if st.button("Lancer l'analyse complète"):
    try:
        # ------- lecture & fenêtrage des points -------
        df_points = pd.DataFrame()
        if src == "GPX":
            if gpx_file is None:
                st.error("⚠️ Upload d'abord un fichier GPX.")
                st.stop()
            gpx, points = parse_gpx_points(gpx_file)
            cum_d, elevs, total_len_m, df_points = gpx_cumulative_distance_and_elev(points)
        else:
            if fit_file is None:
                st.error("⚠️ Upload d'abord un fichier FIT.")
                st.stop()
            df_points = parse_fit_points(fit_file)
            if df_points.empty:
                st.error("Impossible de lire des points valides dans le FIT.")
                st.stop()
            # pour homogénéité avec GPX
            cum_d = df_points["cum_d_m"].tolist()
            elevs = df_points["elev"].tolist()
            total_len_m = float(df_points["cum_d_m"].iloc[-1])

        st.write(f"📈 Points lus : {len(df_points)}")
        st.write(f"Distance fichier : {total_len_m:.1f} m")

        # Fenêtre de temps
        t_debut = hms_to_seconds(t_debut_txt) if t_debut_txt.strip() else 0
        t_fin = hms_to_seconds(t_fin_txt) if t_fin_txt.strip() else None
        if df_points["t_rel_s"].notna().any():
            if t_fin is None:
                mask = df_points["t_rel_s"] >= t_debut
            else:
                if t_fin <= t_debut:
                    st.warning("⏱️ 'À' doit être > 'De'. Fenêtre ignorée.")
                    mask = slice(None)
                else:
                    mask = (df_points["t_rel_s"] >= t_debut) & (df_points["t_rel_s"] <= t_fin)
            df_points = df_points.loc[mask].copy()
            # réindexe / recalc cum_d si besoin
            if src == "GPX":
                # recalc distances de la fenêtre pour cohérence
                # simple haversine
                R = 6371000
                dists = [0.0]
                for i in range(1, len(df_points)):
                    lat1, lon1 = np.radians(df_points.iloc[i-1]["lat"]), np.radians(df_points.iloc[i-1]["lon"])
                    lat2, lon2 = np.radians(df_points.iloc[i]["lat"]), np.radians(df_points.iloc[i]["lon"])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
                    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    d = R*c
                    dists.append(d)
                df_points["cum_d_m"] = np.cumsum(dists)
                cum_d = df_points["cum_d_m"].tolist()
                elevs = df_points["elev"].tolist()
            total_len_m = float(df_points["cum_d_m"].iloc[-1]) if not df_points.empty else 0.0

        # Distance utilisée
        distance_analyse = total_len_m if utiliser_distance_gpx else distance_officielle
        st.info(f"Distance utilisée pour calcul : {distance_analyse:.1f} m")

        # ✅ Création segments (1 km + dernier partiel)
        km_marks = [i * 1000 for i in range(1, int(distance_analyse // 1000) + 1)]
        if distance_analyse % 1000 != 0:
            km_marks.append(distance_analyse)

        def interp_elevs_series(cum_d_series, elev_series, targets_m):
            out = []
            for d in targets_m:
                out.append(interp_elevation_at(d, cum_d_series, elev_series))
            return out

        elev_at_km = [interp_elevation_at(k, cum_d, elevs) for k in [0] + km_marks]
        per_km = []
        for i in range(1, len(elev_at_km)):
            delta_h = elev_at_km[i] - elev_at_km[i-1]
            seg_len = (km_marks[i-1] - (km_marks[i-2] if i > 1 else 0)) / 1000
            per_km.append({
                "D_up": max(0, delta_h),
                "D_down": max(0, -delta_h),
                "length_km": seg_len
            })

        # Régression log-log
        temps_sec, dists_ref = [], []
        for c in courses:
            t = hms_to_seconds(c["temps"])
            t_adj = t * (k_up ** c["D_up"]) * (k_down ** c["D_down"])
            temps_sec.append(t_adj)
            dists_ref.append(c["distance"])
        # moyenne des 3 paires (3 valeurs pour 3 références)
        K_pred = sum(
            math.log(temps_sec[j]/temps_sec[i]) / math.log(dists_ref[j]/dists_ref[i])
            for i in range(len(courses)) for j in range(i+1,len(courses))
        ) / 3
        st.info(f"Exposant log-log estimé K = {K_pred:.4f}")

        if objectif_temps.strip() != "":
            base_total_time = hms_to_seconds(objectif_temps)
            st.success(f"Objectif de temps : {objectif_temps}")
        else:
            base_total_time = temps_sec[-1]*(distance_analyse/dists_ref[-1])**K_pred

        base_s_per_km = base_total_time / (distance_analyse/1000)

        # Météo (heure locale de la course)
        weather_cache = fetch_weather_forecast(API_KEY, latitude, longitude)
        dt_depart = datetime.combine(date_course, heure_course)

        km_results = []
        cum_time = 0.0
        for idx, km in enumerate(per_km):
            t_km = base_s_per_km * km["length_km"]
            t_km *= (k_up ** km["D_up"]) * (k_down ** km["D_down"])
            passage_dt = dt_depart + timedelta(seconds=(cum_time + t_km))
            weather = find_closest_weather_entry(weather_cache, passage_dt) if weather_cache else None
            temp = weather["temp"] if weather and weather.get("temp") is not None else 20
            if temp > 20:
                t_km *= (k_temp_sup ** (temp - 20))
            else:
                t_km *= (k_temp_inf ** (20 - temp))
            cum_time += t_km

            dist_cumulee_km = round(sum(seg["length_km"] for seg in per_km[:idx+1]), 3)
            km_label = f"Km {idx+1}" if km["length_km"] >= 0.999 else f"Km {dist_cumulee_km}"

            km_results.append({
                "Segment": km_label,
                "Km cumulé": dist_cumulee_km,
                "Distance segment (km)": round(km["length_km"], 3),
                "D+ (m)": round(km["D_up"], 1),
                "D- (m)": round(km["D_down"], 1),
                "Temp (°C)": round(float(temp), 1) if temp is not None else None,
                "Temps segment (s)": round(t_km, 1),
                "Allure (min/km)": f"{int((t_km/km['length_km'])//60)}:{int((t_km/km['length_km'])%60):02d}"
            })

        # Affichage des km cumulés
        liste_km = [round(sum(seg["length_km"] for seg in per_km[:i+1]), 3) for i in range(len(per_km))]
        st.markdown(f"**Kilomètres cumulés :** {' - '.join(map(str, liste_km))}")

        total_sec = sum(r["Temps segment (s)"] for r in km_results)
        avg_pace = total_sec / (distance_analyse / 1000)
        st.subheader("⏱️ Résultats")
        st.write(f"Temps total prévisionnel = {seconds_to_hms(total_sec)}")
        st.write(f"Allure moyenne = {int(avg_pace//60)}:{int(avg_pace%60):02d} min/km")

        # -------- VISU CARTE ----------
        st.subheader("🗺️ Carte interactive du parcours")
        if not df_points.empty:
            mid_lat = float(df_points["lat"].mean())
            mid_lon = float(df_points["lon"].mean())
            path_coords = df_points[["lon","lat"]].to_numpy().tolist()
            deck = pdk.Deck(
                initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=12, pitch=45),
                map_style="mapbox://styles/mapbox/outdoors-v12",
                layers=[
                    pdk.Layer(
                        "PathLayer",
                        data=[{"path": path_coords, "name": "Parcours"}],
                        get_path="path",
                        get_width=4,
                        width_min_pixels=2,
                    )
                ],
                tooltip={"text": "{name}"}
            )
            st.pydeck_chart(deck)
        else:
            st.info("Pas de points pour afficher la carte.")

        # -------- PROFIL & ALLURE 2D ----------
        st.subheader("📊 Profil & Allure")
        km_idxs = list(range(1, len(km_results)+1))
        elevs_km = [interp_elevation_at(min(i*1000, cum_d[-1]), cum_d, elevs) for i in range(len(km_idxs))]
        paces = [r["Temps segment (s)"]/60 for r in km_results]

        fig, ax1 = plt.subplots(figsize=(14,4))
        ax1.plot(km_idxs, elevs_km)
        ax1.set_xlabel("Segment")
        ax1.set_ylabel("Altitude (m)")
        ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.plot(km_idxs, paces)
        ax2.set_ylabel("Allure (min/km)")
        st.pyplot(fig)

        st.dataframe(pd.DataFrame(km_results))

        # -------- PROFIL 3D ----------
        st.subheader("🧭 Profil 3D (lon / lat / altitude)")
        if not df_points.empty:
            # on échantillonne pour performance si énorme trace
            df3d = df_points.iloc[::max(1, len(df_points)//3000)].copy()
            fig3d = go.Figure(
                data=[go.Scatter3d(
                    x=df3d["lon"], y=df3d["lat"], z=df3d["elev"],
                    mode="lines",
                    line=dict(width=3),
                )]
            )
            fig3d.update_layout(
                scene=dict(
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    zaxis_title="Altitude (m)",
                    aspectmode="data"
                ),
                margin=dict(l=0,r=0,t=0,b=0),
                height=500
            )
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("Pas de points pour afficher le profil 3D.")

    except Exception as e:
        st.error(f"Erreur durant l'analyse : {e}")
