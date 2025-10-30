import streamlit as st
import math
import gpxpy
import gpxpy.gpx
import requests
from datetime import datetime, timedelta, date, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fitparse import FitFile
import pydeck as pdk
import plotly.graph_objects as go

st.set_page_config(page_title="Prédiction course - GPX + Références FIT + Météo", layout="wide")
st.title("🏃‍♂️ Prédiction de course (GPX) + Références (FIT ou manuel) + Météo des références")

# ---------------- utilitaires ----------------
def hms_to_seconds(hms: str) -> int:
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
        if i == 0:
            elevs[0] = e
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
    """Lit un .FIT et renvoie lon/lat/elev/time,t_rel_s,cum_d_m (croissant)."""
    fit = FitFile(file)
    fit.parse()
    rows = []
    for msg in fit.get_messages("record"):
        vals = {d.name: d.value for d in msg}
        lat = vals.get("position_lat")
        lon = vals.get("position_long")
        if lat is None or lon is None:
            continue
        lat = lat * (180 / 2**31)
        lon = lon * (180 / 2**31)
        elev = vals.get("altitude")
        ts = vals.get("timestamp")
        dist = vals.get("distance")
        rows.append({"lon": lon, "lat": lat, "elev": elev, "time": ts, "dist_m": dist})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    t0 = df["time"].iloc[0]
    df["t_rel_s"] = (df["time"] - t0).dt.total_seconds()
    if df["dist_m"].isna().all():
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
        df["cum_d_m"] = df["dist_m"].ffill().fillna(0.0)
    df["elev"] = df["elev"].ffill().bfill().fillna(0.0)
    return df[["lon", "lat", "elev", "time", "t_rel_s", "cum_d_m"]]

def compute_gain_loss(elev_series: pd.Series) -> tuple[float, float]:
    """Retourne (D+, D-) en mètres à partir d'une série d'altitudes."""
    if elev_series is None or len(elev_series) < 2:
        return 0.0, 0.0
    diffs = np.diff(elev_series.fillna(method="ffill").fillna(method="bfill"))
    up = np.sum(diffs[diffs > 0])
    down = -np.sum(diffs[diffs < 0])
    return float(up), float(down)

@st.cache_data(ttl=600)
def fetch_weather_forecast(api_key, lat, lon):
    """Prévision proche (pour la course cible) via OpenWeather si clé fournie."""
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

@st.cache_data(ttl=3600)
def fetch_historical_weather_openmeteo(lat: float, lon: float, when_dt: datetime):
    """
    Météo historique (température 2m, vent 10m) via Open-Meteo ERA5 (sans clé API).
    On récupère l'heure la plus proche du datetime fourni (timezone=auto).
    """
    start = when_dt.date().isoformat()
    end = start
    url = (
        "https://archive-api.open-meteo.com/v1/era5?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,wind_speed_10m"
        "&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        hours = data.get("hourly", {}).get("time", [])
        temps = data.get("hourly", {}).get("temperature_2m", [])
        winds = data.get("hourly", {}).get("wind_speed_10m", [])
        if not hours:
            return None
        # choisir l'heure la plus proche
        target = pd.to_datetime(when_dt)
        tseries = pd.to_datetime(hours)
        idx = int(np.argmin(np.abs((tseries - target).values)))
        temp = float(temps[idx]) if idx < len(temps) else None
        wind = float(winds[idx]) if idx < len(winds) else None
        return {"temp": temp, "wind": wind, "time": str(tseries[idx])}
    except Exception:
        return None

# ---------------- UI ----------------
st.markdown("### 1️⃣ Upload GPX (parcours à analyser)")
gpx_file = st.file_uploader("Fichier GPX", type=["gpx"])

st.markdown("### 2️⃣ Temps de référence (manuel **ou** import .FIT)")
# gestion dynamique 3 -> 6 références
if "n_refs" not in st.session_state:
    st.session_state.n_refs = 3

cols_btn = st.columns([1,1,6])
with cols_btn[0]:
    if st.button("➕ Ajouter (max 6)") and st.session_state.n_refs < 6:
        st.session_state.n_refs += 1
with cols_btn[1]:
    if st.button("➖ Retirer") and st.session_state.n_refs > 1:
        st.session_state.n_refs -= 1

references = []
for i in range(1, st.session_state.n_refs + 1):
    st.markdown(f"**Référence {i}**")
    c0, c1, c2, c3, c4 = st.columns([1.1,1.1,1.1,1.1,1.3])
    with c0:
        use_fit = st.checkbox(f"Importer FIT {i}", value=False, key=f"use_fit_{i}")
    with c1:
        dist_i = st.number_input(f"Dist {i} (m)", min_value=1, step=100, value=5000*i, key=f"dist_{i}")
    with c2:
        temps_i = st.text_input(f"Temps {i} (h:mm:ss)", value=f"0:{40+i*2}:00", key=f"temps_{i}")
    with c3:
        up_i = st.number_input(f"D+ {i} (m)", min_value=0, step=1, value=0, key=f"up_{i}")
    with c4:
        down_i = st.number_input(f"D- {i} (m)", min_value=0, step=1, value=0, key=f"down_{i}")

    c5, c6, c7 = st.columns(3)
    with c5:
        ref_date = st.date_input(f"Date ref {i}", value=date.today(), key=f"date_{i}")
        ref_time = st.time_input(f"Heure ref {i}", value=time(9,0,0), key=f"time_{i}")
    with c6:
        lat_i = st.number_input(f"Lat ref {i}", value=48.8566, format="%.6f", key=f"lat_{i}")
        lon_i = st.number_input(f"Lon ref {i}", value=2.3522, format="%.6f", key=f"lon_{i}")
    with c7:
        fit_file = st.file_uploader(f"FIT {i}", type=["fit"], key=f"fit_{i}") if use_fit else None
        fit_info_placeholder = st.empty()

    # si FIT coché et fourni -> remplir automatiquement
    auto_vals = {}
    if use_fit and fit_file is not None:
        df_fit = parse_fit_points(fit_file)
        if not df_fit.empty:
            dist_auto = float(df_fit["cum_d_m"].iloc[-1])
            dup, ddn = compute_gain_loss(df_fit["elev"])
            start_dt = pd.to_datetime(df_fit["time"].iloc[0]).to_pydatetime()
            lat_mean = float(df_fit["lat"].mean())
            lon_mean = float(df_fit["lon"].mean())
            auto_vals = {
                "distance": round(dist_auto),
                "D_up": round(dup),
                "D_down": round(ddn),
                "date": start_dt.date(),
                "time": start_dt.time().replace(microsecond=0),
                "lat": round(lat_mean, 6),
                "lon": round(lon_mean, 6)
            }
            fit_info_placeholder.info(
                f"FIT détecté → Dist ~ {auto_vals['distance']} m | D+ {auto_vals['D_up']} m | D- {auto_vals['D_down']} m | "
                f"Départ {auto_vals['date']} {auto_vals['time']} | ({auto_vals['lat']}, {auto_vals['lon']})"
            )
            # on utilise les valeurs auto dans la régression, mais on NE modifie pas tes champs (tu peux les écraser manuellement)
            dist_i = auto_vals["distance"]
            up_i = auto_vals["D_up"]
            down_i = auto_vals["D_down"]
            ref_date = auto_vals["date"]
            ref_time = auto_vals["time"]
            lat_i = auto_vals["lat"]
            lon_i = auto_vals["lon"]
        else:
            fit_info_placeholder.error("Impossible de lire des points valides dans ce FIT.")

    references.append({
        "distance": dist_i,
        "temps": temps_i,
        "D_up": up_i,
        "D_down": down_i,
        "date": ref_date,
        "time": ref_time,
        "lat": lat_i,
        "lon": lon_i
    })

st.markdown("### 3️⃣ Coefficients (modifiable)")
c1, c2 = st.columns(2)
with c1:
    k_up = st.number_input("k_montée (exponentiel)", min_value=1.0, max_value=2.0, step=0.00001, value=1.00100, format="%.5f")
    k_down = st.number_input("k_descente", min_value=0.90, max_value=1.0, step=0.00001, value=0.99900, format="%.5f")
with c2:
    k_temp_sup = st.number_input("k_temp_sup (>20°C, course cible)", min_value=0.90, max_value=1.1, step=0.00001, value=1.00200, format="%.5f")
    k_temp_inf = st.number_input("k_temp_inf (<20°C, course cible)", min_value=0.90, max_value=1.1, step=0.00001, value=0.99800, format="%.5f")

st.markdown("### 4️⃣ Paramètres de la course (cible à prédire)")
col1, col2, col3 = st.columns(3)
with col1:
    utiliser_distance_gpx = st.checkbox("Utiliser la distance du GPX", value=True)
    distance_officielle = st.number_input("Distance manuelle (m, si non GPX)", min_value=1000, step=1, value=42195)
with col2:
    objectif_temps = st.text_input("Objectif de temps (h:mm:ss, optionnel)", value="")
with col3:
    latitude = st.number_input("Latitude (course cible - météo)", value=48.8566)
    longitude = st.number_input("Longitude (course cible - météo)", value=2.3522)
    API_KEY = st.text_input("Clé API OpenWeather (optionnel)", type="password")
date_course = st.date_input("Date de la course cible")
heure_course = st.time_input("Heure de départ (course cible)")

# bouton
if st.button("Lancer l'analyse complète"):
    if gpx_file is None:
        st.error("⚠️ Upload d'abord un fichier GPX (parcours cible).")
        st.stop()

    try:
        # ------- Lecture GPX cible -------
        gpx, points = parse_gpx_points(gpx_file)
        st.write(f"📈 Points GPX : {len(points)}")
        cum_d, elevs, total_len_m, df_points = gpx_cumulative_distance_and_elev(points)
        st.write(f"Distance GPX réelle : {total_len_m:.1f} m")

        # Distance analysée
        distance_analyse = total_len_m if utiliser_distance_gpx else distance_officielle
        st.info(f"Distance utilisée pour calcul : {distance_analyse:.1f} m")

        # Segments (1 km + dernier partiel)
        km_marks = [i * 1000 for i in range(1, int(distance_analyse // 1000) + 1)]
        if distance_analyse % 1000 != 0:
            km_marks.append(distance_analyse)

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

        # ------- Régression log-log à partir des références -------
        if len(references) < 1:
            st.error("Ajoute au moins une référence (manuel ou FIT).")
            st.stop()

        temps_sec, dists_ref = [], []
        refs_weather_rows = []
        for idx, ref in enumerate(references, start=1):
            t = hms_to_seconds(ref["temps"])
            # ajustement dénivelé pour comparer des efforts "à plat"
            t_adj = t * (k_up ** ref["D_up"]) * (k_down ** ref["D_down"])
            temps_sec.append(t_adj)
            dists_ref.append(ref["distance"])

            # météo historique de la référence
            ref_dt = datetime.combine(ref["date"], ref["time"])
            met = fetch_historical_weather_openmeteo(ref["lat"], ref["lon"], ref_dt)
            refs_weather_rows.append({
                "Réf": idx,
                "Date/heure": ref_dt.strftime("%Y-%m-%d %H:%M"),
                "Lat": ref["lat"], "Lon": ref["lon"],
                "Distance (m)": ref["distance"],
                "Temps (h:mm:ss)": ref["temps"],
                "D+ (m)": ref["D_up"], "D- (m)": ref["D_down"],
                "Temp ref (°C)": None if not met else round(met["temp"],1) if met.get("temp") is not None else None,
                "Vent ref (m/s)": None if not met else round(met["wind"],1) if met.get("wind") is not None else None,
            })

        if len(temps_sec) == 1:
            # avec 1 réf, on extrapole par K=1.06 (valeur typique route) — mais on garde ton calcul si >1
            K_pred = 1.06
        else:
            # moyenne des rapports log pour toutes paires (n*(n-1)/2)
            pairs = []
            for i in range(len(temps_sec)):
                for j in range(i+1, len(temps_sec)):
                    if dists_ref[j] != dists_ref[i]:
                        pairs.append(math.log(temps_sec[j]/temps_sec[i]) / math.log(dists_ref[j]/dists_ref[i]))
            K_pred = sum(pairs)/len(pairs) if pairs else 1.06

        st.info(f"Exposant log-log estimé K = {K_pred:.4f}")

        if objectif_temps.strip() != "":
            base_total_time = hms_to_seconds(objectif_temps)
            st.success(f"Objectif de temps : {objectif_temps}")
        else:
            # on prend la dernière réf comme ancre par défaut
            base_total_time = temps_sec[-1]*(distance_analyse/dists_ref[-1])**K_pred

        base_s_per_km = base_total_time / (distance_analyse/1000)

        # ------- Météo course cible (prévision) -------
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

        # ------- Sorties -------
        # Météo des références
        st.subheader("🌤️ Météo du jour de chaque référence (temp & vent)")
        st.dataframe(pd.DataFrame(refs_weather_rows))

        # Kilomètres cumulés
        liste_km = [round(sum(seg["length_km"] for seg in per_km[:i+1]), 3) for i in range(len(per_km))]
        st.markdown(f"**Kilomètres cumulés (course cible) :** {' - '.join(map(str, liste_km))}")

        # Résultats
        total_sec = sum(r["Temps segment (s)"] for r in km_results)
        avg_pace = total_sec / (distance_analyse / 1000)
        st.subheader("⏱️ Résultats course cible")
        st.write(f"Temps total prévisionnel = {seconds_to_hms(total_sec)}")
        st.write(f"Allure moyenne = {int(avg_pace//60)}:{int(avg_pace%60):02d} min/km")

        # Carte du GPX cible
        st.subheader("🗺️ Carte interactive du parcours (GPX)")
        if not df_points.empty:
            mid_lat = float(df_points["lat"].mean())
            mid_lon = float(df_points["lon"].mean())
            path_coords = df_points[["lon","lat"]].to_numpy().tolist()
            deck = pdk.Deck(
                initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=12, pitch=45),
                map_style="mapbox://styles/mapbox/outdoors-v12",
                layers=[pdk.Layer("PathLayer", data=[{"path": path_coords, "name": "Parcours"}],
                                  get_path="path", get_width=4, width_min_pixels=2)],
                tooltip={"text": "{name}"}
            )
            st.pydeck_chart(deck)
        else:
            st.info("Pas de points pour afficher la carte.")

        # Profil & allure 2D
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

        # Profil 3D
        st.subheader("🧭 Profil 3D (lon / lat / altitude)")
        if not df_points.empty:
            df3d = df_points.iloc[::max(1, len(df_points)//3000)].copy()
            fig3d = go.Figure(data=[go.Scatter3d(x=df3d["lon"], y=df3d["lat"], z=df3d["elev"], mode="lines", line=dict(width=3))])
            fig3d.update_layout(scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Altitude (m)", aspectmode="data"),
                                margin=dict(l=0,r=0,t=0,b=0), height=500)
            st.plotly_chart(fig3d, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur durant l'analyse : {e}")
