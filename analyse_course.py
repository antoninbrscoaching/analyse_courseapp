import streamlit as st
import math, requests, gpxpy
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime, timedelta, date, time
from fitparse import FitFile
import pydeck as pdk
import plotly.graph_objects as go

# -----------------------------------------------------------
# ⚙️ CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Prédiction course - GPX + Références FIT", layout="wide")
st.title("🏃‍♂️ Analyse & Prédiction de course (GPX + Références FIT + Météo)")

# -----------------------------------------------------------
# 🧩 FONCTIONS UTILES
# -----------------------------------------------------------
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

def safe_parse_gpx(file):
    try:
        gpx = gpxpy.parse(file)
        pts = []
        for trk in gpx.tracks:
            for seg in trk.segments:
                for p in seg.points:
                    pts.append(p)
        if not pts:
            return None, []
        return gpx, pts
    except Exception:
        return None, []

def gpx_to_df(points):
    """Convertit la trace GPX en DataFrame + distances + altitudes"""
    if not points:
        return pd.DataFrame(), [], [], 0
    rows, cum_d, elevs = [], [0.0], [points[0].elevation or 0]
    total = 0.0
    for i in range(1, len(points)):
        d = points[i].distance_3d(points[i-1]) or 0
        total += d
        cum_d.append(total)
        elevs.append(points[i].elevation or elevs[-1])
        rows.append({
            "lat": points[i].latitude,
            "lon": points[i].longitude,
            "elev": points[i].elevation or elevs[-1]
        })
    df = pd.DataFrame(rows)
    return df, cum_d, elevs, total

def interp_elev(d_target, cum_d, elevs):
    if not cum_d or d_target <= 0:
        return elevs[0] if elevs else 0
    if d_target >= cum_d[-1]:
        return elevs[-1]
    for i in range(1, len(cum_d)):
        if cum_d[i] >= d_target:
            d0, d1 = cum_d[i-1], cum_d[i]
            e0, e1 = elevs[i-1], elevs[i]
            frac = (d_target - d0)/(d1-d0) if (d1-d0)>0 else 0
            return e0 + frac*(e1-e0)
    return elevs[-1]

def parse_fit(file):
    """Lit un fichier FIT et retourne distance, D+, D-, coords moyennes et date"""
    try:
        fit = FitFile(file)
        fit.parse()
        rows = []
        for msg in fit.get_messages("record"):
            data = {d.name: d.value for d in msg}
            if data.get("position_lat") is None or data.get("position_long") is None:
                continue
            lat = data["position_lat"] * (180 / 2**31)
            lon = data["position_long"] * (180 / 2**31)
            elev = data.get("altitude", 0.0)
            dist = data.get("distance", 0.0)
            t = data.get("timestamp")
            rows.append((lat, lon, elev, dist, t))
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["lat","lon","elev","dist","time"]).sort_values("time")
        dup = np.sum(np.diff(df["elev"]).clip(min=0))
        ddn = -np.sum(np.diff(df["elev"]).clip(max=0))
        d_tot = float(df["dist"].max() or 0)
        d_moy = (df["lat"].mean(), df["lon"].mean())
        t0 = pd.to_datetime(df["time"].iloc[0]).to_pydatetime()
        return dict(distance=round(d_tot), D_up=round(dup), D_down=round(ddn), lat=round(d_moy[0],5),
                    lon=round(d_moy[1],5), date=t0.date(), time=t0.time().replace(microsecond=0))
    except:
        return None

def get_weather(lat, lon, when):
    """Température & vent via Open-Meteo ERA5 (gratuit, sans clé)"""
    url = (
        f"https://archive-api.open-meteo.com/v1/era5?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={when.date()}&end_date={when.date()}"
        "&hourly=temperature_2m,wind_speed_10m&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return {}
        data = r.json()["hourly"]
        tseries = pd.to_datetime(data["time"])
        temps = data["temperature_2m"]
        winds = data["wind_speed_10m"]
        idx = int(np.argmin(np.abs(tseries - pd.to_datetime(when))))
        return {"temp": round(temps[idx],1), "wind": round(winds[idx],1)}
    except:
        return {}

# -----------------------------------------------------------
# 📍 1. PARCOURS GPX
# -----------------------------------------------------------
st.header("1️⃣ Parcours GPX")
gpx_file = st.file_uploader("➡️ Importer un fichier GPX", type=["gpx"])

# -----------------------------------------------------------
# 🏁 2. RÉFÉRENCES (manuel ou FIT)
# -----------------------------------------------------------
st.header("2️⃣ Références (temps connus, manuels ou FIT)")

if "n_refs" not in st.session_state:
    st.session_state.n_refs = 3

cols = st.columns([1,1,5])
with cols[0]:
    if st.button("➕ Ajouter réf (max 6)") and st.session_state.n_refs < 6:
        st.session_state.n_refs += 1
with cols[1]:
    if st.button("➖ Retirer") and st.session_state.n_refs > 1:
        st.session_state.n_refs -= 1

refs = []
for i in range(1, st.session_state.n_refs+1):
    st.markdown(f"#### Référence {i}")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: use_fit = st.checkbox(f"FIT ?", key=f"fitbox{i}")
    with c2: dist = st.number_input(f"Dist {i} (m)", value=5000*i, step=100)
    with c3: temps = st.text_input(f"Temps {i} (h:mm:ss)", value=f"0:{40+i*2}:00")
    with c4: dup = st.number_input(f"D+ {i}", value=0)
    with c5: ddn = st.number_input(f"D- {i}", value=0)
    fit_file = st.file_uploader(f"FIT {i}", type=["fit"], key=f"fit{i}") if use_fit else None
    date_i = st.date_input(f"Date {i}", value=date.today(), key=f"d{i}")
    time_i = st.time_input(f"Heure {i}", value=time(9,0), key=f"t{i}")
    lat_i = st.number_input(f"Lat {i}", value=48.85, key=f"lat{i}")
    lon_i = st.number_input(f"Lon {i}", value=2.35, key=f"lon{i}")
    if fit_file:
        parsed = parse_fit(fit_file)
        if parsed:
            dist, dup, ddn = parsed["distance"], parsed["D_up"], parsed["D_down"]
            date_i, time_i, lat_i, lon_i = parsed["date"], parsed["time"], parsed["lat"], parsed["lon"]
            st.info(f"✔ FIT détecté : {dist} m, D+{dup}, D-{ddn}, {date_i} {time_i}, {lat_i},{lon_i}")
    refs.append(dict(distance=dist, temps=temps, D_up=dup, D_down=ddn, date=date_i, time=time_i, lat=lat_i, lon=lon_i))

# -----------------------------------------------------------
# ⚙️ 3. PARAMÈTRES COURSE CIBLE
# -----------------------------------------------------------
st.header("3️⃣ Paramètres course à prédire")
c1, c2 = st.columns(2)
with c1:
    k_up = st.number_input("k_montée", value=1.001)
    k_down = st.number_input("k_descente", value=0.999)
with c2:
    k_temp_sup = st.number_input("k_temp_sup (>20°C)", value=1.002)
    k_temp_inf = st.number_input("k_temp_inf (<20°C)", value=0.998)

col_a, col_b, col_c = st.columns(3)
with col_a:
    use_gpx_dist = st.checkbox("Utiliser distance GPX", value=True)
    dist_manual = st.number_input("Distance manuelle (m)", value=42195)
with col_b:
    objectif = st.text_input("Objectif (h:mm:ss)", value="")
with col_c:
    lat_course = st.number_input("Lat course", value=48.85)
    lon_course = st.number_input("Lon course", value=2.35)
date_course = st.date_input("Date course")
heure_course = st.time_input("Heure départ")

# -----------------------------------------------------------
# 🚀 4. ANALYSE
# -----------------------------------------------------------
if st.button("▶️ Lancer l'analyse"):
    if not gpx_file:
        st.error("Importe d'abord un GPX.")
        st.stop()
    _, pts = safe_parse_gpx(gpx_file)
    if len(pts) < 2:
        st.error("GPX invalide ou trop court.")
        st.stop()

    df, cum_d, elevs, total_len = gpx_to_df(pts)
    dist_course = total_len if use_gpx_dist else dist_manual
    st.success(f"📏 Distance utilisée : {dist_course:.0f} m")

    # segments 1 km
    km_marks = [i*1000 for i in range(1, int(dist_course//1000)+1)]
    if dist_course % 1000:
        km_marks.append(dist_course)

    elev_at = [interp_elev(k, cum_d, elevs) for k in [0]+km_marks]
    per_km = []
    for i in range(1, len(elev_at)):
        dh = elev_at[i]-elev_at[i-1]
        per_km.append({"up": max(0,dh), "down": max(0,-dh), "len": (km_marks[i-1]-(km_marks[i-2] if i>1 else 0))/1000})

    # régression log-log
    temps_sec = []
    dists = []
    meteo_tab = []
    for i,r in enumerate(refs,1):
        t = hms_to_seconds(r["temps"])
        t_adj = t*(k_up**r["D_up"])*(k_down**r["D_down"])
        temps_sec.append(t_adj)
        dists.append(r["distance"])
        met = get_weather(r["lat"],r["lon"],datetime.combine(r["date"],r["time"]))
        meteo_tab.append({"Réf":i,"Dist(m)":r["distance"],"Temp":met.get("temp"),"Vent":met.get("wind")})

    pairs=[]
    for i in range(len(temps_sec)):
        for j in range(i+1,len(temps_sec)):
            try:
                pairs.append(math.log(temps_sec[j]/temps_sec[i]) / math.log(dists[j]/dists[i]))
            except: pass
    K = sum(pairs)/len(pairs) if pairs else 1.06
    st.info(f"Exposant log-log K = {K:.3f}")

    t_base = hms_to_seconds(objectif) if objectif else temps_sec[-1]*(dist_course/dists[-1])**K
    base_spkm = t_base / (dist_course/1000)

    # temps cumulé + météo
    meteo_future = []
    cum_time=0
    for seg in per_km:
        t = base_spkm*seg["len"]*(k_up**seg["up"])*(k_down**seg["down"])
        cum_time+=t
    t_tot=cum_time
    st.subheader("Résultat prévisionnel")
    st.success(f"⏱ Temps total : {seconds_to_hms(t_tot)}  |  Allure moyenne : {int((t_tot/(dist_course/1000))//60)}:{int((t_tot/(dist_course/1000))%60):02d}/km")

    st.subheader("🌦️ Météo des références")
    st.dataframe(pd.DataFrame(meteo_tab))

    # carte
    st.subheader("🗺️ Carte du parcours")
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/outdoors-v12",
        initial_view_state=pdk.ViewState(latitude=df["lat"].mean(), longitude=df["lon"].mean(), zoom=12, pitch=45),
        layers=[pdk.Layer("PathLayer", data=[{"path":df[["lon","lat"]].values.tolist()}], get_path="path", get_width=4)]
    )
    st.pydeck_chart(deck)

    # profil 3D
    st.subheader("🧭 Profil 3D")
    df3d=df.iloc[::max(1,len(df)//2000)]
    fig3d=go.Figure(data=[go.Scatter3d(x=df3d["lon"],y=df3d["lat"],z=df3d["elev"],mode="lines",line=dict(width=3))])
    fig3d.update_layout(scene=dict(xaxis_title="Lon",yaxis_title="Lat",zaxis_title="Alt(m)",aspectmode="data"))
    st.plotly_chart(fig3d,use_container_width=True)

