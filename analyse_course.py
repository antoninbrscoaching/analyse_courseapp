# analyse_course_refactor.py
import streamlit as st
import math
import gpxpy
from fitparse import FitFile
from datetime import datetime, timedelta, date, time
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from io import BytesIO
import requests

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Prédiction course route (refactor)", layout="wide")
st.title("🏃‍♂️ Analyse & Prédiction de course — Refactorisé")

# ============================================================
# MÉTÉO
# ============================================================

OW_API_KEY = st.secrets["openweather"]["api_key"]

@st.cache_data(show_spinner=False)
def get_weather_openmeteo_minutely(lat, lon, dt):
    """
    Météo future ultra-précise : interpolation à la minute.
    Basée sur Open-Meteo forecast (horaire).
    """
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,relativehumidity_2m,wind_speed_10m"
            "&timezone=UTC"
        )

        r = requests.get(url)
        data = r.json()

        if "hourly" not in data:
            return None

        times = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
        temps = data["hourly"]["temperature_2m"]
        winds = data["hourly"]["wind_speed_10m"]
        hums  = data["hourly"]["relativehumidity_2m"]

        before = None
        after = None

        for i in range(len(times) - 1):
            if times[i] <= dt <= times[i+1]:
                before = (times[i], temps[i], winds[i], hums[i])
                after  = (times[i+1], temps[i+1], winds[i+1], hums[i+1])
                break

        if before is None:
            idx = min(range(len(times)), key=lambda i: abs(times[i] - dt))
            return {
                "temp": temps[idx],
                "wind": winds[idx],
                "humidity": hums[idx],
            }

        t1, temp1, wind1, hum1 = before
        t2, temp2, wind2, hum2 = after

        ratio = (dt - t1).total_seconds() / (t2 - t1).total_seconds()
        temp_interp = temp1 + ratio * (temp2 - temp1)
        wind_interp = wind1 + ratio * (wind2 - wind1)
        hum_interp  = hum1  + ratio * (hum2  - hum1)

        return {
            "temp": float(temp_interp),
            "wind": float(wind_interp),
            "humidity": float(hum_interp),
        }

    except Exception as e:
        st.error(f"Erreur météo minute : {e}")
        return None


# -------------------------
# MÉTÉO HISTORIQUE - Open-Meteo (Références)
# -------------------------

@st.cache_data(show_spinner=False)
def get_weather_openmeteo_day(lat, lon, date_obj):
    date_str = date_obj.strftime("%Y-%m-%d")

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        "&hourly=temperature_2m,relativehumidity_2m,wind_speed_10m"
        "&timezone=UTC"
    )

    r = requests.get(url)
    data = r.json()

    if "hourly" not in data:
        return None

    times = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
    temps = data["hourly"]["temperature_2m"]
    winds = data["hourly"]["wind_speed_10m"]
    hums  = data["hourly"]["relativehumidity_2m"]

    return times, temps, winds, hums


def get_avg_weather_for_period(lat, lon, start_dt, end_dt):
    if start_dt is None or end_dt is None:
        return None, None, None

    if (end_dt - start_dt).total_seconds() < 300:
        start_dt -= timedelta(minutes=2)
        end_dt += timedelta(minutes=2)

    meteo_day = get_weather_openmeteo_day(lat, lon, start_dt.date())
    if not meteo_day:
        return None, None, None

    times, temps, winds, hums = meteo_day

    selT = [T for t,T in zip(times, temps) if start_dt <= t <= end_dt]
    selW = [W for t,W in zip(times, winds) if start_dt <= t <= end_dt]
    selH = [H for t,H in zip(times, hums)  if start_dt <= t <= end_dt]

    if not selT:
        closest_index = min(range(len(times)), key=lambda i: abs(times[i] - start_dt))
        return float(temps[closest_index]), float(winds[closest_index]), float(hums[closest_index])

    return float(np.mean(selT)), float(np.mean(selW)), float(np.mean(selH))


# -------------------------
# UTILITAIRES
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
            h = 0 ; m, s = parts
        elif len(parts) == 1:
            h = 0 ; m = 0 ; s = parts[0]
        else:
            return 0
        return max(0, h*3600 + m*60 + s)
    except:
        return 0

def seconds_to_hms(seconds: float) -> str:
    try:
        seconds = int(round(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    except:
        return "0:00:00"


# -------------------------
# AJOUT : extraction d’un segment temporel FIT/TCX (hh:mm:ss)
# -------------------------

def extract_segment_from_points(points, start_td, end_td):
    """
    points : dict (FIT) ou SimplePoint (TCX)
    start_td / end_td : timedelta
    """
    if not points or len(points) < 2:
        return points

    def get_time(p):
        return p["time"] if isinstance(p, dict) else p.time

    times = [get_time(p) for p in points if get_time(p)]
    if len(times) < 2:
        return points

    t0 = min(times)
    start_dt = t0 + start_td
    end_dt = t0 + end_td

    seg = [p for p in points if get_time(p) and start_dt <= get_time(p) <= end_dt]

    return seg if len(seg) >= 2 else points


# -------------------------
# SIMPLEPOINT + HAVERSINE
# -------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

class SimplePoint:
    def __init__(self, lat, lon, elev=0.0, time=None):
        self.latitude = float(lat)
        self.longitude = float(lon)
        self.elevation = float(elev) if elev is not None else 0.0
        self.time = time

    def distance_3d(self, other):
        horiz = haversine_m(self.latitude, self.longitude, other.latitude, other.longitude)
        vert = self.elevation - other.elevation
        return math.sqrt(horiz*horiz + vert*vert)

# -------------------------
# Parsers GPX / FIT / TCX
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
        st.error(f"Erreur parsing GPX : {e}")
        return None, []


def gpx_to_df(points):
    return pd.DataFrame([
        {
            "lat": p.latitude,
            "lon": p.longitude,
            "elev": p.elevation or 0,
            "time": getattr(p, "time", None)
        }
        for p in points
    ])


# -------------------------
# FIT PARSER (MODIFIÉ POUR RETOURNER TOUS LES POINTS)
# -------------------------

def parse_fit(file):
    try:
        file.seek(0)
        fit = FitFile(file)
        fit.parse()

        records = []
        times_points = []

        start_global = None
        elapsed_global = None

        # Méta
        for msg in fit.get_messages("session"):
            vals = {d.name: d.value for d in msg}
            if isinstance(vals.get("start_time"), datetime):
                start_global = vals["start_time"].replace(tzinfo=None)
            if isinstance(vals.get("total_elapsed_time"), (int, float)):
                elapsed_global = vals["total_elapsed_time"]

        # Points GPS (lat, lon, elev, dist, time)
        for msg in fit.get_messages("record"):
            vals = {d.name: d.value for d in msg}

            lat_raw = vals.get("position_lat")
            lon_raw = vals.get("position_long")
            ts = vals.get("timestamp")

            if lat_raw and lon_raw:
                lat = lat_raw * (180 / 2**31)
                lon = lon_raw * (180 / 2**31)
                elev = vals.get("altitude", 0)
                dist = vals.get("distance", 0)

                dt_local = None
                if isinstance(ts, datetime):
                    dt_local = ts.replace(tzinfo=None)
                elif isinstance(ts, (int, float)):
                    dt_local = datetime(1989, 12, 31) + timedelta(seconds=float(ts))

                records.append((lat, lon, elev, dist))
                times_points.append(dt_local)

        df = pd.DataFrame(records, columns=["lat","lon","elev","dist"])

        # Détermination start/end fiables
        valid_times = [t for t in times_points if t]

        if len(valid_times) >= 2:
            start_dt = min(valid_times)
            end_dt   = max(valid_times)
        else:
            start_dt = start_global
            if start_global and elapsed_global:
                end_dt = start_global + timedelta(seconds=elapsed_global)
            elif start_global:
                end_dt = start_global + timedelta(minutes=5)
            else:
                start_dt = datetime.now().replace(hour=12,minute=0,second=0,microsecond=0) - timedelta(days=1)
                end_dt = start_dt + timedelta(minutes=5)

        # Météo robuste
        avgT, avgW, avgH = get_avg_weather_for_period(records[0][0], records[0][1], start_dt, end_dt)

        # AJOUT : retourner tous les points utiles
        fit_points = []
        for (lat, lon, elev, dist), t in zip(records, times_points):
            fit_points.append({
                "lat": lat,
                "lon": lon,
                "elev": elev,
                "dist": dist,
                "time": t
            })

        return {
            "points": fit_points,
            "distance": float(df["dist"].max()),
            "D_up": float(np.sum(np.diff(df.elev).clip(min=0))),
            "D_down": float(-np.sum(np.diff(df.elev).clip(max=0))),
            "duration_hms": seconds_to_hms((end_dt - start_dt).total_seconds()),
            "avg_temp": avgT,
            "avg_wind": avgW,
            "avg_humidity": avgH
        }

    except Exception as e:
        st.error(f"Erreur FIT robuste : {e}")
        return None


# -------------------------
# TCX PARSER (DÉJÀ COMPATIBLE, RETOURNE pts)
# -------------------------

def parse_tcx(file):
    try:
        file.seek(0)
        tree = ET.parse(file)
        root = tree.getroot()
    except:
        return None

    ns = {"tcx":"http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    tps = root.findall(".//tcx:Trackpoint", ns)

    pts = []
    times = []
    elevs = []

    for tp in tps:
        lat = tp.find("tcx:Position/tcx:LatitudeDegrees", ns)
        lon = tp.find("tcx:Position/tcx:LongitudeDegrees", ns)
        tim = tp.find("tcx:Time", ns)
        ele = tp.find("tcx:AltitudeMeters", ns)

        if lat is None or lon is None:
            continue

        lat = float(lat.text)
        lon = float(lon.text)
        elev = float(ele.text) if ele is not None else 0.0

        try:
            t = datetime.fromisoformat(tim.text.replace("Z","+00:00")).replace(tzinfo=None)
        except:
            t = None

        p = SimplePoint(lat, lon, elev, t)
        pts.append(p)
        times.append(t)
        elevs.append(elev)

    if len(pts) < 2:
        return None

    valid_times = [t for t in times if t]

    if len(valid_times) >= 2:
        start_dt = valid_times[0]
        end_dt   = valid_times[-1]
    elif len(valid_times) == 1:
        start_dt = valid_times[0]
        end_dt = start_dt + timedelta(minutes=5)
    else:
        start_dt = datetime.now().replace(hour=12,minute=0,second=0,microsecond=0) - timedelta(days=1)
        end_dt   = start_dt + timedelta(minutes=5)

    avgT, avgW, avgH = get_avg_weather_for_period(pts[0].latitude, pts[0].longitude, start_dt, end_dt)

    total = sum(pts[i].distance_3d(pts[i-1]) for i in range(1,len(pts)))
    dup = float(np.sum(np.diff(np.array(elevs)).clip(min=0)))
    ddn = float(-np.sum(np.diff(np.array(elevs)).clip(max=0)))

    return {
        "points": pts,
        "distance": round(total),
        "D_up": round(dup),
        "D_down": round(ddn),
        "duration_hms": seconds_to_hms((end_dt - start_dt).total_seconds()),
        "avg_temp": avgT,
        "avg_wind": avgW,
        "avg_humidity": avgH
    }

# -------------------------
# UI : Entrées & Références
# -------------------------
st.header("2️⃣ Courses de référence (manuel ou FIT/TCX)")

if "n_refs" not in st.session_state:
    st.session_state.n_refs = 3

cols = st.columns([1, 1])
with cols[0]:
    if st.button("➕ Ajouter (max 6)") and st.session_state.n_refs < 6:
        st.session_state.n_refs += 1
with cols[1]:
    if st.button("➖ Retirer") and st.session_state.n_refs > 1:
        st.session_state.n_refs -= 1

refs_raw = []

# --------------------------------------------
# BOUCLE PRINCIPALE DES RÉFÉRENCES
# --------------------------------------------

for i in range(1, st.session_state.n_refs + 1):

    st.markdown(f"#### Référence {i}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    # Import fichier ?
    with c1:
        use_file = st.checkbox(f"Importer fichier (FIT/TCX) ?", key=f"use_file_{i}")

    # valeurs par défaut mémoire
    default_dist = st.session_state.get(f"dist_{i}", 5000 * i)
    default_temps = st.session_state.get(f"temps_{i}", "0:40:00")
    default_dup = st.session_state.get(f"dup_{i}", 0.0)
    default_ddn = st.session_state.get(f"ddn_{i}", 0.0)

    # Entrées manuelles
    with c2:
        dist = st.number_input(f"Dist {i} (m)", value=float(default_dist), key=f"dist_{i}")
    with c3:
        temps = st.text_input(f"Temps {i} (h:mm:ss)", value=str(default_temps), key=f"temps_{i}")
    with c4:
        dup = st.number_input(f"D+ {i}", value=float(default_dup), key=f"dup_{i}")
    with c5:
        ddn = st.number_input(f"D- {i}", value=float(default_ddn), key=f"ddn_{i}")

    # Upload FIT / TCX
    with c6:
        file_in = st.file_uploader(
            f"FIT/TCX {i}", 
            type=["fit", "tcx"], 
            key=f"fileref_{i}"
        ) if use_file else None

    # --------------------------------------------
    # INTERVALLE TEMPOREL (UNE SEULE LIGNE)
    # --------------------------------------------
    col_a, col_b = st.columns([1, 1])

    start_hms = col_a.text_input(
        f"Début réf {i} (hh:mm:ss)",
        value="00:00:00",
        key=f"start_{i}"
    )

    end_hms = col_b.text_input(
        f"Fin réf {i} (hh:mm:ss)",
        value="23:59:59",
        key=f"end_{i}"
    )

    def hms_to_td(hms):
        try:
            h, m, s = map(int, hms.split(":"))
            return timedelta(hours=h, minutes=m, seconds=s)
        except:
            return timedelta(seconds=0)

    start_td = hms_to_td(start_hms)
    end_td = hms_to_td(end_hms)

    # --------------------------------------------
    # TRAITEMENT FIT / TCX (+ éventuel découpage)
    # --------------------------------------------
    duration_hms_file = None
    avg_temp_ref = None
    avg_wind_ref = None
    avg_hum_ref = None

    if file_in:
        filename = file_in.name.lower()

        # FIT
        if filename.endswith(".fit"):
            fit_data = parse_fit(file_in)
            if fit_data:
                dist = fit_data["distance"]
                dup = fit_data["D_up"]
                ddn = fit_data["D_down"]
                duration_hms_file = fit_data["duration_hms"]

                avg_temp_ref = fit_data["avg_temp"]
                avg_wind_ref = fit_data["avg_wind"]
                avg_hum_ref  = fit_data["avg_humidity"]

                pts = fit_data["points"]

        # TCX
        elif filename.endswith(".tcx"):
            tcx_data = parse_tcx(file_in)
            if tcx_data:
                dist = tcx_data["distance"]
                dup = tcx_data["D_up"]
                ddn = tcx_data["D_down"]
                duration_hms_file = tcx_data["duration_hms"]

                avg_temp_ref = tcx_data["avg_temp"]
                avg_wind_ref = tcx_data["avg_wind"]
                avg_hum_ref  = tcx_data["avg_humidity"]

                pts = tcx_data["points"]

        # -------------------------------------------------------
        # SI INTERVALLE ≠ (0 → fin séance) → découpage FIT/TCX
        # -------------------------------------------------------
        if (start_td.total_seconds() > 0 or end_td.total_seconds() < 86399) and file_in:

            seg = extract_segment_from_points(pts, start_td, end_td)

            # recalcul distance + dénivelé + durée
            new_dist = 0
            elevs = []
            times = []

            for j in range(1, len(seg)):
                p1 = seg[j-1]
                p2 = seg[j]

                lat1 = p1["lat"] if isinstance(p1, dict) else p1.latitude
                lon1 = p1["lon"] if isinstance(p1, dict) else p1.longitude
                lat2 = p2["lat"] if isinstance(p2, dict) else p2.latitude
                lon2 = p2["lon"] if isinstance(p2, dict) else p2.longitude

                elev2 = p2["elev"] if isinstance(p2, dict) else p2.elevation
                t2    = p2["time"] if isinstance(p2, dict) else p2.time

                new_dist += haversine_m(lat1, lon1, lat2, lon2)
                elevs.append(elev2)
                if t2:
                    times.append(t2)

            dist = round(new_dist)

            if len(elevs) >= 2:
                dup = float(np.sum(np.diff(np.array(elevs)).clip(min=0)))
                ddn = float(-np.sum(np.diff(np.array(elevs)).clip(max=0)))

            if len(times) >= 2:
                duration_hms_file = seconds_to_hms(
                    (times[-1] - times[0]).total_seconds()
                )

    # temps final utilisé
    temps_effectif = duration_hms_file if duration_hms_file else temps

    # --------------------------------------------
    # REMPLISSAGE refs_raw
    # --------------------------------------------
    refs_raw.append({
        "distance": float(dist),
        "temps": str(temps_effectif),
        "D_up": float(dup),
        "D_down": float(ddn),
        "duration_hms_file": duration_hms_file,

        # Météo éventuelle
        "avg_temp": avg_temp_ref,
        "avg_wind": avg_wind_ref,
        "avg_humidity": avg_hum_ref,

        # Nouveaux champs intervalle hh:mm:ss
        "start_td": start_td,
        "end_td": end_td,
        "start_hms": start_hms,
        "end_hms": end_hms,
    })

# --------------------------------------------
# RÉCAP
# --------------------------------------------
st.subheader("⏱️ Récap références (raw)")

for idx, r in enumerate(refs_raw, start=1):
    st.write(
        f"Réf {idx} — Dist: {r['distance']:.0f} m | Brut: {r['temps']} | "
        f"D+ {r['D_up']:.0f} m / D- {r['D_down']:.0f} m | "
        f"Dur file: {r.get('duration_hms_file')} | "
        f"T° moy: {r.get('avg_temp')}°C | "
        f"Intervalle : {r['start_hms']} → {r['end_hms']}"
    )

# -------------------------
# Paramètres modèle UI
# -------------------------
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
        k_temp_hot = st.number_input("Sensibilité chaude (k_temp_hot)", value=0.002, format="%.4f", step=0.0005)
        k_temp_cold = st.number_input("Sensibilité froide (k_temp_cold)", value=0.002, format="%.4f", step=0.0005)
        opt_temp = st.number_input("Température optimale (°C)", value=12.0, format="%.1f", step=0.5)
    else:
        k_temp_hot = 0.0
        k_temp_cold = 0.0
        opt_temp = 12.0

col1, col2 = st.columns(2)
with col1:
    date_course = st.date_input("Date de la course (Jour J)", value=date.today())

with col2:
    heure_course = st.time_input("Heure de départ (Jour J)", value=time(9, 0))

st.info("La prédiction utilise OpenWeather pour chaque segment de course (température, vent, humidité).")

# -------------------------
# Recalibrage des références (brut / idéal)
# -------------------------
st.subheader("⏱️ Références recalibrées (plat 0% & T° optimale)")

refs_calibrated = []
for r in refs_raw:

    t_brut = hms_to_seconds(r["temps"])

    t_ideal = recalibrate_ref_to_ideal(
        ref=r,
        k_up=k_up,
        k_down=k_down,
        k_temp_hot=k_temp_hot,
        k_temp_cold=k_temp_cold,
        opt_temp=opt_temp
    )

    refs_calibrated.append({
        "distance": r["distance"],
        "D_up": r["D_up"],
        "D_down": r["D_down"],
        "temps_brut": t_brut,
        "temps_ideal": t_ideal,
        "origine": r.get("duration_hms_file", None),
        "temp_moy": r.get("avg_temp"),
        "vent_moy": r.get("avg_wind"),
        "hum_moy": r.get("avg_humidity"),
    })

df_refs = pd.DataFrame([{
    "Distance (m)": r["distance"],
    "D+ (m)": r["D_up"],
    "D- (m)": r["D_down"],
    "Temps brut": seconds_to_hms(r["temps_brut"]),
    "Temps conditions idéales": seconds_to_hms(r["temps_ideal"]),
    "Temp moy (°C)": r["temp_moy"],
    "Vent moy (m/s)": r["vent_moy"],
    "Hum moy (%)": r["hum_moy"],
} for r in refs_calibrated])

st.dataframe(df_refs, use_container_width=True)

# -------------------------
# FATIGUE
# -------------------------
st.header("3️⃣ bis. Fatigue linéaire")

fatigue_active = st.checkbox("Activer fatigue ?", value=False)
fatigue_rate = 0.0

if fatigue_active:
    fatigue_rate = st.slider("Régression finale (%)", 0.0, 30.0, 5.0, 0.5)

# -------------------------
# Option : utiliser conditions idéales pour le fit
# -------------------------
st.markdown("---")

ideal_refs = st.checkbox(
    "🔧 Utiliser les références recalibrées en CONDITIONS IDÉALES pour le fit (plat 0% & T° opt.) ?",
    value=True
)

# -------------------------
# Calculs : Base & Forcé
# -------------------------
st.header("4️⃣ Calcul & Comparaison")

if st.button("▶️ Calculer prédiction (BASE, d'après références)"):

    if not gpx_file or points is None:
        st.error("Importe un fichier GPX d'abord.")
    else:
        try:
            res_base = run_prediction_df(
                distance_cible_km=None,
                refs_input=refs_raw,
                points=points,
                date_course_local=date_course,
                heure_course_local=heure_course,
                ideal_refs=ideal_refs,
                apply_elev=use_elev_coeff,
                apply_temp=use_temp_coeff,
                apply_fatigue=fatigue_active,
                objective_time_hms=None,
                k_up=k_up,
                k_down=k_down,
                k_temp_hot=k_temp_hot,
                k_temp_cold=k_temp_cold,
                opt_temp=opt_temp,
                fatigue_rate=fatigue_rate
            )
            st.session_state["res_base"] = res_base
            st.success(
                f"Base calculée — distance GPX détectée: {res_base['distance_gpx_km']:.3f} km"
            )
        except Exception as e:
            st.error(f"Erreur lors du calcul base : {e}")

# -----------------------------------------------------
# FORCÉ
# -----------------------------------------------------
st.markdown("---")
st.markdown("**Forcer distance et/ou temps objectif (produit un tableau 'FORCÉ' distinct)**")

colf1, colf2 = st.columns(2)

with colf1:
    force_distance_checkbox = st.checkbox("Forcer la distance pour la prédiction finale ?", value=False)
    if "dist_forced" not in st.session_state:
        st.session_state["dist_forced"] = 5.17

    distance_forced_km = (
        st.number_input("Distance forcée (km)",
        value=st.session_state["dist_forced"],
        format="%.2f",
        key="dist_forced")
        if force_distance_checkbox else None
    )

with colf2:
    force_time_checkbox = st.checkbox("Forcer un temps objectif ?", value=False)
    if "time_forced" not in st.session_state:
        st.session_state["time_forced"] = "0:18:30"

    time_forced_hms = (
        st.text_input("Temps objectif (h:mm:ss)",
        value=st.session_state["time_forced"],
        key="time_forced")
        if force_time_checkbox else None
    )

if st.button("📊 Calculer prédiction finale (FORCÉ)"):

    if not gpx_file or points is None:
        st.error("Importe un fichier GPX d'abord.")
    else:
        dist_target = distance_forced_km if (force_distance_checkbox and distance_forced_km) else None

        try:
            res_forced = run_prediction_df(
                distance_cible_km=dist_target,
                refs_input=refs_raw,
                points=points,
                date_course_local=date_course,
                heure_course_local=heure_course,
                ideal_refs=ideal_refs,
                apply_elev=use_elev_coeff,
                apply_temp=use_temp_coeff,
                apply_fatigue=fatigue_active,
                objective_time_hms=(time_forced_hms if force_time_checkbox else None),
                k_up=k_up,
                k_down=k_down,
                k_temp_hot=k_temp_hot,
                k_temp_cold=k_temp_cold,
                opt_temp=opt_temp,
                fatigue_rate=fatigue_rate
            )

            st.session_state["res_forced"] = res_forced

            st.success(
                f"Prédiction forcée calculée — cible: {distance_forced_km if distance_forced_km else 'GPX'} km"
            )

        except Exception as e:
            st.error(f"Erreur lors du calcul forcé : {e}")


# -------------------------
# Affichage BASE / FORCÉ
# -------------------------

if "res_base" in st.session_state or "res_forced" in st.session_state:

    base = st.session_state.get("res_base", None)
    forced = st.session_state.get("res_forced", None)

    left, right = st.columns(2)

    # BASE
    with left:
        st.subheader("📈 Base (d'après références)")

        if base:
            avg_pace_base = base["total_seconds"] / max(base["distance_gpx_km"], 1e-6)

            st.write(f"Distance GPX détectée: {base['distance_gpx_km']:.3f} km")
            st.write(
                f"Temps total (base): {base['total_human']} "
                f"({pace_seconds_to_str_per_km(avg_pace_base)} / km)"
            )
            st.dataframe(base["df"], use_container_width=True)

        else:
            st.info("Clique sur 'Calculer prédiction (BASE)' pour générer ce tableau.")

    # FORCÉ
    with right:
        st.subheader("🎯 Forcé (distance/temps forcés)")

        if forced:
            dist_display = (
                distance_forced_km if (force_distance_checkbox and distance_forced_km)
                else round(forced['distance_gpx_km'], 3)
            )

            avg_pace_forced = forced["total_seconds"] / max(float(dist_display), 1e-6)

            st.write(f"Distance cible: {dist_display} km")
            st.write(
                f"Temps total (forcé): {forced['total_human']} "
                f"({pace_seconds_to_str_per_km(avg_pace_forced)} / km)"
            )

            st.dataframe(forced["df"], use_container_width=True)

        else:
            st.info("Clique sur 'Calculer prédiction finale (FORCÉ)' pour générer ce tableau.")


# -------------------------
# CARTE & PROFIL (GPX)
# -------------------------

if gpx_file and points:

    try:
        df_points = gpx_to_df(points)

        st.subheader("🗺️ Carte & Profil (GPX importé)")

        # -------------------
        # CARTE
        # -------------------
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
            tooltip={"text": "{name}"}
        )

        st.pydeck_chart(deck, use_container_width=True)

        # -------------------
        # PROFIL D'ALTITUDE
        # -------------------
        st.subheader("📊 Profil d'altitude")

        plt.figure(figsize=(10, 4))

        total_m = 0.0
        cumd = [0.0]

        for i in range(1, len(points)):
            d = SimplePoint(
                points[i-1].latitude,
                points[i-1].longitude,
                points[i-1].elevation
            ).distance_3d(
                SimplePoint(
                    points[i].latitude,
                    points[i].longitude,
                    points[i].elevation
                )
            )
            total_m += d
            cumd.append(total_m)

        x_km = np.array(cumd) / 1000.0
        y_elev = np.array([p.elevation for p in points])

        plt.plot(x_km, y_elev, lw=2)
        plt.xlabel("Distance (km)")
        plt.ylabel("Altitude (m)")
        plt.title("Profil d'altitude du parcours")
        plt.grid(alpha=0.3)

        st.pyplot(plt)

    except Exception as e:
        st.error(f"Impossible d'afficher la carte/profil : {e}")
