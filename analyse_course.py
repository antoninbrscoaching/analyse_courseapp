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
#   - OpenWeather : prédiction jour J (segment par segment)
#   - Open-Meteo  : historique pour les références FIT/TCX
# ============================================================

# -------------------------
# MÉTÉO - OpenWeather (Jour J)
# -------------------------

OW_API_KEY = st.secrets["openweather"]["api_key"]

@st.cache_data(show_spinner=False)
def get_weather_openweather(lat, lon, dt):
    """
    - Si dt est dans le passé → utilise timemachine
    - Si dt est dans le futur → utilise forecast (prévision)
    """

    timestamp = int(dt.timestamp())

    # -------------------
    # 1. CAS FUTUR : FORECAST
    # -------------------
    if dt > datetime.utcnow():
        url = (
            "https://api.openweathermap.org/data/2.5/forecast"
            f"?lat={lat}&lon={lon}&appid={OW_API_KEY}&units=metric"
        )
        r = requests.get(url)
        data = r.json()

        if "list" not in data:
            return None

        # Chercher la prévision la plus proche
        closest = min(
            data["list"],
            key=lambda x: abs(datetime.fromtimestamp(x["dt"]) - dt)
        )

        return {
            "temp": closest["main"].get("temp"),
            "wind": closest["wind"].get("speed"),
            "humidity": closest["main"].get("humidity"),
        }

    # -------------------
    # 2. CAS PASSÉ : TIMEMACHINE
    # -------------------
    url = (
        "https://api.openweathermap.org/data/3.0/onecall/timemachine"
        f"?lat={lat}&lon={lon}&dt={timestamp}"
        f"&appid={OW_API_KEY}&units=metric"
    )

    r = requests.get(url)
    data = r.json()

    if "data" not in data or not data["data"]:
        return None

    entry = data["data"][0]

    return {
        "temp": entry.get("temp"),
        "wind": entry.get("wind_speed"),
        "humidity": entry.get("humidity"),
    }

    except Exception as e:
        st.error(f"Erreur météo OpenWeather : {e}")
        return None

# -------------------------
# MÉTÉO HISTORIQUE - Open-Meteo (Références)
# -------------------------

@st.cache_data(show_spinner=False)
def get_weather_openmeteo_day(lat, lon, date_obj):
    """
    Récupère TOUTE la journée météo (24 valeurs horaires) en un seul appel.
    Retourne :
        - times : liste datetime
        - temps : liste température
        - winds : liste vent
        - hums  : liste humidité
    """
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
    """
    Récupère une météo même si start/end sont courts ou en dehors exact des heures.
    Étend automatiquement la fenêtre et choisit la valeur la plus proche.
    """
    if start_dt is None or end_dt is None:
        return None, None, None

    # 1) ELARGISSEMENT de la fenêtre si trop petite (< 5 minutes)
    if (end_dt - start_dt).total_seconds() < 300:
        start_dt -= timedelta(minutes=2)
        end_dt += timedelta(minutes=2)

    meteo_day = get_weather_openmeteo_day(lat, lon, start_dt.date())
    if not meteo_day:
        return None, None, None

    times, temps, winds, hums = meteo_day

    # 2) Sélection stricte dans l'intervalle
    selT = [T for t,T in zip(times, temps) if start_dt <= t <= end_dt]
    selW = [W for t,W in zip(times, winds) if start_dt <= t <= end_dt]
    selH = [H for t,H in zip(times, hums)  if start_dt <= t <= end_dt]

    # 3) Si rien trouvé, prendre la valeur la plus proche dans l’heure
    if not selT:
        # prendre l'heure la plus proche de start_dt
        closest_index = min(range(len(times)), key=lambda i: abs(times[i] - start_dt))
        return float(temps[closest_index]), float(winds[closest_index]), float(hums[closest_index])

    # 4) Sinon on retourne la moyenne strictes
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
            h = 0
            m, s = parts
        elif len(parts) == 1:
            h = 0
            m = 0
            s = parts[0]
        else:
            return 0
        return max(0, h * 3600 + m * 60 + s)
    except Exception:
        return 0

def seconds_to_hms(seconds: float) -> str:
    try:
        seconds = int(round(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    except Exception:
        return "0:00:00"

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

# -------------------------
# Modèle & facteurs
# -------------------------
def temp_multiplier_nonlin(temp, opt_temp=12.0, k_hot=0.002, k_cold=0.002):
    """
    Multiplicateur simple : >opt -> 1 + k_hot*(temp-opt)
    <opt -> 1 + k_cold*(opt-temp)
    On retourne >= 0.1 (sécurité).
    """
    try:
        if temp is None:
            return 1.0
        diff = float(temp) - float(opt_temp)
        if diff > 0:
            mult = 1.0 + float(k_hot) * diff
        else:
            mult = 1.0 + float(k_cold) * (-diff)
        return max(0.1, mult)
    except Exception:
        return 1.0

def apply_elevation_gradient_route(t_flat, d_up, d_down, segment_length_m=1000.0, k_up=1.04, k_down=0.996):
    try:
        seg_len = float(segment_length_m) if segment_length_m and segment_length_m > 0 else 1000.0
        up_factor = (float(k_up) - 1.0) * (float(d_up) / seg_len)
        down_factor = (1.0 - float(k_down)) * (float(d_down) / seg_len)
        factor = 1.0 + up_factor + down_factor
        return float(t_flat) * max(0.01, factor)
    except Exception:
        return float(t_flat)

def fit_loglog_model(refs):
    """
    Fit log-log: secs = a * (distance_km ** K)
    refs: list of dicts with 'distance' (m) and 'temps' (secs or h:mm:ss)
    Retourne a, K
    """
    X = []
    Y = []
    for r in refs:
        d_m = r.get("distance", None)
        t_raw = r.get("temps")
        if d_m is None or d_m <= 0:
            continue
        if isinstance(t_raw, (int, float, np.number)):
            secs = float(t_raw)
        else:
            secs = hms_to_seconds(str(t_raw))
        if secs <= 0:
            continue
        d_km = float(d_m) / 1000.0
        X.append(math.log(max(1e-6, d_km)))
        Y.append(math.log(max(1e-6, secs)))
    if len(X) >= 2:
        coeffs = np.polyfit(X, Y, 1)  # Y = K * X + log(a)
        K = float(coeffs[0])
        loga = float(coeffs[1])
        a = math.exp(loga)
        if not (0 < a < 1e6):  # sanity
            a = 240.0
        if abs(K) > 5:
            K = 1.0
        return a, K
    elif len(X) == 1:
        d_km = math.exp(X[0])
        secs = math.exp(Y[0])
        a = secs / max(1e-6, d_km)
        return a, 1.0
    else:
        return 240.0, 1.0

def predict_time_flat(distance_m, a, K):
    d_km = float(distance_m) / 1000.0
    return float(a) * (d_km ** float(K))

def override_with_objective(distance_m, objective_time_hms, K):
    objective_seconds = hms_to_seconds(objective_time_hms)
    d_km = float(distance_m) / 1000.0
    if d_km <= 0:
        return None
    a = float(objective_seconds) / (d_km ** float(K))
    return a

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
        {"lat": p.latitude, "lon": p.longitude, "elev": p.elevation or 0, "time": getattr(p, "time", None)}
        for p in points
    ])

def parse_fit(file):
    try:
        file.seek(0)
        fit = FitFile(file)
        fit.parse()

        records = []
        times_points = []

        start_global = None
        elapsed_global = None

        # Extraction metadata
        for msg in fit.get_messages("session"):
            vals = {d.name: d.value for d in msg}
            if isinstance(vals.get("start_time"), datetime):
                start_global = vals["start_time"].replace(tzinfo=None)
            if isinstance(vals.get("total_elapsed_time"), (int,float)):
                elapsed_global = vals["total_elapsed_time"]

        # Extraction record (lat/lon + timestamps)
        for msg in fit.get_messages("record"):
            vals = {d.name: d.value for d in msg}
            lat_raw = vals.get("position_lat")
            lon_raw = vals.get("position_long")
            ts = vals.get("timestamp")

            if lat_raw and lon_raw:
                lat = lat_raw * (180 / 2**31)
                lon = lon_raw * (180 / 2**31)

                # Convert timestamp FIT
                dt_local = None
                if isinstance(ts, datetime):
                    dt_local = ts.replace(tzinfo=None)
                elif isinstance(ts, (int,float)):
                    dt_local = datetime(1989,12,31) + timedelta(seconds=float(ts))

                records.append((lat, lon, vals.get("altitude",0), vals.get("distance",0)))
                times_points.append(dt_local)

        df = pd.DataFrame(records, columns=["lat","lon","elev","dist"])

        # Détermination start/end
        valid_times = [t for t in times_points if t]

        if len(valid_times)>=2:
            start_dt = min(valid_times)
            end_dt = max(valid_times)

        else:
            start_dt = start_global
            if start_global and elapsed_global:
                end_dt = start_global + timedelta(seconds=elapsed_global)
            elif start_global:
                end_dt = start_global + timedelta(minutes=5)
            else:
                # Fallback ultime : on invente une fenêtre sûre
                start_dt = datetime.now().replace(hour=12,minute=0,second=0,microsecond=0) - timedelta(days=1)
                end_dt = start_dt + timedelta(minutes=5)

        # Météo robuste
        avgT, avgW, avgH = get_avg_weather_for_period(records[0][0], records[0][1], start_dt, end_dt)

        return {
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

        pts.append(SimplePoint(lat,lon,elev,t))
        times.append(t)
        elevs.append(elev)

    if len(pts)<2:
        return None

    valid_times = [t for t in times if t]

    if len(valid_times)>=2:
        start_dt = valid_times[0]
        end_dt = valid_times[-1]
    elif len(valid_times)==1:
        start_dt = valid_times[0]
        end_dt = start_dt + timedelta(minutes=5)
    else:
        start_dt = datetime.now().replace(hour=12,minute=0,second=0,microsecond=0) - timedelta(days=1)
        end_dt = start_dt + timedelta(minutes=5)

    avgT, avgW, avgH = get_avg_weather_for_period(pts[0].latitude, pts[0].longitude, start_dt, end_dt)

    total = sum(pts[i].distance_3d(pts[i-1]) for i in range(1,len(pts)))
    dup = float(np.sum(np.diff(np.array(elevs)).clip(min=0)))
    ddn = float(-np.sum(np.diff(np.array(elevs)).clip(max=0)))

    return {
        "points": pts,
        "distance": round(total),
        "D_up": round(dup),
        "D_down": round(ddn),
        "duration_hms": seconds_to_hms((end_dt-start_dt).total_seconds()),
        "avg_temp": avgT,
        "avg_wind": avgW,
        "avg_humidity": avgH
    }

# -------------------------
# Helpers safe
# -------------------------
def safe_float(val, default=0.0):
    try:
        if val is None:
            return float(default)
        if isinstance(val, str):
            s = val.strip()
            if s == "" or s.lower() in ("nan", "none"):
                return float(default)
            return float(s.replace(",", "."))
        if isinstance(val, (float, int, np.number)):
            if np.isnan(val) or np.isinf(val):
                return float(default)
            return float(val)
        return float(val)
    except Exception:
        return float(default)

def clean_time_input(v):
    if v is None:
        return "0:00:00"
    if isinstance(v, (int, float, np.number)):
        return seconds_to_hms(float(v))
    s = str(v).strip()
    if s == "" or s.lower() in ("none", "nan"):
        return "0:00:00"
    return s

# -------------------------
# Recalibration : applique correction élévation & température
# -------------------------
def recalibrate_ref_to_ideal(ref, k_up, k_down, k_temp_hot, k_temp_cold, opt_temp):
    """
    Recalibre une référence dans des conditions idéales :
    - plat (0 %)
    - température optimale choisie par l’utilisateur (opt_temp)
    """

    # --- 1) Temps brut
    secs = hms_to_seconds(ref.get("temps")) if ref.get("temps") is not None else 0

    # --- 2) Retirer effet dénivelé
    D_up = safe_float(ref.get("D_up", 0.0))
    D_down = safe_float(ref.get("D_down", 0.0))
    seg_len = safe_float(ref.get("distance", 1000.0))
    seg_len = seg_len if seg_len > 0 else 1000.0

    up_factor = (k_up - 1.0) * (D_up / seg_len)
    down_factor = (1.0 - k_down) * (D_down / seg_len)
    factor_elev = 1.0 + up_factor + down_factor
    if factor_elev == 0:
        factor_elev = 1.0

    secs_no_elev = secs / factor_elev

    # --- 3) Retirer effet température réelle (si connue)
    temp_real = ref.get("avg_temp")
    if temp_real is not None:
        mult_real = temp_multiplier_nonlin(
            temp_real,
            opt_temp=opt_temp,
            k_hot=k_temp_hot,
            k_cold=k_temp_cold
        )
        if mult_real != 0:
            secs_no_temp = secs_no_elev / mult_real
        else:
            secs_no_temp = secs_no_elev
    else:
        secs_no_temp = secs_no_elev

    # --- 4) Appliquer l'effet température optimale définie par l’utilisateur
    mult_opt = temp_multiplier_nonlin(
        opt_temp,
        opt_temp=opt_temp,
        k_hot=k_temp_hot,
        k_cold=k_temp_cold
    )
    secs_ideal = secs_no_temp * mult_opt

    return max(0.0, secs_ideal)

def recalibrate_ref_using_current(ref, k_up, k_down, k_temp_hot, k_temp_cold, opt_temp, assumed_temp=None):
    """
    Recalibre la référence en retirant l'effet élévation, et éventuellement
    l'effet température si assumed_temp est connu.
    """
    secs = hms_to_seconds(ref.get("temps")) if ref.get("temps") is not None else 0
    D_up = safe_float(ref.get("D_up", 0.0))
    D_down = safe_float(ref.get("D_down", 0.0))
    seg_len = safe_float(ref.get("distance", 1000.0))
    seg_len = seg_len if seg_len > 0 else 1000.0

    up_factor = (k_up - 1.0) * (D_up / seg_len)
    down_factor = (1.0 - k_down) * (D_down / seg_len)
    factor_elev = 1.0 + up_factor + down_factor
    if factor_elev == 0:
        factor_elev = 1.0

    secs_no_elev = secs / factor_elev
    if assumed_temp is None:
        return max(0.0, secs_no_elev)
    else:
        mult_temp = temp_multiplier_nonlin(
            assumed_temp,
            opt_temp=opt_temp,
            k_hot=k_temp_hot,
            k_cold=k_temp_cold
        )
        if mult_temp == 0:
            mult_temp = 1.0
        return max(0.0, secs_no_elev / mult_temp)

# -------------------------
# Prépare les références AVANT fit
# -------------------------
def prepare_refs_for_fit(refs_input, k_up, k_down, k_temp_hot, k_temp_cold, opt_temp, ideal_refs=False):
    """
    refs_input: liste dicts {distance, temps, D_up, D_down, duration_hms_file...}
    Si ideal_refs True => on normalise vers 0% (retrait élévation, et correction temp via opt_temp).
    Sinon => retrait élévation, on garde l'effet température implicite.
    """
    prepared = []
    for r in refs_input:
        d = safe_float(r.get("distance", 0.0))
        file_dur = r.get("duration_hms_file")
        raw_t = file_dur if file_dur else r.get("temps", "0:00:00")

        ref_for_calib = {
            "distance": d,
            "temps": raw_t,
            "D_up": r.get("D_up", 0.0),
            "D_down": r.get("D_down", 0.0),
            "avg_temp": r.get("avg_temp")
        }

        if ideal_refs:
            secs_recal = recalibrate_ref_to_ideal(
                ref_for_calib,
                k_up, k_down,
                k_temp_hot, k_temp_cold,
                opt_temp
            )
        else:
            secs_recal = recalibrate_ref_using_current(
                ref_for_calib,
                k_up, k_down,
                k_temp_hot, k_temp_cold,
                opt_temp,
                assumed_temp=None
            )

        prepared.append({
            "distance": float(d),
            "temps": float(secs_recal)
        })
    return prepared

# -------------------------
# Calcul principal de prédiction (avec météo segmentaire)
# -------------------------
def run_prediction_df(
    distance_cible_km,
    refs_input,
    points,
    date_course_local,
    heure_course_local,
    ideal_refs=False,
    apply_elev=True,
    apply_temp=True,
    apply_fatigue=True,
    objective_time_hms=None,
    k_up=1.040, k_down=0.996,
    k_temp_hot=0.002, k_temp_cold=0.002, opt_temp=12.0,
    fatigue_rate=0.0
):
    if not points or len(points) < 2:
        raise ValueError("GPX invalide ou trop court.")

    # Distances cumulées
    total_m = 0.0
    cum = [0.0]
    for i in range(1, len(points)):
        d = SimplePoint(
            points[i-1].latitude,
            points[i-1].longitude,
            getattr(points[i-1], "elevation", 0)
        ).distance_3d(
            SimplePoint(
                points[i].latitude,
                points[i].longitude,
                getattr(points[i], "elevation", 0)
            )
        )
        total_m += d
        cum.append(total_m)
    distance_gpx_km = total_m / 1000.0

    if distance_cible_km is None or distance_cible_km <= 0:
        distance_cible_km = distance_gpx_km

    facteur_dist = distance_cible_km / max(distance_gpx_km, 1e-9)
    total_corr = total_m * facteur_dist
    dists_corr = np.asarray([d * facteur_dist for d in cum])

    # Élèvations resamplées
    elev_list = np.asarray([getattr(p, "elevation", 0) or 0 for p in points])
    if len(dists_corr) != len(elev_list):
        xs = np.linspace(0, total_m, len(elev_list))
        new_x = np.linspace(0, total_m, len(dists_corr))
        elev_list = np.interp(new_x, xs, elev_list)

    # Références préparées pour le fit
    refs_for_fit = prepare_refs_for_fit(
        refs_input,
        k_up=k_up,
        k_down=k_down,
        k_temp_hot=k_temp_hot,
        k_temp_cold=k_temp_cold,
        opt_temp=opt_temp,
        ideal_refs=ideal_refs,
    )

    # Fit log-log
    a, K = fit_loglog_model(refs_for_fit)

    # Override temps objectif éventuel
    a_override = None
    if objective_time_hms:
        a_override = override_with_objective(
            int(distance_cible_km * 1000),
            objective_time_hms,
            K
        )
    baseline_seconds_per_km = (a_override if a_override is not None else a)

    distance_cible_m = int(distance_cible_km * 1000)
    base_flat_total = predict_time_flat(distance_cible_m, baseline_seconds_per_km, K)
    base_s_per_km_flat = (
        base_flat_total / distance_cible_km if distance_cible_km > 0 else base_flat_total
    )

    # Marqueurs km
    km_marks = [i * 1000 for i in range(1, int(total_corr // 1000) + 1)]
    last_seg = total_corr - (int(total_corr // 1000) * 1000)
    if last_seg > 1e-6:
        km_marks.append(total_corr)

    # Dataframe GPX pour interp lat/lon
    df_points = pd.DataFrame([
        {
            "lat": p.latitude,
            "lon": p.longitude,
            "elev": getattr(p, "elevation", 0),
            "time": getattr(p, "time", None)
        }
        for p in points
    ])

    segment_infos = []
    cum_time_temp = 0.0
    dt_depart = datetime.combine(date_course_local, heure_course_local)

    for i, d in enumerate(km_marks):
        # Élèvation segment
        e_cur = float(np.interp(d, dists_corr, elev_list))
        e_prev = float(np.interp(max(d - 1000.0, 0.0), dists_corr, elev_list)) if i > 0 else e_cur
        d_up = max(0.0, e_cur - e_prev)
        d_down = max(0.0, e_prev - e_cur)

        # Longueur du segment
        seg_length_m = (
            1000.0
            if (i < len(km_marks) - 1 or last_seg < 1e-6)
            else (d - km_marks[-2] if len(km_marks) >= 2 else d)
        )

        # Temps plat théorique
        t_km_flat = base_s_per_km_flat * (seg_length_m / 1000.0)

        # Effet dénivelé
        if apply_elev:
            t_after_elev = apply_elevation_gradient_route(
                t_km_flat,
                d_up,
                d_down,
                segment_length_m=seg_length_m,
                k_up=k_up,
                k_down=k_down
            )
        else:
            t_after_elev = t_km_flat

        # Effet fatigue linéaire
        if apply_fatigue and fatigue_rate > 0 and total_corr > 0:
            progression = d / total_corr
            t_after_fatigue = t_after_elev * (1.0 + (fatigue_rate / 100.0) * progression)
        else:
            t_after_fatigue = t_after_elev

        # Datetime du passage au milieu du segment
        passage_dt = dt_depart + timedelta(
            seconds=cum_time_temp + t_after_fatigue / 2.0
        )

        # Lat/lon du segment
        lat_seg = np.interp(d, dists_corr, df_points["lat"].values)
        lon_seg = np.interp(d, dists_corr, df_points["lon"].values)

        meteo = get_weather_openweather(lat_seg, lon_seg, passage_dt) if apply_temp else None
        temp_here = meteo["temp"] if meteo else None
        wind_here = meteo["wind"] if meteo else None
        hum_here = meteo["humidity"] if meteo else None

        # Effet température
        if apply_temp and temp_here is not None:
            temp_mult = temp_multiplier_nonlin(
                temp_here,
                opt_temp=opt_temp,
                k_hot=k_temp_hot,
                k_cold=k_temp_cold
            )
            t_after_temp = t_after_fatigue * temp_mult
        else:
            temp_mult = 1.0
            t_after_temp = t_after_fatigue

        segment_infos.append({
            "idx": i,
            "d": d,
            "seg_length_m": seg_length_m,
            "d_up": d_up,
            "d_down": d_down,
            "temp": temp_here,
            "wind": wind_here,
            "humidity": hum_here,
            "temp_mult": temp_mult,
            "t_raw": t_after_temp
        })
        cum_time_temp += t_after_temp

    # Ajustement global si objectif temps
    if objective_time_hms:
        objective_seconds = hms_to_seconds(objective_time_hms)
        sum_raw = sum(s["t_raw"] for s in segment_infos)
        scale = (objective_seconds / sum_raw) if (sum_raw > 0) else 1.0
    else:
        scale = 1.0

    # Construction DF résultats
    results = []
    cum_time = 0.0
    for seg in segment_infos:
        t_seg = seg["t_raw"] * scale
        cum_time += t_seg
        pace_per_km = (
            (t_seg / seg["seg_length_m"]) * 1000.0
            if seg["seg_length_m"] > 0
            else t_seg
        )

        results.append({
            "Km": (
                seg["idx"] + 1
                if seg["seg_length_m"] >= 1000 - 1e-6
                else f"{seg['idx']+1} ({seg['seg_length_m']:.0f}m)"
            ),
            "D+ (m)": round(seg["d_up"], 1),
            "D- (m)": round(seg["d_down"], 1),
            "Temp (°C)": round(seg["temp"], 1) if seg["temp"] is not None else None,
            "Vent (m/s)": round(seg["wind"], 1) if seg["wind"] is not None else None,
            "Humidité (%)": round(seg["humidity"], 1) if seg["humidity"] is not None else None,
            "Temp Mult.": round(seg["temp_mult"], 4),
            "Temps segment (s)": round(t_seg, 1),
            "Allure (min/km)": pace_seconds_to_str_per_km(pace_per_km),
            "Temps cumulé": seconds_to_hms(cum_time),
        })

    df = pd.DataFrame(results)
    total_seconds = sum(s["t_raw"] for s in segment_infos) * scale

    return {
        "df": df,
        "total_seconds": total_seconds,
        "total_human": seconds_to_hms(total_seconds),
        "distance_gpx_km": distance_gpx_km,
        "method_used": "3d_haversine",
        "base_flat_total": base_flat_total,
        "a": baseline_seconds_per_km,
        "K": K
    }

# -------------------------
# UI : Entrées & Références
# -------------------------
st.header("1️⃣ Parcours GPX")
gpx_file = st.file_uploader("📂 Importer un fichier GPX", type=["gpx"])
points = None
if gpx_file:
    gpx, points = parse_gpx_points(gpx_file)
    if points:
        total_m_tmp = sum(
            SimplePoint(
                points[i-1].latitude,
                points[i-1].longitude,
                getattr(points[i-1], "elevation", 0)
            ).distance_3d(
                SimplePoint(
                    points[i].latitude,
                    points[i].longitude,
                    getattr(points[i], "elevation", 0)
                )
            )
            for i in range(1, len(points))
        )
        st.session_state["gpx_original_distance_km"] = total_m_tmp / 1000.0
    else:
        st.session_state["gpx_original_distance_km"] = None

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

# Collect raw refs (no recalculation here)
refs_raw = []

for i in range(1, st.session_state.n_refs + 1):

    st.markdown(f"#### Référence {i}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    # --- Checkbox : fichier FIT/TCX ou manuel ---
    with c1:
        use_file = st.checkbox(f"Importer fichier (FIT/TCX) ?", key=f"use_file_{i}")

    # valeurs par défaut
    default_dist = st.session_state.get(f"dist_{i}", 5000 * i)
    default_temps = st.session_state.get(f"temps_{i}", "0:40:00")
    default_dup = st.session_state.get(f"dup_{i}", 0.0)
    default_ddn = st.session_state.get(f"ddn_{i}", 0.0)

    # --- Entrées manuelles ---
    with c2:
        dist = st.number_input(f"Dist {i} (m)", value=float(default_dist), key=f"dist_{i}")
    with c3:
        temps = st.text_input(f"Temps {i} (h:mm:ss)", value=str(default_temps), key=f"temps_{i}")
    with c4:
        dup = st.number_input(f"D+ {i}", value=float(default_dup), key=f"dup_{i}")
    with c5:
        ddn = st.number_input(f"D- {i}", value=float(ddn), key=f"ddn_{i}") if False else st.number_input(f"D- {i}", value=float(default_ddn), key=f"ddn_{i}")  # petite sécurité au cas où

    # --- Import FIT/TCX (dans la boucle) ---
    with c6:
        file_in = st.file_uploader(
            f"FIT/TCX {i}", type=["fit", "tcx"], key=f"fileref_{i}"
        ) if use_file else None

    duration_hms_file = None
    avg_temp_ref = None
    avg_wind_ref = None
    avg_hum_ref = None

    if file_in:
        filename = file_in.name.lower()

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

    # temps utilisé
    temps_effectif = duration_hms_file if duration_hms_file else temps

    # --- Ajout à refs_raw (IMPORTANT : dans la boucle) ---
    refs_raw.append({
        "distance": float(dist),
        "temps": str(temps_effectif),
        "D_up": float(dup),
        "D_down": float(ddn),
        "duration_hms_file": duration_hms_file,
        "avg_temp": avg_temp_ref,
        "avg_wind": avg_wind_ref,
        "avg_humidity": avg_hum_ref,
    })

# Récap brut
st.subheader("⏱️ Récap références (raw)")
for idx, r in enumerate(refs_raw, start=1):
    st.write(
        f"Réf {idx} — Dist: {r['distance']:.0f} m | Brut: {r['temps']} | "
        f"D+ {r['D_up']:.0f} m / D- {r['D_down']:.0f} m | "
        f"Dur file: {r.get('duration_hms_file')} | "
        f"Temp moy: {r.get('avg_temp')}°C"
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
        k_temp_hot = st.number_input(
            "Sensibilité chaude (k_temp_hot)",
            value=0.002,
            format="%.4f",
            step=0.0005
        )
        k_temp_cold = st.number_input(
            "Sensibilité froide (k_temp_cold)",
            value=0.002,
            format="%.4f",
            step=0.0005
        )
        opt_temp = st.number_input(
            "Température optimale (°C)",
            value=12.0,
            format="%.1f",
            step=0.5
        )
    else:
        k_temp_hot = 0.0
        k_temp_cold = 0.0
        opt_temp = 12.0

col1, col2 = st.columns(2)
with col1:
    lat_input = st.number_input(
        "Latitude (pour météo refs fallback)",
        value=48.8566,
        format="%.6f"
    )
    lon_input = st.number_input(
        "Longitude (pour météo refs fallback)",
        value=2.3522,
        format="%.6f"
    )
with col2:
    date_course = st.date_input("Date de la course (Jour J)", value=date.today())
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

# FATIGUE
st.header("3️⃣ bis. Fatigue linéaire")
fatigue_active = st.checkbox("Activer fatigue ?", value=False)
fatigue_rate = 0.0
if fatigue_active:
    fatigue_rate = st.slider("Régression finale (%)", 0.0, 30.0, 5.0, 0.5)

# Option : utiliser CONDITIONS IDÉALES pour le fit
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

st.markdown("---")
st.markdown("**Forcer distance et/ou temps objectif (produit un tableau 'FORCÉ' distinct)**")
colf1, colf2 = st.columns(2)
with colf1:
    force_distance_checkbox = st.checkbox("Forcer la distance pour la prédiction finale ?", value=False)
    if "dist_forced" not in st.session_state:
        st.session_state["dist_forced"] = 5.17
    distance_forced_km = st.number_input(
        "Distance forcée (km)",
        value=st.session_state["dist_forced"],
        format="%.2f",
        key="dist_forced"
    ) if force_distance_checkbox else None

with colf2:
    force_time_checkbox = st.checkbox("Forcer un temps objectif ?", value=False)
    if "time_forced" not in st.session_state:
        st.session_state["time_forced"] = "0:18:30"
    time_forced_hms = st.text_input(
        "Temps objectif (h:mm:ss)",
        value=st.session_state["time_forced"],
        key="time_forced"
    ) if force_time_checkbox else None

if st.button("📊 Calculer prédiction finale (FORCÉ si activé)"):
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
                objective_time_hms=time_forced_hms if force_time_checkbox else None,
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

# display side-by-side
if "res_base" in st.session_state or "res_forced" in st.session_state:
    base = st.session_state.get("res_base", None)
    forced = st.session_state.get("res_forced", None)

    left, right = st.columns(2)
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

        # Carte
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

        # Profil d'altitude
        st.subheader("📊 Profil d'altitude")
        plt.figure(figsize=(10, 4))

        total_m = 0.0
        cumdists = [0.0]
        for i in range(1, len(points)):
            d = SimplePoint(
                points[i-1].latitude,
                points[i-1].longitude,
                getattr(points[i-1], "elevation", 0)
            ).distance_3d(
                SimplePoint(
                    points[i].latitude,
                    points[i].longitude,
                    getattr(points[i], "elevation", 0)
                )
            )
            total_m += d
            cumdists.append(total_m)
        x_km = np.array(cumdists) / 1000.0
        y_elev = np.array([p.elevation or 0 for p in points])

        plt.plot(x_km, y_elev, lw=2)
        plt.xlabel("Distance (km)")
        plt.ylabel("Altitude (m)")
        plt.title("Profil d'altitude du parcours")
        plt.grid(alpha=0.3)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Impossible d'afficher la carte/profil : {e}")
