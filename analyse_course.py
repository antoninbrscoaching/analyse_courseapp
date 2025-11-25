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
st.title("🏃‍♂️ Analyse & Prédiction de course (GPX + FIT + Météo + Fatigue linéaire) — Route")

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

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def parse_gpx_points(file):
    gpx = gpxpy.parse(file)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                points.append(p)
    return gpx, points

def gpx_to_df(points):
    return pd.DataFrame([{"lat": p.latitude, "lon": p.longitude, "elev": p.elevation or 0, "time": getattr(p, "time", None)} for p in points])

def compute_total_and_cumdist(points):
    if not points or len(points) < 2:
        return 0.0, [0.0], "none", {}
    total_3d = 0.0
    dists_3d = [0.0]
    for i in range(1, len(points)):
        try:
            seg = points[i].distance_3d(points[i-1])
        except Exception:
            seg = 0.0
        total_3d += seg
        dists_3d.append(total_3d)

    total_hav = 0.0
    dists_hav = [0.0]
    for i in range(1, len(points)):
        lat1, lon1 = points[i-1].latitude, points[i-1].longitude
        lat2, lon2 = points[i].latitude, points[i].longitude
        seg = haversine_m(lat1, lon1, lat2, lon2)
        total_hav += seg
        dists_hav.append(total_hav)

    debug = {"total_3d": total_3d, "total_hav": total_hav, "n_points": len(points)}
    if total_3d <= 0 or abs(total_3d - total_hav) / max(total_hav, 1e-6) > 0.02:
        return total_hav, dists_hav, "haversine", debug
    else:
        return total_3d, dists_3d, "3d", debug

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
        dup = float(np.sum(np.diff(df["elev"]).clip(min=0)))
        ddn = float(-np.sum(np.diff(df["elev"]).clip(max=0)))
        return dict(distance=round(float(df["dist"].max())), D_up=round(dup), D_down=round(ddn))
    except Exception:
        return None

# ------------------------------------------------------
# météo historique
# ------------------------------------------------------
@st.cache_data(ttl=60*60)
def fetch_open_meteo_hourly(lat, lon, start_date_str, end_date_str):
    base = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "hourly": "temperature_2m",
        "timezone": "UTC"
    }
    try:
        r = requests.get(base, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
        times = j.get("hourly", {}).get("time", [])
        temps = j.get("hourly", {}).get("temperature_2m", [])
        out = {}
        for t, temp in zip(times, temps):
            dt = datetime.fromisoformat(t)
            out[dt] = float(temp)
        return out
    except Exception:
        return {}

def get_temp_for_datetime(hourly_dict, target_dt):
    if not hourly_dict:
        return None
    keys = sorted(hourly_dict.keys())
    if target_dt in hourly_dict:
        return hourly_dict[target_dt]
    lower = None
    upper = None
    for k in keys:
        if k <= target_dt:
            lower = k
        if k > target_dt:
            upper = k
            break
    if lower is None:
        return hourly_dict[keys[0]]
    if upper is None:
        return hourly_dict[keys[-1]]
    t0 = lower; t1 = upper
    v0 = hourly_dict[t0]; v1 = hourly_dict[t1]
    frac = (target_dt - t0).total_seconds() / (t1 - t0).total_seconds()
    return float(v0 + (v1 - v0) * frac)

# ------------------------------------------------------
# Modèle log-log
# ------------------------------------------------------
def fit_loglog_model(refs, k_up=1.0, k_down=1.0):
    xs, ys = [], []
    for r in refs:
        d = float(r.get("distance", 0) or 0)
        t_raw = hms_to_seconds(r.get("temps", "0:00:00") or "0:00:00")
        dup = float(r.get("D_up", 0) or 0)
        ddn = float(r.get("D_down", 0) or 0)
        if d <= 0 or t_raw <= 0:
            continue
        elev_factor = (k_up ** dup) * (k_down ** ddn)
        t_eq = t_raw / elev_factor if elev_factor > 0 else t_raw
        if t_eq <= 0:
            continue
        xs.append(math.log(d))
        ys.append(math.log(t_eq))
    n = len(xs)
    if n < 2:
        raise ValueError("Il faut deux références minimum valides.")
    sum_x = sum(xs); sum_y = sum(ys)
    sum_xx = sum(x*x for x in xs)
    sum_xy = sum(x*y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x**2
    if denom == 0:
        raise ValueError("Distances identiques ou colinéaires.")
    K = (n * sum_xy - sum_x * sum_y) / denom
    a = math.exp((sum_y - K * sum_x) / n)
    return a, K

def predict_time_flat(distance_m, a, K):
    return a * (distance_m ** K)

def override_with_objective(distance_obj_m, time_obj_hms, K):
    t_obj = hms_to_seconds(time_obj_hms or "0:00:00")
    if t_obj <= 0 or distance_obj_m <= 0:
        raise ValueError("Temps objectif invalide ou distance nulle.")
    return t_obj / (distance_obj_m ** K)

def temp_multiplier_nonlin(temp_c, opt_temp=12.0, k_hot=0.002, k_cold=0.002, power=1.6):
    if temp_c is None:
        return 1.0
    delta = temp_c - opt_temp
    if delta >= 0:
        return 1.0 + k_hot * (delta ** power)
    else:
        return 1.0 + k_cold * ((-delta) ** power)

def apply_elevation_gradient_route(time_flat_s, d_up_m, d_down_m, segment_length_m=1000.0, k_up=1.040, k_down=0.996):
    if segment_length_m <= 0:
        return time_flat_s
    g_up = (d_up_m / segment_length_m) * 100.0
    g_down = (d_down_m / segment_length_m) * 100.0
    delta_up = float(k_up) - 1.0
    delta_down = float(k_down) - 1.0
    uphill_factor = math.exp(delta_up * g_up)
    downhill_factor = math.exp(delta_down * g_down)
    return float(time_flat_s * uphill_factor * downhill_factor)

# ---------------- UI & Inputs (conservés et ajustés) ----------------
st.header("1️⃣ Parcours GPX")
gpx_file = st.file_uploader("📂 Importer un fichier GPX", type=["gpx"])

st.header("2️⃣ Courses de référence (manuel ou FIT)")
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
for i in range(1, st.session_state.n_refs + 1):
    st.markdown(f"#### Référence {i}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        use_fit = st.checkbox(f"FIT ?", key=f"use_fit_{i}")
    with c2:
        dist = st.number_input(f"Dist {i} (m)", value=5000 * i, key=f"dist_{i}")
    with c3:
        temps = st.text_input(f"Temps {i} (h:mm:ss)", value="0:40:00", key=f"temps_{i}")
    with c4:
        dup = st.number_input(f"D+ {i}", value=0.0, key=f"dup_{i}")
    with c5:
        ddn = st.number_input(f"D- {i}", value=0.0, key=f"ddn_{i}")
    with c6:
        file_fit = st.file_uploader(f"FIT {i}", type=["fit"], key=f"fit_{i}") if use_fit else None
        if file_fit:
            data_fit = parse_fit(file_fit)
            if data_fit:
                dist, dup, ddn = data_fit["distance"], data_fit["D_up"], data_fit["D_down"]
                st.info(f"✔ FIT détecté : {dist}m | D+{dup} | D-{ddn}")
    refs.append(dict(distance=dist, temps=temps, D_up=dup, D_down=ddn))

st.header("3️⃣ Paramètres modèle")
c1, c2 = st.columns(2)
with c1:
    use_elev_coeff = st.checkbox("Activer coefficients montée/descente 🎢", value=True)
    if use_elev_coeff:
        k_up = st.number_input("Coefficient montée (k_up)", value=1.040, format="%.3f", step=0.001)
        k_down = st.number_input("Coefficient descente (k_down)", value=0.996, format="%.3f", step=0.001)
    else:
        k_up = 1.0; k_down = 1.0
with c2:
    use_temp_coeff = st.checkbox("Activer coefficients température 🌡️", value=True)
    if use_temp_coeff:
        k_temp_hot = st.number_input("Sensibilité chaude (k_temp_hot)", value=0.002, format="%.3f", step=0.001)
        k_temp_cold = st.number_input("Sensibilité froide (k_temp_cold)", value=0.002, format="%.3f", step=0.001)
        opt_temp = st.number_input("Température optimale (°C)", value=12.0, format="%.1f", step=0.5)
    else:
        k_temp_hot = 0.0; k_temp_cold = 0.0; opt_temp = 12.0

col1, col2 = st.columns(2)
with col1:
    lat_input = st.number_input("Latitude (pour météo)", value=48.8566, format="%.6f")
    lon_input = st.number_input("Longitude (pour météo)", value=2.3522, format="%.6f")
    use_hist_refs = st.checkbox("Recalibrer les références avec météo historique ?", value=False)
with col2:
    date_course = st.date_input("Date de la course (Jour J)", value=date.today())
    heure_course = st.time_input("Heure de départ (Jour J)", value=time(9, 0))

if use_hist_refs:
    st.markdown("**Dates/Heures pour les références**")
    for idx, r in enumerate(refs):
        cA, cB = st.columns(2)
        with cA:
            ref_date = st.date_input(f"Date référence #{idx+1}", key=f"ref_date_{idx}", value=date.today())
        with cB:
            ref_time = st.time_input(f"Heure référence #{idx+1}", key=f"ref_time_{idx}", value=time(9, 0))
        refs[idx]["ref_datetime"] = datetime.combine(ref_date, ref_time)

st.header("3️⃣ bis. Fatigue linéaire")
fatigue_active = st.checkbox("Activer fatigue ?", value=False)
fatigue_rate = 0.0
if fatigue_active:
    fatigue_rate = st.slider("Régression finale (%)", 0.0, 30.0, 5.0, 0.5)

# ---------------- run_prediction (retourne df + total) ----------------
@st.cache_data(ttl=60)
def fetch_historical_range(lat, lon, start_date, end_date):
    return fetch_open_meteo_hourly(lat, lon, start_date.isoformat(), end_date.isoformat())

def run_prediction_df(distance_cible_km,
                      refs_input,
                      points,
                      date_course_local,
                      heure_course_local,
                      use_hist_for_refs_local=False,
                      apply_elev=True,
                      apply_temp=True,
                      apply_fatigue=True,
                      objective_time_hms=None,
                      local_k_up=1.040, local_k_down=0.996,
                      local_k_temp_hot=0.002, local_k_temp_cold=0.002, local_opt_temp=12.0,
                      local_fatigue_rate=0.0):
    """
    Calcule et renvoie DataFrame km-par-km et temps total.
    Ne fait PAS d'affichage Streamlit — permet d'avoir base_df et forced_df distincts.
    """
    # points must be non-empty
    if not points or len(points) < 2:
        raise ValueError("GPX invalide ou trop court.")

    # distance/cum dists robustes
    total_m, dists, method_used, debug = compute_total_and_cumdist(points)
    distance_gpx_km = total_m / 1000.0

    # sécurité distance cible
    if not distance_cible_km or distance_cible_km <= 0:
        distance_cible_km = distance_gpx_km

    facteur_dist = distance_cible_km / max(distance_gpx_km, 1e-6)
    total_corr = total_m * facteur_dist
    dists_corr = np.asarray([d * facteur_dist for d in dists])

    elev_list = np.asarray([p.elevation or 0 for p in points])
    if len(dists_corr) != len(elev_list):
        # si arrays ont tailles différentes, on resample elev_list to len(dists_corr) via simple interp
        xs = np.linspace(0, total_m, len(elev_list))
        new_x = np.linspace(0, total_m, len(dists_corr))
        elev_list = np.interp(new_x, xs, elev_list)

    # préparer références (recalées si historique demandé)
    # récup historique pour plage si demandée
    center_lat = np.mean([p.latitude for p in points])
    center_lon = np.mean([p.longitude for p in points])
    min_ref_date = None; max_ref_date = date_course_local
    if use_hist_for_refs_local:
        for r in refs_input:
            rd = r.get("ref_datetime")
            if rd:
                d = rd.date()
                if min_ref_date is None or d < min_ref_date:
                    min_ref_date = d
                if d > max_ref_date:
                    max_ref_date = d
    if min_ref_date is None:
        min_ref_date = date_course_local
    hourly_temps_cache = {}
    try:
        hourly_temps_cache = fetch_historical_range(center_lat, center_lon, min_ref_date, max_ref_date)
    except Exception:
        hourly_temps_cache = {}

    # Si on recalibre refs (on neutralise effet temp réel pour ramener à "plat")
    refs_for_fit = []
    if use_hist_for_refs_local and hourly_temps_cache:
        for r in refs_input:
            rr = r.copy()
            rd = r.get("ref_datetime")
            if rd:
                t_ref = get_temp_for_datetime(hourly_temps_cache, rd)
                mult = temp_multiplier_nonlin(t_ref, opt_temp=local_opt_temp, k_hot=local_k_temp_hot, k_cold=local_k_temp_cold)
                secs = hms_to_seconds(r.get("temps", "0:00:00"))
                if secs > 0 and mult > 0:
                    new_secs = secs / mult
                    rr["temps"] = seconds_to_hms(new_secs)
                    rr["_temp_ref"] = t_ref
            refs_for_fit.append(rr)
    else:
        refs_for_fit = [r.copy() for r in refs_input]

    # Fit log-log
    a, K = fit_loglog_model(refs_for_fit, k_up=(local_k_up if apply_elev else 1.0), k_down=(local_k_down if apply_elev else 1.0))

    # override with objective if provided
    if objective_time_hms:
        a = override_with_objective(int(distance_cible_km * 1000), objective_time_hms, K)

    distance_cible_m = int(distance_cible_km * 1000)
    base_flat_total = predict_time_flat(distance_cible_m, a, K)
    base_s_per_km_flat = base_flat_total / distance_cible_km if distance_cible_km > 0 else base_flat_total

    # km marks
    km_marks = [i * 1000 for i in range(1, int(total_corr // 1000) + 1)]
    if (total_corr % 1000) != 0:
        km_marks.append(total_corr)

    results = []
    cum_time = 0.0
    dt_depart = datetime.combine(date_course_local, heure_course)

    for i, d in enumerate(km_marks):
        e_cur = float(np.interp(d, dists_corr, elev_list))
        e_prev = float(np.interp(max(d - 1000.0, 0.0), dists_corr, elev_list)) if i > 0 else e_cur
        d_up = max(0.0, e_cur - e_prev)
        d_down = max(0.0, e_prev - e_cur)

        t_km = base_s_per_km_flat

        if apply_elev:
            t_km = apply_elevation_gradient_route(t_km, d_up, d_down, segment_length_m=1000.0, k_up=local_k_up, k_down=local_k_down)

        if apply_fatigue and local_fatigue_rate > 0 and total_corr > 0:
            progression = d / total_corr
            t_km *= (1.0 + (local_fatigue_rate / 100.0) * progression)

        passage_dt = dt_depart + timedelta(seconds=cum_time + t_km / 2.0)
        temp_at_passage = get_temp_for_datetime(hourly_temps_cache, passage_dt)

        if apply_temp and temp_at_passage is not None:
            mult_temp = temp_multiplier_nonlin(temp_at_passage, opt_temp=local_opt_temp, k_hot=local_k_temp_hot, k_cold=local_k_temp_cold)
            t_km *= mult_temp
        else:
            mult_temp = 1.0

        cum_time += t_km

        results.append({
            "Km": i + 1,
            "D+ (m)": round(d_up, 1),
            "D- (m)": round(d_down, 1),
            "Temp (°C)": round(temp_at_passage, 1) if temp_at_passage is not None else None,
            "Temp Mult.": round(mult_temp, 4),
            "Temps segment (s)": round(t_km, 1),
            "Allure (min/km)": f"{int(t_km // 60)}:{int(t_km % 60):02d}",
            "Temps cumulé": seconds_to_hms(cum_time),
        })

    df = pd.DataFrame(results)
    return {
        "df": df,
        "total_seconds": cum_time,
        "total_human": seconds_to_hms(cum_time),
        "distance_gpx_km": distance_gpx_km,
        "method_used": method_used,
        "debug": debug,
        "base_flat_total": base_flat_total,
        "a": a, "K": K
    }

# ----------------- Interactions : bouton base et forcé (affichage côte-à-côte) -----------------
st.subheader("4️⃣ Calcul & Comparaison")

# bouton pour calcul base (références)
if st.button("▶️ Calculer prédiction (BASE, d'après références)"):
    if not gpx_file:
        st.error("Importe un fichier GPX d'abord.")
    else:
        gpx, points = parse_gpx_points(gpx_file)
        # run base (no force)
        res_base = run_prediction_df(
            distance_cible_km=None,
            refs_input=refs,
            points=points,
            date_course_local=date_course,
            heure_course_local=heure_course,
            use_hist_for_refs_local=use_hist_refs,
            apply_elev=use_elev_coeff,
            apply_temp=use_temp_coeff,
            apply_fatigue=fatigue_active,
            objective_time_hms=None,
            local_k_up=k_up, local_k_down=k_down,
            local_k_temp_hot=k_temp_hot, local_k_temp_cold=k_temp_cold, local_opt_temp=opt_temp,
            local_fatigue_rate=fatigue_rate
        )
        st.session_state["res_base"] = res_base
        st.success(f"Base calculée — distance GPX détectée: {res_base['distance_gpx_km']:.3f} km (méthode: {res_base['method_used']})")

# zone for forcing options and final calculation
st.markdown("---")
st.markdown("**Forcer distance et/ou temps objectif (produit un tableau 'FORCÉ' distinct)**")
colf1, colf2 = st.columns(2)
with colf1:
    force_distance_checkbox = st.checkbox("Forcer la distance pour la prédiction finale ?", value=False)
    distance_forced_km = st.number_input("Distance forcée (km)", value=5.17, format="%.2f") if force_distance_checkbox else None
with colf2:
    force_time_checkbox = st.checkbox("Forcer un temps objectif ?", value=False)
    time_forced_hms = st.text_input("Temps objectif (h:mm:ss)", value="0:18:30") if force_time_checkbox else None

if st.button("📊 Calculer prédiction finale (FORCÉ si activé)"):
    if not gpx_file:
        st.error("Importe un fichier GPX d'abord.")
    else:
        gpx, points = parse_gpx_points(gpx_file)
        # determine distance to use for forced run
        dist_target = None
        if force_distance_checkbox and distance_forced_km:
            dist_target = distance_forced_km
        else:
            dist_target = None  # run_prediction_df will use GPX distance if None

        res_forced = run_prediction_df(
            distance_cible_km=dist_target,
            refs_input=refs,
            points=points,
            date_course_local=date_course,
            heure_course_local=heure_course,
            use_hist_for_refs_local=use_hist_refs,
            apply_elev=use_elev_coeff,
            apply_temp=use_temp_coeff,
            apply_fatigue=fatigue_active,
            objective_time_hms=time_forced_hms if force_time_checkbox else None,
            local_k_up=k_up, local_k_down=k_down,
            local_k_temp_hot=k_temp_hot, local_k_temp_cold=k_temp_cold, local_opt_temp=opt_temp,
            local_fatigue_rate=fatigue_rate
        )
        st.session_state["res_forced"] = res_forced
        st.success(f"Prédiction forcée calculée — cible: {distance_forced_km if distance_forced_km else 'GPX'} km")

# Afficher côte-à-côte si disponibles
if "res_base" in st.session_state or "res_forced" in st.session_state:
    base = st.session_state.get("res_base", None)
    forced = st.session_state.get("res_forced", None)

    left, right = st.columns(2)
    with left:
        st.subheader("📈 Base (d'après références)")
        if base:
            st.write(f"Distance GPX détectée: {base['distance_gpx_km']:.3f} km (méthode: {base['method_used']})")
            st.write(f"Temps total (base): {base['total_human']}")
            st.dataframe(base["df"], use_container_width=True)
        else:
            st.info("Clique sur 'Calculer prédiction (BASE)' pour générer ce tableau.")

    with right:
        st.subheader("🎯 Forcé (distance/temps forcés)")
        if forced:
            st.write(f"Distance cible: {distance_forced_km if force_distance_checkbox else round(forced['distance_gpx_km'],3)} km")
            st.write(f"Temps total (forcé): {forced['total_human']}")
            st.dataframe(forced["df"], use_container_width=True)
        else:
            st.info("Clique sur 'Calculer prédiction finale (FORCÉ)' pour générer ce tableau.")

# ---------------- Carte + profil (affiche using last used points if exist) ----------------
if gpx_file:
    try:
        gpx, points = parse_gpx_points(gpx_file)
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
            tooltip={"text":"{name}"}
        )
        st.pydeck_chart(deck, use_container_width=True)

        # Profil d'altitude
        st.subheader("📊 Profil d'altitude")
        plt.figure(figsize=(10, 4))

        # Calcul distance cumulée corrigée
        total_m, cumdists, method_used, debug = compute_total_and_cumdist(points)
        x_km = np.array(cumdists) / 1000.0  # distance en km
        y_elev = np.array([p.elevation or 0 for p in points])

        plt.plot(x_km, y_elev, color="blue", lw=2)
        plt.xlabel("Distance (km)")
        plt.ylabel("Altitude (m)")
        plt.title(f"Profil d'altitude du parcours (méthode {method_used})")
        plt.grid(alpha=0.3)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Impossible d'afficher la carte/profil : {e}")
