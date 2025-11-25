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
        dup = np.sum(np.diff(df["elev"]).clip(min=0))
        ddn = -np.sum(np.diff(df["elev"]).clip(max=0))
        return dict(distance=round(df["dist"].max()), D_up=round(dup), D_down=round(ddn))
    except Exception:
        return None

# ------------------------------------------------------
# 🌐 Open-Meteo historique (cache pour limiter appels)
# ------------------------------------------------------
@st.cache_data(ttl=60*60)
def fetch_open_meteo_hourly(lat, lon, start_date_str, end_date_str):
    """Récupère hourly temperature_2m depuis archive-api.open-meteo.com pour latitude, longitude, date range.
    Retourne dict: {datetime (UTC) : temp_c} (datetimes are naive UTC).
    start_date_str and end_date_str format 'YYYY-MM-DD'.
    """
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
            # times are ISO strings like '2023-11-20T12:00'
            dt = datetime.fromisoformat(t)
            out[dt] = temp
        return out
    except Exception as e:
        # en cas d'erreur, renvoyer dict vide
        return {}

def get_temp_for_datetime(hourly_dict, target_dt):
    """Interpole temperature horaire (hourly_dict keys are datetimes at hours) pour un target_dt (naive UTC)."""
    if not hourly_dict:
        return None
    keys = sorted(hourly_dict.keys())
    # si exact
    if target_dt in hourly_dict:
        return hourly_dict[target_dt]
    # find nearest lower and upper hours
    lower = None
    upper = None
    for k in keys:
        if k <= target_dt:
            lower = k
        if k > target_dt:
            upper = k
            break
    if lower is None:
        # use earliest
        return hourly_dict[keys[0]]
    if upper is None:
        return hourly_dict[keys[-1]]
    t0 = lower
    t1 = upper
    v0 = hourly_dict[t0]
    v1 = hourly_dict[t1]
    # fraction between hours
    frac = (target_dt - t0).total_seconds() / (t1 - t0).total_seconds()
    return v0 + (v1 - v0) * frac

# ------------------------------------------------------
# 🧠 MODÈLE LOG-LOG
# ------------------------------------------------------
def fit_loglog_model(refs, k_up=1.0, k_down=1.0):
    xs = []
    ys = []
    for r in refs:
        d = r["distance"]
        t_raw = hms_to_seconds(r["temps"])
        dup = r.get("D_up", 0)
        ddn = r.get("D_down", 0)

        # apply elevation coefficients at reference-level (these are multiplicative corrections of time)
        elev_factor = (k_up ** dup) * (k_down ** ddn)
        t_eq = t_raw / elev_factor if elev_factor > 0 else t_raw

        if d > 0 and t_eq > 0:
            xs.append(math.log(d))
            ys.append(math.log(t_eq))

    n = len(xs)
    if n < 2:
        raise ValueError("Il faut deux références minimum.")

    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x*x for x in xs)
    sum_xy = sum(x*y for x, y in zip(xs, ys))

    denom = n * sum_xx - sum_x**2
    if denom == 0:
        raise ValueError("Distances identiques.")

    K = (n * sum_xy - sum_x * sum_y) / denom
    a = math.exp((sum_y - K * sum_x) / n)

    return a, K

def predict_time_flat(distance_m, a, K):
    return a * (distance_m ** K)

# ------------------------------------------------------
# 🛠️ Modèle d'effet température non linéaire (en U)
# ------------------------------------------------------
def temp_multiplier_nonlin(temp_c, opt_temp=12.0, k_hot=0.002, k_cold=0.002, power=1.6):
    """
    Renvoie un multiplicateur > 1 quand la temp s'éloigne de l'optimum.
    - opt_temp : température optimale (°C), par défaut 12°C
    - k_hot : sensibilité côté chaud (par unité de (°C^power))
    - k_cold: sensibilité côté froid
    - power : exponent (1.5-2 donne une 'courbure' non linéaire)
    Exemple: à opt=12°C, temp=20C, delta=8 -> multiplier = 1 + k_hot * 8^power
    """
    if temp_c is None:
        return 1.0
    delta = temp_c - opt_temp
    if delta >= 0:
        return 1.0 + k_hot * (delta ** power)
    else:
        return 1.0 + k_cold * ((-delta) ** power)

# ------------------------------------------------------
# 🗺️ 1. CHARGEMENT GPX
# ------------------------------------------------------
st.header("1️⃣ Parcours GPX")
gpx_file = st.file_uploader("📂 Importer un fichier GPX", type=["gpx"])

# ------------------------------------------------------
# 🧮 2. COURSES DE RÉFÉRENCE
# ------------------------------------------------------
st.header("2️⃣ Courses de référence (manuel ou FIT)")
if "n_refs" not in st.session_state:
    st.session_state.n_refs = 3

cols = st.columns([1, 1])
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
        dist = st.number_input(f"Dist {i} (m)", value=5000*i, key=f"dist_{i}")
    with c3:
        temps = st.text_input(f"Temps {i} (h:mm:ss)", value="0:40:00", key=f"temps_{i}")
    with c4:
        dup = st.number_input(f"D+ {i}", value=0, key=f"dup_{i}")
    with c5:
        ddn = st.number_input(f"D- {i}", value=0, key=f"ddn_{i}")
    with c6:
        file_fit = st.file_uploader(f"FIT {i}", type=["fit"], key=f"fit_{i}") if use_fit else None
        if file_fit:
            data_fit = parse_fit(file_fit)
            if data_fit:
                dist, dup, ddn = data_fit["distance"], data_fit["D_up"], data_fit["D_down"]
                st.info(f"✔ FIT détecté : {dist}m | D+{dup} | D-{ddn}")

    refs.append(dict(distance=dist, temps=temps, D_up=dup, D_down=ddn))

# ------------------------------------------------------
# ⚙️ 3. PARAMÈTRES
# ------------------------------------------------------
st.header("3️⃣ Paramètres modèle")

c1, c2 = st.columns(2)

# ----------------- Colonne 1 : coefficients élévation -----------------
with c1:
    use_elev_coeff = st.checkbox("Activer coefficients montée/descente 🎢", value=True)
    if use_elev_coeff:
        k_up = st.number_input("Coefficient montée (k_up)", value=1.040, format="%.3f", step=0.001)
        k_down = st.number_input("Coefficient descente (k_down)", value=0.996, format="%.3f", step=0.001)
    else:
        k_up = 1.0
        k_down = 1.0

# ----------------- Colonne 2 : coefficients température -----------------
with c2:
    use_temp_coeff = st.checkbox("Activer coefficients température 🌡️", value=True)
    if use_temp_coeff:
        # ici on demande les *échelles* de sensibilité (trois décimales)
        k_temp_hot = st.number_input("Sensibilité chaude (k_temp_hot)", value=0.002, format="%.3f", step=0.001)
        k_temp_cold = st.number_input("Sensibilité froide (k_temp_cold)", value=0.002, format="%.3f", step=0.001)
        opt_temp = st.number_input("Température optimale (°C)", value=12.0, format="%.1f", step=0.5)
    else:
        k_temp_hot = 0.0
        k_temp_cold = 0.0
        opt_temp = 12.0

# ----------------- Latitude / Longitude / API / historical refs -----------------
col1, col2 = st.columns(2)
with col1:
    lat_input = st.number_input("Latitude (pour météo)", value=48.8566, format="%.6f")
    lon_input = st.number_input("Longitude (pour météo)", value=2.3522, format="%.6f")
    # Option pour utiliser Open-Meteo historique pour recalibrer les références
    use_hist_refs = st.checkbox("Recalibrer les références avec météo historique ?", value=False)
with col2:
    date_course = st.date_input("Date de la course (Jour J)", value=date.today())
    heure_course = st.time_input("Heure de départ (Jour J)", value=time(9, 0))

# Si user veut indiquer une date/heure pour chaque référence (utile pour recalibrage historique)
if use_hist_refs:
    st.markdown("**Dates/Heures pour les références** (utilisées pour récupérer la température historique).")
    for idx, r in enumerate(refs):
        cA, cB = st.columns(2)
        with cA:
            ref_date = st.date_input(f"Date référence #{idx+1}", key=f"ref_date_{idx}", value=date.today())
        with cB:
            ref_time = st.time_input(f"Heure référence #{idx+1}", key=f"ref_time_{idx}", value=time(9, 0))
        # store into refs so run_prediction can use them
        refs[idx]["ref_datetime"] = datetime.combine(ref_date, ref_time)

# ------------------------------------------------------
# 💤 3bis. FATIGUE
# ------------------------------------------------------
st.header("3️⃣ bis. Fatigue linéaire")
fatigue_active = st.checkbox("Activer fatigue ?", value=False)
fatigue_rate = 0.0
if fatigue_active:
    fatigue_rate = st.slider("Régression finale (%)", 0.0, 30.0, 5.0, 0.5)

# ------------------------------------------------------
# 🧠 4. ANALYSE
# ------------------------------------------------------
st.header("4️⃣ Analyse et prédiction route")
st.caption("Calcul sur la distance GPX avec facteurs montée/descente et correction température non-linéaire")

# État pour savoir si un premier calcul a déjà été fait
if "first_run_done" not in st.session_state:
    st.session_state.first_run_done = False
if "distance_gpx_km" not in st.session_state:
    st.session_state.distance_gpx_km = None

# Cache local des températures historiques pour la plage demandée
@st.cache_data(ttl=60*60)
def fetch_historical_range(lat, lon, start_date, end_date):
    return fetch_open_meteo_hourly(lat, lon, start_date.isoformat(), end_date.isoformat())

# ------------------ REMPLACER ICI : fonction run_prediction (corrigée) ------------------
def run_prediction(distance_cible_km, objectif_temps_forced=None, show_map=False, use_hist_for_refs=False):
    """Exécute la prédiction complète route. Utilise gradient + coefficients k_up/k_down et modèle temp non-linéaire."""
    if not gpx_file:
        st.error("⚠️ Importer un fichier GPX d’abord.")
        return

    # --- Lecture GPX ---
    gpx, points = parse_gpx_points(gpx_file)
    df_points = gpx_to_df(points)
    if df_points.empty:
        st.error("Fichier GPX invalide.")
        return

    # --- Capturer les variables UI localement (plus fiable que globals()) ---
    use_elev = bool(use_elev_coeff)
    local_k_up = float(k_up) if use_elev else 1.0
    local_k_down = float(k_down) if use_elev else 1.0

    use_temp = bool(use_temp_coeff)
    local_k_temp_hot = float(k_temp_hot) if use_temp else 0.0
    local_k_temp_cold = float(k_temp_cold) if use_temp else 0.0
    local_opt_temp = float(opt_temp) if use_temp else 12.0

    local_fatigue_active = bool(fatigue_active)
    local_fatigue_rate = float(fatigue_rate)

    # centre du parcours pour météo (si besoin)
    center_lat = df_points["lat"].mean() if "lat" in df_points.columns else float(lat_input)
    center_lon = df_points["lon"].mean() if "lon" in df_points.columns else float(lon_input)

    # Distance cumulée brute (GPX)
    dists = [0.0]
    total = 0.0
    for i in range(1, len(points)):
        total += points[i].distance_3d(points[i - 1])
        dists.append(total)
    if total <= 0:
        st.error("Distance GPX nulle ou invalide.")
        return

    distance_gpx_km = total / 1000.0
    st.session_state.distance_gpx_km = distance_gpx_km

    # Facteur pour adapter la distance cible (si distance forcée)
    # protection si distance_cible_km invalide
    if not distance_cible_km or distance_cible_km <= 0:
        distance_cible_km = distance_gpx_km
    facteur_dist = distance_cible_km / distance_gpx_km if distance_gpx_km > 0 else 1.0
    total_corr = total * facteur_dist
    dists_corr = np.asarray([d * facteur_dist for d in dists])

    # sécurité : elev_list en array et vérification longueur
    elev_list = np.asarray([p.elevation or 0 for p in points])
    if len(dists_corr) < 2 or len(elev_list) < 2:
        st.error("Impossible d'interpoler le profil — nombre de points insuffisant ou incohérent dans le GPX.")
        return

    # --- Modèle log-log (on passe les coefficients locaux)
    try:
        a, K = fit_loglog_model(refs, k_up=(local_k_up if use_elev else 1.0), k_down=(local_k_down if use_elev else 1.0))
    except ValueError as e:
        st.error(f"Problème lors de l’ajustement du modèle log-log : {e}")
        return
    st.info(f"📐 Exposant log-log estimé : {K:.4f}")

    # Recalage éventuel avec un temps objectif forcé
    if objectif_temps_forced:
        a = override_with_objective(int(distance_cible_km * 1000), objectif_temps_forced, K)
        st.success(f"🎯 Modèle recalé pour {distance_cible_km:.2f} km en {objectif_temps_forced} (sur plat)")

    # Temps plat global sur la distance cible
    distance_cible_m = int(distance_cible_km * 1000)
    base_flat_total = predict_time_flat(distance_cible_m, a, K)
    # protection division par zéro
    base_s_per_km_flat = base_flat_total / distance_cible_km if distance_cible_km > 0 else base_flat_total

    # Préparer km_marks (on garde km entiers + éventuel dernier segment)
    km_marks = [i * 1000 for i in range(1, int(total_corr // 1000) + 1)]
    if total_corr % 1000 != 0:
        km_marks.append(total_corr)

    # ----- récup historique météo si nécessaire -----
    min_ref_date = None
    max_ref_date = date_course
    if use_hist_for_refs:
        for r in refs:
            rd = r.get("ref_datetime")
            if rd:
                donly = rd.date()
                if min_ref_date is None or donly < min_ref_date:
                    min_ref_date = donly
                if donly > max_ref_date:
                    max_ref_date = donly
    if min_ref_date is None:
        min_ref_date = date_course

    # récupère les températures horaires entre min_ref_date et max_ref_date pour le centre du parcours
    try:
        hourly_temps_cache = fetch_historical_range(center_lat, center_lon, min_ref_date, max_ref_date)
    except Exception:
        hourly_temps_cache = {}

    # --- Boucle km par km ---
    results = []
    cum_time = 0.0
    dt_depart = datetime.combine(date_course, heure_course)

    for i, d in enumerate(km_marks):
        # altitude interpolée (safe : dists_corr et elev_list sont des arrays)
        e_cur = float(np.interp(d, dists_corr, elev_list))
        # pour le premier segment, on prend la même altitude précédente (pas de négatif)
        e_prev = float(np.interp(max(d - 1000.0, 0.0), dists_corr, elev_list)) if i > 0 else e_cur

        d_up = max(0.0, e_cur - e_prev)
        d_down = max(0.0, e_prev - e_cur)

        # Temps plat sur ce km
        t_km_flat = base_s_per_km_flat

        # Appliquer effet élévation (gradient-based) si activé
        if use_elev:
            t_km = apply_elevation_gradient_route(t_km_flat, d_up, d_down, segment_length_m=1000, k_up=local_k_up, k_down=local_k_down)
        else:
            t_km = t_km_flat

        # Fatigue linéaire
        if local_fatigue_active and local_fatigue_rate > 0 and total_corr > 0:
            progression = d / total_corr
            t_km *= (1.0 + (local_fatigue_rate / 100.0) * progression)

        # Météo : température au passage (milieu du segment)
        passage_dt = dt_depart + timedelta(seconds=cum_time + t_km / 2.0)
        temp_at_passage = get_temp_for_datetime(hourly_temps_cache, passage_dt)

        # Appliquer modèle température non-linéaire (si activé)
        if use_temp and temp_at_passage is not None:
            mult_temp = temp_multiplier_nonlin(temp_at_passage, opt_temp=local_opt_temp, k_hot=local_k_temp_hot, k_cold=local_k_temp_cold)
            t_km *= mult_temp
        else:
            mult_temp = 1.0

        # Cumuler
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

    # résultats finaux (affichage inchangé)
    total_time_hms = seconds_to_hms(cum_time)
    st.success(f"⏱️ Temps total prévisionnel : {total_time_hms}")

    df_results = pd.DataFrame(results)
    st.subheader("📋 Détails km par km")
    st.dataframe(df_results, use_container_width=True)

    # Afficher les références et leur éventuelle température historique si on a recalibré
    if use_hist_for_refs:
        st.subheader("🔁 Références recalibrées (température historique appliquée)")
        # refs_used_for_fit n'existait pas : on affiche 'refs' (ou tu peux calculer un sous-ensemble)
        refs_used_for_fit = refs
        display_refs = []
        for r in refs_used_for_fit:
            temp_info = r.get("_temp_ref", None)
            display_refs.append({
                "Distance (m)": r.get("distance"),
                "Temps original": r.get("temps"),
                "Temp réf (°C)": round(temp_info, 1) if temp_info is not None else None,
                "Temps recalé (h:mm:ss)": r.get("temps")
            })
        st.table(pd.DataFrame(display_refs))

    # Carte + profil
    if show_map:
        st.subheader("🗺️ Carte du parcours")
        view = pdk.ViewState(latitude=df_points.lat.mean(), longitude=df_points.lon.mean(), zoom=13, pitch=0)
        path_layer = pdk.Layer("PathLayer", data=[{"path": df_points[["lon", "lat"]].values.tolist(), "name": "Parcours"}], get_path="path", get_color=[255, 0, 0], width_min_pixels=4)
        deck = pdk.Deck(map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json", initial_view_state=view, layers=[path_layer], tooltip={"text": "{name}"})
        st.pydeck_chart(deck, use_container_width=True)

        st.subheader("📊 Profil d'altitude")
        plt.figure(figsize=(10, 4))
        x_km = np.linspace(0, distance_cible_km, len(df_points))
        plt.plot(x_km, df_points["elev"])
        plt.xlabel("Distance (km)")
        plt.ylabel("Altitude (m)")
        plt.title("Profil d'altitude du parcours")
        plt.grid(alpha=0.3)
        st.pyplot(plt)

    return distance_gpx_km
# ------------------ FIN DE LA FONCTION CORRIGÉE ------------------


# ------------------ REMPLACER ICI : sections 4.1 / 4.2 boutons (corrigées) ------------------

# 4.1 PREMIER CALCUL (distance GPX)
st.subheader("4️⃣.1 Calcul initial (distance GPX)")
if st.button("🚀 Lancer l’analyse sur la distance GPX"):
    if not gpx_file:
        st.error("⚠️ Importer un fichier GPX d’abord.")
    else:
        dummy_distance = 1.0  # juste pour initialiser la distance
        distance_gpx_km = run_prediction(
            distance_cible_km=dummy_distance,
            objectif_temps_forced=None,
            show_map=True,
            use_hist_for_refs=use_hist_refs
        )
        if distance_gpx_km:
            st.session_state.first_run_done = True
            st.session_state.distance_gpx_km = distance_gpx_km

# 4.2 SECOND CALCUL (distance / temps forcés)
if st.session_state.first_run_done and st.session_state.distance_gpx_km:
    st.subheader("4️⃣.2 Ajuster distance et/ou temps objectif")

    distance_gpx_km = st.session_state.distance_gpx_km

    # Forcer la distance
    use_forced_distance = st.checkbox("Forcer la distance ?", value=False)
    distance_cible_km = distance_gpx_km
    if use_forced_distance:
        distance_cible_km = st.number_input(
            "Distance forcée (km)",
            value=float(round(distance_gpx_km, 2)),
            min_value=0.5,
            step=0.1
        )

    # Forcer le temps objectif
    use_forced_time = st.checkbox("Forcer un temps objectif ?", value=False)
    objectif_temps_forced = None
    if use_forced_time:
        objectif_temps_forced = st.text_input("Temps objectif (h:mm:ss)", value="0:17:30")

    # Bouton pour lancer la prédiction finale
    if st.button("📊 Calculer prédiction finale"):
        run_prediction(
            distance_cible_km=distance_cible_km,
            objectif_temps_forced=objectif_temps_forced,
            show_map=True,
            use_hist_for_refs=use_hist_refs
        )
