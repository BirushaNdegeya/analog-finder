"""Analog Finder — Dual-Mode Interactive Decision-Support Dashboard.

Provides two operational personas driven by a single Optimal Transport core:
  - Mode A: 🚀 Planetary Mission Analogs (Earth vs Moon/Mars)
  - Mode B: 🌋 Geohazard & Volcanic Risk Monitor (Bukavu landslides & Virunga volcanism)

Usage
-----
  streamlit run app/streamlit_app.py

Pre-requisite:
  python -m analog_finder.pipeline --mode all --synthetic
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import folium
from folium.plugins import Draw
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# Ensure the package is importable when launching from the app/ directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analog_finder import config as C            # noqa: E402
from analog_finder.scoring import global_score   # noqa: E402
from analog_finder import temporal as T          # noqa: E402
from analog_finder import interactive as IA      # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Analog Finder — GeoCongo AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar Persona Selector
# ---------------------------------------------------------------------------
st.sidebar.title("Operational Focus")
app_mode = st.sidebar.radio(
    "Choose Platform Persona:",
    [
        "🚀 Planetary Mission Analogs (Mode A)",
        "🌋 Geohazard & Risk Monitor (Mode B)",
    ],
    index=0,
)
st.sidebar.markdown("---")


# ===========================================================================
# PERSONA 1: MODE A — PLANETARY MISSION ANALOGS
# ===========================================================================
if "Planetary" in app_mode:
    scores_path = C.PROCESSED_DIR / "scores.json"
    if not scores_path.exists():
        st.error(
            "**scores.json not found.**  "
            "Run the pipeline first:\n\n"
            "```\npython -m analog_finder.pipeline --mode analog --synthetic\n```"
        )
        st.stop()

    data = json.loads(scores_path.read_text(encoding="utf-8"))

    st.title("🚀 Planetary Analog Finder")
    st.caption(
        f"Target: **{data['target']['name']}** — "
        "Candidate Earth sites ranked by geomorphological and geological similarity to the planetary target "
        "(relative Wasserstein metric across the candidate panel)."
    )

    if data.get("synthetic"):
        st.warning(
            "⚠️ **SYNTHETIC DATA — All values are INVENTED.**  "
            "This mode is for pipeline and interface testing only.  "
            "Do not present these rankings as scientific results.",
            icon="⚠️",
        )

    # Sidebar: Criterion weights
    st.sidebar.header("Criterion Weights")
    st.sidebar.caption("Adjust weight distribution for multi-criteria aggregation:")
    weights = {
        C.CRITERIA_LABELS.get(c, c): st.sidebar.slider(
            C.CRITERIA_LABELS.get(c, c), 0.0, 1.0, 0.25, 0.05, key=f"analog_w_{c}"
        )
        for c in C.CRITERIA
    }

    _label_to_key = {v: k for k, v in C.CRITERIA_LABELS.items()}
    internal_weights = {_label_to_key.get(lab, lab): w for lab, w in weights.items()}

    if sum(internal_weights.values()) == 0:
        st.sidebar.error("At least one criterion weight must be greater than zero.")
        st.stop()

    # Build ranked dataframe
    rows = []
    for s in data["sites"]:
        score = round(global_score(s["criteria"], internal_weights), 3)
        criteria_cols = {
            C.CRITERIA_LABELS.get(c, c): (None if v is None else round(v, 2))
            for c, v in s["criteria"].items()
        }
        rows.append({
            "id": s["id"],
            "Site": s["name"],
            "Role": s["role"],
            "Global Score": score,
            "Confidence": round(s["confidence"], 2),
            **criteria_cols,
        })

    df = pd.DataFrame(rows).sort_values("Global Score", ascending=False).reset_index(drop=True)
    df.index += 1

    # Tabs
    tab_map, tab_site, tab_val, tab_custom = st.tabs([
        "🗺️ Global Map & Ranking",
        "🔬 Site Detail & Validity Card",
        "📈 Scientific Validation",
        "🎯 Custom Live Study (Draw & Analyze)",
    ])

    with tab_map:
        left, right = st.columns([3, 2])
        with left:
            m = folium.Map(location=[10, 0], zoom_start=2, tiles=None)
            folium.TileLayer("OpenStreetMap").add_to(m)

            # Target marker (symbolic reference)
            folium.Marker(
                [data["target"]["lat"], data["target"]["lon"]],
                tooltip=f"Planetary Target: {data['target']['name']}",
                icon=folium.Icon(color="blue", icon="star"),
            ).add_to(m)

            # Candidate sites
            for s in data["sites"]:
                sc = float(df.loc[df["id"] == s["id"], "Global Score"].iloc[0])
                color = "#2e7d32" if sc >= C.STRONG else "#f9a825" if sc >= C.WEAK else "#c62828"
                folium.CircleMarker(
                    [s["lat"], s["lon"]],
                    radius=10,
                    color=color,
                    fill=True,
                    fill_opacity=0.85,
                    tooltip=f"{s['name']} (Score: {sc:.2f})",
                ).add_to(m)

            st_folium(m, height=480, width="100%")

        with right:
            st.dataframe(df.drop(columns=["id"]), use_container_width=True)
            st.caption("🟢 Strong analog (≥ 0.70) · 🟡 Partial (0.40–0.70) · 🔴 Weak (< 0.40)")

    with tab_site:
        site_name = st.selectbox("Select Candidate Site", [s["name"] for s in data["sites"]])
        s = next(x for x in data["sites"] if x["name"] == site_name)

        crit_labels = [C.CRITERIA_LABELS.get(c, c) for c in C.CRITERIA]
        crit_values = [s["criteria"].get(c) or 0.0 for c in C.CRITERIA]

        fig = go.Figure(go.Scatterpolar(
            r=crit_values + [crit_values[0]],
            theta=crit_labels + [crit_labels[0]],
            fill="toself",
            name=s["name"],
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1], tickvals=[0, 0.4, 0.7, 1.0])),
            showlegend=False,
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        c1, c2 = st.columns(2)
        c1.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown(f"### {s['name']}")
            st.caption(f"Role: **{s['role']}**")
            st.write(s["explanation"])
            st.write(f"**Data Confidence:** {s['confidence']:.0%} of indicators computed.")
            if s.get("processes"):
                st.write(f"**Testable Processes:** {s['processes']}")
            if s.get("local_data"):
                st.write(f"**Local Data:** {s['local_data']}")

        # Validity Card
        v = s["validity"]
        st.subheader("Analog Validity Card")
        k1, k2, k3 = st.columns(3)
        k1.success("✅ Reproduced: " + (", ".join(v.get("reproduit", [])) or "none"))
        k2.warning("⚠️ Partial: " + (", ".join(v.get("partiel", [])) or "none"))
        k3.error("❌ Not reproduced: " + (", ".join(v.get("non_reproduit", [])) or "none"))

        st.write(f"**Not evaluated:** {', '.join(v.get('non_evalue', [])) or 'none'}")
        st.write(f"**Never reproducible on Earth:** {', '.join(v.get('jamais_reproductible', [])) or 'none'}")
        st.write(f"**Actionable Testing Domains:** {'; '.join(v.get('usages_adaptes', [])) or 'none'}")

        # Measured values
        st.subheader("Measured Physical Distributions vs Target (Medians)")
        if s.get("values"):
            rows_val = [
                {
                    "Indicator": C.INDICATOR_LABELS.get(k, k),
                    "Site Value": round(x["site"], 3),
                    "Target Value": round(x["target"], 3),
                    "Unit": x["unit"],
                    "Score": round(s["indicators"].get(k, 0), 2),
                }
                for k, x in s["values"].items()
            ]
            st.dataframe(pd.DataFrame(rows_val), hide_index=True, use_container_width=True)

    with tab_val:
        exp = data["validation"]["expectations"]
        st.subheader("Known Analogs & Negative Control Verification")
        status = "✅ Passed" if exp["passed"] else "❌ Failed"
        st.write(f"Benchmark Outcome: **{status}** (Top {exp['top_k']} Ranking Requirement).")

        known_rows = [
            {"Site": k, "Rank": v["rank"], "Top-Tier Placement": "✅" if v["in_top"] else "❌"}
            for k, v in exp["known_analogs"].items()
        ]
        if known_rows:
            st.dataframe(pd.DataFrame(known_rows), hide_index=True)

        neg_rank = exp["negative_control_rank"]
        neg_last = exp["negative_control_last"]
        st.write(
            f"**Negative Control Rank:** {neg_rank} "
            f"({'last (rank 9) — ✅ expected' if neg_last else '⚠️ not last'})."
        )

        sens = data["validation"]["sensitivity"]
        st.subheader("Weight Sensitivity Stability (Monte Carlo Dirichlet Permutations)")
        st.write(
            f"Mean Spearman Rank Correlation: **ρ = {sens['spearman_mean']:.2f}** "
            "(high stability across diverse weighting preferences)."
        )
        if "per_site" in sens:
            st.dataframe(pd.DataFrame(sens["per_site"]).T, use_container_width=True)

    with tab_custom:
        st.subheader("🎯 Studio Analogue Personnalisé — Évaluez n'importe quelle zone sur Terre")
        st.write(
            "Ce module interactif permet à l'évaluateur de tester une **nouvelle zone d'étude personnalisée (AOI)** sur Terre "
            "face à une cible planétaire (Lune ou Mars). Vous pouvez choisir parmi des analogues emblématiques, saisir vos propres coordonnées, "
            "ou tracer directement un rectangle sur la carte satellite."
        )

        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            target_opt = st.selectbox(
                "1. Choisissez la cible planétaire :",
                [
                    "🌕 Marius Hills (Lune - Océan des Tempêtes)",
                    "🔴 Jezero Crater (Mars - Cratère d'impact & delta lacustre)",
                ],
            )
            target_key = "marius_hills" if "Marius" in target_opt else "jezero_crater"

        with col_t2:
            preset_choice = st.selectbox(
                "2. Choisissez un site terrestre pré-configuré ou personnalisez :",
                [
                    "Personnalisé (Coordonnées manuelles ou tracé carte)",
                    "🌋 Mont Etna, Italie (Stratovolcan basaltique actif)",
                    "🌋 Ol Doinyo Lengai, Tanzanie (Volcanisme alcalin unique)",
                    "🌋 Piton de la Fournaise, La Réunion (Champs de basalte & calderas)",
                    "🌋 Kilauea Caldera, Hawaii (Basalte tholéiitique & cônes)",
                    "🏜️ Erg Chebbi, Maroc (Dunes géantes & analogues éoliens martiens)",
                    "🌋 Volcan Nyamuragira, RDC (Coulées récentes & tubes de lave)",
                ],
            )

        PRESETS = {
            "🌋 Mont Etna, Italie (Stratovolcan basaltique actif)": ("Mont Etna (Italie)", (14.95, 37.70, 15.05, 37.80)),
            "🌋 Ol Doinyo Lengai, Tanzanie (Volcanisme alcalin unique)": ("Ol Doinyo Lengai (Tanzanie)", (35.88, -2.78, 35.95, -2.72)),
            "🌋 Piton de la Fournaise, La Réunion (Champs de basalte & calderas)": ("Piton de la Fournaise", (55.67, -21.28, 55.75, -21.20)),
            "🌋 Kilauea Caldera, Hawaii (Basalte tholéiitique & cônes)": ("Kilauea Caldera (Hawaii)", (-155.32, 19.38, -155.24, 19.45)),
            "🏜️ Erg Chebbi, Maroc (Dunes géantes & analogues éoliens martiens)": ("Erg Chebbi (Maroc)", (-4.05, 31.10, -3.95, 31.25)),
            "🌋 Volcan Nyamuragira, RDC (Coulées récentes & tubes de lave)": ("Nyamuragira (RDC)", (29.15, -1.45, 29.25, -1.35)),
        }

        default_name = "Custom Terrestrial AOI"
        default_bbox = (29.15, -1.60, 29.35, -1.45)
        if preset_choice in PRESETS:
            default_name, default_bbox = PRESETS[preset_choice]

        c_name = st.text_input("Nom du site candidat :", value=default_name)

        col_coords = st.columns(4)
        with col_coords[0]:
            w_in = st.number_input("Longitude Ouest (Min Lon)", value=float(default_bbox[0]), format="%.4f")
        with col_coords[1]:
            s_in = st.number_input("Latitude Sud (Min Lat)", value=float(default_bbox[1]), format="%.4f")
        with col_coords[2]:
            e_in = st.number_input("Longitude Est (Max Lon)", value=float(default_bbox[2]), format="%.4f")
        with col_coords[3]:
            n_in = st.number_input("Latitude Nord (Max Lat)", value=float(default_bbox[3]), format="%.4f")

        active_bbox = (float(w_in), float(s_in), float(e_in), float(n_in))

        st.caption("🗺️ Carte de localisation avec outil de tracé rectangle (dessinez un cadre pour ajuster la zone) :")
        center_lat = (active_bbox[1] + active_bbox[3]) / 2
        center_lon = (active_bbox[0] + active_bbox[2]) / 2
        m_draw = folium.Map(location=[center_lat, center_lon], zoom_start=9)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satellite",
        ).add_to(m_draw)
        folium.Rectangle(
            bounds=[[active_bbox[1], active_bbox[0]], [active_bbox[3], active_bbox[2]]],
            color="#2196f3",
            fill=True,
            fill_opacity=0.25,
            tooltip=c_name,
        ).add_to(m_draw)
        Draw(
            draw_options={"polyline": False, "polygon": False, "circle": False, "marker": False, "circlemarker": False, "rectangle": True},
            edit_options={"edit": False},
        ).add_to(m_draw)

        draw_output = st_folium(m_draw, height=340, width=None, key="custom_analog_map")
        if draw_output and draw_output.get("last_active_drawing"):
            geom = draw_output["last_active_drawing"].get("geometry", {})
            if geom.get("type") == "Polygon":
                coords = geom["coordinates"][0]
                dlons = [pt[0] for pt in coords]
                dlats = [pt[1] for pt in coords]
                active_bbox = (round(min(dlons), 4), round(min(dlats), 4), round(max(dlons), 4), round(max(dlats), 4))
                st.info(f"📍 Coordonnées capturées depuis la carte : BBox = `{active_bbox}`")

        col_exec, col_mode = st.columns([2, 1])
        with col_mode:
            exec_real = st.toggle("🛰️ Mode Réel (Planetary Computer STAC)", value=not data.get("synthetic", True), help="Télécharge et traite en direct les rasters Sentinel-2, Copernicus DEM et TerraClimate")
        with col_exec:
            launch_analog = st.button("🚀 Lancer l'Évaluation Analogue en Direct", type="primary", use_container_width=True)

        if launch_analog or st.session_state.get(f"res_analog_{c_name}"):
            with st.spinner("Analyse du profil planétaire et extraction des rasters satellites en cours..."):
                if launch_analog or f"res_analog_{c_name}" not in st.session_state:
                    res_a = IA.run_custom_analog_study(
                        bbox=active_bbox,
                        target_id=target_key,
                        custom_name=c_name,
                        synthetic=not exec_real,
                        weights=weights,
                    )
                    st.session_state[f"res_analog_{c_name}"] = res_a
                else:
                    res_a = st.session_state[f"res_analog_{c_name}"]

            cs = res_a["custom_site"]
            st.success(f"✅ Analyse terminée avec succès pour **{cs['name']}** contre la cible **{res_a['target']['name']}** !")

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("🏆 Rang Global", f"#{res_a['rank']} / {res_a['total_sites']} sites")
            m_col2.metric("🎯 Score d'Analogie W₁", f"{cs['global']:.4f}")
            m_col3.metric("📡 Confiance Données", f"{cs['confidence'] * 100:.1f} %")
            m_col4.metric("🪐 Cible Spatiale", res_a["target"]["name"])

            # Spider / Radar Chart
            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.markdown("#### 🕸️ Profil Comparatif Multi-Critères")
                crit_cats = [C.CRITERIA_LABELS.get(k, k) for k in C.CRITERIA.keys()]
                crit_vals = [cs["criteria"].get(k) or 0.0 for k in C.CRITERIA.keys()]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=crit_vals + [crit_vals[0]], theta=crit_cats + [crit_cats[0]], fill="toself", name=cs["name"], line_color="#1976d2"))
                fig_radar.add_trace(go.Scatterpolar(r=[1.0] * (len(crit_cats) + 1), theta=crit_cats + [crit_cats[0]], mode="lines", name="Cible Planétaire Idéale (1.0)", line=dict(dash="dash", color="#43a047")))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=360, margin=dict(l=40, r=40, t=30, b=30))
                st.plotly_chart(fig_radar, use_container_width=True)

            with c_right:
                st.markdown("#### 📋 Fiche de Validité Scientifique")
                st.caption(cs["explanation"])
                v = cs["validity"]
                if v["reproduit"]:
                    st.success("**Critères Reproduits (Fidélité Forte) :** " + ", ".join(v["reproduit"]))
                if v["partiel"]:
                    st.warning("**Critères Partiels (Dérive Modérée) :** " + ", ".join(v["partiel"]))
                if v["non_reproduit"]:
                    st.error("**Critères Non Reproduits :** " + ", ".join(v["non_reproduit"]))
                with st.expander("Paramètres non évalués ou jamais reproductibles sur Terre"):
                    st.write("**Non évalués :**", v["non_evalue"])
                    st.write("**Jamais reproductibles (Lois physiques terrestres) :**", v["jamais_reproductible"])

            # Indicators breakdown table
            st.markdown("#### 🔬 Métriques Détaillées par Indicateur")
            ind_rows = []
            for ind, vinfo in cs.get("values", {}).items():
                ind_rows.append({
                    "Indicateur": C.INDICATOR_LABELS.get(ind, ind),
                    "Valeur Site Terrestre": f"{vinfo['site']:.2f} {vinfo['unit']}",
                    "Valeur Cible Planétaire": f"{vinfo['target']:.2f} {vinfo['unit']}",
                    "Distance Wasserstein W₁": vinfo["wasserstein"],
                    "Score de Ressemblance [0 - 1]": vinfo["score"],
                })
            if ind_rows:
                st.dataframe(pd.DataFrame(ind_rows), hide_index=True, use_container_width=True)

            # Comparative table
            st.markdown("#### 🏅 Classement Intégré parmi les Sites Mondiaux")
            lb_df = pd.DataFrame(res_a["leaderboard"])
            lb_df = lb_df[["name", "role", "global", "confidence"]].rename(columns={"name": "Site", "role": "Rôle", "global": "Score Global", "confidence": "Indice Confiance"})
            lb_df.index = range(1, len(lb_df) + 1)
            st.dataframe(lb_df, use_container_width=True)


# ===========================================================================
# PERSONA 2: MODE B — GEOHAZARD & VOLCANIC RISK MONITOR
# ===========================================================================
else:
    changes_path = C.PROCESSED_DIR / "changes.json"
    if not changes_path.exists():
        st.error(
            "**changes.json not found.**  "
            "Run the multi-temporal pipeline first:\n\n"
            "```\npython -m analog_finder.pipeline --mode temporal --synthetic\n```"
        )
        st.stop()

    tdata = json.loads(changes_path.read_text(encoding="utf-8"))

    st.title("🌋 Geohazard & Volcanic Risk Monitor")
    st.caption(
        "Active multi-temporal surveillance of geomorphic deformation, volcanic fissures, and urban landslide hazards "
        "in the **East African Rift System (Bukavu & Virunga)** using Optimal Transport (Wasserstein Distance)."
    )

    if tdata.get("synthetic"):
        st.warning(
            "⚠️ **SYNTHETIC DEMO MODE — Deformation values are simulated for demonstration.** "
            "Connect local LiDAR/InSAR rasters for production deployment.",
            icon="⚠️",
        )

    # Sidebar: Hazard criteria weights
    st.sidebar.header("Hazard Criticality Weights")
    st.sidebar.caption("Fine-tune hazard sensitivity by domain:")
    t_weights = {
        C.CRITERIA_LABELS.get(c, c): st.sidebar.slider(
            C.CRITERIA_LABELS.get(c, c), 0.0, 1.0, 0.25, 0.05, key=f"temp_w_{c}"
        )
        for c in C.CRITERIA
    }

    _label_to_key = {v: k for k, v in C.CRITERIA_LABELS.items()}
    internal_t_weights = {_label_to_key.get(lab, lab): w for lab, w in t_weights.items()}

    tab_tmap, tab_tzone, tab_heatmap, tab_tcustom = st.tabs([
        "🗺️ Surveillance Map & Alert Tiers",
        "🚨 Zone Criticality & Decision Card",
        "🔬 2D Spatial Patch Differencing Simulation",
        "⚡ On-Demand Hazard Surveillance",
    ])

    with tab_tmap:
        col_m, col_t = st.columns([3, 2])
        with col_m:
            # Map centered on Kivu / Virunga region
            m_rift = folium.Map(location=[-2.0, 29.0], zoom_start=9, tiles=None)
            folium.TileLayer("OpenStreetMap").add_to(m_rift)

            # Color scheme for alerts
            alert_colors = {
                "critical": "#d32f2f", # Red
                "warning": "#f57c00",  # Orange
                "watch": "#fbc02d",    # Yellow
                "stable": "#388e3c",   # Green
            }

            for z in tdata["zones"]:
                # Re-calculate criticality with current sidebar weights
                g_crit = T.temporal_global_criticality(z["criteria"], internal_t_weights)
                alert_enum = T.classify_alert_level(g_crit)
                color = alert_colors.get(alert_enum.value, "#1976d2")

                folium.CircleMarker(
                    [z["lat"], z["lon"]],
                    radius=14,
                    color=color,
                    fill=True,
                    fill_opacity=0.85,
                    tooltip=f"<b>{z['name']}</b><br>Alert: {alert_enum.value.upper()}<br>Index: {g_crit:.2f}",
                ).add_to(m_rift)

            st_folium(m_rift, height=480, width="100%")

        with col_t:
            z_rows = []
            for z in tdata["zones"]:
                g_crit = round(T.temporal_global_criticality(z["criteria"], internal_t_weights), 3)
                alert_enum = T.classify_alert_level(g_crit)
                z_rows.append({
                    "Zone": z["name"],
                    "Hazard Type": z["hazard_type"].capitalize(),
                    "Criticality Index": g_crit,
                    "Alert Level": alert_enum.value.upper(),
                    "Authority": z.get("authority", "Civil Protection"),
                })
            df_zones = pd.DataFrame(z_rows).sort_values("Criticality Index", ascending=False)
            st.dataframe(df_zones, hide_index=True, use_container_width=True)
            st.caption(
                "🔴 CRITICAL (≥ 0.65) · 🟠 WARNING (0.40–0.65) · 🟡 WATCH (0.20–0.40) · 🟢 STABLE (< 0.20)"
            )

    with tab_tzone:
        zone_name = st.selectbox("Select Geohazard Zone for In-Depth Assessment:", [z["name"] for z in tdata["zones"]])
        z = next(x for x in tdata["zones"] if x["name"] == zone_name)

        g_crit = round(T.temporal_global_criticality(z["criteria"], internal_t_weights), 3)
        alert_enum = T.classify_alert_level(g_crit)
        v_card = T.temporal_validity_card(z["criteria"], alert_enum)

        zc1, zc2 = st.columns([1, 1])
        with zc1:
            st.markdown(f"### {z['name']}")
            st.caption(f"Hazard: **{z['hazard_type'].upper()}** | Authority: **{z.get('authority', '')}**")
            st.write(z["description"])
            st.write(f"**Baseline Window T₀:** `{z['baseline_t0']}`")
            st.write(f"**Surveillance Window T₁:** `{z['observation_t1']}`")

            # Alert Badge
            if alert_enum == T.AlertLevel.CRITICAL:
                st.error(f"🚨 **ALERT LEVEL: CRITICAL (Index: {g_crit:.2f})**")
            elif alert_enum == T.AlertLevel.WARNING:
                st.warning(f"⚠️ **ALERT LEVEL: WARNING (Index: {g_crit:.2f})**")
            elif alert_enum == T.AlertLevel.WATCH:
                st.info(f"🟡 **ALERT LEVEL: WATCH (Index: {g_crit:.2f})**")
            else:
                st.success(f"🟢 **ALERT LEVEL: STABLE (Index: {g_crit:.2f})**")

        with zc2:
            # Radar chart of criteria deformation
            crit_labels = [C.CRITERIA_LABELS.get(c, c) for c in C.CRITERIA]
            crit_vals = [z["criteria"].get(c) or 0.0 for c in C.CRITERIA]
            fig_rad = go.Figure(go.Scatterpolar(
                r=crit_vals + [crit_vals[0]],
                theta=crit_labels + [crit_labels[0]],
                fill="toself",
                line_color="#d32f2f" if g_crit >= 0.40 else "#1976d2",
                name="Deformation Index",
            ))
            fig_rad.update_layout(
                polar=dict(radialaxis=dict(range=[0, 1], tickvals=[0, 0.2, 0.4, 0.65, 1.0])),
                showlegend=False,
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_rad, use_container_width=True)

        # Decision Card
        st.subheader("Actionable Civil Protection Decision Card")
        d1, d2 = st.columns(2)
        with d1:
            st.write("**Identified Deformations & Physical Anomalies:**")
            for crit_item in v_card["divergent_or_critical"]:
                st.error(f"🔴 {crit_item}")
            for drift_item in v_card["partial_drift"]:
                st.warning(f"🟡 {drift_item}")
            for stable_item in v_card["reproduced_or_stable"]:
                st.success(f"🟢 {stable_item}")

        with d2:
            st.write("**Mandated Municipal & Protective Measures:**")
            for act in v_card["recommended_actions"]:
                st.info(f"📋 **Action:** {act}")
            st.write("**Subsurface Exclusions (Require In-Situ Geotechnics):**")
            st.caption("; ".join(v_card["not_evaluated"]))

        # Detailed indicator KS-test table
        st.subheader("Statistical Deformation Indicators (Kolmogorov-Smirnov Testing)")
        ind_table_rows = []
        for ind_id, m in z["indicators"].items():
            ind_table_rows.append({
                "Indicator": m["label"],
                "Shift (Δ Median)": f"{m['delta_median']:+.3f} {m['unit']}",
                "Wasserstein W₁": m["wasserstein_distance"],
                "Criticality Score": m["score"],
                "p-value (KS-test)": f"{m['p_value']:.4e}" if m["p_value"] < 0.001 else f"{m['p_value']:.4f}",
                "Significant (p < 0.05)": "✅ YES" if m["is_significant"] else "❌ NO (Noise)",
                "Physical Domain Interpretation": m["interpretation"],
            })
        st.dataframe(pd.DataFrame(ind_table_rows), hide_index=True, use_container_width=True)

    with tab_heatmap:
        st.subheader("🔬 Resolving Spatial Blind Spots: 2D Patch-Wise Wasserstein Heatmap")
        st.write(
            "While 1D empirical distributions quantify total deformation energy, local hazards (e.g. an active landslide toe or volcanic scarp) "
            "require spatial localization. The engine divides the territory into sliding metric patches and computes the Earth Mover's Distance patch-by-patch."
        )

        # Generate a demonstration 2D patch-wise localized anomaly
        np.random.seed(42)
        grid_dim = 16
        base_t0 = np.random.normal(12.0, 1.5, (grid_dim * 10, grid_dim * 10)).astype("float32")
        base_t1 = base_t0.copy()

        # Simulate localized slope steepening in a 4x4 patch block (e.g. Kadutu scarp)
        base_t1[50:90, 60:100] += np.random.uniform(8.0, 16.0, (40, 40)).astype("float32")

        heatmap_grid = T.patch_wasserstein_grid(base_t0, base_t1, patch_size_px=10)

        fig_heat = px.imshow(
            heatmap_grid,
            labels=dict(x="Easting Patch (x500m)", y="Northing Patch (x500m)", color="Wasserstein W₁ (m)"),
            color_continuous_scale="Reds",
            title=f"Spatialized Ground Deformation Heatmap — {z['name']}",
        )
        fig_heat.update_layout(height=450)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("🔴 Red peak denotes the active localized ground displacement zone (W₁ > 8.0 m).")

    with tab_tcustom:
        st.subheader("⚡ Surveillance Temporelle à la Demande — Détection de Déformations Libres")
        st.write(
            "Analysez l'évolution géomorphologique et spectrale d'une zone quelconque sur Terre entre deux dates satellites. "
            "Le moteur calcule les divergences statistiques de Wasserstein pour quantifier la déstabilisation de pente, l'érosion ou les coulées de lave."
        )

        t_col1, t_col2 = st.columns([1, 1])
        with t_col1:
            t_preset = st.selectbox(
                "1. Choisissez une zone d'étude ou personnalisez :",
                [
                    "Personnalisé (Coordonnées manuelles ou tracé carte)",
                    "🏚️ Bukavu - Escarpement d'Ibanda (Faille active & glissement)",
                    "🏚️ Bukavu - Ravin de Funu / Kadutu (Érosion torrentielle critique)",
                    "🌋 Volcan Nyiragongo - Flanc Sud (Éruption 2021 & champ de lave)",
                    "🌋 Volcan Nyamuragira - Caldera & cônes adventifs",
                ],
            )
            t_hazard = st.selectbox(
                "Type d'aléa prédominant :",
                ["landslide", "volcanic", "erosion", "general"],
                format_func=lambda x: {"landslide": "Glissement de terrain / Éboulement", "volcanic": "Activité volcanique / Coulée de lave", "erosion": "Érosion hydrique / Ravinement", "general": "Déformation environnementale générale"}[x],
            )

        T_PRESETS = {
            "🏚️ Bukavu - Escarpement d'Ibanda (Faille active & glissement)": ("Bukavu - Escarpement Ibanda", (28.8559, -2.5297, 28.8653, -2.5167), "2019-06-01/2019-09-01", "2024-06-01/2024-09-01"),
            "🏚️ Bukavu - Ravin de Funu / Kadutu (Érosion torrentielle critique)": ("Bukavu - Ravin Funu", (28.8399, -2.5259, 28.8484, -2.5156), "2018-06-01/2018-09-01", "2024-06-01/2024-09-01"),
            "🌋 Volcan Nyiragongo - Flanc Sud (Éruption 2021 & champ de lave)": ("Nyiragongo Flanc Sud", (29.15, -1.60, 29.35, -1.45), "2021-01-01/2021-05-15", "2021-06-01/2021-09-30"),
            "🌋 Volcan Nyamuragira - Caldera & cônes adventifs": ("Nyamuragira", (29.15, -1.45, 29.25, -1.35), "2020-01-01/2020-06-01", "2024-01-01/2024-06-01"),
        }

        default_tname = "Custom Geohazard AOI"
        default_tbbox = (28.8559, -2.5297, 28.8653, -2.5167)
        default_t0 = "2019-06-01/2019-09-01"
        default_t1 = "2024-06-01/2024-09-01"
        if t_preset in T_PRESETS:
            default_tname, default_tbbox, default_t0, default_t1 = T_PRESETS[t_preset]

        with t_col2:
            t_name = st.text_input("Nom de la zone surveillée :", value=default_tname)
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                t0_input = st.text_input("Période T0 (Référence / ISO)", value=default_t0)
            with d_col2:
                t1_input = st.text_input("Période T1 (Surveillance / ISO)", value=default_t1)

        t_coords = st.columns(4)
        with t_coords[0]:
            tw_in = st.number_input("Lon Ouest (Min Lon) ", value=float(default_tbbox[0]), format="%.4f")
        with t_coords[1]:
            ts_in = st.number_input("Lat Sud (Min Lat) ", value=float(default_tbbox[1]), format="%.4f")
        with t_coords[2]:
            te_in = st.number_input("Lon Est (Max Lon) ", value=float(default_tbbox[2]), format="%.4f")
        with t_coords[3]:
            tn_in = st.number_input("Lat Nord (Max Lat) ", value=float(default_tbbox[3]), format="%.4f")

        active_tbbox = (float(tw_in), float(ts_in), float(te_in), float(tn_in))

        # Folium draw map
        st.caption("🗺️ Carte de surveillance avec outil de tracé rectangle :")
        t_center_lat = (active_tbbox[1] + active_tbbox[3]) / 2
        t_center_lon = (active_tbbox[0] + active_tbbox[2]) / 2
        m_tdraw = folium.Map(location=[t_center_lat, t_center_lon], zoom_start=12)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satellite",
        ).add_to(m_tdraw)
        folium.Rectangle(
            bounds=[[active_tbbox[1], active_tbbox[0]], [active_tbbox[3], active_tbbox[2]]],
            color="#d32f2f",
            fill=True,
            fill_opacity=0.3,
            tooltip=t_name,
        ).add_to(m_tdraw)
        Draw(
            draw_options={"polyline": False, "polygon": False, "circle": False, "marker": False, "circlemarker": False, "rectangle": True},
            edit_options={"edit": False},
        ).add_to(m_tdraw)

        tdraw_output = st_folium(m_tdraw, height=340, width=None, key="custom_temporal_map")
        if tdraw_output and tdraw_output.get("last_active_drawing"):
            tgeom = tdraw_output["last_active_drawing"].get("geometry", {})
            if tgeom.get("type") == "Polygon":
                tcoords = tgeom["coordinates"][0]
                tdlons = [pt[0] for pt in tcoords]
                tdlats = [pt[1] for pt in tcoords]
                active_tbbox = (round(min(tdlons), 4), round(min(tdlats), 4), round(max(tdlons), 4), round(max(tdlats), 4))
                st.info(f"📍 Coordonnées capturées : BBox = `{active_tbbox}`")

        texec_col, tmode_col = st.columns([2, 1])
        with tmode_col:
            texec_real = st.toggle("🛰️ Mode Données Réelles (STAC)", value=not tdata.get("synthetic", True), key="toggle_real_temporal")
        with texec_col:
            launch_temporal = st.button("🚨 Lancer la Surveillance Temporelle à la Demande", type="primary", use_container_width=True)

        if launch_temporal or st.session_state.get(f"res_temporal_{t_name}"):
            with st.spinner("Téléchargement des scènes Sentinel-2 bi-temporelles et calcul différentiel de Wasserstein..."):
                if launch_temporal or f"res_temporal_{t_name}" not in st.session_state:
                    res_t = IA.run_custom_temporal_study(
                        bbox=active_tbbox,
                        baseline_t0=t0_input,
                        observation_t1=t1_input,
                        zone_name=t_name,
                        hazard_type=t_hazard,
                        synthetic=not texec_real,
                        weights=internal_t_weights,
                    )
                    st.session_state[f"res_temporal_{t_name}"] = res_t
                else:
                    res_t = st.session_state[f"res_temporal_{t_name}"]

            st.success(f"✅ Surveillance terminée pour **{res_t['name']}** !")
            alert_lvl = res_t["alert_level"]
            alert_labels = {"critical": "🚨 CRITIQUE", "warning": "⚠️ ATTENTION", "watch": "🟡 VIGILANCE", "stable": "🟢 STABLE"}

            tm1, tm2, tm3, tm4 = st.columns(4)
            tm1.metric("Niveau d'Alerte", alert_labels.get(alert_lvl, alert_lvl))
            tm2.metric("Indice de Criticité Global", f"{res_t['criticality_index']:.4f}")
            tm3.metric("Type d'Aléa", res_t["hazard_type"].capitalize())
            tm4.metric("Période T1 / T0", "Multi-temporel")

            # Risk validity card
            tv = res_t["validity_card"]
            st.markdown("#### 📋 Fiche de Risque & Déformations Détectées")
            if tv["divergent_or_critical"]:
                for item in tv["divergent_or_critical"]:
                    st.error(f"🚨 {item}")
            if tv["partial_drift"]:
                for item in tv["partial_drift"]:
                    st.warning(f"⚠️ {item}")
            if tv["reproduced_or_stable"]:
                for item in tv["reproduced_or_stable"]:
                    st.success(f"🟢 {item}")

            # Recommandations de sécurité civile
            if tv.get("recommended_actions"):
                st.info("**Actions Recommandées :**\n" + "\n".join(f"- {act}" for act in tv["recommended_actions"]))

            # Table KS
            st.markdown("#### 🔬 Métriques Statistiques par Indicateur (Test Kolmogorov-Smirnov)")
            tind_rows = []
            for ind, m in res_t.get("indicators", {}).items():
                tind_rows.append({
                    "Indicateur": m["label"],
                    "Déviation Médiane (Δ)": f"{m['delta_median']:+.3f} {m['unit']}",
                    "Distance Wasserstein W₁": m["wasserstein_distance"],
                    "Score de Criticité": m["score"],
                    "Significativité (p < 0.05)": "✅ OUI (Anomalie réelle)" if m["is_significant"] else "❌ NON (Bruit)",
                    "Interprétation Physique": m["interpretation"],
                })
            st.dataframe(pd.DataFrame(tind_rows), hide_index=True, use_container_width=True)
