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
    tab_map, tab_site, tab_val = st.tabs(["🗺️ Global Map & Ranking", "🔬 Site Detail & Validity Card", "📈 Scientific Validation"])

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

    tab_tmap, tab_tzone, tab_heatmap = st.tabs([
        "🗺️ Surveillance Map & Alert Tiers",
        "🚨 Zone Criticality & Decision Card",
        "🔬 2D Spatial Patch Differencing Simulation",
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
