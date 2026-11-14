"""Analog Finder — Interactive decision-support dashboard.

Reads ``data/processed/scores.json`` (pre-computed by the pipeline) and
renders an interactive map, similarity radar charts, analog validity cards,
and sensitivity analysis.

Usage
-----
  streamlit run app/streamlit_app.py

The pipeline must be run first:
  python -m analog_finder.pipeline --synthetic
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# Ensure the package is importable when launching from the app/ directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analog_finder import config as C            # noqa: E402
from analog_finder.scoring import global_score   # noqa: E402

# ---------------------------------------------------------------------------
# Page config & data loading
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Analog Finder — GeoCongo AI", layout="wide")

scores_path = C.PROCESSED_DIR / "scores.json"
if not scores_path.exists():
    st.error(
        "**scores.json not found.**  "
        "Run the pipeline first:\n\n"
        "```\npython -m analog_finder.pipeline --synthetic\n```"
    )
    st.stop()

data = json.loads(scores_path.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🌍 Analog Finder — GeoCongo AI")
st.caption(
    f"Target: **{data['target']['name']}** — "
    "Sites are ranked by their geomorphological and geological similarity to the target "
    "(relative index across the candidate panel)."
)

# Prominent synthetic data warning
if data["synthetic"]:
    st.warning(
        "⚠️ **SYNTHETIC DATA — All values are INVENTED.**  "
        "This mode is for pipeline and interface testing only.  "
        "Do not present these rankings as scientific results.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Sidebar — criterion weights
# ---------------------------------------------------------------------------
st.sidebar.header("Criterion Weights")
st.sidebar.caption("Adjust how each criterion contributes to the global score.")

weights: dict[str, float] = {
    C.CRITERIA_LABELS.get(c, c): st.sidebar.slider(
        C.CRITERIA_LABELS.get(c, c), 0.0, 1.0, 0.25, 0.05, key=f"w_{c}"
    )
    for c in C.CRITERIA
}

# Map label weights back to internal keys for global_score()
_label_to_key = {v: k for k, v in C.CRITERIA_LABELS.items()}
internal_weights = {_label_to_key.get(lab, lab): w for lab, w in weights.items()}

if sum(internal_weights.values()) == 0:
    st.sidebar.error("At least one criterion weight must be greater than zero.")
    st.stop()

# ---------------------------------------------------------------------------
# Build ranked dataframe (re-computed on every weight change)
# ---------------------------------------------------------------------------
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
df.index += 1  # 1-based ranking

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_map, tab_site, tab_val = st.tabs(["Map & Ranking", "Site Detail", "Validation"])

# ── Tab 1: Map & Ranking ────────────────────────────────────────────────────
with tab_map:
    left, right = st.columns([3, 2])

    with left:
        m = folium.Map(location=[10, 0], zoom_start=2, tiles=None)
        folium.TileLayer("OpenStreetMap").add_to(m)

        # Target marker — symbolic terrestrial reference coordinates
        target_tooltip = (
            f"{data['target']['name']} (symbolic reference — not geocoded on Earth)"
        )
        folium.Marker(
            [data["target"]["lat"], data["target"]["lon"]],
            tooltip=target_tooltip,
            icon=folium.Icon(color="blue", icon="star"),
        ).add_to(m)

        # Candidate sites — colour-coded by global score
        for s in data["sites"]:
            sc = float(df.loc[df["id"] == s["id"], "Global Score"].iloc[0])
            color = "#2e7d32" if sc >= C.STRONG else "#f9a825" if sc >= C.WEAK else "#c62828"
            folium.CircleMarker(
                [s["lat"], s["lon"]],
                radius=10,
                color=color,
                fill=True,
                fill_opacity=0.8,
                tooltip=f"{s['name']} : {sc:.2f}",
            ).add_to(m)

        st_folium(m, height=450, width="100%")

    with right:
        st.dataframe(df.drop(columns=["id"]), use_container_width=True)
        st.caption(
            "🟢 Strong analog (≥ 0.70) · 🟡 Partial (0.40–0.70) · 🔴 Weak (< 0.40)"
        )

# ── Tab 2: Site Detail ──────────────────────────────────────────────────────
with tab_site:
    site_name = st.selectbox("Select a site", [s["name"] for s in data["sites"]])
    s = next(x for x in data["sites"] if x["name"] == site_name)

    # Radar chart — criteria scores (replace None with 0 for display only)
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
        st.markdown(f"**{s['name']}** — *{s['role']}*")
        st.write(s["explanation"])
        st.write(f"**Data confidence:** {s['confidence']:.0%} of indicators computed.")
        if s.get("processes"):
            st.write(f"**Testable processes:** {s['processes']}")
        if s.get("local_data"):
            st.write(f"**Local data available:** {s['local_data']}")
        st.caption(
            "Score is relative to the candidate panel — not an absolute analog quality measure."
        )

    # Validity card
    v = s["validity"]
    st.subheader("Analog Validity Card")
    k1, k2, k3 = st.columns(3)
    k1.success("✅ Reproduced: " + (", ".join(v.get("reproduit", [])) or "none"))
    k2.warning("⚠️ Partial: " + (", ".join(v.get("partiel", [])) or "none"))
    k3.error("❌ Not reproduced: " + (", ".join(v.get("non_reproduit", [])) or "none"))

    not_eval = ", ".join(v.get("non_evalue", []))
    never = ", ".join(v.get("jamais_reproductible", []))
    st.write(f"**Not evaluated:** {not_eval or 'none'}")
    st.write(f"**Never reproducible on Earth:** {never or 'none'}")

    usages = "; ".join(v.get("usages_adaptes", [])) or "none convincingly"
    partial_uses = v.get("usages_partiels", [])
    usage_text = usages
    if partial_uses:
        usage_text += "  |  partially: " + "; ".join(partial_uses)
    st.write(f"**Testable use-cases here:** {usage_text}")

    # Values table
    st.subheader("Measured Values vs. Target (medians)")
    if s.get("values"):
        rows_val = [
            {
                "Indicator": C.INDICATOR_LABELS.get(k, k),
                "Site": round(x["site"], 3),
                "Target": round(x["target"], 3),
                "Unit": x["unit"],
                "Score": round(s["indicators"].get(k, 0), 2),
            }
            for k, x in s["values"].items()
        ]
        st.dataframe(pd.DataFrame(rows_val), hide_index=True, use_container_width=True)

# ── Tab 3: Validation ───────────────────────────────────────────────────────
with tab_val:
    exp = data["validation"]["expectations"]
    st.subheader("Known Analogs & Negative Control")
    status = "✅ Passed" if exp["passed"] else "❌ Failed"
    st.write(f"Validation: **{status}** (expected top {exp['top_k']} sites).")

    known_rows = [
        {
            "Site": k,
            "Rank": v["rank"],
            "In top-k": "✅" if v["in_top"] else "❌",
        }
        for k, v in exp["known_analogs"].items()
    ]
    if known_rows:
        st.dataframe(pd.DataFrame(known_rows), hide_index=True)

    neg_rank = exp["negative_control_rank"]
    neg_last = exp["negative_control_last"]
    st.write(
        f"**Negative control rank:** {neg_rank} "
        f"({'last — ✅ as expected' if neg_last else '⚠️ not last'})."
    )

    sens = data["validation"]["sensitivity"]
    st.subheader("Weight Sensitivity (Spearman ρ)")
    st.write(
        f"Mean Spearman correlation with baseline ranking under random weight permutations: "
        f"**{sens['spearman_mean']:.2f}**"
    )
    st.caption(
        "A high ρ (> 0.80) means the ranking is stable regardless of how criteria are weighted."
    )
    if "per_site" in sens:
        st.dataframe(pd.DataFrame(sens["per_site"]).T, use_container_width=True)
