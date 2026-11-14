# Dual-Mode Architecture & Socio-Economic Impact Whitepaper

> **Analog Finder — GeoCongo AI**  
> *NASA Space Apps Challenge 2026*  
> *Unified Optimal Transport Geospatial Engine for Planetary Exploration & Terrestrial Geohazard Early Warning*

---

## 1. Executive Summary: One Algorithmic Core, Dual Reading of the World

A foundational innovation of **Analog Finder** is the recognition that:

> **Analog Finder is neither strictly a space simulation platform nor strictly a humanitarian disaster tool. It is a unified geospatial optimal transport engine.**  
> What is compared — an Earth site against a lunar target, or a single territory against its own historical past — is simply a configuration parameter. The mathematical core remains identical.

```
                  ┌───────────────────────────────────────────────┐
                  │    Unified Optimal Transport Engine (W₁)     │
                  │   1st Wasserstein Earth Mover's Distance     │
                  └───────────────────────┬───────────────────────┘
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│     MODE A: PLANETARY MISSION         │   │      MODE B: GEOHAZARD SURVEILLANCE   │
│  "Does this Earth site mimic Moon?"   │   │ "Has this terrain moved dangerously?" │
├───────────────────────────────────────┤   ├───────────────────────────────────────┤
│ • Reference: Planetary Target         │   │ • Reference: Historical Baseline T₀   │
│ • Beneficiaries: NASA / Mission teams │   │ • Beneficiaries: Civil Protection     │
│ • Focus: Rover mobility, EVA, ISRU    │   │ • Focus: Landslides, Volcanic Crises  │
│ • Key sites: Virunga, Hawaii, Atacama │   │ • Key zones: Bukavu (Kadutu), Virunga │
└───────────────────────────────────────┘   └───────────────────────────────────────┘
```

---

## 2. Mathematical Foundation: Optimal Transport

### 2.1 The Earth Mover's Distance ($W_1$)

For two one-dimensional empirical probability distributions $P$ and $Q$, the first Wasserstein distance is defined as:

$$W_1(P, Q) = \int_{-\infty}^{+\infty} |F_P(x) - F_Q(x)|\, dx$$

Where $F_P$ and $F_Q$ are cumulative distribution functions (CDFs). 

- **Physical Meaning**: $W_1$ represents the minimal physical work (mass $\times$ distance) required to morph distribution $P$ into distribution $Q$.
- **In Mode A (Analog)**: Measures how much work is required to transform the Earth site's slope, roughness, or mineralogy into the target lunar profile.
- **In Mode B (Temporal)**: Measures the **energy of physical deformation** undergone by the territory between baseline $T_0$ and surveillance $T_1$.

---

## 3. Methodological Rigor: Solving Real-World Geospatial Challenges

### 3.1 Resolving the Spatial Blind Spot: 2D Patch-Wise Heatmaps
A global 1D distribution can conceal localized mass movements (e.g. an active landslide toe or fresh volcanic fissure). 

To resolve this limitation, Mode B implements **localized metric patch differencing** (`patch_wasserstein_grid`):
1. Multi-temporal rasters are partitioned into sliding blocks ($500\,\text{m} \times 500\,\text{m}$).
2. $W_1$ is computed patch-by-patch across corresponding spatial coordinates.
3. Produces a georeferenced **2D critical deformation heatmap**, accurately isolating active scarp heads and debris flow tracks.

### 3.2 Filtering Sensor Noise: Two-Sample Kolmogorov-Smirnov Test
Satellite observations exhibit minor reflectance and elevation fluctuations due to seasonal sun angles and atmospheric noise. Every temporal indicator is subjected to hypothesis testing:
- $H_0$: $P_{T_0} = P_{T_1}$ (no true ground displacement)
- Reject $H_0$ if $p\text{-value} < \alpha$ ($\alpha = 0.05$). Indicators failing this test are classified as **STABLE**.

---

## 4. Socio-Economic Impact Case Studies (East African Rift System)

### Case Study 1: Catastrophic Landslides in Bukavu (Kadutu & Funu Valley)
- **Context**: Bukavu (South Kivu, DRC) is built on steep volcanic terrain. Uncontrolled informal urbanization and heavy tropical rains trigger chronic rotational landslides and gully erosion.
- **Analog Finder Application**:
  - Baseline $T_0$ (2018 dry season) vs Surveillance $T_1$ (2024 post-rainy season).
  - Detected critical slope steepening ($\Delta\text{Slope} > +4.2^\circ$) and vegetative cover loss ($\Delta\text{NDVI} < -0.35$).
  - **Decision-Support Output**: Automated **WARNING / CRITICAL** alert trigger recommending immediate drainage civil works and building permit moratoria.

### Case Study 2: Fissure Eruption & Lava Reshaping at Mount Nyiragongo (May 2021)
- **Context**: On May 22, 2021, Mount Nyiragongo erupted via flank fissures, sending ultra-fluid basaltic lava towards Goma and Lake Kivu.
- **Analog Finder Application**:
  - Pre-eruption $T_0$ (April 2021) vs Post-eruption $T_1$ (June 2021).
  - Detected major surface basalt expansion ($\Delta\text{Basalt} > +45\%$) and elevation changes ($\Delta\text{DEM} > +8\,\text{m}$).
  - **Dual-Use Synergies**: While the newly formed lava flows serve as fresh lunar analog surfaces for space testing (Mode A), they simultaneously track infrastructural burial and hazard zones for disaster response (Mode B).

---

## 5. Decision-Support Action Matrix

| Alert Level | Criticality Index | Geomorphic Status | Mandatory Municipal Action |
|:---|:---|:---|:---|
| 🟢 **STABLE** | $< 0.20$ | Normal seasonal variation | Routine quarterly satellite monitoring |
| 🟡 **WATCH** | $0.20 - 0.40$ | Creeping surface displacement | High-frequency InSAR and drone survey scheduling |
| 🟠 **WARNING** | $0.40 - 0.65$ | Substantial scarp destabilization | Restrict construction permits; reinforce runoff drainage |
| 🔴 **CRITICAL** | $\ge 0.65$ | Severe structural collapse risk | Immediate field inspection & evacuation readiness |

---

## 6. Enterprise REST API Integration

The platform provides an asynchronous, cloud-ready REST API (`src/analog_finder/api/`):
- `POST /v1/analyze`: Unified analysis endpoint supporting `mode="analog"` and `mode="temporal"`.
- `GET /v1/temporal-pairs`: Active surveillance zones with baseline observation intervals.
- `GET /v1/health`: System status and engine capabilities.
- **Swagger Documentation**: `http://localhost:8000/docs`
