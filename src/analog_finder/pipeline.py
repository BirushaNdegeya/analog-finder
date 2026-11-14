"""Main pipeline — computes indicators, similarity scores, and writes scores.json.

Usage
-----
  # Synthetic mode (INVENTED data — for UI development and integration testing only)
  python -m analog_finder.pipeline --synthetic

  # Real mode (requires Planetary Computer network access + local rasters)
  python -m analog_finder.pipeline

Output
------
  data/processed/scores.json  — full ranking with indicators, criteria, global
  score, validity card, explanation, and validation results.

WARNING — SYNTHETIC MODE
------------------------
In synthetic mode, all site distributions are randomly generated from a fixed
baseline.  Results DO NOT reflect any real geological measurements and MUST NOT
be presented as scientific findings.  The dashboard displays a prominent yellow
warning banner when ``"synthetic": true`` is present in scores.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import yaml

from . import config as C
from . import scoring as S
from . import validation as V


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _load_yaml(name: str) -> dict:
    return yaml.safe_load((C.CONFIG_DIR / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

#: Baseline indicator values used to seed synthetic distributions.
#: Units match UNITS dict in config.py (degrees, dimensionless, mm yr⁻¹, etc.)
_SYNTHETIC_BASELINE: dict[str, float] = {
    "pente": 8.0,
    "rugosite": 3.0,
    "basalte": 0.70,
    "oxydes_fer": 1.20,
    "argiles": 1.10,
    "sol_nu": 0.70,
    "aridite": 300.0,
    "cavites": 0.02,
    "basalte_accessible": 0.50,
}


def _synthetic(sites_cfg: list[dict], seed: int = 0) -> tuple[dict, dict]:
    """Generate randomly perturbed synthetic indicator distributions.

    WARNING — ALL VALUES ARE INVENTED for interface/pipeline testing.
    Do not interpret or report these results scientifically.

    Parameters
    ----------
    sites_cfg : list of site config dicts (from sites.yaml).
    seed : random seed for reproducibility.

    Returns
    -------
    target : {indicator_id: 1-D array}
    sites  : {site_id: {indicator_id: 1-D array}}
    """
    rng = np.random.default_rng(seed)
    target = {
        k: S.samples_from_expected(v, abs(v) * 0.1 + 1e-3)
        for k, v in _SYNTHETIC_BASELINE.items()
    }
    sites: dict[str, dict] = {}
    for s in sites_cfg:
        profile = {k: v * rng.uniform(0.6, 1.5) for k, v in _SYNTHETIC_BASELINE.items()}
        # Force the negative control to be a poor analog
        if s["role"] == "negative_control":
            profile.update(
                sol_nu=0.05,
                aridite=1800.0,
                basalte=0.05,
                basalte_accessible=0.01,
                pente=2.0,
                rugosite=0.5,
            )
        sites[s["id"]] = {
            k: rng.normal(v, abs(v) * 0.15 + 1e-3, 500)
            for k, v in profile.items()
        }
    return target, sites


# ---------------------------------------------------------------------------
# Real data loaders
# ---------------------------------------------------------------------------

def _target_profile(tcfg: dict) -> dict:
    """Build the target distribution from the DEM raster and targets.yaml values.

    Indicators whose ``mean`` or ``std`` is null in targets.yaml are skipped
    (they will be missing from scoring — not set to zero).

    Parameters
    ----------
    tcfg : target configuration dict (from targets.yaml).

    Returns
    -------
    {indicator_id: 1-D float32 array}
    """
    from . import indicators as I
    import rasterio

    profile: dict = {}
    dem_path = C.ROOT / tcfg["dem_raster"]
    if dem_path.exists():
        with rasterio.open(dem_path) as src:
            profile.update(I.dem_indicators(src.read(1).astype("float32"), abs(src.transform.a)))
    else:
        print(f"[WARNING] Target DEM not found: {dem_path} — slope/roughness skipped.")

    for ind, spec in tcfg["indicators"].items():
        if spec.get("mean") is not None and spec.get("std") is not None:
            profile[ind] = S.samples_from_expected(spec["mean"], spec["std"])

    return profile


def _real_site(site: dict) -> dict:
    """Download and compute all indicators for one terrestrial site.

    Parameters
    ----------
    site : site config dict from sites.yaml (must have 'bbox' and 'id').

    Returns
    -------
    {indicator_id: 1-D float32 array}  — missing data indicators are absent.

    Raises
    ------
    Any exception from the STAC or rasterio stack is re-raised so the
    caller can catch it and continue with remaining sites.
    """
    from . import indicators as I
    from . import ingest

    bbox = site["bbox"]
    dem100 = ingest.load_dem(bbox, C.COMMON_RES_M)
    dem30 = ingest.load_dem(bbox, 30)
    s2 = ingest.load_s2_composite(bbox)
    basalt = ingest.load_local_raster(
        C.RAW_DIR / site["id"] / "basalt_mask.tif", bbox, dem100.shape
    )
    precip = ingest.load_local_raster(
        C.RAW_DIR / "climate" / "annual_precip.tif", bbox, dem100.shape
    )
    return I.compute_site_indicators(dem100, dem30, s2, basalt, precip)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(synthetic: bool) -> dict:
    """Execute the full scoring pipeline and write ``data/processed/scores.json``.

    Parameters
    ----------
    synthetic : if True, use invented data (safe for offline testing).

    Returns
    -------
    Full result dict (also written to disk as JSON).
    """
    sites_cfg = _load_yaml("sites.yaml")
    targets_cfg = _load_yaml("targets.yaml")
    target_id = sites_cfg["target"]
    tcfg = targets_cfg[target_id]

    # Build indicator distributions
    if synthetic:
        print("[WARNING] Running in SYNTHETIC mode — all values are INVENTED.")
        target, samples = _synthetic(sites_cfg["sites"])
    else:
        target = _target_profile(tcfg)
        samples: dict = {}
        for s in sites_cfg["sites"]:
            try:
                samples[s["id"]] = _real_site(s)
            except Exception as exc:
                print(f"[FAILED] {s['id']}: {exc} — site skipped, others continue.")

    # Compute similarity scores
    sims = S.indicator_similarities(target, samples)
    meta = {s["id"]: s for s in sites_cfg["sites"]}

    out_sites: list[dict] = []
    crit_by_site: dict[str, dict] = {}

    for sid, sim in sims.items():
        crit = S.criterion_scores(sim)
        crit_by_site[sid] = crit
        b = meta[sid]["bbox"]
        lat = (b[1] + b[3]) / 2
        lon = (b[0] + b[2]) / 2

        # Median values for display (site vs. target)
        values = {
            ind: {
                "site": float(np.median(samples[sid][ind])),
                "target": float(np.median(target[ind])),
                "unit": C.UNITS[ind],
            }
            for ind in sim
        }

        out_sites.append({
            "id": sid,
            "name": meta[sid]["name"],
            "role": meta[sid]["role"],
            "lat": lat,
            "lon": lon,
            "indicators": {k: round(v, 4) for k, v in sim.items()},
            "criteria": {k: (round(v, 4) if v is not None else None) for k, v in crit.items()},
            "confidence": round(S.confidence(sim), 4),
            "global": round(S.global_score(crit), 4),
            "explanation": S.explain(crit),
            "values": values,
            "validity": S.validity_card(crit),
            "processes": meta[sid].get("processes"),
            "local_data": meta[sid].get("local_data"),
        })

    # Validation
    ranks = V.ranks({s["id"]: s["global"] for s in out_sites})
    validation_result = {
        "expectations": V.check_expectations(
            ranks,
            sites_cfg["expected_top"],
            sites_cfg["negative_control"],
        ),
        "sensitivity": V.sensitivity(crit_by_site),
    }

    result = {
        "synthetic": synthetic,
        "target": {
            "id": target_id,
            "name": tcfg["name"],
            "lat": tcfg["lat"],
            "lon": tcfg["lon"],
        },
        "sites": out_sites,
        "validation": validation_result,
    }

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = C.PROCESSED_DIR / "scores.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"scores.json written ({len(out_sites)} sites, synthetic={synthetic}).\n"
        f"Path: {output_path}"
    )
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analog Finder pipeline — compute similarity scores for all candidate sites."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use INVENTED synthetic data (for UI testing only — not scientific results).",
    )
    args = parser.parse_args()
    run(args.synthetic)
