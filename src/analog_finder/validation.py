"""Scientific validation suite: expected rank checks, negative control, and sensitivity.

NASA Space Apps Challenge 2026 - GeoCongo AI
"""
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import spearmanr

from . import config as C
from .scoring import global_score


def ranks(scores: Dict[str, float]) -> Dict[str, int]:
    """Compute 1-based ranks from highest score to lowest."""
    order = sorted(scores, key=scores.get, reverse=True)
    return {s: i + 1 for i, s in enumerate(order)}


def check_expectations(rank: Dict[str, int], expected_top: List[str], negative_control: str, top_k: Optional[int] = None) -> dict:
    """Verify that known analogs rank near the top and negative control ranks dead last."""
    n = len(rank)
    top_k = top_k or max(len(expected_top), n // 2)
    known = {s: {"rank": rank[s], "in_top": rank[s] <= top_k} for s in expected_top if s in rank}
    neg = rank.get(negative_control)
    return {
        "top_k": top_k,
        "known_analogs": known,
        "negative_control_rank": neg,
        "negative_control_last": neg == n if neg else None,
        "passed": bool(known) and all(v["in_top"] for v in known.values()) and (neg == n if neg else False),
    }


def sensitivity(criteria_by_site: Dict[str, Dict[str, Optional[float]]], n_draws: int = 500, seed: int = 0) -> dict:
    """Evaluate stability of site rankings against random Dirichlet weighting permutations."""
    rng = np.random.default_rng(seed)
    sites = list(criteria_by_site)
    base = {s: global_score(criteria_by_site[s]) for s in sites}
    base_vec = np.array([base[s] for s in sites])
    all_ranks, rhos = [], []

    for _ in range(n_draws):
        w = dict(zip(C.CRITERIA, rng.dirichlet(np.ones(len(C.CRITERIA)))))
        sc = {s: global_score(criteria_by_site[s], w) for s in sites}
        all_ranks.append([ranks(sc)[s] for s in sites])
        rho = spearmanr(base_vec, [sc[s] for s in sites]).statistic
        rhos.append(rho)

    arr = np.array(all_ranks)
    base_ranks = ranks(base)
    return {
        "spearman_mean": float(np.nanmean(rhos)),
        "per_site": {
            s: {
                "rank_base": base_ranks[s],
                "rank_min": int(arr[:, i].min()),
                "rank_max": int(arr[:, i].max()),
            }
            for i, s in enumerate(sites)
        },
    }
