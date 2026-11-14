"""Unit tests for the similarity scoring engine and scientific validation."""
import numpy as np
from analog_finder import scoring as S
from analog_finder import validation as V


def test_identical_distribution_scores_one():
    """Verify that an identical distribution yields 1.0, and farthest yields 0.0."""
    t = np.linspace(0, 1, 100)
    sims = S.indicator_similarities({"pente": t}, {"a": {"pente": t}, "b": {"pente": t + 5}})
    assert sims["a"]["pente"] == 1.0
    assert sims["b"]["pente"] == 0.0   # Farthest site in comparative panel


def test_missing_indicator_is_not_zero():
    """Verify that missing indicators are omitted rather than defaulted to zero."""
    sims = S.indicator_similarities({"pente": np.arange(10.0)}, {"a": {}})
    assert "pente" not in sims["a"]


def test_criteria_and_global():
    """Verify criteria aggregation and that missing criteria (None) are excluded from global score."""
    crit = S.criterion_scores({"pente": 1.0, "rugosite": 0.5})
    assert crit["topographie"] == 0.75
    assert crit["geologie"] is None
    # Missing criteria are cleanly ignored and weights renormalized
    assert S.global_score(crit) == 0.75


def test_validation_checks():
    """Verify rank expectation checks with known analogs and negative control."""
    rk = V.ranks({"hawaii": 0.9, "lanzarote": 0.5, "foret_congo": 0.1})
    res = V.check_expectations(rk, ["hawaii"], "foret_congo")
    assert res["passed"]
    assert res["negative_control_last"]


def test_validity_card():
    """Verify validity breakdown categorization (reproduced, partial, not evaluated, never reproducible)."""
    card = S.validity_card({"topographie": 0.9, "geologie": 0.5, "environnement": 0.1, "ressources": None})
    assert any("Topography" in x or "topographie" in x for x in card["reproduit"])
    assert any("Geology" in x or "geologie" in x for x in card["partiel"])
    assert any("Environment" in x or "environnement" in x for x in card["non_reproduit"])
    assert any("Resources" in x or "ressources" in x for x in card["non_evalue"])
    assert len(card["jamais_reproductible"]) > 0
