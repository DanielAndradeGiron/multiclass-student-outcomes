"""Escenarios temporales (v3.0): integridad temporal, reserva idéntica y cohortes inferidas."""

import numpy as np
import pytest

from eduoutcomes.config import load_config
from eduoutcomes.constants import (MACRO, SCENARIOS, SEM1_PREFIX, SEM2_PREFIX, TIMING_UNCERTAIN,
                                   assert_temporal_integrity, scenario_predictors)
from eduoutcomes.data import load_dataset


def _cfg(ci_config, scenario):
    d = ci_config.to_dict()
    d["scenario"] = scenario
    return type(ci_config)(**d)


def test_scenario_predictor_sets_are_nested_and_temporal(ci_config):
    ds = load_dataset(ci_config)
    sets = {s: scenario_predictors(ds.all_pred, s) for s in SCENARIOS}
    assert len(sets["T2"]) == 36 and len(sets["T1"]) == 30 and len(sets["T0"]) == 24 and len(sets["T0_strict"]) == 21
    assert set(sets["T0_strict"]) < set(sets["T0"]) < set(sets["T1"]) < set(sets["T2"])
    assert not [c for c in sets["T1"] if c.startswith(SEM2_PREFIX)]
    assert not [c for c in sets["T0"] if c.startswith((SEM1_PREFIX, SEM2_PREFIX))]
    assert not set(TIMING_UNCERTAIN) & set(sets["T0_strict"]) and set(MACRO) <= set(sets["T0_strict"])


def test_temporal_gate_rejects_future_predictors():
    with pytest.raises(AssertionError):
        assert_temporal_integrity(["Age at enrollment", "Curricular units 2nd sem (approved)"], "T1")
    with pytest.raises(AssertionError):
        assert_temporal_integrity(["Debtor"], "T0_strict")


def test_holdout_identical_across_scenarios(ci_config):
    dss = {s: load_dataset(_cfg(ci_config, s)) for s in SCENARIOS}
    hashes = {ds.holdout_hash() for ds in dss.values()}
    assert len(hashes) == 1
    ref = dss["T2"]
    for ds in dss.values():
        assert np.array_equal(ds.dev_pos, ref.dev_pos) and np.array_equal(ds.g_all, ref.g_all)
        assert ds.cols == scenario_predictors(ds.all_pred, ds.scenario)


def test_cohorts_inferred_for_every_record(ci_config):
    ds = load_dataset(ci_config)
    assert (ds.cohort >= 2008).all() and (ds.cohort <= 2017).all()


def test_unknown_scenario_rejected():
    with pytest.raises(AssertionError):
        load_config("configs/default.yaml", ["scenario='T3'"])


@pytest.mark.slow
def test_scenarios_compare_and_temporal(ci_config):
    from eduoutcomes.scenarios import run_scenarios
    from eduoutcomes.temporal import run_temporal

    d = ci_config.to_dict()
    d["scenarios"] = ("T0_strict", "T2")
    d["temporal_test_cohorts"] = (2017,)
    cfg = type(ci_config)(**d)
    out = run_scenarios(cfg)
    for f in ("tables/Table10_scenarios_cv.csv", "tables/Table11_scenarios_holdout.csv",
              "tables/Table12_scenario_contrasts.csv", "results_scenarios_draft.md"):
        assert (out / f).exists(), f
    assert any((out / "figures").glob("Fig8_prediction_scenarios.*"))
    t = run_temporal(cfg, n_workers=1)
    assert (t / "supplementary" / "TableS14_temporal_validation.csv").exists()
