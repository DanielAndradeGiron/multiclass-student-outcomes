import numpy as np
import pytest

from eduoutcomes.metrics import (clopper, ece_binary, logistic_recalibration, mc_metrics, nll, reliability_q,
                                 verify_metrics)


def test_vectorised_metrics_match_sklearn():
    assert verify_metrics(n_trials=30) < 1e-9


def test_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    P = np.eye(3)[y]
    m = mc_metrics(y, P)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["mcc"] == pytest.approx(1.0)
    assert m["brier"] == pytest.approx(0.0)
    assert m["cw_ece"] == pytest.approx(0.0, abs=1e-12)


def test_ece_of_calibrated_constant_is_zero():
    y = np.array([1] * 30 + [0] * 70)
    assert ece_binary(y, np.full(100, 0.3)) == pytest.approx(0.0)


def test_logistic_recalibration_ideal_on_well_calibrated_data():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(float)
    a, b = logistic_recalibration(y, p)
    assert abs(a) < 0.05 and abs(b - 1) < 0.08


def test_nll_and_clopper_and_reliability():
    assert nll(np.array([0]), np.array([[1.0, 0.0, 0.0]])) == pytest.approx(0.0, abs=1e-5)
    lo, hi = clopper(5, 10)
    assert 0 < lo < 0.5 < hi < 1
    r = reliability_q(np.r_[np.zeros(50), np.ones(50)], np.linspace(0, 1, 100), bins=5)
    assert r.shape[1] == 5 and (r[:, 3] <= r[:, 1]).all() and (r[:, 1] <= r[:, 4]).all()
