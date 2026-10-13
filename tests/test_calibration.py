import numpy as np

from eduoutcomes.calibration import CalibrationMap
from eduoutcomes.metrics import nll


def _overconfident(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 1.0
    P = np.exp(4 * logits)          # sobreconfianza deliberada
    return y, P / P.sum(1, keepdims=True)


def test_temperature_preserves_argmax_and_improves_nll():
    y, P = _overconfident()
    cm = CalibrationMap("temperature").fit(P, y)
    Q = cm.transform(P)
    assert (Q.argmax(1) == P.argmax(1)).all()
    assert cm.T_ > 1
    assert nll(y, Q) < nll(y, P)


def test_all_methods_return_valid_probabilities():
    y, P = _overconfident()
    for m in ("uncalibrated", "temperature", "isotonic", "dirichlet"):
        Q = CalibrationMap(m).fit(P, y).transform(P)
        assert np.allclose(Q.sum(1), 1) and (Q >= 0).all(), m
