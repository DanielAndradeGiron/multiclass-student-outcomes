import numpy as np
import pandas as pd

from eduoutcomes.inference import bayes_rope, friedman_iman_davenport, holm, nb_test, verdict


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert np.all(adj <= 1) and np.all(adj >= [0.01, 0.04, 0.03, 0.5])
    assert adj[0] == 0.04


def test_nb_test_widens_with_correction():
    d = np.array([0.02, 0.03, 0.01, 0.025, 0.015, 0.02, 0.03, 0.01, 0.02, 0.025])
    _, lo0, hi0, _, _ = nb_test(d, 0.0)
    _, lo1, hi1, _, _ = nb_test(d, 0.25)
    assert (hi1 - lo1) > (hi0 - lo0)


def test_bayes_rope_sums_to_one_and_verdicts():
    pw, pr, pb = bayes_rope(np.array([0.05, 0.06, 0.04, 0.05, 0.055]), 0.01, 0.25)
    assert abs(pw + pr + pb - 1) < 1e-9 and pb > 0.9
    assert verdict(0.05, 0.03, 0.07, 0.001, 0.01) == "superior (practically relevant)"
    assert verdict(0.0, -0.005, 0.005, 0.9, 0.01) == "practically equivalent"


def test_friedman_ranks():
    pv = pd.DataFrame({"a": [0.9, 0.8, 0.85, 0.9], "b": [0.5, 0.4, 0.45, 0.5], "c": [0.7, 0.6, 0.65, 0.7]})
    fr = friedman_iman_davenport(pv)
    assert fr["ranks"]["a"] == 1 and fr["ranks"]["b"] == 3 and fr["cd"] > 0
