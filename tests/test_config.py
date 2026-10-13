import pytest

from eduoutcomes.config import load_config


def test_default_matches_manuscript():
    cfg = load_config("configs/default.yaml")
    assert (cfg.outer_splits, cfg.outer_repeats, cfg.inner_splits) == (5, 2, 3)
    assert cfg.calibration_primary == "temperature" and cfg.expected_counts == (1421, 794, 2209)
    assert not cfg.quick


def test_extends_and_overrides():
    cfg = load_config("configs/quick.yaml", ["n_workers=3", "capacity_levels=[0.1, 0.5]"])
    assert cfg.quick and cfg.outer_splits == 2 and cfg.n_workers == 3 and cfg.capacity_levels == (0.1, 0.5)
    assert cfg.meta_base == ("RandomForest", "ExtraTrees", "XGBoost", "LightGBM")   # heredado


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        load_config("configs/default.yaml", ["not_a_field=1"])
