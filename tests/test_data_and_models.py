import numpy as np

from eduoutcomes.constants import CLASSES, K
from eduoutcomes.data import load_dataset
from eduoutcomes.models import ModelFactory, ResampledPipeline, gsplits


def test_holdout_is_locked_and_grouped(ci_config):
    ds = load_dataset(ci_config)
    assert set(ds.dev_ids).isdisjoint(ds.hold_ids)
    assert set(ds.g_dev).isdisjoint(ds.g_hold)
    assert len(ds.y_dev) + len(ds.y_hold) == len(ds.df)
    assert abs(len(ds.y_hold) / len(ds.df) - ci_config.test_size) < 0.03
    assert len(ds.cols) == 36 and len(ds.pred_sets["Enrolment only"]) == 24


def test_resampling_never_touches_prediction(ci_config):
    ds = load_dataset(ci_config)
    fac = ModelFactory(ci_config)
    m = fac.build("RandomForest", ds.cols, 0)
    assert isinstance(m, ResampledPipeline)
    m.set_params(estimator__n_estimators=10).fit(ds.X_dev, ds.y_dev)
    assert m.n_resampled_ >= len(ds.y_dev)                     # sobremuestreo en el ajuste
    P = m.predict_proba(ds.X_hold)
    assert P.shape == (len(ds.y_hold), K) and np.allclose(P.sum(1), 1)   # una fila por estudiante real
    assert list(m.classes_) == list(range(len(CLASSES)))


def test_meta_ensembles_reuse_identical_base_learners(ci_config):
    ds = load_dataset(ci_config)
    fac = ModelFactory(ci_config)
    for name in ("VotingSoft", "Stacking"):
        assert fac.audit_meta(fac.build(name, ds.cols, 0))
    folds = gsplits(3, ds.y_dev, ds.g_dev, 0)
    assert all(set(ds.g_dev[tr]).isdisjoint(ds.g_dev[va]) for tr, va in folds)
