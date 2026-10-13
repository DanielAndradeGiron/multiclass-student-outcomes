"""Calibración multiclase *post hoc* ajustada sobre probabilidades fuera de pliegue."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .constants import K
from .metrics import logp, nll, softmax

METHODS = ("uncalibrated", "temperature", "isotonic", "dirichlet")


class CalibrationMap:
    """Mapa de calibración multiclase.

    * ``temperature``: un parámetro (T); conserva la clase predicha (primario).
    * ``isotonic``: regresión isotónica uno-contra-resto, renormalizada.
    * ``dirichlet``: LR multinomial sobre log-probabilidades (*matrix scaling*, L2; Kull et al., 2019).
    """

    def __init__(self, method: str):
        if method not in METHODS:
            raise ValueError(method)
        self.method = method

    def fit(self, P, y):
        m, y = self.method, np.asarray(y).astype(int)
        if m == "temperature":
            L = logp(P)
            res = minimize_scalar(lambda lt: nll(y, softmax(L / np.exp(lt))), bounds=(-3, 3), method="bounded")
            self.T_ = float(np.exp(res.x))
        elif m == "isotonic":
            self.iso_ = [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(P[:, k], (y == k).astype(float))
                         for k in range(K)]
        elif m == "dirichlet":
            self.lr_ = LogisticRegression(C=1.0, max_iter=5000).fit(logp(P), y)
        return self

    def transform(self, P):
        m, P = self.method, np.asarray(P, float)
        if m == "uncalibrated":
            return P
        if m == "temperature":
            return softmax(logp(P) / self.T_)
        if m == "isotonic":
            Q = np.column_stack([iso.predict(P[:, k]) for k, iso in enumerate(self.iso_)])
            s = Q.sum(1, keepdims=True)
            return np.where(s > 0, Q / np.where(s > 0, s, 1.0), P)
        return self.lr_.predict_proba(logp(P))


class CVCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """Ajusta el estimador en pliegues internos → probabilidades OOF → mapas de calibración;
    después reajusta el estimador con todo el entrenamiento."""

    def __init__(self, estimator=None, method="temperature", cv=None, methods=None):
        self.estimator, self.method, self.cv, self.methods = estimator, method, cv, methods

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        oof = np.full((len(y), K), np.nan)
        for tr, va in self.cv:
            m = clone(self.estimator).fit(X.iloc[tr], y[tr])
            assert list(m.classes_) == list(range(K)), "Every class must be present in each calibration fold"
            oof[va] = m.predict_proba(X.iloc[va])
        assert np.isfinite(oof).all(), "Calibration folds must cover every training observation"
        meths = set(self.methods or ()) | {self.method}
        self.maps_ = {mm: CalibrationMap(mm).fit(oof, y) for mm in meths}
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = np.arange(K)
        return self

    def predict_proba(self, X, method=None):
        return self.maps_[method or self.method].transform(self.estimator_.predict_proba(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
