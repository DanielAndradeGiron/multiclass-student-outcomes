"""Arquitecturas de los candidatos: remuestreo → codificación → clasificador, metaensambles y ajuste interno.

Todo se ajusta SOLO con los datos que recibe ``fit``: ningún registro sintético llega a validación,
calibración ni reserva. La memoización (caché por proceso) solo afecta al tiempo de ejecución.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .calibration import CVCalibratedClassifier
from .config import Config
from .constants import BINARY, ENSEMBLE_NAMES, K, META_MODELS, MODEL_OFFSET, NOMINAL, REF
from .data import profile_codes


def _optional(module: str, attr: str):
    try:
        return getattr(importlib.import_module(module), attr)
    except Exception:  # noqa: BLE001 — cualquier fallo de importación deja la librería como ausente
        return None


XGBClassifier = _optional("xgboost", "XGBClassifier")
LGBMClassifier = _optional("lightgbm", "LGBMClassifier")
CatBoostClassifier = _optional("catboost", "CatBoostClassifier")
SMOTENC = _optional("imblearn.over_sampling", "SMOTENC")
RandomOverSampler = _optional("imblearn.over_sampling", "RandomOverSampler")


# =============================================================================================
# Caché por proceso (cada proceso de trabajo tiene la suya; no se serializa entre procesos)
# =============================================================================================
class _Store:
    def __init__(self):
        self.res: OrderedDict = OrderedDict()
        self.fit: OrderedDict = OrderedDict()
        self.stats: Counter = Counter()
        self.write_fit = False
        self.pinned: frozenset = frozenset()
        self.max_fit, self.max_res = 24, 64


STORE = _Store()


def cache_reset(pinned=(), max_fit: int = 24, max_res: int = 64) -> _Store:
    STORE.res.clear()
    STORE.fit.clear()
    STORE.stats.clear()
    STORE.write_fit, STORE.pinned, STORE.max_fit, STORE.max_res = False, frozenset(pinned), int(max_fit), int(max_res)
    return STORE


@contextlib.contextmanager
def fit_cache_writes(flag: bool):
    old = STORE.write_fit
    STORE.write_fit = flag
    try:
        yield
    finally:
        STORE.write_fit = old


def _lru_put(d: OrderedDict, key, value, cap: int) -> None:
    d[key] = value
    d.move_to_end(key)
    while len(d) > cap:
        d.popitem(last=False)


def data_key(X: pd.DataFrame, y) -> str:
    """Huella del contenido exacto de (X, y): mismo contenido y orden → misma huella."""
    h = hashlib.blake2b(digest_size=16)
    h.update(pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes())
    h.update(np.ascontiguousarray(np.asarray(y, dtype=np.int64)).tobytes())
    h.update("|".join(map(str, X.columns)).encode())
    return h.hexdigest()


# =============================================================================================
# Remuestreo y codificación
# =============================================================================================
class SimpleROS(BaseEstimator):
    """Sobremuestreo aleatorio mínimo (respaldo si imbalanced-learn no está instalado; solo pruebas)."""

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y)
        n_max = np.bincount(y).max()
        idx = np.concatenate([np.r_[ic, rng.choice(ic, n_max - len(ic), replace=True)]
                              for ic in (np.where(y == c)[0] for c in np.unique(y))])
        return X.iloc[idx].reset_index(drop=True), y[idx]


def make_prep(cols: list[str], min_frequency: int = 20) -> ColumnTransformer:
    nom = [c for c in cols if c in NOMINAL]
    quant = [c for c in cols if c not in NOMINAL]
    parts = [("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), quant)]
    if nom:
        parts.append(("nom", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                  sparse_output=False))]), nom))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


class ResampledPipeline(ClassifierMixin, BaseEstimator):
    """Remuestreo → codificación → clasificador, todo ajustado SOLO con los datos que recibe ``fit``.

    SMOTENC opera sobre las variables originales (nominales y binarias como categóricas), de modo que los
    registros sintéticos conservan categorías válidas; ``predict_proba`` solo transforma y nunca remuestrea.
    Optimización sin efecto en los resultados: (i) el conjunto remuestreado se memoiza por huella de los datos y
    semilla (todos los modelos de un pliegue comparten los mismos registros sintéticos: diseño pareado); (ii) los
    ajustes de los aprendices base de los metaensambles se memoizan y se reutilizan (mismo ajuste, mismas semillas).
    """

    def __init__(self, estimator=None, cols=None, resampling="smotenc", random_state=None, min_frequency=20,
                 allow_fallback=False):
        self.estimator, self.cols, self.resampling, self.random_state = estimator, cols, resampling, random_state
        self.min_frequency, self.allow_fallback = min_frequency, allow_fallback

    def _sampler(self, y):
        kind = self.resampling
        if kind in (None, "none"):
            return None
        if kind == "smotenc":
            if SMOTENC is not None:
                cat = [i for i, c in enumerate(self.cols) if c in NOMINAL or c in BINARY]
                k = int(max(1, min(5, np.bincount(y).min() - 1)))
                return SMOTENC(categorical_features=cat, k_neighbors=k, random_state=self.random_state)
            if not self.allow_fallback:
                raise ImportError("SMOTENC requires imbalanced-learn")
            return SimpleROS(self.random_state)  # solo prueba estructural
        if kind == "random_oversampling":
            return RandomOverSampler(random_state=self.random_state) if RandomOverSampler else SimpleROS(
                self.random_state)
        raise ValueError(f"Unknown resampling strategy: {kind}")

    def _restore(self, Xr):
        Xr = Xr if isinstance(Xr, pd.DataFrame) else pd.DataFrame(Xr, columns=self.cols)
        Xr = Xr[self.cols].reset_index(drop=True).copy()
        for c in self.cols:
            Xr[c] = Xr[c].astype(str).astype(object) if c in NOMINAL else Xr[c].astype(float)
        return Xr

    def _resample(self, X, y, dk):
        s = self._sampler(y)
        if s is None:
            return X.reset_index(drop=True), y
        key = (dk, tuple(self.cols), self.resampling, self.random_state)
        hit = STORE.res.get(key)
        if hit is not None:
            STORE.res.move_to_end(key)
            STORE.stats["resample_hit"] += 1
            return hit
        Xr, yr = s.fit_resample(X, y)
        out = (self._restore(Xr), np.asarray(yr).astype(int))
        _lru_put(STORE.res, key, out, STORE.max_res)
        STORE.stats["resample_miss"] += 1
        return out

    def fit(self, X, y):
        X, y = X[self.cols], np.asarray(y).astype(int)
        dk = data_key(X, y)
        pinned = type(self.estimator).__name__ in STORE.pinned
        fkey = None
        if pinned:
            fkey = (dk, joblib.hash((type(self.estimator).__name__, self.estimator.get_params(), tuple(self.cols),
                                     self.resampling, self.random_state, self.min_frequency)))
            hit = STORE.fit.get(fkey)
            if hit is not None:
                STORE.fit.move_to_end(fkey)
                STORE.stats["fit_hit"] += 1
                self.prep_, self.estimator_, self.n_resampled_ = hit
                self.classes_ = np.asarray(self.estimator_.classes_)
                return self
        Xr, yr = self._resample(X, y, dk)
        self.prep_ = make_prep(self.cols, self.min_frequency).fit(Xr)
        self.estimator_ = clone(self.estimator).fit(self.prep_.transform(Xr), yr)
        self.classes_ = np.asarray(self.estimator_.classes_)
        self.n_resampled_ = int(len(yr))
        if pinned and STORE.write_fit:
            _lru_put(STORE.fit, fkey, (self.prep_, self.estimator_, self.n_resampled_), STORE.max_fit)
            STORE.stats["fit_store"] += 1
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.prep_.transform(X[self.cols]))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class GroupedStackingClassifier(ClassifierMixin, BaseEstimator):
    """*Stacking* con probabilidades fuera de pliegue obtenidas con pliegues ESTRATIFICADOS Y AGRUPADOS POR PERFIL.
    Metaaprendiz: LR multinomial sobre las K × B probabilidades de los aprendices base."""

    def __init__(self, estimators=None, final_estimator=None, n_splits=5, random_state=None):
        self.estimators, self.final_estimator = estimators, final_estimator
        self.n_splits, self.random_state = n_splits, random_state

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        cv = list(StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.random_state)
                  .split(np.zeros(len(y)), y, profile_codes(X)))
        Z = np.full((len(y), K * len(self.estimators)), np.nan)
        with fit_cache_writes(False):  # ajustes internos de un solo uso: no se memoizan
            for j, (_, est) in enumerate(self.estimators):
                for tr, va in cv:
                    m = clone(est).fit(X.iloc[tr], y[tr])
                    assert list(m.classes_) == list(range(K)), "Every class must be present in each stacking fold"
                    Z[va, j * K:(j + 1) * K] = m.predict_proba(X.iloc[va])
        assert np.isfinite(Z).all(), "Stacking folds must cover every training observation"
        self.estimators_ = [clone(est).fit(X, y) for _, est in self.estimators]
        self.final_estimator_ = clone(self.final_estimator).fit(Z, y)
        self.classes_ = np.arange(K)
        return self

    def predict_proba(self, X):
        return self.final_estimator_.predict_proba(np.hstack([e.predict_proba(X) for e in self.estimators_]))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# =============================================================================================
# Estimadores y rejillas
# =============================================================================================
def make_estimator(name: str, seed: int):
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=1)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=1)
    if name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=seed)
    if name == REF:
        return LogisticRegression(max_iter=5000, C=1.0)  # multinomial (lbfgs) con penalización L2
    if name == "XGBoost" and XGBClassifier:
        return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", tree_method="hist", n_estimators=400,
                             max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.8, random_state=seed,
                             n_jobs=1, verbosity=0)
    if name == "LightGBM" and LGBMClassifier:
        return LGBMClassifier(objective="multiclass", n_estimators=400, learning_rate=0.05, num_leaves=15,
                              subsample_freq=1, random_state=seed, n_jobs=1, verbose=-1)
    if name == "CatBoost" and CatBoostClassifier:
        return CatBoostClassifier(loss_function="MultiClass", iterations=500, depth=6, learning_rate=0.05,
                                  random_seed=seed, verbose=False, allow_writing_files=False, thread_count=1)
    return None


_P = "estimator__"
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "RandomForest": {_P + "n_estimators": [200, 400, 600], _P + "max_depth": [None, 10, 20],
                     _P + "min_samples_leaf": [1, 2, 4], _P + "max_features": ["sqrt", "log2", 0.5]},
    "ExtraTrees": {_P + "n_estimators": [200, 400, 600], _P + "max_depth": [None, 10, 20],
                   _P + "min_samples_leaf": [1, 2, 4], _P + "max_features": ["sqrt", "log2", 0.5]},
    "GradientBoosting": {_P + "n_estimators": [100, 200, 300], _P + "learning_rate": [0.03, 0.05, 0.1],
                         _P + "max_depth": [2, 3, 4], _P + "subsample": [0.7, 0.85, 1.0]},
    "XGBoost": {_P + "n_estimators": [200, 400, 600], _P + "max_depth": [3, 4, 6],
                _P + "learning_rate": [0.03, 0.05, 0.1],
                _P + "subsample": [0.7, 0.85, 1.0], _P + "colsample_bytree": [0.6, 0.8, 1.0],
                _P + "min_child_weight": [1, 3, 5]},
    "LightGBM": {_P + "n_estimators": [200, 400, 600], _P + "learning_rate": [0.03, 0.05, 0.1],
                 _P + "num_leaves": [7, 15, 31], _P + "min_child_samples": [10, 20, 40],
                 _P + "subsample": [0.7, 0.85, 1.0], _P + "colsample_bytree": [0.6, 0.8, 1.0],
                 _P + "reg_lambda": [0.0, 1.0, 5.0]},
    "CatBoost": {_P + "iterations": [300, 500, 800], _P + "depth": [4, 6, 8], _P + "learning_rate": [0.03, 0.05, 0.1],
                 _P + "l2_leaf_reg": [1, 3, 10]},
    REF: {_P + "C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
}


def gsplits(n_splits: int, y, groups, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Particiones estratificadas por clase y agrupadas por perfil (reutilizables)."""
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))


def short_params(bp: dict) -> str:
    def _c(v):
        return v if isinstance(v, (int, float, str, type(None))) else str(v)
    if "base_learners" in bp:
        return json.dumps({b: {k.split("__")[-1]: _c(v) for k, v in p.items()} for b, p in bp["base_learners"].items()})
    return json.dumps({k.split("__")[-1]: _c(v) for k, v in bp.items()})


# =============================================================================================
# Fábrica de modelos (picklable: se envía a los procesos de trabajo)
# =============================================================================================
@dataclass
class ModelFactory:
    cfg: Config
    ensembles: list[str] = field(init=False)
    meta_base: list[str] = field(init=False)
    params: dict[str, dict[str, list]] = field(init=False)

    def __post_init__(self):
        self.ensembles = [m for m in ENSEMBLE_NAMES if make_estimator(m, self.cfg.random_state) is not None]
        mb = [b for b in self.cfg.meta_base if b in self.ensembles]
        if len(mb) < 3:  # respaldo solo si faltan librerías (pruebas estructurales)
            mb = [b for b in ["RandomForest", "ExtraTrees", "GradientBoosting"] if b in self.ensembles]
        self.meta_base = mb
        self.params = {k: dict(v) for k, v in PARAM_GRIDS.items()}
        if self.cfg.quick:  # solo prueba estructural: árboles mucho más pequeños
            for g in self.params.values():
                for k in [k for k in g if k.endswith(("n_estimators", "iterations"))]:
                    g[k] = [20, 40]

    # --- listas de modelos -------------------------------------------------------------------
    @property
    def model_names(self) -> list[str]:
        """Los individuales primero: los metaensambles reutilizan sus ajustes."""
        return self.ensembles + META_MODELS + [REF]

    @property
    def candidates(self) -> list[str]:
        return self.ensembles + META_MODELS

    @property
    def pinned(self) -> frozenset:
        return frozenset(type(make_estimator(b, self.cfg.random_state)).__name__ for b in self.meta_base)

    @property
    def fit_cache_n(self) -> int:
        return max(self.cfg.fit_cache_size, (self.cfg.calibration_splits + 1) * len(self.meta_base) + 4)

    def reset_cache(self) -> _Store:
        return cache_reset(self.pinned, self.fit_cache_n, self.cfg.resample_cache_size)

    # --- construcción ------------------------------------------------------------------------
    def _rp(self, name, cols, seed, resampling):
        return ResampledPipeline(make_estimator(name, seed + MODEL_OFFSET[name]), cols, resampling, seed,
                                 min_frequency=self.cfg.min_category_frequency,
                                 allow_fallback=self.cfg.quick or self.cfg.allow_missing_libraries)

    def build(self, name: str, cols: list[str], seed: int, resampling: str | None = None,
              base_params: dict | None = None):
        """seed = semilla del bloque. El remuestreo usa la semilla del bloque (datos sintéticos compartidos por todos
        los modelos del pliegue) y cada estimador una semilla propia y fija; así un aprendiz base dentro de un
        metaensamble es idéntico al mismo modelo evaluado individualmente."""
        resampling = self.cfg.resampling if resampling is None else resampling
        if name in META_MODELS:
            base_params = base_params or {}
            est = []
            for b in self.meta_base:
                rp = self._rp(b, cols, seed, resampling)
                if base_params.get(b):
                    rp.set_params(**base_params[b])
                est.append((b, rp))
            if name == "VotingSoft":
                return VotingClassifier(est, voting="soft", n_jobs=1)
            return GroupedStackingClassifier(est, final_estimator=LogisticRegression(max_iter=5000, C=1.0),
                                             n_splits=self.cfg.stacking_splits, random_state=seed + 24)
        return self._rp(name, cols, seed, resampling)

    def audit_meta(self, model) -> bool:
        """Cada aprendiz base tiene su propio remuestreo y codificación; el stacking usa pliegues agrupados."""
        for nm, est in model.estimators:
            assert isinstance(est, ResampledPipeline), nm
            assert est.resampling == self.cfg.resampling, nm
        if isinstance(model, GroupedStackingClassifier):
            assert model.n_splits >= 2 and getattr(model.final_estimator, "class_weight", None) is None
        return True

    # --- ajuste ------------------------------------------------------------------------------
    def tune(self, name, X, y, seed, cols, resampling=None, groups=None, n_jobs=None):
        """Búsqueda aleatoria interna con las MISMAS particiones internas para todos los modelos del bloque.
        Devuelve el estimador SIN ajustar con los mejores hiperparámetros."""
        n_jobs = self.cfg.n_jobs if n_jobs is None else n_jobs
        model, grid = self.build(name, cols, seed, resampling), self.params[name]
        cv = gsplits(self.cfg.inner_splits, y, groups, seed + 1)
        n_iter = min(self.cfg.search_n_iter, math.prod(len(v) for v in grid.values()))
        search = RandomizedSearchCV(model, grid, n_iter=n_iter,
                                    scoring=self.cfg.tuning_scoring, cv=cv,
                                    random_state=seed + MODEL_OFFSET[name] + 2, n_jobs=n_jobs, refit=False,
                                    error_score="raise")
        search.fit(X, y)
        return clone(model).set_params(**search.best_params_), search.best_params_, search.best_score_

    def fit_candidate(self, name, X, y, seed, cols, resampling=None, groups=None, base_cache=None, n_jobs=None):
        """Ajusta (o, si es un metaensamble, construye con los hiperparámetros de sus aprendices base) y calibra
        dentro de X. Devuelve (clasificador calibrado, hiperparámetros, F1 macro interno)."""
        if name in META_MODELS:
            bp = {b: (base_cache[b] if base_cache and b in base_cache
                      else self.tune(b, X, y, seed, cols, resampling, groups, n_jobs)[1]) for b in self.meta_base}
            est, inner, params = self.build(name, cols, seed, resampling, base_params=bp), np.nan, {"base_learners": bp}
        else:
            est, params, inner = self.tune(name, X, y, seed, cols, resampling, groups, n_jobs)
        with fit_cache_writes(True):
            cal = CVCalibratedClassifier(est, method=self.cfg.calibration_primary, methods=self.cfg.calibration_methods,
                                         cv=gsplits(self.cfg.calibration_splits, y, groups, seed + 101)).fit(X, y)
        return cal, params, inner
