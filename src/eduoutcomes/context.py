"""Contexto compartido entre etapas: configuración, rutas, datos, fábrica de modelos y resultados intermedios."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import time
from dataclasses import asdict
from typing import Any

import pandas as pd

from .config import Config
from .models import ModelFactory
from .paths import Paths
from .reporting import Reporter

log = logging.getLogger(__name__)

TRACKED_LIBS = ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib", "imbalanced-learn", "xgboost",
                "lightgbm", "catboost", "shap", "lime", "python-docx", "openpyxl", "PyYAML")
REQUIRED_FOR_REPORTING = ("imbalanced-learn", "xgboost", "lightgbm", "catboost", "shap", "lime")


def lib_version(name: str) -> str:
    try:
        import importlib.metadata as im
        return im.version(name)
    except Exception:  # noqa: BLE001
        return "not installed"


class StageTimer:
    """Registro de la duración de cada etapa (``metadata/runtime_log.csv``)."""

    def __init__(self, paths: Paths, workers: int):
        self.paths, self.workers, self.log = paths, workers, []

    def run(self, name: str, fn, *args, **kwargs):
        log.info("▶ %s", name)
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        self.log.append({"stage": name, "seconds": round(dt, 1), "workers": self.workers})
        pd.DataFrame(self.log).to_csv(self.paths.metadata / "runtime_log.csv", index=False)
        log.info("✔ %s: %.1f min", name, dt / 60)
        return out


class Context:
    """Estado del *pipeline*. Cada etapa lee lo que necesita y añade sus resultados como atributos."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.paths = Paths(cfg.output_path).make()
        self.rep = Reporter(cfg, self.paths)
        self.factory = ModelFactory(cfg)
        self.n_workers = int(cfg.n_workers or max(1, min(os.cpu_count() or 1, cfg.max_workers)))
        self.timer = StageTimer(self.paths, self.n_workers)
        self.versions = self._record_versions()
        self.results: dict[str, Any] = {}

    # ------------------------------------------------------------------------------------------
    def _record_versions(self) -> dict[str, Any]:
        v = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
             "workers": self.n_workers, **{k: lib_version(k) for k in TRACKED_LIBS}, "quick_mode": self.cfg.quick}
        meta = self.paths.metadata
        (meta / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2, default=str), encoding="utf-8")
        (meta / "versions_runtime.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
        (meta / "requirements_lock.txt").write_text("\n".join(
            f"{k}=={val}" for k, val in v.items() if k in TRACKED_LIBS and val != "not installed"), encoding="utf-8")
        return v

    def check_libraries(self) -> list[str]:
        missing = [k for k in REQUIRED_FOR_REPORTING if self.versions[k] == "not installed"]
        if missing and not (self.cfg.quick or self.cfg.allow_missing_libraries):
            raise ImportError(f"Required packages missing for a reportable run: {missing}. "
                              "Install them with `pip install -e .`")
        if missing:
            log.warning("Missing libraries (structural test only): %s", missing)
        return missing

    def cache_key(self) -> tuple[str, str]:
        """Clave de caché: todo lo que puede alterar los resultados (configuración, rejillas, datos, librerías)."""
        keys = ("pipeline_version", "scenario", "outer_splits", "outer_repeats", "inner_splits", "search_n_iter", "tuning_scoring",
                "calibration_primary", "calibration_methods", "calibration_splits", "resampling", "stacking_splits",
                "min_category_frequency", "random_state", "test_size", "group_by_predictor_profile", "meta_base",
                "ece_bins", "quick")
        key = json.dumps({k: v for k, v in asdict(self.cfg).items() if k in keys}
                         | {"models": self.factory.model_names, "meta_base_used": self.factory.meta_base,
                            "n": len(self.ds.y_dev), "cols": self.ds.cols, "params": self.factory.params, "data_sha256": self.ds.data_sha,
                            "libs": {k: self.versions[k] for k in ("scikit-learn", "imbalanced-learn", "xgboost",
                                                                    "lightgbm", "catboost")}},
                         sort_keys=True, default=str)
        return key, hashlib.sha256(key.encode()).hexdigest()[:12]
