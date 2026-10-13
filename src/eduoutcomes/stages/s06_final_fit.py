"""§9 — Ajuste final de todos los candidatos (solo desarrollo) y punto de congelación antes de abrir la reserva."""

from __future__ import annotations

import json
import logging
import time

import joblib
import pandas as pd

from ..constants import CLASSES, K, REF, dn
from ..models import cache_reset, short_params

log = logging.getLogger(__name__)


def run(ctx) -> None:
    cfg, ds, fac = ctx.cfg, ctx.ds, ctx.factory
    seed_final = cfg.random_state + 12000
    cache_path = ctx.paths.cache / f"final_models_{ctx.key_hash}.joblib"
    if cfg.reuse_cache and cache_path.exists():
        final, final_params, final_seconds = joblib.load(cache_path)
        log.info("Final models loaded from cache (%s)", cache_path.name)
    else:
        final, final_params, final_seconds, base_cache = {}, {}, {}, {}
        st = fac.reset_cache()
        for name in fac.model_names:
            t0 = time.perf_counter()
            final[name], final_params[name], _ = fac.fit_candidate(name, ds.X_dev, ds.y_dev, seed_final, ds.cols,
                                                                   groups=ds.g_dev, base_cache=base_cache,
                                                                   n_jobs=cfg.n_jobs)
            if name in fac.ensembles:
                base_cache[name] = final_params[name]
            final_seconds[name] = time.perf_counter() - t0
            log.info("final fit: %-20s %7.0f s", dn(name), final_seconds[name])
        log.info("Cache statistics (final fit): %s", dict(st.stats))
        cache_reset()   # libera memoria; los modelos finales conservan sus propios ajustes
        joblib.dump((final, final_params, final_seconds), cache_path, compress=3)
    for name in fac.model_names:
        assert list(final[name].classes_) == list(range(K)), "predict_proba columns must follow CLASSES"
    ctx.FINAL, ctx.FINAL_PARAMS, ctx.FINAL_SECONDS = final, final_params, final_seconds

    best = ctx.BEST
    joblib.dump(final[best], ctx.paths.models / "final_selected_model.joblib", compress=3)
    joblib.dump(final[REF], ctx.paths.models / "final_reference_lr.joblib", compress=3)
    ctx.FROZEN = {
        "selected_model": best, "reference": REF, "class_order": list(CLASSES),
        "hyperparameters": {m: json.loads(short_params(p)) for m, p in final_params.items()},
        "temperature": {m: final[m].maps_["temperature"].T_ for m in fac.model_names if "temperature" in final[m].maps_},
        "calibration_primary": cfg.calibration_primary, "resampling": cfg.resampling,
        "decision_rule": "argmax of calibrated probabilities", "frozen_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "note": "Models were serialised with joblib; reloading requires `pip install eduoutcomes` (same versions)"}
    (ctx.paths.metadata / "frozen_configuration_before_holdout.json").write_text(
        json.dumps(ctx.FROZEN, indent=2, default=str), encoding="utf-8")
