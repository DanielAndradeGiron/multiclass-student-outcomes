"""§6 — Validación cruzada anidada repetida (bloques externos en paralelo y reanudables)."""

from __future__ import annotations

import logging
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..constants import CLASSES, META_MODELS
from ..metrics import mc_metrics
from ..models import cache_reset, gsplits, short_params

log = logging.getLogger(__name__)


def run_block(k, tr, va, X_dev, y_dev, g_dev, dev_ids, cols, factory, key, path):
    """Un bloque externo completo: ajuste, calibración y evaluación de todos los modelos (secuencial en el proceso).

    Es una función de módulo con argumentos explícitos para que los procesos de trabajo no dependan de estado global.
    """
    cfg = factory.cfg
    t_block = time.perf_counter()
    st = factory.reset_cache()
    seed = cfg.random_state + 1000 * k
    rep, fold = k // cfg.outer_splits + 1, k % cfg.outer_splits + 1
    Xtr, ytr, gtr, Xva, yva = X_dev.iloc[tr], y_dev[tr], g_dev[tr], X_dev.iloc[va], y_dev[va]
    rows, prm, oofs, base_cache, tune_s = [], [], [], {}, {}
    for name in factory.model_names:
        t0 = time.perf_counter()
        cal, bp, inner = factory.fit_candidate(name, Xtr, ytr, seed, cols, groups=gtr, base_cache=base_cache, n_jobs=1)
        fit_s = time.perf_counter() - t0
        if name in factory.ensembles:
            base_cache[name], tune_s[name] = bp, fit_s
        if name in META_MODELS:  # el costo incluye el ajuste de sus aprendices base
            fit_s += sum(tune_s.get(b, 0.0) for b in factory.meta_base)
        for meth in cfg.calibration_methods:
            Pv = cal.predict_proba(Xva, method=meth)
            rows.append({"block": k + 1, "repeat": rep, "fold": fold, "model": name, "calibration": meth,
                         "fit_s": fit_s, **mc_metrics(yva, Pv, cfg.ece_bins)})
            if meth in (cfg.calibration_primary, "uncalibrated"):
                oofs.append(pd.DataFrame({"block": k + 1, "model": name, "calibration": meth, "row_id": dev_ids[va],
                                          "observed": yva, **{f"p_{c}": Pv[:, i] for i, c in enumerate(CLASSES)}}))
        prm.append({"Block": k + 1, "Model": name, "Inner macro-F1": None if np.isnan(inner) else round(inner, 4),
                    "Temperature (T)": round(cal.maps_["temperature"].T_, 3) if "temperature" in cal.maps_ else None,
                    "Selected hyperparameters": short_params(bp)})
    res = {"key": key, "rows": rows, "prm": prm, "oof": pd.concat(oofs, ignore_index=True),
           "cache_stats": dict(st.stats), "seconds": time.perf_counter() - t_block}
    cache_reset()
    joblib.dump(res, path)
    return k, res["seconds"], res["cache_stats"]


def run(ctx) -> None:
    cfg, ds, fac = ctx.cfg, ctx.ds, ctx.factory
    y_dev, g_dev = ds.y_dev, ds.g_dev
    splits = [sp for r in range(cfg.outer_repeats)
              for sp in gsplits(cfg.outer_splits, y_dev, g_dev, cfg.random_state + 100 + r)]
    n_blocks = len(splits)
    key, key_hash = ctx.cache_key()
    ctx.key_hash = key_hash
    block_dir = ctx.paths.cache / "cv_blocks"
    block_dir.mkdir(parents=True, exist_ok=True)

    def block_path(k):
        return block_dir / f"block{k + 1:02d}_{key_hash}.joblib"

    def cached(k):
        return cfg.reuse_cache and block_path(k).exists() and joblib.load(block_path(k))["key"] == key

    todo = [k for k in range(n_blocks) if not cached(k)]
    n_jobs = min(ctx.n_workers, max(1, len(todo)))
    log.info("%d block(s) loaded from cache; %d to compute with %d worker(s)", n_blocks - len(todo), len(todo), n_jobs)
    if todo:
        X_dev, dev_ids = ds.X_dev, ds.dev_ids
        info = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_block)(k, *splits[k], X_dev, y_dev, g_dev, dev_ids, ds.cols, fac, key, block_path(k))
            for k in todo)
        for k, sec, cs in sorted(info):
            log.info("  block %2d: %5.1f min | cache %s", k + 1, sec / 60, cs)

    blocks = [joblib.load(block_path(k)) for k in range(n_blocks)]
    ctx.n_blocks = n_blocks
    ctx.fold_df = pd.DataFrame([r for b in blocks for r in b["rows"]])
    ctx.param_df = pd.DataFrame([r for b in blocks for r in b["prm"]])
    ctx.oof_df = pd.concat([b["oof"] for b in blocks], ignore_index=True)
    ctx.cv_cache_stats = pd.DataFrame([b["cache_stats"] for b in blocks]).sum().to_dict()
    ctx.cv_block_seconds = [b["seconds"] for b in blocks]
    ctx.FP = ctx.fold_df[ctx.fold_df.calibration == cfg.calibration_primary].copy()   # calibración primaria

    prim = ctx.oof_df[ctx.oof_df.calibration == cfg.calibration_primary]
    ctx.validation_counts = prim.groupby(["block", "model"]).size().unstack()
    assert ctx.validation_counts.nunique(axis=1).eq(1).all(), "Models did not share validation observations"
    assert ctx.oof_df.groupby(["model", "calibration"]).size().eq(len(y_dev) * cfg.outer_repeats).all(), \
        "Incomplete repeated OOF"
    pc = ctx.oof_df[[f"p_{c}" for c in CLASSES]].to_numpy()
    assert np.allclose(pc.sum(1), 1, atol=1e-6) and (pc >= 0).all(), "Invalid probability vectors"
    ctx.oof_df.to_csv(ctx.paths.metadata / "nested_cv_oof_predictions.csv.gz", index=False)
    log.info("\n%s", ctx.FP.groupby("model")[["f1_macro", "mcc", "roc_auc_macro", "brier", "cw_ece"]].mean().round(3))
