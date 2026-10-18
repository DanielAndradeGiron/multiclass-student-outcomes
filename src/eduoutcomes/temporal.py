"""Validación temporal EXPLORATORIA por cohortes inferidas (v3.0; solo conjunto de desarrollo).

La UCI no publica el año de ingreso, pero las 10 combinaciones de las variables macroeconómicas coinciden con las series
anuales de Portugal 2008–2017 (``constants.COHORT_BY_MACRO``). Con esa cohorte INFERIDA se hace una validación de origen
móvil: para cada año de prueba (por defecto 2015, 2016 y 2017) se entrena con todas las cohortes anteriores del conjunto
de DESARROLLO y se evalúa en la cohorte del año. La reserva bloqueada no se usa.

Advertencias que acompañan al resultado:

* la cohorte es inferida, no documentada;
* las variables macroeconómicas son constantes dentro de cada cohorte: en la cohorte de prueba toman valores nunca
  vistos, por lo que también se evalúa una variante SIN ellas;
* el modelo evaluado en cada escenario es el seleccionado por la CV aleatoria (no se vuelve a seleccionar);
* es un análisis declarado después de la v2.0 y no sustituye a una validación externa ni prospectiva.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import Config
from .constants import CLASSES, LABELS, MACRO, REF, SCENARIO_LABEL, SCENARIO_SHORT, SCENARIOS, dn
from .data import load_dataset
from .formatting import fmt_ms, num
from .metrics import mc_metrics
from .models import ModelFactory, cache_reset
from .paths import Paths
from .reporting import Reporter
from .scenarios import scenario_config
from .style import apply_style

log = logging.getLogger(__name__)
METS = ["f1_macro", "mcc", "roc_auc_macro", "recall_Enrolled", "brier", "cw_ece"]


def _fit_eval(task, cfg_dict, X_dev, y_dev, g_dev, cohort):
    cfg = Config(**cfg_dict)
    fac = ModelFactory(cfg)
    s, model, cols, year, variant = task
    tr, te = np.where(cohort < year)[0], np.where(cohort == year)[0]
    fac.reset_cache()
    t0 = time.perf_counter()
    cal, *_ = fac.fit_candidate(model, X_dev.iloc[tr], y_dev[tr], cfg.random_state + 50000 + year, cols,
                                groups=g_dev[tr], n_jobs=1)
    P = cal.predict_proba(X_dev.iloc[te])
    cache_reset()
    return {"scenario": s, "model": model, "variant": variant, "test_cohort": year, "n_train": len(tr), "n_test": len(te),
            "seconds": time.perf_counter() - t0, **{m: v for m, v in mc_metrics(y_dev[te], P, cfg.ece_bins).items()},
            "test_counts": json.dumps(np.bincount(y_dev[te], minlength=len(CLASSES)).tolist())}


def run_temporal(cfg: Config, n_workers: int | None = None) -> Path:
    """Requiere haber ejecutado los escenarios (para conocer el modelo seleccionado en cada uno)."""
    root = Path(cfg.output_dir)
    apply_style()
    paths = Paths(root / "temporal_validation").make()
    rep = Reporter(cfg, paths)
    tasks, sel = [], {}
    ds = None
    for s in [s for s in SCENARIOS if s in cfg.scenarios]:
        fz = root / s / "metadata" / "frozen_configuration_before_holdout.json"
        if not fz.exists():
            raise FileNotFoundError(f"Run the scenarios first: {fz} not found")
        sel[s] = json.loads(fz.read_text(encoding="utf-8"))["selected_model"]
        ds = load_dataset(scenario_config(cfg, s))
        assert (ds.cohort >= 0).all(), "Some records could not be assigned to an inferred cohort"
        cols = ds.cols
        no_macro = [c for c in cols if c not in MACRO]
        for model in dict.fromkeys([sel[s], REF]):
            for year in cfg.temporal_test_cohorts:
                tasks.append((s, model, cols, year, "With macroeconomic predictors"))
                tasks.append((s, model, no_macro, year, "Without macroeconomic predictors"))
    X_dev, y_dev, g_dev, coh = ds.X_dev, ds.y_dev, ds.g_dev, ds.cohort_dev
    n_jobs = n_workers or max(1, min(len(tasks), cfg.n_workers or cfg.max_workers))
    log.info("Temporal validation: %d fits with %d worker(s)", len(tasks), n_jobs)
    res = Parallel(n_jobs=n_jobs, verbose=10)(delayed(_fit_eval)(t, cfg.to_dict(), X_dev, y_dev, g_dev, coh)
                                              for t in tasks)
    df = pd.DataFrame(res)
    df.to_csv(paths.metadata / "temporal_validation_by_cohort.csv", index=False)

    # Referencia: CV aleatoria del mismo escenario y modelo (media ± DE sobre los pliegues externos)
    rows = []
    for (s, model, variant), g in df.groupby(["scenario", "model", "variant"], sort=False):
        fm = pd.read_csv(root / s / "metadata" / "nested_cv_fold_metrics.csv")
        fm = fm[(fm.model == model) & (fm.calibration == cfg.calibration_primary)]
        row = {"Scenario": SCENARIO_LABEL[s], "Model": dn(model) + (" (selected)" if model == sel[s] else ""),
               "Variant": variant}
        for m in METS:
            row[f"{LABELS[m]} — random CV"] = fmt_ms(fm[m].mean(), fm[m].std(ddof=1), 3 if m != "brier" else 4)
            row[f"{LABELS[m]} — by cohort"] = " / ".join(num(v, 3 if m != "brier" else 4) for v in g.sort_values(
                "test_cohort")[m])
        rows.append(row)
    yrs = ", ".join(map(str, cfg.temporal_test_cohorts))
    rep.register_table(
        pd.DataFrame(rows), "TableS14_temporal_validation", "S14",
        "Exploratory rolling-origin temporal validation by inferred enrolment cohort (development set)",
        f"For each test cohort ({yrs}, values separated by '/'), the model was re-tuned, re-trained (SMOTENC) and "
        "re-calibrated on all earlier cohorts of the development set and evaluated on the test cohort; the locked "
        "holdout was not used. Random CV: mean ± SD over the outer folds of the main analysis. Cohorts were inferred "
        "from the macroeconomic predictors (10 distinct combinations matching Portugal 2008–2017) and are not "
        "documented in the source. Macroeconomic predictors are constant within a cohort and take unseen values in the "
        "test cohort, so a variant without them is shown. Exploratory analysis added after the v2.0 protocol; it is "
        "neither an external nor a prospective validation", landscape=True, group_col="Scenario", supplementary=True)
    rep.export_all(extra_sheets={"temporal_by_cohort": df})
    log.info("Temporal validation written to %s", paths.root)
    return paths.root


__all__ = ["run_temporal", "SCENARIO_SHORT"]
