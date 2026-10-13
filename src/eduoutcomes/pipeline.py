"""Orquestación de las etapas del estudio, en el orden preespecificado."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable

from .config import Config
from .context import Context
from .metrics import verify_metrics
from .stages import (s01_data, s02_describe, s03_nested_cv, s04_rq1_comparison, s05_rq2_calibration, s06_final_fit,
                     s07_holdout, s08_decision, s09_subgroups, s10_sensitivity, s11_explainability, s12_export)
from .style import apply_style

log = logging.getLogger(__name__)

# (identificador, nombre en el registro de tiempos, función)
STAGES: list[tuple[str, str, Callable]] = [
    ("data", "Data audit and holdout lock", s01_data.run),
    ("describe", "Sample description and design", s02_describe.run),
    ("cv", "Repeated nested cross-validation", s03_nested_cv.run),
    ("rq1", "RQ1 statistical comparison", s04_rq1_comparison.run),
    ("rq2", "RQ2 calibration in cross-validation", s05_rq2_calibration.run),
    ("final", "Final fit on the development set", s06_final_fit.run),
    ("holdout", "Holdout evaluation", s07_holdout.run),
    ("decision", "Decision-oriented analysis", s08_decision.run),
    ("subgroups", "Subgroup analysis", s09_subgroups.run),
    ("sensitivity", "Sensitivity analyses", s10_sensitivity.run),
    ("explain", "Explainability", s11_explainability.run),
    ("export", "Integrity checks and export", s12_export.run),
]
STAGE_IDS = [s for s, _, _ in STAGES]


def _configure_warnings() -> None:
    # Solo se silencian avisos de obsolescencia y de nombres de variables; los demás (p. ej., convergencia) se muestran.
    for cat in (FutureWarning, DeprecationWarning):
        warnings.filterwarnings("ignore", category=cat)
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


def run(cfg: Config, until: str | None = None) -> Context:
    """Ejecuta el *pipeline* completo (o hasta la etapa ``until``, inclusive).

    Las etapas costosas (CV anidada, ajuste final y sensibilidad) guardan su resultado en ``<output>/cache``; al
    volver a ejecutar, se reanudan sin recalcular lo ya hecho si la configuración, los datos y las librerías coinciden.
    """
    if until is not None and until not in STAGE_IDS:
        raise ValueError(f"Unknown stage {until!r}; choose from {STAGE_IDS}")
    _configure_warnings()
    if cfg.quick:
        log.warning("QUICK MODE — STRUCTURAL TEST ONLY. DO NOT REPORT THESE RESULTS.")
    font = apply_style()
    ctx = Context(cfg)
    log.info("Output: %s | workers: %d | figure font: %s", ctx.paths.root.resolve(), ctx.n_workers, font)
    ctx.check_libraries()
    ctx.metrics_max_abs_diff = verify_metrics()
    log.info("Vectorised metrics verified against scikit-learn (max |Δ| = %.1e)", ctx.metrics_max_abs_diff)
    log.info("Candidates: %s | reference: LR | meta-ensemble base learners: %s",
             ctx.factory.candidates, ctx.factory.meta_base)
    for sid, name, fn in STAGES:
        ctx.timer.run(name, fn, ctx)
        if sid == "data":
            assert all(ctx.factory.audit_meta(ctx.factory.build(m, ctx.ds.cols, cfg.random_state))
                       for m in ("VotingSoft", "Stacking"))
        if sid == until:
            log.info("Stopped after stage '%s' as requested", sid)
            return ctx
    zip_path = s12_export.package(ctx)
    log.info("Done. Results in %s and %s", ctx.paths.root, zip_path)
    return ctx
