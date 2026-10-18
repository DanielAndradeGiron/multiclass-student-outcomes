"""§2 — Auditoría de datos y bloqueo inmediato del conjunto de reserva."""

from __future__ import annotations

import json
import logging

from ..constants import SCENARIO_LABEL
from ..data import load_dataset

log = logging.getLogger(__name__)


def run(ctx) -> None:
    ctx.ds = load_dataset(ctx.cfg, ctx.paths.metadata)
    log.info("Data: %s | SHA-256 %s… %s", ctx.ds.df.shape, ctx.ds.data_sha[:16],
             "(matches reference)" if ctx.ds.sha_match else "(DIFFERS from reference)")
    log.info("Split manifest:\n%s", json.dumps(ctx.ds.split_manifest, indent=2))
    ctx.rep.scenario_note = (f"Prediction scenario {SCENARIO_LABEL[ctx.ds.scenario]}: {len(ctx.ds.cols)} predictors "
                             "available at the time of prediction")
    log.info("Scenario %s — %d predictors", ctx.ds.scenario, len(ctx.ds.cols))
    # QUALITY GATE: X_hold / y_hold no se usan hasta la etapa de reserva (s07_holdout).
