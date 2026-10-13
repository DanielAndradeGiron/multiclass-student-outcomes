"""Prueba de humo de extremo a extremo (modo rápido, datos sintéticos). Tarda ~1–3 min."""

import json

import pytest

from eduoutcomes.pipeline import run


@pytest.mark.slow
def test_full_pipeline_runs(ci_config):
    ctx = run(ci_config)
    root = ctx.paths.root
    gates = json.loads((root / "metadata" / "quality_checks.json").read_text())["gates"]
    assert gates["all_passed"]
    for f in ("tables/Tables_Springer.docx", "tables/tables_sn-jnl.tex", "results_all.xlsx", "results_draft.md",
              "figures/Fig2_benchmarking.pdf", "metadata/frozen_configuration_before_holdout.json"):
        assert (root / f).exists(), f
    assert root.with_suffix(".zip").exists()
