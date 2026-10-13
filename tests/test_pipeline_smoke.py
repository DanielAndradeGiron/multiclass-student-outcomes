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
    explainability = json.loads((root / "metadata" / "explainability_status.json").read_text())
    for f in ("tables/Tables_Springer.docx", "tables/tables_sn-jnl.tex", "results_all.xlsx", "results_draft.md",
              "figures/Fig2_benchmarking.pdf", "metadata/frozen_configuration_before_holdout.json"):
        assert (root / f).exists(), f
    if explainability.get("SHAP", "").startswith("permutation explainer"):
        assert any((root / "figures" / f"Fig6_local_explanation_enrolled.{ext}").exists()
                   for ext in ("pdf", "eps", "tiff", "png"))
    if (explainability.get("SHAP", "").startswith("permutation explainer")
            and explainability.get("LIME", "").startswith("median R²")):
        assert any((root / "supplementary" / f"FigS4_lime_local_enrolled.{ext}").exists()
                   for ext in ("pdf", "eps", "tiff", "png"))
        assert "SHAP vs LIME (matched cases/classes)" in (
            root / "supplementary" / "TableS7_explanation_quality.csv").read_text(encoding="utf-8-sig")
    assert root.with_suffix(".zip").exists()
