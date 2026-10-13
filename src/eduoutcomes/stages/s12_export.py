"""§15 — Verificaciones de integridad, exportación Springer, borrador de resultados y paquete reproducible."""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import CLASSES, K, META_MODELS, REF, dn
from ..data import sha256
from ..formatting import fmt_ci, fmt_p, num
from ..models import GroupedStackingClassifier

log = logging.getLogger(__name__)

TRIPOD = [
    ("Title/abstract", "Multiclass prediction (Dropout, Enrolled, Graduate); design; ensembles and meta-ensembles",
     "Title, Abstract"),
    ("Source of data", "UCI 697 (Realinho et al., 2021, 2022); SHA-256", "metadata/source_manifest.json"),
    ("Participants", "Undergraduates, Polytechnic Institute of Portalegre", "Table 1"),
    ("Outcome", "Three classes at the end of the normal programme duration", "Methods; Table 1"),
    ("Predictors", "36 predictors; type and time of availability", "Table S1"),
    ("Sample size", "n per partition and class", "Fig. 1; split_manifest.json"),
    ("Missing data", "None in source; imputers inside pipeline", "Table S1"),
    ("Data preparation", "Fold-local SMOTENC, one-hot encoding and scaling", "Methods"),
    ("Analysis — data use", "Locked grouped holdout; repeated nested CV; grouped stacking folds", "Fig. 1"),
    ("Analysis — models", "6 ensembles, voting, stacking, multinomial LR reference; grids", "Table S4"),
    ("Calibration", "Temperature scaling (primary); isotonic; Dirichlet; per-class ECE; calibration slope and intercept",
     "Tables 4, 6, S5; Fig. 3"),
    ("Class imbalance", "SMOTENC in training folds; ROS and none (sensitivity)", "Table 9"),
    ("Performance measures", "Macro-F1, MCC, κ, AUC, per-class metrics, Brier, log-loss, ECE with CIs", "Tables 2, 5–6"),
    ("Model comparison", "Friedman/Iman–Davenport–Nemenyi; Nadeau–Bengio + Holm; Bayesian correlated t test (ROPE)",
     "Table 3; Fig. 2"),
    ("Decision use", "Capacity-constrained prioritisation; expected vs observed; decision curve", "Table 7; Table S9; Fig. 4"),
    ("Fairness/subgroups", "Gender, scholarship, age, displaced, debtor", "Table S6; Fig. S3"),
    ("Explainability", "SHAP by class; permutation; fidelity-gated LIME; concordance and stability",
     "Table 8; Figs. 5–6; Tables S7–S8"),
    ("Limitations", "Single institution; internal validation; prediction horizon", "Discussion"),
    ("Code/data availability", "Repository; versions; checksums; runtime log", "metadata/"),
]


def integrity_checks(ctx, fqa: pd.DataFrame) -> dict[str, bool]:
    cfg, ds, fac = ctx.cfg, ctx.ds, ctx.factory
    checks = {
        "split_disjoint": set(ds.dev_ids).isdisjoint(ds.hold_ids),
        "profiles_disjoint_dev_holdout": set(ds.g_dev).isdisjoint(ds.g_hold),
        "class_counts_match_source": (cfg.expected_counts is None or tuple(int(c) for c in np.bincount(
            ds.y_all, minlength=K)) == tuple(cfg.expected_counts)),
        "predict_proba_column_order": all(list(ctx.FINAL[m].classes_) == list(range(K)) for m in fac.model_names),
        "split_complete": len(ds.dev_ids) + len(ds.hold_ids) == len(ds.df),
        "holdout_hash_unchanged": ds.holdout_hash() == ds.split_manifest["holdout_ids_sha256"],
        "meta_architecture_passed": all(fac.audit_meta(fac.build(m, ds.cols, cfg.random_state)) for m in META_MODELS),
        "stacking_folds_grouped_by_profile": isinstance(fac.build("Stacking", ds.cols, cfg.random_state),
                                                        GroupedStackingClassifier),
        "same_validation_rows_across_models": bool(ctx.validation_counts.nunique(axis=1).eq(1).all()),
        "complete_repeated_oof": bool(ctx.oof_df.groupby(["model", "calibration"]).size()
                                      .eq(len(ds.y_dev) * cfg.outer_repeats).all()),
        "all_calibration_methods_evaluated": set(ctx.fold_df.calibration) == set(cfg.calibration_methods),
        "vectorised_metrics_match_sklearn": ctx.metrics_max_abs_diff < 1e-9,
        "holdout_probabilities_valid": all(np.allclose(ctx.PH[m].sum(1), 1, atol=1e-6) and (ctx.PH[m] >= 0).all()
                                           for m in fac.model_names),
        "all_candidates_frozen_and_evaluated": set(ctx.PH) == set(fac.model_names) == set(ctx.fold_df.model),
        "selected_is_candidate_not_reference": ctx.BEST in fac.candidates and ctx.BEST != REF,
        "frozen_selected_matches": ctx.FROZEN["selected_model"] == ctx.BEST,
        "figures_springer_width": bool(fqa.width_ok.all()) if len(fqa) else True,
        "figures_max_height_234mm": bool(fqa.height_ok.all()) if len(fqa) else True,
        "figures_min_font_7pt": bool(fqa.font_ok.all()) if len(fqa) else True,
    }
    return {k: bool(v) for k, v in checks.items()}


def results_draft(ctx) -> str:
    cfg, ds, best, S, summary = ctx.cfg, ctx.ds, ctx.BEST, ctx.S, ctx.summary
    MH, CH, BB, BD, ql, qh = ctx.MH, ctx.CH, ctx.BB, ctx.BD, ctx.ql, ctx.qh
    fr, fam_a, fam_b, ms = ctx.friedman, ctx.famA, ctx.famB, ctx.ms
    N = len(ds.df)
    cnt = np.bincount(ds.y_all, minlength=K)
    fa = fam_a[fam_a.metric == "f1_macro"].set_index("model")
    n_sup_prac = int(fa.verdict.str.startswith("superior (practically").sum())
    n_sup = int(fa.verdict.str.startswith("superior").sum())
    cal_best = ctx.CM[ctx.CM.model == best].set_index("calibration")
    imp = ctx.imp
    sens = getattr(ctx, "sens", None)
    meta_txt = "; ".join(f"{dn(r.model)} vs {dn(r.comparator)} {fmt_ci(r['diff'], r.lo, r.hi, 3, True)}, "
                         f"P(better) = {num(r.p_better, 2)}, {r.verdict}" for _, r in fam_b.iterrows())
    sens_txt = "; ".join(
        f"{s_}: {num(sens[(sens.Analysis == 'Prediction horizon') & (sens.Setting == s_)].f1_macro.mean())}"
        for s_ in ds.pred_sets) if sens is not None else "not computed"
    shap_txt = ("The most influential predictors by mean absolute SHAP value were " + ", ".join(imp.index[:5])
                + " (Table 8, Fig. 5).") if "shap" in imp else \
        "SHAP was not computed; permutation importance is reported (Table 8)."
    return f"""# Results (draft generated by the pipeline — verify and edit before submission)

## Sample
Of {N:,} students ({ds.n_dup} exact duplicates removed), {cnt[0]:,} ({cnt[0] / N:.1%}) dropped out, {cnt[1]:,}
({cnt[1] / N:.1%}) remained enrolled and {cnt[2]:,} ({cnt[2] / N:.1%}) graduated (Table 1). The development set comprised
{len(ds.y_dev):,} students and the locked holdout {len(ds.y_hold):,} (Fig. 1).

## RQ1 — Discrimination
In repeated nested cross-validation, mean macro-F1 ranged from {num(summary.f1_macro_mean.min())}
({dn(summary.loc[summary.f1_macro_mean.idxmin(), 'model'])}) to {num(summary.f1_macro_mean.max())}
({dn(summary.loc[summary.f1_macro_mean.idxmax(), 'model'])}); the calibrated multinomial LR reached {ms(REF, 'f1_macro')}
(Table 2). The Friedman test {'indicated' if fr['p'] < cfg.alpha else 'did not indicate'} differences between models
(χ²({fr['k'] - 1}) = {fr['stat']:.2f}, p = {fmt_p(fr['p'])}; Iman–Davenport F({fr['id_df'][0]}, {fr['id_df'][1]}) =
{fr['id_f']:.2f}, p = {fmt_p(fr['id_p'])}; critical difference = {fr['cd']:.2f}; Fig. 2c). Compared with the LR, {n_sup}
of {len(ctx.factory.candidates)} candidates were superior in macro-F1 after Holm adjustment, and {n_sup_prac} exceeded the
practical margin of {cfg.practical_margin_f1m:.2f} (Table 3, panel A). Relative to their comparators, the meta-ensembles
were: {meta_txt} (Table 3, panel B). F1 for the Enrolled class was the lowest for every model (range
{num(summary.f1_Enrolled_mean.min())}–{num(summary.f1_Enrolled_mean.max())}). {dn(best)} was selected by the
prespecified tolerance hierarchy. On the holdout it achieved a macro-F1 of
{fmt_ci(MH[best]['f1_macro'], BB.f1_macro.quantile(ql), BB.f1_macro.quantile(qh))} versus {num(MH[REF]['f1_macro'])} for
the LR (Δ {fmt_ci(MH[best]['f1_macro'] - MH[REF]['f1_macro'], BD.f1_macro.quantile(ql), BD.f1_macro.quantile(qh), 3,
True)}; Tables 5–6).

## RQ2 — Calibration
Temperature scaling changed the class-wise ECE of {dn(best)} from {num(cal_best.loc['uncalibrated', 'cw_ece_mean'])} to
{num(cal_best.loc[cfg.calibration_primary, 'cw_ece_mean'])} in cross-validation (Table 4; Table S5). On the holdout, the
class-wise ECE was {num(MH[best]['cw_ece'])} and the ECE for Enrolled {num(MH[best]['ece_Enrolled'])} (Dropout
{num(MH[best]['ece_Dropout'])}; Graduate {num(MH[best]['ece_Graduate'])}); the multiclass Brier score was
{num(MH[best]['brier'], 4)}. One-vs-rest calibration slopes were
{', '.join(f"{c} {num(CH[best][f'slope_{c}'], 2)}" for c in CLASSES)} and intercepts
{', '.join(f"{c} {num(CH[best][f'citl_{c}'], 2, True)}" for c in CLASSES)} (Table 6, Fig. 3). Expected and observed class
counts are compared in Table S9; capacity-based prioritisation is reported in Table 7 and Fig. 4.

## RQ3 — Explainability
{shap_txt}
Global concordance: {ctx.conc_txt}. LIME status: {ctx.EXPL.get('LIME')}. Fidelity, local agreement and stability are
reported in Tables S7–S8; a local explanation for a student in the Enrolled class is shown in Fig. 6.

## Sensitivity analyses
Macro-F1 by prediction horizon: {sens_txt} (Table 9, Fig. 7).
"""


def _excluded(p: Path, root: Path) -> bool:
    return p.relative_to(root).parts[0] == "cache"


def run(ctx) -> None:
    cfg, paths = ctx.cfg, ctx.paths
    fqa = pd.DataFrame(ctx.rep.figure_qa)
    fqa.to_csv(paths.metadata / "figure_quality_control.csv", index=False)
    checks = integrity_checks(ctx, fqa)
    info = {"data_sha256_matches_reference": ctx.ds.sha_match, "cv_cache_statistics": ctx.cv_cache_stats,
            "cv_block_minutes": [round(s / 60, 2) for s in ctx.cv_block_seconds], "workers": ctx.n_workers,
            "quick_mode": cfg.quick}
    failed = [k for k, v in checks.items() if not v]
    if failed:
        log.error("\n%s", fqa.to_string(index=False))
        raise AssertionError(f"Quality gates failed: {failed}")
    checks["all_passed"] = True
    (paths.metadata / "quality_checks.json").write_text(json.dumps({"gates": checks, "information": info}, indent=2,
                                                                   default=str), encoding="utf-8")
    log.info("All integrity checks passed.")

    ctx.rep.export_all(extra_sheets={
        "nested_cv_folds": ctx.fold_df, "familyA_vs_LR": ctx.famA, "familyB_meta": ctx.famB,
        "sensitivity_folds": getattr(ctx, "sens", None), "holdout_bootstrap": ctx.BB.describe().T.reset_index(),
        "importance_all": ctx.imp.reset_index(), "figure_QC": fqa, "runtime": pd.DataFrame(ctx.timer.log)})
    draft = results_draft(ctx)
    if cfg.quick:
        draft = "> **WARNING — QUICK MODE: STRUCTURAL TEST ONLY. DO NOT REPORT THESE RESULTS.**\n\n" + draft
    (paths.root / "results_draft.md").write_text(draft, encoding="utf-8")
    (paths.root / "TRIPOD_AI_checklist.md").write_text(
        "# TRIPOD+AI — where each reporting domain is addressed\n\nMap to the official TRIPOD+AI item numbers "
        "(Collins et al., BMJ 2024) before submission.\n\n| Domain | Content | Location |\n|---|---|---|\n"
        + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in TRIPOD) + "\n", encoding="utf-8")


def package(ctx) -> Path:
    """Manifiesto SHA-256 y ZIP de resultados (sin cachés). Se ejecuta tras registrar el tiempo de la última etapa."""
    root = ctx.paths.root
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and not _excluded(p, root)
             and p.name != "artifact_checksum_manifest.csv"]
    art = [{"relative_path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256(p.read_bytes())}
           for p in files]
    pd.DataFrame(art).to_csv(ctx.paths.metadata / "artifact_checksum_manifest.csv", index=False)
    zip_path = root.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file() and not _excluded(p, root):
                zf.write(p, p.relative_to(root))
    log.info("Result package: %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)
    return zip_path
