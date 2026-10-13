"""§7 — PI1: ranking, Friedman (Iman–Davenport) + Nemenyi, Nadeau–Bengio + Holm, ROPE bayesiano y selección."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy import stats

from ..constants import LABELS, META_MODELS, REF, dn, family_of
from ..formatting import best_cells, fmt_ci, fmt_ms, fmt_p, num
from ..inference import bayes_rope, friedman_iman_davenport, holm, nb_test, verdict
from ..style import C_META, C_NEG, C_POS, C_REF, GRID, OKABE_ITO, new_canvas, panel_label, tint

log = logging.getLogger(__name__)

METS_D = ["f1_macro", "balanced_accuracy", "mcc", "kappa", "roc_auc_macro", "f1_Dropout", "f1_Enrolled", "f1_Graduate",
          "recall_Enrolled", "f1_weighted", "brier", "cw_ece", "log_loss"]


def practical_select(summary: pd.DataFrame, candidates, cfg):
    """Jerarquía de tolerancias preespecificada: F1 macro → MCC → AUC macro → Brier → menor costo."""
    cand = summary[summary.model.isin(candidates)].copy()
    aud = []
    for col, tol, d in [("f1_macro_mean", cfg.tol_f1m, "max"), ("mcc_mean", cfg.tol_mcc, "max"),
                        ("roc_auc_macro_mean", cfg.tol_auc, "max"), ("brier_mean", cfg.tol_brier, "min")]:
        best = cand[col].max() if d == "max" else cand[col].min()
        keep = cand[col] >= best - tol if d == "max" else cand[col] <= best + tol
        aud.append({"Criterion": LABELS[col.replace("_mean", "")], "Direction": "maximise" if d == "max" else "minimise",
                    "Best": num(best, 4), "Tolerance": num(tol, 3), "Before": len(cand), "After": int(keep.sum()),
                    "Retained models": ", ".join(dn(m) for m in cand.loc[keep, "model"])})
        cand = cand[keep]
        if len(cand) == 1:
            break
    return cand.sort_values(["fit_s_mean", "model"], kind="mergesort").iloc[0].model, pd.DataFrame(aud)


def col_of(m, best):
    return C_POS if m == best else (C_REF if m == REF else (C_META if m in META_MODELS else C_NEG))


def run(ctx) -> None:
    cfg, fac, rep, FP = ctx.cfg, ctx.factory, ctx.rep, ctx.FP
    names, cands = fac.model_names, fac.candidates
    n_blocks = ctx.n_blocks
    summary = FP.groupby("model")[METS_D + ["fit_s"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    S = summary.set_index("model")
    test_ratio = 1 / (cfg.outer_splits - 1)   # n_test / n_train de cada pliegue externo
    ctx.summary, ctx.S, ctx.test_ratio = summary, S, test_ratio

    def pivot_of(metric):
        return FP.pivot(index="block", columns="model", values=metric)[names]

    PV = {m: pivot_of(m) for m in ("f1_macro", "mcc", "roc_auc_macro", "brier", "cw_ece", "ece_Enrolled")}
    ctx.PV = PV
    fr = friedman_iman_davenport(PV["f1_macro"], cfg.alpha)
    ctx.friedman = fr
    ranks, CD, k_mod = fr["ranks"], fr["cd"], fr["k"]
    margin = {"f1_macro": cfg.practical_margin_f1m, "mcc": cfg.practical_margin_mcc}

    # --- Familia A: cada candidato frente a la LR calibrada ---
    fam_a = []
    for metric in ("f1_macro", "mcc"):
        rr = []
        for m in cands:
            d = PV[metric][m] - PV[metric][REF]
            mu, lo, hi, t, p = nb_test(d, test_ratio, cfg.alpha)
            pw, pr, pb = bayes_rope(d, margin[metric], test_ratio)
            rr.append({"metric": metric, "model": m, "diff": mu, "lo": lo, "hi": hi, "t": t, "p": p,
                       "p_worse": pw, "p_rope": pr, "p_better": pb})
        rr = pd.DataFrame(rr)
        rr["p_holm"] = holm(rr.p)
        fam_a.append(rr)
    fam_a = pd.concat(fam_a, ignore_index=True)
    fam_a["verdict"] = [verdict(r["diff"], r.lo, r.hi, r.p_holm, margin[r.metric], cfg.alpha)
                        for _, r in fam_a.iterrows()]

    # --- Familia B: metaensamble frente a su mejor aprendiz base y al mejor ensamble individual ---
    best_base = max(fac.meta_base, key=lambda b: S.loc[b, "f1_macro_mean"])
    best_single = max(fac.ensembles, key=lambda b: S.loc[b, "f1_macro_mean"])
    fam_b = []
    for m in META_MODELS:
        for lab, comp in [("best base learner", best_base), ("best individual ensemble", best_single)]:
            if lab == "best individual ensemble" and comp == best_base:
                continue
            d = PV["f1_macro"][m] - PV["f1_macro"][comp]
            mu, lo, hi, t, p = nb_test(d, test_ratio, cfg.alpha)
            pw, pr, pb = bayes_rope(d, cfg.practical_margin_f1m, test_ratio)
            fam_b.append({"model": m, "comparator": comp, "comparator_role": lab, "diff": mu, "lo": lo, "hi": hi,
                          "t": t, "p": p, "p_worse": pw, "p_rope": pr, "p_better": pb})
    fam_b = pd.DataFrame(fam_b)
    fam_b["p_holm"] = holm(fam_b.p)
    fam_b["verdict"] = [verdict(r["diff"], r.lo, r.hi, r.p_holm, cfg.practical_margin_f1m, cfg.alpha)
                        for _, r in fam_b.iterrows()]
    ctx.famA, ctx.famB = fam_a, fam_b

    best, sel_audit = practical_select(summary, cands, cfg)
    ctx.BEST = best
    log.info("Selected candidate: %s | Friedman p = %s | Iman–Davenport p = %s | CD = %.3f",
             best, fmt_p(fr["p"]), fmt_p(fr["id_p"]), CD)

    def ms(m, col, d=3):
        return fmt_ms(S.loc[m, col + "_mean"], S.loc[m, col + "_std"], d)

    ctx.ms = ms
    order_f1 = summary.sort_values("f1_macro_mean", ascending=False).model.tolist()
    ctx.order_f1 = order_f1
    t2 = pd.DataFrame([{
        "Model": dn(m) + ("*" if m == best else ""), "Family": family_of(m),
        "Macro-F1": ms(m, "f1_macro"), "Balanced accuracy": ms(m, "balanced_accuracy"), "MCC": ms(m, "mcc"),
        "Cohen's κ": ms(m, "kappa"), "Macro AUC": ms(m, "roc_auc_macro"), "F1 Dropout": ms(m, "f1_Dropout"),
        "F1 Enrolled": ms(m, "f1_Enrolled"), "F1 Graduate": ms(m, "f1_Graduate"), "Weighted F1": ms(m, "f1_weighted"),
        "Mean rank": num(ranks[m], 2), "Time (s)": f"{S.loc[m, 'fit_s_mean']:.0f}"} for m in order_f1])
    rep.register_table(
        t2, "Table2_nested_cv_discrimination", "2",
        f"Discrimination of the candidates in repeated nested cross-validation ({cfg.outer_splits}-fold × "
        f"{cfg.outer_repeats} repeats; development set, n = {len(ctx.ds.y_dev):,})",
        f"Mean ± SD over {n_blocks} outer validation folds; bold, best value per column. Each candidate was tuned in the "
        f"inner loop ({cfg.inner_splits}-fold grouped CV, {cfg.search_n_iter} random configurations, macro-F1), with "
        "SMOTENC applied only to training data, and calibrated by temperature scaling on inner out-of-fold predictions. "
        "Classes were predicted by the highest probability. Mean rank: average Friedman rank on macro-F1 (1 = best); "
        f"Friedman χ²({k_mod - 1}) = {fr['stat']:.2f}, p = {fmt_p(fr['p'])}; Iman–Davenport F({fr['id_df'][0]}, "
        f"{fr['id_df'][1]}) = {fr['id_f']:.2f}, p = {fmt_p(fr['id_p'])}; Nemenyi critical difference = {CD:.2f}. "
        f"* Selected by the prespecified tolerance hierarchy (macro-F1 ± {cfg.tol_f1m}, MCC ± {cfg.tol_mcc}, macro AUC "
        f"± {cfg.tol_auc}, Brier ± {cfg.tol_brier}, then lowest time); the logistic regression reference was not "
        "eligible. Weighted F1 is shown only for comparability with previous studies. Time: mean single-core tuning and "
        "calibration time per fold (meta-ensembles include the tuning of their base learners). AUC, area under the ROC "
        "curve (one-vs-rest); MCC, Matthews correlation coefficient; SD, standard deviation", landscape=True,
        bold=best_cells(t2, cols_max=["Macro-F1", "Balanced accuracy", "MCC", "Cohen's κ", "Macro AUC", "F1 Dropout",
                                      "F1 Enrolled", "F1 Graduate", "Weighted F1"], cols_min=["Mean rank"]))

    def row3(r, contrast, metric):
        return {"Panel": None, "Contrast": contrast, "Metric": LABELS[metric],
                "Mean difference (95% CI)": fmt_ci(r["diff"], r.lo, r.hi, 3, signed=True), "Corrected t": num(r.t, 2),
                "p": fmt_p(r.p), "Holm p": fmt_p(r.p_holm), "P(better)": num(r.p_better, 2),
                "P(ROPE)": num(r.p_rope, 2), "Verdict": r.verdict}

    t3a = [{**row3(r, f"{dn(r.model)} − LR", r.metric), "Panel": "A. Each candidate versus the calibrated multinomial LR"}
           for _, r in fam_a.sort_values(["metric", "diff"], ascending=[True, False]).iterrows()]
    t3b = [{**row3(r, f"{dn(r.model)} − {dn(r.comparator)} ({r.comparator_role})", "f1_macro"),
            "Panel": "B. Meta-ensemble versus its best base learner and the best individual ensemble"}
           for _, r in fam_b.iterrows()]
    rep.register_table(
        pd.DataFrame(t3a + t3b), "Table3_statistical_comparisons", "3",
        "Paired comparisons in repeated nested cross-validation: (A) each candidate versus the calibrated multinomial "
        "logistic regression (LR) and (B) each meta-ensemble versus its best base learner and the best individual ensemble",
        f"Differences across {n_blocks} outer folds. Nadeau–Bengio corrected resampled t test (variance inflated by "
        f"n_test/n_train = {test_ratio:.2f}); Holm adjustment within each panel and metric. Practical margin (region of "
        f"practical equivalence, ROPE): ± {cfg.practical_margin_f1m:.2f} for macro-F1 and ± {cfg.practical_margin_mcc:.2f} "
        "for MCC. 'Practically relevant' requires the whole 95% CI beyond the margin; 'practically equivalent' requires it "
        "within ± margin. P(better) and P(ROPE): posterior probabilities from the Bayesian correlated t test (Benavoli et "
        "al., 2017) that the difference exceeds the margin or lies within it. Base learners inside the meta-ensembles are "
        "the very same fitted models evaluated individually, so panel B isolates the effect of combination. Repeated CV "
        "folds are not independent, so the tests are approximate; non-significance does not establish equivalence",
        landscape=True, group_col="Panel")
    rep.register_table(sel_audit, "TableS3_selection_audit", "S3", "Audit of the prespecified tolerance-based selection",
                       "Each criterion retains the candidates within the stated tolerance of the best value among those "
                       "still eligible", supplementary=True)
    pdf_ = ctx.param_df.copy()
    pdf_["Model"] = pdf_["Model"].map(dn)
    rep.register_table(pdf_, "TableS4_hyperparameters", "S4", "Hyperparameters selected in the inner loop of each outer fold",
                       "Meta-ensembles reuse the hyperparameters tuned for their base learners in the same outer fold. "
                       "T, fitted temperature", supplementary=True)
    figure2(ctx, summary, PV, fam_a, ranks, CD, k_mod, best)


def figure2(ctx, summary, PV, fam_a, ranks, CD, k_mod, best) -> None:
    """a) F1 macro por pliegue; b) Δ frente a LR con ROPE; c) diagrama de diferencias críticas."""
    cfg = ctx.cfg
    fig = new_canvas("double", 160, wspace=0.06, hspace=0.04)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 0.8])
    a1, a3, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    order = summary.sort_values("f1_macro_mean").model.tolist()
    for i, m in enumerate(order):
        v = PV["f1_macro"][m].values
        col = col_of(m, best)
        a1.scatter(v, i + (np.random.default_rng(i).random(len(v)) - 0.5) * 0.35, s=7, color=tint(col, 0.45), lw=0,
                   zorder=2)
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0
        a1.errorbar(v.mean(), i, xerr=h, fmt="o", ms=4, color=col, mec="black", mew=0.4, capsize=2, elinewidth=0.9,
                    zorder=3)
    a1.set_yticks(range(len(order)), [dn(m) for m in order])
    a1.set_xlabel("Macro-F1 (outer folds)")
    a1.grid(axis="x", lw=0.4, color=GRID)
    a1.set_axisbelow(True)
    q = fam_a[fam_a.metric == "f1_macro"].sort_values("diff").reset_index(drop=True)
    a3.axvspan(-cfg.practical_margin_f1m, cfg.practical_margin_f1m, color=tint(OKABE_ITO["yellow"], 0.45), lw=0,
               zorder=0, label=f"ROPE (± {cfg.practical_margin_f1m:.2f})")
    a3.axvline(0, color="black", lw=0.6, zorder=1)
    for i, r in q.iterrows():
        err = [[r["diff"] - r.lo], [r.hi - r["diff"]]] if np.isfinite(r.lo) else None
        a3.errorbar(r["diff"], i, xerr=err, fmt="o", ms=4, color=col_of(r.model, best), mec="black", mew=0.4,
                    capsize=2, elinewidth=0.9, zorder=3)
        if r.p_holm < cfg.alpha:
            a3.annotate("*", (r.hi, i), xytext=(2, -1), textcoords="offset points", va="center", fontsize=8)
    a3.set_yticks(range(len(q)), [dn(m) for m in q.model])
    a3.set_xlabel("Δ macro-F1 (candidate − LR; 95% CI)")
    a3.grid(axis="x", lw=0.4, color=GRID)
    a3.set_axisbelow(True)
    a3.legend(loc="lower right", handlelength=1.0)
    # diagrama de diferencias críticas (Demšar, 2006)
    rk = ranks.sort_values()
    lo_r, hi_r = 1, k_mod
    a2.set_xlim(lo_r - 2.2, hi_r + 2.2)
    a2.set_ylim(0, 1.08)
    a2.axis("off")
    a2.plot([lo_r, hi_r], [0.86, 0.86], color="black", lw=0.7)
    for r_ in range(lo_r, hi_r + 1):
        a2.plot([r_, r_], [0.86, 0.90], color="black", lw=0.6)
        a2.text(r_, 0.93, str(r_), ha="center", fontsize=7)
    half = math.ceil(len(rk) / 2)
    for i, (m, r_) in enumerate(rk.items()):
        left = i < half
        yy = 0.66 - (i if left else len(rk) - 1 - i) * 0.12
        xt = lo_r - 0.2 if left else hi_r + 0.2
        a2.plot([r_, r_, xt], [0.86, yy, yy], color=col_of(m, best), lw=0.8)
        a2.text(xt + (-0.06 if left else 0.06), yy, f"{dn(m)} ({r_:.2f})", ha="right" if left else "left",
                va="center", fontsize=7)
    cliques, vals = [], rk.values
    for i in range(len(vals)):
        j = max(jj for jj in range(i, len(vals)) if vals[jj] - vals[i] <= CD)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for n_, (i, j) in enumerate(cliques):
        yy = 0.80 - n_ * 0.04
        a2.plot([vals[i] - 0.04, vals[j] + 0.04], [yy, yy], color="black", lw=1.8, solid_capstyle="butt")
    cd_end = min(lo_r + CD, hi_r)
    a2.plot([lo_r, cd_end], [1.03, 1.03], color="black", lw=1.0)
    for xx in (lo_r, cd_end):
        a2.plot([xx, xx], [1.01, 1.05], color="black", lw=0.8)
    a2.text(cd_end + 0.1, 1.03, f"CD = {CD:.2f}", ha="left", va="center", fontsize=7)
    panel_label(a1, "a")
    panel_label(a3, "b")
    panel_label(a2, "c")
    ctx.rep.save_figure(
        fig, "Fig2_benchmarking",
        "Fig. 2 Benchmarking in repeated nested cross-validation. a Macro-F1 in each outer fold (light dots) with mean "
        f"and 95% CI; vermillion, selected candidate ({dn(best)}); purple, meta-ensembles; grey, logistic regression (LR) "
        "reference. b Mean paired difference in macro-F1 between each candidate and the calibrated LR with Nadeau–Bengio "
        "corrected 95% CIs; shaded band, region of practical equivalence (ROPE); * Holm-adjusted p < 0.05. "
        "c Critical-difference (CD) diagram of average Friedman ranks (1 = best); candidates joined by a thick bar do "
        "not differ significantly (Nemenyi test, α = 0.05)")
