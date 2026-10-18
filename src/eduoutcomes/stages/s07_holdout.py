"""§10 — Apertura única de la reserva bloqueada: PI1 y PI2 en datos no vistos (Tablas 5–6, Fig. 3)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from ..constants import CLASSES, K, LABELS, REF, dn, family_of
from ..formatting import best_cells, fmt_ci, num
from ..metrics import cal_lr_metrics, ece_binary, mc_metrics, reliability_q, strat_boot_idx
from ..style import C_CLASS, C_POS, C_REF, CMAP_SEQ, new_canvas, panel_label, tint

SECTIONS6 = [
    ("Discrimination", ["f1_macro", "balanced_accuracy", "mcc", "kappa", "roc_auc_macro", "accuracy", "f1_weighted"]),
    ("Per-class discrimination", ["recall_Dropout", "recall_Enrolled", "recall_Graduate", "precision_Dropout",
                                  "precision_Enrolled", "precision_Graduate", "f1_Dropout", "f1_Enrolled", "f1_Graduate"]),
    ("Probabilistic accuracy and calibration", ["brier", "log_loss", "ece", "cw_ece", "ece_Dropout", "ece_Enrolled",
                                                "ece_Graduate"]),
    ("Weak calibration, one-vs-rest (ideal: intercept 0, slope 1)", [f"{p_}_{c}" for p_ in ("citl", "slope")
                                                                     for c in CLASSES]),
]


def _reliability_panel(ax, yb, pb, pbu, xlabel, cfg):
    ax.plot([0, 1], [0, 1], ls=(0, (3, 2)), color=C_REF, lw=0.6, zorder=1)
    h_, e_ = np.histogram(pb, bins=20, range=(0, 1))
    ax.bar(e_[:-1], h_ / max(h_.max(), 1) * 0.12, width=0.05, align="edge", color=tint(C_REF, 0.35), lw=0, zorder=0)
    ru, rc = reliability_q(yb, pbu, cfg.reliability_bins), reliability_q(yb, pb, cfg.reliability_bins)
    ax.plot(ru[:, 0], ru[:, 1], "o--", color=C_REF, ms=3.2, mfc="white", mew=0.7, lw=0.7, label="Uncalibrated", zorder=2)
    ax.errorbar(rc[:, 0], rc[:, 1], yerr=[rc[:, 1] - rc[:, 3], rc[:, 4] - rc[:, 1]], fmt="s-", color=C_POS, ms=3.2,
                mec="black", mew=0.3, lw=0.9, elinewidth=0.6, capsize=1.2, label="Temperature scaling", zorder=3)
    ax.set(xlabel=xlabel, ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.text(0.97, 0.16, f"ECE {num(ece_binary(yb, pb, cfg.ece_bins))}", transform=ax.transAxes, ha="right", fontsize=7)


def run(ctx) -> None:
    cfg, ds, rep, fac, best = ctx.cfg, ctx.ds, ctx.rep, ctx.factory, ctx.BEST
    X_hold, y_hold = ds.X_hold, ds.y_hold
    t0 = time.perf_counter()
    PH = {m: ctx.FINAL[m].predict_proba(X_hold) for m in fac.model_names}
    ctx.infer_s = time.perf_counter() - t0
    PH_unc = {m: ctx.FINAL[m].predict_proba(X_hold, method="uncalibrated") for m in (best, REF)}
    P_best, P_ref = PH[best], PH[REF]
    ctx.PH, ctx.PH_unc, ctx.P_best, ctx.P_ref = PH, PH_unc, P_best, P_ref
    pd.DataFrame({"row_id": ds.hold_ids, "observed": [CLASSES[i] for i in y_hold],
                  **{f"{best}_p_{c}": P_best[:, i] for i, c in enumerate(CLASSES)},
                  **{f"LR_p_{c}": P_ref[:, i] for i, c in enumerate(CLASSES)}}
                 ).to_csv(ctx.paths.metadata / "holdout_predictions.csv", index=False)
    pd.DataFrame({"row_id": ds.hold_ids, "observed": y_hold,
                  **{f"{m}|p_{c}": PH[m][:, i] for m in fac.model_names for i, c in enumerate(CLASSES)}}
                 ).to_csv(ctx.paths.metadata / "holdout_predictions_all_models.csv.gz", index=False)
    MH = {m: mc_metrics(y_hold, PH[m], cfg.ece_bins) for m in fac.model_names}
    CH = {m: cal_lr_metrics(y_hold, PH[m]) for m in (best, REF)}
    ctx.MH, ctx.CH = MH, CH

    t5 = pd.DataFrame([{
        "Model": dn(m) + ("*" if m == best else ""), "Family": family_of(m),
        "Macro-F1": num(MH[m]["f1_macro"]), "Balanced accuracy": num(MH[m]["balanced_accuracy"]),
        "MCC": num(MH[m]["mcc"]), "Macro AUC": num(MH[m]["roc_auc_macro"]),
        "Recall (Enrolled)": num(MH[m]["recall_Enrolled"]), "Brier score": num(MH[m]["brier"], 4),
        "Class-wise ECE": num(MH[m]["cw_ece"]), "ECE (Enrolled)": num(MH[m]["ece_Enrolled"])}
        for m in sorted(fac.model_names, key=lambda m: -MH[m]["f1_macro"])])
    rep.register_table(
        t5, "Table5_holdout_all", "5", f"Performance of all frozen candidates on the locked holdout (n = {len(y_hold):,})",
        "All candidates were refitted and calibrated (temperature scaling) on the development set and frozen before the "
        "holdout was opened once. Point estimates; bold, best value per column. The selected candidate (*) was chosen in "
        "cross-validation, not on the holdout. ECE, expected calibration error (15 equal-width bins); MCC, Matthews "
        "correlation coefficient", landscape=True,
        bold=best_cells(t5, cols_max=["Macro-F1", "Balanced accuracy", "MCC", "Macro AUC", "Recall (Enrolled)"],
                        cols_min=["Brier score", "Class-wise ECE", "ECE (Enrolled)"]))

    BB, BR, BD = [], [], []
    for idx in strat_boot_idx(y_hold, cfg.holdout_bootstrap, cfg.random_state + 14000):
        a = {**mc_metrics(y_hold[idx], P_best[idx], cfg.ece_bins), **cal_lr_metrics(y_hold[idx], P_best[idx])}
        b = {**mc_metrics(y_hold[idx], P_ref[idx], cfg.ece_bins), **cal_lr_metrics(y_hold[idx], P_ref[idx])}
        BB.append(a)
        BR.append(b)
        BD.append({k_: a[k_] - b[k_] for k_ in a})
    BB, BR, BD = pd.DataFrame(BB), pd.DataFrame(BR), pd.DataFrame(BD)
    ql, qh = cfg.alpha / 2, 1 - cfg.alpha / 2
    ctx.BB, ctx.BR, ctx.BD, ctx.ql, ctx.qh = BB, BR, BD, ql, qh
    rows6 = []
    for sec, mets in SECTIONS6:
        for m in mets:
            d = 4 if m == "brier" else 3
            eb = MH[best][m] if m in MH[best] else CH[best][m]
            er = MH[REF][m] if m in MH[REF] else CH[REF][m]
            sg = m.startswith("citl")
            rows6.append({"Section": sec, "Metric": LABELS[m],
                          dn(best): fmt_ci(eb, BB[m].quantile(ql), BB[m].quantile(qh), d, signed=sg),
                          "LR reference": fmt_ci(er, BR[m].quantile(ql), BR[m].quantile(qh), d, signed=sg),
                          f"Δ ({dn(best)} − LR)": fmt_ci(eb - er, BD[m].quantile(ql), BD[m].quantile(qh), d,
                                                         signed=True)})
    rep.register_table(
        pd.DataFrame(rows6), "Table6_holdout_selected_vs_lr", "6",
        f"Discrimination and calibration of the frozen {dn(best)} and the logistic regression (LR) reference on the "
        f"locked holdout, overall and by class (n = {len(y_hold):,}; Dropout = {int((y_hold == 0).sum())}, Enrolled = "
        f"{int((y_hold == 1).sum())}, Graduate = {int((y_hold == 2).sum())})",
        f"95% CIs from {cfg.holdout_bootstrap:,} bootstrap resamples stratified by class, conditional on the frozen models "
        "(percentile method); Δ from paired resamples. Classes assigned by the highest calibrated probability. For Brier "
        "score, log-loss and ECE, lower is better. Calibration intercept (calibration-in-the-large) and slope from a "
        "one-vs-rest logistic recalibration of logit(p) (Van Calster et al., 2019). ECE is positively biased in bootstrap "
        "resamples (sampling noise adds apparent miscalibration), so its percentile interval may lie above the point "
        "estimate; paired Δ are less affected. ECE, expected calibration error; MCC, "
        "Matthews correlation coefficient", landscape=True, group_col="Section")
    figure3(ctx)


def figure3(ctx) -> None:
    cfg, ds, best = ctx.cfg, ctx.ds, ctx.BEST
    y_hold, P_best, P_ref, MH = ds.y_hold, ctx.P_best, ctx.P_ref, ctx.MH
    fig = new_canvas("double", 128, wspace=0.03, hspace=0.05)
    gs = fig.add_gridspec(2, 3)
    axs = [fig.add_subplot(gs[i, j]) for i in (0, 1) for j in (0, 1, 2)]
    cm = confusion_matrix(y_hold, P_best.argmax(1), labels=range(K))
    cr = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    axs[0].imshow(cr, cmap=CMAP_SEQ, vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            axs[0].text(j, i, f"{cm[i, j]}\n({cr[i, j]:.0%})", ha="center", va="center", fontsize=7,
                        color="white" if cr[i, j] > 0.55 else "black")
    axs[0].set_xticks(range(K), CLASSES)
    axs[0].set_yticks(range(K), CLASSES, rotation=90, va="center")
    axs[0].set_xlabel("Predicted class")
    axs[0].set_ylabel("Observed class")
    axs[0].tick_params(length=0)
    for s_ in axs[0].spines.values():
        s_.set_visible(False)
    for i, c in enumerate(CLASSES):
        f_, t_, _ = roc_curve((y_hold == i).astype(int), P_best[:, i])
        axs[1].plot(f_, t_, color=C_CLASS[c], lw=1.1, label=f"{c} ({num(MH[best][f'auc_{c}'])})")
        f2, t2_, _ = roc_curve((y_hold == i).astype(int), P_ref[:, i])
        axs[1].plot(f2, t2_, color=C_CLASS[c], lw=0.7, ls=(0, (3, 1.5)))
    axs[1].plot([0, 1], [0, 1], ls=":", color=C_REF, lw=0.6)
    axs[1].set(xlabel="1 − Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01), aspect="equal")
    axs[1].set_xticks([0, 0.5, 1])
    axs[1].set_yticks([0, 0.5, 1])
    axs[1].legend(loc="lower right", title="AUC", handlelength=1.2)
    conf_, corr_ = P_best.max(1), (P_best.argmax(1) == y_hold).astype(float)
    cu = ctx.PH_unc[best]
    _reliability_panel(axs[2], corr_, conf_, cu.max(1), "Top-label confidence", cfg)
    for i, c in enumerate(CLASSES):
        _reliability_panel(axs[3 + i], (y_hold == i).astype(float), P_best[:, i], cu[:, i], f"Predicted P({c})", cfg)
    fig.legend(*axs[2].get_legend_handles_labels(), loc="outside lower center", ncol=2,
               title="Reliability diagrams (c–f)")
    for ax_, l_ in zip(axs, "abcdef"):
        panel_label(ax_, l_)
    ctx.rep.save_figure(
        fig, "Fig3_holdout_discrimination_calibration",
        f"Fig. 3 Discrimination and calibration of the frozen {dn(best)} on the locked holdout (n = {len(y_hold):,}). "
        "a Confusion matrix (counts and row percentages). b One-vs-rest ROC curves for each class (solid, selected "
        "candidate, AUC in brackets; dashed, logistic regression reference). c Top-label reliability diagram. d–f "
        "Class-wise reliability diagrams for Dropout, Enrolled and Graduate before (open circles) and after (squares, "
        f"Wilson 95% CIs) temperature scaling, using {cfg.reliability_bins} equal-frequency bins; grey bars, distribution "
        f"of the calibrated probabilities. ECE ({cfg.ece_bins} equal-width bins) refers to the calibrated probabilities. "
        "ECE, expected calibration error")
