"""§8 — PI2: calibración por clase en la validación anidada (Tabla 4, Tabla S5, Figs. S1–S2)."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd

from ..constants import CAL_LABEL, CLASSES, K, LABELS, dn
from ..formatting import best_cells, fmt_ci, fmt_ms, fmt_p, num
from ..inference import nb_test
from ..style import C_CLASS, C_NEG, C_POS, C_REF, GRID, OKABE_ITO, new_figure, panel_label
from .s04_rq1_comparison import col_of

CAL_METS = ["brier", "log_loss", "ece", "cw_ece", "ece_Dropout", "ece_Enrolled", "ece_Graduate"]


def run(ctx) -> None:
    cfg, rep, fold_df, names = ctx.cfg, ctx.rep, ctx.fold_df, ctx.factory.model_names
    CM = fold_df.groupby(["model", "calibration"])[CAL_METS + ["f1_macro"]].agg(["mean", "std"])
    CM.columns = [f"{a}_{b}" for a, b in CM.columns]
    CM = CM.reset_index()
    ctx.CM = CM
    PU = {m: fold_df[fold_df.calibration == "uncalibrated"].pivot(index="block", columns="model", values=m)[names]
          for m in ("cw_ece", "ece_Enrolled")}
    t4 = []
    for m in ctx.order_f1:
        r = CM[(CM.model == m) & (CM.calibration == cfg.calibration_primary)].iloc[0]
        row = {"Model": dn(m) + ("*" if m == ctx.BEST else "")}
        for c in CAL_METS:
            row[LABELS[c]] = fmt_ms(r[c + "_mean"], r[c + "_std"], 4 if c == "brier" else 3)
        for c in ("cw_ece", "ece_Enrolled"):
            mu, lo, hi, _, _ = nb_test(ctx.PV[c][m] - PU[c][m], ctx.test_ratio, cfg.alpha)
            row[f"Δ {LABELS[c]} (95% CI)"] = fmt_ci(mu, lo, hi, 3, signed=True)
        t4.append(row)
    t4 = pd.DataFrame(t4)
    rep.register_table(
        t4, "Table4_calibration_cv", "4",
        "Probabilistic accuracy and class-wise calibration of the candidates in repeated nested cross-validation "
        "(temperature scaling)",
        f"Mean ± SD over {ctx.n_blocks} outer folds; bold, best value per column. Brier: multiclass Brier score (sum over "
        "classes); ECE (top-label): expected calibration error of the predicted class; class-wise ECE: mean of the "
        f"one-vs-rest ECEs; {cfg.ece_bins} equal-width bins. Δ: calibrated − uncalibrated with Nadeau–Bengio corrected 95% "
        "CIs (negative values indicate improvement). The temperature was fitted on inner out-of-fold predictions only. "
        "Lower values are better", landscape=True, bold=best_cells(t4, cols_min=[LABELS[c] for c in CAL_METS]))

    t_s5 = []
    for m in ctx.order_f1:
        sub = CM[CM.model == m].set_index("calibration").reindex(list(cfg.calibration_methods))
        best_m = sub["cw_ece_mean"].idxmin()
        for meth, r in sub.iterrows():
            t_s5.append({"Model": dn(m), "Calibration": CAL_LABEL[meth] + (" †" if meth == best_m else ""),
                         "Macro-F1": num(r.f1_macro_mean), "Brier score": num(r.brier_mean, 4),
                         "Log-loss": num(r.log_loss_mean), "ECE (top-label)": num(r.ece_mean),
                         "Class-wise ECE": num(r.cw_ece_mean), "ECE (Enrolled)": num(r.ece_Enrolled_mean)})
    rep.register_table(
        pd.DataFrame(t_s5), "TableS5_calibration_methods", "S5",
        "Comparison of calibration methods in repeated nested cross-validation",
        "Means over outer folds; all methods evaluated on the same validation observations. † Lowest class-wise ECE for "
        "the model. Temperature scaling was prespecified as primary because it preserves the predicted class; isotonic "
        "regression was fitted one-vs-rest and renormalised; Dirichlet calibration is a multinomial logistic regression "
        "on log-probabilities (matrix scaling, L2)", supplementary=True, group_col="Model")

    # --- Fig. S2: efecto de la calibración ---
    ordc = ctx.order_f1[::-1]
    mk = {"uncalibrated": ("o", C_REF), "temperature": ("s", C_POS), "isotonic": ("^", C_NEG),
          "dirichlet": ("D", OKABE_ITO["green"])}
    fig, axs = new_figure("double", 34 + 8 * len(names), ncols=2, sharey=True)
    for ax, metric in zip(axs, ["cw_ece", "ece_Enrolled"]):
        for i, m in enumerate(ordc):
            sub = CM[CM.model == m].set_index("calibration")
            xs_ = [sub.loc[mm, metric + "_mean"] for mm in cfg.calibration_methods]
            ax.plot([min(xs_), max(xs_)], [i, i], color="#BBBBBB", lw=0.6, zorder=1)
            for mm in cfg.calibration_methods:
                ax.scatter(sub.loc[mm, metric + "_mean"], i, marker=mk[mm][0], s=16, color=mk[mm][1],
                           edgecolor="black", lw=0.3, zorder=2, label=CAL_LABEL[mm] if i == 0 else None)
        ax.set_xlabel(LABELS[metric] + " (lower is better)")
        ax.grid(axis="x", lw=0.4, color=GRID)
        ax.set_axisbelow(True)
        ax.set_xlim(left=0)
    axs[0].set_yticks(range(len(ordc)), [dn(m) for m in ordc])
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    panel_label(axs[0], "a")
    panel_label(axs[1], "b")
    rep.save_figure(fig, "FigS2_calibration_methods",
                    "Fig. S2 Effect of post-hoc calibration in repeated nested cross-validation. a Class-wise expected "
                    "calibration error (ECE) and b ECE for the Enrolled class, averaged over outer folds, without "
                    "calibration and with temperature scaling (primary), isotonic and Dirichlet calibration",
                    supplementary=True)

    # --- Fig. S1: distribución de desenlaces y rangos de Friedman ---
    N = len(ctx.ds.df)
    fr = ctx.friedman
    fig, (a1, a2) = new_figure("double", 66, ncols=2, gridspec_kw=dict(width_ratios=[1, 1.2]))
    cnt3 = pd.Series(np.bincount(ctx.ds.y_all, minlength=K), index=CLASSES)
    a1.bar(range(K), cnt3.values, color=[C_CLASS[c] for c in CLASSES], width=0.6, edgecolor="black", lw=0.4)
    for i, v in enumerate(cnt3.values):
        a1.text(i, v + cnt3.max() * 0.02, f"{v:,}\n({v / N:.1%})", ha="center", va="bottom", fontsize=7)
    a1.set_xticks(range(K), CLASSES)
    a1.set_ylabel("Students")
    a1.set_ylim(0, cnt3.max() * 1.28)
    a1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    rk_ = fr["ranks"].sort_values()
    a2.barh(range(len(rk_)), rk_.values, color=[col_of(m, ctx.BEST) for m in rk_.index], height=0.6,
            edgecolor="black", lw=0.3)
    a2.set_yticks(range(len(rk_)), [dn(m) for m in rk_.index])
    a2.invert_yaxis()
    a2.set_xlabel("Mean Friedman rank (macro-F1; 1 = best)")
    a2.grid(axis="x", lw=0.4, color=GRID)
    a2.set_axisbelow(True)
    panel_label(a1, "a")
    panel_label(a2, "b")
    rep.save_figure(fig, "FigS1_outcomes_and_ranks",
                    f"Fig. S1 a Distribution of the three academic outcomes (n = {N:,}). b Mean Friedman ranks of the "
                    f"candidates across {ctx.n_blocks} outer folds (Friedman χ²({fr['k'] - 1}) = {fr['stat']:.2f}, "
                    f"p = {fmt_p(fr['p'])}); vermillion, selected candidate; purple, meta-ensembles; grey, logistic "
                    "regression reference", supplementary=True)
