"""§12 — Desempeño y calibración por subgrupos en la reserva (Tabla S6, Fig. S3). Análisis exploratorio de equidad."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import K, dn
from ..formatting import fmt_ci, num
from ..metrics import mc_metrics, strat_boot_idx
from ..style import C_POS, C_REF, GRID, new_figure


def subgroup_levels(X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "Gender": X["Gender"].map({0.0: "Female", 1.0: "Male"}).to_numpy(),
        "Scholarship holder": X["Scholarship holder"].map({0.0: "No", 1.0: "Yes"}).to_numpy(),
        "Age at enrolment": np.where(X["Age at enrollment"] <= 20, "≤ 20 years", "> 20 years"),
        "Displaced": X["Displaced"].map({0.0: "No", 1.0: "Yes"}).to_numpy(),
        "Debtor": X["Debtor"].map({0.0: "No", 1.0: "Yes"}).to_numpy(),
    }


def run(ctx) -> None:
    cfg, ds, rep, best = ctx.cfg, ctx.ds, ctx.rep, ctx.BEST
    y_hold, P_best = ds.y_hold, ctx.P_best
    rows, forest = [], []
    for var, lev in subgroup_levels(ds.X_hold).items():
        lev = pd.Series(lev).astype(str).to_numpy()
        for L in sorted(set(lev)):
            msk = lev == L
            n_ = int(msk.sum())
            if n_ < cfg.subgroup_min_n or len(np.unique(y_hold[msk])) < K:
                rows.append({"Subgroup": var, "Level": L, "n": n_, "Macro-F1 (95% CI)": "NE"})
                continue
            mm = mc_metrics(y_hold[msk], P_best[msk], cfg.ece_bins)
            bs = np.array([mc_metrics(y_hold[msk][i], P_best[msk][i], cfg.ece_bins)["f1_macro"]
                           for i in strat_boot_idx(y_hold[msk], max(200, cfg.holdout_bootstrap // 5),
                                                   cfg.random_state + 16000)])
            lo_, hi_ = np.quantile(bs, ctx.ql), np.quantile(bs, ctx.qh)
            forest.append({"var": var, "level": L, "n": n_, "est": mm["f1_macro"], "lo": lo_, "hi": hi_})
            rows.append({"Subgroup": var, "Level": L, "n": n_, "Macro-F1 (95% CI)": fmt_ci(mm["f1_macro"], lo_, hi_),
                         "Recall Dropout": num(mm["recall_Dropout"]), "Recall Enrolled": num(mm["recall_Enrolled"]),
                         "Recall Graduate": num(mm["recall_Graduate"]), "Brier score": num(mm["brier"], 4),
                         "Class-wise ECE": num(mm["cw_ece"])})
    rep.register_table(pd.DataFrame(rows).fillna(""), "TableS6_subgroups", "S6",
                       f"Performance and calibration of the frozen {dn(best)} by subgroup (holdout)",
                       f"Exploratory fairness check; levels with n < {cfg.subgroup_min_n} or without all three classes are "
                       "not estimated (NE). 95% CIs from bootstrap resamples stratified by class within the subgroup. "
                       "Subgroup estimates are imprecise and should not be over-interpreted", supplementary=True,
                       group_col="Subgroup")
    if not forest:
        return
    fo = pd.DataFrame(forest)
    fig, ax = new_figure("single", 16 + 6.5 * (len(fo) + fo["var"].nunique()))
    ypos, labels_, y_ = [], [], 0
    for var, g in fo.groupby("var", sort=False):
        labels_.append((y_, var, True))
        y_ += 1
        for _, r in g.iterrows():
            ypos.append(y_)
            labels_.append((y_, f"{r.level} (n = {r.n})", False))
            y_ += 1
    ax.axvline(ctx.MH[best]["f1_macro"], color=C_REF, ls=(0, (3, 2)), lw=0.7, zorder=0)
    ax.errorbar(fo.est, ypos, xerr=[fo.est - fo.lo, fo.hi - fo.est], fmt="s", ms=3.5, color=C_POS, mec="black",
                mew=0.3, elinewidth=0.8, capsize=1.5)
    ax.set_yticks([p for p, _, _ in labels_], [t for _, t, _ in labels_])
    for tl, (_, _, hdr) in zip(ax.get_yticklabels(), labels_):
        tl.set_fontweight("bold" if hdr else "normal")
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 (95% CI)")
    ax.grid(axis="x", lw=0.4, color=GRID)
    ax.set_axisbelow(True)
    rep.save_figure(fig, "FigS3_subgroups",
                    f"Fig. S3 Macro-F1 of the frozen {dn(best)} by subgroup on the locked holdout with stratified "
                    "bootstrap 95% CIs; dashed line, overall holdout macro-F1. Exploratory analysis", supplementary=True)
