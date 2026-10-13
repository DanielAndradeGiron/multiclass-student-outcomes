"""§11 — Uso orientado a la decisión: capacidad limitada, recuentos esperados y curva de decisión (Tabla 7, S9, Fig. 4)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..constants import CLASSES, dn
from ..formatting import fmt_ci, num
from ..metrics import clopper, net_benefit
from ..style import C_NEG, C_POS, C_REF, GRID, new_figure, panel_label


def run(ctx) -> None:
    cfg, ds, rep, best = ctx.cfg, ctx.ds, ctx.rep, ctx.BEST
    y_hold, P_best, P_ref = ds.y_hold, ctx.P_best, ctx.P_ref
    n_h = len(y_hold)
    is_d, is_e = (y_hold == 0), (y_hold == 1)
    rows7 = []
    for score_name, sc_fn, target in [("P(Dropout)", lambda P: P[:, 0], is_d),
                                      ("P(Dropout) + P(Enrolled)", lambda P: P[:, 0] + P[:, 1], is_d | is_e)]:
        for cap in cfg.capacity_levels:
            n_flag = int(math.ceil(cap * n_h))
            for lab, P_ in [(dn(best), P_best), ("LR reference", P_ref)]:
                top = np.argsort(-sc_fn(P_), kind="mergesort")[:n_flag]
                fl = np.zeros(n_h, bool)
                fl[top] = True
                hit = int((fl & target).sum())
                n_t = int(target.sum())
                rows7.append({
                    "Score": f"Ranking by {score_name}", "Capacity": f"{cap:.0%} (n = {n_flag})", "Model": lab,
                    "Target captured, % (95% CI)": fmt_ci(hit / n_t * 100, *(np.array(clopper(hit, n_t)) * 100), 1),
                    "Precision, % (95% CI)": fmt_ci(hit / n_flag * 100, *(np.array(clopper(hit, n_flag)) * 100), 1),
                    "Dropouts captured": f"{int((fl & is_d).sum())}/{int(is_d.sum())}",
                    "Enrolled captured": f"{int((fl & is_e).sum())}/{int(is_e.sum())}"})
    rep.register_table(pd.DataFrame(rows7), "Table7_capacity", "7",
                       "Capacity-constrained prioritisation of students on the locked holdout",
                       "Students were ranked by the calibrated probability and the top fraction was flagged for support. "
                       "Target: dropouts for P(Dropout); dropouts and still-enrolled students (not graduating on time) for "
                       "P(Dropout) + P(Enrolled). Exact Clopper–Pearson 95% CIs. Capacities were prespecified; none was "
                       "chosen using the holdout", landscape=True, group_col="Score")
    exp_rows = []
    for lab, P_ in [(f"{dn(best)} (temperature scaling)", P_best), (f"{dn(best)} (uncalibrated)", ctx.PH_unc[best]),
                    ("LR reference", P_ref)]:
        for i, c in enumerate(CLASSES):
            e_, o_ = P_[:, i].sum(), int((y_hold == i).sum())
            exp_rows.append({"Model": lab, "Class": c, "Observed": o_, "Expected (Σ p)": num(e_, 1),
                             "Expected/observed": num(e_ / o_)})
    rep.register_table(pd.DataFrame(exp_rows), "TableS9_expected_vs_observed", "S9",
                       "Expected versus observed number of students per class on the holdout",
                       "Expected counts are sums of predicted probabilities; ratios close to 1 indicate "
                       "calibration-in-the-large, which determines whether probabilities can be used to plan support "
                       "capacity", supplementary=True, group_col="Model")

    fig, (a1, a2) = new_figure("double", 72, ncols=2)
    xs = np.arange(1, n_h + 1) / n_h * 100
    for cap in cfg.capacity_levels:
        a1.axvline(cap * 100, color="#C8C8C8", lw=0.5, zorder=0)
    styles = [(dn(best), P_best, C_POS, "-"), ("LR reference", P_ref, C_NEG, (0, (4, 2)))]
    for lab, P_, col, ls in styles:
        o = np.argsort(-P_[:, 0], kind="mergesort")
        a1.plot(xs, np.cumsum(is_d[o]) / is_d.sum() * 100, color=col, ls=ls, lw=1.1, label=lab)
    a1.plot([0, 100], [0, 100], color=C_REF, ls=":", lw=0.8, label="Random ranking")
    a1.set(xlabel="Students flagged (%)", ylabel="Dropouts captured (%)", xlim=(0, 100), ylim=(0, 101))
    a1.legend(loc="lower right")
    ts = np.round(np.arange(0.05, 0.81, 0.01), 2)
    yb = is_d.astype(int)
    prev = yb.mean()
    for lab, P_, col, ls in styles:
        a2.plot(ts, [net_benefit(yb, P_[:, 0], t) for t in ts], color=col, ls=ls, lw=1.1, label=lab)
    a2.plot(ts, prev - (1 - prev) * ts / (1 - ts), color=C_REF, ls="-.", lw=0.8, label="Intervene in all")
    a2.axhline(0, color="black", lw=0.7, label="Intervene in none")
    a2.set(xlabel="Threshold probability of dropout", ylabel="Net benefit", xlim=(0.05, 0.8), ylim=(-0.05, prev + 0.04))
    a2.legend(loc="upper right")
    for a_ in (a1, a2):
        a_.grid(lw=0.4, color=GRID)
        a_.set_axisbelow(True)
    panel_label(a1, "a")
    panel_label(a2, "b")
    rep.save_figure(fig, "Fig4_decision",
                    "Fig. 4 Decision-oriented use of calibrated probabilities on the locked holdout. a Cumulative share of "
                    f"dropouts captured when students are ranked by P(Dropout) ({dn(best)}, solid; logistic regression "
                    "reference, dashed); vertical lines mark the prespecified capacities. b Exploratory decision curve for "
                    "intervening in students at predicted risk of dropout (one-vs-rest)")
