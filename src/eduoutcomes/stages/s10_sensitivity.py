"""§13 — Análisis de sensibilidad preespecificados (horizonte de predicción y tratamiento del desbalance).

Tabla 9, Tabla S10 (contrastes pareados) y Fig. 7. Cada (configuración × pliegue) es independiente y se ejecuta en
paralelo; el resultado se guarda en caché.
"""

from __future__ import annotations

import logging
import math

import joblib
import matplotlib as mpl
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from ..constants import CLASSES, LABELS, dn
from ..formatting import fmt_ci, fmt_ms, fmt_p, num
from ..inference import holm
from ..metrics import mc_metrics
from ..models import cache_reset, gsplits
from ..style import C_CLASS, C_REF, GRID, M_CLASS, OKABE_ITO, new_canvas, panel_label, tint

log = logging.getLogger(__name__)

ANALYSES = ["Prediction horizon", "Imbalance handling"]
PRIMARY = {"Prediction horizon": "End of first year (all)", "Imbalance handling": "SMOTENC (primary)"}
CLAB = {"Enrolment only": "Enrolment only vs end of first year",
        "Enrolment + 1st semester": "Enrolment + 1st semester vs end of first year",
        "Random oversampling": "Random oversampling vs SMOTENC", "No resampling": "No resampling vs SMOTENC"}
TICKS = {"Enrolment only": "Enrolment\nonly", "Enrolment + 1st semester": "Enrolment +\n1st semester",
         "End of first year (all)": "End of\nfirst year", "SMOTENC (primary)": "SMOTENC",
         "Random oversampling": "Random\noversampling", "No resampling": "No\nresampling"}
DELTA_METS = ["f1_macro", "recall_Enrolled", "cw_ece"]
T9_METS = [("f1_macro", "Macro-F1", "max", 3), ("mcc", "MCC", "max", 3), ("roc_auc_macro", "Macro AUC", "max", 3),
           ("recall_Dropout", "Recall Dropout", "max", 3), ("recall_Enrolled", "Recall Enrolled", "max", 3),
           ("recall_Graduate", "Recall Graduate", "max", 3), ("brier", "Brier score", "min", 4),
           ("cw_ece", "Class-wise ECE", "min", 3)]


def run_sens(u, b, cols, rs, tr, va, X_dev, y_dev, g_dev, factory, model):
    """Un ajuste (configuración u × pliegue b), con reajuste y recalibración en el pliegue de entrenamiento."""
    cfg = factory.cfg
    factory.reset_cache()
    cal, *_ = factory.fit_candidate(model, X_dev.iloc[tr], y_dev[tr], cfg.random_state + 20000 + 100 * b, cols,
                                    resampling=rs, groups=g_dev[tr], n_jobs=1)
    out = {"u": u, "fold": b + 1, **mc_metrics(y_dev[va], cal.predict_proba(X_dev.iloc[va]), cfg.ece_bins)}
    cache_reset()
    return out


def nb_paired(d, ratio: float, alpha: float = 0.05):
    """t corregido de Nadeau–Bengio sobre diferencias pareadas por pliegue: (media, lo, hi, p)."""
    d = np.asarray(d, float)
    n = len(d)
    mu = float(d.mean())
    v = d.var(ddof=1) if n > 1 else np.nan
    se = math.sqrt((1 / n + ratio) * v) if n > 1 and v > 0 else np.nan
    if not np.isfinite(se):
        return mu, np.nan, np.nan, np.nan
    tq = stats.t.ppf(1 - alpha / 2, n - 1)
    return mu, mu - tq * se, mu + tq * se, float(2 * stats.t.sf(abs(mu / se), n - 1))


def run(ctx) -> None:
    cfg, ds, rep, fac = ctx.cfg, ctx.ds, ctx.rep, ctx.factory
    model = ctx.BEST
    s_splits = [sp for r in range(cfg.sensitivity_repeats)
                for sp in gsplits(cfg.sensitivity_splits, ds.y_dev, ds.g_dev, cfg.random_state + 300 + r)]
    cols_all = ds.cols
    settings = [("Prediction horizon", nm, cols, cfg.resampling) for nm, cols in ds.pred_sets.items()] + \
               [("Imbalance handling", lab, cols_all, rs) for lab, rs in [("SMOTENC (primary)", cfg.resampling),
                                                                        ("Random oversampling", "random_oversampling"),
                                                                        ("No resampling", "none")]]
    uniq = list(dict.fromkeys((tuple(cols), rs) for _, _, cols, rs in settings))  # el ajuste primario se hace una vez
    cache = ctx.paths.cache / (f"sensitivity_{ctx.key_hash}_{model}_{cfg.sensitivity_splits}x"
                               f"{cfg.sensitivity_repeats}.joblib")
    tasks = [(u, b) for u in range(len(uniq)) for b in range(len(s_splits))]
    if cfg.reuse_cache and cache.exists():
        sens_raw = joblib.load(cache)
        log.info("Sensitivity analyses loaded from cache")
    else:
        X_dev, y_dev, g_dev = ds.X_dev, ds.y_dev, ds.g_dev
        sens_raw = Parallel(n_jobs=min(ctx.n_workers, len(tasks)), verbose=10)(
            delayed(run_sens)(u, b, list(uniq[u][0]), uniq[u][1], *s_splits[b], X_dev, y_dev, g_dev, fac, model)
            for u, b in tasks)
        joblib.dump(sens_raw, cache)
    by_u = pd.DataFrame(sens_raw)
    sens = pd.concat([by_u[by_u.u == uniq.index((tuple(cols), rs))].assign(Analysis=an, Setting=st)
                      for an, st, cols, rs in settings], ignore_index=True).drop(columns="u")
    ctx.sens = sens

    def fold_values(an, st, m):
        """Valores por pliegue ordenados (todos los ajustes usan las MISMAS particiones: diseño pareado)."""
        return sens[(sens.Analysis == an) & (sens.Setting == st)].sort_values("fold")[m].to_numpy()

    # ---------------------------------------------------- Tabla 9 (media ± DE; negrita = mejor por análisis)
    t9, bold9 = [], set()
    for an in ANALYSES:
        sts = [st for a_, st, *_ in settings if a_ == an]
        first = len(t9)
        for st in sts:
            row = {"Analysis": an, "Setting": st + (" †" if st == PRIMARY[an] and "primary" not in st else "")}
            for m, lab, _, d in T9_METS:
                v = fold_values(an, st, m)
                row[lab] = fmt_ms(v.mean(), v.std(ddof=1), d) if len(v) > 1 else num(v.mean(), d)
            t9.append(row)
        for m, lab, direction, _ in T9_METS:
            means = np.array([fold_values(an, st, m).mean() for st in sts])
            bst = np.nanmax(means) if direction == "max" else np.nanmin(means)
            bold9 |= {(first + int(i), lab) for i in np.where(np.isclose(means, bst))[0]}
    rep.register_table(
        pd.DataFrame(t9), "Table9_sensitivity", "9", f"Prespecified sensitivity analyses for {dn(model)} (development set)",
        f"Mean ± SD over {len(s_splits)} folds of stratified, profile-grouped cross-validation; bold, best value within "
        "each analysis; † primary setting. In every setting the model was re-tuned and re-calibrated (temperature "
        "scaling) within each training fold. Enrolment-only predictors exclude all curricular-unit variables; the "
        "first-semester horizon adds the six first-semester variables. Paired contrasts against the primary setting are "
        "reported in Table S10. For Brier score and ECE, lower is better", landscape=True, group_col="Analysis",
        bold=bold9)

    # ---------------------------------------------------- Tabla S10: contrastes pareados frente al ajuste primario
    sens_ratio = 1 / (cfg.sensitivity_splits - 1)
    contrasts = [(an, st) for an, st, *_ in settings if st != PRIMARY[an]]
    dl = []
    for m in DELTA_METS:
        r_ = []
        for an, st in contrasts:
            mu, lo, hi, p = nb_paired(fold_values(an, st, m) - fold_values(an, PRIMARY[an], m), sens_ratio, cfg.alpha)
            r_.append({"metric": m, "analysis": an, "setting": st, "diff": mu, "lo": lo, "hi": hi, "p": p})
        r_ = pd.DataFrame(r_)
        r_["p_holm"] = holm(r_.p.fillna(1.0).to_numpy())
        dl.append(r_)
    sens_delta = pd.concat(dl, ignore_index=True)
    ctx.sens_delta = sens_delta
    t_s10 = pd.DataFrame([{"Analysis": an, "Contrast": CLAB.get(st, st), "Metric": LABELS[m],
                           "Mean difference (95% CI)": fmt_ci(r["diff"], r.lo, r.hi, 3, signed=True),
                           "p": fmt_p(r.p), "Holm p": fmt_p(r.p_holm)}
                          for an in ANALYSES for st in [s for a_, s in contrasts if a_ == an] for m in DELTA_METS
                          for _, r in sens_delta[(sens_delta.analysis == an) & (sens_delta.setting == st)
                                                 & (sens_delta.metric == m)].iterrows()])
    rep.register_table(
        t_s10, "TableS10_sensitivity_contrasts", "S10",
        f"Paired differences between each sensitivity setting and the primary setting ({dn(model)}, development set)",
        f"Differences (setting − primary) across the same {len(s_splits)} folds; Nadeau–Bengio corrected t test "
        f"(n_test/n_train = {sens_ratio:.2f}) with Holm adjustment within each metric over the {len(contrasts)} "
        "contrasts. For class-wise ECE, negative differences indicate better calibration. Cross-validation folds are not "
        "independent, so the tests are approximate", supplementary=True, group_col="Analysis")
    figure7(ctx, settings, contrasts, sens_delta, fold_values, len(s_splits), model)


def figure7(ctx, settings, contrasts, sens_delta, fold_values, n_folds, model) -> None:
    cfg = ctx.cfg
    families = [("Discrimination", "Score", [("f1_macro", "o", OKABE_ITO["black"]), ("mcc", "D", OKABE_ITO["green"]),
                                             ("roc_auc_macro", "v", OKABE_ITO["purple"])]),
                ("Recall by class", "Recall", [(f"recall_{c}", M_CLASS[c], C_CLASS[c]) for c in CLASSES]),
                ("Calibration error", "ECE (lower is better)", [("cw_ece", "D", C_REF),
                                                                ("ece_Enrolled", "h", C_CLASS["Enrolled"])])]
    band = tint(OKABE_ITO["yellow"], 0.25)
    fig = new_canvas("double", 200, wspace=0.03, hspace=0.025)
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 0.17, 0.95])
    axes = np.empty((2, 3), dtype=object)
    for r in range(2):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c], sharey=axes[0, c] if r else None)
    handles = {}
    for r, an in enumerate(ANALYSES):
        st_ = [st for a_, st, *_ in settings if a_ == an]
        xs_ = np.arange(len(st_))
        ip = st_.index(PRIMARY[an])
        for c, (_fam, ylab, mets) in enumerate(families):
            ax = axes[r, c]
            ax.axvspan(ip - 0.45, ip + 0.45, color=band, lw=0, zorder=0)
            w = 0.24 if len(mets) == 3 else 0.28
            for j, (m, mk_, col) in enumerate(mets):
                off = (j - (len(mets) - 1) / 2) * w
                vals = [fold_values(an, s_, m) for s_ in st_]
                for i, v in enumerate(vals):
                    ax.scatter(np.full(len(v), i + off), v, s=6, color=tint(col, 0.40), lw=0, zorder=2)
                mu = np.array([v.mean() for v in vals])
                sd = np.array([v.std(ddof=1) if len(v) > 1 else 0.0 for v in vals])
                if an == "Prediction horizon":  # eje ordinal: se unen las medias
                    ax.plot(xs_ + off, mu, "-", color=col, lw=0.7, zorder=3)
                handles.setdefault(m, ax.errorbar(xs_ + off, mu, yerr=sd, fmt=mk_, ms=4, color=col, mec="black",
                                                  mew=0.35, elinewidth=0.8, capsize=1.5, zorder=4))
            ax.set_xticks(xs_, [TICKS.get(s_, s_) for s_ in st_])
            ax.set_xlim(-0.5, len(st_) - 0.5)
            ax.tick_params(axis="x", length=0)
            ax.set_ylabel(ylab)
            ax.grid(axis="y", lw=0.4, color=GRID)
            ax.set_axisbelow(True)
            if c == 1:
                ax.set_title(an)
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 2].set_ylim(bottom=0)
    for (r, c), letter in zip([(r, c) for r in range(2) for c in range(3)], "abcdef"):
        panel_label(axes[r, c], letter)
    lax = fig.add_subplot(gs[2, :])
    lax.axis("off")
    order_h = [m for _, _, mets in families for m, *_ in mets]
    lax.legend([handles[m] for m in order_h] + [mpl.patches.Patch(color=band)],
               [LABELS[m] for m in order_h] + ["Primary setting"], loc="center", ncol=5, handletextpad=0.4,
               columnspacing=1.4, handlelength=1.4)
    fg = gs[3, :].subgridspec(1, 2)
    ag = fig.add_subplot(fg[0, 0])
    ah = fig.add_subplot(fg[0, 1], sharey=ag)
    ypos, y_ = {}, 0.0
    for an in ANALYSES:
        for a_, st in contrasts:
            if a_ == an:
                ypos[(an, st)] = y_
                y_ += 1
        y_ += 0.6
    an_style = {"Prediction horizon": (OKABE_ITO["blue"], "o"), "Imbalance handling": (OKABE_ITO["vermillion"], "s")}

    def forest(ax, metric, xlabel, rope=None):
        q = sens_delta[sens_delta.metric == metric]
        if rope:
            ax.axvspan(-rope, rope, color=tint(OKABE_ITO["yellow"], 0.45), lw=0, zorder=0)
        ax.axvline(0, color="black", lw=0.6, zorder=1)
        for _, r in q.iterrows():
            y0 = ypos[(r.analysis, r.setting)]
            col, mk_ = an_style[r.analysis]
            err = [[r["diff"] - r.lo], [r.hi - r["diff"]]] if np.isfinite(r.lo) else None
            ax.errorbar(r["diff"], y0, xerr=err, fmt=mk_, ms=4.2, color=col, mec="black", mew=0.35, elinewidth=0.9,
                        capsize=1.8, zorder=3)
            if np.isfinite(r.p_holm) and r.p_holm < cfg.alpha:
                ax.annotate("*", (r.hi if np.isfinite(r.hi) else r["diff"], y0), xytext=(2, -1),
                            textcoords="offset points", va="center", fontsize=8)
        sep = max(v for (a_, _), v in ypos.items() if a_ == ANALYSES[0]) + 0.8
        ax.axhline(sep, color="#BBBBBB", lw=0.5, zorder=1)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", lw=0.4, color=GRID)
        ax.set_axisbelow(True)

    forest(ag, "f1_macro", "Δ macro-F1 (setting − primary; 95% CI)", rope=cfg.practical_margin_f1m)
    forest(ah, "cw_ece", "Δ class-wise ECE (setting − primary; 95% CI)")
    ag.set_yticks(list(ypos.values()), [CLAB.get(st, st) for (_, st) in ypos])
    ag.set_ylim(max(ypos.values()) + 0.6, -0.6)
    ag.tick_params(axis="y", length=0)
    ah.tick_params(axis="y", labelleft=False, length=0)
    panel_label(ag, "g")
    panel_label(ah, "h")
    ctx.rep.save_figure(
        fig, "Fig7_sensitivity",
        f"Fig. 7 Prespecified sensitivity analyses for {dn(model)} in profile-grouped cross-validation on the "
        f"development set ({n_folds} folds). a–c Prediction horizon and d–f imbalance handling: discrimination "
        "(macro-F1, Matthews correlation coefficient and macro AUC), recall by class, and calibration error (class-wise "
        "ECE and ECE for Enrolled; lower is better). Light dots, individual folds; markers and whiskers, mean ± SD; "
        "shaded column, primary setting. g, h Paired differences between each setting and the primary setting in "
        f"macro-F1 and class-wise ECE with Nadeau–Bengio corrected 95% CIs; shaded band in g, practical margin "
        f"(± {cfg.practical_margin_f1m:.2f}); * Holm-adjusted p < 0.05 (Table S10). The model was re-tuned and "
        "re-calibrated within every training fold. AUC, area under the ROC curve; ECE, expected calibration error")
