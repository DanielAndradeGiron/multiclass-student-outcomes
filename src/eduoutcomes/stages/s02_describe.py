"""§3 — Descripción de la muestra por desenlace (Tabla 1), auditorías (Tablas S1–S2) y diseño del estudio (Fig. 1)."""

from __future__ import annotations

import math

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats

from ..constants import (BINARY, CLASSES, K, NOMINAL, SCENARIO_SHORT, SCENARIOS, availability_of, scenario_predictors,
                         short_name)
from ..formatting import fmt_p, num
from ..style import OKABE_ITO, new_figure, tint

QUANT_T1 = ["Age at enrollment", "Admission grade", "Previous qualification (grade)", "Application order",
            "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)", "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)", "Unemployment rate", "Inflation rate", "GDP"]
BIN_T1 = ["Gender", "Scholarship holder", "Debtor", "Tuition fees up to date", "Displaced", "International",
          "Daytime/evening attendance", "Educational special needs"]


def _pct(k, n):
    return f"{k:,} ({k / n * 100:.1f})" if n else "—"


def _med(s):
    return f"{num(s.median(), 1)} ({num(s.quantile(.25), 1)}–{num(s.quantile(.75), 1)})"


def table1(ds) -> pd.DataFrame:
    df, y_all = ds.df, ds.y_all
    N = len(df)
    rows = [{"Characteristic": "Students, n (%)", "Overall": f"{N:,}",
             **{c: _pct(int((y_all == i).sum()), N) for i, c in enumerate(CLASSES)}, "p value": "",
             "Effect size": ""}]
    for c in QUANT_T1:
        groups = [df.loc[y_all == i, c].astype(float) for i in range(K)]
        H, p = stats.kruskal(*groups)
        eps2 = max(0.0, (H - K + 1) / (N - K))
        rows.append({"Characteristic": f"{c}, median (IQR)", "Overall": _med(df[c].astype(float)),
                     **{CLASSES[i]: _med(groups[i]) for i in range(K)}, "p value": fmt_p(p),
                     "Effect size": f"ε² = {eps2:.2f}"})
    for c in BIN_T1:
        tab = pd.crosstab(df[c], y_all)
        chi2, p, *_ = stats.chi2_contingency(tab, correction=False)
        v = math.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
        pos = (df[c] == 1).to_numpy()
        rows.append({"Characteristic": f"{c} = yes, n (%)" if c != "Gender" else "Male, n (%)",
                     "Overall": _pct(int(pos.sum()), N),
                     **{CLASSES[i]: _pct(int((pos & (y_all == i)).sum()), int((y_all == i).sum())) for i in range(K)},
                     "p value": fmt_p(p), "Effect size": f"V = {v:.2f}"})
    t1 = pd.DataFrame(rows)
    t1.insert(0, "Block", ["Sample"] + ["Quantitative characteristics"] * len(QUANT_T1)
              + ["Binary characteristics"] * len(BIN_T1))
    return t1


def _box(ax, xy, text, w, h, fc="white", fs=7):
    ax.add_patch(mpl.patches.FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                            boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec="black", lw=0.6))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs, linespacing=1.3)


def _arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black", mutation_scale=7,
                                                    shrinkA=0, shrinkB=0))


def figure1(ctx) -> None:
    ds, cfg = ctx.ds, ctx.cfg
    N = len(ds.df)
    cnt = np.bincount(ds.y_all, minlength=K)
    hold_fc = tint(OKABE_ITO["orange"], 0.18)
    fig, ax = new_figure("double", 100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    W_, H_ = 0.27, 0.22
    X1, X2, X3 = 0.165, 0.50, 0.835
    Y1, Y2, Y3 = 0.84, 0.50, 0.16
    e = W_ / 2
    _box(ax, (X1, Y1), f"UCI data set (n = {N:,})\nScenario {ds.scenario.replace('_', '-')}: {len(ds.cols)} predictors\n"
                       f"Dropout {cnt[0]:,} · Enrolled {cnt[1]:,}\nGraduate {cnt[2]:,}", W_, H_)
    _box(ax, (X1, Y2), f"Development set\nn = {len(ds.y_dev):,}", W_, H_)
    _box(ax, (X1, Y3), f"Locked holdout\nn = {len(ds.y_hold):,}\nopened once", W_, H_, fc=hold_fc)
    _box(ax, (X2, Y1), "Sensitivity analyses\n" + ("prediction horizon\n" if ds.scenario == "T2" else "")
         + "resampling strategy", W_, H_)
    _box(ax, (X2, Y2), f"Repeated nested CV ({cfg.outer_splits} × {cfg.outer_repeats})\nLR reference · 6 ensembles\n"
                       "soft voting · stacking\nSMOTENC + calibration in folds", W_, H_)
    _box(ax, (X2, Y3), "Final fit on development set\nall candidates frozen\nbefore holdout", W_, H_)
    _box(ax, (X3, Y1), "Explainability (RQ3)\nSHAP · permutation · LIME\nfidelity and stability", W_, H_)
    _box(ax, (X3, Y2), "Statistical comparison (RQ1)\nFriedman · Nemenyi\nNadeau–Bengio + Holm\nBayesian ROPE", W_, H_)
    _box(ax, (X3, Y3), "Holdout evaluation\ndiscrimination · calibration (RQ2)\ncapacity · subgroups", W_, H_,
         fc=hold_fc)
    _arrow(ax, (X1, Y1 - H_ / 2), (X1, Y2 + H_ / 2))
    ax.text(X1 + 0.012, (Y1 + Y2) / 2, "grouped, stratified 80/20", fontsize=7, style="italic", va="center")
    _arrow(ax, (X1 + e, Y2), (X2 - e, Y2))
    _arrow(ax, (X2, Y2 + H_ / 2), (X2, Y1 - H_ / 2))
    _arrow(ax, (X2 + e, Y2), (X3 - e, Y2))
    _arrow(ax, (X2, Y2 - H_ / 2), (X2, Y3 + H_ / 2))
    _arrow(ax, (X2 + e, Y3), (X3 - e, Y3))
    ax.plot([X1, X1, X3 - 0.07], [Y3 - H_ / 2, 0.012, 0.012], color="black", lw=0.6)
    _arrow(ax, (X3 - 0.07, 0.012), (X3 - 0.07, Y3 - H_ / 2))
    ax.plot([X3 + e, 0.993, 0.993], [Y3, Y3, Y1], color="black", lw=0.6)
    _arrow(ax, (0.993, Y1), (X3 + e, Y1))
    ax.plot([X1 - e, 0.007, 0.007], [Y1, Y1, Y3], color="black", lw=0.6)
    _arrow(ax, (0.007, Y3), (X1 - e, Y3))
    ctx.rep.save_figure(fig, "Fig1_study_design",
                        "Fig. 1 Study design. A holdout stratified by the three outcomes and grouped by predictor profile "
                        "was locked before any analysis. Encoding, SMOTENC oversampling, hyperparameter tuning, stacking "
                        "and calibration were fitted only on training folds of repeated nested cross-validation; all "
                        "candidates were refitted on the development set and frozen before the holdout was opened once. "
                        "CV, cross-validation; LR, logistic regression; RQ, research question; ROPE, region of practical "
                        "equivalence")


def run(ctx) -> None:
    ds, cfg, rep = ctx.ds, ctx.cfg, ctx.rep
    rep.register_table(table1(ds), "Table1_characteristics", "1", "Characteristics of the students by academic outcome",
                       "Quantitative variables: median (interquartile range, IQR), Kruskal–Wallis test, ε² effect size. "
                       "Binary variables: n (column %), Pearson χ² test without continuity correction, Cramér's V. "
                       "Descriptive statistics use the full sample (development and holdout). Exact duplicate records "
                       f"removed: {ds.n_dup}", landscape=True, group_col="Block")
    df, ap = ds.df, ds.all_pred
    audit = pd.DataFrame({
        "Variable": ap,
        "Type": ["nominal" if c in NOMINAL else "binary" if c in BINARY else "quantitative" for c in ap],
        "Missing": [int(df[c].isna().sum()) for c in ap],
        "Unique values": [int(df[c].nunique()) for c in ap],
        "Time of measurement (declared)": [availability_of(c)[0] for c in ap],
        "Certainty": [availability_of(c)[1] for c in ap],
        "Source": [availability_of(c)[2] for c in ap],
        "Admitted in scenarios": [", ".join(SCENARIO_SHORT[s] for s in SCENARIOS if c in scenario_predictors(ap, s))
                                  for c in ap],
        "In this run": ["yes" if c in ds.cols else "no" for c in ap],
        "Figure label": [short_name(c) for c in ap]})
    rep.register_table(audit, "TableS1_data_audit", "S1", "Predictor audit: type, completeness, time of measurement and "
                       "prediction scenarios",
                       "Nominal predictors are integer-coded in the source file and were one-hot encoded within training "
                       f"folds (categories with < {cfg.min_category_frequency} training records pooled as infrequent). "
                       "Time of measurement follows the data descriptor (Realinho et al., 2022), which states that enrolment "
                       "information includes demographic, socioeconomic and macroeconomic factors but does not give the "
                       "date on which debtor status, tuition-fee status and scholarship status were recorded (certainty "
                       "'unknown'); T0-strict therefore excludes them. Macroeconomic values are annual national "
                       "indicators whose 10 distinct combinations match Portugal 2008–2017 (certainty 'inferred'). "
                       "Scenarios: T0-strict ⊂ T0 (enrolment) ⊂ T1 (end of 1st semester) ⊂ T2 (end of 1st year); no "
                       "scenario uses information recorded after its time of prediction. Figure label: abbreviation "
                       "used in the figures (CU1/CU2, curricular units in the first/second semester)",
                       supplementary=True)
    rep.register_table(ds.profile_audit, "TableS2_profile_audit", "S2", "Audit of predictor-profile multiplicity",
                       "A profile is the full vector of the 36 predictors. All partitions (holdout, outer and inner "
                       "cross-validation, calibration and stacking folds) were grouped by profile so that identical "
                       "profiles never appear on both sides of a split", supplementary=True)
    # v3.0 — cohortes inferidas (descriptivo, muestra completa) y relación de las variables de momento incierto con el
    # rendimiento académico posterior (orientativo: no permite fechar la medición)
    coh = pd.Series(ds.cohort, name="cohort")
    rows = []
    for yr in sorted(coh.unique()):
        m = (coh == yr).to_numpy()
        rows.append({"Inferred cohort": "unmatched" if yr < 0 else str(int(yr)), "Students": f"{int(m.sum()):,}",
                     **{c: _pct(int((ds.y_all[m] == i).sum()), int(m.sum())) for i, c in enumerate(CLASSES)},
                     "Unemployment rate (%)": num(df.loc[m, "Unemployment rate"].iloc[0], 1),
                     "Inflation rate (%)": num(df.loc[m, "Inflation rate"].iloc[0], 1),
                     "GDP growth (%)": num(df.loc[m, "GDP"].iloc[0], 2)})
    tab = pd.crosstab(coh, ds.y_all)
    chi2, p, *_ = stats.chi2_contingency(tab, correction=False)
    rep.register_table(pd.DataFrame(rows), "TableS15_inferred_cohorts", "S15",
                       "Academic outcome by enrolment cohort inferred from the macroeconomic predictors",
                       "n (row %). The source does not report the year of enrolment; each of the 10 distinct combinations "
                       "of unemployment rate, inflation rate and GDP growth matches one year of the Portuguese annual "
                       "series 2008–2017 (PORDATA), which identifies the cohort. The mapping is an inference to be "
                       f"verified against PORDATA/INE. Pearson χ² test of outcome by cohort: p = {fmt_p(p)}",
                       supplementary=True)
    rows = []
    for c in ["Debtor", "Tuition fees up to date", "Scholarship holder"]:
        for v in (0, 1):
            m = (df[c] == v).to_numpy()
            rows.append({"Predictor": c, "Value": "yes" if v else "no", "Students": f"{int(m.sum()):,}",
                         "Dropout, %": num(100 * (ds.y_all[m] == 0).mean(), 1),
                         "No CU approved in 1st semester, %": num(
                             100 * (df.loc[m, "Curricular units 1st sem (approved)"] == 0).mean(), 1),
                         "No CU approved in 2nd semester, %": num(
                             100 * (df.loc[m, "Curricular units 2nd sem (approved)"] == 0).mean(), 1)})
    rep.register_table(pd.DataFrame(rows), "TableS16_timing_uncertain_predictors", "S16",
                       "Predictors with undocumented time of measurement: association with the outcome and with "
                       "first-year academic results",
                       "Full sample. Strong associations with later academic results are compatible with these statuses "
                       "having been recorded after enrolment (temporal leakage, possibly reverse causality), but cannot "
                       "date the measurement; hence T0 is reported with (T0) and without (T0-strict) them. CU, curricular "
                       "units", supplementary=True)
    figure1(ctx)
