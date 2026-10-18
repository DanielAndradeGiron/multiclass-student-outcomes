"""Escenarios temporales de predicción (v3.0): ejecución por escenario y comparación entre escenarios.

Cada escenario (T0-strict, T0, T1, T2) se ejecuta con el *pipeline* completo y el MISMO protocolo — mismas particiones
(incluida la reserva sellada), mismos modelos, rejillas, remuestreo, calibración y pruebas —, reajustando y recalibrando
todos los modelos con los predictores disponibles en su momento de predicción. Después, ``compare`` contrasta los
escenarios:

* en validación cruzada: diferencias pareadas por pliegue (prueba t corregida de Nadeau–Bengio, ajuste de Holm dentro
  de cada métrica, prueba t bayesiana correlacionada y la misma regla ROPE que en la PI1);
* en la reserva: *bootstrap* pareado y estratificado sobre los mismos estudiantes;
* en explicabilidad: cómo cambia la importancia SHAP de los predictores al incorporar información académica.

Contrastes preespecificados: T0 − T0-strict (efecto de las variables de momento incierto), T1 − T0, T2 − T1 y T2 − T0.
El modelo comparado es el seleccionado en cada escenario por la regla preespecificada (solo con la CV), y, como
referencia de modelo fijo, la LR multinomial calibrada.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

from .config import Config, load_config
from .constants import CLASSES, LABELS, REF, SCENARIO_LABEL, SCENARIO_MOMENT, SCENARIO_SHORT, SCENARIOS, dn
from .formatting import fmt_ci, fmt_ms, fmt_p, num
from .inference import bayes_rope, holm, nb_test, verdict
from .metrics import mc_metrics, strat_boot_idx
from .paths import Paths
from .reporting import Reporter
from .style import C_REF, GRID, OKABE_ITO, apply_style, new_canvas, panel_label, tint

log = logging.getLogger(__name__)

CONTRASTS = [("T0", "T0_strict"), ("T1", "T0"), ("T2", "T1"), ("T2", "T0")]
CMP_METRICS = ["f1_macro", "mcc", "recall_Enrolled", "cw_ece"]
TAB_METRICS = [("f1_macro", 3), ("balanced_accuracy", 3), ("mcc", 3), ("roc_auc_macro", 3), ("pr_auc_macro", 3),
               ("recall_Enrolled", 3), ("f1_Enrolled", 3), ("brier", 4), ("cw_ece", 3), ("ece_Enrolled", 3)]
SC_COLOR = {"T0_strict": OKABE_ITO["sky"], "T0": OKABE_ITO["blue"], "T1": OKABE_ITO["orange"],
            "T2": OKABE_ITO["vermillion"]}


# =============================================================================================
# Ejecución
# =============================================================================================
def scenario_config(base: Config | str | Path, scenario: str, overrides: list[str] | None = None,
                    quick: bool | None = None) -> Config:
    """Configuración de un escenario: mismo diseño, salida en ``<output_dir>/<escenario>``."""
    cfg0 = base if isinstance(base, Config) else load_config(base, overrides, quick=quick)
    d = cfg0.to_dict()
    d.update(scenario=scenario, output_dir=str(Path(cfg0.output_dir) / scenario))
    return Config(**d)


def run_scenarios(cfg: Config, until: str | None = None) -> Path:
    """Ejecuta el *pipeline* en cada escenario de ``cfg.scenarios`` y, si se completa, la comparación."""
    from .pipeline import run

    for s in cfg.scenarios:
        log.info("================ Scenario %s ================", SCENARIO_LABEL[s])
        run(scenario_config(cfg, s), until=until)
    if until is None:
        return compare(cfg)
    return Path(cfg.output_dir)


# =============================================================================================
# Lectura de resultados por escenario
# =============================================================================================
def _load(root: Path, s: str, cfg: Config) -> dict:
    meta = root / s / "metadata"
    if not (meta / "frozen_configuration_before_holdout.json").exists():
        raise FileNotFoundError(f"Scenario {s} has not been run to completion ({meta})")
    frozen = json.loads((meta / "frozen_configuration_before_holdout.json").read_text(encoding="utf-8"))
    split = json.loads((meta / "split_manifest.json").read_text(encoding="utf-8"))
    folds = pd.read_csv(meta / "nested_cv_fold_metrics.csv")
    folds = folds[folds.calibration == cfg.calibration_primary]
    oof = pd.read_csv(meta / "nested_cv_oof_predictions.csv.gz", usecols=["block", "model", "calibration", "row_id"])
    hold = pd.read_csv(meta / "holdout_predictions_all_models.csv.gz")
    imp_path = meta / "importance_all.csv"
    imp = pd.read_csv(imp_path, index_col=0) if imp_path.exists() else None
    status_path = meta / "explainability_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    return {"best": frozen["selected_model"], "split": split, "folds": folds, "oof": oof, "hold": hold, "imp": imp,
            "status": status, "n_pred": int(split.get("n_predictors", 0)), "quick": bool(cfg.quick)}


def _probs(hold: pd.DataFrame, model: str) -> np.ndarray:
    return hold[[f"{model}|p_{c}" for c in CLASSES]].to_numpy()


# =============================================================================================
# Comparación
# =============================================================================================
def compare(cfg: Config) -> Path:  # noqa: C901 — informe lineal
    root = Path(cfg.output_dir)
    scen = [s for s in SCENARIOS if s in cfg.scenarios]
    if len(scen) < 2:
        raise ValueError("At least two scenarios are needed for a comparison (config field `scenarios`)")
    R = {s: _load(root, s, cfg) for s in scen}
    apply_style()
    paths = Paths(root / "comparison").make()
    rep = Reporter(cfg, paths)

    # --- puertas de calidad: mismo diseño en todos los escenarios ---
    hashes = {s: R[s]["split"]["holdout_ids_sha256"] for s in scen}
    assert len(set(hashes.values())) == 1, f"Holdouts differ between scenarios: {hashes}"
    ref_oof = None
    for s in scen:
        o = R[s]["oof"]
        o = o[(o.model == REF) & (o.calibration == cfg.calibration_primary)]
        sets = o.groupby("block").row_id.apply(lambda v: tuple(sorted(v)))
        ref_oof = sets if ref_oof is None else ref_oof
        assert sets.equals(ref_oof), f"Outer validation folds differ in scenario {s}"
    y_hold = R[scen[0]]["hold"].observed.to_numpy().astype(int)
    for s in scen:
        assert np.array_equal(R[s]["hold"].observed.to_numpy().astype(int), y_hold)
        assert np.array_equal(R[s]["hold"].row_id.to_numpy(), R[scen[0]]["hold"].row_id.to_numpy())
    n_blocks = R[scen[0]]["folds"].block.nunique()
    test_ratio = 1 / (cfg.outer_splits - 1)

    def fold_vec(s, model, metric):
        f = R[s]["folds"]
        return f[f.model == model].sort_values("block")[metric].to_numpy()

    # --- Tabla 10: rendimiento por escenario (CV y reserva) ---
    boots = list(strat_boot_idx(y_hold, cfg.holdout_bootstrap, cfg.random_state + 31000))
    rows10, rows11, BOOT = [], [], {}
    for s in scen:
        for role, m in [("Selected", R[s]["best"]), ("Reference", REF)]:
            key = (s, role)
            P = _probs(R[s]["hold"], m)
            MH = mc_metrics(y_hold, P, cfg.ece_bins)
            BOOT[key] = pd.DataFrame([mc_metrics(y_hold[i], P[i], cfg.ece_bins) for i in boots])
            BOOT[key].attrs["point"] = MH
            lab = f"{dn(m)}{' (selected)' if role == 'Selected' else ''}"
            r10 = {"Scenario": SCENARIO_LABEL[s], "Predictors": R[s]["n_pred"], "Model": lab}
            r11 = dict(r10)
            for met, d in TAB_METRICS:
                v = fold_vec(s, m, met)
                r10[LABELS[met]] = fmt_ms(v.mean(), v.std(ddof=1), d)
                b = BOOT[key][met]
                r11[LABELS[met]] = fmt_ci(MH[met], b.quantile(cfg.alpha / 2), b.quantile(1 - cfg.alpha / 2), d)
            rows10.append(r10)
            rows11.append(r11)
    defs = ("T0-strict, enrolment information whose time of measurement is documented; T0, enrolment information "
            "including debtor, tuition-fee and scholarship status (time of measurement undocumented); T1, T0 + six "
            "first-semester curricular-unit variables; T2, T1 + six second-semester variables (all 36 predictors). "
            "Every model was re-tuned, re-trained with SMOTENC and re-calibrated (temperature scaling) in each scenario "
            "with the same partitions")
    rep.register_table(
        pd.DataFrame(rows10), "Table10_scenarios_cv", "10",
        "Performance and calibration by prediction scenario in repeated nested cross-validation (development set)",
        f"Mean ± SD over the same {n_blocks} outer folds in every scenario. Selected: candidate chosen in each scenario by "
        f"the prespecified tolerance hierarchy (cross-validation only). {defs}. AUC, area under the ROC curve "
        "(one-vs-rest); PR-AUC, area under the precision–recall curve (average precision, one-vs-rest); ECE, expected "
        f"calibration error ({cfg.ece_bins} equal-width bins); MCC, Matthews correlation coefficient",
        landscape=True, group_col="Scenario")
    rep.register_table(
        pd.DataFrame(rows11), "Table11_scenarios_holdout", "11",
        f"Performance and calibration by prediction scenario on the locked holdout (n = {len(y_hold):,})",
        f"Point estimates with 95% stratified bootstrap percentile intervals ({cfg.holdout_bootstrap:,} resamples). The "
        f"same {len(y_hold):,} students form the holdout in every scenario (identical SHA-256 of identifiers). {defs}. "
        "The holdout is a random partition of the same retrospective cohorts and is not an external, temporal or "
        "prospective validation. ECE is positively biased in bootstrap resamples, so its percentile interval may lie "
        "above the point estimate", landscape=True, group_col="Scenario")

    # --- Tabla 12: contrastes entre escenarios ---
    margin = {"f1_macro": cfg.practical_margin_f1m, "mcc": cfg.practical_margin_mcc}
    pairs = [c for c in CONTRASTS if c[0] in scen and c[1] in scen]
    if not pairs:   # subconjunto de escenarios no previsto: contrastes entre escenarios consecutivos
        pairs = [(scen[i + 1], scen[i]) for i in range(len(scen) - 1)]
    crow = []
    for role in ("Selected", "Reference"):
        for met in CMP_METRICS:
            tmp = []
            for a, b in pairs:
                ma = R[a]["best"] if role == "Selected" else REF
                mb = R[b]["best"] if role == "Selected" else REF
                d = fold_vec(a, ma, met) - fold_vec(b, mb, met)
                mu, lo, hi, t, p = nb_test(d, test_ratio, cfg.alpha)
                pw, pr, pb = bayes_rope(d, margin.get(met, np.nan), test_ratio) if met in margin else (np.nan,) * 3
                bd = BOOT[(a, role)][met].to_numpy() - BOOT[(b, role)][met].to_numpy()
                hd = BOOT[(a, role)].attrs["point"][met] - BOOT[(b, role)].attrs["point"][met]
                tmp.append({"role": role, "metric": met, "a": a, "b": b, "ma": ma, "mb": mb, "diff": mu, "lo": lo,
                            "hi": hi, "t": t, "p": p, "p_rope": pr, "p_better": pb, "h_diff": hd,
                            "h_lo": np.quantile(bd, cfg.alpha / 2), "h_hi": np.quantile(bd, 1 - cfg.alpha / 2)})
            tmp = pd.DataFrame(tmp)
            tmp["p_holm"] = holm(tmp.p.fillna(1.0))
            crow.append(tmp)
    cdf = pd.concat(crow, ignore_index=True)

    def _verdict(r):
        if r.metric in margin:
            return verdict(r["diff"], r.lo, r.hi, r.p_holm, margin[r.metric], cfg.alpha)
        if not np.isfinite(r.lo):
            return "not estimable"
        if r.p_holm < cfg.alpha:
            better = r["diff"] < 0 if r.metric == "cw_ece" else r["diff"] > 0
            return "better" if better else "worse"
        return "inconclusive (no practical margin prespecified)"

    cdf["verdict"] = [_verdict(r) for _, r in cdf.iterrows()]
    n_contr = int(cdf.groupby(["role", "metric"]).size().iloc[0])
    cdf.to_csv(paths.metadata / "scenario_contrasts.csv", index=False)
    t12 = pd.DataFrame([{
        "Model": "Selected model of each scenario" if r.role == "Selected" else "LR reference (fixed model)",
        "Contrast": f"{SCENARIO_SHORT[r.a]} − {SCENARIO_SHORT[r.b]}"
                    + (f" ({dn(r.ma)} − {dn(r.mb)})" if r.role == "Selected" else ""),
        "Metric": LABELS[r.metric],
        "CV difference (95% CI)": fmt_ci(r["diff"], r.lo, r.hi, 3, signed=True), "Holm p": fmt_p(r.p_holm),
        "P(ROPE)": num(r.p_rope, 2), "CV verdict": r.verdict,
        "Holdout difference (95% CI)": fmt_ci(r.h_diff, r.h_lo, r.h_hi, 3, signed=True)} for _, r in cdf.iterrows()])
    rep.register_table(
        t12, "Table12_scenario_contrasts", "12", "Paired comparisons between prediction scenarios",
        f"CV: differences across the same {n_blocks} outer folds; Nadeau–Bengio corrected t test (n_test/n_train = "
        f"{test_ratio:.2f}) with Holm adjustment within each model role and metric over the {n_contr} contrasts; "
        " P(ROPE), posterior probability of the correlated Bayesian t test that the difference lies within the "
        f"practical margin (± {cfg.practical_margin_f1m:.2f} for macro-F1 and MCC; no margin was prespecified for recall "
        "and ECE). Verdict rule as in Table 3 (95% CIs not adjusted for multiplicity). Holdout: paired stratified "
        f"bootstrap over the same students ({cfg.holdout_bootstrap:,} resamples). For class-wise ECE, negative "
        "differences indicate better calibration. Differences describe how much predictive information is added by "
        "later data; they do not show that later prediction is preferable, because intervention options shrink over "
        "time", landscape=True, group_col="Model")

    # --- Tabla S12: importancia SHAP por escenario ---
    imps = {s: R[s]["imp"] for s in scen if R[s]["imp"] is not None and "shap" in R[s]["imp"]}
    if imps:
        top = []
        for s, im in imps.items():
            top += list(im.sort_values("shap", ascending=False).index[:10])
        top = list(dict.fromkeys(top))
        rows = []
        for f in top:
            row = {"Predictor": f}
            for s, im in imps.items():
                if f in im.index:
                    rk = int(im.shap.rank(ascending=False)[f])
                    row[SCENARIO_SHORT[s]] = f"{im.shap[f] * 100:.2f} ({rk})"
                else:
                    row[SCENARIO_SHORT[s]] = "not available"
            rows.append(row)
        rep.register_table(
            pd.DataFrame(rows), "TableS12_importance_by_scenario", "S12",
            "Global importance of the predictors of the selected model in each prediction scenario",
            "Mean absolute SHAP value × 100 over the three classes (rank in parentheses), calibrated probabilities, "
            "holdout students. Union of the 10 most influential predictors of each scenario. 'Not available': the "
            "predictor is not known at the time of prediction of that scenario. Attributions describe each model, not "
            "causal effects; for debtor, tuition-fee and scholarship status the direction of influence may be reversed "
            "(status recorded after academic difficulties)", supplementary=True)

    # --- Tabla S13: estado de la explicabilidad por escenario ---
    rep.register_table(
        pd.DataFrame([{"Scenario": SCENARIO_LABEL[s], "Selected model": dn(R[s]["best"]),
                       "SHAP": R[s]["status"].get("SHAP", "—"), "LIME (prespecified)": R[s]["status"].get("LIME", "—"),
                       "LIME sensitivity": R[s]["status"].get("LIME sensitivity", "—")} for s in scen]),
        "TableS13_explainability_by_scenario", "S13", "Status and fidelity of the explanations in each scenario",
        "LIME is interpretable only if its local R² ≥ the prespecified threshold; the sensitivity configuration was chosen "
        "on development students and applied once to the holdout (Table S11 of each scenario)", supplementary=True)

    figure8(rep, cfg, scen, R, fold_vec, BOOT, y_hold)
    rep.export_all(extra_sheets={"scenario_contrasts": cdf})
    draft = results_text(cfg, scen, R, cdf, BOOT)
    if any(R[s]["quick"] for s in scen):
        draft = "> **WARNING — QUICK MODE: STRUCTURAL TEST ONLY. DO NOT REPORT THESE RESULTS.**\n\n" + draft
    (paths.root / "results_scenarios_draft.md").write_text(draft, encoding="utf-8")
    log.info("Scenario comparison written to %s", paths.root)
    return paths.root


# =============================================================================================
# Figura 8
# =============================================================================================
def figure8(rep, cfg, scen, R, fold_vec, BOOT, y_hold) -> None:
    fig = new_canvas("double", 140, wspace=0.04, hspace=0.03)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.42, 1, 0.07])
    ax0 = fig.add_subplot(gs[0, :])
    # a — línea temporal
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    xs = {"enrol": 0.08, "s1": 0.36, "s2": 0.62, "end": 0.92}
    ax0.annotate("", xy=(0.98, 0.70), xytext=(0.02, 0.70),
                 arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black", mutation_scale=8))
    for k_, lab in [("enrol", "Enrolment"), ("s1", "End of 1st semester"), ("s2", "End of 1st year"),
                    ("end", "End of normal duration\n(outcome recorded)")]:
        ax0.plot([xs[k_]] * 2, [0.64, 0.76], color="black", lw=0.8)
        ax0.text(xs[k_], 0.80, lab, ha="center", va="bottom", fontsize=7)
    pos = {"T0_strict": (xs["enrol"] - 0.06, 0.30), "T0": (xs["enrol"] + 0.06, 0.30), "T1": (xs["s1"], 0.30),
           "T2": (xs["s2"], 0.30)}
    for s in scen:
        x, y = pos[s]
        ax0.text(x, y, f"{SCENARIO_SHORT[s]}\n{R[s]['n_pred']} predictors", ha="center", va="center", fontsize=7,
                 bbox=dict(boxstyle="round,pad=0.3", fc=tint(SC_COLOR[s], 0.25), ec=SC_COLOR[s], lw=0.7))
    ax0.text(xs["end"], 0.30, "Dropout · Enrolled\n· Graduate", ha="center", va="center", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.6))
    panel_label(ax0, "a")
    # b–d — métricas
    mets = [("f1_macro", "Macro-F1"), ("recall_Enrolled", "Recall (Enrolled)"), ("cw_ece", "Class-wise ECE (lower is better)")]
    xs_ = np.arange(len(scen))
    for j, (met, ylab) in enumerate(mets):
        ax = fig.add_subplot(gs[1, j])
        for role, off, mk, col_fn in [("Selected", -0.13, "o", lambda s: SC_COLOR[s]), ("Reference", 0.13, "s",
                                                                                         lambda s: C_REF)]:
            mus, sds, hp, hl, hh = [], [], [], [], []
            for s in scen:
                m = R[s]["best"] if role == "Selected" else REF
                v = fold_vec(s, m, met)
                ax.scatter(np.full(len(v), xs_[scen.index(s)] + off - 0.06), v, s=5, color=tint(col_fn(s), 0.45), lw=0,
                           zorder=2)
                mus.append(v.mean())
                sds.append(v.std(ddof=1))
                b = BOOT[(s, role)]
                hp.append(b.attrs["point"][met])
                hl.append(b[met].quantile(cfg.alpha / 2))
                hh.append(b[met].quantile(1 - cfg.alpha / 2))
            for i, s in enumerate(scen):
                c = col_fn(s)
                ax.errorbar(xs_[i] + off - 0.06, mus[i], yerr=sds[i], fmt=mk, ms=3.6, color=c, mec="black", mew=0.3,
                            elinewidth=0.7, capsize=1.2, zorder=3)
                ax.errorbar(xs_[i] + off + 0.06, hp[i], yerr=[[max(0.0, hp[i] - hl[i])], [max(0.0, hh[i] - hp[i])]], fmt=mk, ms=3.6,
                            mfc="white", color=c, mew=0.8, elinewidth=0.7, capsize=1.2, zorder=3)
        ax.set_xticks(xs_, [SCENARIO_SHORT[s] for s in scen])
        ax.set_xlim(-0.5, len(scen) - 0.5)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", lw=0.4, color=GRID)
        ax.set_axisbelow(True)
        panel_label(ax, "bcd"[j])
    h = [mpl.lines.Line2D([], [], marker="o", ls="", color="black", ms=3.6, label="Selected model, CV (mean ± SD)"),
         mpl.lines.Line2D([], [], marker="o", ls="", mfc="white", color="black", ms=3.6,
                          label="Selected model, holdout (95% CI)"),
         mpl.lines.Line2D([], [], marker="s", ls="", color=C_REF, ms=3.6, label="LR reference (CV / holdout)")]
    lax = fig.add_subplot(gs[2, :])
    lax.axis("off")
    lax.legend(handles=h, loc="center", ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.5)
    rep.save_figure(
        fig, "Fig8_prediction_scenarios",
        "Fig. 8 Prediction scenarios defined by the information available at the time of prediction. a Time line: T0 "
        "uses enrolment information (T0-strict excludes debtor, tuition-fee and scholarship status, whose time of "
        "measurement is undocumented), T1 adds first-semester results and T2 first-year results; the outcome is "
        "recorded at the end of the normal programme duration. b–d Macro-F1, recall for Enrolled and class-wise ECE of "
        "the model selected in each scenario (coloured) and of the calibrated multinomial logistic regression (grey): "
        "light dots, outer cross-validation folds; filled markers, CV mean ± SD; open markers, locked holdout with 95% "
        "stratified bootstrap intervals. All models were re-tuned and re-calibrated in every scenario with identical "
        "partitions. Results are retrospective; later scenarios gain information but leave less time to intervene. "
        "CV, cross-validation; ECE, expected calibration error; LR, logistic regression")


# =============================================================================================
# Borrador de resultados con la interpretación obligatoria
# =============================================================================================
def results_text(cfg, scen, R, cdf, BOOT) -> str:
    def hv(s, role, met, d=3):
        b = BOOT[(s, role)]
        return fmt_ci(b.attrs["point"][met], b[met].quantile(cfg.alpha / 2), b[met].quantile(1 - cfg.alpha / 2), d)

    lines = ["# Results — prediction scenarios (draft generated by the pipeline; verify before submission)", "",
             "## Scenarios", ""]
    for s in scen:
        lines.append(f"* **{SCENARIO_LABEL[s]}** — {R[s]['n_pred']} predictors. {SCENARIO_MOMENT[s]}. Selected "
                     f"model: {dn(R[s]['best'])}. Holdout macro-F1 {hv(s, 'Selected', 'f1_macro')}; recall for "
                     f"Enrolled {hv(s, 'Selected', 'recall_Enrolled')}; class-wise ECE {hv(s, 'Selected', 'cw_ece')} "
                     f"(LR: macro-F1 {hv(s, 'Reference', 'f1_macro')}).")
    lines += ["", "## Between-scenario contrasts (selected model of each scenario; Table 12)", ""]
    for _, r in cdf[cdf.role == "Selected"].iterrows():
        lines.append(f"* {SCENARIO_SHORT[r.a]} − {SCENARIO_SHORT[r.b]}, {LABELS[r.metric]}: CV "
                     f"{fmt_ci(r['diff'], r.lo, r.hi, 3, True)} (Holm p = {fmt_p(r.p_holm)}; {r.verdict}); holdout "
                     f"{fmt_ci(r.h_diff, r.h_lo, r.h_hi, 3, True)}.")
    lines += ["", "## Mandatory interpretation", "",
              "1. **Enrolled is not longitudinal persistence.** It is the status at the end of the normal programme "
              "duration (still enrolled, no degree) and pools delayed graduation, later dropout and interruptions, "
              "which these data cannot separate. Do not describe it as retention, persistence or partial success.",
              "2. **Retrospective prediction is not early detection.** All scenarios were evaluated on historical "
              "cohorts with known outcomes; the holdout is a random partition of the same cohorts. Only T0/T0-strict "
              "correspond to identification at enrolment; T1 and T2 identify students after the first semester or "
              "year. Any 'early warning' claim must name the scenario and requires prospective, cohort-based "
              "validation before deployment.",
              "3. **T2 performance is not the performance at enrolment.** Report the scenario next to every metric.",
              "4. **T0 vs T0-strict.** The difference measures how much the model relies on debtor, tuition-fee and "
              "scholarship status, whose time of measurement is undocumented; if T0 clearly outperforms T0-strict, the "
              "T0 estimate may be optimistic because of temporal leakage or reverse causality.",
              "5. Predictor importance describes the models, not causes."]
    return "\n".join(lines) + "\n"
