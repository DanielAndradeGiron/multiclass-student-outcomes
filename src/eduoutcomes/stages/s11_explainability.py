"""§14 — PI3: explicabilidad *post hoc* con control de fidelidad, concordancia y estabilidad.

Todas las explicaciones se calculan sobre las probabilidades CALIBRADAS del modelo congelado, a nivel de variable
original y con la misma función de predicción (verificada). SHAP se calcula una sola vez para todos los estudiantes
explicados (muestra global, casos LIME y caso de la Fig. 6). La concordancia SHAP–LIME se calcula sobre los mismos
casos, clases objetivo y predictores seleccionados localmente por LIME.
"""

from __future__ import annotations

import importlib
import json
import logging
import math

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats

from ..constants import BINARY, CLASSES, K, NOMINAL, dn, short_name
from ..formatting import fmt_ci, num
from ..metrics import nll
from ..style import C_NEG, C_POS, C_REF, CMAP_DIV, GRID, new_figure, panel_label, tint

log = logging.getLogger(__name__)


def _optional(mod):
    try:
        return importlib.import_module(mod)
    except Exception:  # noqa: BLE001
        return None


def kt(a, b) -> float:
    return float(stats.kendalltau(a, b)[0])


def jac(a, b) -> float:
    return len(set(a) & set(b)) / len(set(a) | set(b))


def topk(v, k: int = 10) -> list[int]:
    return list(np.argsort(-np.asarray(v))[:k])


class NumericView:
    """Convierte entre la vista numérica (la que ven SHAP/LIME) y el DataFrame original del *pipeline*."""

    def __init__(self, cols: list[str]):
        self.cols = cols

    def to_num(self, X: pd.DataFrame) -> pd.DataFrame:
        D = X[self.cols].copy()
        for c in NOMINAL:
            if c in D:
                D[c] = D[c].astype(float)
        return D.astype(float)

    def from_num(self, A) -> pd.DataFrame:
        D = pd.DataFrame(np.asarray(A, float), columns=self.cols)
        for c in NOMINAL:
            if c in D:
                D[c] = D[c].round().astype(int).astype(str).astype(object)
        return D


def perm_importance(model, X, y, cols, reps, seed):
    """Importancia por permutación (Δ log-loss) con predicción por lotes (una llamada por variable)."""
    rng = np.random.default_rng(seed)
    Xv = X[cols].reset_index(drop=True)
    n = len(Xv)
    base = nll(y, model.predict_proba(Xv))
    out = np.zeros((len(cols), reps))
    for j, c in enumerate(cols):
        big = pd.concat([Xv] * reps, ignore_index=True)
        big[c] = np.concatenate([Xv[c].to_numpy()[rng.permutation(n)] for _ in range(reps)])
        Pb = model.predict_proba(big)
        out[j] = [nll(y, Pb[r * n:(r + 1) * n]) - base for r in range(reps)]
    return out


def run(ctx) -> None:  # noqa: C901 — etapa larga pero lineal; se mantiene junta para preservar el orden de las semillas
    cfg, ds, rep, best = ctx.cfg, ctx.ds, ctx.rep, ctx.BEST
    COLS = ds.cols
    FM = ctx.FINAL[best]
    X_hold, y_hold, X_dev, P_best = ds.X_hold, ds.y_hold, ds.X_dev, ctx.P_best
    SEED = cfg.random_state
    nv = NumericView(COLS)
    shap = _optional("shap")
    lime_mod = _optional("lime.lime_tabular")
    expl: dict[str, str] = {}

    def f_proba(A):
        return FM.predict_proba(nv.from_num(A))

    assert np.allclose(f_proba(nv.to_num(X_hold.iloc[:50]).values), FM.predict_proba(X_hold.iloc[:50]), atol=1e-10), \
        "Prediction function differs from the frozen pipeline"

    pi = perm_importance(FM, X_hold, y_hold, COLS, cfg.perm_repeats, SEED)
    imp = pd.DataFrame({"perm": pi.mean(1), "perm_lo": np.percentile(pi, 2.5, axis=1),
                        "perm_hi": np.percentile(pi, 97.5, axis=1)}, index=COLS)
    summary = ctx.summary
    stab_models = [best] + [m for m in summary[summary.model.isin(ctx.factory.candidates)]
                            .sort_values("f1_macro_mean", ascending=False).model if m != best][:cfg.stability_models - 1]
    cross = []
    for m in stab_models[1:]:
        v = perm_importance(ctx.FINAL[m], X_hold, y_hold, COLS, max(5, cfg.perm_repeats // 2), SEED).mean(1)
        cross.append({"Comparison": f"{dn(best)} vs {dn(m)}", "Kendall τ (all predictors)": num(kt(imp.perm, v), 2),
                      "Top-10 overlap (Jaccard)": num(jac(topk(imp.perm), topk(v)), 2)})

    # --- Estudiantes a explicar: muestra global, casos LIME y caso de la Fig. 6 ---
    rng_e = np.random.default_rng(SEED + 1)
    EXi = np.sort(rng_e.choice(len(X_hold), min(cfg.shap_explain, len(X_hold)), replace=False))
    rng_l = np.random.default_rng(SEED + 31)
    per = int(math.ceil(cfg.lime_cases / K))
    cases = np.concatenate([rng_l.choice(np.where(y_hold == i)[0], min(per, int((y_hold == i).sum())), replace=False)
                            for i in range(K)])
    ok_e = np.where((y_hold == 1) & (P_best.argmax(1) == 1))[0]
    CASE = (int(ok_e[np.argmin(np.abs(P_best[ok_e, 1] - np.median(P_best[ok_e, 1])))]) if len(ok_e)
            else int(np.argmax(np.where(y_hold == 1, P_best[:, 1], -1))))
    cases = np.unique(np.r_[cases, CASE])
    ALL_EX = np.unique(np.r_[EXi, cases, CASE])
    pos_of = {r: i for i, r in enumerate(ALL_EX)}

    # --- SHAP por clase ---
    SV = SV_ALL = BASE_ALL = None
    taus: list[float] = []
    if shap is not None:
        try:
            bg = nv.to_num(X_dev.sample(min(cfg.shap_background, len(X_dev)), random_state=SEED))
            explainer = shap.PermutationExplainer(f_proba, shap.maskers.Independent(bg.values), seed=SEED)
            maxe = 2 * len(COLS) + 1 if cfg.quick else 4 * len(COLS) + 1
            sv_all = explainer(nv.to_num(X_hold.iloc[ALL_EX]).values, max_evals=maxe, silent=True)
            SV_ALL = np.asarray(sv_all.values)
            SV_ALL = SV_ALL[:, :, None] if SV_ALL.ndim == 2 else SV_ALL
            BASE_ALL = np.asarray(sv_all.base_values).reshape(len(ALL_EX), -1)
            SV = SV_ALL[[pos_of[r] for r in EXi]]
            np.save(ctx.paths.metadata / "shap_values_holdout.npy", SV)
            shap_cls = np.abs(SV).mean(0)
            imp["shap"] = shap_cls.mean(1)
            for i, c in enumerate(CLASSES):
                imp[f"shap_{c}"] = shap_cls[:, i]
            rng = np.random.default_rng(SEED + 40)
            keep10 = np.zeros(len(COLS))
            for _ in range(cfg.shap_bootstrap):
                ib = rng.integers(0, len(SV), len(SV))
                ob = np.abs(SV[ib]).mean(0).mean(1)
                taus.append(kt(imp.shap, ob))
                keep10[topk(ob)] += 1
            imp["shap_top10_freq"] = keep10 / cfg.shap_bootstrap
            expl["SHAP"] = (f"permutation explainer; {len(SV)} students; background {len(bg)}; {maxe} evaluations per "
                            f"student; bootstrap τ median {np.median(taus):.2f}")
        except Exception as e:  # noqa: BLE001
            expl["SHAP"] = f"failed: {e!r}"
            log.warning("SHAP failed: %r", e)
            SV = SV_ALL = None
    else:
        expl["SHAP"] = "shap not installed"

    # --- LIME con control de fidelidad y acuerdo local con SHAP ---
    lime_rows, lime_agg, lime_freq = [], np.zeros(len(COLS)), np.zeros(len(COLS))
    matched_shap_agg = np.zeros(len(COLS))
    lime_explanations = {}
    lime_df, lime = pd.DataFrame(), None
    if lime_mod is not None:
        try:
            lime = lime_mod.LimeTabularExplainer(
                nv.to_num(X_dev).values, feature_names=COLS, class_names=list(CLASSES),
                categorical_features=[COLS.index(c) for c in NOMINAL + BINARY if c in COLS], mode="classification",
                random_state=SEED, discretize_continuous=True)
            for i in cases:
                x = nv.to_num(X_hold.iloc[[i]]).values[0]
                lab = int(P_best[i].argmax())
                e_ = lime.explain_instance(x, f_proba, labels=(lab,), num_features=10, num_samples=cfg.lime_num_samples)
                sc = e_.score[lab] if isinstance(e_.score, dict) else e_.score
                wmap = dict(e_.as_map()[lab])
                fidx = list(wmap)
                lw = np.array([wmap[f] for f in fidx])
                lime_agg[fidx] += np.abs(lw)
                lime_freq[fidx] += 1
                lime_explanations[int(i)] = e_
                row = {"case": int(i), "observed": CLASSES[y_hold[i]], "predicted": CLASSES[lab], "r2": float(sc)}
                if SV_ALL is not None:
                    s_ = SV_ALL[pos_of[i], :, lab]
                    sw = s_[fidx]
                    matched_shap_agg[fidx] += np.abs(sw)
                    row.update({
                        "spearman_abs": float(stats.spearmanr(np.abs(sw), np.abs(lw))[0]) if len(fidx) >= 3 else np.nan,
                        "sign_agreement": float(np.mean(np.sign(sw) == np.sign(lw))),
                        "top5_jaccard": jac(topk(np.abs(s_), 5), [fidx[j] for j in np.argsort(-np.abs(lw))[:5]])})
                lime_rows.append(row)
            lime_df = pd.DataFrame(lime_rows)
            lime_df.to_csv(ctx.paths.supplementary / "lime_cases.csv", index=False)
            imp["lime_freq"] = lime_freq / len(cases)
            imp["lime_abs"] = lime_agg / len(cases)
            if SV_ALL is not None:
                imp["lime_matched_shap_abs"] = matched_shap_agg / len(cases)
            fid = lime_df.r2.to_numpy()
            expl["LIME"] = f"median R² {np.median(fid):.3f}; " + (
                "reportable" if np.median(fid) >= cfg.lime_min_fidelity else "not reportable (low fidelity)")
        except Exception as e:  # noqa: BLE001
            expl["LIME"] = f"failed: {e!r}"
            log.warning("LIME failed: %r", e)
            lime_rows = []
            lime_explanations.clear()
    else:
        expl["LIME"] = "lime not installed"

    # --- Tabla 8: predictores más influyentes y concordancia entre métodos ---
    imp = imp.sort_values("shap" if "shap" in imp else "perm", ascending=False)
    ctx.imp = imp
    imp.rename_axis("predictor").to_csv(ctx.paths.metadata / "importance_all.csv")
    top = imp.head(15)
    t8: dict = {"Predictor": top.index}
    if "shap" in top:
        t8["Mean |SHAP| × 100 (all classes)"] = (top.shap * 100).map(lambda v: num(v, 2))
        for c in CLASSES:
            t8[f"Mean |SHAP| × 100 ({c})"] = (top[f"shap_{c}"] * 100).map(lambda v: num(v, 2))
        t8["Top-10 in SHAP bootstrap (%)"] = (top.shap_top10_freq * 100).map(lambda v: f"{v:.0f}")
    t8["Permutation Δ log-loss (95% range)"] = [fmt_ci(a, b, c, 4) for a, b, c in zip(top.perm, top.perm_lo, top.perm_hi)]
    if "lime_freq" in top:
        t8["In LIME top-10 (% of cases)"] = (top.lime_freq * 100).map(lambda v: f"{v:.0f}")
    conc = []
    if "shap" in imp:
        conc.append(("SHAP vs permutation", kt(imp.shap, imp.perm), jac(topk(imp.shap), topk(imp.perm))))
        if "lime_matched_shap_abs" in imp:
            conc.append(("SHAP vs LIME (matched cases/classes)", kt(imp.lime_matched_shap_abs, imp.lime_abs),
                         jac(topk(imp.lime_matched_shap_abs), topk(imp.lime_abs))))
    conc_txt = "; ".join(f"{a}: τ = {num(b, 2)}, top-10 Jaccard = {num(c, 2)}" for a, b, c in conc) or "not computed"
    ctx.conc_txt, ctx.EXPL = conc_txt, expl
    rep.register_table(
        pd.DataFrame(t8), "Table8_explainability", "8",
        f"Most influential predictors of the calibrated probabilities of the frozen {dn(best)} (holdout)",
        "Mean |SHAP|: mean absolute SHAP value of the calibrated probability of each class (percentage points), "
        "permutation explainer with a development-set background; 'all classes' is the mean over classes. Top-10 in SHAP "
        f"bootstrap: share of {cfg.shap_bootstrap} bootstrap resamples of the explained students in which the predictor "
        "ranked in the top 10. Permutation Δ log-loss: increase in multiclass log-loss after permuting the predictor "
        f"({cfg.perm_repeats} repeats; 2.5th–97.5th percentiles). LIME top-10: share of explained students whose local "
        f"explanation included the predictor. SHAP–LIME concordance uses the same LIME cases and predicted classes, "
        "aggregating SHAP absolute values only for features selected in each corresponding LIME explanation. "
        f"Concordance — {conc_txt}. Attributions describe the fitted model, "
        "not causal effects; correlated curricular-unit variables share attribution", landscape=True)
    s7 = []
    if lime_rows:
        for lab_, g in [("All explained students", lime_df)] + [(f"Predicted {c}", lime_df[lime_df.predicted == c])
                                                                 for c in CLASSES]:
            if len(g):
                s7.append({"Measure": f"LIME local R², median (IQR) — {lab_} (n = {len(g)})",
                           "Value": f"{num(g.r2.median())} ({num(g.r2.quantile(.25))}–{num(g.r2.quantile(.75))})"})
        s7.append({"Measure": f"LIME R² ≥ {cfg.lime_min_fidelity:.2f}, n (%)",
                   "Value": f"{int((lime_df.r2 >= cfg.lime_min_fidelity).sum())} "
                            f"({(lime_df.r2 >= cfg.lime_min_fidelity).mean():.0%})"})
        case_lime = lime_df.loc[lime_df.case == CASE, "r2"]
        if not case_lime.empty:
            case_r2 = float(case_lime.iloc[0])
            s7.append({"Measure": "LIME local R² — illustrative Enrolled case",
                       "Value": f"{case_r2:.3f} ({'meets' if case_r2 >= cfg.lime_min_fidelity else 'below'} "
                                f"the {cfg.lime_min_fidelity:.2f} fidelity threshold)"})
        if "spearman_abs" in lime_df:
            s7 += [{"Measure": "Local SHAP–LIME Spearman ρ of |attributions|, median (IQR)",
                    "Value": fmt_ci(lime_df.spearman_abs.median(), lime_df.spearman_abs.quantile(.25),
                                    lime_df.spearman_abs.quantile(.75), 2)},
                   {"Measure": "Local SHAP–LIME sign agreement, median", "Value": num(lime_df.sign_agreement.median(), 2)},
                   {"Measure": "Local SHAP–LIME top-5 overlap (Jaccard), median",
                    "Value": num(lime_df.top5_jaccard.median(), 2)}]
    if SV is not None:
        s7.append({"Measure": "SHAP bootstrap Kendall τ with full-sample ranking, median (2.5th–97.5th percentile)",
                   "Value": fmt_ci(np.median(taus), np.quantile(taus, .025), np.quantile(taus, .975), 2)})
    s7 += [{"Measure": a, "Value": f"τ = {num(b, 2)}; top-10 Jaccard = {num(c, 2)}"} for a, b, c in conc]
    if s7:
        rep.register_table(
            pd.DataFrame(s7), "TableS7_explanation_quality", "S7", "Fidelity, agreement and stability of the explanations",
            f"LIME ({cfg.lime_num_samples:,} perturbations per student; nominal and binary predictors declared "
            "categorical) explained the predicted class of students sampled evenly from the three observed classes, "
            "plus the prespecified illustrative Enrolled case. Explanations are interpretable only if "
            f"R² ≥ {cfg.lime_min_fidelity:.2f}. Local agreement compares SHAP and LIME for the same student, predicted "
            "class and frozen model over the predictors selected by LIME; aggregated SHAP–LIME concordance uses this "
            "same matched set. Low-fidelity LIME is supplementary and should not be interpreted as a faithful explanation.",
            supplementary=True)
    if cross:
        rep.register_table(
            pd.DataFrame(cross), "TableS8_cross_model_stability", "S8",
            "Stability of global importance across the best-performing frozen candidates (permutation importance, holdout)",
            f"Kendall τ over all {len(COLS)} predictors and overlap of the top-10 predictors between the selected candidate and the "
            "next-best candidates in nested cross-validation", supplementary=True)

    # --- Figura 5: SHAP por clase ---
    if SV is not None:
        feats = list(imp.index[:12])[::-1]
        XE = nv.to_num(X_hold.iloc[EXi])
        fig, axs = new_figure("double", 108, ncols=K, sharey=True)
        for k_, (ax, c) in enumerate(zip(axs, CLASSES)):
            for i, f in enumerate(feats):
                j = COLS.index(f)
                sv_ = SV[:, j, k_] * 100
                xv = XE[f].to_numpy()
                h_, e_ = np.histogram(sv_, bins=30)
                dens = h_[np.clip(np.digitize(sv_, e_[1:-1]), 0, 29)] / max(h_.max(), 1)
                jit = (np.random.default_rng(i + 10 * k_).random(len(sv_)) - 0.5) * 0.75 * np.sqrt(dens)
                col = ([tint(C_REF, 0.7)] * len(sv_) if f in NOMINAL
                       else CMAP_DIV(pd.Series(xv).rank(pct=True).to_numpy()))
                ax.scatter(sv_, i + jit, s=3.5, c=col, lw=0, zorder=2)
            ax.axvline(0, color="black", lw=0.6, zorder=1)
            ax.set_xlabel(f"SHAP value for P({c}) (pp)")
            ax.grid(axis="x", lw=0.4, color=GRID)
            ax.set_axisbelow(True)
            panel_label(ax, "abc"[k_])
        axs[0].set_yticks(range(len(feats)), [short_name(f) for f in feats])
        cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=CMAP_DIV), ax=axs, fraction=0.015, pad=0.01, aspect=45, shrink=0.8)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Predictor value (percentile)")
        cb.outline.set_linewidth(0.4)
        rep.save_figure(
            fig, "Fig5_shap_by_class",
            "Fig. 5 SHAP values of the 12 most influential predictors for the calibrated probabilities of a Dropout, "
            f"b Enrolled and c Graduate ({len(SV)} holdout students; {dn(best)}). Each point is one student; colour shows "
            "the percentile of the predictor value (grey for nominal predictors). Positive values increase the probability "
            "of the class. CU1/CU2, curricular units in the first/second semester; pp, percentage points. Attributions "
            "describe the model, not causal effects")

    # --- Figura 6: SHAP y LIME del MISMO estudiante, clase Enrolled, mismo modelo congelado ---
    x_case = nv.to_num(X_hold.iloc[[CASE]]).values
    consistency = {"case_row_id": int(ds.hold_ids[CASE]), "observed": CLASSES[y_hold[CASE]],
                   "class_explained": "Enrolled", "p_pipeline": float(P_best[CASE, 1]),
                   "p_prediction_function": float(f_proba(x_case)[0, 1])}
    assert abs(consistency["p_pipeline"] - consistency["p_prediction_function"]) < 1e-9
    loc_shap = loc_lime = None
    if SV_ALL is not None:
        v1, b1 = SV_ALL[pos_of[CASE]], BASE_ALL[pos_of[CASE]]
        loc_shap = pd.Series(v1[:, 1], index=COLS)
        consistency.update({"shap_base": float(b1[1]), "shap_base_plus_sum": float(b1[1] + loc_shap.sum())})
        assert abs(b1[1] + loc_shap.sum() - consistency["p_pipeline"]) < 1e-6, "SHAP additivity check failed"
    if CASE in lime_explanations:
        e_ = lime_explanations[CASE]
        loc_lime = pd.DataFrame(e_.as_list(label=1), columns=["condition", "weight"])
        sc = e_.score[1] if isinstance(e_.score, dict) else e_.score
        consistency["lime_local_R2"] = float(sc)
    (ctx.paths.metadata / "shap_lime_consistency.json").write_text(json.dumps(consistency, indent=2), encoding="utf-8")
    if loc_shap is not None:
        figure6(ctx, loc_shap, consistency, X_hold.iloc[CASE])
    if loc_lime is not None:
        figureS4_lime(ctx, loc_lime, consistency)
    # --- v3.0: sensibilidad de LIME (post hoc, elegida en desarrollo, aplicada una vez a la reserva) ---
    if cfg.lime_sensitivity and lime_mod is not None and len(lime_df):
        try:
            from ..lime_sensitivity import run as lime_sens_run
            res = lime_sens_run(ctx, lime_mod, nv, f_proba, cases, CASE, SV_ALL, pos_of, lime_df)
            expl["LIME sensitivity"] = (f"{res['chosen']} — {res['rule']}; holdout median R² "
                                        f"{res['holdout_r2_median']:.3f}; Fig. 6 case R² {res['fig6_case_r2']:.3f}")
        except Exception as e:  # noqa: BLE001
            expl["LIME sensitivity"] = f"failed: {e!r}"
            log.warning("LIME sensitivity failed: %r", e)
    (ctx.paths.metadata / "explainability_status.json").write_text(json.dumps(expl, indent=2), encoding="utf-8")
    log.info("Explainability: %s", expl)


def _short_condition(s, cols):
    for c in sorted(cols, key=len, reverse=True):
        if c in s:
            return s.replace(c, short_name(c))
    return s


def _val(c, v):
    return f"{v}" if c in NOMINAL else (f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}")


def figure6(ctx, loc_shap, consistency, case_row) -> None:
    best = ctx.BEST
    fig, ax = new_figure("single", 84)
    tl = loc_shap.reindex(loc_shap.abs().sort_values().index[-10:])
    ax.barh(range(len(tl)), tl.values * 100, color=[C_POS if v > 0 else C_NEG for v in tl.values], height=0.62,
            edgecolor="black", lw=0.3)
    ax.set_yticks(range(len(tl)), [f"{short_name(c)} = {_val(c, case_row[c])}" for c in tl.index])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("SHAP value for P(Enrolled) (pp)")
    ax.set_title(f"P(Enrolled) = {consistency['p_pipeline']:.3f}; base value = {consistency['shap_base']:.3f}",
                 loc="right", fontsize=7)
    ax.grid(axis="x", lw=0.4, color=GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "a")
    ctx.rep.save_figure(
        fig, "Fig6_local_explanation_enrolled",
        "Fig. 6 Local explanations of the same holdout student in the Enrolled class (prespecified: correctly classified "
        f"student closest to the median P(Enrolled)) from the same frozen {dn(best)} and prediction function. a SHAP "
        "contributions (10 largest; base value plus contributions reproduces the calibrated probability). Vermillion bars "
        "increase and blue bars decrease the probability of remaining enrolled. CU1/CU2, curricular units in the "
        "first/second semester; pp, percentage points")


def figureS4_lime(ctx, loc_lime, consistency) -> None:
    cfg = ctx.cfg
    fig, ax = new_figure("single", 84)
    q_ = loc_lime.iloc[::-1]
    ax.barh(range(len(q_)), q_.weight * 100, color=[C_POS if v > 0 else C_NEG for v in q_.weight], height=0.62,
            edgecolor="black", lw=0.3)
    ax.set_yticks(range(len(q_)), [_short_condition(c, ctx.ds.cols) for c in q_.condition])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("LIME weight for P(Enrolled) (pp)")
    ax.set_title(f"Local R² = {consistency['lime_local_R2']:.2f}", loc="right", fontsize=7)
    ax.grid(axis="x", lw=0.4, color=GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "a")
    meets_threshold = consistency["lime_local_R2"] >= cfg.lime_min_fidelity
    fidelity_note = (f"The surrogate {'meets' if meets_threshold else 'does not meet'} the prespecified fidelity "
                     f"threshold (R² ≥ {cfg.lime_min_fidelity:.2f}) and should {'be interpreted cautiously' if meets_threshold else 'not be interpreted as a faithful local explanation'}.")
    ctx.rep.save_figure(
        fig, "FigS4_lime_local_enrolled",
        "Fig. S4 Supplementary LIME surrogate weights for the same prespecified holdout student and calibrated "
        f"P(Enrolled) shown in Fig. 6 ({dn(ctx.BEST)}). {fidelity_note} Vermillion bars increase and blue bars decrease "
        "the locally fitted probability; pp, percentage points.", supplementary=True)
