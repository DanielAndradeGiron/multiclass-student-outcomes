"""Análisis de sensibilidad de LIME (post hoc, declarado como tal; v3.0).

El R² local de LIME con su configuración por defecto (ancho de núcleo 0.75·√p, discretización en cuartiles) suele ser
bajo en modelos de *boosting* muy no lineales. Este análisis comprueba si alguna configuración razonable alcanza una
fidelidad suficiente SIN usar la reserva para elegirla:

1. Se evalúan varias configuraciones solo con estudiantes del conjunto de DESARROLLO.
2. La configuración se elige con una regla fijada de antemano: mayor mediana del R² local entre las que son ESTABLES
   entre semillas (Jaccard medio del top-5 ≥ ``lime_sensitivity_min_stability``), con 10 predictores como en el
   análisis principal. Si ninguna es estable, se conserva la original (un R² más alto e inestable no es creíble:
   un núcleo muy estrecho "ajusta" mejor porque ajusta muy pocos puntos y cambia de explicación con cada semilla).
3. La configuración elegida se aplica UNA vez a los mismos casos de la reserva y al caso de la Fig. 6.

La Fig. 6, la Tabla 8 y la Tabla S7 conservan la configuración preespecificada; este análisis es suplementario.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy import stats

from .constants import BINARY, CLASSES, K, NOMINAL

log = logging.getLogger(__name__)

CONFIGS = [
    {"name": "Prespecified (kernel 0.75√p, quartiles)", "kw": 0.75, "disc": True, "discretizer": "quartile",
     "around": False},
    {"name": "Kernel 0.50√p, quartiles", "kw": 0.50, "disc": True, "discretizer": "quartile", "around": False},
    {"name": "Kernel 0.25√p, quartiles", "kw": 0.25, "disc": True, "discretizer": "quartile", "around": False},
    {"name": "Kernel 0.75√p, deciles", "kw": 0.75, "disc": True, "discretizer": "decile", "around": False},
    {"name": "Kernel 0.50√p, deciles", "kw": 0.50, "disc": True, "discretizer": "decile", "around": False},
    {"name": "Continuous, local sampling, kernel 0.75√p", "kw": 0.75, "disc": False, "discretizer": "quartile",
     "around": True},
    {"name": "Continuous, local sampling, kernel 0.50√p", "kw": 0.50, "disc": False, "discretizer": "quartile",
     "around": True},
]


def _top5(wmap: dict) -> set:
    return set(sorted(wmap, key=lambda f: -abs(wmap[f]))[:5])


def _jac(a, b) -> float:
    return len(set(a) & set(b)) / len(set(a) | set(b))


def _iqr(v, d: int = 2) -> str:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "—"
    return f"{np.median(v):.{d}f} ({np.percentile(v, 25):.{d}f}–{np.percentile(v, 75):.{d}f})"


def run(ctx, lime_mod, nv, f_proba, cases, CASE, SV_ALL, pos_of, lime_df) -> dict:
    """Ejecuta el análisis y registra la Tabla S11 (selección en desarrollo y comparación en la reserva)."""
    cfg, ds, rep = ctx.cfg, ctx.ds, ctx.rep
    COLS = ds.cols
    p = len(COLS)
    cat_idx = [COLS.index(c) for c in NOMINAL + BINARY if c in COLS]
    XD = nv.to_num(ds.X_dev).values
    seeds = (cfg.random_state, cfg.random_state + 101, cfg.random_state + 202)
    n_samples = cfg.lime_sensitivity_samples if not cfg.quick else cfg.lime_num_samples
    min_stab = cfg.lime_sensitivity_min_stability

    def make(conf, seed):
        return lime_mod.LimeTabularExplainer(
            XD, feature_names=COLS, class_names=list(CLASSES), categorical_features=cat_idx, mode="classification",
            random_state=seed, kernel_width=conf["kw"] * math.sqrt(p), discretize_continuous=conf["disc"],
            discretizer=conf["discretizer"], sample_around_instance=conf["around"])

    def one(expl, x, lab, n_feat=10):
        e = expl.explain_instance(x, f_proba, labels=(lab,), num_features=n_feat, num_samples=n_samples)
        sc = e.score[lab] if isinstance(e.score, dict) else e.score
        return float(sc), dict(e.as_map()[lab])

    # 1) estudiantes de SELECCIÓN: desarrollo, estratificados por clase (la reserva no se toca)
    rng = np.random.default_rng(cfg.random_state + 77)
    per = int(math.ceil(cfg.lime_cases / K))
    sel = np.concatenate([rng.choice(np.where(ds.y_dev == i)[0], min(per, int((ds.y_dev == i).sum())), replace=False)
                          for i in range(K)])
    XS = nv.to_num(ds.X_dev.iloc[sel]).values
    lab_s = f_proba(XS).argmax(1)
    n_stab = min(10, len(sel))
    rows = []
    for conf in CONFIGS:
        exps = [make(conf, s) for s in seeds]
        r2s, jacs = [], []
        for j, x in enumerate(XS):
            sc, wmap = one(exps[0], x, int(lab_s[j]))
            r2s.append(sc)
            if j < n_stab:
                sets = [_top5(wmap)] + [_top5(one(ex, x, int(lab_s[j]))[1]) for ex in exps[1:]]
                jacs.append(np.mean([_jac(a, b) for i_, a in enumerate(sets) for b in sets[i_ + 1:]]))
        rows.append({"name": conf["name"], "r2_median": float(np.median(r2s)), "r2_iqr": _iqr(r2s),
                     "pct_meeting": float(np.mean(np.asarray(r2s) >= cfg.lime_min_fidelity)),
                     "stability": float(np.mean(jacs))})
        log.info("LIME sensitivity | %-44s R² median %.3f | stability %.2f", conf["name"], rows[-1]["r2_median"],
                 rows[-1]["stability"])
    sel_df = pd.DataFrame(rows)

    # 2) regla de selección fijada de antemano
    elig = sel_df[sel_df.stability >= min_stab]
    if len(elig):
        chosen_name = elig.sort_values("r2_median", ascending=False).iloc[0]["name"]
        rule_out = "selected by the prespecified rule"
    else:
        chosen_name = CONFIGS[0]["name"]
        rule_out = f"no configuration reached a stability ≥ {min_stab:.2f}; the prespecified configuration is retained"
    chosen = next(c for c in CONFIGS if c["name"] == chosen_name)
    log.info("LIME sensitivity: %s (%s)", chosen_name, rule_out)

    # 3) aplicación ÚNICA a los casos de la reserva y al caso de la Fig. 6
    ex = make(chosen, cfg.random_state)
    hold_rows = []
    for i in cases:
        x = nv.to_num(ds.X_hold.iloc[[i]]).values[0]
        lab = int(ctx.P_best[i].argmax())
        sc, wmap = one(ex, x, lab)
        sc_all, _ = one(ex, x, lab, n_feat=p)
        row = {"case": int(i), "r2_sel": sc, "r2_all_predictors": sc_all}
        if SV_ALL is not None:
            fidx = list(wmap)
            lw = np.array([wmap[f] for f in fidx])
            sw = SV_ALL[pos_of[i], :, lab][fidx]
            row["spearman_abs"] = float(stats.spearmanr(np.abs(sw), np.abs(lw))[0]) if len(fidx) >= 3 else np.nan
            row["sign_agreement"] = float(np.mean(np.sign(sw) == np.sign(lw)))
        hold_rows.append(row)
    hs = pd.DataFrame(hold_rows)
    if len(lime_df):
        hs = hs.merge(lime_df[["case", "r2"]].rename(columns={"r2": "r2_prespecified"}), on="case", how="left")
    case_r2, _ = one(ex, nv.to_num(ds.X_hold.iloc[[CASE]]).values[0], 1)
    hs.to_csv(ctx.paths.supplementary / "lime_sensitivity_holdout_cases.csv", index=False)

    thr = cfg.lime_min_fidelity
    table = [{"Step": "Selection (development set)", "Configuration": r["name"],
              "Local R², median (IQR)": r["r2_iqr"], f"R² ≥ {thr:.2f}, %": f"{100 * r['pct_meeting']:.0f}",
              "Top-5 stability (Jaccard)": f"{r['stability']:.2f}",
              "Chosen": "yes" if r["name"] == chosen_name else ""} for r in rows]
    pre = hs.get("r2_prespecified")
    table += [
        {"Step": "Holdout (applied once)", "Configuration": "Prespecified",
         "Local R², median (IQR)": _iqr(pre) if pre is not None else "—",
         f"R² ≥ {thr:.2f}, %": f"{100 * np.mean(pre >= thr):.0f}" if pre is not None else "—",
         "Top-5 stability (Jaccard)": "", "Chosen": ""},
        {"Step": "Holdout (applied once)", "Configuration": chosen_name, "Local R², median (IQR)": _iqr(hs.r2_sel),
         f"R² ≥ {thr:.2f}, %": f"{100 * np.mean(hs.r2_sel >= thr):.0f}", "Top-5 stability (Jaccard)": "",
         "Chosen": "yes"},
        {"Step": "Holdout (applied once)", "Configuration": f"{chosen_name}, all {p} predictors (fidelity ceiling)",
         "Local R², median (IQR)": _iqr(hs.r2_all_predictors),
         f"R² ≥ {thr:.2f}, %": f"{100 * np.mean(hs.r2_all_predictors >= thr):.0f}", "Top-5 stability (Jaccard)": "",
         "Chosen": ""}]
    rep.register_table(
        pd.DataFrame(table), "TableS11_lime_sensitivity", "S11",
        "Post hoc sensitivity of LIME fidelity to its configuration",
        f"Exploratory analysis declared after the prespecified LIME showed low fidelity. Configurations were compared on "
        f"{len(sel)} development students only ({n_samples:,} perturbations, 10 predictors, predicted class); stability "
        f"is the mean pairwise Jaccard overlap of the top-5 predictors across {len(seeds)} random seeds "
        f"({n_stab} students). Rule fixed before use: highest median R² among configurations with stability ≥ "
        f"{min_stab:.2f}; otherwise the prespecified configuration is retained ({rule_out}). The chosen configuration "
        f"was then applied once to the same {len(hs)} holdout students used in Table S7; R² of the Fig. 6 case: "
        f"{case_r2:.2f}. The main figures and tables keep the prespecified configuration. Kernel width is expressed as a "
        "multiple of √p (p, number of predictors); IQR, interquartile range", supplementary=True)
    return {"chosen": chosen_name, "rule": rule_out, "fig6_case_r2": case_r2,
            "holdout_r2_median": float(hs.r2_sel.median())}
