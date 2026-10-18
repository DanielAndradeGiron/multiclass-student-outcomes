"""Constantes fijas del conjunto de datos y rótulos de presentación.

Todo lo que aquí aparece describe el *dataset* (UCI id 697) o la forma de rotular resultados;
nada de esto es un hiperparámetro ajustable (para eso está :mod:`eduoutcomes.config`).
"""

from __future__ import annotations

import re

# --- Fuente de datos --------------------------------------------------------------------------
UCI_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
REFERENCE_SHA256 = "3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae"
TARGET = "Target"
# Huella de los identificadores de la reserva bloqueada en la v2.0 (seed 42, 20 %): debe ser idéntica en v3.0
HOLDOUT_SHA256_V2 = "436359fdfc27a8f43ae99918d78a5f6f5ae6e56b5d18eb314f914379ac5f8f80"
N_COLUMNS = 37  # 36 predictores + desenlace

# --- Desenlace multiclase (orden fijo de las columnas de predict_proba) -------------------------
CLASSES: tuple[str, ...] = ("Dropout", "Enrolled", "Graduate")
K = len(CLASSES)
IDX = {c: i for i, c in enumerate(CLASSES)}

# --- Tipos de predictores ---------------------------------------------------------------------
NOMINAL = [
    "Marital status", "Application mode", "Course", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation",
]
BINARY = [
    "Daytime/evening attendance", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "International",
]
SEM1_PREFIX = "Curricular units 1st sem"
SEM2_PREFIX = "Curricular units 2nd sem"
MACRO = ["Unemployment rate", "Inflation rate", "GDP"]

# --- Escenarios temporales de predicción (v3.0) -------------------------------------------------
# Realinho et al. (2022) agrupan Debtor, Tuition fees up to date y Scholarship holder como datos socioeconómicos,
# pero no documentan CUÁNDO se registraron. Si reflejan la situación durante o después del primer año (p. ej., quien
# abandona deja de pagar), su uso al ingreso sería fuga temporal y posible causalidad inversa. Por eso T0 se evalúa
# en una versión estricta (sin ellas) y en otra amplia (con ellas).
TIMING_UNCERTAIN = ["Debtor", "Tuition fees up to date", "Scholarship holder"]
SCENARIOS = ("T0_strict", "T0", "T1", "T2")
SCENARIO_LABEL = {
    "T0_strict": "T0-strict (enrolment, without fee/scholarship status)",
    "T0": "T0 (enrolment)",
    "T1": "T1 (end of 1st semester)",
    "T2": "T2 (end of 1st year)",
}
SCENARIO_SHORT = {"T0_strict": "T0-strict", "T0": "T0", "T1": "T1", "T2": "T2"}
SCENARIO_MOMENT = {
    "T0_strict": "At enrolment, before classes start; excludes predictors whose time of measurement is undocumented",
    "T0": "At enrolment, before classes start (fee and scholarship status assumed known at enrolment)",
    "T1": "When first-semester grades are closed",
    "T2": "When first-year (second-semester) grades are closed",
}

# Cohorte de ingreso INFERIDA a partir de las variables macroeconómicas. Las 10 combinaciones distintas presentes en
# los datos coinciden con las series anuales de Portugal 2008–2017 (PORDATA: tasa de desempleo, %, inflación, %, y
# variación del PIB, %). La correspondencia es una inferencia (la UCI no publica el año de ingreso) y debe
# verificarse con PORDATA/INE antes de citarla.
COHORT_BY_MACRO: dict[tuple[float, float, float], int] = {
    (7.6, 2.6, 0.32): 2008, (9.4, -0.8, -3.12): 2009, (10.8, 1.4, 1.74): 2010, (12.7, 3.7, -1.70): 2011,
    (15.5, 2.8, -4.06): 2012, (16.2, 0.3, -0.92): 2013, (13.9, -0.3, 0.79): 2014, (12.4, 0.5, 1.79): 2015,
    (11.1, 0.6, 2.02): 2016, (8.9, 1.4, 3.51): 2017,
}

# --- Modelos ----------------------------------------------------------------------------------
REF = "LogisticRegression (reference)"
ENSEMBLE_NAMES = ["RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost", "LightGBM", "CatBoost"]
META_MODELS = ["VotingSoft", "Stacking"]
# Desplazamiento de semilla fijo por estimador: un aprendiz base dentro de un metaensamble usa la
# misma semilla que el mismo modelo evaluado de forma individual (comparación pareada exacta).
MODEL_OFFSET = {n: 7 * (i + 1) for i, n in enumerate(ENSEMBLE_NAMES + [REF])}
DISPLAY = {
    "RandomForest": "Random forest", "ExtraTrees": "Extra trees", "GradientBoosting": "Gradient boosting",
    "XGBoost": "XGBoost", "LightGBM": "LightGBM", "CatBoost": "CatBoost", "VotingSoft": "Soft voting",
    "Stacking": "Stacking", REF: "Logistic regression",
}


def dn(model: str) -> str:
    """Nombre legible de un modelo."""
    return DISPLAY.get(model, model)


def family_of(model: str) -> str:
    if model == REF:
        return "Reference"
    return "Meta-ensemble" if model in META_MODELS else "Ensemble"


# --- Rótulos de métricas ----------------------------------------------------------------------
LABELS = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "f1_macro": "Macro-F1",
    "f1_weighted": "Weighted F1", "mcc": "MCC", "kappa": "Cohen's κ", "roc_auc_macro": "Macro AUC",
    "pr_auc_macro": "Macro PR-AUC", "brier": "Brier score", "log_loss": "Log-loss", "ece": "ECE (top-label)", "cw_ece": "Class-wise ECE",
    **{
        f"{m}_{c}": f"{n} ({c})"
        for c in CLASSES
        for m, n in [
            ("recall", "Recall"), ("precision", "Precision"), ("f1", "F1"), ("auc", "AUC"), ("ece", "ECE"),
            ("spec", "Specificity"), ("ap", "PR-AUC"),
            ("citl", "Calibration intercept"), ("slope", "Calibration slope"),
        ]
    },
}
CAL_LABEL = {
    "uncalibrated": "Uncalibrated", "temperature": "Temperature scaling", "isotonic": "Isotonic (OvR)",
    "dirichlet": "Dirichlet (matrix scaling)",
}


def scenario_predictors(all_pred: list[str], scenario: str) -> list[str]:
    """Predictores admitidos en cada escenario temporal, en el orden original del archivo.

    T0-strict ⊂ T0 ⊂ T1 ⊂ T2. Ningún escenario contiene información académica posterior a su momento de predicción.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}")
    sem1 = [c for c in all_pred if c.startswith(SEM1_PREFIX)]
    sem2 = [c for c in all_pred if c.startswith(SEM2_PREFIX)]
    if scenario == "T2":
        cols = list(all_pred)
    elif scenario == "T1":
        cols = [c for c in all_pred if c not in sem2]
    else:
        cols = [c for c in all_pred if c not in sem1 + sem2]
        if scenario == "T0_strict":
            cols = [c for c in cols if c not in TIMING_UNCERTAIN]
    assert_temporal_integrity(cols, scenario)
    return cols


def assert_temporal_integrity(cols: list[str], scenario: str) -> None:
    """Puerta de calidad: ningún predictor posterior al momento de predicción del escenario."""
    s1 = [c for c in cols if c.startswith(SEM1_PREFIX)]
    s2 = [c for c in cols if c.startswith(SEM2_PREFIX)]
    if scenario in ("T0_strict", "T0", "T1"):
        assert not s2, f"{scenario}: second-semester predictors are not available at prediction time: {s2}"
    if scenario in ("T0_strict", "T0"):
        assert not s1, f"{scenario}: first-semester predictors are not available at prediction time: {s1}"
    if scenario == "T0_strict":
        bad = [c for c in cols if c in TIMING_UNCERTAIN]
        assert not bad, f"T0_strict must exclude predictors with undocumented timing: {bad}"


def availability_of(col: str) -> tuple[str, str, str]:
    """(momento declarado, certeza, fuente) de cada predictor, según Realinho et al. (2022) y la UCI."""
    if col.startswith(SEM1_PREFIX):
        return "End of 1st semester", "documented", "Realinho et al. (2022), Table 1"
    if col.startswith(SEM2_PREFIX):
        return "End of 2nd semester", "documented", "Realinho et al. (2022), Table 1"
    if col in TIMING_UNCERTAIN:
        return ("Not stated (grouped as socioeconomic; may reflect status during the first year)", "unknown",
                "Realinho et al. (2022), Table 1; UCI 697")
    if col in MACRO:
        return ("Annual national indicator (year not stated; values match Portugal 2008–2017)", "inferred",
                "Realinho et al. (2022), PORDATA")
    return "Enrolment", "documented", "Realinho et al. (2022): information known at enrolment"


def short_name(col: str) -> str:
    """Rótulo abreviado para figuras (CU1/CU2 = unidades curriculares del 1.er/2.º semestre)."""
    s = col.replace(SEM1_PREFIX, "CU1").replace(SEM2_PREFIX, "CU2")
    s = s.replace("Nacionality", "Nationality").replace("Daytime/evening attendance", "Daytime attendance")
    return re.sub(r"\s*\((.*)\)", r" \1", s) if s.startswith("CU") else s
