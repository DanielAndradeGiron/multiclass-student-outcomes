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
    "brier": "Brier score", "log_loss": "Log-loss", "ece": "ECE (top-label)", "cw_ece": "Class-wise ECE",
    **{
        f"{m}_{c}": f"{n} ({c})"
        for c in CLASSES
        for m, n in [
            ("recall", "Recall"), ("precision", "Precision"), ("f1", "F1"), ("auc", "AUC"), ("ece", "ECE"),
            ("citl", "Calibration intercept"), ("slope", "Calibration slope"),
        ]
    },
}
CAL_LABEL = {
    "uncalibrated": "Uncalibrated", "temperature": "Temperature scaling", "isotonic": "Isotonic (OvR)",
    "dirichlet": "Dirichlet (matrix scaling)",
}


def short_name(col: str) -> str:
    """Rótulo abreviado para figuras (CU1/CU2 = unidades curriculares del 1.er/2.º semestre)."""
    s = col.replace(SEM1_PREFIX, "CU1").replace(SEM2_PREFIX, "CU2")
    s = s.replace("Nacionality", "Nationality").replace("Daytime/evening attendance", "Daytime attendance")
    return re.sub(r"\s*\((.*)\)", r" \1", s) if s.startswith("CU") else s
