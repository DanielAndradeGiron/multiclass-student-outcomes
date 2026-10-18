"""Configuración preespecificada del estudio.

Los valores por defecto reproducen exactamente el *pipeline* v2.0 del manuscrito. Los archivos YAML de
``configs/`` agrupan los parámetros por sección (solo por legibilidad); al cargarlos se aplanan en un
único :class:`Config`. Cualquier campo se puede sobrescribir desde la CLI con ``--set clave=valor``.
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    pipeline_version: str = "3.0"
    # --- escenario temporal de predicción (v3.0): T0_strict, T0, T1 o T2 (= 36 predictores, diseño v2.0) ---
    scenario: str = "T2"
    scenarios: tuple[str, ...] = ("T0_strict", "T0", "T1", "T2")   # los que ejecuta `run-scenarios`
    # --- datos y salidas ---
    data_path: str = "data/raw/data.csv"
    output_dir: str = "outputs"
    download_if_missing: bool = True
    expected_counts: tuple[int, ...] | None = (1421, 794, 2209)   # None desactiva la verificación
    require_reference_sha: bool = False                            # True: aborta si el SHA-256 difiere
    test_size: float = 0.20
    random_state: int = 42
    min_category_frequency: int = 20
    group_by_predictor_profile: bool = True
    # --- validación anidada repetida ---
    outer_splits: int = 5
    outer_repeats: int = 2
    inner_splits: int = 3
    calibration_splits: int = 3
    stacking_splits: int = 5
    search_n_iter: int = 15
    tuning_scoring: str = "f1_macro"
    # --- desbalance y calibración ---
    resampling: str = "smotenc"
    calibration_primary: str = "temperature"
    calibration_methods: tuple[str, ...] = ("uncalibrated", "temperature", "isotonic", "dirichlet")
    ece_bins: int = 15
    reliability_bins: int = 10
    # --- metaensambles ---
    meta_base: tuple[str, ...] = ("RandomForest", "ExtraTrees", "XGBoost", "LightGBM")
    # --- selección y relevancia práctica ---
    practical_margin_f1m: float = 0.010
    practical_margin_mcc: float = 0.010
    tol_f1m: float = 0.005
    tol_mcc: float = 0.010
    tol_auc: float = 0.005
    tol_brier: float = 0.005
    alpha: float = 0.05
    # --- reserva, decisión y subgrupos ---
    holdout_bootstrap: int = 2000
    capacity_levels: tuple[float, ...] = (0.10, 0.20, 0.30)
    subgroup_min_n: int = 30
    # --- sensibilidad ---
    sensitivity_splits: int = 5
    sensitivity_repeats: int = 1
    # --- explicabilidad ---
    perm_repeats: int = 20
    shap_background: int = 100
    shap_explain: int = 200
    shap_bootstrap: int = 200
    stability_models: int = 3
    lime_min_fidelity: float = 0.50
    lime_cases: int = 30
    lime_num_samples: int = 3000
    # Sensibilidad de LIME (post hoc, declarada): configuraciones elegidas SOLO con estudiantes de desarrollo
    lime_sensitivity: bool = True
    lime_sensitivity_samples: int = 5000
    lime_sensitivity_min_stability: float = 0.60
    # --- validación temporal exploratoria (cohorte inferida; solo conjunto de desarrollo) ---
    temporal_test_cohorts: tuple[int, ...] = (2015, 2016, 2017)
    # --- cómputo (no afectan a los resultados, solo al tiempo) ---
    n_jobs: int = -1
    n_workers: int = 0
    max_workers: int = 8
    fit_cache_size: int = 24
    resample_cache_size: int = 64
    reuse_cache: bool = True
    # --- figuras (Springer Nature) ---
    fig_formats: tuple[str, ...] = ("pdf", "eps", "tiff", "png")
    tiff_dpi: int = 600
    png_dpi: int = 300
    min_font_pt: float = 7.0
    show_figures: bool = False
    # --- modo rápido: SOLO prueba estructural, no reportable ---
    quick: bool = False
    allow_missing_libraries: bool = False   # permite respaldos (p. ej., sin SMOTENC) en pruebas estructurales
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


_TUPLE_FIELDS = {f.name for f in fields(Config) if "tuple" in str(f.type)}
_FIELD_NAMES = {f.name for f in fields(Config)}


def _flatten(d: dict[str, Any]) -> dict[str, Any]:
    """Aplana secciones del YAML (``cv: {outer_splits: 5}`` → ``outer_splits: 5``)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict) and k not in _FIELD_NAMES:
            for kk, vv in _flatten(v).items():
                if kk in out:
                    raise ValueError(f"Duplicated configuration key: {kk}")
                out[kk] = vv
        else:
            out[k] = v
    return out


def _coerce(name: str, value: Any) -> Any:
    if name in _TUPLE_FIELDS and isinstance(value, list):
        return tuple(value)
    return value


def _parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"Override must be key=value, got {item!r}")
    k, v = item.split("=", 1)
    k = k.strip()
    try:
        val = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        val = {"true": True, "false": False, "null": None, "none": None}.get(v.strip().lower(), v)
    if isinstance(val, list):
        val = tuple(val)
    return k, val


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                quick: bool | None = None) -> Config:
    """Carga un YAML (con herencia opcional mediante ``extends``) y aplica sobrescrituras."""
    values: dict[str, Any] = {}
    if path is not None:
        values = _load_yaml_with_extends(Path(path))
    for item in overrides or []:
        k, v = _parse_override(item)
        values[k] = v
    if quick is not None:
        values["quick"] = quick
    unknown = set(values) - _FIELD_NAMES
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    cfg = Config(**{k: _coerce(k, v) for k, v in values.items()})
    validate(cfg)
    return cfg


def _load_yaml_with_extends(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    base = _load_yaml_with_extends((path.parent / parent).resolve()) if parent else {}
    base.update(_flatten(raw))
    return base


def validate(cfg: Config) -> None:
    assert 0 < cfg.test_size < 1, "test_size must be in (0, 1)"
    assert cfg.outer_splits >= 2 and cfg.inner_splits >= 2 and cfg.calibration_splits >= 2
    assert cfg.calibration_primary in cfg.calibration_methods, "Primary calibration must be evaluated"
    assert "uncalibrated" in cfg.calibration_methods, "'uncalibrated' is needed for the calibration contrasts"
    assert cfg.resampling in ("smotenc", "random_oversampling", "none")
    from .constants import SCENARIOS
    assert cfg.scenario in SCENARIOS, f"scenario must be one of {SCENARIOS}"
    assert set(cfg.scenarios) <= set(SCENARIOS) and len(cfg.scenarios) >= 1
    assert set(cfg.calibration_methods) <= {"uncalibrated", "temperature", "isotonic", "dirichlet"}
    if cfg.expected_counts is not None:
        assert len(cfg.expected_counts) == 3
