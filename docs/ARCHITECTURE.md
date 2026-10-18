# Arquitectura y correspondencia con el notebook v2.0

El paquete reproduce el notebook `notebooks/00_pipeline_original_v2.0.ipynb` sección por sección. Las semillas,
particiones, rejillas, métricas y textos de tablas/figuras son los mismos; los cambios son de **organización**
(sin estado global, funciones con argumentos explícitos, configuración en YAML, registro con `logging`).

| Notebook | Módulo | Contenido |
|---|---|---|
| §1 Configuración | `config.py`, `configs/default.yaml`, `context.py` | `Config`, versiones, `StageTimer` |
| §1b Estilo | `style.py`, `formatting.py`, `reporting.py` | dimensiones Springer, formato, registro de tablas/figuras |
| §2 Datos y reserva | `data.py`, `stages/s01_data.py` | SHA-256, auditoría de perfiles, reserva agrupada y estratificada |
| §3 Tabla 1, S1–S2, Fig. 1 | `stages/s02_describe.py` | |
| §4 Arquitecturas | `models.py` | `ResampledPipeline`, `GroupedStackingClassifier`, `ModelFactory`, memoización |
| §5 Métricas y calibración | `metrics.py`, `calibration.py`, `models.ModelFactory.tune/fit_candidate` | |
| §6 CV anidada | `stages/s03_nested_cv.py` | `run_block` paralelo y reanudable |
| §7 PI1 | `inference.py`, `stages/s04_rq1_comparison.py` | Tablas 2–3, S3–S4, Fig. 2 |
| §8 PI2 | `stages/s05_rq2_calibration.py` | Tabla 4, S5, Figs. S1–S2 |
| §9 Ajuste final | `stages/s06_final_fit.py` | congelación antes de la reserva |
| §10 Reserva | `stages/s07_holdout.py` | Tablas 5–6, Fig. 3 |
| §11 Decisión | `stages/s08_decision.py` | Tabla 7, S9, Fig. 4 |
| §12 Subgrupos | `stages/s09_subgroups.py` | Tabla S6, Fig. S3 |
| §13 Sensibilidad | `stages/s10_sensitivity.py` | Tabla 9, S10, Fig. 7 |
| §14 PI3 | `stages/s11_explainability.py` | Tabla 8, S7–S8, Figs. 5–6 |
| §15 Integridad y exportación | `stages/s12_export.py`, `reporting.py` | `quality_checks.json`, DOCX/LaTeX/Excel, ZIP |

## Flujo

```
configs/*.yaml ──► Config ──► Context(paths, Reporter, ModelFactory)
                                  │
  s01 data ─► s02 describe ─► s03 nested CV ─► s04 RQ1 ─► s05 RQ2 ─► s06 final fit (FREEZE)
                                                                              │
                         s12 export ◄─ s11 explain ◄─ s10 sensitivity ◄─ s09 ◄─ s08 ◄─ s07 holdout (opened once)
```

## Garantías contra la fuga de información

* La reserva se separa en `s01` (agrupada por perfil y estratificada por las tres clases) y solo se lee en `s07`
  en adelante; su huella SHA-256 se comprueba al final.
* SMOTENC, codificación, escalado, búsqueda de hiperparámetros, *stacking* y calibración se ajustan **dentro** de
  `fit()` con los datos de entrenamiento del pliegue; `predict_proba` nunca remuestrea.
* Todas las particiones (externas, internas, calibración, *stacking*, sensibilidad) están agrupadas por perfil.
* La memoización (`models.STORE`) usa la huella exacta de datos + hiperparámetros + semillas: solo ahorra tiempo.

## Cachés

`outputs/<run>/cache/` guarda los bloques de la CV, los modelos finales y la sensibilidad, con una clave que incluye
configuración, rejillas, huella de los datos y versiones de librerías. Cambiar cualquiera de ellos invalida la caché.
`--set reuse_cache=false` fuerza el recálculo. Las cachés no entran en el ZIP de resultados.

## Añadir un modelo

1. `constants.ENSEMBLE_NAMES` y `MODEL_OFFSET` (semilla fija), `DISPLAY`.
2. `models.make_estimator` y `models.PARAM_GRIDS`.
3. Ejecuta `pytest`; la caché se invalida sola porque la lista de modelos forma parte de la clave.

## v3.0 — escenarios temporales

| Módulo | Contenido |
|---|---|
| `constants.py` | `SCENARIOS`, `scenario_predictors`, `assert_temporal_integrity`, `TIMING_UNCERTAIN`, `COHORT_BY_MACRO` |
| `data.py` | `Dataset.cols` = predictores del escenario; reserva y particiones con los perfiles de los 36 predictores (idénticas en todos los escenarios); cohorte inferida |
| `scenarios.py` | `run_scenarios` y `compare` (Tablas 10–12, S12–S13, Fig. 8) |
| `temporal.py` | validación temporal exploratoria por cohorte inferida (Tabla S14; solo desarrollo) |
| `lime_sensitivity.py` | sensibilidad de LIME elegida en desarrollo (Tabla S11) |

Garantía adicional: `assert_temporal_integrity` se comprueba al cargar los datos y como puerta de calidad final
(`no_predictor_after_prediction_time`); `holdout_identical_to_v2_0` verifica la huella de la reserva.
