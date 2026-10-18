# Multiclass Prediction of Student Academic Outcomes

**Benchmarking calibrated and explainable ensemble and meta-ensemble learning**
Código reproducible del manuscrito enviado a *Education and Information Technologies* (Springer) — *pipeline* v3.0
(escenarios temporales de predicción; `scenario: T2` reproduce el diseño de la v2.0).

[![CI](https://github.com/OWNER/multiclass-student-outcomes/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/multiclass-student-outcomes/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> Reemplaza `OWNER` por tu usuario u organización de GitHub en las insignias y en `CITATION.cff`.

Datos: *Predict Students' Dropout and Academic Success* (Realinho et al., 2021, 2022; [UCI id 697](https://archive.ics.uci.edu/dataset/697)),
4 424 estudiantes y 36 predictores. Desenlace: `0 = Dropout`, `1 = Enrolled`, `2 = Graduate`.

| PI | Pregunta | Salidas |
|---|---|---|
| **PI1** | ¿Los ensambles y metaensambles discriminan mejor (estadística **y** prácticamente) que una LR multinomial calibrada? | Tablas 2, 3, 5, 6 · Fig. 2 |
| **PI2** | ¿Qué tan bien calibradas están las probabilidades por clase, en especial *Enrolled*? | Tablas 4, 6, S5 · Figs. 3, S2 |
| **PI3** | ¿Qué predictores impulsan las predicciones calibradas y qué tan consistentes son las explicaciones? | Tabla 8, S7–S8 · Figs. 5–6 |


## Novedades de la v3.0 — escenarios temporales de predicción

La v2.0 usaba los 36 predictores, incluidos los resultados del primer y segundo semestre; su rendimiento corresponde,
por tanto, al **final del primer año**, no al ingreso. La v3.0 evalúa por separado cuatro escenarios, con el mismo
protocolo y la **misma reserva** (huella SHA-256 idéntica a la v2.0):

| Escenario | Momento | Predictores |
|---|---|---|
| `T0_strict` | Ingreso | 21 (sin `Debtor`, `Tuition fees up to date`, `Scholarship holder`: momento de medición no documentado) |
| `T0` | Ingreso | 24 |
| `T1` | Fin del 1.er semestre | 30 |
| `T2` | Fin del 1.er año | 36 (= v2.0) |

```bash
eduoutcomes run-scenarios -c configs/default.yaml          # 4 escenarios + comparación (Tablas 10–12, Fig. 8)
eduoutcomes temporal-validation -c configs/default.yaml    # exploratorio: origen móvil por cohorte inferida
eduoutcomes run -c configs/default.yaml --set scenario=T1  # un solo escenario
```

Salidas: `outputs/main/<escenario>/` (mismas tablas y figuras de la v2.0 para cada escenario),
`outputs/main/comparison/` y `outputs/main/temporal_validation/`. Plan preespecificado:
[`docs/ANALYSIS_PLAN_v3.md`](docs/ANALYSIS_PLAN_v3.md); auditoría: [`docs/AUDIT_v3.md`](docs/AUDIT_v3.md).

Además: puertas de calidad temporales (ningún predictor posterior al momento de predicción), PR-AUC y especificidad
por clase, sensibilidad de LIME elegida en desarrollo (Tabla S11), cohortes inferidas (Tabla S15) y auditoría de las
variables de momento incierto (Tabla S16). Interpretación obligatoria: *Enrolled* no es permanencia longitudinal y la
predicción retrospectiva no es detección temprana.

> Duración orientativa: la v2.0 (un escenario) tardó ≈ 2.3 h con 8 procesos; `run-scenarios` multiplica ese tiempo
> por cuatro (T0 y T1 son algo más rápidos).

---

## Inicio rápido (VS Code)

1. **Clona** el repositorio y ábrelo en VS Code (`File → Open Folder…`). Acepta instalar las **extensiones recomendadas**.
2. **Crea el entorno**: `Terminal → Run Task… → Crear entorno (.venv) e instalar`, o en una terminal:

   ```bash
   python -m venv .venv
   # Linux/macOS:  source .venv/bin/activate      Windows (PowerShell):  .venv\Scripts\Activate.ps1
   pip install -U pip
   pip install -e ".[dev]"
   ```

   Selecciona el intérprete `.venv` (`Ctrl+Shift+P → Python: Select Interpreter`).
3. **Comprueba que todo funciona** (≈ 1–3 min, datos sintéticos): pestaña *Run and Debug* → **“Pipeline: prueba rápida (datos sintéticos)”**, o

   ```bash
   eduoutcomes make-synthetic
   eduoutcomes run -c configs/ci.yaml
   ```
4. **Corrida reportable** con los datos reales (se descargan y verifican automáticamente):

   ```bash
   eduoutcomes download-data          # opcional: run lo hace si falta data/raw/data.csv
   eduoutcomes run -c configs/default.yaml
   ```

   La duración depende de los núcleos (`outputs/main/metadata/runtime_log.csv` registra cada etapa). Si la
   ejecución se interrumpe, **vuelve a lanzar el mismo comando**: los bloques de CV, el ajuste final y la
   sensibilidad ya calculados se reanudan desde `outputs/main/cache/`.

Para reproducir **exactamente** las versiones del manuscrito: `pip install -r requirements-lock.txt && pip install -e . --no-deps`.

## Comandos

| Comando | Qué hace |
|---|---|
| `eduoutcomes run -c CONFIG [--quick] [--until ETAPA] [--set k=v ...]` | Ejecuta el *pipeline* (todo o hasta una etapa) |
| `eduoutcomes stages` | Lista las etapas en orden |
| `eduoutcomes show-config -c CONFIG [--set k=v]` | Muestra la configuración resuelta |
| `eduoutcomes download-data` | Descarga UCI 697 y verifica el SHA-256 de referencia |
| `eduoutcomes make-synthetic` | Datos sintéticos con el mismo esquema (solo pruebas/CI) |
| `eduoutcomes verify-metrics` | Contrasta las métricas vectorizadas con scikit-learn |

Ejemplos: `--set n_workers=4` (procesos), `--set output_dir=outputs/run2`, `--set reuse_cache=false`.

### Configuraciones

| Archivo | Uso | ¿Reportable? |
|---|---|---|
| `configs/default.yaml` | Diseño preespecificado del manuscrito (CV anidada 5 × 2, 2 000 *bootstrap*, …) | **Sí** |
| `configs/quick.yaml` | Datos reales con rejillas mínimas, para depurar | No |
| `configs/ci.yaml` | Datos sintéticos + modo rápido (GitHub Actions y pruebas) | No |

Los parámetros de cómputo (`n_workers`, cachés) **solo afectan al tiempo**, nunca a los resultados.

## Estructura del repositorio

```
├── configs/                     # default.yaml (manuscrito) · quick.yaml · ci.yaml
├── src/eduoutcomes/
│   ├── config.py                # Config preespecificada + carga YAML (extends, --set)
│   ├── constants.py             # clases, tipos de predictores, rótulos
│   ├── data.py                  # descarga, SHA-256, auditoría, reserva bloqueada agrupada por perfil
│   ├── models.py                # ResampledPipeline (SMOTENC en el ajuste), stacking agrupado, rejillas, ajuste
│   ├── calibration.py           # temperature · isotónica OvR · Dirichlet; calibración con OOF internos
│   ├── metrics.py               # métricas vectorizadas (verificadas), ECE, calibración débil, bootstrap
│   ├── inference.py             # Nadeau–Bengio, ROPE bayesiano, Holm, Friedman–Iman–Davenport–Nemenyi
│   ├── style.py · formatting.py # figuras Springer Nature y formato tipográfico
│   ├── reporting.py             # registro de tablas/figuras → DOCX, LaTeX sn-jnl, Excel, PDF/EPS/TIFF/PNG
│   ├── context.py · pipeline.py # estado compartido, registro de tiempos y orquestación
│   ├── cli.py                   # interfaz de línea de comandos
│   └── stages/                  # s01_data … s12_export (una etapa = una sección del manuscrito)
├── tests/                       # unitarias + prueba de humo de extremo a extremo
├── notebooks/                   # 00 = notebook original v2.0 (referencia) · 01 = ejecución interactiva
├── docs/ARCHITECTURE.md         # correspondencia notebook ↔ módulos y garantías de no-fuga
├── .vscode/                     # intérprete, pruebas, depuración y tareas
└── .github/                     # CI (lint + pruebas + humo), corrida completa manual, Dependabot
```

## Salidas (`outputs/<run>/`)

| Carpeta / archivo | Contenido |
|---|---|
| `tables/` | Tablas 1–9 en CSV, `Tables_Springer.docx` (tres líneas) y `tables_sn-jnl.tex` |
| `supplementary/` | Tablas S1–S10, Figs. S1–S3, casos LIME |
| `figures/` | Figs. 1–7 en PDF/EPS vectoriales, TIFF 600 ppp y PNG |
| `metadata/` | manifiestos de datos y partición, configuración congelada antes de la reserva, `quality_checks.json`, versiones, tiempos, log de la corrida, predicciones |
| `results_all.xlsx`, `results_draft.md`, `figure_captions.txt`, `TRIPOD_AI_checklist.md` | Resumen, borrador de resultados por PI, leyendas y lista TRIPOD+AI |
| `<run>.zip` | Paquete con todo lo anterior (sin cachés) y manifiesto SHA-256 |

Si una verificación de integridad falla (fuga entre particiones, columnas de `predict_proba`, tamaño/tipografía de
figuras, …) la corrida se detiene con un error explícito.

## Desarrollo y GitHub

```bash
pip install -e ".[dev]" && pre-commit install   # ruff + limpieza de salidas de notebooks en cada commit
pytest -m "not slow"                            # unitarias (segundos)
pytest                                          # + prueba de humo de extremo a extremo
ruff check src tests
```

Primer *push*:

```bash
git init -b main
git add .
git commit -m "Pipeline v2.0 as a Python package"
git remote add origin https://github.com/OWNER/multiclass-student-outcomes.git
git push -u origin main
```

* **CI** (`.github/workflows/ci.yml`): lint, pruebas en Ubuntu (Python 3.10/3.12) y Windows, y prueba de humo con datos sintéticos cuyo ZIP queda como *artifact*.
* **Corrida completa** (`full-run.yml`): `Actions → Full pipeline (manual) → Run workflow`; usa las versiones exactas de `requirements-lock.txt`.
* `data/raw/` y `outputs/` no se versionan. Publica el ZIP de resultados como *Release* (y archívalo en Zenodo para obtener un DOI).

## Cómo citar

Ver `CITATION.cff`. Datos: Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student
dropout and academic success. *Data*, 7(11), 146.

## Licencia

Código bajo licencia MIT (`LICENSE`). Los datos de la UCI se distribuyen bajo CC BY 4.0 y no se incluyen en el repositorio.
