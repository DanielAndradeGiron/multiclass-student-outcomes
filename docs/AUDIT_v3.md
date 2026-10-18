# Informe de auditoría — pipeline v2.0 → v3.0

Auditoría del paquete `eduoutcomes` (repositorio público, *commit* `37903b1`, 13-oct-2026), del notebook
`00_pipeline_original_v2.0.ipynb`, de sus salidas publicadas (`notebooks/outputs_multiclass_springer/`) y del manuscrito
`Andrade-Giron_et_al_EAIT_ES_limpio12-10.docx`. Todo lo que aquí se afirma sobre los datos se calculó con el archivo
oficial de la UCI (SHA-256 `3ef126de…`, verificado).

## 1. Hallazgos

| # | Componente | Problema identificado | Gravedad | Consecuencia metodológica | Corrección | Justificación |
|---|---|---|---|---|---|---|
| 1 | Manuscrito ↔ código | Las cifras del manuscrito no corresponden a las salidas publicadas del repositorio. Manuscrito: reserva 274/161/450, desarrollo 1 147/633/1 759, Votación suave − LR en F1 macro +0.015 (−0.003 a +0.032, *no concluyente*), Fig. 6 con XGBoost, P(Enrolled) = 0.591, R² LIME = 0.35. Repositorio: reserva 284/159/442, desarrollo 1 137/635/1 767, Votación suave − LR +0.025 (+0.013 a +0.037, *superior y prácticamente relevante*), modelo seleccionado Votación suave, P(Enrolled) = 0.631, R² = 0.27 | **Crítica** | Las conclusiones de la PI1 cambian según la corrida; un revisor que ejecute el código no reproducirá el artículo | Regenerar **todas** las cifras, tablas y figuras del manuscrito a partir de una única corrida reportable de la v3.0 (escenario T2 = diseño v2.0) | Reproducibilidad (TRIPOD+AI; política de datos y código de Springer) |
| 2 | Momento de la predicción | El análisis principal usa los 36 predictores, incluidos los resultados del 1.er y 2.º semestre, pero el texto habla de «sistema de alerta» | **Mayor** | El rendimiento corresponde al final del primer año; presentarlo como detección temprana sobrestima la utilidad práctica | Escenarios T0-strict, T0, T1 y T2 como análisis primario, con el mismo protocolo y la misma reserva; comparación entre escenarios (Tablas 10–12, Fig. 8) | Fuga temporal; interpretación honesta del momento de uso |
| 3 | Debtor, Tuition fees up to date, Scholarship holder | Clasificadas como «disponibles en la matrícula» (Tabla S1 de la v2.0) sin fuente. Realinho et al. (2022) no documentan cuándo se registraron. Con los datos: quien no tiene la matrícula al día abandona en el 86.6 % de los casos (24.7 % si la tiene al día), y el 50.2 % no aprobó ninguna unidad en el 2.º semestre (15.5 %) | **Mayor** | Posible fuga temporal y causalidad inversa (la deuda puede ser consecuencia del abandono); explica el peso de +25 pp que LIME dio a «matrícula al día» | T0-strict sin estas variables; T0 con ellas; contraste T0 − T0-strict; Tablas S1 (certeza del momento) y S16 | No suponer la temporalidad de una variable sin fuente |
| 4 | Validación temporal | El manuscrito afirma que no es posible porque falta el año de ingreso. Sin embargo, las 3 variables macroeconómicas toman exactamente 10 combinaciones que coinciden con las series anuales de Portugal 2008–2017 (desempleo 7.6 % en 2008 … 16.2 % en 2013 … 8.9 % en 2017) | **Mayor** (oportunidad) | Se puede aproximar una validación por cohortes | Cohorte inferida (Tabla S15) y validación de origen móvil exploratoria en el conjunto de desarrollo (Tabla S14), con y sin variables macroeconómicas | La cohorte es inferida: debe confirmarse con PORDATA/INE y declararse como tal |
| 5 | Variables macroeconómicas | Son constantes dentro de cada cohorte: en una partición aleatoria actúan como identificador de cohorte | Menor | El modelo puede aprender efectos de cohorte que no se trasladan a cohortes nuevas | Variante sin ellas en la validación temporal | Validez de transporte |
| 6 | Datos en el repositorio | `.gitattributes` (`* text=auto eol=lf`) convierte los finales de línea de `data.csv`; la copia clonada tiene SHA-256 `19737f…` ≠ `3ef126de…`, y `require_reference_sha: false` solo emite un aviso | Menor | Quien clone el repositorio usa un archivo que no supera la verificación de integridad | `*.csv -text` en `.gitattributes` | Integridad de la fuente |
| 7 | LIME | `categorical_features=[COLS.index(c) for c in NOMINAL + BINARY]` falla si falta algún predictor binario | Menor (error) | Impedía ejecutar T0-strict | Se filtra por los predictores del escenario | — |
| 8 | Calibración en la reserva | Los IC *bootstrap* percentil del ECE pueden no contener la estimación puntual (Tabla 6 de la v2.0: ECE por clase 0.036 con IC 0.038–0.058) | Menor | Un revisor lo verá como un error | Nota explicativa en las Tablas 6 y 11 (sesgo positivo del ECE en remuestras) | El ECE es un estimador sesgado |
| 9 | Friedman / Nemenyi | Se presentan como contraste de significación sobre 10 pliegues no independientes | Menor | Supuestos no satisfechos | El borrador los describe como descriptivos; los contrastes primarios siguen siendo Nadeau–Bengio + Holm + ROPE | Benavoli et al. (2017) |
| 10 | Métricas | Faltaban PR-AUC y especificidad por clase | Menor | Informe incompleto | Añadidas (`ap_*`, `pr_auc_macro`, `spec_*`), verificadas contra scikit-learn (máx. \|Δ\| = 2·10⁻¹⁶) | — |
| 11 | LIME | Fidelidad local baja (mediana R² = 0.34 en la v2.0) | Menor | LIME no es interpretable con el umbral preespecificado | Se conserva el umbral; sensibilidad post hoc elegida en desarrollo y aplicada una vez (Tabla S11) | Evitar ajustar LIME mirando la reserva |
| 12 | Textos | Notas con «36 predictores» fijo | Menor | Incorrecto en escenarios con menos predictores | Calculado a partir de los datos | — |

## 2. Componentes verificados que se conservan

Reserva estratificada y agrupada por perfil con huella SHA-256 (idéntica en v3.0: `436359fd…`); CV anidada 5 × 2 con
particiones agrupadas; SMOTENC, codificación y escalado dentro de `fit()`; *stacking* con predicciones OOF en pliegues
agrupados; *temperature scaling* primario con isotónica y Dirichlet como secundarios, ajustados con OOF internos;
Nadeau–Bengio + Holm + prueba t bayesiana correlacionada con ROPE ± 0.01; jerarquía de selección preespecificada;
congelación de modelos antes de abrir la reserva; SHAP por permutación sobre probabilidades calibradas con verificación
de aditividad; puertas de calidad de figuras Springer; exportación DOCX/LaTeX/Excel.

## 3. Resultados ya verificados con los datos reales

**Regresión logística de referencia, diseño reportable completo (CV anidada 5 × 2 en desarrollo; la reserva no se
usó).** El ajuste de la LR no depende de los demás modelos, de modo que estas cifras son exactamente las que dará la
corrida reportable (con las mismas versiones de librerías). En T2 reproduce la v2.0 (F1 macro 0.700 ± 0.017).

| Escenario | F1 macro | MCC | AUC macro | PR-AUC macro | Recall Enrolled | ECE por clase |
|---|---|---|---|---|---|---|
| T0-strict (21) | 0.501 ± 0.015 | 0.276 ± 0.020 | 0.699 ± 0.014 | 0.519 ± 0.015 | 0.403 ± 0.044 | 0.101 ± 0.004 |
| T0 (24) | 0.556 ± 0.019 | 0.367 ± 0.027 | 0.759 ± 0.012 | 0.598 ± 0.012 | 0.436 ± 0.034 | 0.093 ± 0.007 |
| T1 (30) | 0.662 ± 0.016 | 0.540 ± 0.020 | 0.850 ± 0.010 | 0.710 ± 0.012 | 0.489 ± 0.060 | 0.076 ± 0.006 |
| T2 (36) | 0.699 ± 0.017 | 0.597 ± 0.020 | 0.878 ± 0.009 | 0.748 ± 0.013 | 0.538 ± 0.062 | 0.070 ± 0.005 |

Contrastes (Nadeau–Bengio, Holm en 4 contrastes, ROPE ± 0.01) en F1 macro: T0 − T0-strict +0.056 (+0.029 a +0.083;
p Holm = 0.001), T1 − T0 +0.106 (+0.074 a +0.138), T2 − T1 +0.037 (+0.027 a +0.048), T2 − T0 +0.143 (+0.109 a
+0.177); todos «superiores y prácticamente relevantes». Es decir: (i) el rendimiento al ingreso es mucho menor que el
del final del primer año, y (ii) solo tres variables de momento no documentado aportan +0.056 de F1 macro al ingreso,
lo que hace imprescindible informar T0-strict.

**Corrida completa en modo rápido (no reportable) con los datos reales**: los cuatro escenarios, la comparación y la
validación temporal se ejecutaron sin errores y superaron todas las puertas de calidad, incluidas
`holdout_identical_to_v2_0` y `no_predictor_after_prediction_time`. Tablas S1, S15 y S16 (descriptivas) verificadas.

## 4. Lo que queda pendiente (no se ejecutó aquí)

* **La corrida reportable** (`eduoutcomes run-scenarios -c configs/default.yaml` y `temporal-validation`). En este
  entorno (2 núcleos, sin acceso a PyPI) no es viable: la v2.0 tardó ≈ 2.3 h con 8 procesos por escenario. Se verificó
  la ejecución completa en modo rápido con los datos reales (sin CatBoost, que no pudo compilarse aquí) y en modo CI con
  datos sintéticos.
* Confirmar con PORDATA/INE la correspondencia entre las variables macroeconómicas y los años 2008–2017.
* Confirmar con los autores del conjunto de datos (Realinho et al.) cuándo se registraron Debtor, Tuition fees up to
  date y Scholarship holder.

## 5. Limitaciones que permanecen

Una sola institución; validación interna y retrospectiva; *Enrolled* es una situación transversal al término de la
duración prevista (no permanencia longitudinal); cohorte inferida; 10 estimaciones pareadas con poca potencia.
