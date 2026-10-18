# Plan de análisis preespecificado — pipeline v3.0

Redactado **antes** de ejecutar con datos reales cualquier análisis nuevo de la v3.0. Su huella SHA-256 debe
registrarse (por ejemplo, en el mensaje del *commit* o en el material suplementario) antes de la corrida reportable.

## 1. Punto de partida

* La v2.0 (escenario equivalente a **T2**, 36 predictores) ya abrió una vez la reserva bloqueada
  (n = 885; SHA-256 de los identificadores `436359fd…`). La v3.0 reutiliza **exactamente** esa reserva
  (puerta de calidad `holdout_identical_to_v2_0`).
* Ningún análisis de la v3.0 usa la reserva para seleccionar modelos, hiperparámetros, umbrales, métodos de
  calibración ni configuraciones de explicabilidad.

## 2. Análisis primario (nuevo): escenarios temporales de predicción

| Escenario | Momento de la predicción | Predictores |
|---|---|---|
| T0-strict | Ingreso | 21: datos de ingreso con momento documentado (sin Debtor, Tuition fees up to date ni Scholarship holder) |
| T0 | Ingreso | 24: T0-strict + las 3 variables de momento no documentado |
| T1 | Fin del 1.er semestre | 30: T0 + 6 variables CU1 |
| T2 | Fin del 1.er año | 36: T1 + 6 variables CU2 (diseño v2.0) |

* Mismo protocolo en cada escenario: mismas particiones externas, internas, de calibración y de *stacking*;
  mismos modelos, rejillas (15 configuraciones por modelo), SMOTENC, *temperature scaling* y pruebas.
* Selección del modelo de cada escenario: la jerarquía de tolerancias de la v2.0 (F1 macro ± 0.005 → MCC ± 0.010 →
  AUC macro ± 0.005 → Brier ± 0.005 → menor tiempo), **solo con la CV**.
* Métrica primaria: F1 macro. Secundarias: MCC, recall de Enrolled, ECE por clase (y el resto de la Tabla 10).
* Comparación dentro de cada escenario: idéntica a la PI1 de la v2.0 (Tabla 3 de cada escenario).

## 3. Análisis secundario: comparación entre escenarios

* Contrastes preespecificados: **T0 − T0-strict**, **T1 − T0**, **T2 − T1**, **T2 − T0**.
* Modelos: el seleccionado en cada escenario y, como modelo fijo, la LR multinomial calibrada.
* CV: diferencias pareadas por pliegue; t corregida de Nadeau–Bengio; Holm dentro de cada (modelo, métrica) sobre
  los 4 contrastes; prueba t bayesiana correlacionada; ROPE ± 0.01 en F1 macro y MCC (sin margen preespecificado
  para recall ni ECE: solo dirección y Holm).
* Reserva: *bootstrap* estratificado y pareado (2 000 remuestras) sobre los mismos estudiantes.
* Explicabilidad: importancia SHAP (media |SHAP|) por escenario; estado de LIME por escenario.

## 4. Análisis exploratorios (declarados como tales)

1. **Sensibilidad de LIME** (Tabla S11 de cada escenario): 7 configuraciones evaluadas en 30 estudiantes de
   desarrollo; regla: mayor mediana de R² entre las de estabilidad top-5 (Jaccard, 3 semillas) ≥ 0.60; si ninguna
   cumple, se conserva la preespecificada. Se aplica una sola vez a la reserva.
2. **Validación temporal por cohorte inferida** (Tabla S14): origen móvil 2015, 2016, 2017; entrenamiento con las
   cohortes anteriores del **conjunto de desarrollo**; con y sin variables macroeconómicas; modelo seleccionado de cada
   escenario y LR. La cohorte se infiere de las variables macroeconómicas (10 combinaciones = Portugal 2008–2017).

## 5. Regla para la explicación local

Igual que en la v2.0, en cada escenario: estudiante de la reserva con desenlace Enrolled correctamente clasificado
cuya P(Enrolled) está más cerca de la mediana de P(Enrolled) entre esos estudiantes.

## 6. Interpretación obligatoria

* Enrolled = situación al término de la duración prevista (matriculado sin titulación); no es permanencia
  longitudinal.
* Todos los resultados son retrospectivos; solo T0/T0-strict corresponden al ingreso. T1 y T2 no son detección
  temprana al ingreso.
* Si T0 supera claramente a T0-strict, T0 puede ser optimista por fuga temporal o causalidad inversa.

## 7. Desviaciones respecto de la v2.0

| Cambio | Motivo |
|---|---|
| Escenarios temporales como análisis primario | Evitar presentar el rendimiento del fin del primer año como detección al ingreso |
| T0-strict | Momento de medición no documentado de Debtor, Tuition fees up to date y Scholarship holder |
| Análisis de horizonte de la Tabla 9 solo en T2 | En otros escenarios añadiría información posterior al momento de predicción |
| Nuevas métricas (PR-AUC por clase y macro, especificidad) | Solicitadas para el informe; no cambian la selección |
| Sensibilidad de LIME y validación temporal | Exploratorios, declarados |
