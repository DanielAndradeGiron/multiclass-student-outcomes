# Cambios propuestos al manuscrito (v3.0)

Los valores entre corchetes `[…]` deben copiarse de la corrida reportable (`outputs/main/comparison/` y
`outputs/main/<escenario>/`). **Todas** las cifras actuales del manuscrito deben sustituirse por las de esa única corrida:
las actuales no coinciden con las salidas publicadas del repositorio (véase `AUDIT_v3.md`, hallazgo 1).

## Resumen

* Actual: «…predecir el abandono (Dropout), la permanencia sin titulación (Enrolled) y la titulación (Graduate)…»
* Propuesto: «…predecir el abandono (Dropout), la situación de matriculado sin titulación al término de la duración
  prevista (Enrolled) y la titulación (Graduate)…», y añadir: «Los modelos se evaluaron en cuatro escenarios definidos
  por la información disponible en el momento de la predicción —al ingreso, con y sin las variables de momento de
  medición no documentado; al final del primer semestre, y al final del primer año—. Al ingreso, el F1 macro fue
  [T0-strict] ([T0]); al final del primer año, [T2].»

## Palabras clave

«Sistemas de alerta temprana» puede mantenerse si el texto deja claro que solo T0/T0-strict corresponden al ingreso.

## Introducción (párrafo de la PI)

Añadir una pregunta o subobjetivo: «¿Cómo cambian la discriminación, la calibración y las explicaciones cuando solo se
usa la información disponible al ingreso, al final del primer semestre o al final del primer año?»

## Métodos — 3.1 Conjunto de datos

* Actual: «…por su disponibilidad, 24 se conocen en la matrícula, 6 al final del primer semestre y 6 al final del
  segundo.»
* Propuesto: «…por su disponibilidad, 21 se registran en la matrícula según el descriptor de los datos; para la condición
  de deudor, el pago de la matrícula al día y la condición de becario, el descriptor no indica cuándo se registraron
  (Tabla S1), por lo que podrían reflejar la situación durante el primer año; 6 predictores se conocen al final del
  primer semestre y 6 al final del segundo.»
* Añadir: «La base no incluye el año de ingreso; sin embargo, las diez combinaciones de las variables macroeconómicas
  coinciden con las series anuales de Portugal 2008–2017, lo que permite inferir la cohorte (Tabla S15).»
* Variable objetivo: «Enrolled designa a quienes seguían matriculados sin titularse al término de la duración prevista;
  es una situación transversal que agrupa trayectorias heterogéneas (titulación con retraso, abandono posterior,
  interrupciones) y no equivale a la permanencia medida longitudinalmente.»

## Métodos — nueva subsección «Escenarios de predicción»

«Para no utilizar información posterior al momento en que se pretende predecir, cada análisis se repitió en cuatro
escenarios: T0-strict (21 predictores de ingreso con momento documentado), T0 (24; añade deudor, matrícula al día y
beca), T1 (30; añade el primer semestre) y T2 (36; añade el segundo semestre). En cada escenario se reajustaron,
remuestrearon y recalibraron todos los modelos con las mismas particiones —incluida la misma reserva— y se seleccionó el
candidato con la regla preespecificada. Los escenarios se compararon con diferencias pareadas por pliegue (Nadeau–Bengio,
Holm y ROPE) y, en la reserva, con *bootstrap* pareado. Como análisis exploratorio se realizó una validación de origen
móvil por cohorte inferida en el conjunto de desarrollo (2015–2017), con y sin variables macroeconómicas.»

## Métodos — análisis de sensibilidad

El análisis de horizonte de la Tabla 9 queda subsumido por los escenarios (se conserva solo en T2). Añadir la
sensibilidad de LIME como análisis post hoc (Tabla S11).

## Resultados

* Nueva subsección con las Tablas 10–12 y la Fig. 8: rendimiento por escenario, contrastes T0 − T0-strict,
  T1 − T0, T2 − T1, T2 − T0 y cambio de la importancia SHAP (Tabla S12).
* En cada cifra de rendimiento, indicar el escenario (p. ej., «en T2, F1 macro = […]»).
* Explicaciones locales: usar el párrafo que declara que LIME no supera el umbral de fidelidad (R² = […] < 0.50) y que la
  discrepancia ilustra el riesgo de interpretar LIME sin control de fidelidad.

## Discusión

* Sustituir «la permanencia» por «la situación Enrolled» en todo el texto.
* Añadir: «Las diferencias entre escenarios cuantifican la información que aportan los resultados académicos
  posteriores; no implican que predecir más tarde sea preferible, porque el margen de intervención se reduce.»
* Interpretar T0 − T0-strict: si es grande, el rendimiento de T0 puede deberse a variables registradas después del
  ingreso (fuga temporal o causalidad inversa).

## Limitaciones

* Actual: «Los datos son transversales y no incluyen el año de ingreso, lo que impidió una validación temporal por
  cohortes…»
* Propuesto: «Los datos no incluyen el año de ingreso; la cohorte se infirió de las variables macroeconómicas y la
  validación temporal se limitó a un análisis exploratorio en el conjunto de desarrollo (Tabla S14). Todas las
  evaluaciones son retrospectivas: la reserva es una partición aleatoria de las mismas cohortes, por lo que no
  sustituye una validación externa ni prospectiva. El momento de registro de la deuda, el pago de la matrícula y la
  beca no está documentado.»

## Conclusión

* Actual: «…los resultados indican que un ensamble de boosting calibrado basta para un sistema de alerta útil…»
* Propuesto: «…con la información del primer año, un ensamble calibrado ofrece probabilidades fiables (T2: […]); al
  ingreso, el rendimiento es menor (T0-strict: […]) y la clase Enrolled apenas se distingue. Antes de usar el modelo
  como alerta temprana se requiere una validación prospectiva y por cohortes en el momento de uso previsto.»
* Sustituir «predecir el abandono, la permanencia y la titulación» por «predecir el abandono, la situación Enrolled y la
  titulación».

## Declaraciones

* Disponibilidad de datos: añadir la cita del conjunto (Realinho et al., 2021, UCI, https://doi.org/10.24432/C5MC89).
* Disponibilidad del código: actualizar la versión de Zenodo a la v3.0.
