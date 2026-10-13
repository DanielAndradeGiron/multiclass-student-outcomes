# Contribuir

1. Crea una rama: `git checkout -b feature/descripcion`.
2. Instala el entorno de desarrollo: `pip install -e ".[dev]" && pre-commit install`.
3. Cambios que **no** alteran resultados (refactor, tiempo, documentación): basta con que `pytest` pase.
4. Cambios que **sí** alteran el diseño preespecificado (rejillas, márgenes, métricas, particiones): actualiza
   `configs/default.yaml`, sube `pipeline_version` y documenta el motivo en el PR. No modifiques el diseño después de
   haber abierto la reserva para una corrida que se vaya a reportar.
5. Mantén la convención: una etapa = un módulo en `src/eduoutcomes/stages/` con `run(ctx)`.
