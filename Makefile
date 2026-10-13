# Atajos (Linux/macOS/WSL). En Windows usa las tareas de VS Code o los comandos equivalentes del README.
PY ?= python

.PHONY: install data synthetic test test-fast lint format quick ci run clean

install:            ## Entorno de desarrollo
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

data:               ## Descarga UCI 697 y verifica el SHA-256
	eduoutcomes download-data

synthetic:          ## Datos sintéticos (esquema UCI 697) para pruebas
	eduoutcomes make-synthetic

test:               ## Todas las pruebas (incluye la prueba de humo de extremo a extremo)
	$(PY) -m pytest

test-fast:          ## Pruebas unitarias
	$(PY) -m pytest -m "not slow"

lint:
	ruff check src tests

format:
	ruff format src tests && ruff check --fix src tests

quick:              ## Prueba estructural con los datos reales (NO reportable)
	eduoutcomes run -c configs/quick.yaml

ci: synthetic       ## Igual que GitHub Actions
	eduoutcomes run -c configs/ci.yaml

run:                ## Corrida completa reportable del manuscrito
	eduoutcomes run -c configs/default.yaml

clean:
	rm -rf outputs/quick outputs/ci .pytest_cache .ruff_cache
