"""Fixtures compartidas: datos sintéticos con el esquema UCI 697 (no se necesita descargar nada)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from eduoutcomes.config import load_config  # noqa: E402
from eduoutcomes.data import make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "data.csv"
    make_synthetic(600, seed=1).to_csv(path, sep=";", index=False)
    return path


@pytest.fixture()
def ci_config(synthetic_csv, tmp_path):
    return load_config("configs/ci.yaml", [f"data_path={str(synthetic_csv)!r}", f"output_dir={str(tmp_path / 'out')!r}",
                                           "n_workers=1"])
