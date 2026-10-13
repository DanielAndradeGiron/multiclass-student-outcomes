"""Carga, auditoría y partición de los datos (reserva bloqueada agrupada por perfil)."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import Config
from .constants import (BINARY, CLASSES, IDX, K, N_COLUMNS, NOMINAL, REFERENCE_SHA256, SEM1_PREFIX, SEM2_PREFIX,
                        TARGET, UCI_URL)

log = logging.getLogger(__name__)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def download_uci(dest: str | Path, url: str = UCI_URL, timeout: int = 60) -> Path:
    """Descarga el ZIP oficial de la UCI (id 697) y extrae ``data.csv``."""
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob = urllib.request.urlopen(url, timeout=timeout).read()  # noqa: S310 (URL fija y conocida)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        dest.write_bytes(z.read([n for n in z.namelist() if n.endswith(".csv")][0]))
    log.info("data.csv downloaded from UCI (id 697) → %s", dest)
    return dest


def read_raw(raw_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw_bytes), sep=";", encoding="utf-8-sig")
    if df.shape[1] == 1:
        df = pd.read_csv(io.BytesIO(raw_bytes), sep=",", encoding="utf-8-sig")
    df.columns = [re.sub(r"\s+", " ", c.replace("﻿", "")).strip() for c in df.columns]
    if TARGET not in df.columns or df.shape[1] != N_COLUMNS:
        raise ValueError(f"Unexpected data layout: shape {df.shape}; '{TARGET}' column required")
    return df


def predictor_columns(columns) -> list[str]:
    return [c for c in columns if c != TARGET]


def profile_codes(X: pd.DataFrame, cols: list[str] | None = None) -> np.ndarray:
    """Código entero del perfil completo de predictores (filas idénticas → mismo grupo)."""
    cols = cols if cols is not None else [c for c in X.columns if c not in (TARGET, "_row_id")]
    h = pd.util.hash_pandas_object(X[cols], index=False).to_numpy()
    return np.unique(h, return_inverse=True)[1].ravel()


@dataclass
class Dataset:
    df: pd.DataFrame
    all_pred: list[str]
    sem1: list[str]
    sem2: list[str]
    pred_sets: dict[str, list[str]]
    X_all: pd.DataFrame
    y_all: np.ndarray
    g_all: np.ndarray
    row_ids: np.ndarray
    dev_pos: np.ndarray
    hold_pos: np.ndarray
    n_dup: int
    data_sha: str
    sha_match: bool
    profile_audit: pd.DataFrame
    split_manifest: dict = field(default_factory=dict)

    # --- vistas convenientes -------------------------------------------------------------------
    @property
    def cols(self) -> list[str]:
        return self.pred_sets["End of first year (all)"]

    @property
    def X_dev(self) -> pd.DataFrame:
        return self.X_all.iloc[self.dev_pos].reset_index(drop=True)

    @property
    def y_dev(self) -> np.ndarray:
        return self.y_all[self.dev_pos]

    @property
    def g_dev(self) -> np.ndarray:
        return self.g_all[self.dev_pos]

    @property
    def dev_ids(self) -> np.ndarray:
        return self.row_ids[self.dev_pos]

    @property
    def X_hold(self) -> pd.DataFrame:
        return self.X_all.iloc[self.hold_pos].reset_index(drop=True)

    @property
    def y_hold(self) -> np.ndarray:
        return self.y_all[self.hold_pos]

    @property
    def g_hold(self) -> np.ndarray:
        return self.g_all[self.hold_pos]

    @property
    def hold_ids(self) -> np.ndarray:
        return self.row_ids[self.hold_pos]

    def holdout_hash(self) -> str:
        return sha256(",".join(map(str, sorted(self.hold_ids))).encode())


def load_dataset(cfg: Config, metadata_dir: Path | None = None) -> Dataset:
    """Lee y audita el archivo, construye X/y y bloquea la reserva (estratificada por las tres clases y
    agrupada por perfil de predictores). La reserva NO se usa hasta la evaluación final."""
    path = Path(cfg.data_path)
    if not path.exists():
        if not cfg.download_if_missing:
            raise FileNotFoundError(f"{path} not found. Run `eduoutcomes download-data` or set data_path.")
        download_uci(path)
    raw_bytes = path.read_bytes()
    data_sha = sha256(raw_bytes)
    sha_match = data_sha == REFERENCE_SHA256
    if not sha_match:
        msg = "The SHA-256 of the data file differs from the recorded reference (UCI 697). Verify the source."
        if cfg.require_reference_sha:
            raise ValueError(msg)
        log.warning(msg)
    if metadata_dir is not None:
        (metadata_dir / "source_manifest.json").write_text(json.dumps({
            "file": str(path), "sha256": data_sha, "bytes": len(raw_bytes),
            "reference_sha256_uci_697": REFERENCE_SHA256, "matches_reference": sha_match,
            "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat()}, indent=2), encoding="utf-8")

    df_raw = read_raw(raw_bytes)
    all_pred = predictor_columns(df_raw.columns)
    sem1 = [c for c in all_pred if c.startswith(SEM1_PREFIX)]
    sem2 = [c for c in all_pred if c.startswith(SEM2_PREFIX)]
    missing = [c for c in NOMINAL + BINARY if c not in all_pred]
    if missing or len(sem1) != 6 or len(sem2) != 6:
        raise ValueError(f"Unexpected predictor set (missing: {missing})")
    pred_sets = {"Enrolment only": [c for c in all_pred if c not in sem1 + sem2],
                 "Enrolment + 1st semester": [c for c in all_pred if c not in sem2],
                 "End of first year (all)": all_pred}

    n_dup = int(df_raw.duplicated().sum())
    df = df_raw.drop_duplicates().reset_index(drop=True)
    df.insert(0, "_row_id", np.arange(len(df)))
    outcome = df[TARGET].astype(str).str.strip()
    if set(outcome) != set(CLASSES):
        raise ValueError(f"Unexpected outcome labels: {set(outcome)}")
    y_all = outcome.map(IDX).to_numpy().astype(int)   # 0 = Dropout, 1 = Enrolled, 2 = Graduate
    counts = np.bincount(y_all, minlength=K)
    if cfg.expected_counts is not None and tuple(int(c) for c in counts) != tuple(cfg.expected_counts):
        raise ValueError(f"Unexpected class counts: {counts.tolist()} (expected {list(cfg.expected_counts)})")

    X_all = df[all_pred].copy()
    for c in NOMINAL:
        X_all[c] = X_all[c].astype(int).astype(str).astype(object)
    for c in [c for c in all_pred if c not in NOMINAL]:
        X_all[c] = X_all[c].astype(float)

    g_all = profile_codes(df, all_pred) if cfg.group_by_predictor_profile else np.arange(len(df))
    mult = pd.Series(g_all).value_counts()
    conflict = pd.DataFrame({"g": g_all, "y": y_all}).groupby("g").y.nunique().gt(1).sum()
    profile_audit = pd.DataFrame([
        {"Measure": "Records", "Value": f"{len(df):,}"},
        {"Measure": "Distinct predictor profiles", "Value": f"{len(mult):,}"},
        {"Measure": "Profiles shared by ≥ 2 records", "Value": f"{int((mult > 1).sum()):,}"},
        {"Measure": "Records in shared profiles", "Value": f"{int(mult[mult > 1].sum()):,}"},
        {"Measure": "Shared profiles with discordant outcomes", "Value": f"{int(conflict):,}"},
        {"Measure": "Largest profile size", "Value": f"{int(mult.max())}"}])

    # Reserva agrupada por perfil y estratificada por las tres clases (1 de 5 pliegues = 20 %)
    sgk = StratifiedGroupKFold(n_splits=round(1 / cfg.test_size), shuffle=True, random_state=cfg.random_state)
    dev_pos, hold_pos = next(sgk.split(np.zeros(len(df)), y_all, groups=g_all))
    if not set(g_all[dev_pos]).isdisjoint(g_all[hold_pos]):
        raise AssertionError("A predictor profile appears in both development and holdout")

    ds = Dataset(df=df, all_pred=all_pred, sem1=sem1, sem2=sem2, pred_sets=pred_sets, X_all=X_all, y_all=y_all,
                 g_all=g_all, row_ids=df["_row_id"].to_numpy(), dev_pos=dev_pos, hold_pos=hold_pos, n_dup=n_dup,
                 data_sha=data_sha, sha_match=sha_match, profile_audit=profile_audit)
    assert set(ds.dev_ids).isdisjoint(ds.hold_ids) and len(ds.dev_ids) + len(ds.hold_ids) == len(df)
    ds.split_manifest = {
        "n_total": len(df), "duplicates_removed": n_dup, "n_development": len(ds.y_dev), "n_holdout": len(ds.y_hold),
        "class_order": list(CLASSES),
        "dev_counts": dict(zip(CLASSES, map(int, np.bincount(ds.y_dev, minlength=K)))),
        "holdout_counts": dict(zip(CLASSES, map(int, np.bincount(ds.y_hold, minlength=K)))),
        "holdout_ids_sha256": ds.holdout_hash(), "status": "HOLDOUT_LOCKED"}
    if metadata_dir is not None:
        (metadata_dir / "split_manifest.json").write_text(json.dumps(ds.split_manifest, indent=2), encoding="utf-8")
        pd.DataFrame({"row_id": np.r_[ds.dev_ids, ds.hold_ids],
                      "partition": ["development"] * len(ds.dev_ids) + ["locked_holdout"] * len(ds.hold_ids)}
                     ).sort_values("row_id").to_csv(metadata_dir / "split_membership.csv", index=False)
    return ds


def make_synthetic(n: int = 900, seed: int = 0) -> pd.DataFrame:
    """Datos sintéticos con el MISMO esquema que UCI 697 (36 predictores + Target).

    Solo para pruebas y CI: permiten ejecutar el *pipeline* completo sin descargar los datos reales.
    Los resultados obtenidos con estos datos no tienen ningún valor sustantivo.
    """
    rng = np.random.default_rng(seed)
    y = rng.choice(3, n, p=[0.32, 0.18, 0.50])
    ability = rng.normal(0, 1, n) + (y - 1) * 1.0
    cols: dict[str, np.ndarray] = {}
    card = {"Marital status": 6, "Application mode": 18, "Course": 17, "Previous qualification": 17,
            "Nacionality": 21, "Mother's qualification": 29, "Father's qualification": 34,
            "Mother's occupation": 32, "Father's occupation": 46}
    order = ["Marital status", "Application mode", "Application order", "Course", "Daytime/evening attendance",
             "Previous qualification", "Previous qualification (grade)", "Nacionality", "Mother's qualification",
             "Father's qualification", "Mother's occupation", "Father's occupation", "Admission grade",
             "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date", "Gender",
             "Scholarship holder", "Age at enrollment", "International"]
    for c in order:
        if c in card:
            p = rng.dirichlet(np.ones(card[c]) * 0.6)
            cols[c] = rng.choice(np.arange(1, card[c] + 1), n, p=p)
        elif c in BINARY:
            logit = -0.2 + 0.6 * ability * (1 if c in ("Tuition fees up to date", "Scholarship holder") else -0.3)
            cols[c] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        elif c == "Application order":
            cols[c] = rng.integers(0, 9, n)
        elif c == "Age at enrollment":
            cols[c] = np.clip(np.round(20 - 2 * ability + rng.gamma(2, 2, n)), 17, 70).astype(int)
        else:
            cols[c] = np.round(np.clip(130 + 8 * ability + rng.normal(0, 10, n), 95, 190), 1)
    for sem in ("1st", "2nd"):
        enrolled = rng.integers(4, 9, n)
        approved = np.clip(np.round(enrolled * (0.6 + 0.25 * ability + rng.normal(0, 0.15, n))), 0, enrolled)
        base = f"Curricular units {sem} sem"
        cols[f"{base} (credited)"] = rng.poisson(0.3, n)
        cols[f"{base} (enrolled)"] = enrolled
        cols[f"{base} (evaluations)"] = enrolled + rng.poisson(1.5, n)
        cols[f"{base} (approved)"] = approved.astype(int)
        cols[f"{base} (grade)"] = np.where(approved > 0, np.round(np.clip(12 + ability + rng.normal(0, 1, n), 10, 18),
                                                                  4), 0.0)
        cols[f"{base} (without evaluations)"] = rng.poisson(0.15, n)
    cols["Unemployment rate"] = rng.choice([7.6, 8.9, 9.4, 10.8, 11.1, 12.4, 12.7, 13.9, 15.5, 16.2], n)
    cols["Inflation rate"] = rng.choice([-0.8, -0.3, 0.3, 0.5, 0.6, 1.4, 2.6, 2.8, 3.7], n)
    cols["GDP"] = rng.choice([-4.06, -3.12, -1.7, -0.92, 0.32, 0.79, 1.74, 1.79, 2.02, 3.51], n)
    df = pd.DataFrame(cols)
    df[TARGET] = np.array(CLASSES)[y]
    assert df.shape[1] == N_COLUMNS
    return df
