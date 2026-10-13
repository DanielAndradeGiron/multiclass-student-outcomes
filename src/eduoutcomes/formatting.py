"""Formato numérico tipográfico para tablas (signo menos U+2212, IC, valores p)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

MINUS = "−"


def num(x, d: int = 3, signed: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "NE"
    s = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    if float(s.replace("+", "")) == 0:  # evita “−0.000”
        s = s.replace("-", "+" if signed else "")
    return s.replace("-", MINUS)


def fmt_ci(est, lo, hi, d: int = 3, signed: bool = False) -> str:
    sep = " to " if signed or (np.isfinite(lo) and lo < 0) else "–"
    return f"{num(est, d, signed)} ({num(lo, d, signed)}{sep}{num(hi, d, signed)})"


def fmt_ms(mean, sd, d: int = 3) -> str:
    return f"{num(mean, d)} ± {num(sd, d)}"


def fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "NE"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


_LEAD = re.compile(r"[−\-+]?\d+(\.\d+)?")


def _lead(s) -> float:
    m = _LEAD.match(str(s).replace(MINUS, "-"))
    return float(m.group(0)) if m else np.nan


def best_cells(df: pd.DataFrame, cols_max=(), cols_min=()) -> set[tuple[int, str]]:
    """Celdas con el mejor valor (según el número inicial de la celda) para resaltarlas en negrita."""
    out: set[tuple[int, str]] = set()
    for cols, fn in ((cols_max, np.nanmax), (cols_min, np.nanmin)):
        for c in cols:
            v = df[c].map(_lead).to_numpy(float)
            if np.isfinite(v).any():
                out |= {(int(i), c) for i in np.where(np.isclose(v, fn(v)))[0]}
    return out
