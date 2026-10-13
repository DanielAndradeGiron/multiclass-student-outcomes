"""Contrastes estadísticos para CV repetida: Nadeau–Bengio, ROPE bayesiano, Holm, Friedman–Iman–Davenport–Nemenyi."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import studentized_range


def nb_test(d, test_ratio: float, alpha: float = 0.05):
    """t corregido de Nadeau–Bengio para CV repetida (Bouckaert y Frank, 2004): (media, lo, hi, t, p)."""
    d = np.asarray(d, float)
    n = len(d)
    mu = float(d.mean())
    v = d.var(ddof=1) if n > 1 else np.nan
    se = math.sqrt((1 / n + test_ratio) * v) if n > 1 and v > 0 else np.nan
    if not np.isfinite(se):
        return mu, np.nan, np.nan, np.nan, 1.0
    t = mu / se
    tq = stats.t.ppf(1 - alpha / 2, n - 1)
    return mu, mu - tq * se, mu + tq * se, t, float(2 * stats.t.sf(abs(t), n - 1))


def bayes_rope(d, rope: float, test_ratio: float):
    """Prueba t bayesiana correlacionada (Benavoli et al., 2017): (P(peor), P(ROPE), P(mejor))."""
    d = np.asarray(d, float)
    n = len(d)
    v = d.var(ddof=1) if n > 1 else np.nan
    sc = math.sqrt((1 / n + test_ratio) * v) if n > 1 and v > 0 else np.nan
    if not np.isfinite(sc):
        return np.nan, np.nan, np.nan
    dist = stats.t(n - 1, loc=d.mean(), scale=sc)
    lo, hi = dist.cdf(-rope), dist.cdf(rope)
    return float(lo), float(hi - lo), float(1 - hi)


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    adj[o] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    return adj


def verdict(diff, lo, hi, p_adj, margin, alpha: float = 0.05) -> str:
    if not np.isfinite(lo):
        return "not estimable"
    if lo > margin:
        return "superior (practically relevant)"
    if hi < -margin:
        return "inferior (practically relevant)"
    if p_adj < alpha:
        return ("superior" if diff > 0 else "inferior") + " (below practical margin)"
    if lo > -margin and hi < margin:
        return "practically equivalent"
    return "inconclusive"


def friedman_iman_davenport(pivot: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Friedman con corrección de Iman–Davenport, rangos medios (1 = mejor) y diferencia crítica de Nemenyi."""
    n_blocks, k = pivot.shape
    fr_stat, fr_p = stats.friedmanchisquare(*[pivot[c] for c in pivot.columns])
    denom = n_blocks * (k - 1) - fr_stat
    id_f = (n_blocks - 1) * fr_stat / denom if denom > 0 else np.inf
    id_df = (k - 1, (k - 1) * (n_blocks - 1))
    id_p = float(stats.f.sf(id_f, *id_df)) if np.isfinite(id_f) else 0.0
    ranks = pivot.rank(axis=1, ascending=False).mean()
    q_alpha = float(studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2))
    cd = q_alpha * math.sqrt(k * (k + 1) / (6 * n_blocks))
    return {"stat": float(fr_stat), "p": float(fr_p), "id_f": float(id_f), "id_df": id_df, "id_p": id_p,
            "ranks": ranks, "cd": float(cd), "k": k, "n_blocks": n_blocks}
