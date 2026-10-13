"""Métricas multiclase vectorizadas (verificadas contra scikit-learn) y medidas de calibración."""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import stats
from scipy.special import expit
from scipy.stats import beta as beta_dist
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

from .constants import CLASSES, K

EPS = 1e-6
DEFAULT_ECE_BINS = 15


def logp(P) -> np.ndarray:
    return np.log(np.clip(np.asarray(P, float), EPS, 1.0))


def softmax(Z) -> np.ndarray:
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def nll(y, P) -> float:
    P = np.clip(np.asarray(P, float), EPS, 1.0)
    return float(-np.mean(np.log(P[np.arange(len(y)), np.asarray(y).astype(int)])))


def ece_binary(yb, pb, bins: int = DEFAULT_ECE_BINS) -> float:
    """ECE con intervalos de igual anchura cerrados por la derecha (idéntico a ``np.digitize(right=True)``)."""
    yb, pb = np.asarray(yb, float), np.clip(np.asarray(pb, float), 0, 1)
    idx = np.searchsorted(np.linspace(0, 1, bins + 1)[1:-1], pb, side="left")
    return float(np.abs(np.bincount(idx, yb, bins) - np.bincount(idx, pb, bins)).sum() / len(pb))


def auc_binary(yk, s) -> float:
    """AUC = estadístico de Mann–Whitney con empates a 0.5 (idéntico a ``roc_auc_score``)."""
    yk = np.asarray(yk).astype(bool)
    n1 = int(yk.sum())
    n0 = len(yk) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return float((r[yk].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mc_metrics(y, P, ece_bins: int = DEFAULT_ECE_BINS) -> dict[str, float]:
    """Todas las métricas de discriminación y calibración de un vector de probabilidades (n × K)."""
    y = np.asarray(y).astype(int)
    P = np.asarray(P, float)
    n = len(y)
    z = P.argmax(1)
    C = np.bincount(y * K + z, minlength=K * K).reshape(K, K).astype(float)
    t, p, tp = C.sum(1), C.sum(0), np.diag(C)
    fp, fn = p - tp, t - tp
    rec = np.divide(tp, t, out=np.zeros(K), where=t > 0)
    prec = np.divide(tp, p, out=np.zeros(K), where=p > 0)
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros(K), where=den > 0)
    c, s = tp.sum(), float(n)
    cov_tp, cov_pp, cov_tt = c * s - t @ p, s * s - p @ p, s * s - t @ t
    pe = (t @ p) / s ** 2
    Y = np.eye(K)[y]
    aucs = [auc_binary(y == k, P[:, k]) for k in range(K)]
    out = {
        "accuracy": c / s, "balanced_accuracy": float(rec[t > 0].mean()), "f1_macro": float(f1.mean()),
        "f1_weighted": float((f1 * t).sum() / s),
        "mcc": float(cov_tp / math.sqrt(cov_tt * cov_pp)) if cov_tt * cov_pp > 0 else 0.0,
        "kappa": float((c / s - pe) / (1 - pe)) if pe < 1 else np.nan,
        "roc_auc_macro": float(np.mean(aucs)) if (t > 0).all() else np.nan,
        "brier": float(np.mean(np.sum((P - Y) ** 2, 1))), "log_loss": nll(y, P),
        "ece": ece_binary((z == y).astype(float), P.max(1), ece_bins),
    }
    eces = []
    for k, cl in enumerate(CLASSES):
        out[f"recall_{cl}"], out[f"precision_{cl}"], out[f"f1_{cl}"], out[f"auc_{cl}"] = rec[k], prec[k], f1[k], aucs[k]
        out[f"ece_{cl}"] = ece_binary(y == k, P[:, k], ece_bins)
        eces.append(out[f"ece_{cl}"])
    out["cw_ece"] = float(np.mean(eces))
    return out


def mc_metrics_sklearn(y, P) -> dict[str, float]:
    """Implementación de referencia con scikit-learn, usada solo para verificar :func:`mc_metrics`."""
    y = np.asarray(y).astype(int)
    P = np.asarray(P, float)
    z = P.argmax(1)
    full = len(np.unique(y)) == K
    out = {"accuracy": accuracy_score(y, z), "balanced_accuracy": balanced_accuracy_score(y, z),
           "f1_macro": f1_score(y, z, average="macro", labels=range(K), zero_division=0),
           "f1_weighted": f1_score(y, z, average="weighted", labels=range(K), zero_division=0),
           "mcc": matthews_corrcoef(y, z), "kappa": cohen_kappa_score(y, z),
           "roc_auc_macro": roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(K)))
           if full else np.nan}
    for k, cl in enumerate(CLASSES):
        yk = (y == k).astype(int)
        out[f"recall_{cl}"] = recall_score(yk, (z == k).astype(int), zero_division=0)
        out[f"precision_{cl}"] = precision_score(yk, (z == k).astype(int), zero_division=0)
        out[f"f1_{cl}"] = f1_score(yk, (z == k).astype(int), zero_division=0)
        out[f"auc_{cl}"] = roc_auc_score(yk, P[:, k]) if 0 < yk.sum() < len(yk) else np.nan
    return out


def verify_metrics(n_trials: int = 60, seed: int = 7) -> float:
    """Compara :func:`mc_metrics` con scikit-learn en casos aleatorios (incluye empates y clases sin predicción)."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr in range(n_trials):
            n = int(rng.integers(30, 400))
            y = rng.integers(0, K, n)
            P = rng.dirichlet(np.ones(K) * rng.uniform(0.3, 3), n)
            if tr % 5 == 0:  # sin predicciones de una clase
                P[:, 1] *= 0.01
                P /= P.sum(1, keepdims=True)
            if tr % 7 == 0:  # empates en las puntuaciones
                P = np.round(P, 1) + 1e-9
                P /= P.sum(1, keepdims=True)
            a, b = mc_metrics(y, P), mc_metrics_sklearn(y, P)
            for k_, v in b.items():
                if np.isnan(v):
                    assert np.isnan(a[k_]), k_
                else:
                    worst = max(worst, abs(a[k_] - v))
    if worst >= 1e-9:
        raise AssertionError(f"Vectorised metrics differ from scikit-learn (max |Δ| = {worst})")
    return worst


def logistic_recalibration(yb, pb, iters: int = 100) -> tuple[float, float]:
    """Calibración débil uno-contra-resto (Van Calster et al., 2019): intercepto con desplazamiento logit(p)
    ('calibration-in-the-large', ideal 0) y pendiente de la recalibración logística (ideal 1). Newton–Raphson."""
    yb = np.asarray(yb, float)
    pb = np.clip(np.asarray(pb, float), EPS, 1 - EPS)
    x = np.log(pb / (1 - pb))
    if yb.min() == yb.max():
        return np.nan, np.nan
    a = 0.0
    for _ in range(iters):
        mu = expit(a + x)
        h = (mu * (1 - mu)).sum()
        st_ = (yb - mu).sum() / max(h, 1e-12)
        a += st_
        if abs(st_) < 1e-10:
            break
    Xd = np.column_stack([np.ones_like(x), x])
    b = np.array([0.0, 1.0])
    for _ in range(iters):
        mu = expit(Xd @ b)
        W = mu * (1 - mu)
        step = np.linalg.solve(Xd.T @ (Xd * W[:, None]) + 1e-10 * np.eye(2), Xd.T @ (yb - mu))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return (float(a), float(b[1])) if np.isfinite([a, b[1]]).all() else (np.nan, np.nan)


def cal_lr_metrics(y, P) -> dict[str, float]:
    out = {}
    for k, cl in enumerate(CLASSES):
        out[f"citl_{cl}"], out[f"slope_{cl}"] = logistic_recalibration(np.asarray(y) == k, np.asarray(P)[:, k])
    return out


def reliability_q(yb, pb, bins: int = 10) -> np.ndarray:
    """Fiabilidad con intervalos de igual frecuencia e IC de Wilson al 95 % (columnas: p̄, ȳ, n, lo, hi)."""
    yb, pb = np.asarray(yb, float), np.asarray(pb, float)
    edges = np.unique(np.quantile(pb, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges[1:-1], pb, side="right"), 0, max(len(edges) - 2, 0))
    out = []
    z = 1.959964
    for b in np.unique(idx):
        s = idx == b
        n_ = int(s.sum())
        k_ = yb[s].sum()
        c_ = (k_ + z * z / 2) / (n_ + z * z)
        h_ = z * math.sqrt(k_ * (n_ - k_) / n_ + z * z / 4) / (n_ + z * z)
        out.append((pb[s].mean(), k_ / n_, n_, c_ - h_, c_ + h_))
    return np.array(out)


def strat_boot_idx(y, B: int, seed: int):
    """Índices *bootstrap* estratificados por clase."""
    rng = np.random.default_rng(seed)
    g = [np.where(y == c)[0] for c in range(K)]
    for _ in range(B):
        yield np.concatenate([rng.choice(x, len(x), replace=True) for x in g if len(x)])


def clopper(k: int, n: int, a: float = 0.05) -> tuple[float, float]:
    """IC exacto de Clopper–Pearson."""
    if n == 0:
        return np.nan, np.nan
    lo = 0.0 if k == 0 else beta_dist.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def net_benefit(yb, pb, t: float) -> float:
    n = len(yb)
    tp = ((pb >= t) & (yb == 1)).sum()
    fp = ((pb >= t) & (yb == 0)).sum()
    return tp / n - fp / n * t / (1 - t)
