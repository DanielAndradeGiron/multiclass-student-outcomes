"""Estructura de carpetas de salida."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def supplementary(self) -> Path:
        return self.root / "supplementary"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata"

    @property
    def cache(self) -> Path:
        """Cachés reanudables (bloques de CV, sensibilidad, ajuste final). Excluidas del paquete final."""
        return self.root / "cache"

    def make(self) -> Paths:
        for p in (self.root, self.figures, self.tables, self.supplementary, self.models, self.metadata, self.cache):
            p.mkdir(parents=True, exist_ok=True)
        return self
