"""Estilo de figuras de Springer Nature: dimensiones físicas exactas, paleta Okabe–Ito y tipografía."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Anchos de Springer Nature: 84 mm (una columna), 129 mm (1,5) y 174 mm (dos columnas); alto máximo 234 mm.
COL_MM = {"single": 84, "onehalf": 129, "double": 174}
MAX_H_MM = 234
MM = 1 / 25.4

OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
    "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F",
}
C_POS, C_NEG, C_REF, C_META = OKABE_ITO["vermillion"], OKABE_ITO["blue"], OKABE_ITO["grey"], OKABE_ITO["purple"]
C_CLASS = {"Dropout": OKABE_ITO["vermillion"], "Enrolled": OKABE_ITO["orange"], "Graduate": OKABE_ITO["blue"]}
M_CLASS = {"Dropout": "o", "Enrolled": "s", "Graduate": "^"}
GRID = "#E3E3E3"

CMAP_DIV = mpl.colors.LinearSegmentedColormap.from_list(
    "oi_div", [OKABE_ITO["blue"], "#DADADA", OKABE_ITO["vermillion"]])
CMAP_SEQ = mpl.colors.LinearSegmentedColormap.from_list("oi_seq", ["#FFFFFF", OKABE_ITO["blue"]])


def tint(color, f: float):
    """Mezcla un color con blanco SIN transparencia (compatible con EPS, TIFF RGB e impresión)."""
    r, g, b = mpl.colors.to_rgb(color)
    return (1 - f * (1 - r), 1 - f * (1 - g), 1 - f * (1 - b))


def pick_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"):
        if name in available:
            return name
    return "DejaVu Sans"


def apply_style() -> str:
    """Aplica los rcParams del estudio y devuelve la fuente elegida."""
    font = pick_font()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [font, "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.title_fontsize": 7, "legend.frameon": False, "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3, "legend.columnspacing": 1.0,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
        "lines.markersize": 3.5, "errorbar.capsize": 1.5, "patch.linewidth": 0.5, "hatch.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 3, "axes.labelpad": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True,
        "mathtext.default": "regular",
        "savefig.bbox": None, "savefig.pad_inches": 0.0, "figure.dpi": 120,
        "axes.prop_cycle": mpl.cycler(color=list(OKABE_ITO.values())),
    })
    return font


def new_figure(width: str = "double", height_mm: float = 90, **kw):
    """Figura con dimensiones físicas exactas y diseño 'constrained' (sin recortes al guardar)."""
    fig, ax = plt.subplots(figsize=(COL_MM[width] * MM, min(height_mm, MAX_H_MM) * MM), layout="constrained", **kw)
    fig.get_layout_engine().set(w_pad=1.5 * MM, h_pad=1.5 * MM)
    return fig, ax


def new_canvas(width: str = "double", height_mm: float = 90, wspace: float = 0.02, hspace: float = 0.02):
    fig = plt.figure(figsize=(COL_MM[width] * MM, min(height_mm, MAX_H_MM) * MM), layout="constrained")
    fig.get_layout_engine().set(w_pad=1.5 * MM, h_pad=1.5 * MM, wspace=wspace, hspace=hspace)
    return fig


def panel_label(ax, letter: str) -> None:
    """Rótulo de panel en negrita minúscula; se alinea al borde izquierdo real del panel al guardar."""
    ax.set_title(letter, loc="left", fontsize=9, fontweight="bold")
    ax._panel_letter = letter


def align_panel_labels(fig) -> None:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        letter = getattr(ax, "_panel_letter", None)
        if not letter:
            continue
        ax.set_title("", loc="left")
        bb = ax.get_tightbbox(r)
        x_ax = ax.transAxes.inverted().transform((bb.x0, 0))[0] if bb is not None else 0.0
        ax.set_title(letter, loc="left", fontsize=9, fontweight="bold", x=x_ax)


def min_font(fig) -> float:
    sizes = [t.get_fontsize() for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
    return float(min(sizes)) if sizes else float("nan")
