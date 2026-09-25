#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared publication plotting style helpers for emo_final figures."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Sequence


BASE_FONT_SIZE = 14
TICK_FONT_SIZE = 13
SINGLE_PANEL_SIZE = (4.0, 5.0)
MODEL_COLORS = {
    "M_nn": "#e74c3c",
    "M_conv": "#2ca25f",
    "M_div": "#3182bd",
}


def setup_publication_style() -> None:
    """Configure Matplotlib for editable-text SVG publication figures."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": TICK_FONT_SIZE,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Arial",
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def panel_figsize(n_panels: int = 1, panel_width: float = 4.0, height: float = 5.0) -> tuple[float, float]:
    """Return a fixed-height figure size for aligned multi-panel figures."""
    return (max(float(panel_width), float(panel_width) * max(1, int(n_panels))), float(height))


def format_p_value(p: float | None) -> str:
    """Format p-values compactly for figure annotations."""
    if p is None:
        return "p=n/a"
    try:
        pv = float(p)
    except (TypeError, ValueError):
        return "p=n/a"
    if not (pv == pv):
        return "p=n/a"
    if pv < 0.001:
        return "p<0.001"
    if pv < 0.01:
        return f"p={pv:.3f}"
    return f"p={pv:.2f}"


def sig_stars(p: float | None) -> str:
    """Return significance stars for a p-value."""
    if p is None:
        return "n.s."
    try:
        pv = float(p)
    except (TypeError, ValueError):
        return "n.s."
    if not (pv == pv):
        return "n.s."
    if pv < 0.001:
        return "***"
    if pv < 0.01:
        return "**"
    if pv < 0.05:
        return "*"
    return "n.s."


def wrap_label(label: str, width: int = 14) -> str:
    """Wrap long axis labels at separators while keeping ROI names readable."""
    text = str(label)
    if len(text) <= int(width):
        return text
    parts = re.split(r"([_/\- ])", text)
    lines: list[str] = []
    cur = ""
    for part in parts:
        if len(cur) + len(part) <= int(width):
            cur += part
            continue
        if cur:
            lines.append(cur.strip())
        cur = part
    if cur:
        lines.append(cur.strip())
    return "\n".join([x for x in lines if x])


def apply_wrapped_xticklabels(ax, labels: Sequence[str], rotation: float = 45.0, ha: str = "right") -> None:
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([wrap_label(x) for x in labels], rotation=rotation, ha=ha)


def save_figure_svg_png(fig, out_svg: Path, save_png: bool = True, dpi: int = 300) -> tuple[Path, Path | None]:
    """Save SVG with editable text and optional PNG preview."""
    out_svg = Path(out_svg)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    out_png = None
    if bool(save_png):
        out_png = out_svg.with_suffix(".png")
        fig.savefig(out_png, format="png", dpi=int(dpi), bbox_inches="tight")
    return out_svg, out_png


def ensure_text_editable_svg(svg_path: Path) -> bool:
    """Quick check used by tests/smoke validation."""
    text = Path(svg_path).read_text(encoding="utf-8", errors="ignore")
    return "<text" in text


def model_color(model: str) -> str:
    return MODEL_COLORS.get(str(model), "#4d4d4d")


def finite_values(values: Iterable[float]) -> list[float]:
    out = []
    for value in values:
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v == v:
            out.append(v)
    return out
