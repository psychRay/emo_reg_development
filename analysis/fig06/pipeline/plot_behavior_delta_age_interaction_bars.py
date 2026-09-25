#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot brain x age-model interaction beta bars for significant ROIs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from plotting_style import MODEL_COLORS, save_figure_svg_png, setup_publication_style, sig_stars  # noqa: E402
    from utils_network_assignment import get_tian_s2_labels  # noqa: E402
else:
    from .plotting_style import MODEL_COLORS, save_figure_svg_png, setup_publication_style, sig_stars
    from .utils_network_assignment import get_tian_s2_labels


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
DEFAULT_SCHAEFER_MAPPING_NAME = "schaefer2018_200parcels_7networks_order_mapping.csv"
EFFECTS = ("brain_x_M_nn", "brain_x_M_conv", "brain_x_M_div")
LEGACY_EFFECTS = {"brain_x_M_nn": "interaction_M_nn", "brain_x_M_conv": "interaction_M_conv", "brain_x_M_div": "interaction_M_div"}
EFFECT_LABELS = {"brain_x_M_nn": "brain x M_nn", "brain_x_M_conv": "brain x M_conv", "brain_x_M_div": "brain x M_div"}
EFFECT_MODELS = {"brain_x_M_nn": "M_nn", "brain_x_M_conv": "M_conv", "brain_x_M_div": "M_div"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot delta age-regression brain x age interaction beta bars")
    p.add_argument("--result-csv", type=Path, required=True, help="roi_isc_behavior_delta_age_regression_joint.csv or compatible CSV")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--p-prefix", type=str, default="p_fdr_perm")
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--schaefer-mapping-file", type=Path, default=None)
    return p.parse_args()


def _p_col(effect: str, prefix: str) -> str:
    return f"{prefix}_{effect}_model_wise"


def _value_from_effect(row: pd.Series, stem: str, effect: str) -> float:
    for candidate in (effect, LEGACY_EFFECTS.get(effect, effect)):
        col = f"{stem}_{candidate}"
        if col in row.index:
            return float(row.get(col, np.nan))
    return float("nan")


def _roi_aliases(roi: str) -> set[str]:
    text = str(roi).strip()
    aliases = {text} if text else set()
    m = re.fullmatch(r"([LRV])_(\d+)", text)
    if not m:
        return aliases
    hemi, raw = m.group(1), int(m.group(2))
    if hemi == "R":
        aliases.add(f"R_{raw - 100}" if raw > 100 else f"R_{raw + 100}")
    return aliases


def _strip_schaefer_prefix(name: str) -> str:
    text = str(name).strip()
    return re.sub(r"^7Networks_", "", text, flags=re.IGNORECASE)


def _display_label_from_record(roi: str, record: dict) -> str:
    structure = str(record.get("structure", "") or "").strip()
    hemi = str(record.get("hemi", "") or "").strip()
    if structure and hemi:
        return f"{hemi}_{structure}"
    if structure:
        return structure
    source = str(record.get("source_name", "") or "").strip()
    if source:
        return _strip_schaefer_prefix(source)
    parcel = str(record.get("parcel_name", "") or "").strip()
    if parcel and hemi:
        return f"{hemi}_{parcel}"
    if parcel:
        return parcel
    network = str(record.get("network", "") or record.get("network_full", "") or "").strip()
    if network and hemi:
        return f"{hemi}_{network}"
    if network:
        return network
    return str(roi)


def _find_schaefer_mapping(explicit: Path | None, result_csv: Path) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    for parent in [result_csv.parent, *result_csv.parents]:
        candidate = parent / DEFAULT_SCHAEFER_MAPPING_NAME
        if candidate.exists():
            return candidate
    return None


def _load_schaefer_mapping(args: argparse.Namespace) -> dict[str, dict]:
    path = _find_schaefer_mapping(args.schaefer_mapping_file, args.result_csv)
    out: dict[str, dict] = {}
    for _, row in get_tian_s2_labels().iterrows():
        roi = str(row.get("roi", "")).strip()
        if not roi:
            continue
        out[roi] = {
            "region_type": "subcortical",
            "structure": str(row.get("structure", "")),
            "structure_group": str(row.get("structure_group", "")),
            "hemi": str(row.get("hemi", "")),
        }
    if path is None:
        expected = Path(args.schaefer_mapping_file) if args.schaefer_mapping_file is not None else args.result_csv.parent / DEFAULT_SCHAEFER_MAPPING_NAME
        print(f"[mapping] Schaefer mapping not found, falling back to ROI labels: {expected}")
        return out
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        rec: dict[str, str] = {}
        for col in ("network", "network_full", "parcel_name", "source_name", "hemi"):
            if col in row.index and pd.notna(row.get(col)):
                rec[col] = str(row.get(col))
        for col in ("roi", "roi_schaefer_style", "source_name"):
            if col in row.index and pd.notna(row.get(col)):
                for alias in _roi_aliases(str(row.get(col))) | {str(row.get(col)).strip()}:
                    if alias:
                        out.setdefault(alias, rec)
    print(f"[mapping] Loaded Schaefer label mapping: {path}")
    return out


def _get_display_label(row: pd.Series, roi: str, mapping: dict[str, dict]) -> str:
    row_record = {
        col: row.get(col, "")
        for col in ("source_name", "network", "network_full", "parcel_name", "roi_schaefer_style", "hemi", "structure", "structure_group")
        if col in row.index and pd.notna(row.get(col))
    }
    if row_record:
        return _display_label_from_record(roi, row_record)
    for alias in _roi_aliases(str(roi)):
        if alias in mapping:
            return _display_label_from_record(roi, mapping[alias])
    return str(roi)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.result_csv)
    out_dir = Path(args.out_dir) if args.out_dir is not None else args.result_csv.parent / "figures" / "interaction_bars"
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = _load_schaefer_mapping(args)
    setup_publication_style()
    import matplotlib.pyplot as plt

    # One grouped bar chart of every ROI with FDR p < alpha on any effect.
    # Regions are on the x-axis; models are grouped within each region.
    sig_rois: list[dict] = []
    for _, row in df.iterrows():
        roi = str(row.get("roi", ""))
        pvals = {}
        for effect in EFFECTS:
            pvals[effect] = float("nan")
            for candidate in (effect, LEGACY_EFFECTS.get(effect, effect)):
                col = _p_col(candidate, str(args.p_prefix))
                if col in row.index:
                    pvals[effect] = float(row.get(col, np.nan))
                    break
        if not any(np.isfinite(p) and p < float(args.alpha) for p in pvals.values()):
            continue
        betas = [_value_from_effect(row, "beta", effect) for effect in EFFECTS]
        display_label = _get_display_label(row, roi, mapping)
        sig_rois.append({"roi": roi, "display_label": display_label, "betas": betas, "pvals": pvals})

    summary_rows: list[dict] = []
    if sig_rois:
        n_roi = len(sig_rois)
        n_eff = len(EFFECTS)
        group_w = 0.8
        bar_w = group_w / n_eff
        group_centers = np.arange(n_roi, dtype=float)
        colors = [MODEL_COLORS[EFFECT_MODELS[e]] for e in EFFECTS]

        fig_w = max(5.0, 1.6 * n_roi + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, 5.0), constrained_layout=True)

        all_betas = np.array(
            [b for rec in sig_rois for b in rec["betas"] if np.isfinite(b)], dtype=float
        )
        beta_min = float(np.nanmin(all_betas)) if all_betas.size else 0.0
        beta_max = float(np.nanmax(all_betas)) if all_betas.size else 0.0
        beta_span = max(beta_max - beta_min, 0.05)
        y_bottom = beta_min - 0.30 * beta_span
        y_top = beta_max + 0.35 * beta_span
        if y_bottom > -0.02:
            y_bottom = -0.02
        if y_top < 0.02:
            y_top = 0.02
        y_span = y_top - y_bottom
        label_offset_pos = 0.04 * y_span
        label_offset_neg = -0.04 * y_span

        for ei, effect in enumerate(EFFECTS):
            offsets = group_centers + (ei - (n_eff - 1) / 2.0) * bar_w
            heights = [rec["betas"][ei] for rec in sig_rois]
            ax.bar(
                offsets,
                heights,
                width=bar_w * 0.92,
                color=colors[ei],
                edgecolor="black",
                linewidth=0.8,
                label=EFFECT_LABELS[effect],
            )
            for x, rec in zip(offsets, sig_rois):
                beta = rec["betas"][ei]
                if not np.isfinite(beta):
                    continue
                pval = rec["pvals"][effect]
                if not np.isfinite(pval):
                    continue
                stars = sig_stars(rec["pvals"][effect])
                if stars == "":
                    continue
                y = beta + (label_offset_pos if beta >= 0 else label_offset_neg)
                ax.text(
                    x,
                    y,
                    stars,
                    ha="center",
                    va="bottom" if beta >= 0 else "top",
                    fontweight="bold" if stars != "n.s." else "normal",
                    fontsize=9 if stars != "n.s." else 8,
                )

        ax.axhline(0, color="#8c8c8c", lw=1.0)
        ax.set_ylim(y_bottom, y_top)
        ax.set_xticks(group_centers)
        ax.set_xticklabels([rec["display_label"] for rec in sig_rois], rotation=25, ha="right")
        ax.set_ylabel("beta coefficient")
        ax.set_title("Brain x age-model interaction")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10)

        out_svg = out_dir / "brain_age_interaction_grouped_bars.svg"
        save_figure_svg_png(fig, out_svg, save_png=not bool(args.no_png))
        plt.close(fig)
        for rec in sig_rois:
            summary_rows.append(
                {
                    "roi": rec["roi"],
                    "display_label": rec["display_label"],
                    "plot_status": "plotted",
                    "output_svg": out_svg.name,
                    **{f"p_{e}": rec["pvals"][e] for e in EFFECTS},
                }
            )
    if not summary_rows:
        summary_rows.append({"plot_status": "no_significant_interaction_roi", "result_csv": Path(args.result_csv).name})
    pd.DataFrame(summary_rows).to_csv(out_dir / "interaction_bars_summary.csv", index=False)


if __name__ == "__main__":
    main()
