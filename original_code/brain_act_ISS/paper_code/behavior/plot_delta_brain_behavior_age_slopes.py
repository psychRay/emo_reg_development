#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize age-varying conditional coupling between delta brain and behavior.

This script is designed as a post-hoc visualization for ROIs with a significant
brain x M_conv interaction in joint_analysis_roi_isc_behavior_delta_age_regression.py.
It uses the fitted joint model coefficients and developmental-model matrices:

    Coupling_ij = beta_brain_delta
                  + beta_interaction_M_nn * M_nn_ij
                  + beta_interaction_M_conv * M_conv_ij
                  + beta_interaction_M_div * M_div_ij

and summarizes how that conditional coupling changes over the empirical age range.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

if __package__ in {None, "", "behavior"}:
    THIS_DIR = Path(__file__).resolve().parent
    EMO_DIR = THIS_DIR.parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    if str(EMO_DIR) not in sys.path:
        sys.path.insert(0, str(EMO_DIR))
    from joint_analysis_roi_isc_behavior_age_regression import (  # noqa: E402
        build_models,
        fisher_z,
        _policy_flag,
        _resolve_prefix,
    )
    from joint_analysis_roi_isc_behavior_delta_age_regression import (  # noqa: E402
        DEFAULT_MATRIX_DIR,
        OUT_PREFIX,
        _align_delta_inputs,
        _load_schaefer_mapping,
        _prepare_vector,
        _roi_aliases,
        _resolve_roi_list,
    )
    from plotting_style import MODEL_COLORS, setup_publication_style  # noqa: E402
else:
    from .joint_analysis_roi_isc_behavior_age_regression import (  # noqa: E402
        build_models,
        fisher_z,
        _policy_flag,
        _resolve_prefix,
    )
    from .joint_analysis_roi_isc_behavior_delta_age_regression import (  # noqa: E402
        DEFAULT_MATRIX_DIR,
        OUT_PREFIX,
        _align_delta_inputs,
        _load_schaefer_mapping,
        _prepare_vector,
        _roi_aliases,
        _resolve_roi_list,
    )
    from ..plotting_style import MODEL_COLORS, setup_publication_style  # noqa: E402

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - scipy is expected in the analysis env.
    scipy_stats = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Post-hoc visualization for significant M_conv interaction ROIs: "
            "age-sorted conditional-coupling matrices and fixed-window coupling trends."
        )
    )
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--brain-stimulus-dir-name", type=str, default=None)
    p.add_argument("--behavior-stimulus-dir-name", type=str, default=None)
    p.add_argument("--reappraisal-name", type=str, default="Reappraisal")
    p.add_argument("--passive-name", type=str, default="Passive_Emo")
    p.add_argument("--brain-isc-method", type=str, default="mahalanobis")
    p.add_argument("--brain-isc-prefix", type=str, default=None)
    p.add_argument("--behavior-isc-method", type=str, default="mahalanobis")
    p.add_argument("--behavior-isc-prefix", type=str, default=None)
    p.add_argument("--fisher-z-brain", type=str, default=None)
    p.add_argument("--fisher-z-behavior", type=str, default=None)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--rank-transform", action="store_true", dest="rank_transform")
    g.add_argument("--no-rank-transform", action="store_false", dest="rank_transform")
    p.set_defaults(rank_transform=True)
    p.add_argument("--result-csv", type=Path, default=None)
    p.add_argument("--roi-p-col", type=str, default="p_fdr_perm_interaction_M_conv_model_wise")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--rois", nargs="+", default=None, help="Optional ROI subset after M_conv-FDR selection.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--schaefer-mapping-file", type=Path, default=None)
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument(
        "--window-width-years",
        type=float,
        default=None,
        help="Sliding-window width for averaging conditional coupling. Default is one quarter of the empirical age range.",
    )
    p.add_argument("--min-subjects-window", type=int, default=12)
    p.add_argument("--min-pairs-window", type=int, default=80)
    p.add_argument("--matrix-grid-size", type=int, default=50)
    p.add_argument(
        "--matrix-bandwidth-years",
        type=float,
        default=None,
        help="Local age-window width for the conditional-coupling matrix. Default is one sixth of the empirical age range.",
    )
    p.add_argument("--min-pairs-matrix", type=int, default=200)
    p.add_argument("--vlim-quantile", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-png", action="store_true")
    return p.parse_args()


def _default_result_csv(args: argparse.Namespace) -> Path:
    matrix_dir = Path(args.matrix_dir)
    brain_dir_name = args.brain_stimulus_dir_name or args.stimulus_dir_name
    behavior_dir_name = args.behavior_stimulus_dir_name or args.stimulus_dir_name
    if str(brain_dir_name) == str(behavior_dir_name):
        root = matrix_dir / str(brain_dir_name) / f"{args.reappraisal_name}_minus_{args.passive_name}"
    else:
        root = (
            matrix_dir
            / f"{brain_dir_name}_brain__{behavior_dir_name}_behavior"
            / f"{args.reappraisal_name}_minus_{args.passive_name}"
        )
    return root / f"{OUT_PREFIX}_joint.csv"


def _clean_label(text: str) -> str:
    label = str(text or "").strip()
    if not label:
        return label
    for prefix in ("7Networks_", "17Networks_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    return label


def _label_for_roi(roi: str, result_rows: pd.DataFrame, mapping: dict[str, dict]) -> str:
    sub = result_rows[result_rows["roi"].astype(str) == str(roi)]
    if not sub.empty:
        for col in ("source_name", "roi_schaefer_style", "parcel_name", "network"):
            if col in sub.columns:
                value = str(sub.iloc[0].get(col, "") or "").strip()
                if value and value.lower() != "nan":
                    return _clean_label(value)
    for key in _roi_aliases(str(roi)):
        if key in mapping and str(mapping[key].get("source_name", "")).strip():
            return _clean_label(str(mapping[key]["source_name"]))
    return str(roi)


def _row_for_roi(df: pd.DataFrame, roi: str) -> pd.Series:
    aliases = _roi_aliases(str(roi))
    for alias in aliases:
        sub = df[df["roi"].astype(str) == str(alias)]
        if not sub.empty:
            return sub.iloc[0]
    raise KeyError(f"Cannot find ROI row for {roi} using aliases {sorted(aliases)}")


def _numeric(row: pd.Series, col: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else float("nan")


def _summarize_coupling(values: np.ndarray, min_pairs: int) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x)
    n = int(mask.sum())
    if n < int(min_pairs):
        return {
            "n_pairs": n,
            "coupling_mean": np.nan,
            "coupling_sd": np.nan,
            "coupling_se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    xv = x[mask]
    mean = float(xv.mean())
    sd = float(xv.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(float(n)) if n > 0 else np.nan
    df = max(n - 1, 1)
    if se > 0.0 and scipy_stats is not None:
        tcrit = float(scipy_stats.t.ppf(0.975, df=df))
        ci_low = mean - tcrit * se
        ci_high = mean + tcrit * se
    else:
        ci_low = mean
        ci_high = mean
    return {
        "n_pairs": n,
        "coupling_mean": mean,
        "coupling_sd": sd,
        "coupling_se": float(se),
        "ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
        "ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
    }


def _box_sum(arr: np.ndarray, radius: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    n0, n1 = arr.shape
    out = np.zeros_like(arr, dtype=np.float64)
    integ = np.pad(arr, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    for i in range(n0):
        r0 = max(0, i - radius)
        r1 = min(n0 - 1, i + radius)
        for j in range(n1):
            c0 = max(0, j - radius)
            c1 = min(n1 - 1, j + radius)
            out[i, j] = (
                integ[r1 + 1, c1 + 1]
                - integ[r0, c1 + 1]
                - integ[r1 + 1, c0]
                + integ[r0, c0]
            )
    return out


def _local_coupling_grid(
    coupling: np.ndarray,
    age_i: np.ndarray,
    age_j: np.ndarray,
    age_min: float,
    age_max: float,
    n_grid: int,
    bandwidth_years: float,
    min_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(float(age_min), float(age_max), int(n_grid) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if not np.isfinite(edges).all() or edges[-1] <= edges[0]:
        raise ValueError("Invalid age range for local coupling grid")

    shape = (int(n_grid), int(n_grid))
    coupling_grid = np.full(shape, np.nan, dtype=np.float64)
    ns = np.zeros(shape, dtype=np.float64)
    coupling = np.asarray(coupling, dtype=np.float64).reshape(-1)
    age_i = np.asarray(age_i, dtype=np.float64).reshape(-1)
    age_j = np.asarray(age_j, dtype=np.float64).reshape(-1)
    valid = np.isfinite(coupling) & np.isfinite(age_i) & np.isfinite(age_j)
    coupling = coupling[valid]
    age_i = age_i[valid]
    age_j = age_j[valid]
    half_bw = max(float(bandwidth_years) * 0.5, 1e-9)

    # Average each cell from the actual dyads whose (age_i, age_j) coordinates
    # fall in the local age neighborhood around that cell center. This avoids
    # smoothing artifacts from applying rectangular prefix sums to the upper
    # triangular dyad space.
    for ii, ci in enumerate(centers):
        row_idx = np.flatnonzero(np.abs(age_i - float(ci)) <= half_bw)
        if row_idx.size < int(min_pairs):
            continue
        row_age_j = age_j[row_idx]
        row_coupling = coupling[row_idx]
        for jj, cj in enumerate(centers):
            local = np.abs(row_age_j - float(cj)) <= half_bw
            n_local = int(local.sum())
            ns[ii, jj] = float(n_local)
            if n_local < int(min_pairs):
                continue
            coupling_grid[ii, jj] = float(np.nanmean(row_coupling[local]))
    return coupling_grid, ns, centers, edges


def _matrix_from_grid(
    value_grid: np.ndarray,
    pair_bin_i: np.ndarray,
    pair_bin_j: np.ndarray,
    iu: np.ndarray,
    ju: np.ndarray,
    n_sub: int,
) -> np.ndarray:
    mat = np.full((int(n_sub), int(n_sub)), np.nan, dtype=np.float32)
    pair_vals = value_grid[pair_bin_i, pair_bin_j].astype(np.float32, copy=False)
    mat[iu, ju] = pair_vals
    mat[ju, iu] = pair_vals
    return mat


def _plot_matrix(
    matrix: np.ndarray,
    ages: np.ndarray,
    roi_label: str,
    out_base: Path,
    vlim_quantile: float,
    save_png: bool,
) -> None:
    import matplotlib.pyplot as plt

    vals = matrix[np.isfinite(matrix)]
    if vals.size == 0:
        return
    q = min(max(float(vlim_quantile), 0.5), 1.0)
    vmin = float(np.nanquantile(vals, 1.0 - q))
    vmax = float(np.nanquantile(vals, q))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    n = len(ages)
    tick_pos = np.linspace(0, n - 1, min(7, n), dtype=int)
    tick_labels = [f"{ages[i]:.1f}" for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("subject age (sorted)")
    ax.set_ylabel("subject age (sorted)")
    ax.set_title(f"{roi_label}: conditional coupling")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("conditional coupling")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    if save_png:
        fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_trend(df: pd.DataFrame, roi_label: str, out_base: Path, save_png: bool) -> None:
    import matplotlib.pyplot as plt

    ok = np.isfinite(pd.to_numeric(df["coupling_mean"], errors="coerce"))
    if not bool(ok.any()):
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    x = df["window_center"].to_numpy(dtype=float)
    y = df["coupling_mean"].to_numpy(dtype=float)
    ci_low = df["ci_low"].to_numpy(dtype=float)
    ci_high = df["ci_high"].to_numpy(dtype=float)
    color = MODEL_COLORS.get("M_conv", "#2ca25f")
    if np.isfinite(ci_low).any() and np.isfinite(ci_high).any():
        ax.fill_between(x, ci_low, ci_high, color=color, alpha=0.16, linewidth=0)
    ax.plot(x, y, color=color, linewidth=2.0)
    ax.scatter(x, y, color=color, edgecolor="black", linewidth=0.5, s=36, zorder=3)
    ax.axhline(0.0, color="#8f8f8f", linewidth=1.0)
    labels = [f"{a:.1f}-{b:.1f}" for a, b in zip(df["window_start"], df["window_end"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("age window")
    ax.set_ylabel("conditional coupling")
    ax.set_title(f"{roi_label}: sliding-window coupling")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    if save_png:
        fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_trend(df: pd.DataFrame, out_base: Path, save_png: bool) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=False)
    colors = ["#2ca25f", "#e34a33", "#3182bd", "#756bb1", "#fdbb84", "#636363"]
    for idx, (roi, sub) in enumerate(df.groupby("roi", sort=False)):
        sub = sub.sort_values("window_index")
        label = str(sub["roi_label"].iloc[0])
        ax.plot(
            sub["window_index"],
            sub["coupling_mean"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=colors[idx % len(colors)],
            label=label,
        )
    ax.axhline(0.0, color="#8f8f8f", linewidth=1.0)
    y_vals = pd.to_numeric(df["coupling_mean"], errors="coerce").to_numpy(dtype=float)
    y_vals = np.concatenate([y_vals[np.isfinite(y_vals)], np.asarray([0.0], dtype=float)])
    if y_vals.size:
        y_min = float(np.nanmin(y_vals))
        y_max = float(np.nanmax(y_vals))
        y_range = max(y_max - y_min, 1e-6)
        y_pad = max(y_range * 0.14, 0.015)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    first = df.sort_values("window_index").drop_duplicates("window_index")
    labels = [f"{a:.1f}-{b:.1f}" for a, b in zip(first["window_start"], first["window_end"])]
    ax.set_xticks(first["window_index"].to_numpy(dtype=float))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("age window", labelpad=8)
    ax.set_ylabel("conditional coupling")
    ax.set_title("Delta brain-behavior conditional coupling")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=2,
        borderaxespad=0.0,
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.subplots_adjust(bottom=0.34)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    if save_png:
        fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_publication_style()

    matrix_dir = Path(args.matrix_dir)
    brain_dir_name = args.brain_stimulus_dir_name or args.stimulus_dir_name
    behavior_dir_name = args.behavior_stimulus_dir_name or args.stimulus_dir_name
    result_csv = Path(args.result_csv) if args.result_csv is not None else _default_result_csv(args)
    if not result_csv.exists():
        raise FileNotFoundError(f"Cannot find joint result CSV: {result_csv}")
    result_df = pd.read_csv(result_csv)
    required = {"roi", str(args.roi_p_col)}
    beta_cols = {
        "beta_brain_delta",
        "beta_interaction_M_nn",
        "beta_interaction_M_conv",
        "beta_interaction_M_div",
    }
    required |= beta_cols
    missing = sorted(required.difference(result_df.columns))
    if missing:
        raise ValueError(f"{result_csv} missing required column(s): {missing}")

    sig_df = result_df[pd.to_numeric(result_df[str(args.roi_p_col)], errors="coerce") < float(args.alpha)].copy()
    sig_rois = sig_df["roi"].astype(str).drop_duplicates().tolist()
    if args.rois is not None:
        wanted_aliases = {alias for roi in args.rois for alias in _roi_aliases(str(roi))}
        sig_rois = [r for r in sig_rois if any(alias in wanted_aliases for alias in _roi_aliases(str(r)))]
    if not sig_rois:
        raise ValueError(f"No ROI with {args.roi_p_col} < {args.alpha}")

    subjects_file = result_csv.parent / f"{OUT_PREFIX}_joint_subjects_sorted.csv"
    if subjects_file.exists():
        sub_df = pd.read_csv(subjects_file)
        if not {"subject", "age"}.issubset(sub_df.columns):
            raise ValueError(f"{subjects_file} must contain subject and age columns")
        subjects = sub_df["subject"].astype(str).tolist()
        ages = pd.to_numeric(sub_df["age"], errors="coerce").to_numpy(dtype=np.float32)
        all_rois = result_df["roi"].astype(str).drop_duplicates().tolist()
    else:
        print(f"[warn] {subjects_file} not found; falling back to loading delta matrices for subject alignment.")
        brain_reapp_dir = matrix_dir / str(brain_dir_name) / str(args.reappraisal_name)
        brain_passive_dir = matrix_dir / str(brain_dir_name) / str(args.passive_name)
        behavior_reapp_dir = matrix_dir / str(behavior_dir_name) / str(args.reappraisal_name)
        behavior_passive_dir = matrix_dir / str(behavior_dir_name) / str(args.passive_name)
        for label, path in {
            "brain Reappraisal": brain_reapp_dir,
            "brain Passive": brain_passive_dir,
            "behavior Reappraisal": behavior_reapp_dir,
            "behavior Passive": behavior_passive_dir,
        }.items():
            if not path.exists():
                raise FileNotFoundError(f"Cannot find {label} directory: {path}")

        brain_prefix = _resolve_prefix(args.brain_isc_prefix, method=str(args.brain_isc_method), fallback_template="roi_isc_{method}_by_age")
        behavior_prefix = _resolve_prefix(
            args.behavior_isc_prefix,
            method=str(args.behavior_isc_method),
            fallback_template="behavior_pattern_isc_{method}_by_age",
        )
        _, _, subjects, ages, all_rois = _align_delta_inputs(
            brain_reapp_dir=brain_reapp_dir,
            brain_passive_dir=brain_passive_dir,
            behavior_reapp_dir=behavior_reapp_dir,
            behavior_passive_dir=behavior_passive_dir,
            brain_prefix=str(brain_prefix),
            behavior_prefix=str(behavior_prefix),
        )
    selected_rois = _resolve_roi_list(sig_rois, [str(r) for r in all_rois])

    order = np.argsort(np.asarray(ages, dtype=np.float64), kind="mergesort")
    subjects_sorted = [subjects[i] for i in order]
    ages_sorted = np.asarray(ages, dtype=np.float64)[order]

    n_sub = len(subjects_sorted)
    iu, ju = np.triu_indices(n_sub, k=1)
    pair_age_i = ages_sorted[iu]
    pair_age_j = ages_sorted[ju]
    age_min = float(np.nanmin(ages_sorted))
    age_max = float(np.nanmax(ages_sorted))
    age_range = max(age_max - age_min, 1e-6)
    window_width = float(args.window_width_years) if args.window_width_years is not None else age_range / 4.0
    matrix_bandwidth = (
        float(args.matrix_bandwidth_years)
        if args.matrix_bandwidth_years is not None
        else max(age_range / 6.0, window_width / 2.0)
    )
    n_windows = int(args.n_windows)
    if n_windows < 1:
        raise ValueError("--n-windows must be >= 1")
    if n_windows == 1:
        starts = np.asarray([age_min], dtype=np.float64)
    else:
        starts = np.linspace(age_min, age_max - window_width, n_windows, dtype=np.float64)
    ends = starts + window_width

    m_raw = build_models(np.asarray(ages_sorted, dtype=np.float32), iu, ju, normalize=False)
    if bool(args.rank_transform):
        age_models = [_prepare_vector(m, rank_transform=True).astype(np.float64) for m in m_raw]
    else:
        age_models = [_prepare_vector(m, rank_transform=False).astype(np.float64) for m in m_raw]
    m_nn, m_conv, m_div = age_models

    mapping = _load_schaefer_mapping(args.schaefer_mapping_file, matrix_dir=matrix_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else result_csv.parent / "figures" / "delta_brain_behavior_conditional_coupling"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_png = not bool(args.no_png)

    age_info = pd.DataFrame({"subject": subjects_sorted, "age": ages_sorted})
    age_info.to_csv(out_dir / "subjects_age_sorted.csv", index=False)
    selected_info = sig_df[sig_df["roi"].astype(str).isin(selected_rois)].copy()
    selected_info.to_csv(out_dir / "selected_mconv_interaction_rois.csv", index=False)

    n_grid = int(args.matrix_grid_size)
    edges = np.linspace(age_min, age_max, n_grid + 1)
    pair_bin_i = np.clip(np.searchsorted(edges, pair_age_i, side="right") - 1, 0, n_grid - 1)
    pair_bin_j = np.clip(np.searchsorted(edges, pair_age_j, side="right") - 1, 0, n_grid - 1)

    all_window_rows: list[dict] = []
    all_grid_rows: list[dict] = []
    for roi in selected_rois:
        roi_label = _label_for_roi(str(roi), sig_df, mapping)
        safe_roi = str(roi).replace("/", "_").replace("\\", "_")
        roi_dir = out_dir / safe_roi
        roi_dir.mkdir(parents=True, exist_ok=True)

        row_for_beta = _row_for_roi(result_df, str(roi))
        beta_brain = _numeric(row_for_beta, "beta_brain_delta")
        beta_nn = _numeric(row_for_beta, "beta_interaction_M_nn")
        beta_conv = _numeric(row_for_beta, "beta_interaction_M_conv")
        beta_div = _numeric(row_for_beta, "beta_interaction_M_div")
        if not np.isfinite([beta_brain, beta_nn, beta_conv, beta_div]).all():
            print(f"[warn] Skipping {roi}: missing beta coefficient(s)")
            continue
        coupling_vec = (
            beta_brain
            + beta_nn * m_nn
            + beta_conv * m_conv
            + beta_div * m_div
        ).astype(np.float64, copy=False)

        window_rows: list[dict] = []
        for wi, (start, end) in enumerate(zip(starts, ends), start=1):
            sub_mask = (ages_sorted >= float(start)) & (ages_sorted <= float(end))
            n_subjects = int(sub_mask.sum())
            pair_mask = sub_mask[iu] & sub_mask[ju]
            fit = _summarize_coupling(coupling_vec[pair_mask], min_pairs=int(args.min_pairs_window))
            if n_subjects < int(args.min_subjects_window):
                fit = {
                    **fit,
                    "coupling_mean": np.nan,
                    "coupling_sd": np.nan,
                    "coupling_se": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            row = {
                "roi": str(roi),
                "roi_label": roi_label,
                "beta_brain_delta": beta_brain,
                "beta_interaction_M_nn": beta_nn,
                "beta_interaction_M_conv": beta_conv,
                "beta_interaction_M_div": beta_div,
                "window_index": wi,
                "window_start": float(start),
                "window_end": float(end),
                "window_center": float((start + end) * 0.5),
                "n_subjects": n_subjects,
                **fit,
            }
            window_rows.append(row)
            all_window_rows.append(row)

        win_df = pd.DataFrame(window_rows)
        win_df.to_csv(roi_dir / f"{safe_roi}_sliding_window_coupling_summary.csv", index=False)
        _plot_trend(win_df, roi_label, roi_dir / f"{safe_roi}_sliding_window_coupling_trend", save_png=save_png)

        coupling_grid, n_grid_pairs, centers, _ = _local_coupling_grid(
            coupling_vec,
            pair_age_i,
            pair_age_j,
            age_min=age_min,
            age_max=age_max,
            n_grid=n_grid,
            bandwidth_years=matrix_bandwidth,
            min_pairs=int(args.min_pairs_matrix),
        )
        coupling_matrix = _matrix_from_grid(coupling_grid, pair_bin_i, pair_bin_j, iu, ju, n_sub)
        np.save(roi_dir / f"{safe_roi}_age_sorted_conditional_coupling_matrix.npy", coupling_matrix)
        _plot_matrix(
            coupling_matrix,
            ages_sorted,
            roi_label,
            roi_dir / f"{safe_roi}_age_sorted_conditional_coupling_matrix",
            vlim_quantile=float(args.vlim_quantile),
            save_png=save_png,
        )

        grid_df = pd.DataFrame(
            {
                "age_i_center": np.repeat(centers, len(centers)),
                "age_j_center": np.tile(centers, len(centers)),
                "n_pairs_local": n_grid_pairs.reshape(-1),
                "conditional_coupling": coupling_grid.reshape(-1),
            }
        )
        grid_df.insert(0, "roi_label", roi_label)
        grid_df.insert(0, "roi", str(roi))
        grid_df.to_csv(roi_dir / f"{safe_roi}_conditional_coupling_grid_summary.csv", index=False)
        all_grid_rows.extend(grid_df.to_dict("records"))

    all_win_df = pd.DataFrame(all_window_rows)
    all_win_df.to_csv(out_dir / "all_rois_sliding_window_coupling_summary.csv", index=False)
    pd.DataFrame(all_grid_rows).to_csv(out_dir / "all_rois_conditional_coupling_grid_summary.csv", index=False)
    _plot_combined_trend(all_win_df, out_dir / "all_rois_sliding_window_coupling_trend", save_png=save_png)

    print(
        "[done] "
        f"selected_rois={len(selected_rois)}, n_subjects={n_sub}, "
        f"age_range={age_min:.3f}-{age_max:.3f}, "
        f"window_width={window_width:.3f}, matrix_bandwidth={matrix_bandwidth:.3f}, "
        f"out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
