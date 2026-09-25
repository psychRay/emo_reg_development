#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot behavior ISC developmental model permutation-null histograms.
"""

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
    from perm_null_io import load_perm_null_npz  # noqa: E402
    from plot_behavior_isc_dev_models_distributions import _bootstrap_rhos, _policy_flag, _prepare_condition, _resolve_prefix  # noqa: E402
    from plotting_style import format_p_value, setup_publication_style  # noqa: E402
else:
    from .perm_null_io import load_perm_null_npz
    from .plot_behavior_isc_dev_models_distributions import _bootstrap_rhos, _policy_flag, _prepare_condition, _resolve_prefix
    from .plotting_style import format_p_value, setup_publication_style


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
DEFAULT_MODELS = ("M_nn", "M_conv", "M_div")
NULL_BLUE = "#5AA1C8"
BOOT_GRAY = "#9e9e9e"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot behavior ISC x developmental model permutation-null histograms"
    )
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_emotion")
    p.add_argument("--result-file", type=str, default="behavior_isc_dev_models_perm_fwer.csv")
    p.add_argument("--null-file", type=str, default="behavior_isc_dev_models_perm_null.npz")
    p.add_argument(
        "--p-col",
        type=str,
        default="p_fdr_bh_model_wise",
        choices=("p_fdr_bh_model_wise", "p_perm", "p_fwer_global", "p_fdr_bh_global"),
        help=(
            "Statistic shown in the upper-right annotation. The default applies BH-FDR "
            "separately within each developmental model across conditions."
        ),
    )
    p.add_argument("--behavior-isc-method", type=str, default="mahalanobis", choices=("spearman", "pearson", "euclidean", "mahalanobis"))
    p.add_argument("--behavior-isc-prefix", type=str, default=None)
    p.add_argument("--assoc-method", type=str, default="spearman", choices=("pearson", "spearman"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument(
        "--include-negative",
        action="store_true",
        help="Plot negative empirical effects instead of skipping them.",
    )
    p.add_argument("--bins", type=int, default=35)
    p.add_argument("--no-png", action="store_true")

    p.add_argument(
        "--bootstrap-file",
        type=Path,
        default=None,
        help="Optional .npz or .csv file containing bootstrap rhos. If omitted, each condition reads bootstrap_rhos.npz when present.",
    )
    p.add_argument(
        "--bootstrap-key",
        type=str,
        default=None,
        help="Key name for bootstrap rhos in npz. If omitted, common names are auto-detected.",
    )
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=202405)
    p.add_argument("--fisher-z-behavior", type=str, default=None)
    p.add_argument("--no-normalize-models", action="store_false", dest="normalize_models", default=True)
    p.add_argument("--compute-bootstrap-if-missing", action="store_true", help="Compute bootstrap_rhos.npz during plotting only when it is missing.")
    p.add_argument("--recompute-bootstrap", action="store_true", help="Recompute bootstrap_rhos.npz even if it already exists.")
    return p.parse_args()


def _safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float).reshape(-1)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * float(ranked.size) / np.arange(1, ranked.size + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    out[valid] = restored
    return out


def _recompute_p_values(
    obs: np.ndarray,
    perm: np.ndarray,
    tail: str,
) -> dict[str, np.ndarray]:
    """Recompute all displayed p-values from the same null array used in the plot."""
    obs = np.asarray(obs, dtype=float).reshape(-1)
    perm = np.asarray(perm, dtype=float)
    if perm.ndim != 2 or perm.shape[0] != obs.size:
        raise ValueError(f"Unexpected permutation shape {perm.shape} for {obs.size} observed models")
    if perm.shape[1] <= 0:
        raise ValueError("Permutation-null array has no permutations")

    two_sided = str(tail).strip().lower() == "two_sided"
    obs_stat = np.abs(obs) if two_sided else obs
    perm_stat = np.abs(perm) if two_sided else perm
    valid_obs = np.isfinite(obs_stat)
    n_perm = int(perm.shape[1])

    p_perm = np.full(obs.shape, np.nan, dtype=float)
    p_fwer = np.full(obs.shape, np.nan, dtype=float)
    for mi in range(obs.size):
        if not valid_obs[mi]:
            continue
        p_perm[mi] = (np.count_nonzero(perm_stat[mi] >= obs_stat[mi]) + 1.0) / (n_perm + 1.0)

    with np.errstate(all="ignore"):
        max_stat = np.nanmax(perm_stat, axis=0)
    for mi in range(obs.size):
        if valid_obs[mi]:
            p_fwer[mi] = (np.count_nonzero(max_stat >= obs_stat[mi]) + 1.0) / (n_perm + 1.0)

    return {
        "p_perm": p_perm,
        "p_fwer_global": p_fwer,
        "p_fdr_bh_global": _bh_fdr(p_perm),
    }


def _compute_model_wise_fdr(
    stim_dirs: list[Path],
    *,
    result_file: str,
    null_file: str,
) -> dict[tuple[str, str], float]:
    """BH-correct condition-level permutation p-values separately for each model."""
    raw_rows: list[tuple[str, str, float]] = []
    for stim_dir in stim_dirs:
        result_csv = stim_dir / str(result_file)
        null_npz = stim_dir / str(null_file)
        if not result_csv.exists() or not null_npz.exists():
            continue

        result_df = pd.read_csv(result_csv)
        null = load_perm_null_npz(null_npz, required=("obs_rho", "perm_rhos", "models"))
        models = [str(x) for x in np.asarray(null["models"]).tolist()]
        obs = np.asarray(null["obs_rho"], dtype=float).reshape(-1)
        perm = np.asarray(null["perm_rhos"], dtype=float)
        if perm.ndim != 2:
            raise ValueError(f"Unexpected perm_rhos shape in {null_npz}: {perm.shape}")
        if perm.shape[0] != len(models) and perm.shape[1] == len(models):
            perm = perm.T
        if obs.size != len(models) or perm.shape[0] != len(models):
            raise ValueError(
                f"Model dimension mismatch in {null_npz}: models={len(models)}, "
                f"obs={obs.shape}, perm={perm.shape}"
            )

        metadata = null.get("metadata", {})
        tail = str(metadata.get("tail", "")).strip().lower()
        if tail not in {"positive", "two_sided"} and "tail" in result_df.columns and not result_df.empty:
            tail = str(result_df.iloc[0]["tail"]).strip().lower()
        if tail not in {"positive", "two_sided"}:
            tail = "positive"
        p_perm = _recompute_p_values(obs, perm, tail=tail)["p_perm"]
        raw_rows.extend((stim_dir.name, model, float(p_perm[mi])) for mi, model in enumerate(models))

    corrected: dict[tuple[str, str], float] = {}
    for model in sorted({model for _, model, _ in raw_rows}):
        model_rows = [(stim, p) for stim, row_model, p in raw_rows if row_model == model]
        qvals = _bh_fdr(np.asarray([p for _, p in model_rows], dtype=float))
        for (stim, _), qval in zip(model_rows, qvals):
            corrected[(stim, model)] = float(qval)
    return corrected


def _assert_values_match(
    *,
    label: str,
    model: str,
    left: float,
    right: float,
    left_source: str,
    right_source: str,
    atol: float = 1e-7,
) -> None:
    if np.isfinite(left) and np.isfinite(right) and not np.isclose(left, right, rtol=1e-6, atol=atol):
        raise ValueError(
            f"Inconsistent {label} for {model}: {left_source}={left:.10g}, "
            f"{right_source}={right:.10g}. The CSV and permutation NPZ likely come "
            "from different analysis runs; rerun the analysis before plotting."
        )


def _format_p_annotation(p_val: float, p_col: str) -> str:
    formatted = format_p_value(p_val)
    suffix = formatted[1:] if formatted.startswith("p") else f"={formatted}"
    labels = {
        "p_perm": "perm p",
        "p_fwer_global": "FWER p",
        "p_fdr_bh_global": "global BH-FDR adjusted p",
        "p_fdr_bh_model_wise": "BH-FDR adjusted p",
    }
    return f"{labels[p_col]}{suffix}"


def _set_svg_editable_text() -> None:
    import matplotlib as mpl

    mpl.rcParams["svg.fonttype"] = "none"   # Keep SVG text editable in Illustrator and Inkscape.
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["axes.unicode_minus"] = False


def _robust_xlim(values: np.ndarray, pad_frac: float = 0.08) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0

    lo, hi = np.percentile(vals, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))

    width = max(hi - lo, 1e-6)
    return float(lo - width * pad_frac), float(hi + width * pad_frac)


def _obs_xlim(obs_r: float, null_xlim: tuple[float, float]) -> tuple[float, float]:
    null_width = max(null_xlim[1] - null_xlim[0], 1e-6)
    obs_pad = max(null_width * 0.04, abs(obs_r) * 0.02, 0.001)
    return float(obs_r - obs_pad), float(obs_r + obs_pad)


def _draw_break_marks(ax_left, ax_right) -> None:
    kwargs = dict(marker=[(-1, -0.8), (1, 0.8)], markersize=9,
                  linestyle="none", color="k", mec="k", mew=1, clip_on=False)

    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **kwargs)
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **kwargs)


def _resolve_bootstrap_path(stim_dir: Path, bootstrap_file: Path | None) -> Path:
    if bootstrap_file is None:
        return stim_dir / "bootstrap_rhos.npz"
    p = Path(bootstrap_file)
    return p if p.is_absolute() else stim_dir / p


def _first_scalar(df: pd.DataFrame, col: str, fallback):
    if col not in df.columns or df.empty:
        return fallback
    vals = df[col].dropna()
    if vals.empty:
        return fallback
    return vals.iloc[0]


def _as_bool_value(value, fallback: bool) -> bool:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return bool(fallback)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return bool(fallback)


def _ensure_bootstrap_file(stim_dir: Path, args: argparse.Namespace, result_df: pd.DataFrame) -> Path | None:
    boot_path = _resolve_bootstrap_path(stim_dir, args.bootstrap_file)
    if args.bootstrap_file is not None:
        return boot_path
    if boot_path.exists() and not bool(args.recompute_bootstrap):
        return boot_path
    if not bool(args.recompute_bootstrap) and not bool(args.compute_bootstrap_if_missing):
        print(f"[bootstrap] {stim_dir.name}: missing {boot_path}; skip bootstrap panel")
        return None
    if int(args.n_bootstrap) <= 0:
        return None

    behavior_isc_method = str(_first_scalar(result_df, "behavior_isc_method", args.behavior_isc_method))
    assoc_method = str(_first_scalar(result_df, "assoc_method", args.assoc_method))
    prefix_from_csv = _first_scalar(result_df, "behavior_isc_prefix", None)
    prefix = str(args.behavior_isc_prefix) if args.behavior_isc_prefix is not None else str(prefix_from_csv) if prefix_from_csv is not None else _resolve_prefix(None, method=behavior_isc_method)
    fisher_z_fallback = _policy_flag(method=behavior_isc_method, override=args.fisher_z_behavior)
    fisher_z_eff = _as_bool_value(_first_scalar(result_df, "fisher_z_behavior", None), fallback=fisher_z_fallback)
    normalize_models = _as_bool_value(_first_scalar(result_df, "normalize_models", None), fallback=bool(args.normalize_models))

    if not (stim_dir / f"{prefix}.npy").exists() or not (stim_dir / f"{prefix}_subjects_sorted.csv").exists():
        print(f"[bootstrap] {stim_dir.name}: missing behavior ISC files for {prefix}; skip bootstrap panel")
        return None

    behavior_isc, subjects, ages, _iu, _ju, _beh_vec = _prepare_condition(
        stim_dir=stim_dir,
        behavior_isc_prefix=str(prefix),
        behavior_isc_method=behavior_isc_method,
        fisher_z_behavior=bool(fisher_z_eff),
    )
    print(f"[bootstrap] {stim_dir.name}: computing bootstrap rhos n={int(args.n_bootstrap)} -> {boot_path}")
    boot_by_model = _bootstrap_rhos(
        behavior_isc=behavior_isc,
        ages=ages,
        assoc_method=assoc_method,
        normalize_models=bool(normalize_models),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.bootstrap_seed),
        fisher_z_behavior=bool(fisher_z_eff),
    )
    boot_arr = np.vstack([np.asarray(boot_by_model[m], dtype=np.float64) for m in DEFAULT_MODELS])
    boot_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        boot_path,
        models=np.asarray(DEFAULT_MODELS, dtype=str),
        bootstrap_rhos=boot_arr,
        M_nn_bootstrap=np.asarray(boot_by_model["M_nn"], dtype=np.float64),
        M_conv_bootstrap=np.asarray(boot_by_model["M_conv"], dtype=np.float64),
        M_div_bootstrap=np.asarray(boot_by_model["M_div"], dtype=np.float64),
        subjects=np.asarray(subjects, dtype=str),
        ages=np.asarray(ages, dtype=np.float32),
        behavior_isc_prefix=np.asarray(str(prefix)),
        behavior_isc_method=np.asarray(behavior_isc_method),
        assoc_method=np.asarray(assoc_method),
        fisher_z_behavior=np.asarray(bool(fisher_z_eff)),
        normalize_models=np.asarray(bool(normalize_models)),
        bootstrap_seed=np.asarray(int(args.bootstrap_seed), dtype=np.int64),
        n_bootstrap=np.asarray(int(args.n_bootstrap), dtype=np.int64),
    )
    return boot_path


def _find_bootstrap_values(
    boot_file: Path | None,
    null_npz: Path,
    model: str,
    models: list[str],
    bootstrap_key: str | None = None,
) -> np.ndarray | None:
    """
    Load bootstrap correlation values.

    Accepted inputs:
    1. --bootstrap-file:
       - .npz with shape [n_models, n_boot] or [n_boot, n_models]
       - .csv with a model column and one of rho, boot_rho, bootstrap_rho, or bootstrapped_rho
    2. Without --bootstrap-file, the caller should pass bootstrap_rhos.npz.
       Common keys inside the permutation-null npz are still accepted.
    """
    source_file = boot_file if boot_file is not None else null_npz
    if source_file is None or not source_file.exists():
        return None

    model_to_i = {m: i for i, m in enumerate(models)}
    mi = model_to_i.get(model)
    if mi is None:
        return None

    if source_file.suffix.lower() == ".csv":
        df = pd.read_csv(source_file)
        if "model" not in df.columns:
            return None

        rho_col = None
        for c in ("rho", "boot_rho", "bootstrap_rho", "bootstrapped_rho", "r"):
            if c in df.columns:
                rho_col = c
                break

        if rho_col is None:
            return None

        vals = df.loc[df["model"].astype(str) == model, rho_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        return vals if vals.size else None

    if source_file.suffix.lower() == ".npz":
        with np.load(source_file, allow_pickle=True) as z:
            candidate_keys = []
            if bootstrap_key is not None:
                candidate_keys.append(bootstrap_key)

            candidate_keys.extend([
                "boot_rhos",
                "bootstrap_rhos",
                "bootstrapped_rhos",
                "boot_rho",
                "bootstrap_rho",
                "rhos_bootstrap",
                f"{model}_bootstrap",
                f"{model}_boot",
            ])

            key = next((k for k in candidate_keys if k in z.files), None)
            if key is None:
                return None

            arr = np.asarray(z[key], dtype=float)

        if arr.ndim == 1:
            vals = arr
        elif arr.ndim == 2:
            if arr.shape[0] == len(models):
                vals = arr[mi, :]
            elif arr.shape[1] == len(models):
                vals = arr[:, mi]
            else:
                return None
        else:
            return None

        vals = vals[np.isfinite(vals)]
        return vals if vals.size else None

    return None


def _save_svg_png(fig, out_svg: Path, save_png: bool = True) -> None:
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, bbox_inches="tight")

    if save_png:
        out_png = out_svg.with_suffix(".png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")


def plot_single_model(
    *,
    stim_name: str,
    model: str,
    obs_r: float,
    null_vals: np.ndarray,
    boot_vals: np.ndarray | None,
    p_val: float,
    p_col: str,
    out_svg: Path,
    bins: int,
    save_png: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    _set_svg_editable_text()
    setup_publication_style()

    null_vals = np.asarray(null_vals, dtype=float)
    null_vals = null_vals[np.isfinite(null_vals)]

    boot_vals = None if boot_vals is None else np.asarray(boot_vals, dtype=float)
    if boot_vals is not None:
        boot_vals = boot_vals[np.isfinite(boot_vals)]
        if boot_vals.size == 0:
            boot_vals = None

    null_xlim = _robust_xlim(null_vals)
    obs_outside_null = obs_r < null_xlim[0] or obs_r > null_xlim[1]

    # Use a broken-axis panel only when the observed value falls outside the null axis.
    # Otherwise draw the observed value as a vertical line on the null panel.
    need_obs_panel = bool(obs_outside_null)

    if boot_vals is not None and need_obs_panel:
        # Null panel, broken-axis observed panel, and bootstrap panel.
        fig = plt.figure(figsize=(8.8, 3.6))
        gs = fig.add_gridspec(
            1, 3,
            width_ratios=[4.2, 1.35, 3.75],
            wspace=0.22,
            left=0.09,
            right=0.98,
            top=0.82,
            bottom=0.22,
        )
        ax_null = fig.add_subplot(gs[0, 0])
        ax_obs = fig.add_subplot(gs[0, 1], sharey=ax_null)
        ax_boot = fig.add_subplot(gs[0, 2])
    elif boot_vals is not None and not need_obs_panel:
        # Null panel and bootstrap panel; the observed value is drawn on the null panel.
        fig = plt.figure(figsize=(7.6, 3.6))
        gs = fig.add_gridspec(
            1, 2,
            width_ratios=[4.2, 3.75],
            wspace=0.24,
            left=0.10,
            right=0.98,
            top=0.82,
            bottom=0.22,
        )
        ax_null = fig.add_subplot(gs[0, 0])
        ax_obs = None
        ax_boot = fig.add_subplot(gs[0, 1])
    elif boot_vals is None and need_obs_panel:
        # Null panel and broken-axis observed panel.
        fig = plt.figure(figsize=(5.9, 3.6))
        gs = fig.add_gridspec(
            1, 2,
            width_ratios=[4.2, 1.35],
            wspace=0.22,
            left=0.12,
            right=0.98,
            top=0.82,
            bottom=0.22,
        )
        ax_null = fig.add_subplot(gs[0, 0])
        ax_obs = fig.add_subplot(gs[0, 1], sharey=ax_null)
        ax_boot = None
    else:
        # Null panel only; the observed value is drawn on it.
        fig = plt.figure(figsize=(4.6, 3.6))
        gs = fig.add_gridspec(
            1, 1,
            left=0.15,
            right=0.97,
            top=0.82,
            bottom=0.22,
        )
        ax_null = fig.add_subplot(gs[0, 0])
        ax_obs = None
        ax_boot = None

    # Null distribution.
    ax_null.hist(
        null_vals,
        bins=int(bins),
        color=NULL_BLUE,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
        density=False,
    )
    ax_null.set_xlim(*null_xlim)
    ax_null.set_xlabel("rho")
    ax_null.set_ylabel("Occurrence")
    ax_null.spines["top"].set_visible(False)
    ax_null.spines["right"].set_visible(False)
    ax_null.xaxis.set_major_locator(MaxNLocator(5))

    # Draw the observed value on the null panel when it falls inside that axis.
    if not need_obs_panel:
        ax_null.axvline(obs_r, color="#d7301f", lw=2.0, ls="--")

    # Broken-axis panel for an observed value outside the null axis.
    if ax_obs is not None:
        obs_xlim = _obs_xlim(obs_r, null_xlim)
        ax_obs.set_xlim(*obs_xlim)
        ax_obs.axvline(obs_r, color="#d7301f", lw=2.0, ls="--")
        ax_obs.set_xlabel("rho")
        ax_obs.spines["top"].set_visible(False)
        ax_obs.spines["left"].set_visible(False)
        ax_obs.spines["right"].set_visible(False)
        ax_obs.tick_params(axis="y", left=False, labelleft=False)
        ax_obs.set_xticks([obs_xlim[0], obs_xlim[1]])
        ax_obs.set_xticklabels([f"{obs_xlim[0]:.3f}", f"{obs_xlim[1]:.3f}"])
        ax_obs.tick_params(axis="x", pad=2)

    ymax = max(ax_null.get_ylim()[1], 1.0)
    ax_null.set_ylim(0, ymax)
    if ax_obs is not None:
        ax_obs.set_ylim(0, ymax)

    # Place annotations on the figure so they do not cover the axes.
    fig.text(
        0.5,
        0.94,
        f"{stim_name} | {model}",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.98,
        0.91,
        f"obs rho={obs_r:.3f}    {_format_p_annotation(p_val, p_col)}",
        ha="right",
        va="center",
        fontsize=11,
    )

    # Bootstrap distribution.
    if ax_boot is not None and boot_vals is not None:
        boot_xlim = _robust_xlim(boot_vals)
        # Keep the observed-value line inside the visible bootstrap axis.
        boot_width = max(boot_xlim[1] - boot_xlim[0], 1e-6)
        pad = boot_width * 0.05
        lo = min(boot_xlim[0], obs_r - pad)
        hi = max(boot_xlim[1], obs_r + pad)
        boot_xlim = (lo, hi)
        ax_boot.hist(
            boot_vals,
            bins=int(bins),
            color=BOOT_GRAY,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            density=False,
        )
        ax_boot.axvline(obs_r, color="#d7301f", lw=2.0, ls="--")
        ax_boot.set_xlim(*boot_xlim)
        ax_boot.set_xlabel("Bootstrapped rhos", fontweight="bold")
        ax_boot.set_ylabel("Occurrence")
        ax_boot.spines["top"].set_visible(False)
        ax_boot.spines["right"].set_visible(False)
        ax_boot.xaxis.set_major_locator(MaxNLocator(5))

    _save_svg_png(fig, out_svg, save_png=save_png)
    plt.close(fig)


def plot_one_condition(
    stim_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    model_wise_fdr: dict[tuple[str, str], float],
) -> list[dict]:
    result_csv = stim_dir / str(args.result_file)
    null_npz = stim_dir / str(args.null_file)

    if not result_csv.exists():
        return [{
            "stimulus_type": stim_dir.name,
            "plot_status": "missing_result_csv",
            "input": (Path(stim_dir.name) / result_csv.name).as_posix(),
        }]

    if not null_npz.exists():
        return [{
            "stimulus_type": stim_dir.name,
            "plot_status": "missing_null_npz",
            "input": (Path(stim_dir.name) / null_npz.name).as_posix(),
        }]

    df = pd.read_csv(result_csv)

    null = load_perm_null_npz(
        null_npz,
        required=("obs_rho", "perm_rhos", "models"),
    )

    models = [str(x) for x in np.asarray(null["models"]).tolist()]
    obs = np.asarray(null["obs_rho"], dtype=float).reshape(-1)
    perm = np.asarray(null["perm_rhos"], dtype=float)

    if perm.ndim != 2:
        raise ValueError(f"Unexpected perm_rhos shape in {null_npz}: {perm.shape}")
    if perm.shape[0] != len(models) and perm.shape[1] == len(models):
        perm = perm.T
    if obs.size != len(models) or perm.shape[0] != len(models):
        raise ValueError(
            f"Model dimension mismatch in {null_npz}: models={len(models)}, "
            f"obs={obs.shape}, perm={perm.shape}"
        )

    if perm.size == 0 or not np.isfinite(perm).any():
        raise ValueError(
            f"{null_npz} does not contain valid permutation null values; "
            "rerun the analysis with --correction-mode perm_fwer_fdr."
        )

    model_to_i = {m: i for i, m in enumerate(models)}
    metadata = null.get("metadata", {})
    tail = str(metadata.get("tail", "")).strip().lower()
    if tail not in {"positive", "two_sided"} and "tail" in df.columns and not df.empty:
        tail = str(df.iloc[0]["tail"]).strip().lower()
    if tail not in {"positive", "two_sided"}:
        tail = "positive"
    p_values = _recompute_p_values(obs, perm, tail=tail)

    summary: list[dict] = []
    bootstrap_file = _ensure_bootstrap_file(stim_dir, args, result_df=df)

    for model in [str(m) for m in args.models]:
        if model not in model_to_i:
            summary.append({
                "stimulus_type": stim_dir.name,
                "model": model,
                "plot_status": "model_missing_in_null",
            })
            continue

        mi = model_to_i[model]
        obs_r = float(obs[mi])

        row = df[df["model"].astype(str) == model]
        if row.empty:
            raise ValueError(f"Result CSV {result_csv} has no row for model {model}")
        result_row = row.iloc[0]
        if "r_obs" not in result_row.index or pd.isna(result_row["r_obs"]):
            raise ValueError(f"Result CSV {result_csv} has no valid r_obs for model {model}")
        _assert_values_match(
            label="observed rho",
            model=model,
            left=obs_r,
            right=float(result_row["r_obs"]),
            left_source="NPZ obs_rho",
            right_source="CSV r_obs",
        )

        if str(args.p_col) == "p_fdr_bh_model_wise":
            key = (stim_dir.name, model)
            if key not in model_wise_fdr:
                raise ValueError(f"Cannot compute model-wise BH-FDR for condition/model {key}")
            p_val = float(model_wise_fdr[key])
        else:
            p_val = float(p_values[str(args.p_col)][mi])
        if str(args.p_col) != "p_fdr_bh_model_wise" and str(args.p_col) in result_row.index and pd.notna(result_row[str(args.p_col)]):
            _assert_values_match(
                label=str(args.p_col),
                model=model,
                left=p_val,
                right=float(result_row[str(args.p_col)]),
                left_source="recomputed NPZ null",
                right_source=f"CSV {args.p_col}",
                atol=1e-10,
            )

        if (not bool(args.include_negative)) and obs_r <= 0:
            summary.append({
                "stimulus_type": stim_dir.name,
                "model": model,
                "r_obs": obs_r,
                "plot_status": "skipped_negative_effect",
            })
            continue

        null_vals = np.asarray(perm[mi], dtype=float)
        null_vals = null_vals[np.isfinite(null_vals)]

        if null_vals.size == 0:
            summary.append({
                "stimulus_type": stim_dir.name,
                "model": model,
                "r_obs": obs_r,
                "plot_status": "empty_null",
            })
            continue

        boot_vals = _find_bootstrap_values(
            boot_file=bootstrap_file,
            null_npz=null_npz,
            model=model,
            models=models,
            bootstrap_key=args.bootstrap_key,
        )

        model_slug = _safe_name(model)
        out_svg = (
            out_dir
            / stim_dir.name
            / model_slug
            / f"{stim_dir.name}_{model_slug}_behavior_isc_dev_model_null_distribution.svg"
        )

        plot_single_model(
            stim_name=stim_dir.name,
            model=model,
            obs_r=obs_r,
            null_vals=null_vals,
            boot_vals=boot_vals,
            p_val=p_val,
            p_col=str(args.p_col),
            out_svg=out_svg,
            bins=int(args.bins),
            save_png=not bool(args.no_png),
        )

        summary.append({
            "stimulus_type": stim_dir.name,
            "model": model,
            "r_obs": obs_r,
            "p_display": p_val,
            "p_display_col": str(args.p_col),
            "p_perm": float(p_values["p_perm"][mi]),
            "p_fwer_global": float(p_values["p_fwer_global"][mi]),
            "p_fdr_bh_global": float(p_values["p_fdr_bh_global"][mi]),
            "p_fdr_bh_model_wise": float(model_wise_fdr[(stim_dir.name, model)]),
            "tail": tail,
            "n_null": int(null_vals.size),
            "n_bootstrap": int(boot_vals.size) if boot_vals is not None else 0,
            "bootstrap_file": (Path(stim_dir.name) / bootstrap_file.name).as_posix() if bootstrap_file is not None else "",
            "plot_status": "plotted_single_model",
            "output_svg": out_svg.relative_to(out_dir).as_posix(),
            "output_png": out_svg.with_suffix(".png").relative_to(out_dir).as_posix() if not bool(args.no_png) else "",
        })

    return summary


def main() -> None:
    args = parse_args()

    base = Path(args.matrix_dir) / str(args.stimulus_dir_name)
    if not base.exists():
        raise FileNotFoundError(f"Cannot find stimulus directory: {base}")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else Path(args.matrix_dir)
        / "figures"
        / str(args.stimulus_dir_name)
        / "behavior_null_histograms"
    )

    stim_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    model_wise_fdr = _compute_model_wise_fdr(
        stim_dirs,
        result_file=str(args.result_file),
        null_file=str(args.null_file),
    )

    rows = []
    for stim_dir in stim_dirs:
        rows.extend(plot_one_condition(stim_dir, out_dir, args, model_wise_fdr))

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(
        out_dir / f"behavior_isc_dev_model_null_distributions_summary_{args.stimulus_dir_name}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
