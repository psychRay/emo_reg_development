#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from perm_null_io import load_perm_null_npz  # noqa: E402
else:
    from .perm_null_io import load_perm_null_npz


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
MODEL_NAMES = ("M_nn", "M_conv", "M_div")
MODEL_LABELS = {
    "M_nn": "nearest_neighbor",
    "M_conv": "convergent",
    "M_div": "divergent",
}
OUT_PREFIX = "behavior_isc_dev_models_perm_fwer"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot behavior ISC vs age-model permutation and bootstrap rho distributions")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus", choices=("by_stimulus", "by_emotion"))
    p.add_argument("--behavior-isc-method", type=str, default="mahalanobis", choices=("spearman", "pearson", "euclidean", "mahalanobis"))
    p.add_argument("--behavior-isc-prefix", type=str, default=None)
    p.add_argument("--assoc-method", type=str, default="spearman", choices=("pearson", "spearman"))
    p.add_argument("--null-file", type=str, default="behavior_isc_dev_models_perm_null.npz", help="Existing analysis-stage permutation null NPZ inside each condition directory.")
    p.add_argument("--recompute-permutation", action="store_true", help="Ignore --null-file and recompute permutation nulls during plotting.")
    p.add_argument("--bootstrap-file", type=str, default="bootstrap_rhos.npz", help="Existing analysis-stage bootstrap NPZ inside each condition directory.")
    p.add_argument("--compute-bootstrap-if-missing", action="store_true", help="Compute bootstrap_rhos.npz during plotting only when it is missing.")
    p.add_argument("--recompute-bootstrap", action="store_true", help="Recompute bootstrap_rhos.npz during plotting even when it exists.")
    p.add_argument("--n-perm", type=int, default=None, help="Defaults to value stored in result CSV, otherwise 10000")
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=None, help="Permutation seed; defaults to value stored in result CSV, otherwise 42")
    p.add_argument("--bootstrap-seed", type=int, default=202405)
    p.add_argument("--fisher-z-behavior", type=str, default=None)
    p.add_argument("--no-normalize-models", action="store_false", dest="normalize_models", default=True)
    p.add_argument("--bins", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def rankdata(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    n = int(arr.size)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_vals = arr[order]
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = 0.5 * (float(start + 1) + float(end))
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if int(m.sum()) < 3:
        return out
    sd = float(x[m].std(ddof=0))
    if sd <= 0 or not np.isfinite(sd):
        return out
    out[m] = (x[m] - float(x[m].mean())) / sd
    return out


def fisher_z(r: np.ndarray) -> np.ndarray:
    x = np.asarray(r, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    m = np.isfinite(x)
    if bool(m.any()):
        out[m] = np.arctanh(np.clip(x[m], -0.999999, 0.999999))
    return out


def build_models(ages: np.ndarray, iu: np.ndarray, ju: np.ndarray, normalize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(ages, dtype=np.float64).reshape(-1)
    amax = float(np.nanmax(a))
    ai = a[iu]
    aj = a[ju]
    nn = amax - np.abs(ai - aj)
    conv = np.minimum(ai, aj)
    div = amax - 0.5 * (ai + aj)
    if bool(normalize):
        return zscore_1d(nn), zscore_1d(conv), zscore_1d(div)
    return nn.astype(np.float64), conv.astype(np.float64), div.astype(np.float64)


def _assoc_one(x: np.ndarray, m: np.ndarray, method: str) -> float:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    mv = np.asarray(m, dtype=np.float64).reshape(-1)
    ok = np.isfinite(xv) & np.isfinite(mv)
    if int(ok.sum()) < 3:
        return float("nan")
    if str(method).strip().lower() == "spearman":
        xz = zscore_1d(rankdata(xv[ok]))
        mz = zscore_1d(rankdata(mv[ok]))
    else:
        xz = zscore_1d(xv[ok])
        mz = zscore_1d(mv[ok])
    keep = np.isfinite(xz) & np.isfinite(mz)
    if int(keep.sum()) < 3:
        return float("nan")
    return float(np.mean(xz[keep] * mz[keep]))


def _policy_flag(method: str, override: Optional[str]) -> bool:
    if override is not None:
        return str(override).strip().lower() == "true"
    return str(method).strip().lower() in {"pearson", "spearman"}


def _resolve_prefix(prefix: Optional[str], method: str) -> str:
    return str(prefix) if prefix is not None else f"behavior_isc_{str(method).strip().lower()}_by_age"


def _load_result_row(stim_dir: Path, model: str) -> Optional[pd.Series]:
    csv_path = stim_dir / f"{OUT_PREFIX}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    sub = df[df["model"].astype(str) == str(model)]
    if sub.empty:
        return None
    return sub.iloc[0]


def _first_int(row: Optional[pd.Series], col: str, fallback: int) -> int:
    if row is not None and col in row.index and pd.notna(row[col]):
        return int(row[col])
    return int(fallback)


def _prepare_condition(
    stim_dir: Path,
    behavior_isc_prefix: str,
    behavior_isc_method: str,
    fisher_z_behavior: bool,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    behavior_isc = np.load(stim_dir / f"{behavior_isc_prefix}.npy")
    sub_df = pd.read_csv(stim_dir / f"{behavior_isc_prefix}_subjects_sorted.csv")
    subjects = sub_df["subject"].astype(str).tolist()
    ages = sub_df["age"].astype(float).to_numpy()
    n_sub = len(subjects)
    if behavior_isc.shape != (n_sub, n_sub):
        raise ValueError(f"{stim_dir.name}: behavior ISC shape {behavior_isc.shape} does not match n_subjects={n_sub}")
    iu, ju = np.triu_indices(n_sub, k=1)
    beh_vec = np.asarray(behavior_isc, dtype=np.float64)[iu, ju]
    if bool(fisher_z_behavior):
        beh_vec = fisher_z(beh_vec)
    return np.asarray(behavior_isc, dtype=np.float64), subjects, ages, iu, ju, beh_vec


def _permutation_rhos(
    beh_vec: np.ndarray,
    ages: np.ndarray,
    iu: np.ndarray,
    ju: np.ndarray,
    assoc_method: str,
    normalize_models: bool,
    n_perm: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    out = {m: np.full(int(n_perm), np.nan, dtype=np.float64) for m in MODEL_NAMES}
    n_sub = int(ages.size)
    for pi in range(int(n_perm)):
        ages_p = ages[rng.permutation(n_sub)]
        models = build_models(ages_p, iu, ju, normalize=bool(normalize_models))
        for mi, model in enumerate(MODEL_NAMES):
            out[model][pi] = _assoc_one(beh_vec, models[mi], method=str(assoc_method))
    return out


def _null_path_for_condition(stim_dir: Path, null_file: str | Path) -> Path:
    candidate = Path(null_file)
    return candidate if candidate.is_absolute() else stim_dir / candidate


def _load_perm_null_by_model(stim_dir: Path, null_file: str | Path) -> tuple[dict[str, float], dict[str, np.ndarray], Path]:
    null_path = _null_path_for_condition(stim_dir, null_file)
    null = load_perm_null_npz(null_path, required=("obs_rho", "perm_rhos", "models"))
    models = [str(x) for x in np.asarray(null["models"]).tolist()]
    obs = np.asarray(null["obs_rho"], dtype=np.float64)
    perm = np.asarray(null["perm_rhos"], dtype=np.float64)
    if perm.ndim != 2:
        raise ValueError(f"{null_path} field perm_rhos must have shape (n_models, n_perm), got {perm.shape}")
    if obs.shape[0] != len(models) or perm.shape[0] != len(models):
        raise ValueError(f"{null_path} model dimension mismatch: models={len(models)}, obs={obs.shape}, perm={perm.shape}")
    model_to_i = {m: i for i, m in enumerate(models)}
    missing = [m for m in MODEL_NAMES if m not in model_to_i]
    if missing:
        raise KeyError(f"{null_path} missing model(s): {missing}")
    obs_by_model = {m: float(obs[model_to_i[m]]) for m in MODEL_NAMES}
    null_by_model = {m: perm[model_to_i[m], :].astype(np.float64, copy=False) for m in MODEL_NAMES}
    return obs_by_model, null_by_model, null_path


def _bootstrap_path_for_condition(stim_dir: Path, bootstrap_file: str | Path) -> Path:
    candidate = Path(bootstrap_file)
    return candidate if candidate.is_absolute() else stim_dir / candidate


def _load_bootstrap_by_model(stim_dir: Path, bootstrap_file: str | Path) -> tuple[dict[str, np.ndarray], Path]:
    boot_path = _bootstrap_path_for_condition(stim_dir, bootstrap_file)
    if not boot_path.exists():
        raise FileNotFoundError(f"Cannot find bootstrap file: {boot_path}")
    with np.load(boot_path, allow_pickle=True) as z:
        if "models" in z.files:
            models = [str(x) for x in np.asarray(z["models"]).tolist()]
        else:
            models = list(MODEL_NAMES)
        if "bootstrap_rhos" in z.files:
            arr = np.asarray(z["bootstrap_rhos"], dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(f"{boot_path} field bootstrap_rhos must have shape (n_models, n_boot), got {arr.shape}")
            model_to_i = {m: i for i, m in enumerate(models)}
            missing = [m for m in MODEL_NAMES if m not in model_to_i]
            if missing:
                raise KeyError(f"{boot_path} missing model(s): {missing}")
            boot_by_model = {m: arr[model_to_i[m], :].astype(np.float64, copy=False) for m in MODEL_NAMES}
        else:
            boot_by_model = {}
            for model in MODEL_NAMES:
                key = f"{model}_bootstrap"
                if key not in z.files:
                    raise KeyError(f"{boot_path} missing bootstrap_rhos and {key}")
                boot_by_model[model] = np.asarray(z[key], dtype=np.float64)
    return boot_by_model, boot_path


def _bootstrap_rhos(
    behavior_isc: np.ndarray,
    ages: np.ndarray,
    assoc_method: str,
    normalize_models: bool,
    n_bootstrap: int,
    seed: int,
    fisher_z_behavior: bool,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    n_sub = int(ages.size)
    iu, ju = np.triu_indices(n_sub, k=1)
    out = {m: np.full(int(n_bootstrap), np.nan, dtype=np.float64) for m in MODEL_NAMES}
    for bi in range(int(n_bootstrap)):
        sample_idx = rng.integers(0, n_sub, size=n_sub)
        bmat = behavior_isc[np.ix_(sample_idx, sample_idx)]
        ages_b = ages[sample_idx]
        beh_vec = bmat[iu, ju].astype(np.float64, copy=False)
        same_subject = sample_idx[iu] == sample_idx[ju]
        beh_vec = beh_vec.copy()
        beh_vec[same_subject] = np.nan
        if bool(fisher_z_behavior):
            beh_vec = fisher_z(beh_vec)
        models = build_models(ages_b, iu, ju, normalize=bool(normalize_models))
        for mi, model in enumerate(MODEL_NAMES):
            mv = models[mi].copy()
            mv[same_subject] = np.nan
            out[model][bi] = _assoc_one(beh_vec, mv, method=str(assoc_method))
    return out


def _finite_limits(x: np.ndarray, pad_frac: float = 0.08) -> tuple[float, float]:
    xv = np.asarray(x, dtype=np.float64)
    xv = xv[np.isfinite(xv)]
    if int(xv.size) == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(xv, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(xv))
        hi = float(np.nanmax(xv))
    width = max(float(hi - lo), 1e-6)
    return float(lo - width * pad_frac), float(hi + width * pad_frac)


def _draw_break_marks(ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    kwargs = dict(marker=[(-1, -0.7), (1, 0.7)], markersize=8, linestyle="none", color="0.25", mec="0.25", mew=1.0, clip_on=False)
    ax_left.plot([1, 1], [0, 0], transform=ax_left.transAxes, **kwargs)
    ax_right.plot([0, 0], [0, 0], transform=ax_right.transAxes, **kwargs)


def _format_p(x: float) -> str:
    return f"{x:.4g}" if np.isfinite(x) else "nan"


def _result_stats(result_row: Optional[pd.Series], boot_f: np.ndarray) -> tuple[float, float, float, float, float]:
    p_perm = float(result_row["p_perm"]) if result_row is not None and "p_perm" in result_row.index and pd.notna(result_row["p_perm"]) else np.nan
    p_fwer = float(result_row["p_fwer_global"]) if result_row is not None and "p_fwer_global" in result_row.index and pd.notna(result_row["p_fwer_global"]) else np.nan
    p_fdr = float(result_row["p_fdr_bh_global"]) if result_row is not None and "p_fdr_bh_global" in result_row.index and pd.notna(result_row["p_fdr_bh_global"]) else np.nan
    ci_lo, ci_hi = (np.nan, np.nan)
    if int(boot_f.size) > 0:
        ci_lo, ci_hi = np.percentile(boot_f, [2.5, 97.5])
    return p_perm, p_fwer, p_fdr, float(ci_lo), float(ci_hi)


def _summary_row(
    stim_name: str,
    model: str,
    obs_rho: float,
    null_f: np.ndarray,
    boot_f: np.ndarray,
    p_perm: float,
    p_fwer: float,
    p_fdr: float,
    ci_lo: float,
    ci_hi: float,
    figure: Path,
) -> dict:
    return {
        "stimulus_type": str(stim_name),
        "model": str(model),
        "obs_rho": float(obs_rho),
        "p_perm": p_perm,
        "p_fwer_global": p_fwer,
        "p_fdr_bh_global": p_fdr,
        "null_mean": float(np.nanmean(null_f)) if int(null_f.size) else np.nan,
        "null_sd": float(np.nanstd(null_f, ddof=1)) if int(null_f.size) > 1 else np.nan,
        "bootstrap_mean": float(np.nanmean(boot_f)) if int(boot_f.size) else np.nan,
        "bootstrap_sd": float(np.nanstd(boot_f, ddof=1)) if int(boot_f.size) > 1 else np.nan,
        "bootstrap_ci_low": float(ci_lo),
        "bootstrap_ci_high": float(ci_hi),
        "figure": str(figure),
    }


def _svg_text(x: float, y: float, text: str, size: int = 13, anchor: str = "middle", weight: str = "normal") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="Arial, sans-serif" text-anchor="{anchor}" font-weight="{weight}">{html.escape(str(text))}</text>'


def _svg_line(x1: float, y1: float, x2: float, y2: float, color: str = "#222", width: float = 1.2, dash: bool = False) -> str:
    dash_attr = ' stroke-dasharray="5,4"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.1f}"{dash_attr}/>'


def _svg_hist(
    values: np.ndarray,
    x0: float,
    y0: float,
    w: float,
    h: float,
    bins: int,
    color: str,
    xlim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    title_lines: tuple[str, ...] = (),
    obs_rho: Optional[float] = None,
) -> str:
    vals = values[np.isfinite(values)]
    counts, edges = np.histogram(vals, bins=int(bins), range=xlim) if int(vals.size) else (np.zeros(int(bins)), np.linspace(xlim[0], xlim[1], int(bins) + 1))
    ymax = max(float(np.max(counts)) if int(counts.size) else 1.0, 1.0)

    def sx(v: float) -> float:
        return x0 + (float(v) - xlim[0]) / max(xlim[1] - xlim[0], 1e-12) * w

    def sy(v: float) -> float:
        return y0 + h - float(v) / ymax * h

    parts = []
    for i, c in enumerate(counts):
        bx0 = sx(float(edges[i]))
        bx1 = sx(float(edges[i + 1]))
        bw = max(0.4, bx1 - bx0 - 1.0)
        bh = y0 + h - sy(float(c))
        parts.append(f'<rect x="{bx0:.1f}" y="{sy(float(c)):.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}" opacity="0.95"/>')
    parts.append(_svg_line(x0, y0 + h, x0 + w, y0 + h))
    parts.append(_svg_line(x0, y0, x0, y0 + h))
    if obs_rho is not None and np.isfinite(obs_rho):
        ox = sx(float(obs_rho))
        if x0 <= ox <= x0 + w:
            parts.append(_svg_line(ox, y0, ox, y0 + h, color="#E53935", width=1.8, dash=True))

    for frac in (0.0, 0.5, 1.0):
        xv = xlim[0] + frac * (xlim[1] - xlim[0])
        xp = sx(xv)
        parts.append(_svg_line(xp, y0 + h, xp, y0 + h + 4, width=1.0))
        parts.append(_svg_text(xp, y0 + h + 18, f"{xv:.3f}", size=11))
    for frac in (0.0, 0.5, 1.0):
        yv = frac * ymax
        yp = sy(yv)
        parts.append(_svg_line(x0 - 4, yp, x0, yp, width=1.0))
        parts.append(_svg_text(x0 - 8, yp + 4, f"{yv:.0f}", size=11, anchor="end"))
    parts.append(_svg_text(x0 + w / 2, y0 + h + 38, xlabel, size=13, weight="bold" if "Boot" in xlabel else "normal"))
    parts.append(_svg_text(x0 - 42, y0 + h / 2, ylabel, size=13, anchor="middle"))
    for i, line in enumerate(title_lines):
        parts.append(_svg_text(x0 + w / 2, y0 - 28 + i * 18, line, size=14, weight="bold" if i == 0 else "normal"))
    return "\n".join(parts)


def _plot_one_svg(
    stim_name: str,
    model: str,
    obs_rho: float,
    null_f: np.ndarray,
    boot_f: np.ndarray,
    result_row: Optional[pd.Series],
    out_path: Path,
    bins: int,
) -> dict:
    p_perm, p_fwer, p_fdr, ci_lo, ci_hi = _result_stats(result_row, boot_f)
    label = MODEL_LABELS.get(str(model), str(model))
    out_svg = Path(out_path).with_suffix(".svg")
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    width, height = 980, 390
    y0, hist_h = 95, 205
    null_xlim = _finite_limits(null_f)
    obs_width = max(0.002, abs(float(obs_rho)) * 0.008)
    obs_xlim = (float(obs_rho) - obs_width, float(obs_rho) + obs_width)
    boot_xlim = _finite_limits(np.concatenate([boot_f, np.array([obs_rho], dtype=np.float64)]))
    title = f"{stim_name} | {model}   p_perm={_format_p(p_perm)}, p_FWER={_format_p(p_fwer)}, p_FDR={_format_p(p_fdr)}   bootstrap 95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 28, title, size=13, weight="bold"),
        _svg_hist(null_f, 65, y0, 315, hist_h, int(bins), "#5AA1C8", null_xlim, "rho", "density", (f"Permutation null ({label})", f"obs rho = {obs_rho:.3f}")),
        _svg_hist(np.array([], dtype=np.float64), 440, y0, 88, hist_h, 1, "#FFFFFF", obs_xlim, "rho", "", (), obs_rho=float(obs_rho)),
        _svg_hist(boot_f, 610, y0, 315, hist_h, int(bins), "#A6A6A6", boot_xlim, "Bootstrapped rhos", "density", (), obs_rho=float(obs_rho)),
        _svg_line(405, y0 + hist_h, 420, y0 + hist_h - 12, color="#333"),
        _svg_line(540, y0 + hist_h, 555, y0 + hist_h - 12, color="#333"),
        "</svg>",
    ]
    out_svg.write_text("\n".join(parts), encoding="utf-8")
    return _summary_row(stim_name, model, obs_rho, null_f, boot_f, p_perm, p_fwer, p_fdr, ci_lo, ci_hi, out_svg)


def plot_one(
    stim_name: str,
    model: str,
    obs_rho: float,
    null_rhos: np.ndarray,
    boot_rhos: np.ndarray,
    result_row: Optional[pd.Series],
    out_png: Path,
    bins: int,
    dpi: int,
) -> dict:
    null_f = null_rhos[np.isfinite(null_rhos)]
    boot_f = boot_rhos[np.isfinite(boot_rhos)]
    label = MODEL_LABELS.get(str(model), str(model))
    p_perm, p_fwer, p_fdr, ci_lo, ci_hi = _result_stats(result_row, boot_f)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib unavailable ({exc}); writing SVG fallback for {stim_name} {model}")
        return _plot_one_svg(
            stim_name=stim_name,
            model=model,
            obs_rho=float(obs_rho),
            null_f=null_f,
            boot_f=boot_f,
            result_row=result_row,
            out_path=Path(out_png),
            bins=int(bins),
        )

    fig = plt.figure(figsize=(8.2, 3.25), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.48, 1.12], wspace=0.06)
    ax_null = fig.add_subplot(gs[0, 0])
    ax_obs = fig.add_subplot(gs[0, 1])
    ax_boot = fig.add_subplot(gs[0, 2])

    ax_null.hist(null_f, bins=int(bins), color="#5AA1C8", edgecolor="white", linewidth=0.35)
    ax_null.set_title(f"Permutation null ({label})\nobs rho = {obs_rho:.3f}", fontsize=11, pad=8)
    ax_null.set_xlabel("rho")
    ax_null.set_ylabel("density")
    ax_null.set_xlim(*_finite_limits(null_f))

    ax_obs.axvline(float(obs_rho), color="#E53935", linestyle="--", linewidth=1.6)
    ax_obs.set_xlabel("rho")
    ax_obs.set_yticks([])
    obs_width = max(0.002, abs(float(obs_rho)) * 0.008)
    ax_obs.set_xlim(float(obs_rho) - obs_width, float(obs_rho) + obs_width)
    for side in ("left", "right", "top"):
        ax_obs.spines[side].set_visible(False)
    _draw_break_marks(ax_null, ax_obs)

    ax_boot.hist(boot_f, bins=int(bins), color="#A6A6A6", edgecolor="white", linewidth=0.35)
    ax_boot.axvline(float(obs_rho), color="#E53935", linestyle="--", linewidth=1.6)
    ax_boot.set_xlabel("Bootstrapped rhos", fontweight="bold")
    ax_boot.set_ylabel("density")
    boot_lo, boot_hi = _finite_limits(np.concatenate([boot_f, np.array([obs_rho], dtype=np.float64)]))
    ax_boot.set_xlim(boot_lo, boot_hi)

    for ax in (ax_null, ax_obs, ax_boot):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)

    fig.suptitle(f"{stim_name} | {model}   p_perm={p_perm:.4g}, p_FWER={p_fwer:.4g}, p_FDR={p_fdr:.4g}   bootstrap 95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]", fontsize=10, y=1.03)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return _summary_row(stim_name, model, obs_rho, null_f, boot_f, p_perm, p_fwer, p_fdr, ci_lo, ci_hi, Path(out_png))


def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
    behavior_isc_method: str,
    behavior_isc_prefix: Optional[str],
    assoc_method: str,
    null_file: str,
    recompute_permutation: bool,
    bootstrap_file: str,
    compute_bootstrap_if_missing: bool,
    recompute_bootstrap: bool,
    n_perm: Optional[int],
    n_bootstrap: int,
    seed: Optional[int],
    bootstrap_seed: int,
    fisher_z_behavior: Optional[str],
    normalize_models: bool,
    bins: int,
    dpi: int,
    out_dir: Optional[Path],
) -> None:
    matrix_dir = Path(matrix_dir)
    by_stim = matrix_dir / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"Cannot find directory: {by_stim}")

    prefix = _resolve_prefix(behavior_isc_prefix, method=str(behavior_isc_method))
    fisher_z_eff = _policy_flag(method=str(behavior_isc_method), override=fisher_z_behavior)
    fig_root = Path(out_dir) if out_dir is not None else matrix_dir / "figures" / str(stimulus_dir_name) / "behavior_isc_dev_models_distributions"
    summary_rows = []

    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        if not (stim_dir / f"{prefix}.npy").exists() or not (stim_dir / f"{prefix}_subjects_sorted.csv").exists():
            print(f"[SKIP] {stim_dir.name}: missing behavior ISC files for {prefix}")
            continue

        behavior_isc, _, ages, iu, ju, beh_vec = _prepare_condition(
            stim_dir=stim_dir,
            behavior_isc_prefix=str(prefix),
            behavior_isc_method=str(behavior_isc_method),
            fisher_z_behavior=bool(fisher_z_eff),
        )
        models = build_models(ages, iu, ju, normalize=bool(normalize_models))
        r_obs = {model: _assoc_one(beh_vec, models[mi], method=str(assoc_method)) for mi, model in enumerate(MODEL_NAMES)}
        first_row = _load_result_row(stim_dir, MODEL_NAMES[0])
        n_perm_eff = int(n_perm) if n_perm is not None else _first_int(first_row, "n_perm", 10000)
        seed_eff = int(seed) if seed is not None else _first_int(first_row, "seed", 42)

        if bool(recompute_permutation):
            print(f"[plot] {stim_dir.name}: recomputing permutation null (n={n_perm_eff}) and bootstrap rhos (n={int(n_bootstrap)})")
            null_by_model = _permutation_rhos(
                beh_vec=beh_vec,
                ages=ages,
                iu=iu,
                ju=ju,
                assoc_method=str(assoc_method),
                normalize_models=bool(normalize_models),
                n_perm=int(n_perm_eff),
                seed=int(seed_eff),
            )
            source_perm_null = "recomputed"
        else:
            try:
                r_obs_loaded, null_by_model, loaded_null_path = _load_perm_null_by_model(stim_dir, null_file=str(null_file))
                r_obs = r_obs_loaded
                n_perm_eff = int(next(iter(null_by_model.values())).size)
                print(f"[plot] {stim_dir.name}: loaded permutation null from {loaded_null_path} (n={n_perm_eff}); bootstrap rhos n={int(n_bootstrap)}")
                source_perm_null = str(loaded_null_path)
            except FileNotFoundError:
                print(f"[plot] {stim_dir.name}: missing {str(null_file)}; recomputing permutation null (n={n_perm_eff})")
                null_by_model = _permutation_rhos(
                    beh_vec=beh_vec,
                    ages=ages,
                    iu=iu,
                    ju=ju,
                    assoc_method=str(assoc_method),
                    normalize_models=bool(normalize_models),
                    n_perm=int(n_perm_eff),
                    seed=int(seed_eff),
                )
                source_perm_null = f"recomputed_missing:{_null_path_for_condition(stim_dir, null_file)}"
        n_bootstrap_eff = int(n_bootstrap)
        if bool(recompute_bootstrap):
            print(f"[plot] {stim_dir.name}: recomputing bootstrap rhos (n={n_bootstrap_eff})")
            boot_by_model = _bootstrap_rhos(
                behavior_isc=behavior_isc,
                ages=ages,
                assoc_method=str(assoc_method),
                normalize_models=bool(normalize_models),
                n_bootstrap=int(n_bootstrap_eff),
                seed=int(bootstrap_seed),
                fisher_z_behavior=bool(fisher_z_eff),
            )
            source_bootstrap = "recomputed"
        else:
            try:
                boot_by_model, loaded_boot_path = _load_bootstrap_by_model(stim_dir, bootstrap_file=str(bootstrap_file))
                n_bootstrap_eff = int(next(iter(boot_by_model.values())).size)
                print(f"[plot] {stim_dir.name}: loaded bootstrap rhos from {loaded_boot_path} (n={n_bootstrap_eff})")
                source_bootstrap = str(loaded_boot_path)
            except FileNotFoundError:
                if bool(compute_bootstrap_if_missing):
                    print(f"[plot] {stim_dir.name}: missing {str(bootstrap_file)}; computing bootstrap rhos (n={n_bootstrap_eff})")
                    boot_by_model = _bootstrap_rhos(
                        behavior_isc=behavior_isc,
                        ages=ages,
                        assoc_method=str(assoc_method),
                        normalize_models=bool(normalize_models),
                        n_bootstrap=int(n_bootstrap_eff),
                        seed=int(bootstrap_seed),
                        fisher_z_behavior=bool(fisher_z_eff),
                    )
                    source_bootstrap = f"recomputed_missing:{_bootstrap_path_for_condition(stim_dir, bootstrap_file)}"
                else:
                    print(f"[plot] {stim_dir.name}: missing {_bootstrap_path_for_condition(stim_dir, bootstrap_file)}; bootstrap panel will be empty")
                    boot_by_model = {m: np.asarray([], dtype=np.float64) for m in MODEL_NAMES}
                    n_bootstrap_eff = 0
                    source_bootstrap = f"missing:{_bootstrap_path_for_condition(stim_dir, bootstrap_file)}"
        np.savez_compressed(
            stim_dir / f"{OUT_PREFIX}_plot_distributions.npz",
            models=np.array(MODEL_NAMES, dtype=str),
            M_nn_null=null_by_model["M_nn"],
            M_conv_null=null_by_model["M_conv"],
            M_div_null=null_by_model["M_div"],
            M_nn_bootstrap=boot_by_model["M_nn"],
            M_conv_bootstrap=boot_by_model["M_conv"],
            M_div_bootstrap=boot_by_model["M_div"],
            seed=np.array([int(seed_eff)], dtype=np.int32),
            bootstrap_seed=np.array([int(bootstrap_seed)], dtype=np.int32),
            n_perm=np.array([int(n_perm_eff)], dtype=np.int32),
            n_bootstrap=np.array([int(n_bootstrap_eff)], dtype=np.int32),
            source_perm_null=np.array([source_perm_null], dtype=str),
            source_bootstrap=np.array([source_bootstrap], dtype=str),
        )

        for model in MODEL_NAMES:
            result_row = _load_result_row(stim_dir, model)
            out_png = fig_root / stim_dir.name / f"behavior_isc_dev_models_{stim_dir.name}_{model}.png"
            summary_rows.append(
                plot_one(
                    stim_name=stim_dir.name,
                    model=model,
                    obs_rho=float(r_obs[model]),
                    null_rhos=null_by_model[model],
                    boot_rhos=boot_by_model[model],
                    result_row=result_row,
                    out_png=out_png,
                    bins=int(bins),
                    dpi=int(dpi),
                )
            )

    if summary_rows:
        out_csv = fig_root / f"behavior_isc_dev_models_distribution_summary_{stimulus_dir_name}.csv"
        pd.DataFrame(summary_rows).sort_values(["stimulus_type", "model"]).to_csv(out_csv, index=False)
        print(f"[plot] saved summary: {out_csv}")


def main() -> None:
    args = parse_args()
    run(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        behavior_isc_method=str(args.behavior_isc_method),
        behavior_isc_prefix=args.behavior_isc_prefix,
        assoc_method=str(args.assoc_method),
        null_file=str(args.null_file),
        recompute_permutation=bool(args.recompute_permutation),
        bootstrap_file=str(args.bootstrap_file),
        compute_bootstrap_if_missing=bool(args.compute_bootstrap_if_missing),
        recompute_bootstrap=bool(args.recompute_bootstrap),
        n_perm=args.n_perm,
        n_bootstrap=int(args.n_bootstrap),
        seed=args.seed,
        bootstrap_seed=int(args.bootstrap_seed),
        fisher_z_behavior=args.fisher_z_behavior,
        normalize_models=bool(args.normalize_models),
        bins=int(args.bins),
        dpi=int(args.dpi),
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
