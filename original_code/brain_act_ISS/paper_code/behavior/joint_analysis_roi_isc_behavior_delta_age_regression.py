#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

if __package__ in {None, "", "behavior"}:
    THIS_DIR = Path(__file__).resolve().parent
    EMO_DIR = THIS_DIR.parent
    if str(EMO_DIR) not in sys.path:
        sys.path.insert(0, str(EMO_DIR))
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))

    from joint_analysis_roi_isc_behavior_age_regression import (  # noqa: E402
        bh_fdr,
        build_models,
        fisher_z,
        rankdata,
        zscore_1d,
        _fit_roi_model,
        _ols_fit_1d,
        _policy_flag,
        _resolve_prefix,
    )
    from perm_null_io import make_metadata, save_perm_null_npz  # noqa: E402
else:
    from .joint_analysis_roi_isc_behavior_age_regression import (  # noqa: E402
        bh_fdr,
        build_models,
        fisher_z,
        rankdata,
        zscore_1d,
        _fit_roi_model,
        _ols_fit_1d,
        _policy_flag,
        _resolve_prefix,
    )
    from ..perm_null_io import make_metadata, save_perm_null_npz  # noqa: E402


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
DEFAULT_SCHAEFER_MAPPING_NAME = "schaefer2018_200parcels_7networks_order_mapping.csv"
DEFAULT_ROIS = (
    "L_24",
    "L_26",
    "L_27",
    "L_3",
    "L_4",
    "R_103",
    "R_124",
    "R_125",
    "R_127",
    "R_129",
    "R_148",
    "V_5",
    "V_8",
)
MODEL_NAMES = ("M_nn", "M_conv", "M_div")
OUT_PREFIX = "roi_isc_behavior_delta_age_regression"
JOINT_EFFECTS = (
    "brain_delta",
    "age_M_nn",
    "age_M_conv",
    "age_M_div",
    "interaction_M_nn",
    "interaction_M_conv",
    "interaction_M_div",
)
JOINT_EFFECT_TO_T_INDEX = {effect: i + 1 for i, effect in enumerate(JOINT_EFFECTS)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Delta RSA regression: (Behavior_ISC_Reappraisal - Behavior_ISC_Passive_Emo) "
            "~ (Brain_ROI_ISC_Reappraisal - Brain_ROI_ISC_Passive_Emo) + Age_Model + interaction"
        )
    )
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument(
        "--brain-stimulus-dir-name",
        type=str,
        default=None,
        help="Directory name for brain ISC inputs. Default uses --stimulus-dir-name.",
    )
    p.add_argument(
        "--behavior-stimulus-dir-name",
        type=str,
        default=None,
        help="Directory name for behavior ISC inputs. Default uses --stimulus-dir-name.",
    )
    p.add_argument("--reappraisal-name", type=str, default="Reappraisal")
    p.add_argument("--passive-name", type=str, default="Passive_Emo")
    p.add_argument(
        "--rois",
        nargs="+",
        default=None,
        help="ROI names to analyze. If omitted, all ROIs in the brain ROI file are analyzed.",
    )
    p.add_argument(
        "--roi-selection-dir-name",
        type=str,
        default=None,
        help=(
            "If set, select ROIs from <matrix-dir>/<dir>/<condition>/roi_isc_dev_models_perm_fwer.csv "
            "before delta regression, typically by_emotion."
        ),
    )
    p.add_argument(
        "--roi-selection-conditions",
        nargs="+",
        default=None,
        help="Conditions used for ROI selection. Default: reappraisal and passive condition names.",
    )
    p.add_argument("--roi-selection-file", type=str, default="roi_isc_dev_models_perm_fwer.csv")
    p.add_argument("--roi-selection-model", nargs="+", default=["M_conv"])
    p.add_argument("--roi-selection-p-col", type=str, default="p_fdr_bh_model_wise")
    p.add_argument("--roi-selection-alpha", type=float, default=0.05)
    p.add_argument("--roi-selection-mode", type=str, default="union", choices=("union", "intersection"))
    p.add_argument("--brain-isc-method", type=str, default="mahalanobis")
    p.add_argument("--brain-isc-prefix", type=str, default=None)
    p.add_argument("--behavior-isc-method", type=str, default="mahalanobis")
    p.add_argument("--behavior-isc-prefix", type=str, default=None)
    p.add_argument("--age-model-mode", type=str, default="separate", choices=("separate", "joint"))
    p.add_argument("--fisher-z-brain", type=str, default=None)
    p.add_argument("--fisher-z-behavior", type=str, default=None)
    p.add_argument("--no-normalize-models", action="store_false", dest="normalize_models", default=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--rank-transform", action="store_true", dest="rank_transform")
    g.add_argument("--no-rank-transform", action="store_false", dest="rank_transform")
    p.set_defaults(rank_transform=True)
    p.add_argument("--tail", type=str, default="two_sided", choices=("two_sided", "positive"))
    p.add_argument("--correction-mode", type=str, default="perm_fwer_fdr", choices=("fdr_only", "perm_fwer_fdr"))
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1, help="Number of ROI workers. FDR/FWER are still merged and written by the main process.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--schaefer-mapping-file",
        type=Path,
        default=None,
        help=(
            "Optional Schaefer mapping CSV. Default: <matrix-dir>/"
            f"{DEFAULT_SCHAEFER_MAPPING_NAME}. The source_name column is merged into the output by ROI."
        ),
    )
    return p.parse_args()


def _read_subjects(path: Path) -> list[str]:
    return pd.read_csv(path)["subject"].astype(str).tolist()


def _read_age_map(paths: list[Path]) -> dict[str, float]:
    age_map: dict[str, float] = {}
    for path in paths:
        df = pd.read_csv(path)
        if "age" not in df.columns:
            continue
        for s, a in zip(df["subject"].astype(str), pd.to_numeric(df["age"], errors="coerce")):
            if np.isfinite(a) and float(a) < 998.0:
                age_map[str(s)] = float(a)
    return age_map


def _check_square(name: str, arr: np.ndarray, n: int) -> None:
    if arr.ndim != 2 or arr.shape != (n, n):
        raise ValueError(f"{name} shape {arr.shape} does not match n_subjects={n}; expected ({n}, {n})")


def _check_brain(name: str, arr: np.ndarray, n_rois: int, n_sub: int) -> None:
    if arr.ndim != 3 or arr.shape[0] != n_rois or arr.shape[1:] != (n_sub, n_sub):
        raise ValueError(f"{name} shape {arr.shape} does not match expected ({n_rois}, {n_sub}, {n_sub})")


def _align_delta_inputs(
    brain_reapp_dir: Path,
    brain_passive_dir: Path,
    behavior_reapp_dir: Path,
    behavior_passive_dir: Path,
    brain_prefix: str,
    behavior_prefix: str,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[str]]:
    brain_reapp = np.load(brain_reapp_dir / f"{brain_prefix}.npy")
    brain_passive = np.load(brain_passive_dir / f"{brain_prefix}.npy")
    beh_reapp = np.load(behavior_reapp_dir / f"{behavior_prefix}.npy")
    beh_passive = np.load(behavior_passive_dir / f"{behavior_prefix}.npy")

    brain_rois_reapp = pd.read_csv(brain_reapp_dir / f"{brain_prefix}_rois.csv")["roi"].astype(str).tolist()
    brain_rois_passive = pd.read_csv(brain_passive_dir / f"{brain_prefix}_rois.csv")["roi"].astype(str).tolist()
    if brain_rois_reapp != brain_rois_passive:
        raise ValueError("Reappraisal and Passive_Emo brain ROI order differ")
    rois = brain_rois_reapp

    br_sub_reapp_path = brain_reapp_dir / f"{brain_prefix}_subjects_sorted.csv"
    br_sub_passive_path = brain_passive_dir / f"{brain_prefix}_subjects_sorted.csv"
    beh_sub_reapp_path = behavior_reapp_dir / f"{behavior_prefix}_subjects_sorted.csv"
    beh_sub_passive_path = behavior_passive_dir / f"{behavior_prefix}_subjects_sorted.csv"

    br_sub_reapp = _read_subjects(br_sub_reapp_path)
    br_sub_passive = _read_subjects(br_sub_passive_path)
    beh_sub_reapp = _read_subjects(beh_sub_reapp_path)
    beh_sub_passive = _read_subjects(beh_sub_passive_path)
    for label, subs in {
        "brain Reappraisal": br_sub_reapp,
        "brain Passive_Emo": br_sub_passive,
        "behavior Reappraisal": beh_sub_reapp,
        "behavior Passive_Emo": beh_sub_passive,
    }.items():
        if len(set(subs)) != len(subs):
            raise ValueError(f"Duplicate subjects in {label} subject file")

    _check_brain("brain Reappraisal", brain_reapp, len(rois), len(br_sub_reapp))
    _check_brain("brain Passive_Emo", brain_passive, len(rois), len(br_sub_passive))
    _check_square("behavior Reappraisal", beh_reapp, len(beh_sub_reapp))
    _check_square("behavior Passive_Emo", beh_passive, len(beh_sub_passive))

    age_map = _read_age_map([br_sub_reapp_path, br_sub_passive_path, beh_sub_reapp_path, beh_sub_passive_path])
    overlap_set = set(br_sub_passive) & set(beh_sub_reapp) & set(beh_sub_passive)
    subjects = [s for s in br_sub_reapp if s in overlap_set and np.isfinite(age_map.get(s, np.nan))]
    if len(subjects) < 6:
        raise ValueError(f"Fewer than 6 common subjects with valid age across all four matrices: n={len(subjects)}")

    idx_br_reapp = {s: i for i, s in enumerate(br_sub_reapp)}
    idx_br_passive = {s: i for i, s in enumerate(br_sub_passive)}
    idx_beh_reapp = {s: i for i, s in enumerate(beh_sub_reapp)}
    idx_beh_passive = {s: i for i, s in enumerate(beh_sub_passive)}
    ib_reapp = [idx_br_reapp[s] for s in subjects]
    ib_passive = [idx_br_passive[s] for s in subjects]
    ih_reapp = [idx_beh_reapp[s] for s in subjects]
    ih_passive = [idx_beh_passive[s] for s in subjects]

    brain_delta = (
        np.asarray(brain_reapp, dtype=np.float32)[:, ib_reapp, :][:, :, ib_reapp]
        - np.asarray(brain_passive, dtype=np.float32)[:, ib_passive, :][:, :, ib_passive]
    )
    behavior_delta = (
        np.asarray(beh_reapp, dtype=np.float32)[np.ix_(ih_reapp, ih_reapp)]
        - np.asarray(beh_passive, dtype=np.float32)[np.ix_(ih_passive, ih_passive)]
    )
    ages = np.asarray([age_map[s] for s in subjects], dtype=np.float32)
    return brain_delta, behavior_delta, subjects, ages, rois


def _prepare_vector(v: np.ndarray, rank_transform: bool) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    if bool(rank_transform):
        mask = np.isfinite(x)
        r = np.full_like(x, np.nan, dtype=np.float32)
        if int(mask.sum()) > 0:
            r[mask] = rankdata(x[mask]).astype(np.float32, copy=False)
        return zscore_1d(r)
    return zscore_1d(x)


def _roi_aliases(roi: str) -> set[str]:
    r = str(roi)
    aliases = {r}
    if "_" not in r:
        return aliases
    hemi, num = r.split("_", 1)
    if hemi not in {"L", "R"}:
        return aliases
    try:
        idx = int(num)
    except ValueError:
        return aliases
    if hemi == "R":
        aliases.add(f"R_{idx - 100}" if idx > 100 else f"R_{idx + 100}")
    return aliases


def _resolve_roi_list(requested_rois: list[str], all_rois: list[str]) -> list[str]:
    lookup: dict[str, str] = {}
    for roi in all_rois:
        for alias in _roi_aliases(str(roi)):
            lookup.setdefault(alias, str(roi))

    resolved: list[str] = []
    missing: list[str] = []
    for roi in requested_rois:
        hit = None
        for alias in _roi_aliases(str(roi)):
            if alias in lookup:
                hit = lookup[alias]
                break
        if hit is None:
            missing.append(str(roi))
        elif hit not in resolved:
            resolved.append(hit)
    if missing:
        raise ValueError(f"Requested ROIs missing from brain ROI file after alias matching: {missing}")
    return resolved


def _select_rois_from_dev_models(
    matrix_dir: Path,
    roi_selection_dir_name: str,
    conditions: list[str],
    roi_selection_file: str,
    roi_selection_model: list[str] | str,
    roi_selection_p_col: str,
    roi_selection_alpha: float,
    roi_selection_mode: str,
) -> list[str]:
    def _clean_model_name(x: str) -> str:
        return str(x).strip().strip(",[]").strip("'\"").strip()

    raw_model_parts = [roi_selection_model] if isinstance(roi_selection_model, str) else list(roi_selection_model)
    roi_selection_models = []
    for part in raw_model_parts:
        for token in str(part).replace(",", " ").split():
            roi_selection_models.append(_clean_model_name(token))
    roi_selection_models = [x for x in roi_selection_models if x]
    if not roi_selection_models:
        raise ValueError("At least one --roi-selection-model is required")

    roi_sets: list[set[str]] = []
    for condition in conditions:
        path = Path(matrix_dir) / str(roi_selection_dir_name) / str(condition) / str(roi_selection_file)
        if not path.exists():
            raise FileNotFoundError(f"ROI selection file not found: {path}")
        df = pd.read_csv(path)
        required = {"roi", "model", str(roi_selection_p_col)}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"ROI selection file {path} misses required columns: {missing}")
        present_models = set(df["model"].astype(str).unique().tolist())
        missing_models = sorted(set(roi_selection_models).difference(present_models))
        if missing_models:
            raise ValueError(f"ROI selection file {path} has no rows for model(s): {missing_models}")
        sub = df[df["model"].astype(str).isin(roi_selection_models)].copy()
        if sub.empty:
            raise ValueError(f"No rows with model={roi_selection_models} in ROI selection file: {path}")
        pvals = pd.to_numeric(sub[str(roi_selection_p_col)], errors="coerce")
        selected = set(sub.loc[np.isfinite(pvals) & (pvals <= float(roi_selection_alpha)), "roi"].astype(str).tolist())
        roi_sets.append(selected)
        print(
            "[roi-selection] "
            f"{condition}: {len(selected)} ROIs with model={','.join(roi_selection_models)}, "
            f"{roi_selection_p_col}<={roi_selection_alpha}"
        )

    mode = str(roi_selection_mode).strip().lower()
    if not roi_sets:
        return []
    if mode == "intersection":
        selected_rois = set.intersection(*roi_sets)
    else:
        selected_rois = set.union(*roi_sets)
    out = sorted(selected_rois)
    if not out:
        raise ValueError(
            "ROI selection produced zero ROIs: "
            f"dir={roi_selection_dir_name}, conditions={conditions}, model={roi_selection_models}, "
            f"p_col={roi_selection_p_col}, alpha={roi_selection_alpha}, mode={roi_selection_mode}"
        )
    print(f"[roi-selection] {mode}: selected {len(out)} unique ROIs")
    return out


def _load_schaefer_mapping(mapping_file: Optional[Path], matrix_dir: Path) -> dict[str, dict]:
    path = Path(mapping_file) if mapping_file is not None else Path(matrix_dir) / DEFAULT_SCHAEFER_MAPPING_NAME
    if not path.exists():
        print(f"[mapping] Schaefer mapping file not found, output will omit network labels: {path}")
        return {}
    df = pd.read_csv(path)
    if "roi" not in df.columns:
        raise ValueError(f"{path} must contain a 'roi' column")
    wanted = ["roi", "roi_schaefer_style", "source_name"]
    keep = [c for c in wanted if c in df.columns]
    if "source_name" not in keep:
        raise ValueError(f"{path} must contain a 'source_name' column")
    out: dict[str, dict] = {}
    for _, row in df[keep].iterrows():
        record = {"source_name": row["source_name"]}
        roi = str(row["roi"])
        for alias in _roi_aliases(roi):
            out.setdefault(alias, record)
        if "roi_schaefer_style" in row.index and pd.notna(row["roi_schaefer_style"]):
            for alias in _roi_aliases(str(row["roi_schaefer_style"])):
                out.setdefault(alias, record)
    print(f"[mapping] Loaded Schaefer mapping: {path}")
    return out


def _add_schaefer_mapping_columns(out_df: pd.DataFrame, mapping: dict[str, dict]) -> pd.DataFrame:
    out = out_df.copy()
    if not mapping:
        return out
    records = []
    missing = []
    for roi in out["roi"].astype(str).tolist():
        rec = None
        for alias in _roi_aliases(roi):
            if alias in mapping:
                rec = mapping[alias]
                break
        if rec is None:
            missing.append(roi)
            rec = {}
        records.append(rec)
    map_df = pd.DataFrame(records)
    if not map_df.empty:
        out = pd.concat([out.reset_index(drop=True), map_df.reset_index(drop=True)], axis=1)
    if missing:
        miss_show = sorted(set(missing))[:10]
        print(f"[mapping] Warning: {len(set(missing))} ROI labels not found in Schaefer mapping: {miss_show}")
    return out


def _perm_stats_for_roi(
    brain_vec: np.ndarray,
    age_vec: np.ndarray,
    beh_vec: np.ndarray,
    pair_pos: np.ndarray,
    iu: np.ndarray,
    ju: np.ndarray,
    n_sub: int,
    n_perm: int,
    seed: int,
    model_index: int,
    tail: str,
    normalize_interaction: bool,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    beta_obs, t_obs, _, _, n_obs, X_obs, base_mask = _fit_roi_model(
        brain_vec=brain_vec,
        age_vec=age_vec,
        beh_vec=beh_vec,
        normalize_interaction=bool(normalize_interaction),
    )
    max_brain = np.full(int(n_perm), -np.inf, dtype=np.float64)
    max_age = np.full(int(n_perm), -np.inf, dtype=np.float64)
    max_interaction = np.full(int(n_perm), -np.inf, dtype=np.float64)
    if int(n_obs) <= 5:
        return np.nan, np.nan, np.nan, max_brain, max_age, max_interaction

    t_obs_b = float(t_obs[1])
    t_obs_a = float(t_obs[2])
    t_obs_i = float(t_obs[3])
    c_b = c_a = c_i = 0
    rng = np.random.default_rng(int(seed) + int(model_index) * 100003)
    tail_mode = str(tail).strip().lower()
    for pi in range(int(n_perm)):
        perm = rng.permutation(int(n_sub))
        idx_perm_full = pair_pos[perm[iu], perm[ju]]
        idx_perm = idx_perm_full[base_mask]
        y_perm = beh_vec[idx_perm].astype(np.float64, copy=False)
        keep = np.isfinite(y_perm) & np.isfinite(X_obs).all(axis=1)
        if int(keep.sum()) <= int(X_obs.shape[1]):
            tp_b = tp_a = tp_i = np.nan
        else:
            _, t_perm, _, _ = _ols_fit_1d(X_obs[keep], y_perm[keep])
            tp_b = float(t_perm[1])
            tp_a = float(t_perm[2])
            tp_i = float(t_perm[3])

        if tail_mode == "two_sided":
            if np.isfinite(tp_b) and np.isfinite(t_obs_b):
                c_b += int(abs(tp_b) >= abs(t_obs_b))
                max_brain[pi] = max(max_brain[pi], abs(tp_b))
            if np.isfinite(tp_a) and np.isfinite(t_obs_a):
                c_a += int(abs(tp_a) >= abs(t_obs_a))
                max_age[pi] = max(max_age[pi], abs(tp_a))
            if np.isfinite(tp_i) and np.isfinite(t_obs_i):
                c_i += int(abs(tp_i) >= abs(t_obs_i))
                max_interaction[pi] = max(max_interaction[pi], abs(tp_i))
        else:
            if np.isfinite(tp_b) and np.isfinite(t_obs_b):
                c_b += int(tp_b >= t_obs_b)
                max_brain[pi] = max(max_brain[pi], tp_b)
            if np.isfinite(tp_a) and np.isfinite(t_obs_a):
                c_a += int(tp_a >= t_obs_a)
                max_age[pi] = max(max_age[pi], tp_a)
            if np.isfinite(tp_i) and np.isfinite(t_obs_i):
                c_i += int(tp_i >= t_obs_i)
                max_interaction[pi] = max(max_interaction[pi], tp_i)

    denom = float(int(n_perm) + 1)
    p_b = (float(c_b) + 1.0) / denom if np.isfinite(t_obs_b) else np.nan
    p_a = (float(c_a) + 1.0) / denom if np.isfinite(t_obs_a) else np.nan
    p_i = (float(c_i) + 1.0) / denom if np.isfinite(t_obs_i) else np.nan
    return p_b, p_a, p_i, max_brain, max_age, max_interaction


def _build_joint_roi_design(
    brain_vec: np.ndarray,
    age_mat: np.ndarray,
    beh_vec: np.ndarray,
    normalize_interaction: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    brain = np.asarray(brain_vec, dtype=np.float64).reshape(-1)
    ages = np.asarray(age_mat, dtype=np.float64)
    beh = np.asarray(beh_vec, dtype=np.float64).reshape(-1)
    if ages.ndim != 2 or ages.shape[0] != len(MODEL_NAMES):
        raise ValueError("age_mat must be 2D with rows M_nn/M_conv/M_div")
    if ages.shape[1] != brain.shape[0] or beh.shape[0] != brain.shape[0]:
        raise ValueError("brain_vec, age_mat, and beh_vec have incompatible shapes")
    base_mask = np.isfinite(brain) & np.isfinite(beh) & np.isfinite(ages).all(axis=0)
    if int(base_mask.sum()) == 0:
        return np.empty((0, 8), dtype=np.float64), np.empty(0, dtype=np.float64), base_mask

    base_idx = np.where(base_mask)[0]
    b = brain[base_mask]
    a = ages[:, base_mask]
    y = beh[base_mask]
    inter = a * b.reshape(1, -1)
    if bool(normalize_interaction):
        inter = np.stack([zscore_1d(row).astype(np.float64, copy=False) for row in inter], axis=0)

    X = np.column_stack(
        [
            np.ones_like(y, dtype=np.float64),
            b,
            a[0],
            a[1],
            a[2],
            inter[0],
            inter[1],
            inter[2],
        ]
    )
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    final_mask = np.zeros_like(base_mask, dtype=bool)
    final_mask[base_idx[keep]] = True
    return X[keep], y[keep], final_mask


def _fit_joint_roi_model(
    brain_vec: np.ndarray,
    age_mat: np.ndarray,
    beh_vec: np.ndarray,
    normalize_interaction: bool,
) -> tuple[np.ndarray, np.ndarray, float, float, int, np.ndarray, np.ndarray, int, float]:
    X, y, base_mask = _build_joint_roi_design(
        brain_vec=brain_vec,
        age_mat=age_mat,
        beh_vec=beh_vec,
        normalize_interaction=bool(normalize_interaction),
    )
    p = 8
    if int(X.shape[0]) <= int(X.shape[1]):
        beta = np.full(p, np.nan, dtype=np.float64)
        tvals = np.full(p, np.nan, dtype=np.float64)
        rank = int(np.linalg.matrix_rank(X)) if X.size else 0
        return beta, tvals, np.nan, np.nan, int(X.shape[0]), X, base_mask, rank, np.nan
    beta, tvals, r2, r2_adj = _ols_fit_1d(X, y)
    rank = int(np.linalg.matrix_rank(X))
    cond = float(np.linalg.cond(X)) if X.size else np.nan
    return beta, tvals, float(r2), float(r2_adj), int(X.shape[0]), X, base_mask, rank, cond


def _perm_stats_for_joint_roi(
    brain_vec: np.ndarray,
    age_mat: np.ndarray,
    beh_vec: np.ndarray,
    pair_pos: np.ndarray,
    iu: np.ndarray,
    ju: np.ndarray,
    n_sub: int,
    n_perm: int,
    seed: int,
    tail: str,
    normalize_interaction: bool,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    p_nan = {effect: np.nan for effect in JOINT_EFFECTS}
    max_stats = {effect: np.full(int(n_perm), -np.inf, dtype=np.float64) for effect in JOINT_EFFECTS}
    perm_t = {effect: np.full(int(n_perm), np.nan, dtype=np.float64) for effect in JOINT_EFFECTS}
    perm_beta = {effect: np.full(int(n_perm), np.nan, dtype=np.float64) for effect in JOINT_EFFECTS}
    if int(n_perm) <= 0:
        return p_nan, max_stats, perm_t, perm_beta

    _, t_obs, _, _, n_obs, X_obs, base_mask, _, _ = _fit_joint_roi_model(
        brain_vec=brain_vec,
        age_mat=age_mat,
        beh_vec=beh_vec,
        normalize_interaction=bool(normalize_interaction),
    )
    if int(n_obs) <= int(X_obs.shape[1]):
        return p_nan, max_stats, perm_t, perm_beta

    counts = {effect: 0 for effect in JOINT_EFFECTS}
    rng = np.random.default_rng(int(seed))
    tail_mode = str(tail).strip().lower()
    for pi in range(int(n_perm)):
        perm = rng.permutation(int(n_sub))
        idx_perm_full = pair_pos[perm[iu], perm[ju]]
        idx_perm = idx_perm_full[base_mask]
        y_perm = beh_vec[idx_perm].astype(np.float64, copy=False)
        keep = np.isfinite(y_perm) & np.isfinite(X_obs).all(axis=1)
        if int(keep.sum()) <= int(X_obs.shape[1]):
            continue
        beta_perm, t_perm, _, _ = _ols_fit_1d(X_obs[keep], y_perm[keep])
        for effect in JOINT_EFFECTS:
            ti = int(JOINT_EFFECT_TO_T_INDEX[effect])
            perm_t[effect][pi] = float(t_perm[ti])
            perm_beta[effect][pi] = float(beta_perm[ti])
            tp = float(t_perm[ti])
            to = float(t_obs[ti])
            if not (np.isfinite(tp) and np.isfinite(to)):
                continue
            stat_p = abs(tp) if tail_mode == "two_sided" else tp
            stat_o = abs(to) if tail_mode == "two_sided" else to
            counts[effect] += int(stat_p >= stat_o)
            max_stats[effect][pi] = max(max_stats[effect][pi], stat_p)

    denom = float(int(n_perm) + 1)
    pvals = {}
    for effect in JOINT_EFFECTS:
        to = float(t_obs[int(JOINT_EFFECT_TO_T_INDEX[effect])])
        pvals[effect] = (float(counts[effect]) + 1.0) / denom if np.isfinite(to) else np.nan
    return pvals, max_stats, perm_t, perm_beta


_DELTA_WORKER_CTX: dict = {}


def _init_delta_worker(ctx: dict) -> None:
    global _DELTA_WORKER_CTX
    _DELTA_WORKER_CTX = ctx


def _make_result_row(
    roi: str,
    model_name: str,
    n_sub: int,
    n_pairs: int,
    beta: np.ndarray,
    tvals: np.ndarray,
    r2: float,
    r2_adj: float,
    p_b: float,
    p_a: float,
    p_i: float,
    ctx: dict,
) -> dict:
    return {
        "roi": str(roi),
        "model": str(model_name),
        "n_subjects": int(n_sub),
        "n_pairs": int(n_pairs),
        "beta_brain_delta": float(beta[1]) if np.isfinite(beta[1]) else np.nan,
        "t_brain_delta": float(tvals[1]) if np.isfinite(tvals[1]) else np.nan,
        "beta_age": float(beta[2]) if np.isfinite(beta[2]) else np.nan,
        "t_age": float(tvals[2]) if np.isfinite(tvals[2]) else np.nan,
        "beta_interaction": float(beta[3]) if np.isfinite(beta[3]) else np.nan,
        "t_interaction": float(tvals[3]) if np.isfinite(tvals[3]) else np.nan,
        "r_squared": float(r2),
        "r_squared_adj": float(r2_adj),
        "p_perm_brain_delta": float(p_b),
        "p_perm_age": float(p_a),
        "p_perm_interaction": float(p_i),
        "p_fwer_brain_delta_model_wise": np.nan,
        "p_fwer_age_model_wise": np.nan,
        "p_fwer_interaction_model_wise": np.nan,
        "tail": str(ctx["tail_mode"]),
        "correction_mode": str(ctx["correction_mode"]),
        "n_perm": int(ctx["n_perm"]),
        "seed": int(ctx["seed"]),
        "n_jobs": int(ctx["n_jobs"]),
        "brain_stimulus_dir_name": str(ctx["brain_stimulus_dir_name"]),
        "behavior_stimulus_dir_name": str(ctx["behavior_stimulus_dir_name"]),
        "brain_isc_prefix": str(ctx["brain_prefix"]),
        "behavior_isc_prefix": str(ctx["behavior_prefix"]),
        "roi_selection_dir_name": str(ctx["roi_selection_dir_name"]),
        "roi_selection_model": str(ctx["roi_selection_model"]),
        "roi_selection_p_col": str(ctx["roi_selection_p_col"]),
        "roi_selection_alpha": ctx["roi_selection_alpha"],
        "roi_selection_mode": str(ctx["roi_selection_mode"]),
        "reappraisal_name": str(ctx["reappraisal_name"]),
        "passive_name": str(ctx["passive_name"]),
        "rank_transform": bool(ctx["rank_transform"]),
        "normalize_models": bool(ctx["normalize_models"]),
        "fisher_z_brain": bool(ctx["fisher_z_brain"]),
        "fisher_z_behavior": bool(ctx["fisher_z_behavior"]),
    }


def _compute_delta_roi_task(local_i: int) -> dict:
    ctx = _DELTA_WORKER_CTX
    brain_vec = ctx["brain_vecs"][int(local_i)]
    roi = str(ctx["selected_rois"][int(local_i)])
    beta, tvals, r2, r2_adj, n_pairs, _, _ = _fit_roi_model(
        brain_vec=brain_vec,
        age_vec=ctx["age_vec"],
        beh_vec=ctx["beh_vec"],
        normalize_interaction=bool(ctx["normalize_interaction"]),
    )
    if bool(ctx["use_perm"]):
        p_b, p_a, p_i, mb, ma, mi_arr = _perm_stats_for_roi(
            brain_vec=brain_vec,
            age_vec=ctx["age_vec"],
            beh_vec=ctx["beh_vec"],
            pair_pos=ctx["pair_pos"],
            iu=ctx["iu"],
            ju=ctx["ju"],
            n_sub=int(ctx["n_sub"]),
            n_perm=int(ctx["n_perm"]),
            seed=int(ctx["seed"]),
            model_index=int(ctx["model_index"]),
            tail=str(ctx["tail_mode"]),
            normalize_interaction=bool(ctx["normalize_interaction"]),
        )
    else:
        p_b = p_a = p_i = np.nan
        mb = ma = mi_arr = None
    row = _make_result_row(
        roi=roi,
        model_name=str(ctx["model_name"]),
        n_sub=int(ctx["n_sub"]),
        n_pairs=int(n_pairs),
        beta=beta,
        tvals=tvals,
        r2=float(r2),
        r2_adj=float(r2_adj),
        p_b=float(p_b),
        p_a=float(p_a),
        p_i=float(p_i),
        ctx=ctx,
    )
    return {
        "roi": roi,
        "row": row,
        "max_brain": mb,
        "max_age": ma,
        "max_interaction": mi_arr,
    }


def _make_joint_result_row(
    roi: str,
    n_sub: int,
    n_pairs: int,
    beta: np.ndarray,
    tvals: np.ndarray,
    r2: float,
    r2_adj: float,
    design_rank: int,
    design_condition_number: float,
    p_perm: dict[str, float],
    ctx: dict,
) -> dict:
    row = {
        "roi": str(roi),
        "model": "joint",
        "n_subjects": int(n_sub),
        "n_pairs": int(n_pairs),
        "n_predictors": 7,
        "design_rank": int(design_rank),
        "design_condition_number": float(design_condition_number) if np.isfinite(design_condition_number) else np.nan,
        "r_squared": float(r2) if np.isfinite(r2) else np.nan,
        "r_squared_adj": float(r2_adj) if np.isfinite(r2_adj) else np.nan,
        "tail": str(ctx["tail_mode"]),
        "correction_mode": str(ctx["correction_mode"]),
        "n_perm": int(ctx["n_perm"]),
        "seed": int(ctx["seed"]),
        "n_jobs": int(ctx["n_jobs"]),
        "brain_stimulus_dir_name": str(ctx["brain_stimulus_dir_name"]),
        "behavior_stimulus_dir_name": str(ctx["behavior_stimulus_dir_name"]),
        "brain_isc_prefix": str(ctx["brain_prefix"]),
        "behavior_isc_prefix": str(ctx["behavior_prefix"]),
        "roi_selection_dir_name": str(ctx["roi_selection_dir_name"]),
        "roi_selection_model": str(ctx["roi_selection_model"]),
        "roi_selection_p_col": str(ctx["roi_selection_p_col"]),
        "roi_selection_alpha": ctx["roi_selection_alpha"],
        "roi_selection_mode": str(ctx["roi_selection_mode"]),
        "reappraisal_name": str(ctx["reappraisal_name"]),
        "passive_name": str(ctx["passive_name"]),
        "rank_transform": bool(ctx["rank_transform"]),
        "normalize_models": bool(ctx["normalize_models"]),
        "fisher_z_brain": bool(ctx["fisher_z_brain"]),
        "fisher_z_behavior": bool(ctx["fisher_z_behavior"]),
    }
    for effect in JOINT_EFFECTS:
        idx = int(JOINT_EFFECT_TO_T_INDEX[effect])
        row[f"beta_{effect}"] = float(beta[idx]) if np.isfinite(beta[idx]) else np.nan
        row[f"t_{effect}"] = float(tvals[idx]) if np.isfinite(tvals[idx]) else np.nan
        row[f"p_perm_{effect}"] = float(p_perm.get(effect, np.nan))
        row[f"p_fwer_{effect}_model_wise"] = np.nan
    return row


def _compute_delta_joint_roi_task(local_i: int) -> dict:
    ctx = _DELTA_WORKER_CTX
    brain_vec = ctx["brain_vecs"][int(local_i)]
    roi = str(ctx["selected_rois"][int(local_i)])
    beta, tvals, r2, r2_adj, n_pairs, _, _, design_rank, design_condition_number = _fit_joint_roi_model(
        brain_vec=brain_vec,
        age_mat=ctx["age_mat"],
        beh_vec=ctx["beh_vec"],
        normalize_interaction=bool(ctx["normalize_interaction"]),
    )
    if bool(ctx["use_perm"]):
        p_perm, max_stats, perm_t, perm_beta = _perm_stats_for_joint_roi(
            brain_vec=brain_vec,
            age_mat=ctx["age_mat"],
            beh_vec=ctx["beh_vec"],
            pair_pos=ctx["pair_pos"],
            iu=ctx["iu"],
            ju=ctx["ju"],
            n_sub=int(ctx["n_sub"]),
            n_perm=int(ctx["n_perm"]),
            seed=int(ctx["seed"]),
            tail=str(ctx["tail_mode"]),
            normalize_interaction=bool(ctx["normalize_interaction"]),
        )
    else:
        p_perm = {effect: np.nan for effect in JOINT_EFFECTS}
        max_stats = {effect: None for effect in JOINT_EFFECTS}
        perm_t = {effect: None for effect in JOINT_EFFECTS}
        perm_beta = {effect: None for effect in JOINT_EFFECTS}
    row = _make_joint_result_row(
        roi=roi,
        n_sub=int(ctx["n_sub"]),
        n_pairs=int(n_pairs),
        beta=beta,
        tvals=tvals,
        r2=float(r2),
        r2_adj=float(r2_adj),
        design_rank=int(design_rank),
        design_condition_number=float(design_condition_number),
        p_perm=p_perm,
        ctx=ctx,
    )
    return {"roi": roi, "row": row, "max_stats": max_stats, "perm_t": perm_t, "perm_beta": perm_beta}


def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
    brain_stimulus_dir_name: Optional[str],
    behavior_stimulus_dir_name: Optional[str],
    reappraisal_name: str,
    passive_name: str,
    rois: Optional[list[str]],
    roi_selection_dir_name: Optional[str],
    roi_selection_conditions: Optional[list[str]],
    roi_selection_file: str,
    roi_selection_model: list[str] | str,
    roi_selection_p_col: str,
    roi_selection_alpha: float,
    roi_selection_mode: str,
    brain_isc_method: str,
    brain_isc_prefix: Optional[str],
    behavior_isc_method: str,
    behavior_isc_prefix: Optional[str],
    age_model_mode: str,
    fisher_z_brain: Optional[str],
    fisher_z_behavior: Optional[str],
    normalize_models: bool,
    rank_transform: bool,
    tail: str,
    correction_mode: str,
    n_perm: int,
    seed: int,
    n_jobs: int,
    out_dir: Optional[Path],
    schaefer_mapping_file: Optional[Path],
) -> pd.DataFrame:
    matrix_dir = Path(matrix_dir)
    brain_dir_name = str(brain_stimulus_dir_name) if brain_stimulus_dir_name is not None else str(stimulus_dir_name)
    behavior_dir_name = str(behavior_stimulus_dir_name) if behavior_stimulus_dir_name is not None else str(stimulus_dir_name)
    brain_by_stim = matrix_dir / brain_dir_name
    behavior_by_stim = matrix_dir / behavior_dir_name
    brain_reapp_dir = brain_by_stim / str(reappraisal_name)
    brain_passive_dir = brain_by_stim / str(passive_name)
    behavior_reapp_dir = behavior_by_stim / str(reappraisal_name)
    behavior_passive_dir = behavior_by_stim / str(passive_name)
    for label, path in {
        "brain Reappraisal": brain_reapp_dir,
        "brain Passive": brain_passive_dir,
        "behavior Reappraisal": behavior_reapp_dir,
        "behavior Passive": behavior_passive_dir,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {label} directory: {path}")

    brain_prefix = _resolve_prefix(brain_isc_prefix, method=str(brain_isc_method), fallback_template="roi_isc_{method}_by_age")
    behavior_prefix = _resolve_prefix(behavior_isc_prefix, method=str(behavior_isc_method), fallback_template="behavior_pattern_isc_{method}_by_age")
    fisher_z_brain_eff = _policy_flag(method=str(brain_isc_method), override=fisher_z_brain)
    fisher_z_behavior_eff = _policy_flag(method=str(behavior_isc_method), override=fisher_z_behavior)

    brain_delta, behavior_delta, subjects, ages, all_rois = _align_delta_inputs(
        brain_reapp_dir=brain_reapp_dir,
        brain_passive_dir=brain_passive_dir,
        behavior_reapp_dir=behavior_reapp_dir,
        behavior_passive_dir=behavior_passive_dir,
        brain_prefix=str(brain_prefix),
        behavior_prefix=str(behavior_prefix),
    )
    rois_from_selection: Optional[list[str]] = None
    if roi_selection_dir_name is not None:
        selection_conditions = (
            [str(x) for x in roi_selection_conditions]
            if roi_selection_conditions is not None
            else [str(reappraisal_name), str(passive_name)]
        )
        rois_from_selection = _select_rois_from_dev_models(
            matrix_dir=matrix_dir,
            roi_selection_dir_name=str(roi_selection_dir_name),
            conditions=selection_conditions,
            roi_selection_file=str(roi_selection_file),
            roi_selection_model=str(roi_selection_model),
            roi_selection_p_col=str(roi_selection_p_col),
            roi_selection_alpha=float(roi_selection_alpha),
            roi_selection_mode=str(roi_selection_mode),
        )

    if rois_from_selection is not None and rois is not None:
        requested = [str(r) for r in rois]
        selection_aliases = {alias for roi in rois_from_selection for alias in _roi_aliases(str(roi))}
        selected_rois = [r for r in requested if any(alias in selection_aliases for alias in _roi_aliases(str(r)))]
        print(f"[roi-selection] Intersected explicit --rois with selected set: {len(selected_rois)} ROIs")
    elif rois_from_selection is not None:
        selected_rois = [str(r) for r in rois_from_selection]
    else:
        selected_rois = [str(r) for r in all_rois] if rois is None else [str(r) for r in rois]

    selected_rois = _resolve_roi_list(selected_rois, [str(r) for r in all_rois])
    if len(set(selected_rois)) != len(selected_rois):
        dup = sorted({r for r in selected_rois if selected_rois.count(r) > 1})
        raise ValueError(f"Duplicate requested ROIs: {dup}")
    if not selected_rois:
        raise ValueError("No ROIs selected for delta regression")
    roi_index = {r: i for i, r in enumerate(all_rois)}
    selected_idx = [roi_index[r] for r in selected_rois]
    print(
        "[inputs] "
        f"brain_dir={brain_dir_name}, behavior_dir={behavior_dir_name}, "
        f"n_subjects={len(subjects)}, n_rois={len(selected_rois)}"
    )
    roi_selection_model_label = (
        ",".join(str(x) for x in roi_selection_model)
        if isinstance(roi_selection_model, (list, tuple))
        else str(roi_selection_model)
    )

    n_sub = len(subjects)
    iu, ju = np.triu_indices(n_sub, k=1)
    pair_pos = np.full((n_sub, n_sub), -1, dtype=np.int32)
    pair_pos[iu, ju] = np.arange(int(iu.size), dtype=np.int32)
    pair_pos[ju, iu] = pair_pos[iu, ju]

    beh_vec0 = behavior_delta[iu, ju].astype(np.float32)
    if bool(fisher_z_behavior_eff):
        beh_vec0 = fisher_z(beh_vec0)
    beh_vec = _prepare_vector(beh_vec0, rank_transform=bool(rank_transform))

    m_obs = build_models(np.asarray(ages, dtype=np.float32), iu, ju, normalize=False)
    if bool(rank_transform):
        age_models = [_prepare_vector(m, rank_transform=True) for m in m_obs]
    else:
        age_models = [zscore_1d(m) if bool(normalize_models) else np.asarray(m, dtype=np.float32) for m in m_obs]

    brain_vecs = []
    for ri in selected_idx:
        v = brain_delta[ri][iu, ju].astype(np.float32)
        if bool(fisher_z_brain_eff):
            v = fisher_z(v)
        brain_vecs.append(_prepare_vector(v, rank_transform=bool(rank_transform)))
    brain_vecs = np.stack(brain_vecs, axis=0).astype(np.float32)

    rows: list[dict] = []
    tail_mode = str(tail).strip().lower()
    mode = str(correction_mode).strip().lower()
    age_mode = str(age_model_mode).strip().lower()
    if age_mode == "joint" and mode != "perm_fwer_fdr":
        raise ValueError("--age-model-mode joint currently requires --correction-mode perm_fwer_fdr")
    use_perm = mode == "perm_fwer_fdr" and int(n_perm) > 0
    common_ctx = {
        "brain_vecs": brain_vecs,
        "selected_rois": selected_rois,
        "beh_vec": beh_vec,
        "pair_pos": pair_pos,
        "iu": iu,
        "ju": ju,
        "n_sub": int(n_sub),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "n_jobs": int(n_jobs),
        "tail_mode": str(tail_mode),
        "correction_mode": str(mode),
        "normalize_interaction": bool(rank_transform or normalize_models),
        "rank_transform": bool(rank_transform),
        "normalize_models": bool(normalize_models),
        "use_perm": bool(use_perm),
        "brain_prefix": str(brain_prefix),
        "behavior_prefix": str(behavior_prefix),
        "brain_stimulus_dir_name": str(brain_dir_name),
        "behavior_stimulus_dir_name": str(behavior_dir_name),
        "roi_selection_dir_name": "" if roi_selection_dir_name is None else str(roi_selection_dir_name),
        "roi_selection_model": "" if roi_selection_dir_name is None else str(roi_selection_model_label),
        "roi_selection_p_col": "" if roi_selection_dir_name is None else str(roi_selection_p_col),
        "roi_selection_alpha": np.nan if roi_selection_dir_name is None else float(roi_selection_alpha),
        "roi_selection_mode": "" if roi_selection_dir_name is None else str(roi_selection_mode),
        "reappraisal_name": str(reappraisal_name),
        "passive_name": str(passive_name),
        "fisher_z_brain": bool(fisher_z_brain_eff),
        "fisher_z_behavior": bool(fisher_z_behavior_eff),
    }
    if age_mode == "joint":
        max_by_effect = {effect: np.full(int(n_perm), -np.inf, dtype=np.float64) for effect in JOINT_EFFECTS}
        worker_ctx = dict(common_ctx)
        worker_ctx.update({"age_mat": np.stack(age_models, axis=0).astype(np.float32)})
        jobs = max(1, min(int(n_jobs), len(selected_rois)))
        tasks = list(range(len(selected_rois)))
        if jobs == 1:
            _init_delta_worker(worker_ctx)
            results = [_compute_delta_joint_roi_task(t) for t in tasks]
        else:
            print(f"[parallel] {reappraisal_name}-{passive_name} joint: running {len(tasks)} ROI tasks with n_jobs={jobs}")
            results = []
            with ProcessPoolExecutor(max_workers=jobs, initializer=_init_delta_worker, initargs=(worker_ctx,)) as ex:
                futures = [ex.submit(_compute_delta_joint_roi_task, t) for t in tasks]
                for fut in as_completed(futures):
                    results.append(fut.result())
        for result in results:
            rows.append(result["row"])
            if use_perm:
                for effect in JOINT_EFFECTS:
                    arr = result["max_stats"].get(effect)
                    if arr is not None:
                        max_by_effect[effect] = np.maximum(max_by_effect[effect], np.asarray(arr, dtype=np.float64))
        if use_perm:
            denom = float(int(n_perm) + 1)
            for row in rows:
                for effect in JOINT_EFFECTS:
                    tval = float(row.get(f"t_{effect}", np.nan))
                    valid_max = np.isfinite(max_by_effect[effect])
                    if not np.isfinite(tval):
                        row[f"p_fwer_{effect}_model_wise"] = np.nan
                    elif tail_mode == "two_sided":
                        row[f"p_fwer_{effect}_model_wise"] = (float(np.sum(max_by_effect[effect][valid_max] >= abs(tval))) + 1.0) / denom
                    else:
                        row[f"p_fwer_{effect}_model_wise"] = (float(np.sum(max_by_effect[effect][valid_max] >= tval)) + 1.0) / denom

        out_df = pd.DataFrame(rows)
        for effect in JOINT_EFFECTS:
            out_df[f"p_fdr_perm_{effect}_model_wise"] = bh_fdr(out_df[f"p_perm_{effect}"].to_numpy(dtype=float))
        schaefer_mapping = _load_schaefer_mapping(schaefer_mapping_file, matrix_dir=Path(matrix_dir))
        out_df = _add_schaefer_mapping_columns(out_df, schaefer_mapping)
        out_df = out_df.sort_values(["model", "roi"]).reset_index(drop=True)

        if out_dir is not None:
            out_root = Path(out_dir)
        elif brain_dir_name == behavior_dir_name:
            out_root = matrix_dir / brain_dir_name / f"{reappraisal_name}_minus_{passive_name}"
        else:
            out_root = matrix_dir / f"{brain_dir_name}_brain__{behavior_dir_name}_behavior" / f"{reappraisal_name}_minus_{passive_name}"
        out_root.mkdir(parents=True, exist_ok=True)
        out_csv = out_root / f"{OUT_PREFIX}_joint.csv"
        out_df.to_csv(out_csv, index=False)
        pd.DataFrame({"subject": subjects, "age": ages}).to_csv(out_root / f"{OUT_PREFIX}_joint_subjects_sorted.csv", index=False)
        pd.DataFrame({"roi": selected_rois}).to_csv(out_root / f"{OUT_PREFIX}_joint_rois.csv", index=False)
        if rois_from_selection is not None:
            pd.DataFrame({"roi": selected_rois}).to_csv(out_root / f"{OUT_PREFIX}_joint_selected_rois_from_dev_models.csv", index=False)
        legacy_effect_names = list(JOINT_EFFECTS)
        effect_names = [
            "brain_delta",
            "age_M_nn",
            "age_M_conv",
            "age_M_div",
            "brain_x_M_nn",
            "brain_x_M_conv",
            "brain_x_M_div",
        ]
        effect_to_legacy = dict(zip(effect_names, legacy_effect_names))
        rois_out = out_df["roi"].astype(str).tolist()
        result_by_roi = {str(result["roi"]): result for result in results}
        obs_beta = np.asarray(
            [[float(row.get(f"beta_{effect_to_legacy[effect]}", np.nan)) for effect in effect_names] for _, row in out_df.iterrows()],
            dtype=np.float64,
        )
        obs_t = np.asarray(
            [[float(row.get(f"t_{effect_to_legacy[effect]}", np.nan)) for effect in effect_names] for _, row in out_df.iterrows()],
            dtype=np.float64,
        )
        perm_t_arr = np.full((len(rois_out), len(effect_names), int(n_perm)), np.nan, dtype=np.float64)
        perm_beta_arr = np.full_like(perm_t_arr, np.nan)
        if use_perm:
            for ri, roi in enumerate(rois_out):
                result = result_by_roi.get(str(roi))
                if result is None:
                    continue
                for ei, effect in enumerate(effect_names):
                    legacy_effect = effect_to_legacy[effect]
                    t_arr = result.get("perm_t", {}).get(legacy_effect)
                    b_arr = result.get("perm_beta", {}).get(legacy_effect)
                    if t_arr is not None:
                        perm_t_arr[ri, ei, :] = np.asarray(t_arr, dtype=np.float64)
                    if b_arr is not None:
                        perm_beta_arr[ri, ei, :] = np.asarray(b_arr, dtype=np.float64)
        perm_max_t = np.asarray(
            [np.asarray(max_by_effect[effect_to_legacy[effect]], dtype=np.float64) for effect in effect_names],
            dtype=np.float64,
        )
        metadata = make_metadata(
            script_name=Path(__file__).name,
            result_type="roi_isc_behavior_delta_age_regression_perm_null",
            input_files={
                "brain_reappraisal_dir": str(brain_reapp_dir),
                "brain_passive_dir": str(brain_passive_dir),
                "behavior_reappraisal_dir": str(behavior_reapp_dir),
                "behavior_passive_dir": str(behavior_passive_dir),
            },
            output_csv=str(out_csv),
            seed=int(seed),
            n_perm=int(n_perm),
            models=["joint"],
            extra={
                "age_model_mode": str(age_mode),
                "effects": effect_names,
                "legacy_effects": legacy_effect_names,
                "effect_to_legacy": effect_to_legacy,
                "brain_prefix": str(brain_prefix),
                "behavior_prefix": str(behavior_prefix),
                "correction_mode": str(mode),
                "tail": str(tail_mode),
                "rank_transform": bool(rank_transform),
                "normalize_models": bool(normalize_models),
            },
        )
        save_perm_null_npz(
            out_root / "roi_isc_behavior_delta_age_regression_perm_null.npz",
            metadata=metadata,
            obs_beta=obs_beta,
            obs_t=obs_t,
            perm_beta=perm_beta_arr,
            perm_t=perm_t_arr,
            perm_max_t=perm_max_t,
            rois=np.asarray(rois_out, dtype=str),
            effects=np.asarray(effect_names, dtype=str),
            legacy_effects=np.asarray(legacy_effect_names, dtype=str),
            selected_rois=np.asarray(selected_rois, dtype=str),
            subjects=np.asarray(subjects, dtype=str),
            ages=np.asarray(ages, dtype=np.float32),
        )
        print(f"Saved: {out_csv}")
        return out_df

    if age_mode != "separate":
        raise ValueError(f"Unsupported age_model_mode: {age_model_mode}")

    for mi, mname in enumerate(MODEL_NAMES):
        age_vec = age_models[mi]
        max_brain_all = np.full(int(n_perm), -np.inf, dtype=np.float64)
        max_age_all = np.full(int(n_perm), -np.inf, dtype=np.float64)
        max_inter_all = np.full(int(n_perm), -np.inf, dtype=np.float64)
        model_rows: list[dict] = []
        worker_ctx = dict(common_ctx)
        worker_ctx.update({
            "age_vec": age_vec,
            "model_index": int(mi),
            "model_name": str(mname),
        })
        jobs = max(1, min(int(n_jobs), len(selected_rois)))
        tasks = list(range(len(selected_rois)))
        if jobs == 1:
            _init_delta_worker(worker_ctx)
            results = [_compute_delta_roi_task(t) for t in tasks]
        else:
            print(f"[parallel] {reappraisal_name}-{passive_name} {mname}: running {len(tasks)} ROI tasks with n_jobs={jobs}")
            results = []
            with ProcessPoolExecutor(max_workers=jobs, initializer=_init_delta_worker, initargs=(worker_ctx,)) as ex:
                futures = [ex.submit(_compute_delta_roi_task, t) for t in tasks]
                for fut in as_completed(futures):
                    results.append(fut.result())
        for result in results:
            model_rows.append(result["row"])
            if use_perm:
                max_brain_all = np.maximum(max_brain_all, np.asarray(result["max_brain"], dtype=np.float64))
                max_age_all = np.maximum(max_age_all, np.asarray(result["max_age"], dtype=np.float64))
                max_inter_all = np.maximum(max_inter_all, np.asarray(result["max_interaction"], dtype=np.float64))
        if use_perm:
            denom = float(int(n_perm) + 1)
            for row in model_rows:
                tb = float(row["t_brain_delta"])
                ta = float(row["t_age"])
                ti = float(row["t_interaction"])
                if tail_mode == "two_sided":
                    row["p_fwer_brain_delta_model_wise"] = (float(np.sum(max_brain_all[np.isfinite(max_brain_all)] >= abs(tb))) + 1.0) / denom if np.isfinite(tb) else np.nan
                    row["p_fwer_age_model_wise"] = (float(np.sum(max_age_all[np.isfinite(max_age_all)] >= abs(ta))) + 1.0) / denom if np.isfinite(ta) else np.nan
                    row["p_fwer_interaction_model_wise"] = (float(np.sum(max_inter_all[np.isfinite(max_inter_all)] >= abs(ti))) + 1.0) / denom if np.isfinite(ti) else np.nan
                else:
                    row["p_fwer_brain_delta_model_wise"] = (float(np.sum(max_brain_all[np.isfinite(max_brain_all)] >= tb)) + 1.0) / denom if np.isfinite(tb) else np.nan
                    row["p_fwer_age_model_wise"] = (float(np.sum(max_age_all[np.isfinite(max_age_all)] >= ta)) + 1.0) / denom if np.isfinite(ta) else np.nan
                    row["p_fwer_interaction_model_wise"] = (float(np.sum(max_inter_all[np.isfinite(max_inter_all)] >= ti)) + 1.0) / denom if np.isfinite(ti) else np.nan
        rows.extend(model_rows)

    out_df = pd.DataFrame(rows)
    for mname in MODEL_NAMES:
        idx = out_df["model"].astype(str) == str(mname)
        out_df.loc[idx, "p_fdr_perm_brain_delta_model_wise"] = bh_fdr(out_df.loc[idx, "p_perm_brain_delta"].to_numpy())
        out_df.loc[idx, "p_fdr_perm_age_model_wise"] = bh_fdr(out_df.loc[idx, "p_perm_age"].to_numpy())
        out_df.loc[idx, "p_fdr_perm_interaction_model_wise"] = bh_fdr(out_df.loc[idx, "p_perm_interaction"].to_numpy())
    schaefer_mapping = _load_schaefer_mapping(schaefer_mapping_file, matrix_dir=Path(matrix_dir))
    out_df = _add_schaefer_mapping_columns(out_df, schaefer_mapping)
    out_df = out_df.sort_values(["model", "p_perm_interaction", "roi"]).reset_index(drop=True)

    if out_dir is not None:
        out_root = Path(out_dir)
    elif brain_dir_name == behavior_dir_name:
        out_root = matrix_dir / brain_dir_name / f"{reappraisal_name}_minus_{passive_name}"
    else:
        out_root = matrix_dir / f"{brain_dir_name}_brain__{behavior_dir_name}_behavior" / f"{reappraisal_name}_minus_{passive_name}"
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / f"{OUT_PREFIX}.csv"
    out_df.to_csv(out_csv, index=False)
    pd.DataFrame({"subject": subjects, "age": ages}).to_csv(out_root / f"{OUT_PREFIX}_subjects_sorted.csv", index=False)
    pd.DataFrame({"roi": selected_rois}).to_csv(out_root / f"{OUT_PREFIX}_rois.csv", index=False)
    if rois_from_selection is not None:
        pd.DataFrame({"roi": selected_rois}).to_csv(out_root / f"{OUT_PREFIX}_selected_rois_from_dev_models.csv", index=False)
    print(f"Saved: {out_csv}")
    return out_df


def main() -> None:
    args = parse_args()
    run(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        brain_stimulus_dir_name=args.brain_stimulus_dir_name,
        behavior_stimulus_dir_name=args.behavior_stimulus_dir_name,
        reappraisal_name=str(args.reappraisal_name),
        passive_name=str(args.passive_name),
        rois=None if args.rois is None else [str(r) for r in args.rois],
        roi_selection_dir_name=args.roi_selection_dir_name,
        roi_selection_conditions=None if args.roi_selection_conditions is None else [str(x) for x in args.roi_selection_conditions],
        roi_selection_file=str(args.roi_selection_file),
        roi_selection_model=[str(x) for x in args.roi_selection_model],
        roi_selection_p_col=str(args.roi_selection_p_col),
        roi_selection_alpha=float(args.roi_selection_alpha),
        roi_selection_mode=str(args.roi_selection_mode),
        brain_isc_method=str(args.brain_isc_method),
        brain_isc_prefix=args.brain_isc_prefix,
        behavior_isc_method=str(args.behavior_isc_method),
        behavior_isc_prefix=args.behavior_isc_prefix,
        age_model_mode=str(args.age_model_mode),
        fisher_z_brain=args.fisher_z_brain,
        fisher_z_behavior=args.fisher_z_behavior,
        normalize_models=bool(args.normalize_models),
        rank_transform=bool(args.rank_transform),
        tail=str(args.tail),
        correction_mode=str(args.correction_mode),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        n_jobs=int(args.n_jobs),
        out_dir=args.out_dir,
        schaefer_mapping_file=args.schaefer_mapping_file,
    )


if __name__ == "__main__":
    main()
