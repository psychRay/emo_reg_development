#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank transform for 1D arrays, matching scipy.stats.rankdata default."""
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


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64).reshape(-1)
    q = np.full(p.shape, np.nan, dtype=np.float64)
    m = np.isfinite(p)
    if int(m.sum()) == 0:
        return q
    pv = p[m]
    n = int(pv.size)
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * float(n) / np.arange(1, n + 1, dtype=np.float64)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_valid = np.empty_like(pv)
    q_valid[order] = q_ranked
    q[m] = q_valid
    return q


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.isfinite(x)
    if int(m.sum()) < 10:
        return np.full_like(x, np.nan, dtype=np.float32)
    mu = float(x[m].mean())
    sd = float(x[m].std(ddof=0))
    if sd <= 0:
        return np.full_like(x, np.nan, dtype=np.float32)
    out = (x - mu) / sd
    out[~m] = np.nan
    return out.astype(np.float32)


def fisher_z(r: np.ndarray) -> np.ndarray:
    x = np.asarray(r, dtype=np.float32)
    m = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=np.float32)
    if not bool(m.any()):
        return out
    xc = np.clip(x[m], -0.999999, 0.999999)
    out[m] = np.arctanh(xc).astype(np.float32, copy=False)
    return out

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multiple Regression RSA: Behavior_ISC = β1*Brain_ISC + β2*Age_Model + β3*(Brain_ISC*Age_Model)")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--brain-isc-method", type=str, default="mahalanobis")
    p.add_argument("--brain-isc-prefix", type=str, default=None)
    p.add_argument("--behavior-isc-method", type=str, default="mahalanobis")
    p.add_argument("--behavior-isc-prefix", type=str, default=None)
    p.add_argument("--age-model-mode", type=str, default="separate", choices=("separate", "joint"))
    p.add_argument("--fisher-z-brain", type=str, default=None)
    p.add_argument("--fisher-z-behavior", type=str, default=None)
    p.add_argument("--no-normalize-models", action="store_false", dest="normalize_models", default=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--rank-transform", action="store_true", dest="rank_transform", help="Rank transform matrices before regression (similar to Spearman)")
    g.add_argument("--no-rank-transform", action="store_false", dest="rank_transform", help="Disable rank transform and use standard z-scoring")
    p.set_defaults(rank_transform=True)
    p.add_argument("--tail", type=str, default="positive", choices=("two_sided", "positive"))
    p.add_argument("--correction-mode", type=str, default="perm_fwer_fdr", choices=("fdr_only", "perm_fwer_fdr"))
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1, help="Number of ROI workers. Results/checkpoints are still written by the main process.")
    return p.parse_args()

def _policy_flag(method: str, override: Optional[str]) -> bool:
    if override is not None:
        return str(override).strip().lower() == "true"
    return str(method).strip().lower() in {"pearson", "spearman"}

def _resolve_prefix(prefix: Optional[str], method: str, fallback_template: str) -> str:
    return str(prefix) if prefix is not None else fallback_template.format(method=str(method).strip().lower())

def has_brain_files(stim_dir: Path, brain_isc_prefix: str) -> bool:
    required = (
        stim_dir / f"{brain_isc_prefix}.npy",
        stim_dir / f"{brain_isc_prefix}_subjects_sorted.csv",
        stim_dir / f"{brain_isc_prefix}_rois.csv",
    )
    return all(p.exists() for p in required)

def has_behavior_files(stim_dir: Path, behavior_isc_prefix: str) -> bool:
    required = (
        stim_dir / f"{behavior_isc_prefix}.npy",
        stim_dir / f"{behavior_isc_prefix}_subjects_sorted.csv",
    )
    return all(p.exists() for p in required)


def _validate_isc_inputs(
    brain_isc: np.ndarray,
    brain_subjects: List[str],
    behavior_isc: np.ndarray,
    behavior_subjects: List[str],
) -> None:
    if len(set(brain_subjects)) != len(brain_subjects):
        dup = sorted({s for s in brain_subjects if brain_subjects.count(s) > 1})
        raise ValueError(f"Duplicate subjects in brain ISC subject file: {dup}")
    if len(set(behavior_subjects)) != len(behavior_subjects):
        dup = sorted({s for s in behavior_subjects if behavior_subjects.count(s) > 1})
        raise ValueError(f"Duplicate subjects in behavior ISC subject file: {dup}")

    b = np.asarray(brain_isc)
    h = np.asarray(behavior_isc)
    n_brain = len(brain_subjects)
    n_beh = len(behavior_subjects)
    if b.ndim != 3 or b.shape[1] != n_brain or b.shape[2] != n_brain:
        raise ValueError(
            f"Brain ISC shape {b.shape} does not match brain subjects ({n_brain}); "
            "expected (n_rois, n_subjects, n_subjects)"
        )
    if h.ndim != 2 or h.shape[0] != n_beh or h.shape[1] != n_beh:
        raise ValueError(
            f"Behavior ISC shape {h.shape} does not match behavior subjects ({n_beh}); "
            "expected (n_subjects, n_subjects)"
        )

def _align_subjects(
    brain_isc: np.ndarray,
    brain_subjects_df: pd.DataFrame,
    behavior_isc: np.ndarray,
    behavior_subjects_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, List[str], List[float]]:
    brain_subjects = brain_subjects_df["subject"].astype(str).tolist()
    behavior_subjects = behavior_subjects_df["subject"].astype(str).tolist()
    _validate_isc_inputs(brain_isc, brain_subjects, behavior_isc, behavior_subjects)
    
    brain_has_age = "age" in brain_subjects_df.columns
    beh_has_age = "age" in behavior_subjects_df.columns
    if not (brain_has_age or beh_has_age):
        raise KeyError("Cannot find 'age' column in either brain or behavior subject CSV")

    age_map: dict[str, float] = {}
    if brain_has_age:
        age_map.update(dict(zip(brain_subjects_df["subject"].astype(str), brain_subjects_df["age"].astype(float))))
    if beh_has_age:
        beh_age_map = dict(zip(behavior_subjects_df["subject"].astype(str), behavior_subjects_df["age"].astype(float)))
        for s, a in beh_age_map.items():
            cur = age_map.get(s, np.nan)
            if not np.isfinite(cur):
                age_map[s] = a
    
    behavior_set = set(behavior_subjects)
    
    # Ensure subjects have valid age (not 999.0 and not NaN)
    overlap = []
    for s in brain_subjects:
        if s in behavior_set:
            age_val = age_map.get(s, np.nan)
            if np.isfinite(age_val) and age_val < 998.0:
                overlap.append(s)
                
    if len(overlap) < 3:
        raise ValueError(f"Fewer than 3 overlapping subjects with valid age between brain ISC and behavior ISC")

    brain_idx = {s: i for i, s in enumerate(brain_subjects)}
    behavior_idx = {s: i for i, s in enumerate(behavior_subjects)}
    
    ib = [brain_idx[s] for s in overlap]
    ih = [behavior_idx[s] for s in overlap]
    aligned_brain_subjects = [brain_subjects[i] for i in ib]
    aligned_behavior_subjects = [behavior_subjects[i] for i in ih]
    if aligned_brain_subjects != aligned_behavior_subjects:
        raise ValueError(
            "Aligned brain/behavior subject order differs: "
            f"brain={aligned_brain_subjects[:10]}, behavior={aligned_behavior_subjects[:10]}"
        )
    
    brain_isc = np.asarray(brain_isc, dtype=np.float32)[:, ib, :][:, :, ib]
    behavior_isc = np.asarray(behavior_isc, dtype=np.float32)[np.ix_(ih, ih)]
    ages = [float(age_map[s]) for s in aligned_brain_subjects]
    if brain_isc.shape[1:] != behavior_isc.shape:
        raise ValueError(f"Aligned brain/behavior ISC shapes differ: brain={brain_isc.shape}, behavior={behavior_isc.shape}")
    
    return brain_isc, behavior_isc, aligned_brain_subjects, ages

def build_models(ages: np.ndarray, iu: np.ndarray, ju: np.ndarray, normalize: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(ages, dtype=np.float32).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("ages contains NaN/Inf")
    amax = float(a.max())
    ai = a[iu]
    aj = a[ju]
    nn = (amax - np.abs(ai - aj)).astype(np.float32)
    conv = np.minimum(ai, aj).astype(np.float32)
    div = (amax - 0.5 * (ai + aj)).astype(np.float32)
    if normalize:
        return zscore_1d(nn), zscore_1d(conv), zscore_1d(div)
    return nn, conv, div

def _ols_tvals(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xv = np.asarray(X, dtype=np.float64)
    Yv = np.asarray(Y, dtype=np.float64)
    if Yv.ndim != 2:
        raise ValueError("Y must be 2D (n_rois, n_obs)")
    if Xv.ndim != 2:
        raise ValueError("X must be 2D (n_obs, p)")
    n_obs = int(Xv.shape[0])
    p = int(Xv.shape[1])
    if int(Yv.shape[1]) != n_obs:
        raise ValueError("X and Y have incompatible shapes")
    dof = float(n_obs - p)
    if dof <= 0:
        raise ValueError("Too few observations for OLS")
    xtx_inv = np.linalg.pinv(Xv.T @ Xv)
    b = xtx_inv @ Xv.T
    beta = b @ Yv.T
    yhat = Xv @ beta
    resid = Yv.T - yhat
    mse = (resid * resid).sum(axis=0) / dof
    diag = np.diag(xtx_inv).reshape(-1, 1)
    se = np.sqrt(np.maximum(0.0, diag * mse.reshape(1, -1)))
    tvals = beta / se
    return beta.T, tvals.T


def _ols_tvals_behavior_y_by_roi(
    brain_pred: np.ndarray,
    age_vec: np.ndarray,
    y_vec: np.ndarray,
    normalize_interaction: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized OLS over ROIs: behavior ~ brain_roi + age + brain_roi:age."""
    B = np.asarray(brain_pred, dtype=np.float64)
    A = np.asarray(age_vec, dtype=np.float64).reshape(-1)
    Y = np.asarray(y_vec, dtype=np.float64).reshape(-1)
    if B.ndim != 2:
        raise ValueError("brain_pred must be 2D (n_rois, n_obs)")
    if B.shape[1] != A.shape[0] or B.shape[1] != Y.shape[0]:
        raise ValueError("brain_pred, age_vec, and y_vec have incompatible shapes")

    n_obs = int(B.shape[1])
    p = 4
    dof = float(n_obs - p)
    if dof <= 0:
        raise ValueError("Too few observations for OLS")

    C = B * A.reshape(1, -1)
    valid_interaction = np.ones(B.shape[0], dtype=bool)
    if bool(normalize_interaction):
        c_mu = C.mean(axis=1, keepdims=True)
        c_sd = C.std(axis=1, keepdims=True, ddof=0)
        valid_interaction = (n_obs >= 10) & (c_sd.reshape(-1) > 0) & np.isfinite(c_sd.reshape(-1))
        C = np.divide(C - c_mu, c_sd, out=np.zeros_like(C), where=c_sd > 0)
    ones = np.ones(B.shape[0], dtype=np.float64)

    s_b = B.sum(axis=1)
    s_a = float(A.sum())
    s_c = C.sum(axis=1)
    s_bb = (B * B).sum(axis=1)
    s_ba = (B * A.reshape(1, -1)).sum(axis=1)
    s_bc = (B * C).sum(axis=1)
    s_aa = float((A * A).sum())
    s_ac = (A.reshape(1, -1) * C).sum(axis=1)
    s_cc = (C * C).sum(axis=1)

    xtx = np.zeros((B.shape[0], p, p), dtype=np.float64)
    xtx[:, 0, 0] = float(n_obs)
    xtx[:, 0, 1] = xtx[:, 1, 0] = s_b
    xtx[:, 0, 2] = xtx[:, 2, 0] = s_a * ones
    xtx[:, 0, 3] = xtx[:, 3, 0] = s_c
    xtx[:, 1, 1] = s_bb
    xtx[:, 1, 2] = xtx[:, 2, 1] = s_ba
    xtx[:, 1, 3] = xtx[:, 3, 1] = s_bc
    xtx[:, 2, 2] = s_aa * ones
    xtx[:, 2, 3] = xtx[:, 3, 2] = s_ac
    xtx[:, 3, 3] = s_cc

    s_y = float(Y.sum())
    xty = np.stack(
        [
            s_y * ones,
            (B * Y.reshape(1, -1)).sum(axis=1),
            float((A * Y).sum()) * ones,
            (C * Y.reshape(1, -1)).sum(axis=1),
        ],
        axis=1,
    )

    xtx_inv = np.linalg.pinv(xtx)
    beta = np.einsum("rij,rj->ri", xtx_inv, xty)
    yy = float((Y * Y).sum())
    sse = yy - np.einsum("ri,ri->r", beta, xty)
    mse = np.maximum(0.0, sse) / dof
    diag = np.diagonal(xtx_inv, axis1=1, axis2=2)
    se = np.sqrt(np.maximum(0.0, diag * mse.reshape(-1, 1)))
    tvals = beta / se
    beta[~valid_interaction, :] = np.nan
    tvals[~valid_interaction, :] = np.nan
    return beta.astype(np.float64), tvals.astype(np.float64)


def _ols_fit_1d(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    Xv = np.asarray(X, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    if Xv.ndim != 2:
        raise ValueError("X must be 2D")
    if yv.ndim != 1 or yv.shape[0] != Xv.shape[0]:
        raise ValueError("y must be 1D and aligned with X")
    n_obs = int(Xv.shape[0])
    p = int(Xv.shape[1])
    dof = float(n_obs - p)
    if dof <= 0:
        raise ValueError("Too few observations for OLS")

    xtx_inv = np.linalg.pinv(Xv.T @ Xv)
    beta = xtx_inv @ Xv.T @ yv
    yhat = Xv @ beta
    resid = yv - yhat
    mse = float((resid * resid).sum() / dof)
    se = np.sqrt(np.maximum(0.0, np.diag(xtx_inv) * mse))
    tvals = beta / se

    ss_res = float((resid * resid).sum())
    y_center = yv - float(np.mean(yv))
    ss_tot = float((y_center * y_center).sum())
    if ss_tot > 0:
        r_squared = 1.0 - (ss_res / ss_tot)
        r_squared_adj = 1.0 - (1.0 - r_squared) * float(n_obs - 1) / float(max(1, n_obs - p))
    else:
        r_squared = np.nan
        r_squared_adj = np.nan
    return beta.astype(np.float64), tvals.astype(np.float64), float(r_squared), float(r_squared_adj)


def _analysis_signature(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_roi_design(
    brain_vec: np.ndarray,
    age_vec: np.ndarray,
    beh_vec: np.ndarray,
    normalize_interaction: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    brain = np.asarray(brain_vec, dtype=np.float64).reshape(-1)
    age = np.asarray(age_vec, dtype=np.float64).reshape(-1)
    beh = np.asarray(beh_vec, dtype=np.float64).reshape(-1)
    base_mask = np.isfinite(brain) & np.isfinite(age) & np.isfinite(beh)
    if int(base_mask.sum()) == 0:
        return np.empty((0, 4), dtype=np.float64), np.empty(0, dtype=np.float64), base_mask

    base_idx = np.where(base_mask)[0]
    b = brain[base_mask]
    a = age[base_mask]
    y = beh[base_mask]
    inter = b * a
    if bool(normalize_interaction):
        inter = zscore_1d(inter).astype(np.float64, copy=False)

    X = np.column_stack([np.ones_like(y, dtype=np.float64), b, a, inter])
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    final_mask = np.zeros_like(base_mask, dtype=bool)
    final_mask[base_idx[keep]] = True
    return X[keep], y[keep], final_mask


def _fit_roi_model(
    brain_vec: np.ndarray,
    age_vec: np.ndarray,
    beh_vec: np.ndarray,
    normalize_interaction: bool,
) -> tuple[np.ndarray, np.ndarray, float, float, int, np.ndarray, np.ndarray]:
    X, y, base_mask = _build_roi_design(
        brain_vec=brain_vec,
        age_vec=age_vec,
        beh_vec=beh_vec,
        normalize_interaction=normalize_interaction,
    )
    if int(X.shape[0]) <= int(X.shape[1]):
        beta = np.full(4, np.nan, dtype=np.float64)
        tvals = np.full(4, np.nan, dtype=np.float64)
        return beta, tvals, np.nan, np.nan, int(X.shape[0]), X, base_mask
    beta, tvals, r_squared, r_squared_adj = _ols_fit_1d(X, y)
    return beta, tvals, r_squared, r_squared_adj, int(X.shape[0]), X, base_mask


JOINT_AGE_MODEL_NAMES = ("M_nn", "M_conv", "M_div")
JOINT_EFFECTS = (
    "brain",
    "age_M_nn",
    "age_M_conv",
    "age_M_div",
    "interaction_M_nn",
    "interaction_M_conv",
    "interaction_M_div",
)
JOINT_EFFECT_TO_T_INDEX = {effect: i + 1 for i, effect in enumerate(JOINT_EFFECTS)}


def _build_joint_roi_design(
    brain_vec: np.ndarray,
    age_mat: np.ndarray,
    beh_vec: np.ndarray,
    normalize_interaction: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    brain = np.asarray(brain_vec, dtype=np.float64).reshape(-1)
    ages = np.asarray(age_mat, dtype=np.float64)
    beh = np.asarray(beh_vec, dtype=np.float64).reshape(-1)
    if ages.ndim != 2 or ages.shape[0] != len(JOINT_AGE_MODEL_NAMES):
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
        inter_z = []
        for row in inter:
            inter_z.append(zscore_1d(row).astype(np.float64, copy=False))
        inter = np.stack(inter_z, axis=0)

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
        normalize_interaction=normalize_interaction,
    )
    p = 8
    if int(X.shape[0]) <= int(X.shape[1]):
        beta = np.full(p, np.nan, dtype=np.float64)
        tvals = np.full(p, np.nan, dtype=np.float64)
        return beta, tvals, np.nan, np.nan, int(X.shape[0]), X, base_mask, int(np.linalg.matrix_rank(X)) if X.size else 0, np.nan
    beta, tvals, r_squared, r_squared_adj = _ols_fit_1d(X, y)
    rank = int(np.linalg.matrix_rank(X))
    cond = float(np.linalg.cond(X)) if X.size else np.nan
    return beta, tvals, r_squared, r_squared_adj, int(X.shape[0]), X, base_mask, rank, cond


def _checkpoint_required_cols() -> set[str]:
    return {
        "roi",
        "model",
        "n_subjects",
        "n_pairs",
        "beta_brain",
        "t_brain",
        "beta_age",
        "t_age",
        "beta_interaction",
        "t_interaction",
        "r_squared",
        "r_squared_adj",
        "p_perm_brain",
        "p_perm_age",
        "p_perm_interaction",
        "p_fwer_brain_model_wise",
        "p_fwer_age_model_wise",
        "p_fwer_interaction_model_wise",
        "analysis_signature",
        "model_complete",
    }


def _load_checkpoint_rows(out_path: Path, expected_signature: str) -> list[dict]:
    path = Path(out_path)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    required = _checkpoint_required_cols()
    if not required.issubset(set(df.columns)):
        print(f"[checkpoint] Existing result has old/incomplete schema, recomputing: {path}")
        return []
    sig = df["analysis_signature"].astype(str)
    if bool((sig != str(expected_signature)).any()):
        print(f"[checkpoint] Existing result belongs to a different analysis signature, recomputing: {path}")
        return []
    df = df.drop_duplicates(["roi", "model"], keep="last").reset_index(drop=True)
    return df.to_dict("records")


def _rows_to_df(rows: list[dict], model_names: tuple[str, ...]) -> pd.DataFrame:
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df
    out_df = out_df.drop_duplicates(["roi", "model"], keep="last").reset_index(drop=True)

    if set(model_names) == {"joint"}:
        for effect in JOINT_EFFECTS:
            p_col = f"p_perm_{effect}"
            if p_col not in out_df.columns:
                continue
            out_df[f"p_fdr_perm_{effect}_model_wise"] = bh_fdr(out_df[p_col].to_numpy(dtype=float))
        sort_cols = [c for c in ["model", "roi"] if c in out_df.columns]
        if sort_cols:
            out_df = out_df.sort_values(sort_cols).reset_index(drop=True)
        return out_df

    out_df["p_fdr_perm_interaction_model_wise"] = np.nan
    for mname in model_names:
        idx = out_df["model"].astype(str) == str(mname)
        out_df.loc[idx, "p_fdr_perm_interaction_model_wise"] = bh_fdr(out_df.loc[idx, "p_perm_interaction"].to_numpy())

    out_df["p_fdr_perm_brain_model_wise"] = np.nan
    for mname in model_names:
        idx = out_df["model"].astype(str) == str(mname)
        out_df.loc[idx, "p_fdr_perm_brain_model_wise"] = bh_fdr(out_df.loc[idx, "p_perm_brain"].to_numpy())

    out_df["p_fdr_perm_age_model_wise"] = np.nan
    for mname in model_names:
        idx = out_df["model"].astype(str) == str(mname)
        out_df.loc[idx, "p_fdr_perm_age_model_wise"] = bh_fdr(out_df.loc[idx, "p_perm_age"].to_numpy())

    sort_cols = [c for c in ["model", "roi"] if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)
    return out_df


def _write_checkpoint(out_path: Path, rows: list[dict], model_names: tuple[str, ...]) -> pd.DataFrame:
    out_df = _rows_to_df(rows, model_names=model_names)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    out_df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)
    return out_df


_ROI_WORKER_CTX: dict = {}


def _init_roi_worker(ctx: dict) -> None:
    global _ROI_WORKER_CTX
    _ROI_WORKER_CTX = ctx


def _permutation_stats_for_roi_core(
    brain_vec0: np.ndarray,
    age_vec: np.ndarray,
    beh_vec: np.ndarray,
    pair_pos: np.ndarray,
    iu: np.ndarray,
    ju: np.ndarray,
    n_sub: int,
    n_perm: int,
    seed: int,
    model_index: int,
    tail_mode: str,
    normalize_interaction: bool,
    collect_max: bool,
) -> tuple[float, float, float, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if int(n_perm) <= 0:
        return np.nan, np.nan, np.nan, None, None, None

    _, t_obs, _, _, n_obs, X_obs, base_mask = _fit_roi_model(
        brain_vec=brain_vec0,
        age_vec=age_vec,
        beh_vec=beh_vec,
        normalize_interaction=bool(normalize_interaction),
    )
    if int(n_obs) <= 5:
        return np.nan, np.nan, np.nan, None, None, None

    t_obs_b = float(t_obs[1])
    t_obs_a = float(t_obs[2])
    t_obs_i = float(t_obs[3])
    c_b = c_a = c_i = 0
    max_brain = np.full(int(n_perm), -np.inf, dtype=np.float64) if collect_max else None
    max_age = np.full(int(n_perm), -np.inf, dtype=np.float64) if collect_max else None
    max_interaction = np.full(int(n_perm), -np.inf, dtype=np.float64) if collect_max else None

    rng_roi = np.random.default_rng(int(seed) + int(model_index) * 100003)
    for pi in range(int(n_perm)):
        perm = rng_roi.permutation(int(n_sub))
        idx_perm_full = pair_pos[perm[iu], perm[ju]]
        idx_perm = idx_perm_full[base_mask]
        Yp = beh_vec[idx_perm].astype(np.float64, copy=False)
        keep = np.isfinite(Yp) & np.isfinite(X_obs).all(axis=1)
        if int(keep.sum()) <= int(X_obs.shape[1]):
            tp_b = tp_a = tp_i = np.nan
        else:
            _, t_perm, _, _ = _ols_fit_1d(X_obs[keep], Yp[keep])
            tp_b = float(t_perm[1])
            tp_a = float(t_perm[2])
            tp_i = float(t_perm[3])

        if tail_mode == "two_sided":
            if np.isfinite(tp_b) and np.isfinite(t_obs_b):
                c_b += int(abs(tp_b) >= abs(t_obs_b))
                if collect_max and max_brain is not None:
                    max_brain[pi] = max(max_brain[pi], abs(tp_b))
            if np.isfinite(tp_a) and np.isfinite(t_obs_a):
                c_a += int(abs(tp_a) >= abs(t_obs_a))
                if collect_max and max_age is not None:
                    max_age[pi] = max(max_age[pi], abs(tp_a))
            if np.isfinite(tp_i) and np.isfinite(t_obs_i):
                c_i += int(abs(tp_i) >= abs(t_obs_i))
                if collect_max and max_interaction is not None:
                    max_interaction[pi] = max(max_interaction[pi], abs(tp_i))
        else:
            if np.isfinite(tp_b) and np.isfinite(t_obs_b):
                c_b += int(tp_b >= t_obs_b)
                if collect_max and max_brain is not None:
                    max_brain[pi] = max(max_brain[pi], tp_b)
            if np.isfinite(tp_a) and np.isfinite(t_obs_a):
                c_a += int(tp_a >= t_obs_a)
                if collect_max and max_age is not None:
                    max_age[pi] = max(max_age[pi], tp_a)
            if np.isfinite(tp_i) and np.isfinite(t_obs_i):
                c_i += int(tp_i >= t_obs_i)
                if collect_max and max_interaction is not None:
                    max_interaction[pi] = max(max_interaction[pi], tp_i)

    denom = float(int(n_perm) + 1)
    p_b = (float(c_b) + 1.0) / denom if np.isfinite(t_obs_b) else np.nan
    p_a = (float(c_a) + 1.0) / denom if np.isfinite(t_obs_a) else np.nan
    p_i = (float(c_i) + 1.0) / denom if np.isfinite(t_obs_i) else np.nan
    return p_b, p_a, p_i, max_brain, max_age, max_interaction


def _make_result_row(
    roi: str,
    model_name: str,
    n_subjects: int,
    n_pairs: int,
    beta_hat: np.ndarray,
    t_hat: np.ndarray,
    r_squared: float,
    r_squared_adj: float,
    p_perm_brain: float,
    p_perm_age: float,
    p_perm_interaction: float,
    tail_mode: str,
    correction_mode: str,
    n_perm: int,
    seed: int,
    analysis_sig: str,
) -> dict:
    row = {
        "roi": str(roi),
        "model": str(model_name),
        "n_subjects": int(n_subjects),
        "n_pairs": int(n_pairs),
        "beta_brain": np.nan,
        "t_brain": np.nan,
        "beta_age": np.nan,
        "t_age": np.nan,
        "beta_interaction": np.nan,
        "t_interaction": np.nan,
        "r_squared": np.nan,
        "r_squared_adj": np.nan,
        "p_perm_brain": float(p_perm_brain),
        "p_perm_age": float(p_perm_age),
        "p_perm_interaction": float(p_perm_interaction),
        "p_fwer_brain_model_wise": np.nan,
        "p_fwer_age_model_wise": np.nan,
        "p_fwer_interaction_model_wise": np.nan,
        "tail": str(tail_mode),
        "correction_mode": str(correction_mode),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "analysis_signature": str(analysis_sig),
        "model_complete": False,
    }
    if np.isfinite(beta_hat).all() and np.isfinite(t_hat).all():
        row.update({
            "beta_brain": float(beta_hat[1]),
            "t_brain": float(t_hat[1]),
            "beta_age": float(beta_hat[2]),
            "t_age": float(t_hat[2]),
            "beta_interaction": float(beta_hat[3]),
            "t_interaction": float(t_hat[3]),
            "r_squared": float(r_squared),
            "r_squared_adj": float(r_squared_adj),
        })
    return row


def _compute_roi_task(task: tuple[int, str, bool, bool, bool]) -> dict:
    ri, roi, need_row, need_perm_p, need_max = task
    ctx = _ROI_WORKER_CTX
    brain_vec = ctx["S_z"][int(ri)]
    p_perm_brain = p_perm_age = p_perm_interaction = np.nan
    max_brain = max_age = max_interaction = None
    if bool(need_perm_p) or bool(need_max):
        p_perm_brain, p_perm_age, p_perm_interaction, max_brain, max_age, max_interaction = _permutation_stats_for_roi_core(
            brain_vec0=brain_vec,
            age_vec=ctx["age_vec"],
            beh_vec=ctx["beh_vec"],
            pair_pos=ctx["pair_pos"],
            iu=ctx["iu"],
            ju=ctx["ju"],
            n_sub=int(ctx["n_sub"]),
            n_perm=int(ctx["n_perm"]),
            seed=int(ctx["seed"]),
            model_index=int(ctx["model_index"]),
            tail_mode=str(ctx["tail_mode"]),
            normalize_interaction=bool(ctx["normalize_interaction"]),
            collect_max=bool(need_max),
        )

    row = None
    if bool(need_row):
        beta_hat, t_hat, r_squared, r_squared_adj, n_pairs, _, _ = _fit_roi_model(
            brain_vec=brain_vec,
            age_vec=ctx["age_vec"],
            beh_vec=ctx["beh_vec"],
            normalize_interaction=bool(ctx["normalize_interaction"]),
        )
        row = _make_result_row(
            roi=str(roi),
            model_name=str(ctx["model_name"]),
            n_subjects=int(ctx["n_sub"]),
            n_pairs=int(n_pairs),
            beta_hat=beta_hat,
            t_hat=t_hat,
            r_squared=float(r_squared),
            r_squared_adj=float(r_squared_adj),
            p_perm_brain=float(p_perm_brain),
            p_perm_age=float(p_perm_age),
            p_perm_interaction=float(p_perm_interaction),
            tail_mode=str(ctx["tail_mode"]),
            correction_mode=str(ctx["correction_mode"]),
            n_perm=int(ctx["n_perm"]),
            seed=int(ctx["seed"]),
            analysis_sig=str(ctx["analysis_sig"]),
        )

    return {
        "ri": int(ri),
        "roi": str(roi),
        "need_row": bool(need_row),
        "need_max": bool(need_max),
        "row": row,
        "max_brain": max_brain,
        "max_age": max_age,
        "max_interaction": max_interaction,
    }


def _permutation_stats_for_joint_roi_core(
    brain_vec0: np.ndarray,
    age_mat: np.ndarray,
    beh_vec: np.ndarray,
    pair_pos: np.ndarray,
    iu: np.ndarray,
    ju: np.ndarray,
    n_sub: int,
    n_perm: int,
    seed: int,
    tail_mode: str,
    normalize_interaction: bool,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    p_nan = {effect: np.nan for effect in JOINT_EFFECTS}
    max_stats = {effect: np.full(int(n_perm), -np.inf, dtype=np.float64) for effect in JOINT_EFFECTS}
    if int(n_perm) <= 0:
        return p_nan, max_stats

    _, t_obs, _, _, n_obs, X_obs, base_mask, _, _ = _fit_joint_roi_model(
        brain_vec=brain_vec0,
        age_mat=age_mat,
        beh_vec=beh_vec,
        normalize_interaction=bool(normalize_interaction),
    )
    if int(n_obs) <= int(X_obs.shape[1]):
        return p_nan, max_stats

    c = {effect: 0 for effect in JOINT_EFFECTS}
    rng_roi = np.random.default_rng(int(seed))
    tail = str(tail_mode).strip().lower()
    for pi in range(int(n_perm)):
        perm = rng_roi.permutation(int(n_sub))
        idx_perm_full = pair_pos[perm[iu], perm[ju]]
        idx_perm = idx_perm_full[base_mask]
        y_perm = beh_vec[idx_perm].astype(np.float64, copy=False)
        keep = np.isfinite(y_perm) & np.isfinite(X_obs).all(axis=1)
        if int(keep.sum()) <= int(X_obs.shape[1]):
            continue
        _, t_perm, _, _ = _ols_fit_1d(X_obs[keep], y_perm[keep])
        for effect in JOINT_EFFECTS:
            ti = int(JOINT_EFFECT_TO_T_INDEX[effect])
            tp = float(t_perm[ti])
            to = float(t_obs[ti])
            if not (np.isfinite(tp) and np.isfinite(to)):
                continue
            stat_p = abs(tp) if tail == "two_sided" else tp
            stat_o = abs(to) if tail == "two_sided" else to
            c[effect] += int(stat_p >= stat_o)
            max_stats[effect][pi] = max(max_stats[effect][pi], stat_p)

    denom = float(int(n_perm) + 1)
    pvals = {}
    for effect in JOINT_EFFECTS:
        to = float(t_obs[int(JOINT_EFFECT_TO_T_INDEX[effect])])
        pvals[effect] = (float(c[effect]) + 1.0) / denom if np.isfinite(to) else np.nan
    return pvals, max_stats


def _make_joint_result_row(
    roi: str,
    n_subjects: int,
    n_pairs: int,
    beta_hat: np.ndarray,
    t_hat: np.ndarray,
    r_squared: float,
    r_squared_adj: float,
    design_rank: int,
    design_condition_number: float,
    p_perm: dict[str, float],
    tail_mode: str,
    correction_mode: str,
    n_perm: int,
    seed: int,
    analysis_sig: str,
) -> dict:
    row = {
        "roi": str(roi),
        "model": "joint",
        "n_subjects": int(n_subjects),
        "n_pairs": int(n_pairs),
        "n_predictors": 7,
        "design_rank": int(design_rank),
        "design_condition_number": float(design_condition_number) if np.isfinite(design_condition_number) else np.nan,
        "r_squared": float(r_squared) if np.isfinite(r_squared) else np.nan,
        "r_squared_adj": float(r_squared_adj) if np.isfinite(r_squared_adj) else np.nan,
        "tail": str(tail_mode),
        "correction_mode": str(correction_mode),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "analysis_signature": str(analysis_sig),
        "model_complete": True,
    }
    for effect in JOINT_EFFECTS:
        idx = int(JOINT_EFFECT_TO_T_INDEX[effect])
        row[f"beta_{effect}"] = float(beta_hat[idx]) if np.isfinite(beta_hat[idx]) else np.nan
        row[f"t_{effect}"] = float(t_hat[idx]) if np.isfinite(t_hat[idx]) else np.nan
        row[f"p_perm_{effect}"] = float(p_perm.get(effect, np.nan))
        row[f"p_fwer_{effect}_model_wise"] = np.nan
    return row


def _compute_joint_roi_task(task: tuple[int, str]) -> dict:
    ri, roi = task
    ctx = _ROI_WORKER_CTX
    brain_vec = ctx["S_z"][int(ri)]
    p_perm, max_stats = _permutation_stats_for_joint_roi_core(
        brain_vec0=brain_vec,
        age_mat=ctx["age_mat"],
        beh_vec=ctx["beh_vec"],
        pair_pos=ctx["pair_pos"],
        iu=ctx["iu"],
        ju=ctx["ju"],
        n_sub=int(ctx["n_sub"]),
        n_perm=int(ctx["n_perm"]),
        seed=int(ctx["seed"]),
        tail_mode=str(ctx["tail_mode"]),
        normalize_interaction=bool(ctx["normalize_interaction"]),
    )
    beta_hat, t_hat, r_squared, r_squared_adj, n_pairs, _, _, design_rank, design_condition_number = _fit_joint_roi_model(
        brain_vec=brain_vec,
        age_mat=ctx["age_mat"],
        beh_vec=ctx["beh_vec"],
        normalize_interaction=bool(ctx["normalize_interaction"]),
    )
    row = _make_joint_result_row(
        roi=str(roi),
        n_subjects=int(ctx["n_sub"]),
        n_pairs=int(n_pairs),
        beta_hat=beta_hat,
        t_hat=t_hat,
        r_squared=float(r_squared),
        r_squared_adj=float(r_squared_adj),
        design_rank=int(design_rank),
        design_condition_number=float(design_condition_number),
        p_perm=p_perm,
        tail_mode=str(ctx["tail_mode"]),
        correction_mode=str(ctx["correction_mode"]),
        n_perm=int(ctx["n_perm"]),
        seed=int(ctx["seed"]),
        analysis_sig=str(ctx["analysis_sig"]),
    )
    return {"ri": int(ri), "roi": str(roi), "row": row, "max_stats": max_stats}

def run_one(
    stim_dir: Path,
    brain_isc_prefix: str,
    behavior_isc_prefix: str,
    age_model_mode: str,
    fisher_z_brain: bool,
    fisher_z_behavior: bool,
    normalize_models: bool,
    rank_transform: bool,
    tail: str,
    correction_mode: str,
    n_perm: int,
    seed: int,
    n_jobs: int = 1,
) -> pd.DataFrame:
    brain_isc = np.load(stim_dir / f"{brain_isc_prefix}.npy")
    behavior_isc = np.load(stim_dir / f"{behavior_isc_prefix}.npy")
    brain_sub_df = pd.read_csv(stim_dir / f"{brain_isc_prefix}_subjects_sorted.csv")
    beh_sub_df = pd.read_csv(stim_dir / f"{behavior_isc_prefix}_subjects_sorted.csv")
    roi_df = pd.read_csv(stim_dir / f"{brain_isc_prefix}_rois.csv")
    rois = roi_df["roi"].astype(str).tolist()

    brain_isc, behavior_isc, subjects, ages = _align_subjects(brain_isc, brain_sub_df, behavior_isc, beh_sub_df)
    n_sub = len(subjects)
    iu, ju = np.triu_indices(n_sub, k=1)

    # Prepare behavior vector (dependent variable Y).
    beh_vec = behavior_isc[iu, ju].astype(np.float32)
    if bool(fisher_z_behavior):
        beh_vec = fisher_z(beh_vec)
        
    if rank_transform:
        mask_b = np.isfinite(beh_vec)
        br = np.full_like(beh_vec, np.nan, dtype=np.float32)
        br[mask_b] = rankdata(beh_vec[mask_b])
        beh_vec = zscore_1d(br)
    else:
        beh_vec = zscore_1d(beh_vec)

    # Prepare age models (X2)
    model_names = ("M_nn", "M_conv", "M_div")
    m_obs = build_models(np.array(ages), iu, ju, normalize=False)
    
    if rank_transform:
        m_obs_z = []
        for m in m_obs:
            mask_m = np.isfinite(m)
            mr = np.full_like(m, np.nan, dtype=np.float32)
            mr[mask_m] = rankdata(m[mask_m])
            m_obs_z.append(zscore_1d(mr))
    else:
        m_obs_z = [zscore_1d(m) if normalize_models else m for m in m_obs]

    # Prepare ROI-wise brain vectors (predictors X1).
    S_z = []
    for ri in range(len(rois)):
        v = brain_isc[ri][iu, ju]
        if bool(fisher_z_brain):
            v = fisher_z(v)
        if rank_transform:
            mask_v = np.isfinite(v)
            vr = np.full_like(v, np.nan, dtype=np.float32)
            vr[mask_v] = rankdata(v[mask_v])
            S_z.append(zscore_1d(vr))
        else:
            S_z.append(zscore_1d(v))
    S_z = np.stack(S_z, axis=0).astype(np.float32)

    age_mode = str(age_model_mode).strip().lower()
    if age_mode == "joint" and str(correction_mode).strip().lower() != "perm_fwer_fdr":
        raise ValueError("--age-model-mode joint currently requires --correction-mode perm_fwer_fdr")

    out_prefix = "roi_isc_behavior_age_regression_joint" if age_mode == "joint" else "roi_isc_behavior_age_regression"
    out_path = stim_dir / f"{out_prefix}.csv"
    signature_payload = {
        "version": "behavior_y_brain_age_interaction_v3_joint" if age_mode == "joint" else "behavior_y_brain_age_interaction_v2",
        "stimulus_type": str(stim_dir.name),
        "brain_isc_prefix": str(brain_isc_prefix),
        "behavior_isc_prefix": str(behavior_isc_prefix),
        "fisher_z_brain": bool(fisher_z_brain),
        "fisher_z_behavior": bool(fisher_z_behavior),
        "normalize_models": bool(normalize_models),
        "rank_transform": bool(rank_transform),
        "tail": str(tail).strip().lower(),
        "correction_mode": str(correction_mode).strip().lower(),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "subjects": [str(x) for x in subjects],
        "ages": [round(float(x), 8) for x in ages],
        "rois": [str(x) for x in rois],
        "brain_shape": list(np.asarray(brain_isc).shape),
        "behavior_shape": list(np.asarray(behavior_isc).shape),
    }
    if age_mode == "joint":
        signature_payload["age_model_mode"] = str(age_mode)
    analysis_sig = _analysis_signature(signature_payload)
    del brain_isc, behavior_isc
    tail_mode = str(tail).strip().lower()
    mode = str(correction_mode).strip().lower()
    pair_pos = np.full((n_sub, n_sub), -1, dtype=np.int32)
    pair_pos[iu, ju] = np.arange(int(iu.size), dtype=np.int32)
    pair_pos[ju, iu] = pair_pos[iu, ju]

    if age_mode == "joint":
        rows = []
        max_by_effect = {effect: np.full(int(n_perm), -np.inf, dtype=np.float64) for effect in JOINT_EFFECTS}
        use_perm = mode == "perm_fwer_fdr" and int(n_perm) > 0
        worker_ctx = {
            "S_z": S_z,
            "age_mat": np.stack(m_obs_z, axis=0).astype(np.float32),
            "beh_vec": beh_vec,
            "pair_pos": pair_pos,
            "iu": iu,
            "ju": ju,
            "n_sub": int(n_sub),
            "n_perm": int(n_perm) if use_perm else 0,
            "seed": int(seed),
            "tail_mode": str(tail_mode),
            "correction_mode": str(mode),
            "normalize_interaction": bool(rank_transform or normalize_models),
            "analysis_sig": str(analysis_sig),
        }
        tasks = [(int(ri), str(roi)) for ri, roi in enumerate(rois)]
        jobs = max(1, min(int(n_jobs), len(tasks)))
        if jobs == 1:
            _init_roi_worker(worker_ctx)
            results = [_compute_joint_roi_task(task) for task in tasks]
        else:
            print(f"[parallel] {stim_dir.name} joint: running {len(tasks)} ROI tasks with n_jobs={jobs}")
            results = []
            with ProcessPoolExecutor(max_workers=jobs, initializer=_init_roi_worker, initargs=(worker_ctx,)) as ex:
                futures = [ex.submit(_compute_joint_roi_task, task) for task in tasks]
                for fut in as_completed(futures):
                    results.append(fut.result())

        for result in results:
            rows.append(result["row"])
            if use_perm:
                for effect in JOINT_EFFECTS:
                    max_by_effect[effect] = np.maximum(
                        max_by_effect[effect],
                        np.asarray(result["max_stats"][effect], dtype=np.float64),
                    )
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
        out_df = _write_checkpoint(out_path, rows, model_names=("joint",))
        pd.DataFrame({"subject": subjects, "age": ages}).to_csv(stim_dir / f"{out_prefix}_subjects_sorted.csv", index=False)
        return out_df

    if age_mode != "separate":
        raise ValueError(f"Unsupported age_model_mode: {age_model_mode}")

    rows = _load_checkpoint_rows(out_path, expected_signature=analysis_sig)
    completed = {(str(r["roi"]), str(r["model"])) for r in rows if "roi" in r and "model" in r}
    if rows:
        print(f"[checkpoint] {stim_dir.name}: loaded {len(rows)} completed ROI/model rows from {out_path}")

    for mi, mname in enumerate(model_names):
        age_vec = m_obs_z[mi]
        use_perm = mode == "perm_fwer_fdr" and int(n_perm) > 0
        ckpt_npz = stim_dir / f"{out_prefix}_{mname}_permmax_checkpoint.npz"
        max_brain = np.full(int(n_perm), -np.inf, dtype=np.float64)
        max_age = np.full(int(n_perm), -np.inf, dtype=np.float64)
        max_interaction = np.full(int(n_perm), -np.inf, dtype=np.float64)
        max_done: set[str] = set()
        if use_perm and ckpt_npz.exists():
            with np.load(ckpt_npz, allow_pickle=False) as z:
                meta_ok = (
                    "analysis_signature" in z.files
                    and str(z["analysis_signature"][0]) == str(analysis_sig)
                    and int(z["n_perm"][0]) == int(n_perm)
                    and int(z["seed"][0]) == int(seed)
                    and str(z["tail"][0]) == str(tail_mode)
                )
                if bool(meta_ok):
                    max_brain = z["max_brain"].astype(np.float64, copy=True)
                    max_age = z["max_age"].astype(np.float64, copy=True)
                    max_interaction = z["max_interaction"].astype(np.float64, copy=True)
                    max_done = {str(x) for x in z["processed_rois"].astype(str).tolist()}
                else:
                    print(f"[checkpoint] {stim_dir.name}/{mname}: ignoring stale permutation checkpoint")

        def _save_permmax_checkpoint() -> None:
            if not use_perm:
                return
            tmp_npz = ckpt_npz.with_suffix(ckpt_npz.suffix + ".tmp.npz")
            np.savez_compressed(
                tmp_npz,
                max_brain=max_brain,
                max_age=max_age,
                max_interaction=max_interaction,
                processed_rois=np.array(sorted(max_done), dtype=str),
                n_perm=np.array([int(n_perm)], dtype=np.int32),
                seed=np.array([int(seed)], dtype=np.int32),
                tail=np.array([str(tail_mode)], dtype=str),
                analysis_signature=np.array([str(analysis_sig)], dtype=str),
            )
            tmp_npz.replace(ckpt_npz)

        def _merge_task_result(result: dict) -> None:
            nonlocal rows, completed, max_brain, max_age, max_interaction, max_done
            roi = str(result["roi"])
            key = (roi, str(mname))
            if bool(result.get("need_max")):
                mb = result.get("max_brain")
                ma = result.get("max_age")
                mi_arr = result.get("max_interaction")
                if mb is not None:
                    max_brain = np.maximum(max_brain, np.asarray(mb, dtype=np.float64))
                if ma is not None:
                    max_age = np.maximum(max_age, np.asarray(ma, dtype=np.float64))
                if mi_arr is not None:
                    max_interaction = np.maximum(max_interaction, np.asarray(mi_arr, dtype=np.float64))
                max_done.add(roi)
                _save_permmax_checkpoint()

            row = result.get("row")
            if row is not None:
                rows.append(row)
                completed.add(key)
                _write_checkpoint(out_path, rows, model_names=model_names)
                print(f"[checkpoint] saved {stim_dir.name} {mname} {roi} ({len(completed)}/{len(rois) * len(model_names)})")
            elif bool(result.get("need_max")):
                print(f"[checkpoint] restored maxT {stim_dir.name} {mname} {roi} ({len(max_done)}/{len(rois)})")

        tasks: list[tuple[int, str, bool, bool, bool]] = []
        for ri, roi in enumerate(rois):
            key = (str(roi), str(mname))
            need_row = key not in completed
            need_perm_p = bool(use_perm and need_row)
            need_max = bool(use_perm and str(roi) not in max_done)
            if need_row or need_perm_p or need_max:
                tasks.append((int(ri), str(roi), bool(need_row), bool(need_perm_p), bool(need_max)))

        worker_ctx = {
            "S_z": S_z,
            "age_vec": age_vec,
            "beh_vec": beh_vec,
            "pair_pos": pair_pos,
            "iu": iu,
            "ju": ju,
            "n_sub": int(n_sub),
            "n_perm": int(n_perm),
            "seed": int(seed),
            "model_index": int(mi),
            "model_name": str(mname),
            "tail_mode": str(tail_mode),
            "correction_mode": str(mode),
            "normalize_interaction": bool(rank_transform or normalize_models),
            "analysis_sig": str(analysis_sig),
        }

        if tasks:
            jobs = max(1, min(int(n_jobs), len(tasks)))
            if jobs == 1:
                _init_roi_worker(worker_ctx)
                for task in tasks:
                    _merge_task_result(_compute_roi_task(task))
            else:
                print(f"[parallel] {stim_dir.name} {mname}: running {len(tasks)} ROI tasks with n_jobs={jobs}")
                with ProcessPoolExecutor(max_workers=jobs, initializer=_init_roi_worker, initargs=(worker_ctx,)) as ex:
                    futures = [ex.submit(_compute_roi_task, task) for task in tasks]
                    for fut in as_completed(futures):
                        _merge_task_result(fut.result())

        model_done = all((str(roi), str(mname)) in completed for roi in rois)
        if bool(model_done):
            for row in rows:
                if str(row.get("model")) != str(mname):
                    continue
                row["model_complete"] = True
                if use_perm:
                    tb = float(row.get("t_brain", np.nan))
                    ta = float(row.get("t_age", np.nan))
                    ti = float(row.get("t_interaction", np.nan))
                    denom = float(int(n_perm) + 1)
                    if tail_mode == "two_sided":
                        row["p_fwer_brain_model_wise"] = (float(np.sum(max_brain[np.isfinite(max_brain)] >= abs(tb))) + 1.0) / denom if np.isfinite(tb) else np.nan
                        row["p_fwer_age_model_wise"] = (float(np.sum(max_age[np.isfinite(max_age)] >= abs(ta))) + 1.0) / denom if np.isfinite(ta) else np.nan
                        row["p_fwer_interaction_model_wise"] = (float(np.sum(max_interaction[np.isfinite(max_interaction)] >= abs(ti))) + 1.0) / denom if np.isfinite(ti) else np.nan
                    else:
                        row["p_fwer_brain_model_wise"] = (float(np.sum(max_brain[np.isfinite(max_brain)] >= tb)) + 1.0) / denom if np.isfinite(tb) else np.nan
                        row["p_fwer_age_model_wise"] = (float(np.sum(max_age[np.isfinite(max_age)] >= ta)) + 1.0) / denom if np.isfinite(ta) else np.nan
                        row["p_fwer_interaction_model_wise"] = (float(np.sum(max_interaction[np.isfinite(max_interaction)] >= ti)) + 1.0) / denom if np.isfinite(ti) else np.nan
            out_df = _write_checkpoint(out_path, rows, model_names=model_names)
            print(f"[checkpoint] finalized model: {stim_dir.name} {mname}")

    out_df = _write_checkpoint(out_path, rows, model_names=model_names)
    pd.DataFrame({"subject": subjects, "age": ages}).to_csv(stim_dir / f"{out_prefix}_subjects_sorted.csv", index=False)
    
    return out_df

def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
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
    n_jobs: int = 1,
) -> None:
    by_stim = Path(matrix_dir) / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"Cannot find directory: {by_stim}")

    brain_isc_prefix = _resolve_prefix(brain_isc_prefix, method=str(brain_isc_method), fallback_template="roi_isc_{method}_by_age")
    behavior_isc_prefix = _resolve_prefix(behavior_isc_prefix, method=str(behavior_isc_method), fallback_template="behavior_pattern_isc_{method}_by_age")
    fisher_z_brain_eff = _policy_flag(method=str(brain_isc_method), override=fisher_z_brain)
    fisher_z_behavior_eff = _policy_flag(method=str(behavior_isc_method), override=fisher_z_behavior)

    summary = []
    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        if not has_brain_files(stim_dir, brain_isc_prefix=str(brain_isc_prefix)):
            print(f"[SKIP] {stim_dir.name}: missing brain ISC files")
            continue
        if not has_behavior_files(stim_dir, behavior_isc_prefix=str(behavior_isc_prefix)):
            print(f"[SKIP] {stim_dir.name}: missing behavior ISC files")
            continue
            
        print(f"Running regression for {stim_dir.name}...")
        out = run_one(
            stim_dir=stim_dir,
            brain_isc_prefix=str(brain_isc_prefix),
            behavior_isc_prefix=str(behavior_isc_prefix),
            age_model_mode=str(age_model_mode),
            fisher_z_brain=bool(fisher_z_brain_eff),
            fisher_z_behavior=bool(fisher_z_behavior_eff),
            normalize_models=normalize_models,
            rank_transform=rank_transform,
            tail=str(tail),
            correction_mode=str(correction_mode),
            n_perm=int(n_perm),
            seed=int(seed),
            n_jobs=int(n_jobs),
        )
        
        if str(age_model_mode).strip().lower() == "joint":
            sub_out = out[out["model"] == "joint"]
            for effect in ("interaction_M_nn", "interaction_M_conv", "interaction_M_div"):
                fdr_col = f"p_fdr_perm_{effect}_model_wise"
                fwer_col = f"p_fwer_{effect}_model_wise"
                beta_col = f"beta_{effect}"
                sig_inter_perm_fdr = sub_out[sub_out[fdr_col] < 0.05] if fdr_col in sub_out.columns else sub_out.iloc[0:0]
                sig_inter_perm_fwer = sub_out[sub_out[fwer_col] < 0.05] if fwer_col in sub_out.columns else sub_out.iloc[0:0]
                summary.append({
                    "stimulus_type": stim_dir.name,
                    "model": "joint",
                    "effect": effect,
                    "n_rois": int(sub_out.shape[0]),
                    "n_sig_inter_perm_fdr": int(sig_inter_perm_fdr.shape[0]),
                    "n_sig_inter_perm_fwer": int(sig_inter_perm_fwer.shape[0]),
                    "n_sig_inter_pos_perm_fdr": int((sig_inter_perm_fdr[beta_col] > 0).sum()) if beta_col in sig_inter_perm_fdr.columns else 0,
                    "n_sig_inter_neg_perm_fdr": int((sig_inter_perm_fdr[beta_col] < 0).sum()) if beta_col in sig_inter_perm_fdr.columns else 0,
                    "n_sig_inter_pos_perm_fwer": int((sig_inter_perm_fwer[beta_col] > 0).sum()) if beta_col in sig_inter_perm_fwer.columns else 0,
                    "n_sig_inter_neg_perm_fwer": int((sig_inter_perm_fwer[beta_col] < 0).sum()) if beta_col in sig_inter_perm_fwer.columns else 0,
                })
        else:
            # Count significant interaction effects separately for each age model.
            for mname in ("M_nn", "M_conv", "M_div"):
                sub_out = out[out["model"] == mname]
                sig_inter_perm_fdr = sub_out[sub_out["p_fdr_perm_interaction_model_wise"] < 0.05]
                sig_inter_perm_fwer = sub_out[sub_out["p_fwer_interaction_model_wise"] < 0.05]

                summary.append({
                    "stimulus_type": stim_dir.name,
                    "model": mname,
                    "effect": "interaction",
                    "n_rois": int(sub_out.shape[0]),
                    "n_sig_inter_perm_fdr": int(sig_inter_perm_fdr.shape[0]),
                    "n_sig_inter_perm_fwer": int(sig_inter_perm_fwer.shape[0]),
                    "n_sig_inter_pos_perm_fdr": int((sig_inter_perm_fdr["beta_interaction"] > 0).sum()),
                    "n_sig_inter_neg_perm_fdr": int((sig_inter_perm_fdr["beta_interaction"] < 0).sum()),
                    "n_sig_inter_pos_perm_fwer": int((sig_inter_perm_fwer["beta_interaction"] > 0).sum()),
                    "n_sig_inter_neg_perm_fwer": int((sig_inter_perm_fwer["beta_interaction"] < 0).sum()),
                })

    out_sum = pd.DataFrame(summary)
    if not out_sum.empty:
        out_sum = out_sum.sort_values(["stimulus_type", "model"])
    summary_suffix = "_joint" if str(age_model_mode).strip().lower() == "joint" else ""
    summary_name = f"roi_isc_behavior_age_regression{summary_suffix}_{str(stimulus_dir_name)}_summary.csv"
    out_sum.to_csv(Path(matrix_dir) / summary_name, index=False)
    print("All done!")

def main() -> None:
    args = parse_args()
    run(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        brain_isc_method=str(args.brain_isc_method),
        brain_isc_prefix=args.brain_isc_prefix,
        behavior_isc_method=str(args.behavior_isc_method),
        behavior_isc_prefix=args.behavior_isc_prefix,
        age_model_mode=str(args.age_model_mode),
        fisher_z_brain=args.fisher_z_brain,
        fisher_z_behavior=args.fisher_z_behavior,
        normalize_models=args.normalize_models,
        rank_transform=args.rank_transform,
        tail=str(args.tail),
        correction_mode=str(args.correction_mode),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        n_jobs=int(args.n_jobs),
    )

if __name__ == "__main__":
    main()
