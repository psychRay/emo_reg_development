#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_roi_isc_by_age.py

Step 2:
From the Step 1 ROI representational matrices (subject x stimulus x stimulus),
compute a subject-by-subject similarity matrix for each ROI and order subjects by age.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute age-ordered ROI inter-subject similarity from ROI representational matrices")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--subject-info", type=Path, default=Path(os.environ.get("SUBJECT_AGE_TABLE", "/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv")))
    p.add_argument("--repr-prefix", type=str, default="roi_repr_matrix_200")
    p.add_argument("--isc-method", type=str, default="mahalanobis", choices=("spearman", "pearson", "euclidean", "mahalanobis"))
    p.add_argument("--isc-prefix", type=str, default=None)
    p.add_argument(
        "--cov-estimator",
        type=str,
        default="ridge_pinv",
        choices=("ridge_pinv", "ledoit_wolf"),
        help="Covariance estimator for Mahalanobis inter-subject similarity. ridge_pinv is the default and preserves previous results.",
    )
    p.add_argument(
        "--ridge",
        type=float,
        default=1e-6,
        help="Diagonal ridge for the Mahalanobis covariance. The default 1e-6 preserves previous results.",
    )
    return p.parse_args()


def has_repr_files(stim_dir: Path, repr_prefix: str) -> bool:
    # Minimal input contract for one stimulus_type directory.
    required = (
        stim_dir / f"{repr_prefix}.npz",
        stim_dir / f"{repr_prefix}_subjects.csv",
        stim_dir / f"{repr_prefix}_rois.csv",
    )
    return all(p.exists() for p in required)


def parse_chinese_age(age_str: object) -> float:
    # Parse numeric ages and year/month/day phrases stored in the age table.
    if pd.isna(age_str) or str(age_str).strip() == "":
        return 999.0
    if isinstance(age_str, (int, float, np.integer, np.floating)):
        return float(age_str)
    s = str(age_str).strip()
    try:
        return float(s)
    except Exception:
        pass
    y_match = re.search(r"(\d+)\s*岁", s)
    m_match = re.search(r"(\d+)\s*个月", s) or re.search(r"(\d+)\s*月", s)
    d_match = re.search(r"(\d+)\s*天", s)
    years = int(y_match.group(1)) if y_match else 0
    months = int(m_match.group(1)) if m_match else 0
    days = int(d_match.group(1)) if d_match else 0
    return float(round(years + months / 12.0 + days / 365.0, 4))


def load_subject_ages_map(subject_info_path: Path) -> Dict[str, float]:
    # Load subject->age mapping from CSV/TSV/XLSX with bilingual column name support.
    path_str = str(subject_info_path)
    if path_str.endswith(".tsv") or path_str.endswith(".txt"):
        info_df = pd.read_csv(path_str, sep="\t")
    elif path_str.endswith(".xlsx") or path_str.endswith(".xls"):
        info_df = pd.read_excel(path_str)
    elif path_str.endswith(".csv"):
        info_df = pd.read_csv(path_str)
    else:
        raise ValueError("Unsupported age-table format")

    if "sub_id" in info_df.columns and "age" in info_df.columns:
        sub_col, age_col = "sub_id", "age"
    elif "被试编号" in info_df.columns and "采集年龄" in info_df.columns:
        sub_col, age_col = "被试编号", "采集年龄"
    else:
        raise ValueError(f"Age table is missing required columns: {list(info_df.columns)}")

    age_map: Dict[str, float] = {}
    for _, row in info_df.iterrows():
        raw = str(row[sub_col]).strip()
        if raw.endswith(".0"):
            raw = raw[:-2]
        sid = raw if raw.startswith("sub-") else f"sub-{raw}"
        age_map[sid] = parse_chinese_age(row[age_col])
    return age_map


def sort_subjects_by_age(subjects: List[str], age_map: Dict[str, float]) -> Tuple[List[str], List[float]]:
    # Stable sort by age, then by subject id; missing ages are pushed to the end (age=999).
    def key(s: str) -> Tuple[float, str]:
        a = age_map.get(s, np.nan)
        if not np.isfinite(a):
            return 999.0, s
        return float(a), s

    ss = sorted(subjects, key=key)
    aa = [float(age_map.get(s, 999.0)) for s in ss]
    return ss, aa


def spearman_corr_matrix_rows(X: np.ndarray) -> np.ndarray:
    ranks = np.apply_along_axis(rankdata, 1, X)
    mean = ranks.mean(axis=1, keepdims=True)
    std = ranks.std(axis=1, keepdims=True, ddof=1)
    std[std == 0] = np.nan
    Z = (ranks - mean) / std
    p = X.shape[1]
    corr = (Z @ Z.T) / float(max(1, p - 1))
    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def pearson_corr_matrix_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True, ddof=1)
    std[std == 0] = np.nan
    Z = (X - mean) / std
    p = X.shape[1]
    corr = (Z @ Z.T) / float(max(1, p - 1))
    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def fisher_z_transform(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # Fisher Z transform for correlation-like values; used before distance-based ISC.
    X = np.asarray(X, dtype=np.float64)
    X = np.clip(X, -1.0 + eps, 1.0 - eps)
    return np.arctanh(X)


def pairwise_euclidean_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    diff = X[:, None, :] - X[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.maximum(dist2, 0.0, out=dist2)
    return np.sqrt(dist2, dtype=np.float64)


def pairwise_mahalanobis_distances(
    X: np.ndarray,
    ridge: float = 1e-6,
    cov_estimator: str = "ridge_pinv",
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    dist = np.full((n_samples, n_samples), np.nan, dtype=np.float64)
    valid_rows = np.all(np.isfinite(X), axis=1)
    if valid_rows.sum() == 0:
        return dist

    Xv = X[valid_rows]
    est = str(cov_estimator).strip().lower()
    if est not in {"ridge_pinv", "ledoit_wolf"}:
        raise ValueError(f"Unsupported cov_estimator: {cov_estimator}")

    if est == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf  # type: ignore

            cov = np.asarray(LedoitWolf().fit(Xv).covariance_, dtype=np.float64)
        except Exception as e:
            print(f"[warn] Ledoit-Wolf is unavailable; falling back to ridge_pinv ({e})")
            cov = np.cov(Xv, rowvar=False)
    else:
        cov = np.cov(Xv, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    cov = cov + float(ridge) * np.eye(n_features, dtype=np.float64)
    inv_cov = np.linalg.pinv(cov)
    diff = Xv[:, None, :] - Xv[None, :, :]
    left = np.einsum("...i,ij->...j", diff, inv_cov)
    dist2 = np.einsum("...i,...i->...", left, diff)
    np.maximum(dist2, 0.0, out=dist2)
    dist_valid = np.sqrt(dist2, dtype=np.float64)
    dist[np.ix_(valid_rows, valid_rows)] = dist_valid
    return dist


def distance_to_similarity(dist: np.ndarray) -> np.ndarray:
    # Convert a distance matrix into a similarity matrix (negate + z-score off-diagonal).
    sim = -np.asarray(dist, dtype=np.float64)
    n = sim.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = sim[mask]
    finite_vals = vals[np.isfinite(vals)]
    mean = finite_vals.mean() if finite_vals.size > 0 else 0.0
    std = finite_vals.std(ddof=1) if finite_vals.size > 1 else 0.0
    if std > 0:
        sim = (sim - mean) / std
    else:
        sim = sim - mean
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)


def compute_isc(
    X: np.ndarray,
    method: str,
    cov_estimator: str = "ridge_pinv",
    ridge: float = 1e-6,
) -> np.ndarray:
    # X: (n_subjects, n_features) where features are flattened upper-tri of the RSM.
    m = str(method).strip().lower()
    if m == "spearman":
        return spearman_corr_matrix_rows(X)
    if m == "pearson":
        return pearson_corr_matrix_rows(X)
    if m == "euclidean":
        Xz = fisher_z_transform(X)
        dist = pairwise_euclidean_distances(Xz)
        return distance_to_similarity(dist)
    if m == "mahalanobis":
        Xz = fisher_z_transform(X)
        dist = pairwise_mahalanobis_distances(Xz, ridge=float(ridge), cov_estimator=str(cov_estimator))
        return distance_to_similarity(dist)
    raise ValueError(f"Unknown isc-method: {method}")


def flatten_upper(rsm: np.ndarray) -> np.ndarray:
    # Vectorize the strict upper triangle of an RSM (excluding diagonal).
    iu = np.triu_indices(rsm.shape[0], k=1)
    return rsm[iu].astype(np.float32)


def run_one_stimulus(
    stim_dir: Path,
    subject_info: Path,
    repr_prefix: str,
    isc_method: str,
    isc_prefix: str,
    cov_estimator: str = "ridge_pinv",
    ridge: float = 1e-6,
) -> Dict[str, object]:
    npz_path = stim_dir / f"{repr_prefix}.npz"
    subjects_path = stim_dir / f"{repr_prefix}_subjects.csv"
    rois_path = stim_dir / f"{repr_prefix}_rois.csv"

    npz = np.load(npz_path)
    subjects = pd.read_csv(subjects_path)["subject"].astype(str).tolist()
    rois = pd.read_csv(rois_path)["roi"].astype(str).tolist()

    age_map = load_subject_ages_map(subject_info)
    subjects_sorted, ages_sorted = sort_subjects_by_age(subjects, age_map)
    idx_map = {s: i for i, s in enumerate(subjects)}
    order = [idx_map[s] for s in subjects_sorted]

    n_sub = len(subjects_sorted)
    isc = np.zeros((len(rois), n_sub, n_sub), dtype=np.float32)
    for ri, roi in enumerate(rois):
        arr = np.asarray(npz[roi], dtype=np.float32)
        arr = arr[order, :, :]
        # Per subject: flatten RSM -> vector; across subjects: compute ISC on vectors.
        vecs = np.stack([flatten_upper(arr[i]) for i in range(n_sub)], axis=0)
        isc[ri] = compute_isc(vecs, method=str(isc_method), cov_estimator=str(cov_estimator), ridge=float(ridge))

    out_prefix = str(isc_prefix)
    np.save(stim_dir / f"{out_prefix}.npy", isc)
    pd.DataFrame({"subject": subjects_sorted, "age": ages_sorted}).to_csv(stim_dir / f"{out_prefix}_subjects_sorted.csv", index=False)
    pd.DataFrame({"roi": rois}).to_csv(stim_dir / f"{out_prefix}_rois.csv", index=False)
    pd.DataFrame({"isc_method": [str(isc_method)], "repr_prefix": [str(repr_prefix)], "isc_prefix": [str(out_prefix)]}).to_csv(
        stim_dir / f"{out_prefix}_meta.csv",
        index=False,
    )
    return {"stimulus_type": stim_dir.name, "n_subjects": n_sub, "n_rois": len(rois), "isc_method": str(isc_method), "isc_prefix": str(out_prefix)}


def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
    subject_info: Path,
    repr_prefix: str,
    isc_method: str,
    isc_prefix: str,
    cov_estimator: str = "ridge_pinv",
    ridge: float = 1e-6,
) -> None:
    by_stim = matrix_dir / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"Directory not found: {by_stim}")

    rows = []
    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        if not has_repr_files(stim_dir, repr_prefix=str(repr_prefix)):
            print(f"[SKIP] {stim_dir.name}: missing inputs for {repr_prefix}; skipped.")
            continue
        rows.append(
            run_one_stimulus(
                stim_dir,
                subject_info,
                repr_prefix,
                isc_method=str(isc_method),
                isc_prefix=str(isc_prefix),
                cov_estimator=str(cov_estimator),
                ridge=float(ridge),
            )
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("stimulus_type")
    out.to_csv(matrix_dir / f"{isc_prefix}_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    isc_prefix = str(args.isc_prefix) if args.isc_prefix is not None else f"roi_isc_{str(args.isc_method)}_by_age"
    run(
        args.matrix_dir,
        args.stimulus_dir_name,
        args.subject_info,
        args.repr_prefix,
        isc_method=str(args.isc_method),
        isc_prefix=str(isc_prefix),
        cov_estimator=str(args.cov_estimator),
        ridge=float(args.ridge),
    )


if __name__ == "__main__":
    main()
