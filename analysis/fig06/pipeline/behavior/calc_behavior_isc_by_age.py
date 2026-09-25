#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

if __package__ in {None, "", "behavior"}:
    THIS_DIR = Path(__file__).resolve().parent
    EMO_DIR = THIS_DIR.parent
    if str(EMO_DIR) not in sys.path:
        sys.path.insert(0, str(EMO_DIR))

    from calc_roi_isc_by_age import (  # noqa: E402
        distance_to_similarity,
        flatten_upper,
        load_subject_ages_map,
        pairwise_euclidean_distances,
        pairwise_mahalanobis_distances,
        pearson_corr_matrix_rows,
        sort_subjects_by_age,
        spearman_corr_matrix_rows,
    )
else:
    from ..calc_roi_isc_by_age import (  # noqa: E402
        distance_to_similarity,
        flatten_upper,
        load_subject_ages_map,
        pairwise_euclidean_distances,
        pairwise_mahalanobis_distances,
        pearson_corr_matrix_rows,
        sort_subjects_by_age,
        spearman_corr_matrix_rows,
    )

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute subject-by-subject behavior similarity matrices sorted by age")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--subject-info", type=Path, required=True)
    p.add_argument("--repr-prefix", type=str, default="behavior_repr_matrix_trial")
    p.add_argument("--input-kind", type=str, default="repr", choices=("repr", "pattern"))
    p.add_argument("--isc-method", type=str, default="mahalanobis", choices=("spearman", "pearson", "euclidean", "mahalanobis"))
    p.add_argument("--isc-prefix", type=str, default=None)
    p.add_argument("--max-missing-fraction", type=float, default=0.2)
    return p.parse_args()


def has_repr_files(stim_dir: Path, repr_prefix: str) -> bool:
    required = (
        stim_dir / f"{repr_prefix}.npz",
        stim_dir / f"{repr_prefix}_subjects.csv",
    )
    return all(p.exists() for p in required)


def compute_behavior_isc(X: np.ndarray, method: str) -> np.ndarray:
    m = str(method).strip().lower()
    X = np.asarray(X, dtype=np.float32)
    if m == "spearman":
        return spearman_corr_matrix_rows(X)
    if m == "pearson":
        return pearson_corr_matrix_rows(X)
    if m == "euclidean":
        dist = pairwise_euclidean_distances(X)
        return distance_to_similarity(dist)
    if m == "mahalanobis":
        dist = pairwise_mahalanobis_distances(X)
        return distance_to_similarity(dist)
    raise ValueError(f"Unknown behavior isc-method: {method}")


def vectorize_subject_input(arr: np.ndarray, input_kind: str) -> np.ndarray:
    kind = str(input_kind).strip().lower()
    arr = np.asarray(arr, dtype=np.float32)
    if kind == "repr":
        return flatten_upper(arr)
    if kind == "pattern":
        return arr.reshape(-1).astype(np.float32, copy=False)
    raise ValueError(f"Unknown behavior input-kind: {input_kind}")


def run_one_stimulus(
    stim_dir: Path,
    subject_info: Path,
    repr_prefix: str,
    isc_method: str,
    isc_prefix: str,
    max_missing_fraction: float | None,
    input_kind: str = "repr",
) -> Dict[str, object]:
    npz = np.load(stim_dir / f"{repr_prefix}.npz")
    subjects = pd.read_csv(stim_dir / f"{repr_prefix}_subjects.csv")["subject"].astype(str).tolist()
    age_map = load_subject_ages_map(subject_info)
    subjects_sorted, ages_sorted = sort_subjects_by_age(subjects, age_map)

    vecs = np.stack(
        [vectorize_subject_input(np.asarray(npz[s], dtype=np.float32), input_kind=input_kind) for s in subjects_sorted],
        axis=0,
    )
    missing_frac = np.mean(~np.isfinite(vecs), axis=1).astype(float)
    keep_mask = np.ones(len(subjects_sorted), dtype=bool)
    threshold_label = ""
    if max_missing_fraction is not None:
        keep_mask = missing_frac < float(max_missing_fraction)
        threshold_label = f"missing_fraction_repr < {float(max_missing_fraction):.6g}"

    filter_df = pd.DataFrame(
        {
            "subject": subjects_sorted,
            "age": ages_sorted,
            "input_kind": str(input_kind),
            "missing_fraction_repr": missing_frac,
            "missing_fraction_input": missing_frac,
            "keep_for_behavior_isc": keep_mask.astype(bool),
        }
    )
    if threshold_label:
        filter_df["filter_rule"] = threshold_label
    filter_df.to_csv(stim_dir / f"{isc_prefix}_subject_filter.csv", index=False)

    subjects_kept = [s for s, keep in zip(subjects_sorted, keep_mask) if bool(keep)]
    ages_kept = [a for a, keep in zip(ages_sorted, keep_mask) if bool(keep)]
    vecs_kept = vecs[keep_mask]
    missing_frac_kept = missing_frac[keep_mask]
    n_excluded = int((~keep_mask).sum())
    if vecs_kept.shape[0] < 3:
        raise ValueError(
            f"Fewer than 3 subjects remain after missing-fraction filtering for {stim_dir.name} "
            f"(kept={vecs_kept.shape[0]}, excluded={n_excluded}, threshold={max_missing_fraction})"
        )

    col_mean = np.nanmean(vecs_kept, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    vecs_filled = np.where(np.isfinite(vecs_kept), vecs_kept, col_mean.reshape(1, -1))
    isc = compute_behavior_isc(vecs_filled, method=str(isc_method))
    np.save(stim_dir / f"{isc_prefix}.npy", isc.astype(np.float32))
    pd.DataFrame(
        {
            "subject": subjects_kept,
            "age": ages_kept,
            "input_kind": str(input_kind),
            "missing_fraction_repr": missing_frac_kept,
            "missing_fraction_input": missing_frac_kept,
        }
    ).to_csv(
        stim_dir / f"{isc_prefix}_subjects_sorted.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "isc_method": [str(isc_method)],
            "repr_prefix": [str(repr_prefix)],
            "input_kind": [str(input_kind)],
            "isc_prefix": [str(isc_prefix)],
            "missing_fill": ["column_mean"],
            "max_missing_fraction": [float(max_missing_fraction) if max_missing_fraction is not None else np.nan],
            "n_subjects_before_filter": [int(len(subjects_sorted))],
            "n_subjects_after_filter": [int(len(subjects_kept))],
            "n_subjects_excluded_missing": [int(n_excluded)],
            "n_input_features": [int(vecs.shape[1])],
        }
    ).to_csv(
        stim_dir / f"{isc_prefix}_meta.csv",
        index=False,
    )
    return {
        "stimulus_type": stim_dir.name,
        "n_subjects_before_filter": int(len(subjects_sorted)),
        "n_subjects_after_filter": int(len(subjects_kept)),
        "n_subjects_excluded_missing": int(n_excluded),
        "input_kind": str(input_kind),
        "n_input_features": int(vecs.shape[1]),
        "isc_method": str(isc_method),
        "isc_prefix": str(isc_prefix),
        "max_missing_fraction": float(max_missing_fraction) if max_missing_fraction is not None else np.nan,
    }


def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
    subject_info: Path,
    repr_prefix: str,
    isc_method: str,
    isc_prefix: str,
    max_missing_fraction: float | None,
    input_kind: str = "repr",
) -> None:
    by_stim = Path(matrix_dir) / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"Cannot find directory: {by_stim}")

    rows = []
    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        if not has_repr_files(stim_dir, repr_prefix=str(repr_prefix)):
            print(f"[SKIP] {stim_dir.name}: missing {repr_prefix} input files")
            continue
        rows.append(
            run_one_stimulus(
                stim_dir,
                subject_info,
                repr_prefix,
                isc_method=str(isc_method),
                isc_prefix=str(isc_prefix),
                max_missing_fraction=max_missing_fraction,
                input_kind=str(input_kind),
            )
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("stimulus_type")
    out.to_csv(Path(matrix_dir) / f"{isc_prefix}_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    isc_prefix = str(args.isc_prefix) if args.isc_prefix is not None else f"behavior_isc_{str(args.isc_method)}_by_age"
    run(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        subject_info=Path(args.subject_info),
        repr_prefix=str(args.repr_prefix),
        isc_method=str(args.isc_method),
        isc_prefix=str(isc_prefix),
        max_missing_fraction=float(args.max_missing_fraction) if args.max_missing_fraction is not None else None,
        input_kind=str(args.input_kind),
    )


if __name__ == "__main__":
    main()
