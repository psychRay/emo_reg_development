#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print and validate brain/behavior ISC subject alignment")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--brain-isc-prefix", type=str, default="roi_isc_mahalanobis_by_age")
    p.add_argument("--behavior-isc-prefix", type=str, default="behavior_pattern_isc_mahalanobis_by_age")
    p.add_argument("--conditions", nargs="*", default=None, help="Condition directory names. Default: all conditions.")
    p.add_argument("--mode", type=str, default="both", choices=("joint", "age_regression", "both"))
    p.add_argument("--print-subjects", type=int, default=20, help="Number of aligned subjects to print per condition.")
    p.add_argument("--print-all", action="store_true", help="Print all aligned subjects.")
    p.add_argument("--verbose", action="store_true", help="Print aligned subject preview to terminal.")
    p.add_argument("--save-csv", action="store_true", help="Save alignment table into each condition directory.")
    return p.parse_args()


def _read_subjects(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "subject" not in df.columns:
        raise ValueError(f"{path} missing required column: subject")
    df["subject"] = df["subject"].astype(str)
    return df


def _check_duplicates(subjects: List[str], label: str) -> None:
    dup = sorted({s for s in subjects if subjects.count(s) > 1})
    if dup:
        raise ValueError(f"Duplicate subjects in {label}: {dup}")


def _validate_shapes(
    brain_isc: np.ndarray,
    brain_subjects: List[str],
    behavior_isc: np.ndarray,
    behavior_subjects: List[str],
) -> None:
    b = np.asarray(brain_isc)
    h = np.asarray(behavior_isc)
    n_brain = len(brain_subjects)
    n_beh = len(behavior_subjects)
    if b.ndim != 3 or b.shape[1] != n_brain or b.shape[2] != n_brain:
        raise ValueError(f"Brain ISC shape {b.shape} does not match {n_brain} brain subjects")
    if h.ndim != 2 or h.shape[0] != n_beh or h.shape[1] != n_beh:
        raise ValueError(f"Behavior ISC shape {h.shape} does not match {n_beh} behavior subjects")


def _age_map(brain_df: pd.DataFrame, beh_df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "age" in brain_df.columns:
        out.update(dict(zip(brain_df["subject"].astype(str), brain_df["age"].astype(float))))
    if "age" in beh_df.columns:
        for sub, age in zip(beh_df["subject"].astype(str), beh_df["age"].astype(float)):
            cur = out.get(str(sub), np.nan)
            if not np.isfinite(cur):
                out[str(sub)] = float(age)
    return out


def _alignment_rows(
    brain_subjects: List[str],
    behavior_subjects: List[str],
    brain_df: pd.DataFrame,
    beh_df: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    brain_idx = {s: i for i, s in enumerate(brain_subjects)}
    behavior_idx = {s: i for i, s in enumerate(behavior_subjects)}
    behavior_set = set(behavior_subjects)
    age_lookup = _age_map(brain_df, beh_df)

    rows = []
    for sub in brain_subjects:
        if sub not in behavior_set:
            continue
        age = float(age_lookup.get(sub, np.nan))
        valid_age = bool(np.isfinite(age) and age < 998.0)
        if mode == "age_regression" and not valid_age:
            continue
        brain_original_index = int(brain_idx[sub])
        behavior_original_index = int(behavior_idx[sub])
        brain_aligned_subject = brain_subjects[brain_original_index]
        behavior_aligned_subject = behavior_subjects[behavior_original_index]
        rows.append(
            {
                "aligned_order": len(rows),
                "brain_aligned_subject": brain_aligned_subject,
                "behavior_aligned_subject": behavior_aligned_subject,
                "subject_order_match": bool(brain_aligned_subject == behavior_aligned_subject),
                "subject": sub,
                "brain_original_index": brain_original_index,
                "behavior_original_index": behavior_original_index,
                "age": age,
                "valid_age_for_age_regression": valid_age,
            }
        )
    return pd.DataFrame(rows)


def _condition_dirs(root: Path, requested: Optional[List[str]]) -> List[Path]:
    if requested:
        out = [root / str(x) for x in requested]
    else:
        out = sorted([p for p in root.iterdir() if p.is_dir()])
    missing = [p for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing condition directories: {[str(p) for p in missing]}")
    return out


def check_one(
    stim_dir: Path,
    brain_prefix: str,
    behavior_prefix: str,
    mode: str,
    print_n: int,
    print_all: bool,
    verbose: bool,
    save_csv: bool,
) -> dict:
    brain_npy = stim_dir / f"{brain_prefix}.npy"
    brain_sub_path = stim_dir / f"{brain_prefix}_subjects_sorted.csv"
    beh_npy = stim_dir / f"{behavior_prefix}.npy"
    beh_sub_path = stim_dir / f"{behavior_prefix}_subjects_sorted.csv"
    for path in (brain_npy, brain_sub_path, beh_npy, beh_sub_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    brain_isc = np.load(brain_npy)
    behavior_isc = np.load(beh_npy)
    brain_df = _read_subjects(brain_sub_path)
    beh_df = _read_subjects(beh_sub_path)
    brain_subjects = brain_df["subject"].astype(str).tolist()
    behavior_subjects = beh_df["subject"].astype(str).tolist()

    _check_duplicates(brain_subjects, f"{brain_sub_path}")
    _check_duplicates(behavior_subjects, f"{beh_sub_path}")
    _validate_shapes(brain_isc, brain_subjects, behavior_isc, behavior_subjects)

    rows = _alignment_rows(brain_subjects, behavior_subjects, brain_df, beh_df, mode=mode)
    if rows.shape[0] < 3:
        raise ValueError(f"{stim_dir.name}: fewer than 3 aligned subjects for mode={mode}")
    if not rows["subject_order_match"].astype(bool).all():
        bad = rows.loc[~rows["subject_order_match"].astype(bool)].head(10)
        raise ValueError(f"{stim_dir.name}: aligned brain/behavior subject order mismatch:\n{bad.to_string(index=False)}")

    brain_aligned_shape = (int(brain_isc.shape[0]), int(rows.shape[0]), int(rows.shape[0]))
    behavior_aligned_shape = (int(rows.shape[0]), int(rows.shape[0]))
    n_brain_only = len(set(brain_subjects) - set(behavior_subjects))
    n_behavior_only = len(set(behavior_subjects) - set(brain_subjects))
    raw_order_exact = bool(brain_subjects == behavior_subjects)

    status = {
        "condition": stim_dir.name,
        "mode": str(mode),
        "status": "PASS",
        "brain_raw_shape": str(tuple(brain_isc.shape)),
        "behavior_raw_shape": str(tuple(behavior_isc.shape)),
        "n_brain_subjects": int(len(brain_subjects)),
        "n_behavior_subjects": int(len(behavior_subjects)),
        "raw_subject_order_exact": bool(raw_order_exact),
        "n_brain_only": int(n_brain_only),
        "n_behavior_only": int(n_behavior_only),
        "n_aligned": int(rows.shape[0]),
        "aligned_brain_shape": str(brain_aligned_shape),
        "aligned_behavior_shape": str(behavior_aligned_shape),
        "aligned_subject_order_exact": True,
    }
    if bool(save_csv):
        out_name = f"brain_behavior_alignment_{mode}_{behavior_prefix}.csv"
        rows.to_csv(stim_dir / out_name, index=False)
        status["detail_csv"] = str(stim_dir / out_name)

    if bool(verbose):
        print("=" * 100)
        for k, v in status.items():
            print(f"{k}: {v}")
        print("alignment rule: overlap subjects in brain subject order, then index behavior matrix by matching subject IDs")
        if mode == "age_regression":
            print("age_regression rule: additionally keep only aligned subjects with finite age < 998")
        n_show = rows.shape[0] if bool(print_all) else min(int(print_n), rows.shape[0])
        print(f"\naligned subjects preview ({n_show}/{rows.shape[0]}):")
        print(rows.head(n_show).to_string(index=False))

    return status


def main() -> None:
    args = parse_args()
    root = Path(args.matrix_dir) / str(args.stimulus_dir_name)
    if not root.exists():
        raise FileNotFoundError(f"Cannot find stimulus root: {root}")

    modes = ["joint", "age_regression"] if str(args.mode) == "both" else [str(args.mode)]
    summary = []
    for stim_dir in _condition_dirs(root, args.conditions):
        for mode in modes:
            try:
                row = check_one(
                    stim_dir=stim_dir,
                    brain_prefix=str(args.brain_isc_prefix),
                    behavior_prefix=str(args.behavior_isc_prefix),
                    mode=str(mode),
                    print_n=int(args.print_subjects),
                    print_all=bool(args.print_all),
                    verbose=bool(args.verbose),
                    save_csv=bool(args.save_csv),
                )
            except Exception as e:
                row = {"condition": stim_dir.name, "mode": str(mode), "status": "FAIL", "error": str(e)}
            summary.append(row)

    summary_df = pd.DataFrame(summary)
    cols = [
        "status",
        "condition",
        "mode",
        "n_brain_subjects",
        "n_behavior_subjects",
        "n_brain_only",
        "n_behavior_only",
        "n_aligned",
        "raw_subject_order_exact",
        "aligned_subject_order_exact",
        "error",
        "detail_csv",
    ]
    show_cols = [c for c in cols if c in summary_df.columns]
    print(summary_df.loc[:, show_cols].to_string(index=False))
    n_fail = int((summary_df["status"].astype(str) == "FAIL").sum()) if not summary_df.empty else 0
    if n_fail:
        raise SystemExit(f"Alignment check FAILED for {n_fail} condition/mode rows")
    print(f"\nAlignment check PASS: {summary_df.shape[0]} condition/mode rows checked.")


if __name__ == "__main__":
    main()
