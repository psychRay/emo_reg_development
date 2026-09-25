#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_roi_emotion_repr_matrix.py

Build emotion-level ROI representational similarity matrices from the
ROI patterns exported in Step 1 (roi_beta_patterns_*.npz).

Within each stimulus type, stimuli are aggregated by emotion keyword and an
emotion-by-emotion ROI RSM is computed. Results are written to a separate
stimulus directory (default by_emotion/<stimulus_type>) and do not overwrite
the trial-level by_stimulus products.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
DEFAULT_EMOTIONS = ("Anger", "Disgust", "Fear", "Sad")


def save_npz_named(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(path), mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, arr in arrays.items():
            # Store each array as an individual .npy entry to keep subject keys intact.
            k = str(name).replace("/", "_").replace("\\", "_")
            bio = io.BytesIO()
            np.save(bio, np.asarray(arr), allow_pickle=False)
            zf.writestr(f"{k}.npy", bio.getvalue())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate ROI patterns into emotion-level RSMs")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--roi-set", type=int, default=232, choices=(200, 232))
    p.add_argument("--pattern-prefix", type=str, default=None)
    p.add_argument("--emotions", nargs="+", default=list(DEFAULT_EMOTIONS))
    p.add_argument("--out-stimulus-dir-name", type=str, default="by_emotion")
    p.add_argument("--out-prefix", type=str, default=None)
    p.add_argument("--rsm-method", type=str, default="spearman", choices=("spearman", "pearson"))
    p.add_argument("--no-cocktail-blank", action="store_false", dest="cocktail_blank", default=True)
    p.add_argument("--no-save-pattern", action="store_false", dest="save_pattern", default=True)
    return p.parse_args()


def spearman_rsm(X: np.ndarray, cocktail_blank: bool) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if bool(cocktail_blank):
        # Cocktail-blank: remove feature-wise mean across conditions to reduce global mean drive.
        X = X - X.mean(axis=0, keepdims=True)
    ranks = np.apply_along_axis(rankdata, 1, X)
    mean = ranks.mean(axis=1, keepdims=True)
    std = ranks.std(axis=1, keepdims=True, ddof=1)
    std[std == 0] = np.nan
    Z = (ranks - mean) / std
    p = ranks.shape[1]
    C = (Z @ Z.T) / float(max(1, p - 1))
    return np.clip(C, -1.0, 1.0).astype(np.float32)


def pearson_rsm(X: np.ndarray, cocktail_blank: bool) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if bool(cocktail_blank):
        # Cocktail-blank: remove feature-wise mean across conditions to reduce global mean drive.
        X = X - X.mean(axis=0, keepdims=True)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True, ddof=1)
    std[std == 0] = np.nan
    Z = (X - mean) / std
    p = X.shape[1]
    C = (Z @ Z.T) / float(max(1, p - 1))
    return np.clip(C, -1.0, 1.0).astype(np.float32)


def compute_rsm(X: np.ndarray, cocktail_blank: bool, method: str) -> np.ndarray:
    m = str(method).strip().lower()
    if m == "spearman":
        return spearman_rsm(X, cocktail_blank=bool(cocktail_blank))
    if m == "pearson":
        return pearson_rsm(X, cocktail_blank=bool(cocktail_blank))
    raise ValueError(f"Unknown rsm-method: {method}")


def detect_emotion(stimulus_name: str, emotions: List[str]) -> str:
    s = str(stimulus_name).lower()
    for emo in emotions:
        if str(emo).lower() in s:
            return str(emo)
    return ""


def run_one(
    stim_dir: Path,
    pattern_prefix: str,
    emotions: List[str],
    out_dir: Path,
    out_prefix: str,
    rsm_method: str,
    cocktail_blank: bool,
    save_pattern: bool,
    roi_set: int,
) -> Optional[dict]:
    order_path = stim_dir / "stimulus_order.csv"
    if not order_path.exists():
        raise FileNotFoundError(f"Missing file: {order_path}")
    stim_order = pd.read_csv(order_path)["stimulus_order"].astype(str).tolist()

    # Detect emotion label from stimulus name by substring matching.
    emo_labels = [detect_emotion(s, emotions) for s in stim_order]
    keep_idx = [i for i, e in enumerate(emo_labels) if e != ""]
    if not keep_idx:
        return None
    stim_order_kept = [stim_order[i] for i in keep_idx]
    emo_labels_kept = [emo_labels[i] for i in keep_idx]

    emo_idx: Dict[str, np.ndarray] = {}
    for emo in emotions:
        idx = np.where(np.array(emo_labels_kept, dtype=object) == emo)[0]
        if int(idx.size) == 0:
            # Skip this stimulus_type if it cannot form a complete fixed-size emotion matrix.
            return None
        emo_idx[str(emo)] = idx

    npz_path = stim_dir / f"{pattern_prefix}.npz"
    subs_path = stim_dir / f"{pattern_prefix}_subjects.csv"
    rois_path = stim_dir / f"{pattern_prefix}_rois.csv"
    sizes_path = stim_dir / f"{pattern_prefix}_roi_feat_sizes.csv"
    for p in (npz_path, subs_path, rois_path, sizes_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    npz = np.load(npz_path)
    subjects = pd.read_csv(subs_path)["subject"].astype(str).tolist()
    rois = pd.read_csv(rois_path)["roi"].astype(str).tolist()
    roi_sizes = pd.read_csv(sizes_path)
    if roi_sizes.shape[0] != len(rois):
        raise ValueError("roi_feat_sizes row count does not match rois")
    feat_sizes = roi_sizes["n_feat"].astype(int).tolist()
    max_feat = int(max(feat_sizes)) if feat_sizes else 0

    n_sub = len(subjects)
    n_emo = len(emotions)
    out_roi: Dict[str, np.ndarray] = {r: np.zeros((n_sub, n_emo, n_emo), dtype=np.float32) for r in rois}
    out_pat_by_sub: Dict[str, np.ndarray] = {}

    for si, sub in enumerate(subjects):
        if str(sub) not in npz:
            raise KeyError(f"Pattern npz is missing subject key: {sub}")
        arr = np.asarray(npz[str(sub)], dtype=np.float32)
        if arr.shape[0] != len(rois) or arr.shape[1] != len(stim_order):
            raise ValueError(f"Pattern dimensions do not match: {sub} {arr.shape}")
        arr = arr[:, keep_idx, :]
        # Aggregate trial-level patterns into emotion-level patterns (mean across stimuli in each emotion).
        pat_emo = np.full((len(rois), n_emo, int(max_feat)), np.nan, dtype=np.float32)
        for ei, emo in enumerate(emotions):
            idx = emo_idx[str(emo)]
            pat_emo[:, ei, :] = np.nanmean(arr[:, idx, :], axis=1).astype(np.float32)
        if bool(save_pattern):
            out_pat_by_sub[str(sub)] = pat_emo
        for ri, roi in enumerate(rois):
            nf = int(feat_sizes[ri])
            X = pat_emo[ri, :, :nf]
            # Recompute emotion-level RSM from aggregated patterns (not block-averaging an existing RSM).
            out_roi[str(roi)][si] = compute_rsm(X, cocktail_blank=bool(cocktail_blank), method=str(rsm_method))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{out_prefix}.npz", **out_roi)
    pd.DataFrame({"roi": rois}).to_csv(out_dir / f"{out_prefix}_rois.csv", index=False)
    pd.DataFrame({"subject": subjects}).to_csv(out_dir / f"{out_prefix}_subjects.csv", index=False)
    pd.DataFrame({"stimulus_order": emotions}).to_csv(out_dir / "stimulus_order.csv", index=False)
    pd.DataFrame({"emotion_order": emotions}).to_csv(out_dir / "emotion_order.csv", index=False)
    pd.DataFrame({"stimulus_order": stim_order_kept, "emotion": emo_labels_kept}).to_csv(out_dir / "stimulus_to_emotion.csv", index=False)
    pd.DataFrame([{"emotion": e, "n_stimuli": int(emo_idx[str(e)].size)} for e in emotions]).to_csv(out_dir / "emotion_stimulus_counts.csv", index=False)
    pd.DataFrame(
        {
            "repr_level": ["emotion"],
            "n_emotions": [int(n_emo)],
            "roi_set": [int(roi_set)],
            "rsm_method": [str(rsm_method)],
            "cocktail_blank": [bool(cocktail_blank)],
            "pattern_prefix_input": [str(pattern_prefix)],
            "pattern_key": ["subject"],
            "pattern_max_feat": [int(max_feat)],
        }
    ).to_csv(out_dir / f"{out_prefix}_meta.csv", index=False)

    if bool(save_pattern):
        pat_out_prefix = f"{out_prefix}_patterns"
        save_npz_named(out_dir / f"{pat_out_prefix}.npz", out_pat_by_sub)
        pd.DataFrame({"roi": rois}).to_csv(out_dir / f"{pat_out_prefix}_rois.csv", index=False)
        pd.DataFrame({"subject": subjects}).to_csv(out_dir / f"{pat_out_prefix}_subjects.csv", index=False)
        pd.DataFrame({"roi": rois, "n_feat": feat_sizes}).to_csv(out_dir / f"{pat_out_prefix}_roi_feat_sizes.csv", index=False)
        pd.DataFrame({"stimulus_order": emotions}).to_csv(out_dir / f"{pat_out_prefix}_stimulus_order.csv", index=False)
        pd.DataFrame(
            {
                "repr_level": ["emotion"],
                "n_emotions": [int(n_emo)],
                "roi_set": [int(roi_set)],
                "pattern_prefix_output": [str(pat_out_prefix)],
                "pattern_key": ["subject"],
                "pattern_max_feat": [int(max_feat)],
            }
        ).to_csv(out_dir / f"{pat_out_prefix}_meta.csv", index=False)

    return {"stimulus_type": stim_dir.name, "n_subjects": int(n_sub), "n_rois": int(len(rois)), "n_emotions": int(n_emo)}


def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
    roi_set: int,
    pattern_prefix: str,
    emotions: List[str],
    out_stimulus_dir_name: str,
    out_prefix: str,
    rsm_method: str,
    cocktail_blank: bool,
    save_pattern: bool,
) -> None:
    by_stim = Path(matrix_dir) / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"Directory not found: {by_stim}")

    rows = []
    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        out_dir = Path(matrix_dir) / str(out_stimulus_dir_name) / stim_dir.name
        row = run_one(
            stim_dir=stim_dir,
            pattern_prefix=str(pattern_prefix),
            emotions=[str(x) for x in emotions],
            out_dir=out_dir,
            out_prefix=str(out_prefix),
            rsm_method=str(rsm_method),
            cocktail_blank=bool(cocktail_blank),
            save_pattern=bool(save_pattern),
            roi_set=int(roi_set),
        )
        if row is None:
            print(f"[SKIP] {stim_dir.name}: cannot build a fixed {len(emotions)}x{len(emotions)} emotion matrix")
            continue
        rows.append(row)

    pd.DataFrame(rows).to_csv(Path(matrix_dir) / f"{out_prefix}_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    roi_set = int(args.roi_set)
    pattern_prefix = str(args.pattern_prefix) if args.pattern_prefix is not None else f"roi_beta_patterns_{roi_set}"
    emotions = [str(x) for x in args.emotions]
    out_prefix = str(args.out_prefix) if args.out_prefix is not None else f"roi_repr_matrix_{roi_set}_emotion{len(emotions)}"
    run(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        roi_set=roi_set,
        pattern_prefix=str(pattern_prefix),
        emotions=emotions,
        out_stimulus_dir_name=str(args.out_stimulus_dir_name),
        out_prefix=str(out_prefix),
        rsm_method=str(args.rsm_method),
        cocktail_blank=bool(args.cocktail_blank),
        save_pattern=bool(args.save_pattern),
    )


if __name__ == "__main__":
    main()
