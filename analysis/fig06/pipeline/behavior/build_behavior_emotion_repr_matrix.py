#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

if __package__ in {None, "", "behavior"}:
    THIS_DIR = Path(__file__).resolve().parent
    EMO_DIR = THIS_DIR.parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    if str(EMO_DIR) not in sys.path:
        sys.path.insert(0, str(EMO_DIR))

    from behavior_matrix_utils import compute_difference_matrix, save_npz_named  # noqa: E402
    from build_roi_emotion_repr_matrix import detect_emotion  # noqa: E402
    from calc_roi_isc_by_age import distance_to_similarity  # noqa: E402
else:
    from .behavior_matrix_utils import compute_difference_matrix, save_npz_named  # noqa: E402
    from ..build_roi_emotion_repr_matrix import detect_emotion  # noqa: E402
    from ..calc_roi_isc_by_age import distance_to_similarity  # noqa: E402


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
DEFAULT_EMOTIONS = ("Anger", "Disgust", "Fear", "Sad")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate trial-level behavior patterns into emotion-level behavior matrices")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--pattern-prefix", type=str, default="behavior_patterns_trial")
    p.add_argument("--diff-method", type=str, default="euclidean", choices=("euclidean", "cityblock", "manhattan", "mahalanobis"))
    p.add_argument("--emotions", nargs="+", default=list(DEFAULT_EMOTIONS))
    p.add_argument("--out-stimulus-dir-name", type=str, default="by_emotion")
    p.add_argument("--score-file-name", type=str, default="behavior_subject_emotion_scores.csv")
    p.add_argument("--out-pattern-prefix", type=str, default="behavior_patterns_emotion4")
    p.add_argument("--out-diff-prefix", type=str, default="behavior_diff_matrix_emotion4")
    p.add_argument("--out-repr-prefix", type=str, default="behavior_repr_matrix_emotion4")
    return p.parse_args()


def run_one(
    stim_dir: Path,
    pattern_prefix: str,
    emotions: List[str],
    diff_method: str,
    out_dir: Path,
    score_file_name: str,
    out_pattern_prefix: str,
    out_diff_prefix: str,
    out_repr_prefix: str,
) -> Optional[dict]:
    order_path = stim_dir / "stimulus_order.csv"
    subjects_path = stim_dir / f"{pattern_prefix}_subjects.csv"
    features_path = stim_dir / f"{pattern_prefix}_features.csv"
    pattern_npz_path = stim_dir / f"{pattern_prefix}.npz"
    for p in (order_path, subjects_path, features_path, pattern_npz_path):
        if not p.exists():
            return None

    stim_order = pd.read_csv(order_path)["stimulus_order"].astype(str).tolist()
    subjects = pd.read_csv(subjects_path)["subject"].astype(str).tolist()
    features = pd.read_csv(features_path)["feature"].astype(str).tolist()
    npz = np.load(pattern_npz_path)

    emo_labels = [detect_emotion(s, emotions) for s in stim_order]
    keep_idx = [i for i, e in enumerate(emo_labels) if e != ""]
    if not keep_idx:
        return None
    stim_order_kept = [stim_order[i] for i in keep_idx]
    emo_labels_kept = [emo_labels[i] for i in keep_idx]

    emo_idx: Dict[str, np.ndarray] = {}
    for emo in emotions:
        idx = np.where(np.array(emo_labels_kept, dtype=object) == str(emo))[0]
        if int(idx.size) == 0:
            return None
        emo_idx[str(emo)] = idx

    out_pattern: Dict[str, np.ndarray] = {}
    out_diff: Dict[str, np.ndarray] = {}
    out_repr: Dict[str, np.ndarray] = {}
    score_rows = []

    for sub in subjects:
        if str(sub) not in npz:
            raise KeyError(f"Pattern npz missing subject key: {sub}")
        arr = np.asarray(npz[str(sub)], dtype=np.float32)
        if arr.shape[0] != len(stim_order):
            raise ValueError(f"Unexpected pattern shape for {sub}: {arr.shape}")
        arr = arr[keep_idx, :]

        pat_emo = np.full((len(emotions), len(features)), np.nan, dtype=np.float32)
        for ei, emo in enumerate(emotions):
            idx = emo_idx[str(emo)]
            pat_emo[ei, :] = np.nanmean(arr[idx, :], axis=0).astype(np.float32)
            row = {"subject": str(sub), "emotion": str(emo), "n_stimuli_agg": int(idx.size)}
            for fi, feat in enumerate(features):
                row[str(feat)] = float(pat_emo[ei, fi])
            row["all_features_missing"] = bool(np.all(~np.isfinite(pat_emo[ei, :])))
            score_rows.append(row)

        if bool(np.all(~np.isfinite(pat_emo))):
            continue

        D = compute_difference_matrix(pat_emo, method=str(diff_method)).astype(np.float32)
        S = distance_to_similarity(D).astype(np.float32)
        out_pattern[str(sub)] = pat_emo
        out_diff[str(sub)] = D
        out_repr[str(sub)] = S

    kept_subjects = sorted(out_pattern.keys())
    if not kept_subjects:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(score_rows).to_csv(out_dir / str(score_file_name), index=False)
    save_npz_named(out_dir / f"{out_pattern_prefix}.npz", out_pattern)
    save_npz_named(out_dir / f"{out_diff_prefix}.npz", out_diff)
    save_npz_named(out_dir / f"{out_repr_prefix}.npz", out_repr)
    pd.DataFrame({"subject": kept_subjects}).to_csv(out_dir / f"{out_pattern_prefix}_subjects.csv", index=False)
    pd.DataFrame({"subject": kept_subjects}).to_csv(out_dir / f"{out_diff_prefix}_subjects.csv", index=False)
    pd.DataFrame({"subject": kept_subjects}).to_csv(out_dir / f"{out_repr_prefix}_subjects.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(out_dir / f"{out_pattern_prefix}_features.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(out_dir / f"{out_diff_prefix}_features.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(out_dir / f"{out_repr_prefix}_features.csv", index=False)
    pd.DataFrame({"emotion_order": emotions}).to_csv(out_dir / "emotion_order.csv", index=False)
    pd.DataFrame({"stimulus_order": emotions}).to_csv(out_dir / "stimulus_order.csv", index=False)
    pd.DataFrame({"stimulus_order": stim_order_kept, "emotion": emo_labels_kept}).to_csv(out_dir / "stimulus_to_emotion.csv", index=False)
    pd.DataFrame([{"emotion": e, "n_stimuli": int(emo_idx[str(e)].size)} for e in emotions]).to_csv(out_dir / "emotion_stimulus_counts.csv", index=False)
    meta = pd.DataFrame(
        {
            "repr_level": ["emotion"],
            "n_subjects": [int(len(kept_subjects))],
            "n_emotions": [int(len(emotions))],
            "n_features": [int(len(features))],
            "feature_cols": ["|".join(features)],
            "diff_method": [str(diff_method)],
            "pattern_prefix_input": [str(pattern_prefix)],
        }
    )
    meta.to_csv(out_dir / f"{out_pattern_prefix}_meta.csv", index=False)
    meta.to_csv(out_dir / f"{out_diff_prefix}_meta.csv", index=False)
    meta.to_csv(out_dir / f"{out_repr_prefix}_meta.csv", index=False)
    return {"stimulus_type": stim_dir.name, "n_subjects": int(len(kept_subjects)), "n_emotions": int(len(emotions)), "n_features": int(len(features))}


def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
    pattern_prefix: str,
    emotions: List[str],
    diff_method: str,
    out_stimulus_dir_name: str,
    score_file_name: str,
    out_pattern_prefix: str,
    out_diff_prefix: str,
    out_repr_prefix: str,
) -> None:
    by_stim = Path(matrix_dir) / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"Cannot find directory: {by_stim}")

    rows = []
    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        out_dir = Path(matrix_dir) / str(out_stimulus_dir_name) / stim_dir.name
        row = run_one(
            stim_dir=stim_dir,
            pattern_prefix=str(pattern_prefix),
            emotions=[str(x) for x in emotions],
            diff_method=str(diff_method),
            out_dir=out_dir,
            score_file_name=str(score_file_name),
            out_pattern_prefix=str(out_pattern_prefix),
            out_diff_prefix=str(out_diff_prefix),
            out_repr_prefix=str(out_repr_prefix),
        )
        if row is None:
            print(f"[SKIP] {stim_dir.name}: cannot build a complete fixed emotion matrix")
            continue
        rows.append(row)

    pd.DataFrame(rows).to_csv(Path(matrix_dir) / f"{out_repr_prefix}_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    run(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        pattern_prefix=str(args.pattern_prefix),
        emotions=[str(x) for x in args.emotions],
        diff_method=str(args.diff_method),
        out_stimulus_dir_name=str(args.out_stimulus_dir_name),
        score_file_name=str(args.score_file_name),
        out_pattern_prefix=str(args.out_pattern_prefix),
        out_diff_prefix=str(args.out_diff_prefix),
        out_repr_prefix=str(args.out_repr_prefix),
    )


if __name__ == "__main__":
    main()
