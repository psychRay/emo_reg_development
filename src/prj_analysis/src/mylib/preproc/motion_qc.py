
"""
fmri_motion_qc.py
-----------------

Subject-level motion QC for task fMRI based on fMRIPrep confounds.tsv.

Core features:
- Frame censoring via FD (>0.5 mm) OR robust-z DVARS (>1.5–2.0) with pre/post padding.
- Run-level metrics: bad-frame proportion, usable volumes, longest contiguous usable segment,
  mean FD, max FD (peak), and optional block retention.
- Run verdict: PASS / FLAG / FAIL using thresholds aligned with best-practice guidance.
- Subject-level verdict for a single task: PASS if any run PASS; otherwise FLAG if any run FLAG; else FAIL.

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd



# -------------------------
# Utility helpers
# -------------------------

def _robust_z(x: np.ndarray) -> np.ndarray:
    """Compute robust z-scores using median and MAD (scaled by 1.4826)."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if mad == 0 or np.isnan(mad):
        # Fallback to standard z if MAD is zero (rare but possible on flat signals)
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=1)
        if sd == 0 or np.isnan(sd):
            return np.zeros_like(x)
        return (x - mu) / sd
    return (x - med) / mad


def _expand_boolean_mask(mask: np.ndarray, pre: int, post: int) -> np.ndarray:
    """Expand True values in a boolean mask by pre and post frames (causal + acausal padding)."""
    n = len(mask)
    out = mask.copy()
    idx = np.flatnonzero(mask)
    for i in idx:
        a = max(0, i - pre)
        b = min(n, i + post + 1)
        out[a:b] = True
    return out


def _longest_contiguous_true(mask: np.ndarray) -> int:
    """Return length of the longest contiguous True segment in a boolean mask."""
    # We compute longest streak of True; typical use is on "usable" frames
    max_run = 0
    current = 0
    for v in mask:
        if v:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


@dataclass
class RunQCConfig:
    # Frame-censor thresholds
    fd_thresh: float = 0.5                 # Power FD threshold (mm)
    dvars_z_thresh: float = 1.5            # robust z-DVARS threshold
    pad_pre: int = 1                       # expand bad frames: include 1 frame before
    pad_post: int = 2                      # expand bad frames: include 2 frames after

    # Design-type specifics
    design: str = "event"                  # "event" or "block"
    tr: float = 2.0                        # seconds

    # Hard-fail thresholds
    min_usable_vols_event: int = 200       # usable volumes for event-related
    min_contig_tr_event: int = 60          # minimum contiguous usable frames for event-related
    min_block_retention: float = 0.80      # fraction of blocks retained (block design)
    max_fd_peak: float = 5.0               # absolute max instantaneous FD peak (mm)
    max_bad_prop: float = 0.40             # maximum proportion of bad frames

    # Soft-flag range (used only for messaging; verdict uses hard thresholds above)
    soft_bad_prop_low: float = 0.20        # 0.20–0.40 => FLAG


@dataclass
class RunQCMetrics:
    n_vols: int
    n_bad: int
    bad_prop: float
    usable_vols: int
    longest_contig_true: int
    mean_fd: float
    max_fd: float
    mean_fd_usable: float
    # Optional block retention (if blocks are provided)
    block_retention: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RunQCDecision:
    verdict: str            # "PASS" | "FLAG" | "FAIL"
    reasons: List[str]
    metrics: RunQCMetrics

    def to_dict(self) -> Dict:
        d = {"verdict": self.verdict, "reasons": self.reasons}
        d.update({"metrics." + k: v for k, v in self.metrics.to_dict().items()})
        return d


# -------------------------
# Core computation
# -------------------------

def compute_run_qc(
    confounds: pd.DataFrame,
    cfg: RunQCConfig,
    fd_col: Optional[str] = None,
    dvars_col: Optional[str] = None,
    blocks_sec: Optional[List[Tuple[float, float]]] = None,
) -> RunQCDecision:
    """
    Compute run-level QC decision and metrics from a single confounds dataframe.

    Parameters
    ----------
    confounds : pd.DataFrame
        fMRIPrep confounds.tsv loaded as DataFrame (one row per volume/timepoint).
    cfg : RunQCConfig
        Thresholds and design-specific parameters.
    fd_col : Optional[str]
        Column name for framewise displacement. Auto-detect if None.
    dvars_col : Optional[str]
        Column name for DVARS. Auto-detect if None.
    blocks_sec : Optional[List[(onset_sec, duration_sec)]]
        Only for block designs: list of block onsets and durations in seconds
        used to compute block retention.

    Returns
    -------
    RunQCDecision
    """

    # --- Detect columns ---
    if fd_col is None:
        for cand in ["framewise_displacement", "fd_power", "FD", "fd"]:
            if cand in confounds.columns:
                fd_col = cand
                break
    if fd_col is None:
        raise ValueError("Could not find an FD column in confounds. Provide fd_col.")

    if dvars_col is None:
        for cand in ["std_dvars", "dvars", "dvars_std", "DVARS"]:
            if cand in confounds.columns:
                dvars_col = cand
                break
    if dvars_col is None:
        raise ValueError("Could not find a DVARS column in confounds. Provide dvars_col.")

    fd = confounds[fd_col].astype(float).to_numpy()
    dvars = confounds[dvars_col].astype(float).to_numpy()

    n = len(confounds)
    if n == 0:
        raise ValueError("Empty confounds dataframe.")

    # fMRIPrep sometimes has NaNs at first volume; replace NaN FD with 0 (conservative)
    fd = np.nan_to_num(fd, nan=0.0, posinf=np.nanmax(np.where(np.isfinite(fd), fd, 0.0)), neginf=0.0)
    dvars = np.nan_to_num(dvars, nan=np.nanmedian(dvars) if np.isfinite(np.nanmedian(dvars)) else 0.0)

    # --- Robust z-DVARS ---
    dvars_z = _robust_z(dvars)

    # --- Identify bad frames ---
    bad_fd = fd > cfg.fd_thresh
    bad_dv = dvars_z > cfg.dvars_z_thresh
    bad = np.logical_or(bad_fd, bad_dv)
    # Expand by pre/post (padding around motion spikes)
    bad = _expand_boolean_mask(bad, pre=cfg.pad_pre, post=cfg.pad_post)

    usable = ~bad

    # --- Metrics ---
    n_bad = int(bad.sum())
    bad_prop = float(n_bad) / float(n)
    usable_vols = int(usable.sum())
    longest_contig = int(_longest_contiguous_true(usable))

    mean_fd = float(np.mean(fd))
    max_fd = float(np.max(fd))
    mean_fd_usable = float(np.mean(fd[usable])) if usable_vols > 0 else float("nan")

    block_ret = None
    if cfg.design.lower() == "block" and blocks_sec is not None and len(blocks_sec) > 0:
        # Convert usable frames to time windows and compute retention of block centers
        # A block is considered retained if its center falls on a usable frame
        block_kept = 0
        for onset, dur in blocks_sec:
            center_t = onset + 0.5 * dur
            # map to nearest TR index
            idx = int(np.round(center_t / cfg.tr))
            if 0 <= idx < n and usable[idx]:
                block_kept += 1
        block_ret = block_kept / len(blocks_sec)

    metrics = RunQCMetrics(
        n_vols=n,
        n_bad=n_bad,
        bad_prop=bad_prop,
        usable_vols=usable_vols,
        longest_contig_true=longest_contig,
        mean_fd=mean_fd,
        max_fd=max_fd,
        mean_fd_usable=mean_fd_usable,
        block_retention=block_ret,
    )

    # --- Verdict ---
    reasons = []
    verdict = "PASS"

    # Hard fails
    if max_fd > cfg.max_fd_peak:
        verdict = "FAIL"; reasons.append(f"max_fd>{cfg.max_fd_peak}mm ({max_fd:.2f}mm)")
    if bad_prop > cfg.max_bad_prop:
        verdict = "FAIL"; reasons.append(f"bad_prop>{cfg.max_bad_prop:.0%} ({bad_prop:.0%})")

    if cfg.design.lower() == "event":
        if metrics.usable_vols < cfg.min_usable_vols_event:
            verdict = "FAIL"; reasons.append(f"usable_vols<{cfg.min_usable_vols_event} ({metrics.usable_vols})")
        if metrics.longest_contig_true < cfg.min_contig_tr_event:
            verdict = "FAIL"; reasons.append(f"longest_contig<{cfg.min_contig_tr_event}TR ({metrics.longest_contig_true}TR)")

    elif cfg.design.lower() == "block":
        if block_ret is None:
            # If no blocks are provided we cannot assess retention; treat as unknown (FLAG)
            verdict = "FLAG"; reasons.append("block_retention=unknown (no blocks provided)")
        else:
            if block_ret < cfg.min_block_retention:
                verdict = "FAIL"; reasons.append(f"block_retention<{cfg.min_block_retention:.0%} ({block_ret:.0%})")

    # Soft flags (only if not already FAIL)
    if verdict != "FAIL":
        if cfg.soft_bad_prop_low <= bad_prop <= cfg.max_bad_prop:
            verdict = "FLAG"; reasons.append(f"bad_prop in [{cfg.soft_bad_prop_low:.0%}, {cfg.max_bad_prop:.0%}] ({bad_prop:.0%})")

    return RunQCDecision(verdict=verdict, reasons=reasons, metrics=metrics)


# -------------------------
# Subject-level aggregation
# -------------------------

def aggregate_subject_task(run_decisions: List[RunQCDecision]) -> Dict:
    """
    Aggregate multiple runs for a single subject and task.
    Rule: PASS if any run PASS; else FLAG if any run FLAG; else FAIL.
    Also return per-run metrics for transparency.

    Returns a flat dict suitable for DataFrame rows.
    """
    if len(run_decisions) == 0:
        return {"subject_task_verdict": "FAIL", "subject_task_reasons": ["no_runs_provided"]}

    verdicts = [d.verdict for d in run_decisions]
    if "PASS" in verdicts:
        subject_verdict = "PASS"
    elif "FLAG" in verdicts:
        subject_verdict = "FLAG"
    else:
        subject_verdict = "FAIL"

    out = {
        "subject_task_verdict": subject_verdict,
        "subject_task_reasons": [],
    }
    # Summaries across runs
    out["n_runs"] = len(run_decisions)
    out["n_PASS"] = sum(v == "PASS" for v in verdicts)
    out["n_FLAG"] = sum(v == "FLAG" for v in verdicts)
    out["n_FAIL"] = sum(v == "FAIL" for v in verdicts)

    # Attach simple averages for transparency
    avg_mean_fd = np.mean([d.metrics.mean_fd for d in run_decisions])
    avg_bad_prop = np.mean([d.metrics.bad_prop for d in run_decisions])
    out["avg_mean_fd"] = float(avg_mean_fd)
    out["avg_bad_prop"] = float(avg_bad_prop)

    return out


# -------------------------
# BIDS-like convenience
# -------------------------

def qc_runs_from_confounds_paths(
    confounds_paths: List[Path],
    cfg: RunQCConfig,
    fd_col: Optional[str] = None,
    dvars_col: Optional[str] = None,
    blocks_per_run_sec: Optional[Dict[str, List[Tuple[float, float]]]] = None,
) -> pd.DataFrame:
    """
    Batch QC across multiple confounds.tsv files (e.g., multiple runs for a subject).
    The output is a tidy DataFrame: one row per run with metrics and verdict.

    Parameters
    ----------
    confounds_paths : list of Paths
        Paths to fMRIPrep confounds.tsv files. The run label will be inferred from filename.
    blocks_per_run_sec : dict (optional)
        Mapping from run_id (string you expect to find in path name) to list of (onset, duration) in seconds.
        Only needed for block designs.
    """
    rows = []
    for p in confounds_paths:
        df = pd.read_csv(p, sep="\t")
        run_id = p.stem  # filename without extension
        blocks = None
        if blocks_per_run_sec is not None:
            # simple heuristic: match keys that are substrings of the run_id
            for k, v in blocks_per_run_sec.items():
                if k in run_id:
                    blocks = v
                    break

        decision = compute_run_qc(df, cfg, fd_col=fd_col, dvars_col=dvars_col, blocks_sec=blocks)
        row = {"run_id": run_id, "path": str(p), "verdict": decision.verdict, "reasons": ";".join(decision.reasons)}
        for k, v in decision.metrics.to_dict().items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def qc_subject_task_summary(
    confounds_paths: List[Path],
    cfg: RunQCConfig,
    fd_col: Optional[str] = None,
    dvars_col: Optional[str] = None,
    blocks_per_run_sec: Optional[Dict[str, List[Tuple[float, float]]]] = None,
) -> pd.DataFrame:
    """
    Run QC across runs and return a two-table summary concatenated vertically:
    - Run-level rows (one per run)
    - A final subject-level summary row (run_id='__SUBJECT_SUMMARY__')
    """
    run_df = qc_runs_from_confounds_paths(
        confounds_paths, cfg, fd_col=fd_col, dvars_col=dvars_col, blocks_per_run_sec=blocks_per_run_sec
    )

    # Build subject-level verdict from run decisions
    decisions = []
    for _, r in run_df.iterrows():
        metrics = RunQCMetrics(
            n_vols=int(r["n_vols"]),
            n_bad=int(r["n_bad"]),
            bad_prop=float(r["bad_prop"]),
            usable_vols=int(r["usable_vols"]),
            longest_contig_true=int(r["longest_contig_true"]),
            mean_fd=float(r["mean_fd"]),
            max_fd=float(r["max_fd"]),
            mean_fd_usable=float(r["mean_fd_usable"]),
            block_retention=float(r["block_retention"]) if pd.notna(r.get("block_retention", np.nan)) else None,
        )
        decisions.append(RunQCDecision(verdict=r["verdict"], reasons=r["reasons"].split(";") if r["reasons"] else [], metrics=metrics))

    subj = aggregate_subject_task(decisions)

    summary_row = {"run_id": "__SUBJECT_SUMMARY__", "path": ""}
    summary_row.update(subj)
    out = pd.concat([run_df, pd.DataFrame([summary_row])], ignore_index=True)
    return out
