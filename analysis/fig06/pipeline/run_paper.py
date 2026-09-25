#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the paper analysis and figure scripts from config_paper.json.

Edit paths and the steps map in config_paper.json. This runner only supplies
command-line arguments; the statistics live in the scripts it calls.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEP_ORDER = (
    "lss",
    "repr_trial",
    "repr_emotion",
    "isc_emotion",
    "brain_dev",
    "delta_dev",
    "behavior_trial",
    "behavior_emotion",
    "behavior_isc",
    "behavior_dev",
    "align_check",
    "delta_regression",
    "export_roi_names",
    "plot_b",
    "plot_c",
    "plot_d_bars",
    "plot_d_brain",
    "plot_d_slopes",
)
STEP_HELP = {
    "lss": "Single-trial LSS GLM. Off by default; betas are usually already on disk.",
    "repr_trial": "Trial-level brain ROI patterns and RSMs.",
    "repr_emotion": "Emotion-level brain RSMs (voxel-pattern average, then 4x4 RSM).",
    "isc_emotion": "Brain inter-subject representational ISC.",
    "brain_dev": "Brain ISC x M_nn / M_conv / M_div, permutation + model-wise FDR. Panel c.",
    "delta_dev": "Delta ISC (Reappraisal - Passive_Emo) x age models. ROI gate for panel d.",
    "behavior_trial": "Trial-level behavior patterns.",
    "behavior_emotion": "Emotion-level behavior patterns.",
    "behavior_isc": "Behavior inter-subject ISC from emotion patterns.",
    "behavior_dev": "Behavior ISC x age models, permutation null + bootstrap. Panel b.",
    "align_check": "Check that brain and behavior ISC subjects can be aligned.",
    "delta_regression": "Joint delta brain x behavior x age regression. Panel d statistics.",
    "export_roi_names": "Schaefer order-info file to ROI display names used by the plots.",
    "plot_b": "Behavior developmental-model null histograms.",
    "plot_c": "Export Reappraisal and Passive_Emo brain-map data for Figure 6c Workbench rendering.",
    "plot_d_bars": "Interaction beta bars for the significant ROIs.",
    "plot_d_brain": "Export surface/volume data for the Figure 6d brain visualization.",
    "plot_d_slopes": "Age-sorted coupling matrices and 10-window trends.",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce the paper ISC analyses and figures.")
    p.add_argument("--config", type=Path, default=ROOT / "config_paper.json")
    p.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step names. Overrides the enabled flags in the config.",
    )
    p.add_argument("--list", action="store_true", help="Print step names and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    for key in ("paths", "runtime", "analysis", "steps"):
        if key not in cfg or not isinstance(cfg[key], dict):
            raise ValueError(f"Config is missing object '{key}'")
    unknown = sorted(set(cfg["steps"]) - set(STEP_ORDER))
    if unknown:
        raise ValueError(f"Unknown steps in config: {unknown}")
    for name, value in cfg["paths"].items():
        if isinstance(value, str) and value.strip():
            candidate = Path(value)
            cfg["paths"][name] = str(candidate if candidate.is_absolute() else (Path(path).resolve().parent / candidate).resolve())
    return cfg


def selected_steps(cfg: dict, steps_arg: str | None) -> list[str]:
    if steps_arg:
        names = [part.strip() for part in steps_arg.split(",") if part.strip()]
        unknown = sorted(set(names) - set(STEP_ORDER))
        if unknown:
            raise ValueError(f"Unknown steps: {unknown}")
        return [name for name in STEP_ORDER if name in set(names)]
    enabled = cfg["steps"]
    return [name for name in STEP_ORDER if bool(enabled.get(name, False))]


def _opt(flag: str, value: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    return [flag, text]


def _join(root: str, *parts: str) -> str:
    """Join a configured data path without rewriting its slash style."""
    text = str(root).rstrip("/\\")
    sep = "\\" if ("\\" in text and "/" not in text) else "/"
    for part in parts:
        piece = str(part).strip("/\\")
        if piece:
            text = f"{text}{sep}{piece}"
    return text


def joint_csv(cfg: dict) -> str:
    paths = cfg["paths"]
    analysis = cfg["analysis"]
    name = f"{analysis['reappraisal_name']}_minus_{analysis['passive_name']}"
    return _join(
        paths["matrix_dir"],
        "by_emotion",
        name,
        "roi_isc_behavior_delta_age_regression_joint.csv",
    )


def mapping_csv(cfg: dict) -> str:
    return _join(cfg["paths"]["matrix_dir"], "schaefer2018_200parcels_7networks_order_mapping.csv")


def build_command(step: str, cfg: dict) -> tuple[list[str], dict[str, str]]:
    py = sys.executable
    paths = cfg["paths"]
    runtime = cfg["runtime"]
    analysis = cfg["analysis"]
    matrix = str(paths["matrix_dir"])
    panel_output = str(paths.get("panel_output_dir") or matrix)
    emotions = [str(x) for x in analysis["emotions"]]
    models = [str(x) for x in analysis["roi_selection_models"]]
    plot_models = [str(x) for x in analysis["plot_models"]]
    reappraisal = str(analysis["reappraisal_name"])
    passive = str(analysis["passive_name"])
    env = os.environ.copy()

    if step == "lss":
        env["BIDS_DATA_DIR"] = str(paths["bids_data_dir"])
        env["LSS_OUTPUT_ROOT"] = str(paths["lss_root"])
        return [py, "lss_main.py"], env

    if step == "repr_trial":
        return [
            py, "build_roi_repr_matrix.py",
            "--lss-root", str(paths["lss_root"]),
            "--out-dir", matrix,
            "--roi-set", str(analysis["roi_set"]),
            "--stim-type-col", str(analysis["stim_type_col"]),
            "--stim-id-col", str(analysis["stim_id_col"]),
            "--threshold-ratio", str(analysis["threshold_ratio"]),
            "--rsm-method", str(analysis["rsm_method"]),
            "--atlas-surf-left", str(paths["atlas_surf_left"]),
            "--atlas-surf-right", str(paths["atlas_surf_right"]),
            "--atlas-volume", str(paths["atlas_volume"]),
        ], env

    if step == "repr_emotion":
        return [
            py, "build_roi_emotion_repr_matrix.py",
            "--matrix-dir", matrix,
            "--stimulus-dir-name", "by_stimulus",
            "--roi-set", str(analysis["roi_set"]),
            "--pattern-prefix", f"roi_beta_patterns_{analysis['roi_set']}",
            "--emotions", *emotions,
            "--out-stimulus-dir-name", "by_emotion",
            "--out-prefix", f"roi_repr_matrix_{analysis['roi_set']}_emotion{len(emotions)}",
            "--rsm-method", str(analysis["rsm_method"]),
        ], env

    if step == "isc_emotion":
        return [
            py, "calc_roi_isc_by_age.py",
            "--matrix-dir", matrix,
            "--stimulus-dir-name", "by_emotion",
            "--subject-info", str(paths["subject_info"]),
            "--repr-prefix", f"roi_repr_matrix_{analysis['roi_set']}_emotion{len(emotions)}",
            "--isc-method", str(analysis["isc_method"]),
        ], env

    if step == "brain_dev":
        return [
            py, "joint_analysis_roi_isc_dev_models.py",
            "--matrix-dir", matrix,
            "--stimulus-dir-name", "by_emotion",
            "--isc-method", str(analysis["isc_method"]),
            "--assoc-method", str(analysis["assoc_method"]),
            "--correction-mode", "perm_fwer_fdr",
            "--n-perm", str(runtime["n_perm"]),
            "--seed", str(runtime["seed"]),
            "--n-jobs", str(runtime["n_jobs"]),
        ], env

    if step == "delta_dev":
        return [
            py, "joint_analysis_roi_isc_delta_dev_models.py",
            "--matrix-dir", matrix,
            "--stimulus-dir-name", "by_emotion",
            "--reappraisal-name", reappraisal,
            "--passive-name", passive,
            "--isc-method", str(analysis["isc_method"]),
            "--assoc-method", str(analysis["assoc_method"]),
            "--correction-mode", "perm_fwer_fdr",
            "--tail", "positive",
            "--n-perm", str(runtime["n_perm"]),
            "--seed", str(runtime["seed"]),
            "--n-jobs", str(runtime["n_jobs"]),
        ], env

    if step == "behavior_trial":
        return [
            py, str(Path("behavior") / "build_behavior_trial_repr_matrix.py"),
            "--matrix-dir", matrix,
            "--stimulus-dir-name", "by_stimulus",
            "--brain-repr-prefix", f"roi_repr_matrix_{analysis['roi_set']}",
            "--beh-data-dir", str(paths["beh_data_dir"]),
            "--fmri-data-dir", str(paths["fmri_data_dir"]),
            "--feature-cols", str(analysis["behavior_feature"]),
            "--agg-func", "mean",
            "--diff-method", "euclidean",
            "--pattern-prefix", "behavior_patterns_trial",
        ], env

    if step == "behavior_emotion":
        return [
            py, str(Path("behavior") / "build_behavior_emotion_repr_matrix.py"),
            "--matrix-dir", matrix,
            "--stimulus-dir-name", "by_stimulus",
            "--pattern-prefix", "behavior_patterns_trial",
            "--emotions", *emotions,
            "--diff-method", "euclidean",
            "--out-stimulus-dir-name", "by_emotion",
            "--out-pattern-prefix", "behavior_patterns_emotion4",
        ], env

    if step == "behavior_isc":
        return [
            py, str(Path("behavior") / "calc_behavior_isc_by_age.py"),
            "--matrix-dir", matrix,
            "--stimulus-dir-name", "by_emotion",
            "--subject-info", str(paths["subject_info"]),
            "--repr-prefix", "behavior_patterns_emotion4",
            "--input-kind", "pattern",
            "--isc-method", str(analysis["isc_method"]),
            "--isc-prefix", str(analysis["behavior_isc_prefix"]),
            "--max-missing-fraction", str(analysis["max_missing_fraction"]),
        ], env

    if step == "behavior_dev":
        return [
            py, str(Path("behavior") / "joint_analysis_behavior_isc_dev_models.py"),
            "--matrix-dir", matrix,
            "--stimulus-dir-name", "by_emotion",
            "--behavior-isc-prefix", str(analysis["behavior_isc_prefix"]),
            "--behavior-isc-method", str(analysis["isc_method"]),
            "--assoc-method", str(analysis["assoc_method"]),
            "--correction-mode", "perm_fwer_fdr",
            "--tail", "positive",
            "--n-perm", str(runtime["n_perm"]),
            "--seed", str(runtime["seed"]),
            "--n-jobs", str(runtime["n_jobs"]),
            "--n-bootstrap", str(runtime["n_bootstrap"]),
            "--bootstrap-seed", str(runtime["bootstrap_seed"]),
        ], env

    if step == "align_check":
        return [
            py, str(Path("behavior") / "check_brain_behavior_alignment.py"),
            "--matrix-dir", matrix,
            "--stimulus-dir-name", "by_emotion",
            "--brain-isc-prefix", f"roi_isc_{analysis['isc_method']}_by_age",
            "--behavior-isc-prefix", str(analysis["behavior_isc_prefix"]),
            "--conditions", passive, reappraisal,
            "--mode", "both",
            "--verbose",
        ], env

    if step == "delta_regression":
        return [
            py, str(Path("behavior") / "joint_analysis_roi_isc_behavior_delta_age_regression.py"),
            "--matrix-dir", matrix,
            "--stimulus-dir-name", "by_emotion",
            "--brain-stimulus-dir-name", "by_emotion",
            "--behavior-stimulus-dir-name", "by_emotion",
            "--reappraisal-name", reappraisal,
            "--passive-name", passive,
            "--roi-selection-dir-name", "by_emotion",
            "--roi-selection-conditions", f"{reappraisal}_minus_{passive}",
            "--roi-selection-file", "roi_isc_delta_dev_models_perm_fwer.csv",
            "--roi-selection-model", *models,
            "--roi-selection-p-col", str(analysis["roi_selection_p_col"]),
            "--roi-selection-alpha", str(analysis["roi_selection_alpha"]),
            "--brain-isc-method", str(analysis["isc_method"]),
            "--behavior-isc-method", str(analysis["isc_method"]),
            "--behavior-isc-prefix", str(analysis["behavior_isc_prefix"]),
            "--age-model-mode", "joint",
            "--rank-transform",
            "--tail", "positive",
            "--correction-mode", "perm_fwer_fdr",
            "--n-perm", str(runtime["n_perm"]),
            "--seed", str(runtime["seed"]),
            "--n-jobs", str(runtime["n_jobs"]),
        ], env

    if step == "export_roi_names":
        return [
            py, "export_schaefer_order_info_mapping.py",
            "--order-info", str(paths["schaefer_order_info"]),
            "--out", mapping_csv(cfg),
            "--right-label-mode", "global",
        ], env

    if step == "plot_b":
        return [
            py, "plot_behavior_isc_dev_model_null_distributions.py",
            "--matrix-dir", matrix,
            "--stimulus-dir-name", "by_emotion",
            "--models", *plot_models,
            "--p-col", "p_fdr_bh_model_wise",
            "--out-dir", _join(panel_output, "fig6b"),
        ], env

    if step == "plot_c":
        cmd = [
            py, "plot_combined_dev_model_brain_map.py",
            "--matrix-dir", matrix,
            "--stimulus-dir-name", "by_emotion",
            "--conditions", passive, reappraisal,
            "--models", *plot_models,
            "--p-col", "p_fdr_bh_model_wise",
            "--alpha", str(analysis["alpha"]),
            "--schaefer-mapping-file", mapping_csv(cfg),
            "--out-dir", _join(panel_output, "fig6c"),
            "--maps-only",
        ]
        cmd += _opt("--atlas-surf-left", paths.get("atlas_surf_left", ""))
        cmd += _opt("--atlas-surf-right", paths.get("atlas_surf_right", ""))
        cmd += _opt("--atlas-volume", paths.get("atlas_volume", ""))
        cmd += _opt("--surface-left", paths.get("surface_left", ""))
        cmd += _opt("--surface-right", paths.get("surface_right", ""))
        cmd += _opt("--surface-bg-left", paths.get("surface_bg_left", ""))
        cmd += _opt("--surface-bg-right", paths.get("surface_bg_right", ""))
        return cmd, env

    if step == "plot_d_bars":
        return [
            py, "plot_behavior_delta_age_interaction_bars.py",
            "--result-csv", joint_csv(cfg),
            "--alpha", str(analysis["alpha"]),
            "--p-prefix", "p_fdr_perm",
            "--schaefer-mapping-file", mapping_csv(cfg),
            "--out-dir", _join(panel_output, "fig6d", "bars"),
        ], env

    if step == "plot_d_brain":
        cmd = [
            py, "plot_brain_surface_delta_age_regression.py",
            "--matrix-dir", matrix,
            "--stimulus-dir-name", "by_emotion",
            "--reappraisal-name", reappraisal,
            "--passive-name", passive,
            "--result-csv", joint_csv(cfg),
            "--joint",
            "--effect", "interaction_M_conv",
            "--alpha", str(analysis["alpha"]),
            "--positive-only",
            "--glass-brain",
            "--out-dir", _join(panel_output, "fig6d", "brain_maps"),
            "--maps-only",
        ]
        cmd += _opt("--atlas-surf-l", paths.get("atlas_surf_left", ""))
        cmd += _opt("--atlas-surf-r", paths.get("atlas_surf_right", ""))
        cmd += _opt("--atlas-vol", paths.get("atlas_volume", ""))
        cmd += _opt("--schaefer-vol", paths.get("schaefer_vol", ""))
        return cmd, env

    if step == "plot_d_slopes":
        return [
            py, str(Path("behavior") / "plot_delta_brain_behavior_age_slopes.py"),
            "--matrix-dir", matrix,
            "--stimulus-dir-name", "by_emotion",
            "--brain-stimulus-dir-name", "by_emotion",
            "--behavior-stimulus-dir-name", "by_emotion",
            "--reappraisal-name", reappraisal,
            "--passive-name", passive,
            "--brain-isc-method", str(analysis["isc_method"]),
            "--behavior-isc-method", str(analysis["isc_method"]),
            "--behavior-isc-prefix", str(analysis["behavior_isc_prefix"]),
            "--result-csv", joint_csv(cfg),
            "--roi-p-col", "p_fdr_perm_interaction_M_conv_model_wise",
            "--alpha", str(analysis["alpha"]),
            "--n-windows", str(analysis["n_windows"]),
            "--min-subjects-window", str(analysis.get("min_subjects_window", 12)),
            "--min-pairs-window", str(analysis.get("min_pairs_window", 80)),
            "--min-pairs-matrix", str(analysis.get("min_pairs_matrix", 200)),
            "--schaefer-mapping-file", mapping_csv(cfg),
            "--out-dir", _join(panel_output, "fig6d", "coupling"),
        ], env

    raise ValueError(f"No command for step {step}")


def format_command(argv: list[str]) -> str:
    return " ".join(argv)


def main() -> None:
    args = parse_args()
    if args.list:
        for name in STEP_ORDER:
            print(f"{name}: {STEP_HELP[name]}")
        return
    cfg = load_config(args.config)
    steps = selected_steps(cfg, args.steps)
    if not steps:
        raise SystemExit("No steps selected. Enable steps in config_paper.json or pass --steps.")
    print(f"Working directory: {ROOT}")
    print(f"Steps: {', '.join(steps)}")
    for name in steps:
        argv, env = build_command(name, cfg)
        print(f"\n=== {name} ===")
        print(format_command(argv))
        if args.dry_run:
            continue
        completed = subprocess.run(argv, cwd=ROOT, env=env)
        if completed.returncode != 0:
            raise SystemExit(f"Step {name} failed with exit code {completed.returncode}")
    print("\nDone.")


if __name__ == "__main__":
    main()
