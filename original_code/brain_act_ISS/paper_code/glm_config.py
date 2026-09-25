#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glm_config.py
Global configuration. Images are read from midprep and confounds from miniprep.

Path conventions shared by the analysis, plotting, and QC scripts:
- BIDS_DATA_DIR: BIDS root (default /public/home/dingrui/BIDS_DATA)
- LSS_OUTPUT_ROOT: LSS output root (default /public/home/dingrui/fmri_analysis/zz_analysis/lss_results)

Downstream similarity, permutation, and plotting steps use LSS_OUTPUT_ROOT
as the entry point for LSS products.
"""
import os
from pathlib import Path


class Config:
    def __init__(self, data_space='surface', hemi='L'):
        # =========================
        # 1. Main control switches
        # =========================
        self.PARALLEL = True  # True: use all available CPUs
        self.DEBUG = True  # Debug mode
        self.DEBUG_N_SUBJECTS = 30
        # Data space: 'volume' (MNI) or 'surface' (fsLR)
        self.DATA_SPACE = data_space
        self.HEMI = hemi  # Hemisphere: 'L' or 'R'
        if self.DATA_SPACE == 'volume':
            self.N_JOBS = 4
        else:
            # Surface jobs: 6 is a stable default (12 was too high).
            self.N_JOBS = 6
        # =========================
        # 2. Path mapping
        # =========================
        # Paths are taken from environment variables when set.
        # BIDS_DATA_DIR controls where the BIDS-like dataset lives on your machine/cluster.
        base_data_dir = os.environ.get("BIDS_DATA_DIR", "/public/home/dingrui/BIDS_DATA")
        self.BASE_DATA_DIR = Path(base_data_dir)

        # Require the base data directory.
        if not self.BASE_DATA_DIR.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.BASE_DATA_DIR}\nSet BIDS_DATA_DIR to the BIDS root.")

        # Output root
        # OUTPUT_ROOT is where LSS betas + aligned index CSVs are written.
        # LSS_OUTPUT_ROOT lets a reproduction run point at another machine without editing this file.
        self.OUTPUT_ROOT = Path(os.environ.get(
            "LSS_OUTPUT_ROOT",
            "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results",
        ))
        # Create the output directory if needed.
        self.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        # Scenario identifier
        self.SCENARIO_ID = f"{self.DATA_SPACE}{'_' + self.HEMI if self.DATA_SPACE == 'surface' else ''}"

        # Run mapping
        # RUN_MAP captures task-specific directory conventions (midprep for images, miniprep for confounds).
        self.RUN_MAP = {
            1: {
                "root": self.BASE_DATA_DIR / "emo_20250623/v1",
                "task": "EMO",
                "folder_task": "emo",
                "fmri_dir": "emo_midprep",  # Functional images are in midprep.
                "conf_dir": "emo_miniprep"  # Confounds are in miniprep.
            },
            2: {
                "root": self.BASE_DATA_DIR / "soc_20250623/v1",
                "task": "SOC",
                "folder_task": "soc",
                "fmri_dir": "soc_midprep",  # Functional images are in midprep.
                "conf_dir": "soc_miniprep"  # Confounds are in miniprep.
            }
        }

        self.RUNS = [1, 2]

        # Subject list from the EMO directory
        run1_root = self.RUN_MAP[1]["root"]
        if run1_root.exists():
            self.SUBJECTS = sorted([p.name for p in run1_root.glob("sub-*") if p.is_dir()])
            if not self.SUBJECTS:
                raise ValueError(f"Data root exists but contains no subject folders: {run1_root}")
        else:
            raise FileNotFoundError(f"Data root not found: {run1_root}")

        # =========================
        # 3. Auxiliary settings
        # =========================
        if self.DATA_SPACE == 'volume':
            self.MNI_MASK_PATH = Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_Binary_Mask.nii.gz")
        else:
            self.MNI_MASK_PATH = None

        self.TR = 0.75
        self.SMOOTHING_FWHM = None

        # --- Path builders ---

    def get_paths(self, sub, run_idx):
        info = self.RUN_MAP.get(run_idx)
        if not info: raise ValueError(f"No configuration for run {run_idx}")

        root = info["root"]
        task = info["task"]
        f_task = info["folder_task"]

        # Image and confound directories
        dir_fmri = info["fmri_dir"]  # midprep
        dir_conf = info["conf_dir"]  # miniprep

        # 1. Events path
        # Standardized input contract: (fmri_path, events.tsv, confounds.tsv).
        event_path = (
                root /
                f"{sub}/beh/func_task_{f_task}/run-1/{sub}_task-{task}_events.tsv"
        )

        # 2. fMRI path (midprep)
        if self.DATA_SPACE == 'volume':
            fmri_path = (
                    root /
                    f"{sub}/func/{dir_fmri}/{sub}_task-{task}_space-MNI152NLin2009cAsym_desc-taskFriston_custom-smooth_bold.nii.gz"
            )
        else:
            fmri_path = (
                    root /
                    f"{sub}/func/{dir_fmri}/"
                    f"{sub}_hemi-{self.HEMI}_space-fsLR_desc-taskFriston_custom-smooth_bold.shape.gii"
            )

        # 3. Confounds path (miniprep)
        confounds_path = (
                root /
                f"{sub}/func/{dir_conf}/{sub}_task-{task}_desc-confounds_timeseries.tsv"
        )

        return str(fmri_path), str(event_path), str(confounds_path)


# Default configuration
config = Config()

# Scenario list
SCENARIOS = [
    # (data_space, hemi)
    ('surface', 'L'),
    ('surface', 'R'),
    ('volume', ''),  # Volume has no hemisphere.
]

# Configuration object for every scenario
def get_scenario_configs():
    configs = []
    for data_space, hemi in SCENARIOS:
        # Volume does not take a hemisphere.
        if data_space == 'volume':
            configs.append(Config(data_space='volume', hemi=''))
        else:
            configs.append(Config(data_space=data_space, hemi=hemi))
    return configs
