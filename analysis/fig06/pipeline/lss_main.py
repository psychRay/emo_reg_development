#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lss_main.py
Least-squares-separate (LSS) entry point. Uses glm_config.py and glm_utils.py.
1. Single-trial models for surface and volume data (surface series are wrapped as NIfTI).
2. Write an index that includes stimulus_content so trials align across subjects.
"""

import os

# 1. Limit low-level BLAS threads.
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import gc
import logging
import argparse
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import nibabel as nb
from pathlib import Path
from joblib import Parallel, delayed
from nilearn.glm.first_level import FirstLevelModel
from nilearn import surface, image
from tqdm import tqdm

try:
    from glm_config import get_scenario_configs
    import glm_utils
except ImportError:
    print("Error: glm_config.py and glm_utils.py must be in the same directory")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

WRITE_MODES = ("overwrite", "append", "sync")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSS analysis with resume and incremental index updates")
    p.add_argument("subject_file", nargs="?", default=None, help="Optional subject-list file (whitespace-separated text)")
    p.add_argument(
        "--write-mode",
        choices=WRITE_MODES,
        default="overwrite",
        help="overwrite rebuilds the index; append adds new trials; sync records betas that already exist",
    )
    return p.parse_args()


def _merge_append(existing: pd.DataFrame, new: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    if existing is None or existing.empty:
        out = new.copy()
    elif new is None or new.empty:
        out = existing.copy()
    else:
        out = pd.concat([existing, new], axis=0, ignore_index=True)
    if out.empty:
        return out
    miss = [c for c in key_cols if c not in out.columns]
    if miss:
        raise ValueError(f"Index is missing columns required to detect duplicates: {miss}")
    out = out.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    return out


def load_subjects_from_file(subject_file):
    """Read a subject list from text, allowing a header and extra columns."""
    subject_file = Path(subject_file)
    if not subject_file.exists():
        raise FileNotFoundError(f"Subject-list file not found: {subject_file}")

    subjects = []
    with subject_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            first_col = line.split()[0]

            # Skip a common header such as: sub_id mean_FD
            if i == 0 and first_col.lower() in {"sub", "sub_id", "subject", "subject_id"}:
                continue

            # Normalize IDs to the sub- prefix.
            if not first_col.startswith("sub-"):
                first_col = f"sub-{first_col}"

            subjects.append(first_col)

    if not subjects:
        raise ValueError(f"Subject-list file is empty or contains no valid subjects: {subject_file}")

    # Drop duplicates and sort so the run order is stable.
    return sorted(set(subjects))


def clean_bids_val(val):
    """Clean strings in a BIDS table: drop a .png suffix and surrounding spaces."""
    s = str(val).strip()
    if s.endswith('.png'):
        s = s[:-4]
    return s


def validate_subject_task_presence(subjects, config):
    """Check that each subject has the core EMO and SOC input files."""
    presence_rows = []
    valid_subjects = []

    for sub in subjects:
        task_presence = {}
        missing_msgs = []

        for run in config.RUNS:
            task_label = config.RUN_MAP[run]["task"]
            try:
                fmri_path_s, event_path_s, conf_path_s = config.get_paths(sub, run)
                fmri_path, event_path, conf_path = Path(fmri_path_s), Path(event_path_s), Path(conf_path_s)

                conf_ok = conf_path.exists()
                if not conf_ok:
                    cand = list(conf_path.parent.glob(f"*{task_label}*confounds_timeseries.tsv"))
                    conf_ok = bool(cand)

                run_ok = fmri_path.exists() and event_path.exists() and conf_ok
                task_presence[task_label] = run_ok

                if not run_ok:
                    missing_msgs.append(
                        f"{task_label}:fmri={fmri_path.exists()},events={event_path.exists()},confounds={conf_ok}"
                    )
            except Exception as e:
                task_presence[task_label] = False
                missing_msgs.append(f"{task_label}:get_paths_failed({e})")

        emo_ok = task_presence.get("EMO", False)
        soc_ok = task_presence.get("SOC", False)
        both_ok = emo_ok and soc_ok

        presence_rows.append(
            {
                "subject": sub,
                "emo_inputs_ok": emo_ok,
                "soc_inputs_ok": soc_ok,
                "inputs_both_ok": both_ok,
                "input_check_note": "; ".join(missing_msgs) if missing_msgs else "",
            }
        )
        if both_ok:
            valid_subjects.append(sub)

    return valid_subjects, presence_rows


def save_beta_as_gifti(beta_data, output_path):
    """Save a NumPy array as a GIfTI file."""
    data_flat = beta_data.flatten().astype(np.float32)
    da = nb.gifti.GiftiDataArray(data_flat, intent='NIFTI_INTENT_ESTIMATE')
    img = nb.gifti.GiftiImage(darrays=[da])
    nb.save(img, str(output_path))


def process_single_trial_lss(args):
    """Fit the model for one trial."""
    (trial_idx, fmri_input, events_df, confounds_df,
     out_dir, unique_label, mask_img, config) = args

    try:
        stage = "prepare_input"
        # 1. Prepare the data object.
        if config.DATA_SPACE == 'volume':
            # Volume: pass the NIfTI path.
            input_img = fmri_input
            current_mask = mask_img
        else:
            # ====================================================
            # Surface: wrap the series as a NIfTI image and force a full mask.
            # ====================================================
            # Surface workaround: wrap (n_vert, n_time) as a fake 4D NIfTI and use an all-ones mask
            # so nilearn does not auto-mask away vertices.
            surf_data = surface.load_surf_data(fmri_input)

            # Cast to float32.
            try:
                surf_data = surf_data.astype(np.float32)
            except ValueError:
                surf_data = pd.to_numeric(surf_data.flatten(), errors='coerce').reshape(surf_data.shape)
                surf_data = np.nan_to_num(surf_data).astype(np.float32)

            # Require a 2D array (vertices, time).
            if surf_data.ndim == 1:
                surf_data = surf_data[:, np.newaxis]

            n_vert, n_time = surf_data.shape

            # Pseudo-4D volume (vertices, 1, 1, time).
            fake_vol_data = surf_data.reshape((n_vert, 1, 1, n_time))

            affine = np.eye(4)
            input_img = nb.Nifti1Image(fake_vol_data, affine=affine)

            # All-ones mask so Nilearn does not drop vertices.
            mask_data = np.ones((n_vert, 1, 1), dtype=np.int8)
            current_mask = nb.Nifti1Image(mask_data, affine=affine)
            # ====================================================

        # 2. Build the event file.
        stage = "create_lss_events"
        # LSS design: 1 target regressor (this trial) + 1 nuisance regressor (all other trials).
        lss_events = glm_utils.create_lss_events(trial_idx, events_df)

        # 3. Specify the model.
        stage = "init_model"
        model = FirstLevelModel(
            t_r=config.TR,
            slice_time_ref=0.5,
            smoothing_fwhm=config.SMOOTHING_FWHM,
            noise_model="ar1",
            standardize=True,
            hrf_model="spm + derivative",
            drift_model="cosine",
            high_pass=1 / 128.0,
            mask_img=current_mask,
            minimize_memory=False,
            verbose=0
        )

        # 4. Fit.
        stage = "fit"
        model.fit(input_img, events=lss_events, confounds=confounds_df)

        # 5. Extract the beta image.
        stage = "compute_contrast"
        beta_img = model.compute_contrast('LSS_TARGET', output_type='effect_size')

        # 6. Save the result.
        stage = "save"
        ext = ".gii" if config.DATA_SPACE == 'surface' else ".nii.gz"
        # Append the hemisphere suffix.
        hemi_suffix = f"_{config.HEMI}" if config.DATA_SPACE == 'surface' else ""
        fname = f"beta_{unique_label}{hemi_suffix}{ext}"

        out_path = out_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if config.DATA_SPACE == 'volume':
            beta_img.to_filename(out_path)
        else:
            beta_data = beta_img.get_fdata()
            save_beta_as_gifti(beta_data, out_path)

        return {"status": "success", "file": fname}

    except Exception as e:
        logger.error(f"Trial {trial_idx} error ({stage}): {e}")
        return {"status": "fail", "stage": stage, "error": str(e)}


def _build_expected_results(
    sub: str,
    run: int,
    task_label: str,
    events: pd.DataFrame,
    out_dir: Path,
    hemi_suffix: str,
    ext: str,
    cond_col: Optional[str],
    emo_col: Optional[str],
) -> Tuple[List[Path], List[dict]]:
    expected_files: List[Path] = []
    expected_results: List[dict] = []
    for idx in range(len(events)):
        row = events.iloc[idx]
        if str(row.get("trial_type", "")) == "Other":
            continue
        fname = f"beta_{row['unique_label']}{hemi_suffix}{ext}"
        expected_files.append(out_dir / fname)
        raw_cond = clean_bids_val(row[cond_col]) if cond_col else "Unknown"
        raw_emo = clean_bids_val(row[emo_col]) if emo_col else "Unknown"
        # Cross-subject alignment key (task + condition + emotion).
        stimulus_content = f"{task_label}_{raw_cond}_{raw_emo}"
        expected_results.append(
            {
                "subject": sub,
                "run": int(run),
                "task": str(task_label),
                "label": str(row["unique_label"]),
                "file": str(fname),
                "stimulus_content": str(stimulus_content),
                "raw_condition": str(raw_cond),
                "raw_emotion": str(raw_emo),
            }
        )
    return expected_files, expected_results


def sync_one_run(sub: str, run: int, config):
    # Sync mode: do not run GLM; only scan existing outputs and rebuild aligned index entries.
    audit = []
    try:
        fmri_path_s, event_path_s, conf_path_s = config.get_paths(sub, run)
        fmri_path, event_path, conf_path = Path(fmri_path_s), Path(event_path_s), Path(conf_path_s)
        if not fmri_path.exists() or not event_path.exists():
            audit.append(
                {
                    "subject": sub,
                    "run": run,
                    "task": config.RUN_MAP[run]["task"],
                    "status": "fail",
                    "stage": "sync_missing_inputs",
                    "error": "fmri_or_events_not_found",
                    "fmri_path": str(fmri_path),
                    "events_path": str(event_path),
                    "confounds_path": str(conf_path),
                }
            )
            return {"meta": [], "audit": audit}
    except Exception as e:
        audit.append(
            {
                "subject": sub,
                "run": run,
                "task": config.RUN_MAP[run]["task"] if run in config.RUN_MAP else "",
                "status": "fail",
                "stage": "sync_get_paths",
                "error": str(e),
            }
        )
        return {"meta": [], "audit": audit}

    task_type = config.RUN_MAP[run]["folder_task"]
    task_label = config.RUN_MAP[run]["task"]
    out_dir = config.OUTPUT_ROOT / task_type / sub
    ext = ".gii" if config.DATA_SPACE == "surface" else ".nii.gz"
    hemi_suffix = f"_{config.HEMI}" if config.DATA_SPACE == "surface" else ""

    try:
        events = pd.read_csv(event_path, sep="\t")
        if events.empty:
            audit.append(
                {
                    "subject": sub,
                    "run": run,
                    "task": task_label,
                    "status": "fail",
                    "stage": "sync_read_events",
                    "error": "events_empty",
                    "events_path": str(event_path),
                }
            )
            return {"meta": [], "audit": audit}
        col_map = {c.lower(): c for c in events.columns}
        cond_col = col_map.get("condition")
        emo_col = col_map.get("emotion")
        events = glm_utils.reclassify_events(events)
        events["unique_label"] = events["trial_type"] + "_tr" + events.index.astype(str)
        expected_files, expected_results = _build_expected_results(
            sub=sub,
            run=int(run),
            task_label=str(task_label),
            events=events,
            out_dir=Path(out_dir),
            hemi_suffix=str(hemi_suffix),
            ext=str(ext),
            cond_col=cond_col,
            emo_col=emo_col,
        )
        present = []
        for f, r in zip(expected_files, expected_results):
            if Path(f).exists():
                present.append(r)
        audit.append(
            {
                "subject": sub,
                "run": run,
                "task": task_label,
                "status": "success",
                "stage": "sync",
                "n_present": int(len(present)),
                "n_expected": int(len(expected_results)),
            }
        )
        return {"meta": present, "audit": audit}
    except Exception as e:
        audit.append(
            {
                "subject": sub,
                "run": run,
                "task": task_label,
                "status": "fail",
                "stage": "sync_failed",
                "error": str(e),
            }
        )
        return {"meta": [], "audit": audit}


def process_one_run(sub, run, config, record_existing_outputs: bool = True):
    audit = []
    # 1. Resolve paths.
    try:
        fmri_path_s, event_path_s, conf_path_s = config.get_paths(sub, run)
        fmri_path, event_path, conf_path = Path(fmri_path_s), Path(event_path_s), Path(conf_path_s)

        if not conf_path.exists():
            cand = list(conf_path.parent.glob(f"*{config.RUN_MAP[run]['task']}*confounds_timeseries.tsv"))
            if cand:
                conf_path = cand[0]
            else:
                audit.append(
                    {
                        "subject": sub,
                        "run": run,
                        "task": config.RUN_MAP[run]["task"],
                        "status": "fail",
                        "stage": "missing_confounds",
                        "error": "confounds_not_found",
                        "fmri_path": str(fmri_path),
                        "events_path": str(event_path),
                        "confounds_path": str(conf_path),
                    }
                )
                return {"meta": [], "audit": audit}

        if not fmri_path.exists() or not event_path.exists():
            audit.append(
                {
                    "subject": sub,
                    "run": run,
                    "task": config.RUN_MAP[run]["task"],
                    "status": "fail",
                    "stage": "missing_inputs",
                    "error": "fmri_or_events_not_found",
                    "fmri_path": str(fmri_path),
                    "events_path": str(event_path),
                    "confounds_path": str(conf_path),
                }
            )
            return {"meta": [], "audit": audit}
    except Exception:
        audit.append(
            {
                "subject": sub,
                "run": run,
                "task": config.RUN_MAP[run]["task"] if run in config.RUN_MAP else "",
                "status": "fail",
                "stage": "get_paths",
                "error": "get_paths_failed",
            }
        )
        return {"meta": [], "audit": audit}

    # 2. Prepare the output directory.
    task_type = config.RUN_MAP[run]['folder_task']
    task_label = config.RUN_MAP[run]['task']  # EMO or SOC
    out_dir = config.OUTPUT_ROOT / task_type / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load data.
    try:
        # Read events.
        events = pd.read_csv(event_path, sep='\t')
        if events.empty:
            audit.append(
                {
                    "subject": sub,
                    "run": run,
                    "task": task_label,
                    "status": "fail",
                    "stage": "read_events",
                    "error": "events_empty",
                    "events_path": str(event_path),
                }
            )
            return {"meta": [], "audit": audit}

        # Match column names without regard to case.
        col_map = {c.lower(): c for c in events.columns}
        cond_col = col_map.get('condition')
        emo_col = col_map.get('emotion')

        # Classify events and write trial_type.
        events = glm_utils.reclassify_events(events)

        # Load confounds.
        confounds = glm_utils.load_complex_confounds(str(conf_path))
        if confounds.empty:
            audit.append(
                {
                    "subject": sub,
                    "run": run,
                    "task": task_label,
                    "status": "fail",
                    "stage": "load_confounds",
                    "error": "confounds_empty",
                    "confounds_path": str(conf_path),
                }
            )
            return {"meta": [], "audit": audit}

        # Build position-based labels.
        events['unique_label'] = events['trial_type'] + '_tr' + events.index.astype(str)
    except Exception:
        audit.append(
            {
                "subject": sub,
                "run": run,
                "task": task_label,
                "status": "fail",
                "stage": "prepare_inputs",
                "error": "prepare_inputs_failed",
                "events_path": str(event_path),
                "confounds_path": str(conf_path),
            }
        )
        return {"meta": [], "audit": audit}

    # 4. Preload the volume and mask (volume runs only).
    run_mask = None
    fmri_input = str(fmri_path)
    if config.DATA_SPACE == 'volume':
        if config.MNI_MASK_PATH and config.MNI_MASK_PATH.exists():
            try:
                fmri_input = nb.load(str(fmri_path))
            except Exception:
                audit.append(
                    {
                        "subject": sub,
                        "run": run,
                        "task": task_label,
                        "status": "fail",
                        "stage": "load_fmri",
                        "error": "fmri_load_failed",
                        "fmri_path": str(fmri_path),
                    }
                )
                return {"meta": [], "audit": audit}

            try:
                mask_img = nb.load(str(config.MNI_MASK_PATH))
                run_mask = image.resample_to_img(mask_img, fmri_input, interpolation="nearest")
            except Exception:
                audit.append(
                    {
                        "subject": sub,
                        "run": run,
                        "task": task_label,
                        "status": "fail",
                        "stage": "mask",
                        "error": "mask_load_or_resample_failed",
                        "mask_path": str(config.MNI_MASK_PATH),
                    }
                )
                return {"meta": [], "audit": audit}
        else:
            audit.append(
                {
                    "subject": sub,
                    "run": run,
                    "task": task_label,
                    "status": "fail",
                    "stage": "mask",
                    "error": "mask_missing",
                    "mask_path": str(config.MNI_MASK_PATH) if config.MNI_MASK_PATH else "",
                }
            )
            return {"meta": [], "audit": audit}

    # 5. Loop over trials.
    results = []

    # Always iterate over the full trial list.
    indices = range(len(events))

    # 5.1 Skip this run when every trial result already exists.
    ext = ".gii" if config.DATA_SPACE == 'surface' else ".nii.gz"
    hemi_suffix = f"_{config.HEMI}" if config.DATA_SPACE == 'surface' else ""
    expected_files, expected_results = _build_expected_results(
        sub=sub,
        run=int(run),
        task_label=str(task_label),
        events=events,
        out_dir=Path(out_dir),
        hemi_suffix=str(hemi_suffix),
        ext=str(ext),
        cond_col=cond_col,
        emo_col=emo_col,
    )

    if expected_files and all(p.exists() for p in expected_files):
        audit.append(
            {
                "subject": sub,
                "run": run,
                "task": task_label,
                "status": "skipped",
                "stage": "all_trial_outputs_exist",
                "file_count": len(expected_files),
            }
        )
        return {"meta": expected_results if bool(record_existing_outputs) else [], "audit": audit}

    for idx in indices:
        try:
            row = events.iloc[idx]

            # Skip events that are not modeled.
            if row['trial_type'] == 'Other':
                audit.append(
                    {
                        "subject": sub,
                        "run": run,
                        "task": task_label,
                        "trial_idx": int(idx),
                        "label": str(row.get("unique_label", "")),
                        "trial_type": str(row.get("trial_type", "")),
                        "status": "skipped",
                        "stage": "trial_type_other",
                    }
                )
                continue

            # Extract the fields used to align trials across subjects.
            # Read condition and emotion from the current row.
            raw_cond = "Unknown"
            raw_emo = "Unknown"

            if cond_col:
                raw_cond = clean_bids_val(row[cond_col])
            if emo_col:
                raw_emo = clean_bids_val(row[emo_col])

            # Alignment label: Task_Condition_Emotion.
            # e.g., EMO_PassiveLook_Negative_001_Negative
            stimulus_content = f"{task_label}_{raw_cond}_{raw_emo}"
            # ===============================

            # Skip trials whose output already exists.
            fname = f"beta_{row['unique_label']}{hemi_suffix}{ext}"
            if (out_dir / fname).exists():
                audit.append(
                    {
                        "subject": sub,
                        "run": run,
                        "task": task_label,
                        "trial_idx": int(idx),
                        "label": str(row.get("unique_label", "")),
                        "trial_type": str(row.get("trial_type", "")),
                        "status": "skipped",
                        "stage": "output_exists",
                        "file": fname,
                    }
                )
                if bool(record_existing_outputs):
                    results.append(
                        {
                            "subject": sub,
                            "run": run,
                            "task": task_label,
                            "label": row["unique_label"],
                            "file": fname,
                            "stimulus_content": stimulus_content,
                            "raw_condition": raw_cond,
                            "raw_emotion": raw_emo,
                        }
                    )
                continue

            # Fit the trial model.
            args = (idx, fmri_input, events, confounds, out_dir, row['unique_label'], run_mask, config)
            res = process_single_trial_lss(args)

            if res and res.get("status") == "success":
                # Store metadata together with the alignment fields.
                results.append({
                    'subject': sub,
                    'run': run,
                    'task': task_label,
                    'label': row['unique_label'],  # Original position label (for example Passive_Neutral_tr0).
                    'file': res['file'],  # Output file name.
                    'stimulus_content': stimulus_content,  # Cross-subject content label.
                    'raw_condition': raw_cond,  # Original condition.
                    'raw_emotion': raw_emo  # Original emotion.
                })
                audit.append(
                    {
                        "subject": sub,
                        "run": run,
                        "task": task_label,
                        "trial_idx": int(idx),
                        "label": str(row.get("unique_label", "")),
                        "trial_type": str(row.get("trial_type", "")),
                        "status": "success",
                        "stage": "trial",
                        "file": str(res.get("file", "")),
                    }
                )
            else:
                audit.append(
                    {
                        "subject": sub,
                        "run": run,
                        "task": task_label,
                        "trial_idx": int(idx),
                        "label": str(row.get("unique_label", "")),
                        "trial_type": str(row.get("trial_type", "")),
                        "status": "fail",
                        "stage": str(res.get("stage", "trial")) if isinstance(res, dict) else "trial",
                        "error": str(res.get("error", "")) if isinstance(res, dict) else "unknown",
                    }
                )
        except Exception:
            continue

    gc.collect()
    return {"meta": results, "audit": audit}


def run_main():
    args = parse_args()
    subject_file = Path(args.subject_file).expanduser() if args.subject_file is not None else None
    run_lss(subject_file=subject_file, write_mode=str(args.write_mode))


def run_lss(subject_file: Optional[Path], write_mode: str) -> None:
    subjects_from_file = None
    if subject_file is not None:
        subjects_from_file = load_subjects_from_file(Path(subject_file))
        logger.info(f"Subject-list file: {subject_file}")
        logger.info(f"Subjects read from file: {len(subjects_from_file)}")
    write_mode = str(write_mode)
    if write_mode not in WRITE_MODES:
        raise ValueError(f"Unsupported write_mode: {write_mode}")

    scenarios = get_scenario_configs()

    for cfg in scenarios:
        scenario_name = f"{cfg.DATA_SPACE}_{cfg.HEMI}" if cfg.DATA_SPACE == 'surface' else "volume"
        logger.info(f"\n>>> Running {scenario_name} (n_jobs={cfg.N_JOBS})")

        # In debug mode, limit the number of subjects.
        original_subjects = list(subjects_from_file) if subjects_from_file is not None else list(cfg.SUBJECTS)
        subjects = list(original_subjects)
        if getattr(cfg, 'DEBUG', False):
            debug_n_sub = getattr(cfg, 'DEBUG_N_SUBJECTS', 30)
            subjects = subjects[:min(debug_n_sub, len(subjects))]
            total_n = len(subjects_from_file) if subjects_from_file is not None else len(cfg.SUBJECTS)
            logger.info(f"[DEBUG] Subject limit: {len(subjects)} / {total_n}")

        checked_subjects, presence_rows = validate_subject_task_presence(subjects, cfg)
        # Strictly require each subject to have both EMO and SOC inputs before running LSS.
        logger.info(f"Subjects with both EMO and SOC inputs: {len(checked_subjects)} / {len(subjects)}")
        tasks = [(s, r) for s in checked_subjects for r in cfg.RUNS]
        if not tasks:
            status_file = cfg.OUTPUT_ROOT / f"lss_subject_status_{cfg.SCENARIO_ID}.csv"
            pd.DataFrame(presence_rows).assign(
                lss_status="fail",
                lss_note="Input check failed; LSS was not run",
            ).to_csv(status_file, index=False)
            logger.info(f"LSS subject-status file saved: {status_file}")
            continue

        all_meta = []
        all_audit = []
        try:
            record_existing = bool(write_mode) == "overwrite"
            if write_mode == "append":
                record_existing = False
            if cfg.PARALLEL:
                if write_mode == "sync":
                    res_list = Parallel(n_jobs=cfg.N_JOBS)(
                        delayed(sync_one_run)(s, r, cfg)
                        for s, r in tqdm(tasks, desc=f"{scenario_name} sync")
                    )
                else:
                    res_list = Parallel(n_jobs=cfg.N_JOBS)(
                        delayed(process_one_run)(s, r, cfg, record_existing_outputs=bool(record_existing))
                        for s, r in tqdm(tasks, desc=f"{scenario_name} parallel")
                    )
                for r in res_list:
                    if isinstance(r, dict):
                        all_meta.extend(r.get("meta", []))
                        all_audit.extend(r.get("audit", []))
            else:
                for s, r in tqdm(tasks, desc=f"{scenario_name} serial"):
                    if write_mode == "sync":
                        res = sync_one_run(s, r, cfg)
                    else:
                        res = process_one_run(s, r, cfg, record_existing_outputs=bool(record_existing))
                    if isinstance(res, dict):
                        all_meta.extend(res.get("meta", []))
                        all_audit.extend(res.get("audit", []))

            out_file = cfg.OUTPUT_ROOT / f"lss_index_{cfg.SCENARIO_ID}_aligned.csv"
            new_df = pd.DataFrame(all_meta) if all_meta else pd.DataFrame()
            key_cols = ["subject", "run", "task", "label", "file"]
            if write_mode == "overwrite":
                # Overwrite mode: replace the aligned index with this run's meta only.
                if not new_df.empty:
                    new_df.to_csv(out_file, index=False)
                    logger.info(f"Full alignment index saved: {out_file}")
            else:
                # Append/sync mode: merge by a stable key to avoid duplicates.
                existing_df = pd.read_csv(out_file) if out_file.exists() else pd.DataFrame()
                merged = _merge_append(existing_df, new_df, key_cols=key_cols)
                if not merged.empty:
                    merged.to_csv(out_file, index=False)
                    logger.info(f"Alignment index updated: {out_file} (mode={write_mode}, rows={int(merged.shape[0])})")

            if all_audit:
                audit_file = cfg.OUTPUT_ROOT / f"lss_audit_{cfg.SCENARIO_ID}.csv"
                audit_new = pd.DataFrame(all_audit)
                if write_mode == "overwrite" or not audit_file.exists():
                    audit_new.to_csv(audit_file, index=False)
                else:
                    audit_old = pd.read_csv(audit_file) if audit_file.exists() else pd.DataFrame()
                    audit_merged = pd.concat([audit_old, audit_new], axis=0, ignore_index=True)
                    audit_merged.to_csv(audit_file, index=False)
                logger.info(f"LSS audit log saved: {audit_file} (mode={write_mode})")

            # Subject-level pass/fail list. A pass requires valid EMO and SOC results.
            # Subject-level status: EMO and SOC must both have at least one success/skipped audit entry.
            audit_file = cfg.OUTPUT_ROOT / f"lss_audit_{cfg.SCENARIO_ID}.csv"
            audit_df = pd.read_csv(audit_file) if audit_file.exists() else pd.DataFrame(all_audit)
            success_like = {"success", "skipped"}
            status_rows = []
            for row in presence_rows:
                sub = row["subject"]
                sub_audit = audit_df[audit_df["subject"] == sub] if not audit_df.empty else pd.DataFrame()

                emo_ok = bool(
                    not sub_audit.empty
                    and ((sub_audit["task"] == "EMO") & (sub_audit["status"].isin(success_like))).any()
                )
                soc_ok = bool(
                    not sub_audit.empty
                    and ((sub_audit["task"] == "SOC") & (sub_audit["status"].isin(success_like))).any()
                )

                lss_ok = bool(row["inputs_both_ok"] and emo_ok and soc_ok)
                note = []
                if not row["inputs_both_ok"]:
                    note.append("Input check failed")
                if row["inputs_both_ok"] and not emo_ok:
                    note.append("No successful or skipped EMO record")
                if row["inputs_both_ok"] and not soc_ok:
                    note.append("No successful or skipped SOC record")

                status_rows.append(
                    {
                        "subject": sub,
                        "emo_inputs_ok": row["emo_inputs_ok"],
                        "soc_inputs_ok": row["soc_inputs_ok"],
                        "inputs_both_ok": row["inputs_both_ok"],
                        "emo_lss_ok": emo_ok,
                        "soc_lss_ok": soc_ok,
                        "lss_status": "success" if lss_ok else "fail",
                        "lss_note": "; ".join(note) if note else "EMO and SOC both passed",
                        "input_check_note": row["input_check_note"],
                    }
                )

            status_file = cfg.OUTPUT_ROOT / f"lss_subject_status_{cfg.SCENARIO_ID}.csv"
            pd.DataFrame(status_rows).to_csv(status_file, index=False)
            logger.info(f"LSS subject pass/fail list saved: {status_file}")

        except Exception as e:
            logger.error(f"Processing failed: {e}")


if __name__ == "__main__":
    run_main()
