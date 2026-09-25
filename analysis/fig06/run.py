"""Run the original Figure 6 ISC and plotting scripts with project-local inputs."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent / "pipeline"
TOY_STEPS = (
    "brain_dev", "delta_dev", "behavior_dev", "align_check",
    "delta_regression", "plot_b", "plot_c", "plot_d_bars",
    "plot_d_brain", "plot_d_slopes",
)


def _extract_surface_labels(dlabel_path: Path, atlas_dir: Path) -> tuple[Path, Path]:
    """Split the project Schaefer CIFTI labels for the original GIFTI readers."""
    import nibabel as nib

    img = nib.load(str(dlabel_path))
    labels = np.asarray(img.get_fdata()[0], dtype=np.int32)
    axis = img.header.get_axis(1)
    outputs: dict[str, Path] = {}
    atlas_dir.mkdir(parents=True, exist_ok=True)
    for structure, slc, model in axis.iter_structures():
        hemi = {"CIFTI_STRUCTURE_CORTEX_LEFT": "L", "CIFTI_STRUCTURE_CORTEX_RIGHT": "R"}.get(structure)
        if hemi is None:
            continue
        values = np.zeros(model.nvertices[structure], dtype=np.int32)
        values[model.vertex] = labels[slc]
        path = atlas_dir / f"Schaefer2018_200Parcels_7Networks_{hemi}.label.gii"
        nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(values, intent="NIFTI_INTENT_LABEL")]), str(path))
        outputs[hemi] = path
    if set(outputs) != {"L", "R"}:
        raise ValueError(f"Both cortical hemispheres are required in {dlabel_path}")
    return outputs["L"], outputs["R"]


def _toy_isc(output: Path) -> dict:
    """Supply small age-ordered ISC matrices at the Fig. 6a analysis boundary."""
    import nibabel as nib

    output.mkdir(parents=True, exist_ok=True)
    matrix = output / "intermediate"
    rng = np.random.default_rng(42)
    n = 28
    subjects = [f"sub-{i:03d}" for i in range(1, n + 1)]
    ages = np.linspace(8.0, 18.0, n)
    rois = ["L_1", "R_101", "V_1", "V_2"]
    iu = np.triu_indices(n, 1)
    age_i, age_j = ages[iu[0]], ages[iu[1]]
    nn = 1.0 - np.abs(age_i - age_j) / 10.0
    conv = (np.minimum(age_i, age_j) - 8.0) / 10.0
    noise = lambda scale: rng.normal(0, scale, len(age_i))

    def matrix_from_pairs(values: np.ndarray) -> np.ndarray:
        result = np.eye(n, dtype=np.float32)
        result[iu] = values
        result[(iu[1], iu[0])] = values
        return result

    behavior_base = 0.08 * nn + 0.08 * conv + noise(0.025)
    brain_delta = 0.10 * conv + noise(0.08)
    behavior_delta = 0.04 * conv + 2.0 * brain_delta * conv
    brain_base = 0.08 * nn + 0.08 * conv
    sub_df = pd.DataFrame({"subject": subjects, "age": ages})
    for condition in ("Passive_Emo", "Reappraisal"):
        directory = matrix / "by_emotion" / condition
        directory.mkdir(parents=True, exist_ok=True)
        is_reappraisal = condition == "Reappraisal"
        brain = np.stack([
            matrix_from_pairs(
                brain_base + (brain_delta if is_reappraisal else 0)
                + noise(0.003 + 0.001 * index)
            ) for index in range(len(rois))
        ])
        behavior = matrix_from_pairs(
            behavior_base + (behavior_delta if is_reappraisal else 0) + noise(0.003)
        )
        brain_prefix = "roi_isc_mahalanobis_by_age"
        behavior_prefix = "behavior_pattern_isc_mahalanobis_by_age"
        np.save(directory / f"{brain_prefix}.npy", brain)
        np.save(directory / f"{behavior_prefix}.npy", behavior)
        sub_df.to_csv(directory / f"{brain_prefix}_subjects_sorted.csv", index=False)
        sub_df.to_csv(directory / f"{behavior_prefix}_subjects_sorted.csv", index=False)
        pd.DataFrame({"roi": rois}).to_csv(directory / f"{brain_prefix}_rois.csv", index=False)

    pd.DataFrame({
        "roi": rois,
        "source_name": ["7Networks_LH_Vis_1", "7Networks_RH_Vis_1", "Tian_1", "Tian_2"],
    }).to_csv(matrix / "schaefer2018_200parcels_7networks_order_mapping.csv", index=False)

    atlas_dir = matrix / "atlas"
    atlas_dir.mkdir(exist_ok=True)
    for hemi, label in (("L", 1), ("R", 101)):
        values = np.zeros(32492, dtype=np.int32)
        values[1000:3000] = label
        img = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(values, intent="NIFTI_INTENT_LABEL")])
        nib.save(img, str(atlas_dir / f"{hemi}.label.gii"))
    volume = np.zeros((24, 24, 24), dtype=np.int16)
    volume[5:10, 5:10, 5:10] = 1
    volume[14:19, 14:19, 14:19] = 2
    nib.save(nib.Nifti1Image(volume, np.diag([2.0, 2.0, 2.0, 1.0])), str(atlas_dir / "Tian.nii.gz"))

    with (PAPER / "config_paper.json").open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["paths"] = {
        "bids_data_dir": "", "lss_root": "", "matrix_dir": str(matrix.resolve()),
        "panel_output_dir": str(output.resolve()),
        "subject_info": "", "beh_data_dir": "", "fmri_data_dir": "",
        "schaefer_order_info": "", "atlas_surf_left": str((atlas_dir / "L.label.gii").resolve()),
        "atlas_surf_right": str((atlas_dir / "R.label.gii").resolve()),
        "atlas_volume": str((atlas_dir / "Tian.nii.gz").resolve()),
        "schaefer_vol": str((atlas_dir / "Tian.nii.gz").resolve()),
        "surface_left": str((ROOT / "resources/templates/fslr_32k/fsaverage.L.inflated.32k_fs_LR.surf.gii").resolve()),
        "surface_right": str((ROOT / "resources/templates/fslr_32k/fsaverage.R.inflated.32k_fs_LR.surf.gii").resolve()),
        "surface_bg_left": "", "surface_bg_right": "",
    }
    cfg["runtime"].update(n_perm=99, n_bootstrap=100, n_jobs=1)
    cfg["analysis"].update(min_subjects_window=3, min_pairs_window=3, min_pairs_matrix=3)
    cfg["steps"] = {name: name in TOY_STEPS for name in cfg["steps"]}
    config = matrix / "toy_config.json"
    config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def run_figure_6(config_path: Path | None, toy: bool, output_dir: Path,
                 steps: str | None = None) -> Path:
    if toy and steps is not None:
        raise ValueError("Figure 6 toy mode runs the complete pipeline")
    output_dir = output_dir.resolve()
    temporary_config: Path | None = None
    if toy:
        _toy_isc(output_dir)
        config = output_dir / "intermediate" / "toy_config.json"
    else:
        config = Path(config_path).resolve()
        if not config.is_file():
            raise FileNotFoundError(config)
        with config.open(encoding="utf-8") as handle:
            cfg = json.load(handle)
        for name, value in cfg["paths"].items():
            if isinstance(value, str) and value.strip():
                path = Path(value)
                cfg["paths"][name] = str(path if path.is_absolute() else (config.parent / path).resolve())
        raw_matrix = str(cfg["paths"].get("matrix_dir", "")).strip()
        if not raw_matrix:
            raise ValueError("Set paths.matrix_dir in the Figure 6 config")
        matrix = Path(raw_matrix)
        if not matrix.is_absolute():
            matrix = (config.parent / matrix).resolve()
        panel_raw = str(cfg["paths"].get("panel_output_dir", "")).strip()
        output_dir = Path(panel_raw) if panel_raw else (matrix.parent if matrix.name == "intermediate" else matrix)
        cfg["paths"]["panel_output_dir"] = str(output_dir)
        selected = {name.strip() for name in steps.split(",")} if steps else {name for name, on in cfg["steps"].items() if on}
        required_paths = {
            "lss": {"bids_data_dir", "lss_root"},
            "repr_trial": {"lss_root", "atlas_volume"},
            "isc_emotion": {"subject_info"},
            "behavior_trial": {"beh_data_dir", "fmri_data_dir"},
            "behavior_isc": {"subject_info"},
            "export_roi_names": {"schaefer_order_info"},
            "plot_c": {"atlas_volume", "surface_left", "surface_right"},
            "plot_d_brain": {"atlas_volume", "schaefer_vol"},
        }
        missing = sorted({key for step in selected for key in required_paths.get(step, ())
                          if not str(cfg["paths"].get(key, "")).strip()})
        if missing:
            raise ValueError(f"Set Figure 6 config paths: {', '.join(missing)}")
        needs_surface = bool(selected.intersection({"repr_trial", "plot_c", "plot_d_brain"}))
        left = cfg["paths"].get("atlas_surf_left", "")
        right = cfg["paths"].get("atlas_surf_right", "")
        if needs_surface and (not left or not right):
            dlabel_raw = cfg["paths"].get("atlas_dlabel", "")
            if not dlabel_raw:
                raise ValueError("Set paths.atlas_dlabel or both surface GIFTI atlases in the Figure 6 config")
            dlabel = Path(dlabel_raw)
            if not dlabel.is_file():
                raise FileNotFoundError(dlabel)
            left_path, right_path = _extract_surface_labels(dlabel, matrix / "atlases")
            cfg["paths"]["atlas_surf_left"] = str(left_path)
            cfg["paths"]["atlas_surf_right"] = str(right_path)
        matrix.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json",
                                         prefix=".fig06_runtime_", dir=matrix,
                                         delete=False) as handle:
            json.dump(cfg, handle, indent=2)
            temporary_config = Path(handle.name)
        config = temporary_config
    cmd = [sys.executable, str(PAPER / "run_paper.py"), "--config", str(config)]
    if steps:
        cmd.extend(["--steps", steps])
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    finally:
        if temporary_config is not None:
            temporary_config.unlink(missing_ok=True)
    if toy:
        from analysis.toy_outputs import finish_toy_outputs

        finish_toy_outputs(6, output_dir)
    return output_dir
