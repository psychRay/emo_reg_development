#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_brain_surface_delta_age_regression.py

Map FDR-significant regions from
joint_analysis_roi_isc_behavior_delta_age_regression.py onto the Schaefer
cortical surfaces (left and right) and the Tian subcortical volume. Export
.func.gii and .nii.gz statistic images.

Also:
1. Project L_/R_ Schaefer parcels into the Schaefer volumetric atlas.
2. Merge V_ Tian subcortical ROIs into the same volume.
3. Write a combined glass-brain NIfTI.
4. Optionally render a glass-brain PNG/SVG; Figure 6d uses the exported maps.

Example:
  python plot_brain_surface_delta_age_regression.py \
      --matrix-dir path/to/fig06/intermediate \
      --stimulus-dir-name by_emotion \
      --joint \
      --effect interaction_M_conv \
      --alpha 0.05 \
      --glass-brain --maps-only \
      --schaefer-vol path/to/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _patch_numpy_for_legacy_nibabel() -> None:
    if not hasattr(np, "sctypes"):
        np.sctypes = {  # type: ignore[attr-defined]
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.bytes_, np.str_, np.void, np.object_],
        }

    if not hasattr(np, "maximum_sctype"):
        def _maximum_sctype(dtype):
            dt = np.dtype(dtype)
            if dt.kind == "f":
                return np.longdouble
            if dt.kind == "c":
                return np.clongdouble
            if dt.kind == "u":
                return np.uint64
            if dt.kind in {"i", "b"}:
                return np.int64
            return dt.type

        np.maximum_sctype = _maximum_sctype  # type: ignore[attr-defined]


_patch_numpy_for_legacy_nibabel()

import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn import datasets, image, plotting


# -- Default paths ─────────────────────────────────────────────────────────────
DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")

DEFAULT_ATLAS_SURF_L = Path(
    "/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"
)
DEFAULT_ATLAS_SURF_R = Path(
    "/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"
)
DEFAULT_ATLAS_VOL = Path(
    "/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_2009cAsym.nii.gz"
)
DEFAULT_ATLAS_SCHAEFER_VOL = Path(
    "/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
)

DEFAULT_STIMULUS_DIR_NAME = "by_emotion"
DEFAULT_REAPPRAISAL_NAME = "Reappraisal"
DEFAULT_PASSIVE_NAME = "Passive_Emo"

DEFAULT_RESULT_FILENAME = "roi_isc_behavior_delta_age_regression.csv"
DEFAULT_JOINT_RESULT_FILENAME = "roi_isc_behavior_delta_age_regression_joint.csv"


# Effects available in separate mode
SEPARATE_EFFECTS: Dict[str, Dict[str, str]] = {
    "interaction": {
        "beta": "beta_interaction",
        "fdr": "p_fdr_perm_interaction_model_wise",
    },
}


# Effects available in joint mode
JOINT_EFFECTS: Dict[str, Dict[str, str]] = {
    "interaction_M_nn": {
        "beta": "beta_interaction_M_nn",
        "fdr": "p_fdr_perm_interaction_M_nn_model_wise",
    },
    "interaction_M_conv": {
        "beta": "beta_interaction_M_conv",
        "fdr": "p_fdr_perm_interaction_M_conv_model_wise",
    },
    "interaction_M_div": {
        "beta": "beta_interaction_M_div",
        "fdr": "p_fdr_perm_interaction_M_div_model_wise",
    },
}


TIAN_S2_LABELS: Dict[str, str] = {
    "V_1": "HIP-head",
    "V_2": "HIP-body",
    "V_3": "AMY",
    "V_4": "THA-anterior",
    "V_5": "THA-posterior",
    "V_6": "THA-lateral",
    "V_7": "THA-medial",
    "V_8": "CAU-head",
    "V_9": "CAU-body",
    "V_10": "PUT-anterior",
    "V_11": "PUT-posterior",
    "V_12": "NAc",
    "V_13": "GP-external",
    "V_14": "GP-internal",
    "V_15": "THA-ventral",
    "V_16": "THA-intralaminar",
    "V_17": "HIP-head",
    "V_18": "HIP-body",
    "V_19": "AMY",
    "V_20": "THA-anterior",
    "V_21": "THA-posterior",
    "V_22": "THA-lateral",
    "V_23": "THA-medial",
    "V_24": "CAU-head",
    "V_25": "CAU-body",
    "V_26": "PUT-anterior",
    "V_27": "PUT-posterior",
    "V_28": "NAc",
    "V_29": "GP-external",
    "V_30": "GP-internal",
    "V_31": "THA-ventral",
    "V_32": "THA-intralaminar",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Map FDR-significant delta-age-regression betas onto cortical surfaces and subcortical volume, with an optional glass brain"
    )

    # -- Input paths ─────────────────────────────────────────────────────────
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default=DEFAULT_STIMULUS_DIR_NAME)
    p.add_argument("--reappraisal-name", type=str, default=DEFAULT_REAPPRAISAL_NAME)
    p.add_argument("--passive-name", type=str, default=DEFAULT_PASSIVE_NAME)
    p.add_argument(
        "--result-csv",
        type=Path,
        default=None,
        help="Regression CSV. If omitted, the path is derived from matrix-dir and the condition names.",
    )
    p.add_argument(
        "--joint",
        action="store_true",
        help="Read roi_isc_behavior_delta_age_regression_joint.csv from the joint model.",
    )

    # -- Selection and display ─────────────────────────────────────────────────
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to display, for example M_conv, M_nn, M_div, or joint. Inferred when omitted.",
    )
    p.add_argument(
        "--effect",
        type=str,
        default=None,
        help=(
            "separate mode: interaction; "
            "joint mode: interaction_M_nn, interaction_M_conv, or interaction_M_div."
        ),
    )
    p.add_argument(
        "--fdr-col",
        type=str,
        default=None,
        help="FDR p-value column. Inferred from the effect when omitted.",
    )
    p.add_argument(
        "--beta-col",
        type=str,
        default=None,
        help="Beta column. Inferred from the effect when omitted.",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--positive-only",
        action="store_true",
        help="Keep only significant ROIs with beta > 0.",
    )
    p.add_argument(
        "--map-value-mode",
        type=str,
        default="beta",
        choices=["beta", "binary"],
        help="beta colors voxels by the coefficient; binary sets significant ROIs to 1.",
    )

    # -- Surface and volume atlases ────────────────────────────────────────
    p.add_argument("--atlas-surf-l", type=Path, default=DEFAULT_ATLAS_SURF_L)
    p.add_argument("--atlas-surf-r", type=Path, default=DEFAULT_ATLAS_SURF_R)
    p.add_argument("--atlas-vol", type=Path, default=DEFAULT_ATLAS_VOL)

    # -- Glass brain ───────────────────────────────────────────────────────
    p.add_argument(
        "--glass-brain",
        action="store_true",
        help="Also write a combined NIfTI and render a glass brain.",
    )
    p.add_argument(
        "--schaefer-vol",
        type=Path,
        default=DEFAULT_ATLAS_SCHAEFER_VOL,
        help="Schaefer2018 200Parcels 7Networks volumetric atlas。",
    )
    p.add_argument(
        "--glass-display-mode",
        type=str,
        default="lyrz",
        help="plot_glass_brain display_mode, for example lyrz, ortho, x, y, or z.",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--no-svg", action="store_true")
    p.add_argument("--maps-only", action="store_true", help="Export Figure 6d brain map data without rendering the brain figure.")

    # -- Output paths ─────────────────────────────────────────────────────────
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: {matrix_dir}/figures/{stimulus_dir_name}_brain_maps.",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Output-file prefix. Generated from the arguments when omitted.",
    )

    return p.parse_args()


def setup_svg_editable_text() -> None:
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["axes.unicode_minus"] = False


def is_left_surface_roi(roi: str) -> bool:
    s = str(roi).strip()
    return s.startswith("L_") and s.split("_", 1)[1].isdigit()


def is_right_surface_roi(roi: str) -> bool:
    s = str(roi).strip()
    return s.startswith("R_") and s.split("_", 1)[1].isdigit()


def is_surface_roi(roi: str) -> bool:
    return is_left_surface_roi(roi) or is_right_surface_roi(roi)


def is_volume_roi(roi: str) -> bool:
    s = str(roi).strip()
    return s.startswith("V_") and s.split("_", 1)[1].isdigit()


def roi_number(roi: str) -> int:
    return int(str(roi).strip().split("_", 1)[1])


def possible_surface_ids(roi: str) -> List[int]:
    """Parcel-index conventions for GIfTI surface atlases.

    Left hemisphere: L_1 -> 1
    Right hemisphere accepts either numbering:
      R_1   -> 1 or 101
      R_101 -> 101 or 1
    """
    roi = str(roi).strip()
    idx = roi_number(roi)

    if roi.startswith("L_"):
        return [idx]

    if roi.startswith("R_"):
        if idx > 100:
            return [idx, idx - 100]
        return [idx, idx + 100]

    return [idx]


def possible_schaefer_volume_ids(roi: str) -> List[int]:
    """The Schaefer-200 volumetric atlas is usually numbered L = 1-100 and R = 101-200.

    Also accepts:
      L_3   -> 3
      R_3   -> 103
      R_103 -> 103
    """
    roi = str(roi).strip()
    idx = roi_number(roi)

    if roi.startswith("L_"):
        return [idx]

    if roi.startswith("R_"):
        if idx > 100:
            return [idx]
        return [idx + 100]

    return [idx]


def possible_tian_volume_ids(roi: str) -> List[int]:
    """Tian S2 volume: V_1-V_32 -> atlas label 1-32。"""
    return [roi_number(roi)]


def load_atlas(
    surf_l_path: Path,
    surf_r_path: Path,
    vol_path: Path,
) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image, np.ndarray]:
    surf_l = nib.load(str(surf_l_path)).darrays[0].data.astype(int)
    surf_r = nib.load(str(surf_r_path)).darrays[0].data.astype(int)

    vol_img = image.load_img(str(vol_path))
    vol_data = np.rint(vol_img.get_fdata()).astype(int)

    print(f"[atlas] surface L vertices: {surf_l.size}")
    print(f"[atlas] surface R vertices: {surf_r.size}")
    print(f"[atlas] Tian volume shape: {vol_img.shape}")

    return surf_l, surf_r, vol_img, vol_data


def map_values_to_brain(
    df: pd.DataFrame,
    value_col: str,
    surf_l_data: np.ndarray,
    surf_r_data: np.ndarray,
    vol_data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Map values onto the left surface, right surface, and Tian volume."""
    out_surf_l = np.zeros_like(surf_l_data, dtype=np.float32)
    out_surf_r = np.zeros_like(surf_r_data, dtype=np.float32)
    out_vol = np.zeros_like(vol_data, dtype=np.float32)

    records = []

    use_df = df.copy()
    use_df[value_col] = pd.to_numeric(use_df[value_col], errors="coerce").fillna(0.0)

    for _, row in use_df.iterrows():
        roi = str(row["roi"]).strip()
        val = float(row[value_col])

        if np.isclose(val, 0.0):
            continue

        space = "unknown"
        possible_ids: List[int] = []
        matched_id = None
        mapped_units = 0
        label_name = ""

        if is_left_surface_roi(roi):
            space = "surface_L"
            possible_ids = possible_surface_ids(roi)
            for roi_id in possible_ids:
                mask = surf_l_data == int(roi_id)
                if np.any(mask):
                    out_surf_l[mask] = val
                    matched_id = int(roi_id)
                    mapped_units = int(np.sum(mask))
                    break

        elif is_right_surface_roi(roi):
            space = "surface_R"
            possible_ids = possible_surface_ids(roi)
            for roi_id in possible_ids:
                mask = surf_r_data == int(roi_id)
                if np.any(mask):
                    out_surf_r[mask] = val
                    matched_id = int(roi_id)
                    mapped_units = int(np.sum(mask))
                    break

        elif is_volume_roi(roi):
            space = "tian_volume"
            label_name = TIAN_S2_LABELS.get(roi, "")
            possible_ids = possible_tian_volume_ids(roi)
            for roi_id in possible_ids:
                mask = vol_data == int(roi_id)
                if np.any(mask):
                    out_vol[mask] = val
                    matched_id = int(roi_id)
                    mapped_units = int(np.sum(mask))
                    break

        records.append(
            {
                "roi": roi,
                "space": space,
                "label_name": label_name,
                "possible_atlas_ids": ",".join(str(x) for x in possible_ids),
                "matched_atlas_id": matched_id,
                "mapped_units": mapped_units,
                "value": val,
                "mapped": matched_id is not None,
            }
        )

    mapping_df = pd.DataFrame(records)

    if not mapping_df.empty:
        missing = mapping_df[~mapping_df["mapped"]]
        if not missing.empty:
            print("[WARN] These ROIs were not found in the surface or volume atlas:")
            print(
                missing[
                    ["roi", "space", "label_name", "possible_atlas_ids"]
                ].to_string(index=False)
            )

    print(f"[mapped] surface L nonzero vertices: {int(np.sum(np.abs(out_surf_l) > 0))}")
    print(f"[mapped] surface R nonzero vertices: {int(np.sum(np.abs(out_surf_r) > 0))}")
    print(f"[mapped] Tian volume nonzero voxels: {int(np.sum(np.abs(out_vol) > 0))}")

    return out_surf_l, out_surf_r, out_vol, mapping_df


def save_brain_maps(
    out_dir: Path,
    prefix: str,
    l_array: np.ndarray,
    r_array: np.ndarray,
    vol_img: nib.Nifti1Image,
    vol_array: np.ndarray,
) -> None:
    """Save left and right .func.gii files and a Tian volume .nii.gz."""
    out_dir.mkdir(parents=True, exist_ok=True)

    out_vol_img = image.new_img_like(vol_img, vol_array.astype(np.float32))
    vol_path = out_dir / f"{prefix}_volume.nii.gz"
    out_vol_img.to_filename(str(vol_path))
    print(f"[saved] {vol_path}")

    gi_l = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                np.asarray(l_array, dtype=np.float32),
                intent=nib.nifti1.intent_codes["NIFTI_INTENT_ESTIMATE"],
            )
        ]
    )
    path_l = out_dir / f"{prefix}_surf_L.func.gii"
    nib.save(gi_l, str(path_l))
    print(f"[saved] {path_l}")

    gi_r = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                np.asarray(r_array, dtype=np.float32),
                intent=nib.nifti1.intent_codes["NIFTI_INTENT_ESTIMATE"],
            )
        ]
    )
    path_r = out_dir / f"{prefix}_surf_R.func.gii"
    nib.save(gi_r, str(path_r))
    print(f"[saved] {path_r}")


def load_schaefer_volume(schaefer_vol: Path):
    """Load the Schaefer volume. Use a local file, otherwise try a Nilearn download."""
    path = Path(schaefer_vol)

    if path.exists():
        img = image.load_img(str(path))
        print(f"[atlas] Loaded local Schaefer volume: {path}")
        return img

    print(f"[atlas] Local Schaefer volume not found: {path}")
    print("[atlas] Try fetching Schaefer volume from nilearn...")

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=200,
        yeo_networks=7,
        resolution_mm=2,
    )
    img = image.load_img(atlas.maps)
    print(f"[atlas] Loaded Schaefer volume from nilearn: {atlas.maps}")

    return img


def load_tian_resampled(tian_vol: Path, target_img):
    """Load the Tian volume and resample it into Schaefer volume space."""
    path = Path(tian_vol)
    if not path.exists():
        raise FileNotFoundError(f"Tian volume atlas not found: {path}")

    tian_img = image.load_img(str(path))

    try:
        tian_resampled = image.resample_to_img(
            tian_img,
            target_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
    except TypeError:
        tian_resampled = image.resample_to_img(
            tian_img,
            target_img,
            interpolation="nearest",
        )

    print(f"[atlas] Loaded and resampled Tian volume: {path}")
    return tian_resampled


def map_values_to_glass_volume(
    df: pd.DataFrame,
    value_col: str,
    schaefer_img,
    tian_img,
) -> Tuple[nib.Nifti1Image, pd.DataFrame]:
    """Merge L_/R_ Schaefer ROIs and V_ Tian ROIs into one volumetric NIfTI."""
    schaefer_data = np.rint(schaefer_img.get_fdata()).astype(int)
    tian_data = np.rint(tian_img.get_fdata()).astype(int)

    if schaefer_data.shape != tian_data.shape:
        raise ValueError(
            "Schaefer and Tian shapes still differ after resampling: "
            f"{schaefer_data.shape} vs {tian_data.shape}"
        )

    out = np.zeros_like(schaefer_data, dtype=np.float32)
    records = []

    use_df = df.copy()
    use_df[value_col] = pd.to_numeric(use_df[value_col], errors="coerce").fillna(0.0)
    use_df = use_df[np.abs(use_df[value_col].to_numpy(dtype=float)) > 0].copy()

    for _, row in use_df.iterrows():
        roi = str(row["roi"]).strip()
        val = float(row[value_col])

        space = "unknown"
        label_name = ""
        possible_ids: List[int] = []
        matched_id = None
        mapped_voxels = 0

        if is_surface_roi(roi):
            space = "schaefer_surface_to_volume"
            possible_ids = possible_schaefer_volume_ids(roi)

            for atlas_id in possible_ids:
                mask = schaefer_data == int(atlas_id)
                if np.any(mask):
                    out[mask] = val
                    matched_id = int(atlas_id)
                    mapped_voxels = int(np.sum(mask))
                    break

        elif is_volume_roi(roi):
            space = "tian_volume"
            label_name = TIAN_S2_LABELS.get(roi, "")
            possible_ids = possible_tian_volume_ids(roi)

            for atlas_id in possible_ids:
                mask = tian_data == int(atlas_id)
                if np.any(mask):
                    out[mask] = val
                    matched_id = int(atlas_id)
                    mapped_voxels = int(np.sum(mask))
                    break

        records.append(
            {
                "roi": roi,
                "space": space,
                "label_name": label_name,
                "possible_atlas_ids": ",".join(str(x) for x in possible_ids),
                "matched_atlas_id": matched_id,
                "mapped_voxels": mapped_voxels,
                "value": val,
                "mapped": matched_id is not None,
            }
        )

    mapping_df = pd.DataFrame(records)

    if not mapping_df.empty:
        missing = mapping_df[~mapping_df["mapped"]]
        if not missing.empty:
            print("[WARN] These ROIs were not found in the glass-brain volume:")
            print(
                missing[
                    ["roi", "space", "label_name", "possible_atlas_ids"]
                ].to_string(index=False)
            )

    out_img = image.new_img_like(schaefer_img, out.astype(np.float32))

    print(f"[glass] nonzero voxels: {int(np.sum(np.abs(out) > 0))}")
    if not mapping_df.empty:
        print(
            "[glass] surface-to-volume voxels: "
            f"{int(mapping_df.loc[mapping_df['space'] == 'schaefer_surface_to_volume', 'mapped_voxels'].sum())}"
        )
        print(
            "[glass] Tian volume voxels: "
            f"{int(mapping_df.loc[mapping_df['space'] == 'tian_volume', 'mapped_voxels'].sum())}"
        )

    return out_img, mapping_df


def save_and_plot_glass_brain(
    img,
    out_dir: Path,
    prefix: str,
    display_mode: str,
    dpi: int,
    save_png: bool,
    save_svg: bool,
) -> None:
    """Save the combined NIfTI and render a glass brain."""
    setup_svg_editable_text()
    out_dir.mkdir(parents=True, exist_ok=True)

    nii_path = out_dir / f"{prefix}_glass_combined.nii.gz"
    img.to_filename(str(nii_path))
    print(f"[saved] {nii_path}")

    data = np.asarray(img.get_fdata(), dtype=float)
    nz = data[np.isfinite(data) & (np.abs(data) > 0)]

    if nz.size == 0:
        print("[WARN] Glass-brain NIfTI has no nonzero voxels; skipping the render.")
        return

    data_min = float(np.nanmin(nz))
    data_max = float(np.nanmax(nz))

    if data_min >= 0:
        cmap = "Reds"
        threshold = 1e-12
        vmin = 0.0
        vmax = max(0.05, float(np.nanpercentile(nz, 99)))
    elif data_max <= 0:
        cmap = "Blues_r"
        threshold = 1e-12
        vmax = 0.0
        vmin = min(-0.05, float(np.nanpercentile(nz, 1)))
    else:
        cmap = "RdBu_r"
        threshold = 1e-12
        vmax = max(0.05, float(np.nanpercentile(np.abs(nz), 99)))
        vmin = -vmax

    fig = plt.figure(figsize=(9.8, 3.6))

    plotting.plot_glass_brain(
        img,
        display_mode=str(display_mode),
        plot_abs=False,
        cmap=cmap,
        threshold=threshold,
        vmin=vmin,
        vmax=vmax,
        colorbar=True,
        figure=fig,
        title=f"{prefix} glass brain",
    )

    if bool(save_png):
        png_path = out_dir / f"{prefix}_glass_brain.png"
        fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
        print(f"[saved] {png_path}")

    if bool(save_svg):
        svg_path = out_dir / f"{prefix}_glass_brain.svg"
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"[saved] {svg_path}")

    plt.close(fig)


def resolve_input_csv(args: argparse.Namespace) -> Path:
    if args.result_csv is not None:
        return Path(args.result_csv)

    csv_dir = (
        Path(args.matrix_dir)
        / str(args.stimulus_dir_name)
        / f"{args.reappraisal_name}_minus_{args.passive_name}"
    )
    filename = DEFAULT_JOINT_RESULT_FILENAME if args.joint else DEFAULT_RESULT_FILENAME
    return csv_dir / filename


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return Path(args.out_dir)

    return Path(args.matrix_dir) / "figures" / f"{args.stimulus_dir_name}_brain_maps"


def determine_model_filter(df: pd.DataFrame, args: argparse.Namespace) -> str:
    models_in_csv = sorted(df["model"].astype(str).unique().tolist()) if "model" in df.columns else []

    if args.model is not None:
        return str(args.model)

    if args.joint or "joint" in models_in_csv:
        return "joint"

    if len(models_in_csv) == 1:
        return str(models_in_csv[0])

    raise ValueError(
        f"The CSV contains more than one model ({models_in_csv}). Set --model."
    )


def determine_effect_columns(df_model: pd.DataFrame, args: argparse.Namespace, is_joint: bool) -> Tuple[str, str, str]:
    effect_map = JOINT_EFFECTS if is_joint else SEPARATE_EFFECTS
    available_effects = list(effect_map.keys())

    if args.fdr_col is not None and args.beta_col is not None:
        effect_label = args.effect if args.effect else "custom"
        return str(effect_label), str(args.beta_col), str(args.fdr_col)

    effect = args.effect

    if effect is None:
        if is_joint:
            effect = "interaction_M_conv" if "interaction_M_conv" in effect_map else available_effects[0]
        else:
            effect = "interaction" if "interaction" in effect_map else available_effects[0]

    if effect not in effect_map:
        raise ValueError(
            f"Effect '{effect}' is not available. "
            f"Effects available in {'joint' if is_joint else 'separate'} mode: {available_effects}"
        )

    beta_col = effect_map[effect]["beta"]
    fdr_col = effect_map[effect]["fdr"]

    for col in [beta_col, fdr_col]:
        if col not in df_model.columns:
            raise ValueError(
                f"Result file is missing column '{col}'. Columns: {list(df_model.columns)}"
            )

    return str(effect), str(beta_col), str(fdr_col)


def main() -> None:
    args = parse_args()

    result_csv = resolve_input_csv(args)
    if not result_csv.exists():
        raise FileNotFoundError(f"Result file not found: {result_csv}")

    out_dir = resolve_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Input: {result_csv}")
    print(f"[info] Output directory: {out_dir}")

    df = pd.read_csv(result_csv)
    print(f"[info] {len(df)} rows; first columns={list(df.columns[:10])}")

    if "roi" not in df.columns:
        raise ValueError("Result file is missing the roi column")
    if "model" not in df.columns:
        raise ValueError("Result file is missing the model column")

    models_in_csv = sorted(df["model"].astype(str).unique().tolist())
    model_filter = determine_model_filter(df, args)

    df_model = df[df["model"].astype(str) == str(model_filter)].copy()
    if df_model.empty:
        raise ValueError(f"No rows for model={model_filter}. Available models: {models_in_csv}")

    is_joint = args.joint or str(model_filter) == "joint"

    effect_label, beta_col, fdr_col = determine_effect_columns(df_model, args, is_joint=is_joint)

    print(f"[info] model={model_filter}, n={len(df_model)}")
    print(f"[info] effect={effect_label}, beta_col={beta_col}, fdr_col={fdr_col}")

    fdr_vals = pd.to_numeric(df_model[fdr_col], errors="coerce")
    beta_vals = pd.to_numeric(df_model[beta_col], errors="coerce")

    sig_mask = np.isfinite(fdr_vals) & (fdr_vals <= float(args.alpha))

    if bool(args.positive_only):
        sig_mask = sig_mask & np.isfinite(beta_vals) & (beta_vals > 0)

    n_sig = int(np.sum(sig_mask))
    print(f"[info] alpha={args.alpha}: {n_sig}/{len(df_model)} ROIs significant")

    if n_sig == 0:
        print("[warn] No ROI is significant; writing an empty brain map.")

    df_plot = df_model.copy()
    df_plot["map_value"] = pd.to_numeric(df_plot[beta_col], errors="coerce").fillna(0.0)

    if str(args.map_value_mode) == "binary":
        df_plot.loc[sig_mask, "map_value"] = 1.0

    df_plot.loc[~sig_mask, "map_value"] = 0.0

    sig_df = df_plot[sig_mask].copy()
    if not sig_df.empty:
        show_cols = ["roi", beta_col, fdr_col, "map_value"]
        sig_show = sig_df[show_cols].copy()
        sig_show.columns = ["roi", "beta", "p_fdr", "map_value"]
        print("[info] Significant ROIs:")
        print(sig_show.to_string(index=False))

    if args.out_prefix is not None:
        prefix = str(args.out_prefix)
    else:
        tag = "pos" if bool(args.positive_only) else "all"
        val_tag = str(args.map_value_mode)
        prefix = f"delta_age_reg_{model_filter}_{effect_label}_fdr{args.alpha:g}_{tag}_{val_tag}"

    # Save the significant-ROI table.
    sig_csv = out_dir / f"{prefix}_significant_rois.csv"
    sig_df.to_csv(sig_csv, index=False)
    print(f"[saved] {sig_csv}")

    # Save the surface and volume images.
    surf_l_data, surf_r_data, vol_img, vol_data = load_atlas(
        args.atlas_surf_l,
        args.atlas_surf_r,
        args.atlas_vol,
    )

    l_arr, r_arr, vol_arr, map_df = map_values_to_brain(
        df_plot,
        "map_value",
        surf_l_data,
        surf_r_data,
        vol_data,
    )

    map_csv = out_dir / f"{prefix}_surface_volume_mapping_check.csv"
    map_df.to_csv(map_csv, index=False)
    print(f"[saved] {map_csv}")

    save_brain_maps(
        out_dir=out_dir,
        prefix=prefix,
        l_array=l_arr,
        r_array=r_arr,
        vol_img=vol_img,
        vol_array=vol_arr,
    )

    # Optional glass-brain output.
    if bool(args.glass_brain):
        schaefer_img = load_schaefer_volume(args.schaefer_vol)
        tian_img = load_tian_resampled(args.atlas_vol, target_img=schaefer_img)

        glass_img, glass_map_df = map_values_to_glass_volume(
            df=df_plot,
            value_col="map_value",
            schaefer_img=schaefer_img,
            tian_img=tian_img,
        )

        glass_map_csv = out_dir / f"{prefix}_glass_mapping_check.csv"
        glass_map_df.to_csv(glass_map_csv, index=False)
        print(f"[saved] {glass_map_csv}")

        if bool(args.maps_only):
            glass_img.to_filename(str(out_dir / f"{prefix}_glass_combined.nii.gz"))
        else:
            save_and_plot_glass_brain(
                img=glass_img,
                out_dir=out_dir,
                prefix=prefix,
                display_mode=str(args.glass_display_mode),
                dpi=int(args.dpi),
                save_png=not bool(args.no_png),
                save_svg=not bool(args.no_svg),
            )

    print(f"[done] Output directory: {out_dir}")


if __name__ == "__main__":
    main()
