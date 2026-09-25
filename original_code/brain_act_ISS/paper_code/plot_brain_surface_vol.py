#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_brain_surface_vol.py

Step 5:
Map significant ROI correlation values onto the Schaefer cortical
surfaces (left and right) and the Tian subcortical volume. Export .func.gii
and .nii.gz statistic images and a preview rendering.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

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
from nilearn import datasets, image, plotting
import matplotlib.pyplot as plt

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
ATLAS_SURF_L = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii")
ATLAS_SURF_R = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii")
ATLAS_VOL = Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_2009cAsym.nii.gz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map significant ROI correlations onto the brain")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--repr-prefix", type=str, default=None)
    p.add_argument("--stimulus-type", type=str, required=True)
    p.add_argument("--model", type=str, default="M_conv", choices=["M_nn", "M_conv", "M_div"])
    p.add_argument("--sig-method", type=str, default="fwer", choices=["fwer", "fdr_model_wise", "fdr_global", "raw_p"])
    p.add_argument("--sig-col", type=str, default="p_perm_one_tailed", choices=["p_perm_one_tailed"])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--no-preview", action="store_true")
    return p.parse_args()


def has_repr_files(stim_dir: Path, repr_prefix: str) -> bool:
    required = (
        stim_dir / f"{repr_prefix}.npz",
        stim_dir / f"{repr_prefix}_subjects.csv",
        stim_dir / f"{repr_prefix}_rois.csv",
    )
    return all(p.exists() for p in required)


def load_atlas() -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image, np.ndarray]:
    """Load the cortical and volumetric atlases."""
    surf_l = nib.load(str(ATLAS_SURF_L)).darrays[0].data.astype(int)
    surf_r = nib.load(str(ATLAS_SURF_R)).darrays[0].data.astype(int)
    
    vol_img = image.load_img(str(ATLAS_VOL))
    vol_data = np.rint(vol_img.get_fdata()).astype(int)
    return surf_l, surf_r, vol_img, vol_data


def map_values_to_brain(df: pd.DataFrame, surf_l_data: np.ndarray, surf_r_data: np.ndarray, vol_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map r_obs from a results table onto left surface, right surface, and volume arrays."""
    # ROI naming convention:
    # - L_<id>, R_<id> for Schaefer surface parcels; V_<id> for Tian subcortex parcels.
    out_surf_l = np.zeros_like(surf_l_data, dtype=np.float32)
    out_surf_r = np.zeros_like(surf_r_data, dtype=np.float32)
    out_vol = np.zeros_like(vol_data, dtype=np.float32)

    for _, row in df.iterrows():
        roi = str(row["roi"])
        r_val = float(row["r_obs"])
        
        if roi.startswith("L_"):
            roi_id = int(roi.split("_")[1])
            out_surf_l[surf_l_data == roi_id] = r_val
        elif roi.startswith("R_"):
            roi_id = int(roi.split("_")[1])
            out_surf_r[surf_r_data == roi_id] = r_val
        elif roi.startswith("V_"):
            roi_id = int(roi.split("_")[1])
            out_vol[vol_data == roi_id] = r_val

    return out_surf_l, out_surf_r, out_vol


def apply_significance(df: pd.DataFrame, model: str, method: str, sig_col: str, alpha: float) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        raise ValueError("The result table is empty, so significant values cannot be mapped")

    if str(method) == "fwer":
        out = out[out["model"].astype(str) == str(model)].copy()
        if out.empty:
            raise ValueError(f"No rows for model={model}")
        if "p_fwer_model_wise" not in out.columns:
            raise ValueError("Result file has no p_fwer_model_wise column, so FWER cannot be mapped")
        sig = out["p_fwer_model_wise"].astype(float) <= float(alpha)
        out["sig_method"] = "fwer"
        out["p_sig"] = out["p_fwer_model_wise"].astype(float)
    elif str(method) == "raw_p":
        out = out[out["model"].astype(str) == str(model)].copy()
        if out.empty:
            raise ValueError(f"No rows for model={model}")
        if str(sig_col) not in out.columns:
            raise ValueError(f"Result file is missing column: {sig_col}")
        sig = out[str(sig_col)].astype(float) <= float(alpha)
        out["sig_method"] = f"raw_{sig_col}"
        out["p_sig"] = out[str(sig_col)].astype(float)
    elif str(method) in ("fdr_model_wise", "fdr_global"):
        if str(method) == "fdr_model_wise":
            out = out[out["model"].astype(str) == str(model)].copy()
            if out.empty:
                raise ValueError(f"No rows for model={model}")
            if "p_fdr_bh_model_wise" not in out.columns:
                raise ValueError("Result file has no p_fdr_bh_model_wise column, so FDR cannot be mapped")
            sig = out["p_fdr_bh_model_wise"].astype(float) <= float(alpha)
            out["sig_method"] = "fdr_model_wise"
            out["p_sig"] = out["p_fdr_bh_model_wise"].astype(float)
        else:
            out = out[out["model"].astype(str) == str(model)].copy()
            if out.empty:
                raise ValueError(f"No rows for model={model}")
            if "p_fdr_bh_global" not in out.columns:
                raise ValueError("Result file has no p_fdr_bh_global column, so FDR cannot be mapped")
            sig = out["p_fdr_bh_global"].astype(float) <= float(alpha)
            out["sig_method"] = "fdr_global"
            out["p_sig"] = out["p_fdr_bh_global"].astype(float)
    else:
        raise ValueError(f"Unknown sig-method: {method}")

    out = out.copy()
    # Non-significant ROIs are set to 0 so the exported brain maps visualize only significant effects.
    out.loc[~sig, "r_obs"] = 0.0
    return out


def save_and_plot_maps(
    out_dir: Path, prefix: str, l_array: np.ndarray, r_array: np.ndarray, 
    vol_img: nib.Nifti1Image, vol_array: np.ndarray, vmax: float, preview: bool
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save NIfTI and GIfTI files for Connectome Workbench.
    # Subcortical volume.
    out_vol_img = image.new_img_like(vol_img, vol_array)
    vol_path = out_dir / f"{prefix}_volume.nii.gz"
    out_vol_img.to_filename(str(vol_path))
    
    # Left cortical surface.
    gi_l = nib.gifti.GiftiImage(
        darrays=[nib.gifti.GiftiDataArray(np.asarray(l_array, dtype=np.float32), intent=nib.nifti1.intent_codes["NIFTI_INTENT_ESTIMATE"])]
    )
    path_l = out_dir / f"{prefix}_surf_L.func.gii"
    nib.save(gi_l, str(path_l))
    
    # Right cortical surface.
    gi_r = nib.gifti.GiftiImage(
        darrays=[nib.gifti.GiftiDataArray(np.asarray(r_array, dtype=np.float32), intent=nib.nifti1.intent_codes["NIFTI_INTENT_ESTIMATE"])]
    )
    path_r = out_dir / f"{prefix}_surf_R.func.gii"
    nib.save(gi_r, str(path_r))
    print(f"Statistic images written to: {out_dir}")

    if not bool(preview):
        return

    # 2. Render a Nilearn preview when a surface template is available.
    try:
        # Fetch the fsaverage surface. This can fail offline.
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
        
        # Figure.
        fig = plt.figure(figsize=(12, 8))
        
        # Left lateral view.
        ax1 = fig.add_subplot(221, projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, l_array, hemi='left', view='lateral',
            cmap='RdBu_r', vmax=vmax, vmin=-vmax, bg_map=fsaverage.sulc_left,
            axes=ax1, title="Left Lateral", colorbar=False
        )
        # Left medial view.
        ax2 = fig.add_subplot(222, projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, l_array, hemi='left', view='medial',
            cmap='RdBu_r', vmax=vmax, vmin=-vmax, bg_map=fsaverage.sulc_left,
            axes=ax2, title="Left Medial", colorbar=False
        )
        # Right lateral view.
        ax3 = fig.add_subplot(223, projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_right, r_array, hemi='right', view='lateral',
            cmap='RdBu_r', vmax=vmax, vmin=-vmax, bg_map=fsaverage.sulc_right,
            axes=ax3, title="Right Lateral", colorbar=False
        )
        # Subcortical slices.
        ax4 = fig.add_subplot(224)
        if np.max(np.abs(vol_array)) > 0:
            plotting.plot_stat_map(
                out_vol_img, display_mode='z', cut_coords=3, cmap='RdBu_r',
                vmax=vmax, axes=ax4, colorbar=True, title="Subcortex Volume"
            )
        else:
            ax4.text(0.5, 0.5, 'No Significant Subcortical ROIs', ha='center', va='center')
            ax4.axis("off")
            
        fig.suptitle(f"Brain Projection: {prefix}", fontsize=16)
        png_path = out_dir / f"{prefix}_brain_map.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Preview written to: {png_path}")
        
    except Exception as e:
        print(f"Nilearn preview failed, usually because fsaverage could not be downloaded offline.\nError: {e}\nThe exported .func.gii and .nii.gz files can be opened in Connectome Workbench.")


def main():
    args = parse_args()
    stim_dir = Path(args.matrix_dir) / str(args.stimulus_dir_name) / args.stimulus_type
    if args.repr_prefix is not None and not has_repr_files(stim_dir, repr_prefix=str(args.repr_prefix)):
        print(f"[SKIP] {stim_dir.name}: missing inputs for {args.repr_prefix}; skipped.")
        return
    result_csv = stim_dir / "roi_isc_dev_models_perm_fwer.csv"
    
    if not result_csv.exists():
        raise FileNotFoundError(f"Result file not found: {result_csv}")

    df = pd.read_csv(result_csv)

    df_sig = apply_significance(
        df=df,
        model=str(args.model),
        method=str(args.sig_method),
        sig_col=str(args.sig_col),
        alpha=float(args.alpha),
    )
    
    if df_sig["r_obs"].abs().max() == 0:
        print("Warning: no ROI is significant at this threshold; writing an empty brain map.")

    # Map values onto the atlases.
    surf_l_data, surf_r_data, vol_img, vol_data = load_atlas()
    l_arr, r_arr, vol_arr = map_values_to_brain(df_sig, surf_l_data, surf_r_data, vol_data)
    
    # Set a color limit that is not dominated by extreme values.
    vmax = max(0.05, float(df_sig["r_obs"].abs().quantile(0.99)))

    # Save and render.
    prefix = f"brain_map_{args.stimulus_dir_name}_{args.stimulus_type}_{args.model}_{args.sig_method}_a{args.alpha:g}"
    out_dir = Path(args.matrix_dir) / "figures" / f"{str(args.stimulus_dir_name)}_brain_maps"
    
    save_and_plot_maps(out_dir, prefix, l_arr, r_arr, vol_img, vol_arr, vmax, preview=not bool(args.no_preview))

if __name__ == "__main__":
    main()
