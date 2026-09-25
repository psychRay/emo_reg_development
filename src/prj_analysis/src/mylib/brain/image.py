#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 03:47:40 2025
    
    sorts of useful tools to manipulate brain images

@author: dingrui
"""


# modules
import warnings
warnings.filterwarnings('ignore', message='.*deprecated.*')

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, Colormap, ListedColormap
from nibabel.gifti import GiftiImage, GiftiDataArray
from typing import Union, Dict, Optional, List, Tuple

import mylib.brain.surface as surf

# functions
def make_diverging_cmap(vmin: float, vmax: float, vcenter: float = 0.0, cmap: str = "coolwarm"):
    """
    Create a diverging colormap and a normalization such that 'vcenter' maps
    exactly to the midpoint (0.5) of the colormap.
    
    Parameters
    ----------
    vmin : float
        Minimum data value.
    vmax : float
        Maximum data value.
    vcenter : float, optional
        The value to place at the center of the colormap (default 0.0).
    cmap : str or Colormap, optional
        A Matplotlib colormap name or object (default 'RdBu_r').
    
    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The diverging colormap.
    norm : matplotlib.colors.TwoSlopeNorm
        Normalization mapping values to [0, 1] with vcenter at 0.5.
    """
    if not (vmin < vcenter < vmax):
        raise ValueError(f"Require vmin < vcenter < vmax, got vmin={vmin}, vcenter={vcenter}, vmax={vmax}")
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cm = plt.get_cmap(cmap)
    return cm, norm

def make_center_anchored_div_cmap(
    vmin: float,
    vmax: float,
    vcenter: float = 0.0,
    cmap: Union[str, Colormap] = "RdBu_r",
    *,
    samples_per_side: int = 128,
    mid_color: Optional[tuple] = None,
    mode: str = "segmented"
) -> Colormap:
    """
    Build a multi-stop colormap that preserves the *shape* of an input Matplotlib
    preset colormap while anchoring the conceptual center (e.g., 0) at the linear
    index f0 = (vcenter - vmin) / (vmax - vmin).

    Parameters
    ----------
    vmin, vmax : float
        Data range that you'll pass later to surfplot as color_range=(vmin, vmax).
    vcenter : float, default 0.0
        The value you want to sit at the 'break' (mid) color.
    cmap : str or Colormap, default "RdBu_r"
        The source colormap (name or object). Its *shape* will be preserved.
    samples_per_side : int, default 128
        Sampling density on each side of the midpoint (total samples ≈ 2*samples_per_side).
        Increase for smoother fidelity.
    mid_color : tuple or None
        If provided, force the exact midpoint color (e.g., (1,1,1,1)) instead of cmap(0.5).
    mode : {"segmented","listed"}, default "segmented"
        - "segmented": returns LinearSegmentedColormap with explicit positions (exact).
        - "listed"   : returns ListedColormap (approximate but pure-listed).

    Returns
    -------
    Colormap
        A Matplotlib Colormap with the center anchored at f0.

    Notes
    -----
    - Use with surfplot:
        p.add_layer(..., cmap=the_cmap, color_range=(vmin, vmax), zero_transparent=False)
    - 'segmented' is mathematically precise (positions supported).
      'listed' approximates by dense sampling; with N≈256+ it’s visually indistinguishable.
    """
    if not (vmin < vcenter < vmax):
        raise ValueError(f"Require vmin < vcenter < vmax (got {vmin}, {vcenter}, {vmax})")

    # Get source colormap
    src = cm.get_cmap(cmap) if isinstance(cmap, (str, bytes)) else cmap
    if not isinstance(src, Colormap):
        raise TypeError("`cmap` must be a Matplotlib colormap name or a Colormap object.")

    # Where should vcenter land in [0,1] after *linear* normalization
    f0 = (vcenter - vmin) / (vmax - vmin)

    # Build input side positions (original parameterization)
    # Left side [0, 0.5], Right side [0.5, 1.0]
    nL = max(2, samples_per_side + 1)      # include endpoints
    nR = max(2, samples_per_side + 1)

    xL_in = np.linspace(0.0, 0.5, nL, endpoint=True)
    xR_in = np.linspace(0.5, 1.0, nR, endpoint=True)

    # Colors sampled from the *source* colormap on each side
    cL = src(xL_in)
    c_mid = (src(0.5) if mid_color is None else mid_color)
    cR = src(xR_in)

    # Replace the exact midpoint color to ensure crisp break
    cL[-1] = c_mid
    cR[0]  = c_mid

    if mode.lower() == "segmented":
        # Output positions (target parameterization) with the midpoint at f0
        xL_out = np.linspace(0.0, f0, nL, endpoint=True)
        xR_out = np.linspace(f0, 1.0, nR, endpoint=True)

        # Concatenate, removing the duplicated midpoint to avoid a tiny plateau
        x_out = np.concatenate([xL_out[:-1], xR_out])
        c_out = np.vstack([cL[:-1], cR])

        # Use explicit (pos, color) pairs for perfect alignment
        colors_with_pos = list(zip(x_out.tolist(), [tuple(row) for row in c_out]))
        return LinearSegmentedColormap.from_list(
            name=f"{getattr(src, 'name', 'cmap')}_center@{vcenter:g}",
            colors=colors_with_pos,
            N=len(colors_with_pos)
        )

    elif mode.lower() == "listed":
        # Build a pure ListedColormap by *re-indexing* samples so that
        # index round(f0*(N-1)) holds the midpoint color and each side keeps shape.
        # Total length (avoid double-counting the midpoint):
        N = (nL + nR - 1)
        i0 = int(round(f0 * (N - 1)))

        # Left block: map x in [0,0.5] into indices [0, i0]
        idxL = np.linspace(0, i0, nL, endpoint=True).astype(int)
        # Right block: map x in [0.5,1.0] into indices [i0, N-1]
        idxR = np.linspace(i0, N - 1, nR, endpoint=True).astype(int)

        # Initialize and fill with RGBA
        colors = np.zeros((N, 4), dtype=float)
        colors[idxL] = cL
        colors[idxR] = cR

        # Fill any potential gaps (due to integer rounding) by linear interpolation
        # to avoid tiny holes in discrete indices
        missing = np.where(~np.all(colors, axis=1))[0]
        if missing.size > 0:
            known = np.where(np.all(colors, axis=1))[0]
            for ch in range(4):
                colors[missing, ch] = np.interp(missing, known, colors[known, ch])

        return ListedColormap(colors, name=f"{getattr(src, 'name','cmap')}_listed_center@{vcenter:g}")

    else:
        raise ValueError("`mode` must be 'segmented' or 'listed'.")

def image_to_network_df(
    img: Union[str, nib.Nifti1Image, np.ndarray],
    atlas: Union[str, nib.Nifti1Image, np.ndarray],
    labels_map: Optional[Dict[int, str]] = None,
    *,
    resample_to_img: bool = True,
    drop_atlas_zero: True,
    drop_nan_inf: bool = True,
    out_csv: Optional[str] = None,
    atlas_is_probabilistic: bool = False,
    prob_rule: str = "argmax"  # only used if atlas_is_probabilistic=True
) -> pd.DataFrame:
    """
    Build a tidy DataFrame with columns: ['voxel_val', 'network_label'].

    Parameters
    ----------
    img : str | Nifti1Image | np.ndarray
        Path to 3D/4D image, or in-memory NIfTI/array. If 4D, it will be averaged across time.
    atlas : str | Nifti1Image | np.ndarray
        Path to network atlas. For label atlases: integer labels per voxel.
        For probabilistic atlases: last dim indexes networks (X,Y,Z,K).
    labels_map : dict[int, str], optional
        Mapping from integer label -> human-readable network name.
        If None, integer labels will be used as strings.
    resample_to_img : bool
        If True and both inputs are NIfTI with different shapes/affines, resample atlas to img space.
        Requires nilearn if resampling is needed.
    drop_atlas_zero : bool
        If True, drop voxels with atlas label == 0 (often background).
    drop_nan_inf : bool
        If True, drop voxels where image is NaN/Inf.
    out_csv : str, optional
        If provided, save the DataFrame to this CSV path.
    atlas_is_probabilistic : bool
        If True, treats `atlas` as a probabilistic atlas with shape (X,Y,Z,K) and assigns voxel to the
        network with maximum probability along K (or other rule in `prob_rule`).
    prob_rule : {'argmax'}
        Rule to convert probabilistic atlas to hard labels. Currently only 'argmax'.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ['voxel_val', 'network_label'].
    """
    # --- Load image ---
    def _to_nifti(x):
        if isinstance(x, str):
            return nib.load(x)
        elif isinstance(x, nib.spatialimages.SpatialImage):
            return x
        elif isinstance(x, np.ndarray):
            # Wrap naked arrays in a NIfTI with identity affine
            return nib.Nifti1Image(x, affine=np.eye(4))
        else:
            raise TypeError("`img`/`atlas` must be path, NIfTI image, or numpy array.")

    img_nii = _to_nifti(img)
    atl_nii = _to_nifti(atlas)

    # --- Collapse 4D img to 3D if needed (mean over time) ---
    img_data = np.asanyarray(img_nii.dataobj)
    if img_data.ndim == 4:
        img_data = np.nanmean(img_data, axis=-1)

    # --- Prepare atlas data ---
    atl_data = np.asanyarray(atl_nii.dataobj)

    # If probabilistic atlas (X,Y,Z,K), convert to hard labels via argmax
    if atlas_is_probabilistic:
        if atl_data.ndim != 4:
            raise ValueError("Probabilistic atlas expected shape (X,Y,Z,K).")
        if prob_rule != "argmax":
            raise NotImplementedError("Only argmax rule is implemented for now.")
        # labels become 1..K (or 0..K-1); we will use 1..K for clarity
        hard = np.argmax(atl_data, axis=-1).astype(np.int32)
        # Shift to 1..K
        hard = hard + 1
        atl_data = hard

    # --- Optionally resample atlas to image space if shapes/affines differ ---
    if resample_to_img:
        if (img_data.shape != atl_data.shape) or not np.allclose(img_nii.affine, atl_nii.affine):
            try:
                from nilearn.image import resample_to_img
            except ImportError as e:
                raise ImportError(
                    "Resampling requested but nilearn is not installed. "
                    "Install nilearn or set resample_to_img=False."
                ) from e
            atl_nii = resample_to_img(atl_nii, img_nii, interpolation='nearest')  # keep labels discrete
            atl_data = np.asanyarray(atl_nii.dataobj)

    if img_data.shape != atl_data.shape:
        raise ValueError(f"Shape mismatch after optional resampling: img {img_data.shape} vs atlas {atl_data.shape}")

    # --- Flatten arrays ---
    vox = img_data.reshape(-1)
    lab = atl_data.reshape(-1)

    # --- Build mask ---
    mask = np.ones_like(vox, dtype=bool)
    if drop_nan_inf:
        mask &= np.isfinite(vox)
    if drop_atlas_zero:
        mask &= (lab != 0)

    vox = vox[mask]
    lab = lab[mask]

    # --- Map integer labels to names (if provided) ---
    if labels_map is None:
        # default: use the integer itself as string
        network_label = lab.astype(int).astype(str)
    else:
        # unknown labels fall back to their integer string
        network_label = np.array([labels_map.get(int(k), str(int(k))) for k in lab], dtype=object)

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "voxel_val": vox.astype(np.float32),
        "network_label": network_label
    })

    if out_csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        df.to_csv(out_csv, index=False)

    return df

def vertex2roi(
    brain_img_lh: List[str],
    brain_img_rh: List[str],
    parc_lh: Union[str, np.ndarray], parc_rh: Union[str, np.ndarray],
    mask_lh: Optional[np.ndarray] = None, 
    mask_rh: Optional[np.ndarray] = None,
    include_labels_lh: Optional[List[int]] = None, 
    include_labels_rh: Optional[List[int]] = None,
    
):
    """
    convert brain image into roi-level data array given parcellations
    (Note: current version returns means of within-roi voxel/vertex value)
    """
    
    # ---- parcellation image/data and mask arrays ----
    if isinstance(parc_lh, str):
        lab_lh = surf._load_surf_labels(parc_lh, hemi_hint="lh", map_idx=0)
    else:
        lab_lh = parc_lh.copy()
    
    if isinstance(parc_rh, str):
        lab_rh = surf._load_surf_labels(parc_rh, hemi_hint="rh", map_idx=0)
    else:
        lab_rh = parc_rh.copy()
    
    if mask_lh is None:
        mask_lh_arr = np.ones(lab_lh.size, dtype=bool)
    else:
        mask_lh_arr = np.asarray(mask_lh, dtype=bool).reshape(-1)
        
    if mask_rh is None:
        mask_rh_arr = np.ones(lab_rh.size, dtype=bool)
    else:
        mask_rh_arr = np.asarray(mask_rh, dtype=bool).reshape(-1)
    
    keep_lh = mask_lh_arr & (lab_lh != 0)
    keep_rh = mask_rh_arr & (lab_rh != 0)
    
    if include_labels_lh is not None:
        keep_lh &= np.isin(lab_lh, np.asarray(include_labels_lh, np.int64))
    if include_labels_rh is not None:
        keep_rh &= np.isin(lab_rh, np.asarray(include_labels_rh, np.int64))
    
    uniq_lh = np.unique(lab_lh[keep_lh]); uniq_lh.sort()
    uniq_rh = np.unique(lab_rh[keep_rh]); uniq_rh.sort()
    roi_keys = [('lh', int(k)) for k in uniq_lh] + [('rh', int(k)) for k in uniq_rh]
    
    if len(roi_keys) < 1:
        raise ValueError("No ROIs found after masking/filtering.")
        
    key_to_idx = {k: i for i,k in enumerate(roi_keys)}
    kept_idx_lh = np.flatnonzero(keep_lh)
    kept_idx_rh = np.flatnonzero(keep_rh)
    roi_index_lh = np.array([key_to_idx[('lh', int(l))] for l in lab_lh[kept_idx_lh]], dtype=np.int64)
    roi_index_rh = np.array([key_to_idx[('rh', int(l))] for l in lab_rh[kept_idx_rh]], dtype=np.int64)
    R = len(roi_keys)
    
    # build ROI table
    roi_table = []
    for hemi, lab in roi_keys:
        if hemi == 'lh':
            nvert = (lab_lh[kept_idx_lh] == lab).sum()
        else:
            nvert = (lab_rh[kept_idx_rh] == lab).sum()
        roi_table.append({'hemi': hemi, 'label': int(lab), 'n_vert': int(nvert)})
    
    # ---- build data matrix: ROI x subject means of brain image ----
    N = len(brain_img_lh)
    M = np.full((R, N), np.nan, dtype=np.float64)
    
    for i in tqdm(range(N),
                  disable=False if N>10 else True,
                  desc='Transforming surface image into data array' if N>10 else None):
        lh = surf._load_surf_metric(brain_img_lh[i], np.float32, "lh")
        rh = surf._load_surf_metric(brain_img_rh[i], np.float32, "rh")

        # LH
        if kept_idx_lh.size > 0:
            vals = lh[kept_idx_lh].astype(np.float64, copy=False)
            fin  = np.isfinite(vals)
            if fin.any():
                sums = np.bincount(roi_index_lh[fin], weights=vals[fin], minlength=R)
                cnts = np.bincount(roi_index_lh[fin], minlength=R)
                means = np.divide(sums, cnts, out=np.full(R, np.nan), where=cnts>0)
                # store only lh ROIs
                lh_mask = np.zeros(R, dtype=bool) 
                lh_mask[[key_to_idx[('lh', int(l))] for l in uniq_lh]] = True
                M[lh_mask, i] = means[lh_mask]
        # RH
        if kept_idx_rh.size > 0:
            vals = rh[kept_idx_rh].astype(np.float64, copy=False)
            fin  = np.isfinite(vals)
            if fin.any():
                sums = np.bincount(roi_index_rh[fin], weights=vals[fin], minlength=R)
                cnts = np.bincount(roi_index_rh[fin], minlength=R)
                means = np.divide(sums, cnts, out=np.full(R, np.nan), where=cnts>0)
                rh_mask = np.zeros(R, dtype=bool)
                rh_mask[[key_to_idx[('rh', int(l))] for l in uniq_rh]] = True
                M[rh_mask, i] = means[rh_mask]
    
    return {'roi_table': roi_table, 'roi_data': M}

def voxel2roi(
    brain_img_ls: List[Union[str, np.ndarray]],
    parcellation_img,
    mask_path: Optional[str] = None,
    include_labels: Optional[List[int]] = None,
    exclude_labels: Optional[List[int]] = None,
    reducer: str = "mean",    # "mean" | "median"
    dtype=np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert N voxel-wise 3D maps into an ROI-by-N matrix by parcel-wise reduction.

    Returns
    -------
    M : (R, N) ROI-by-map matrix
    roi_ids : (R,) kept parcel label IDs (sorted ascending)
    """
    if isinstance(parcellation_img, str):
        parc_img = nib.load(parcellation_img)
    else:
        parc_img = parcellation_img
        
    parc = parc_img.get_fdata()
    shape, aff = parc.shape, parc_img.affine

    if mask_path:
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata().astype(bool)
        if mask.shape != shape or not np.allclose(mask_img.affine, aff):
            raise ValueError("mask must match parcellation shape/affine.")
    else:
        mask = np.ones(shape, dtype=bool)

    labels_all = parc[mask].astype(np.int64)
    keep = labels_all != 0
    if include_labels is not None:
        keep &= np.isin(labels_all, np.asarray(include_labels, np.int64))
    if exclude_labels is not None:
        keep &= ~np.isin(labels_all, np.asarray(exclude_labels, np.int64))

    labels_kept = labels_all[keep]
    uniq = np.unique(labels_kept)
    if uniq.size == 0:
        raise ValueError("No parcels left after filtering.")
    uniq.sort()
    id2idx = {int(k): i for i, k in enumerate(uniq)}
    roi_index = np.array([id2idx[int(l)] for l in labels_kept], dtype=np.int64)
    R = uniq.size
    
    # find indices (in a whole brain array) for voxels corresponding to labels/rois we left/kept
    kept_linear_idx = np.flatnonzero(mask.ravel())[keep]
    
    # build ROI table
    roi_table = []
    for i, lab in enumerate(uniq):
        nvox = (parc.ravel()[kept_linear_idx] == lab).sum()
        roi_table.append({'label': lab, 'n_vox': nvox})
    
    # ---- build data matrix: ROI x subject means of brain image ----
    M = np.zeros((R, len(brain_img_ls)), dtype=np.float32)

    for j, src in tqdm(enumerate(brain_img_ls),
                       total=len(brain_img_ls),
                       disable=False if len(brain_img_ls)>10 else True,
                       desc='Transforming volume images into data array' if len(brain_img_ls)>10 else None):
        arr = nib.load(src).get_fdata(dtype=dtype) if isinstance(src, str) else np.asarray(src, dtype=dtype)
        if arr.shape != shape:
            raise ValueError("All maps must match parcellation shape.")
        v = arr.ravel()[kept_linear_idx].astype(np.float64, copy=False)

        if reducer == "mean":
            sums = np.bincount(roi_index, weights=v, minlength=R)
            cnts = np.bincount(roi_index, minlength=R)
            vals = sums / np.maximum(cnts, 1)
        elif reducer == "median":
            vals = np.zeros(R, dtype=np.float64)
            for r in range(R):
                vals[r] = np.median(v[roi_index == r])
        else:
            raise ValueError("reducer must be 'mean' or 'median'.")

        M[:, j] = vals.astype(np.float32, copy=False)

    return {'roi_table':roi_table, 'roi_data': M}

def parcels_to_surface(
    parcel_values,
    parc_path,
    background_label=0,
    fill_value=0.0,
    return_gifti=False,
    out_path=None,
    split_cifti_hemis=False,
    cifti_hemi_struct_names=("CIFTI_STRUCTURE_CORTEX_LEFT",
                             "CIFTI_STRUCTURE_CORTEX_RIGHT"),
):
    """
    Map parcel-wise values back to vertex-wise surface space.

    This function supports three main scenarios:

    1) Single parcellation file (GIFTI or CIFTI)
       -----------------------------------------
       - parc_path: str, path to a .gii (GIFTI) or .nii (CIFTI) parcellation.
       - parcel_values: shape (n_parcels,) or (n_parcels, n_features)
         where n_parcels = number of unique non-background labels in the
         parcellation image.
       - Returns (when split_cifti_hemis == False):
           surf_data, label_order
         where surf_data is vertex-wise values in the same space as the
         parcellation:
           * GIFTI: surface vertices
           * CIFTI: grayordinates (cortex + subcortex, etc.)

       - If parc_path is CIFTI *and* split_cifti_hemis == True:
         then after mapping parcel_values to grayordinates, we use the
         CIFTI BrainModelAxis to split cortex into LH / RH surface arrays.
         Returns:
           (surf_lh, surf_rh), label_order
         Each element can be:
           * np.ndarray (if return_gifti == False)
           * GiftiImage (if return_gifti == True)

    2) Full-brain parcellation split into LH and RH GIFTI label files
       ----------------------------------------------------------------
       - parc_path: (lh_parc_path, rh_parc_path) or [lh_parc_path, rh_parc_path]
         (both must be GIFTI label files).
       - parcel_values: full-brain parcel array, shape (n_total_parcels,)
                        or (n_total_parcels, n_features), where
                        n_total_parcels = n_parcels_lh + n_parcels_rh.
         The row order is assumed to be:
             [all LH parcels (sorted unique non-background labels),
              all RH parcels (sorted unique non-background labels)]
       - Returns:
           (surf_lh, surf_rh), (labels_lh, labels_rh)

    In all cases, the row order of `parcel_values` is assumed to correspond to
    the sorted unique non-background labels in the parcellation(s).

    Parameters
    ----------
    parcel_values : np.ndarray
        Array with shape (n_parcels,) or (n_parcels, n_features).
    parc_path : str or sequence of str
        - str: path to a GIFTI (.gii) or CIFTI (.nii) parcellation file.
        - (lh_path, rh_path): paths to left- and right-hemisphere GIFTI
          parcellations.
    background_label : int or float or None, optional
        Label value treated as background (will be filled with `fill_value`).
        If None, no label will be treated as background.
    fill_value : float, optional
        Value assigned to background vertices / grayordinates.
    return_gifti : bool, optional
        If True:
          - For GIFTI parcellations: returns GiftiImage instead of np.ndarray.
          - For CIFTI + split_cifti_hemis: returns LH/RH GiftiImage(s).
          - For CIFTI without split: has no effect (still returns np.ndarray).
    out_path : str or sequence of str or None, optional
        If not None, save resulting GIFTI(s) with nibabel.save():
        - Single GIFTI parcellation:
            out_path: str (output func.gii path)
        - LH/RH GIFTI parcellations:
            out_path: (lh_out_path, rh_out_path)
        - CIFTI parcellation with split_cifti_hemis=True and return_gifti=True:
            out_path: (lh_out_path, rh_out_path)
    split_cifti_hemis : bool, optional
        Only relevant when parc_path is a CIFTI parcellation.
        If True:
            1) Map parcel_values -> grayordinates;
            2) Use BrainModelAxis to split cortex into LH/RH;
            3) Return (surf_lh, surf_rh).
    cifti_hemi_struct_names : tuple of str, optional
        Names of the CIFTI structures treated as LH and RH cortex.
        Default:
            ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT")

    Returns
    -------
    Case A: parc_path is (lh_path, rh_path)  [GIFTI parcellations]
        (surf_lh, surf_rh), (labels_lh, labels_rh)
        Each surf_* is either:
            - np.ndarray with shape (n_vertices_hemi,) or
              (n_vertices_hemi, n_features), or
            - GiftiImage, if return_gifti=True.

    Case B: parc_path is single GIFTI (.gii)
        surf_data, label_order
        surf_data: np.ndarray or GiftiImage (if return_gifti=True).

    Case C1: parc_path is CIFTI, split_cifti_hemis == False
        surf_data, label_order
        surf_data: np.ndarray in grayordinate space
                   shape (n_grayordinates,) or (n_grayordinates, n_features).

    Case C2: parc_path is CIFTI, split_cifti_hemis == True
        (surf_lh, surf_rh), label_order
        Each surf_* is either:
            - np.ndarray with shape (n_vertices_hemi,) or
              (n_vertices_hemi, n_features), or
            - GiftiImage, if return_gifti=True.

    Notes
    -----
    - For CIFTI parcellations, splitting into hemispheres uses the BrainModelAxis.
      Only cortical structures specified in `cifti_hemi_struct_names` are
      extracted. Subcortical grayordinates are ignored in the per-hemisphere
      outputs (but still present in the original grayordinate surf_data if you
      call with split_cifti_hemis=False).
    """

    # -----------------------------
    # Internal helper functions
    # -----------------------------
    def _prepare_parcel_values(parcel_values, expected_n_parcels):
        """Standardize parcel_values to 2D and check its first dimension."""
        arr = np.asarray(parcel_values)
        if arr.ndim == 1:
            if arr.shape[0] != expected_n_parcels:
                raise ValueError(
                    f"Length of parcel_values ({arr.shape[0]}) "
                    f"does not match expected #parcels ({expected_n_parcels})."
                )
            arr_2d = arr[:, np.newaxis]
            squeeze_output = True
        elif arr.ndim == 2:
            if arr.shape[0] != expected_n_parcels:
                raise ValueError(
                    f"parcel_values.shape[0] = {arr.shape[0]} "
                    f"does not match expected #parcels ({expected_n_parcels})."
                )
            arr_2d = arr
            squeeze_output = False
        else:
            raise ValueError(
                "parcel_values must be 1D (n_parcels,) or 2D (n_parcels, n_features)."
            )
        return arr_2d, squeeze_output

    def _map_single_parc_to_surface(
        parcel_values_2d,
        labels,
        template_img,
        background_label,
        fill_value,
        squeeze_output,
        return_gifti,
        out_path,
    ):
        """Map parcel-wise values to vertex-wise surface for a single parcellation."""
        n_vertices = labels.size

        # Determine unique parcel labels
        unique_labels = np.unique(labels)
        if background_label is not None:
            unique_labels = unique_labels[unique_labels != background_label]

        # Build vertex-to-parcel index map
        index_map = np.full(n_vertices, -1, dtype=int)
        for idx, lab in enumerate(unique_labels):
            index_map[labels == lab] = idx

        n_parcels, n_features = parcel_values_2d.shape

        # Sanity check: #parcels in values vs unique_labels
        if n_parcels != unique_labels.size:
            raise ValueError(
                f"Number of rows in parcel_values ({n_parcels}) does not "
                f"match number of unique non-background labels "
                f"in parcellation ({unique_labels.size})."
            )

        # Initialize vertex-wise data
        surf_data = np.full((n_vertices, n_features), fill_value, dtype=float)
        valid_mask = index_map >= 0

        # Fill non-background vertices
        surf_data[valid_mask, :] = parcel_values_2d[index_map[valid_mask], :]

        if squeeze_output and n_features == 1:
            surf_data = surf_data[:, 0]

        # If template is GIFTI and return_gifti is True, wrap as GiftiImage
        if isinstance(template_img, GiftiImage) and return_gifti:
            da = GiftiDataArray(data=surf_data.astype(np.float32))
            out_img = GiftiImage(darrays=[da])
            # Copy meta / labeltable if available
            out_img.meta = template_img.meta
            out_img.labeltable = template_img.labeltable
            if out_path is not None:
                nib.save(out_img, out_path)
            return out_img, unique_labels

        # Otherwise just return numpy array
        return surf_data, unique_labels

    def _load_parcellation_labels(path):
        """Load parcellation file and return labels (1D) and the image object."""
        img = nib.load(path)
        if isinstance(img, GiftiImage):
            labels = img.darrays[0].data.astype(int).ravel()
        elif isinstance(img, nib.cifti2.Cifti2Image):
            lab_data = img.get_fdata()
            lab_data = np.squeeze(lab_data)
            if lab_data.ndim != 1:
                raise ValueError(
                    "CIFTI parcellation is expected to be a 1D label map "
                    "(shape (1, n_vertices) or (n_vertices,))."
                )
            labels = lab_data.astype(int).ravel()
        else:
            raise TypeError(
                "Unsupported parcellation type. "
                "Expected a GIFTI (.gii) or CIFTI (.nii) file."
            )
        return labels, img

    def _split_cifti_surface_by_hemis(
        surf_data,
        cifti_img,
        fill_value,
        return_gifti,
        out_path,
        hemi_struct_names,
    ):
        """
        Split CIFTI grayordinate data into per-hemisphere surface arrays or GiftiImages.

        Parameters
        ----------
        surf_data : np.ndarray
            Grayordinate-wise data, shape (n_grayordinates,) or
            (n_grayordinates, n_features). First axis must correspond to the
            BrainModelAxis of the CIFTI image.
        cifti_img : nib.cifti2.Cifti2Image
            Original CIFTI image providing the BrainModelAxis.
        fill_value : float
        return_gifti : bool
        out_path : None or (lh_path, rh_path)
        hemi_struct_names : sequence of str
            Names of CIFTI structures to be treated as LH and RH cortex.

        Returns
        -------
        surf_lh, surf_rh : np.ndarray or GiftiImage
        """
        arr = np.asarray(surf_data)
        if arr.ndim == 1:
            arr_2d = arr[:, np.newaxis]
            squeeze_output = True
        elif arr.ndim == 2:
            arr_2d = arr
            squeeze_output = False
        else:
            raise ValueError(
                "surf_data must be 1D (n_grayordinates,) or "
                "2D (n_grayordinates, n_features) for CIFTI splitting."
            )

        # Find BrainModelAxis
        axes = [cifti_img.header.get_axis(i) for i in range(cifti_img.ndim)]
        bm_axis = None
        for ax in axes:
            # In nibabel >= 3 this is nib.cifti2.BrainModelAxis
            if isinstance(ax, nib.cifti2.BrainModelAxis):
                bm_axis = ax
                break
        if bm_axis is None:
            raise TypeError(
                "CIFTI image does not contain a BrainModelAxis; "
                "cannot split into hemispheres."
            )

        # Handle output paths for hemis
        if out_path is not None:
            if not isinstance(out_path, (list, tuple)) or \
               len(out_path) != len(hemi_struct_names):
                raise ValueError(
                    "For CIFTI hemisphere split, out_path must be a "
                    "(lh_out_path, rh_out_path) pair or None."
                )
            hemi_out_paths = list(out_path)
        else:
            hemi_out_paths = [None] * len(hemi_struct_names)

        hemi_results = []

        for hemi_name, hemi_out_path in zip(hemi_struct_names, hemi_out_paths):
            found = False
            # Iterate over volumetric and surface structures
            for name, data_indices, model in bm_axis.iter_structures():
                if name == hemi_name:
                    found = True
                    # data_indices index into the grayordinate axis
                    data_hemi = arr_2d[data_indices, :]  # (n_coords_hemi, n_features)

                    # For cortical surfaces, model.vertex gives vertex indices
                    vtx_indices = model.vertex  # 1D array
                    n_vertices_hemi = int(vtx_indices.max()) + 1

                    hemi_arr = np.full(
                        (n_vertices_hemi, arr_2d.shape[1]),
                        fill_value,
                        dtype=arr_2d.dtype,
                    )
                    hemi_arr[vtx_indices, :] = data_hemi

                    if squeeze_output and arr_2d.shape[1] == 1:
                        hemi_arr = hemi_arr[:, 0]

                    if return_gifti:
                        da = GiftiDataArray(data=hemi_arr.astype(np.float32))
                        hemi_img = GiftiImage(darrays=[da])
                        if hemi_out_path is not None:
                            nib.save(hemi_img, hemi_out_path)
                        hemi_results.append(hemi_img)
                    else:
                        hemi_results.append(hemi_arr)

                    break

            if not found:
                raise ValueError(
                    f"No structure named {hemi_name} found in "
                    "CIFTI BrainModelAxis."
                )

        # Expect exactly two structures for LH/RH
        if len(hemi_results) != 2:
            raise RuntimeError(
                "Unexpected number of hemisphere results; "
                "expected 2 (LH and RH)."
            )

        return hemi_results[0], hemi_results[1]

    # ------------------------------------------------------------------
    # Case 1: LH/RH GIFTI parcellations for a full-brain parcel vector
    # ------------------------------------------------------------------
    if isinstance(parc_path, (list, tuple)):
        if len(parc_path) != 2:
            raise ValueError(
                "parc_path must be a single string or a (lh_path, rh_path) pair."
            )
        lh_parc_path, rh_parc_path = parc_path

        # Load LH and RH parcellations
        labels_lh, img_lh = _load_parcellation_labels(lh_parc_path)
        labels_rh, img_rh = _load_parcellation_labels(rh_parc_path)

        # Ensure both are GIFTI
        if not isinstance(img_lh, GiftiImage) or not isinstance(img_rh, GiftiImage):
            raise TypeError(
                "When parc_path is a tuple/list, both files must be GIFTI (.gii)."
            )

        # Unique labels per hemisphere (for counting parcels)
        unique_lh = np.unique(labels_lh)
        unique_rh = np.unique(labels_rh)
        if background_label is not None:
            unique_lh = unique_lh[unique_lh != background_label]
            unique_rh = unique_rh[unique_rh != background_label]

        n_lh = unique_lh.size
        n_rh = unique_rh.size
        n_total = n_lh + n_rh

        # Standardize parcel_values to 2D and check total #parcels
        parcel_values_2d, squeeze_output = _prepare_parcel_values(
            parcel_values, expected_n_parcels=n_total
        )

        # Split full-brain parcel_values into LH and RH parts
        values_lh = parcel_values_2d[:n_lh, :]
        values_rh = parcel_values_2d[n_lh:, :]

        # Handle separate out_paths for LH/RH (if provided)
        lh_out_path = rh_out_path = None
        if out_path is not None:
            if isinstance(out_path, (list, tuple)) and len(out_path) == 2:
                lh_out_path, rh_out_path = out_path
            else:
                raise ValueError(
                    "For LH/RH GIFTI parcellations, out_path must be "
                    "a (lh_out_path, rh_out_path) pair or None."
                )

        # Map LH
        surf_lh, labels_lh_sorted = _map_single_parc_to_surface(
            parcel_values_2d=values_lh,
            labels=labels_lh,
            template_img=img_lh,
            background_label=background_label,
            fill_value=fill_value,
            squeeze_output=squeeze_output,
            return_gifti=return_gifti,
            out_path=lh_out_path,
        )

        # Map RH
        surf_rh, labels_rh_sorted = _map_single_parc_to_surface(
            parcel_values_2d=values_rh,
            labels=labels_rh,
            template_img=img_rh,
            background_label=background_label,
            fill_value=fill_value,
            squeeze_output=squeeze_output,
            return_gifti=return_gifti,
            out_path=rh_out_path,
        )

        return (surf_lh, surf_rh), (labels_lh_sorted, labels_rh_sorted)

    # ------------------------------------------------------------------
    # Case 2: Single parcellation file (CIFTI full brain or GIFTI single hemi)
    # ------------------------------------------------------------------
    else:
        labels, img = _load_parcellation_labels(parc_path)

        # Determine unique labels for checking
        unique_labels = np.unique(labels)
        if background_label is not None:
            unique_labels = unique_labels[unique_labels != background_label]

        # Standardize parcel_values to 2D
        parcel_values_2d, squeeze_output = _prepare_parcel_values(
            parcel_values, expected_n_parcels=unique_labels.size
        )

        # For a single parcellation file, out_path only applies if template is GIFTI
        single_out_path = None
        if out_path is not None and isinstance(img, GiftiImage):
            if isinstance(out_path, str):
                single_out_path = out_path
            else:
                raise ValueError(
                    "For a single GIFTI parcellation, out_path must be a string or None."
                )

        # First: map parcels -> vertices (GIFTI) or grayordinates (CIFTI)
        surf_data, sorted_labels = _map_single_parc_to_surface(
            parcel_values_2d=parcel_values_2d,
            labels=labels,
            template_img=img,
            background_label=background_label,
            fill_value=fill_value,
            squeeze_output=squeeze_output,
            return_gifti=return_gifti,
            out_path=single_out_path,
        )

        # If this is a CIFTI parcellation and we want LH/RH split
        if isinstance(img, nib.cifti2.Cifti2Image) and split_cifti_hemis:
            # For CIFTI, surf_data here is always np.ndarray (return_gifti is ignored)
            surf_lh, surf_rh = _split_cifti_surface_by_hemis(
                surf_data=surf_data,
                cifti_img=img,
                fill_value=fill_value,
                return_gifti=return_gifti,
                out_path=out_path,
                hemi_struct_names=cifti_hemi_struct_names,
            )
            return (surf_lh, surf_rh), sorted_labels

        # Otherwise: just return the single surface/grayordinate array
        return surf_data, sorted_labels