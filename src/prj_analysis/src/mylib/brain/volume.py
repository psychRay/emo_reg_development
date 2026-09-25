#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 03:01:08 2025

    sorts of flexible functions to perform brain imaging analyses of volume data

@author: dingrui
"""
# modules
import warnings
warnings.filterwarnings('ignore', message='.*deprecated.*')

import os, re
import numpy as np
import pandas as pd
import nibabel as nib

from typing import Dict, List, Optional, Union, Tuple
from scipy.io import loadmat
from scipy.stats import t as tdist, norm
from nilearn.image import new_img_like
# from nilearn.maskers import NiftiLabelsMasker

# functions
def get_spm_struct(spm_mat_path):
    
    """
    Load SPM.mat and return the SPM struct (scipy.io 'bunch' style)
    """
    
    md = loadmat(spm_mat_path, squeeze_me=True, struct_as_record=False)
    if "SPM" not in md:
        raise RuntimeError("SPM variable not found in SPM.mat")
        
    return md["SPM"]

def find_canonical_cols(colnames, cond_name, include_derivs=False):
    
    """
    Find column indices for a given condition's canonical HRF (bf(1))
    If include_derivs=True, include bf(1|2|3) with weights you'd set later (default: only bf(1))
    """
    
    cond_esc = re.escape(cond_name)
    if include_derivs:
        pat = re.compile(rf"^Sn\(\d+\)\s+{cond_esc}\*bf\((1|2|3)\)$")
    else:
        pat = re.compile(rf"^Sn\(\d+\)\s+{cond_esc}\*bf\(1\)$")
    idx = [i for i, name in enumerate(colnames) if pat.search(name)]
    
    return idx

def to_nifti(ref_img, data):
    
    """
    Write NIfTI with ref_img's affine/header; 
    (cast to float32)
    """
    
    out = nib.Nifti1Image(
        np.asarray(data, dtype=np.float32), 
        ref_img.affine, 
        ref_img.header
    )
    return out

def make_effvar(
    fpath_spmmat, fpath_ResMS,
    name_conds,
    # nameA,
    # nameB,
    include_derivs=False,
    savedir=None,
    out_prefix="",
):
    """
    Generate effect variance maps for conditions of interests
    
    Parameters
    ----------
    fpath_spmmat, fpath_ResMS : Path
        fullpath for 1st-level SPM.mat and ResMS.nii
    name_conds : list
        Condition names list specified in SPM 1st-level analysis (match SPM.xX.name, e.g., 'A', 'B')
    savedir : Path
        the directory provided for saving the estimated effect variance map
    include_derivs : bool
        if True, include bf(2)/bf(3) columns as well (you can later adjust weights)
        default False (use canonical bf(1) only).
    out_prefix : str
        optional filename prefix, e.g., 'sub-001_'
        
    """

    if not os.path.exists(fpath_spmmat):
        raise FileNotFoundError(f"SPM.mat not found at {fpath_spmmat}")
    if not os.path.exists(fpath_ResMS):
        raise FileNotFoundError(f"ResMS.nii not found at {fpath_ResMS}")

    spmmat = get_spm_struct(str(fpath_spmmat))
    colnames = spmmat.xX.name.tolist()
    P = len(colnames)

    # Find canonical HRF columns per condition
    idx_conds = {}
    for name in name_conds:
        idx_conds[name] = find_canonical_cols(colnames, name, include_derivs=include_derivs)
        if idx_conds[name] == 0:
            raise RuntimeError(
                f"Cannot find canonical columns for {name}. "
                f"Check condition names and bf(1) in SPM.xX.name."
            )

    # Effect variance = ResMS * (c' * Bcov * c)
    # Bcov --> covariance matrix (P x P):
    Bcov = np.array(spmmat.xX.Bcov, dtype=np.float32)
    # ResMS (voxelwise residual mean squares):
    resms_img = nib.load(fpath_ResMS)
    resms     = resms_img.get_fdata(dtype=np.float32)
    # (safety) ensure shapes
    if Bcov.shape != (P, P):
        raise RuntimeError(f"Unexpected Bcov shape {Bcov.shape}, expected ({P},{P})")
    
    # Effect variance = ResMS * (c' * Bcov * c)
    effvar_dict = {}
    effvec_chek = {}
    for name_cond, idx in idx_conds.items():
        # Build equal-weight contrast vectors across runs (weights sum to 1 within a condition)
        wCV_cond = np.zeros(P, dtype=np.float32)
        wCV_cond[idx] = 1.0 / len(idx)
        effvec_chek[f'contrast vector for {name_cond}'] = wCV_cond
        
        # Compute effect variance
        # scalars (design-based variance factors):
        scale_cond = float(wCV_cond @ Bcov @ wCV_cond)
        effvar_cond = to_nifti(resms_img, resms * scale_cond)
        effvar_dict[f'effect variance for {name_cond}'] = effvar_cond
    
    # save effect variance maps into local disk
    if savedir is not None:
        for name_cond, nii in effvar_dict.items():
            fname = [name for name in name_conds if name in name_cond][0]
            nii.to_filename(
                os.path.join(savedir, f'{out_prefix}_effvar_{fname}.nii.gz')
            )
    
    return effvar_dict, effvec_chek


def compute_fixed_effects_fast(
    effect_size_imgs: List[str] or List[nib.Nifti1Image],
    effect_varc_imgs: List[str] or List[nib.Nifti1Image],
    brain_mask: Optional[str] or Optional[nib.Nifti1Image] = None,
    out_paths: Optional[Tuple[str, str, str]] = None,
    dtype=np.float32,
    chunk_slices: int = 64,
    var_floor: float = 1e-6,
) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, nib.Nifti1Image]:
    
    """
    Fixed-effects inverse-variance combination across runs.
    effect_i: per-run effect/COPE/β map (linearly combinable quantity, NOT t/z).
    var_i   : per-run effect-variance map (same space as effect_i).

    Returns
    -------
    eff_img, var_img, t_img : NIfTI images
    """
    
    e_imgs = [nib.load(p) if isinstance(p, str) else p for p in effect_size_imgs]
    v_imgs = [nib.load(p) if isinstance(p, str) else p for p in effect_varc_imgs]

    # Basic sanity: same shape/affine
    shape  = e_imgs[0].shape
    affine = e_imgs[0].affine
    for im in e_imgs[1:] + v_imgs:
        if im.shape != shape:
            raise ValueError("All inputs must have identical shapes (avoid resampling).")
        if not np.allclose(im.affine, affine):
            raise ValueError("All inputs must have identical affines (avoid resampling).")

    # Build mask: provided mask ∩ finite ∩ var>0
    if brain_mask is not None:
        if isinstance(brain_mask, os.PathLike):
            mask_data = nib.load(brain_mask).get_fdata()
        else:
            mask_data = brain_mask.get_fdata()
            if mask_data.shape != shape:
                raise ValueError("Mask shape mismatch.")
    else:
        mask_data = None

    # Prepare outputs
    eff = np.zeros(shape, dtype=dtype)
    var = np.zeros(shape, dtype=dtype)

    # Chunk along the last axis to bound memory
    Z = shape[-1]
    for z0 in range(0, Z, chunk_slices):
        z1 = min(Z, z0 + chunk_slices)
        # Stack K runs in memory for this chunk
        E = np.stack([im.dataobj[..., z0:z1].astype(dtype, copy=False) for im in e_imgs], axis=-1)
        V = np.stack([vi.dataobj[..., z0:z1].astype(dtype, copy=False) for vi in v_imgs], axis=-1)

        # Floor the variance to keep weights bounded
        V = np.maximum(V, var_floor)

        W = 1.0 / V  # weights

        # Weighted mean and its variance
        sumW = np.sum(W, axis=-1)
        num = np.sum(W * E, axis=-1)
        eff_chunk = num / sumW
        var_chunk = 1.0 / sumW

        # Apply mask (if provided) and basic validity filter
        if mask_data is not None:
            m = np.bool(mask_data[..., z0:z1])
        else:
            # valid where all finite and sumW>0
            m = np.isfinite(eff_chunk) & np.isfinite(var_chunk) & (sumW > 0)

        out_e = np.zeros_like(eff_chunk, dtype=dtype)
        out_v = np.zeros_like(var_chunk, dtype=dtype)
        out_e[m] = eff_chunk[m]
        out_v[m] = var_chunk[m]

        eff[..., z0:z1] = out_e
        var[..., z0:z1] = out_v

    # t map
    with np.errstate(divide="ignore", invalid="ignore"):
        t_arr = eff / np.sqrt(var)
        t_arr[~np.isfinite(t_arr)] = 0

    # Wrap as NIfTI
    ref_hdr = e_imgs[0].header
    eff_img = nib.Nifti1Image(eff, affine, ref_hdr)
    var_img = nib.Nifti1Image(var, affine, ref_hdr)
    t_img   = nib.Nifti1Image(t_arr.astype(dtype, copy=False), affine, ref_hdr)

    # Optional I/O
    if out_paths is not None:
        e_out, v_out, t_out = out_paths
        if e_out: 
            if not os.path.exists(os.path.dirname(e_out)):
                os.mkdir(os.path.dirname(e_out))
            nib.save(eff_img, e_out)
        if v_out: 
            if not os.path.exists(os.path.dirname(v_out)):
                os.mkdir(os.path.dirname(v_out))
            nib.save(var_img, v_out)
        if t_out:
            if not os.path.exists(os.path.dirname(t_out)):
                os.mkdir(os.path.dirname(t_out))
            nib.save(t_img, t_out)

    return eff_img, var_img, t_img

def glm_age_t(
        img_paths, 
        age, 
        mask_path, 
        out_t_path=None, out_beta_path=None, dtype=np.float32):
    """
    Streaming OLS for Y ~ 1 + age (age should be centered).
    Computes the t map for 'age' (and optionally beta_age image) without stacking all subjects in memory.
    """
    age = np.asarray(age, dtype=np.float64)
    N = age.size
    assert len(img_paths) == N
    
    if isinstance(mask_path, str):
        mask_img = nib.load(mask_path)
    else:
        mask_img = mask_path

    mask = mask_img.get_fdata().astype(bool)
    nvox = mask.sum()

    # Precompute (X'X)^-1 for X = [1, age] with centered age (sum(age)=0)
    age_c = age - age.mean()
    S_aa = np.dot(age_c, age_c)            # sum(age^2)
    XTX_inv = np.linalg.inv(np.array([[N, 0.0],[0.0, S_aa]]))  # because age is centered
    # Accumulators per voxel (in mask)
    XTy0 = np.zeros(nvox, dtype=np.float64)  # sum(y)
    XTy1 = np.zeros(nvox, dtype=np.float64)  # sum(age*y)
    yTy  = np.zeros(nvox, dtype=np.float64)  # sum(y^2)

    # Stream over subjects
    for i, p in enumerate(img_paths):
        y3d = nib.load(str(p)).get_fdata(dtype=dtype)
        y = y3d[mask].astype(np.float64, copy=False)
        a = age_c[i]
        XTy0 += y
        XTy1 += a * y
        yTy  += y * y

    # Betas per voxel: beta = (X'X)^-1 X'Y
    beta0 = XTX_inv[0,0]*XTy0 + XTX_inv[0,1]*XTy1
    beta1 = XTX_inv[1,0]*XTy0 + XTX_inv[1,1]*XTy1   # slope for age

    # SSE = y'y - beta^T X'Y = yTy - (beta0*XTy0 + beta1*XTy1)
    SSE = yTy - (beta0 * XTy0 + beta1 * XTy1)
    dof = N - 2
    sigma2 = np.maximum(SSE / dof, 1e-12)          # variance of residuals
    # Var(beta_age) = sigma2 * (X'X)^-1[1,1] = sigma2 * (1/S_aa)
    se_beta1 = np.sqrt(sigma2 * XTX_inv[1,1])
    t_age_map = np.zeros(mask.shape, dtype=dtype)
    t_age_map[mask] = (beta1 / np.maximum(se_beta1, 1e-12)).astype(dtype, copy=False)

    if out_beta_path:
        beta_age_img = np.zeros(mask.shape, dtype=dtype)
        beta_age_img[mask] = beta1.astype(dtype, copy=False)
        nib.save(nib.Nifti1Image(beta_age_img, mask_img.affine, mask_img.header), out_beta_path)
    
    if out_t_path:
        nib.save(nib.Nifti1Image(t_age_map, mask_img.affine, mask_img.header), out_t_path)
    
    return nib.Nifti1Image(t_age_map, mask_img.affine, mask_img.header)

def glm_AmB_with_covariates(
    A_paths: List[str],
    B_paths: List[str],
    X: pd.DataFrame,
    mask_path: str,
    contrasts: None,
    out_t_paths: None,
    out_effect_paths: None,
    out_beta_stack_path: None,
    center_non_intercept: bool = True,
    intercept_first: bool = True,
    dtype=np.float32,
    var_floor: float = 1e-12,
):
    """
    Streaming OLS for Δ = A - B with arbitrary subject-level covariates:
        Δ ~ 1 + covariates

    Parameters
    ----------
    A_paths, B_paths : list[str]
        Per-subject paths to A and B beta maps (same space/resolution).
    X : (N, p) array or pandas.DataFrame
        Design matrix with p columns. MUST include an intercept column.
        Recommended: center non-intercept columns so that the intercept is the adjusted mean at covariate=0.
    mask_path : str
        Brain mask NIfTI. Computation is restricted to voxels where mask!=0.
    contrasts : dict[name -> (p,)]
        Optional. Linear contrasts over beta (length p). Examples:
          - adjusted mean of Δ (intercept): c = [1,0,0,...]
          - age effect in Δ:               c = [0,1,0,...] (if age is 2nd col)
          - custom combos allowed.
        If None, will default to {"intercept": e0}.
    out_t_paths : dict[name -> path]
        Optional. Where to save t-maps for each contrast.
    out_effect_paths : dict[name -> path]
        Optional. Where to save effect maps (c'β) for each contrast.
    out_beta_stack_path : str
        Optional. Save the full beta stack as a 4D NIfTI with p volumes (diagnostics).
    center_non_intercept : bool
        If True, mean-center all non-intercept columns (recommended).
    intercept_first : bool
        If True, assume the first column of X is the intercept.
    dtype : np.dtype
        Floating dtype for image IO and outputs.
    var_floor : float
        Lower bound for variance to avoid division-by-zero.

    Returns
    -------
    results : dict
        Contains 'beta' (p x nvox array), 'XtX_inv' (p x p), 'sigma2' (nvox,),
        and per-contrast numerics under 'contrasts'.
    """
    # ---------- prep design ----------
    if "pd" in str(type(X)):
        X_names = list(X.columns)
        X = X.values
    else:
        X_names = [f"x{i}" for i in range(X.shape[1])]

    X = np.asarray(X, dtype=np.float64)
    N, p = X.shape
    assert len(A_paths) == len(B_paths) == N, "A/B paths and X rows must match."

    # ensure intercept present
    if intercept_first:
        assert np.allclose(X[:, 0], 1.0), "First column must be intercept (all ones)."
        start_j = 1
    else:
        # try to detect an all-ones column
        ones_cols = np.where(np.all(np.isclose(X, 1.0), axis=0))[0]
        if len(ones_cols) == 0:
            raise ValueError("Design matrix must include an intercept column of all ones.")
        # move first such column to position 0 for convenience
        j0 = int(ones_cols[0])
        if j0 != 0:
            X[:, [0, j0]] = X[:, [j0, 0]]
            X_names[0], X_names[j0] = X_names[j0], X_names[0]
        start_j = 1

    # center non-intercept columns to make intercept an adjusted mean at covariate=0
    if center_non_intercept and p > 1:
        X[:, 1:] = X[:, 1:] - X[:, 1:].mean(axis=0, keepdims=True)

    # Precompute design cross-products
    XtX = X.T @ X                        # (p x p)
    # Check rank; use pinv if ill-conditioned but warn the user
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX, rcond=1e-12)
        print("[WARN] X'X not invertible; using pseudo-inverse. Check collinearity.")

    # ---------- prep mask & accumulators ----------
    if isinstance(mask_path, str):
        mask_img = nib.load(mask_path)
    else:
        mask_img = mask_path

    mask = mask_img.get_fdata().astype(bool)
    nvox = int(mask.sum())

    # Accumulate X'Y (p x nvox) and Y'Y (nvox) in mask
    XTy = np.zeros((p, nvox), dtype=np.float64)
    YTY = np.zeros(nvox, dtype=np.float64)

    # ---------- streaming over subjects ----------
    for i, (ap, bp) in enumerate(zip(A_paths, B_paths)):
        Ai = nib.load(ap).get_fdata(dtype=dtype)
        Bi = nib.load(bp).get_fdata(dtype=dtype)
        d = (Ai - Bi)[mask].astype(np.float64, copy=False)  # Δ_i in mask

        # Update X'Y and Y'Y
        YTY += d * d
        # Outer product d with row X[i,:], but keep memory low by broadcasting:
        # XTy[j, :] += X[i, j] * d
        XTy += (X[i, :].reshape(-1, 1) * d.reshape(1, -1))

    # ---------- solve per-voxel OLS ----------
    # beta (p x nvox) = (X'X)^-1 (X'Y)
    beta = XtX_inv @ XTy

    # SSE per voxel = Y'Y - beta^T (X'Y) = Y'Y - sum_j beta_j * XTy_j
    SSE = YTY - np.sum(beta * XTy, axis=0)
    dof = max(N - p, 1)
    sigma2 = np.maximum(SSE / dof, var_floor)       # (nvox,)

    # ---------- package outputs ----------
    results = {
        "beta": beta,                # shape (p, nvox)
        "XtX_inv": XtX_inv,          # (p x p)
        "sigma2": sigma2,            # (nvox,)
        "names": X_names,
        "dof": dof,
        "mask_shape": mask.shape,
    }

    # Optionally save beta stack as 4D NIfTI (vol 0..p-1)
    if out_beta_stack_path:
        beta_vol = np.zeros(mask.shape + (p,), dtype=dtype)
        for j in range(p):
            tmp = np.zeros(mask.shape, dtype=dtype)
            tmp[mask] = beta[j].astype(dtype, copy=False)
            beta_vol[..., j] = tmp
        nib.save(nib.Nifti1Image(beta_vol, mask_img.affine, mask_img.header), out_beta_stack_path)

    # ---------- contrasts ----------
    if contrasts is None:
        # default: adjusted mean of Δ (intercept)
        c = np.zeros((1, p), dtype=np.float64); c[0, 0] = 1.0
        contrasts = {"intercept": c[0]}

    results["contrasts"] = {}
    for name, c in contrasts.items():
        c = np.asarray(c, dtype=np.float64).reshape(-1)   # (p,)
        if c.size != p:
            raise ValueError(f"Contrast '{name}' has length {c.size}, expected {p}.")
        # effect = c'β (nvox,)
        eff = c @ beta
        # se(effect) = sqrt( sigma2 * c'(X'X)^-1 c )
        scale = float(c @ XtX_inv @ c)
        se = np.sqrt(np.maximum(sigma2 * scale, var_floor))
        t = eff / np.maximum(se, np.sqrt(var_floor))
        
        vol_eff = np.zeros(mask.shape, dtype=dtype); vol_eff[mask] = eff.astype(dtype, copy=False)
        vol_t   = np.zeros(mask.shape, dtype=dtype); vol_t[mask]   = t.astype(dtype, copy=False)
        
        # write outputs if requested
        if out_effect_paths and name in out_effect_paths and out_effect_paths[name]:
            nib.save(nib.Nifti1Image(vol_eff, mask_img.affine, mask_img.header), out_effect_paths[name])
        if out_t_paths and name in out_t_paths and out_t_paths[name]:          
            nib.save(nib.Nifti1Image(vol_t, mask_img.affine, mask_img.header), out_t_paths[name])

        results["contrasts"][name] = {
            "effect": nib.Nifti1Image(vol_eff, mask_img.affine, mask_img.header), 
            "t": nib.Nifti1Image(vol_t, mask_img.affine, mask_img.header), 
            "scale": scale}

    return results

def glm_AmB_roi(
    A_paths: List[str],
    B_paths: List[str],
    X: Union[np.ndarray, "pd.DataFrame"],   # (N, p), col0 must be ones
    mask_path: str,
    parcellation_img_path: str,
    *,
    contrasts: Optional[Dict[str, np.ndarray]] = None,   # name -> (p,)
    include_labels: Optional[List[int]] = None,
    exclude_labels: Optional[List[int]] = None,
    dtype=np.float32,
    var_floor: float = 1e-12,
    save_dir: Optional[str] = None,
):
    
    """
    Streaming ROI-wise OLS for Δ = A - B with arbitrary subject-level covariates:
    ROI-mean(Δ)  ~  1 + covariates

    This function ONLY supports ROI (parcellation) mode for simplicity & maintainability.
    It returns, for each contrast, parcel-constant NIfTI maps (effect / t) and ROI vectors.

    Parameters
    ----------
    A_paths, B_paths : list[str]
        Per-subject paths to A and B beta maps (same geometry).
    X : (N, p) array or pandas.DataFrame
        Design matrix with p columns. MUST include an intercept column of ones.
        Recommended: center non-intercept columns so the intercept is an adjusted mean.
    mask_path : str
        Brain mask NIfTI; computations are restricted to voxels where mask!=0.
    parcellation_img_path : str
        Parcellation NIfTI (label image). Must match mask/data shape & affine.
        Label 0 is treated as background (excluded by default).
    contrasts : dict[name -> (p,)], optional
        Linear contrasts over beta. If None => {"intercept": e0}.
        Example for age slope (assuming columns=[intercept, age, ...]): [0,1,0,...]
    center_non_intercept : bool
        Mean-center non-intercept regressors (recommended).
    intercept_first : bool
        If True, expects intercept to be the first column; otherwise will try to move an all-ones
        column to the first position.
    dtype : np.dtype
        Floating dtype for image IO and outputs.
    var_floor : float
        Variance floor to avoid division-by-zero.
    parcel_include / parcel_exclude : list[int], optional
        Keep / drop specific parcel IDs.
    label_names : dict[int -> str], optional
        Human-readable names for ROI IDs (reported in results['roi_info'] only).
    save_dir : str, optional
        If provided, write NIfTI maps (effect/t) for each contrast into this directory.

    Returns
    -------
    results : dict
        keys:
          'names'     : list of design column names
          'dof'       : degrees of freedom (N - p)
          'roi_ids'   : (R,) int array of kept label IDs (sorted)
          'roi_counts': (R,) voxel counts per ROI
          'roi_info'  : list of dicts (id, name, n_vox)
          'XtX_inv'   : (p,p) design inverse
          For each contrast 'c':
             results['contrasts'][c]['effect_roi'] : (R,) effect per ROI (c'β)
             results['contrasts'][c]['t_roi']      : (R,) t per ROI
             results['contrasts'][c]['effect_map'] : NIfTI (parcel-constant)
             results['contrasts'][c]['t_map']      : NIfTI (parcel-constant)
    """
    
    # ----- Design -----
    if "pd" in str(type(X)):
        names = list(X.columns); X = X.values.copy().astype(np.float64)
    else:
        names = [f"x{i}" for i in range(X.shape[1])]
        X = np.asarray(X, dtype=np.float64)
    N, p = X.shape
    if not np.allclose(X[:, 0], 1.0):
        raise ValueError("First design column must be an intercept (all ones).")
    if len(A_paths) != N or len(B_paths) != N:
        raise ValueError("A_paths/B_paths length must match rows of X.")

    # ----- Geometry / labels -----
    if isinstance(mask_path, str):
        mask_img = nib.load(mask_path)
    else:
        mask_img = mask_path
        
    mask = mask_img.get_fdata().astype(bool)
    
    if isinstance(parcellation_img_path, str):
        parc_img = nib.load(parcellation_img_path)
    else:
        parc_img = parcellation_img_path
        
    parc_f = parc_img.get_fdata()
    
    if parc_img.shape != mask_img.shape or not np.allclose(parc_img.affine, mask_img.affine):
        raise ValueError("Parcellation must match mask shape & affine exactly.")
    # force integer labels; if fractional 
    parc = np.rint(parc_f).astype(np.int64)

    # Build kept-voxel index under mask and non-zero labels (+ include/exclude)
    mask_flat = mask.ravel()
    labels_all = parc.ravel()[mask_flat]
    keep = labels_all != 0
    if include_labels is not None:
        keep &= np.isin(labels_all, np.asarray(include_labels, np.int64))
    if exclude_labels is not None:
        keep &= ~np.isin(labels_all, np.asarray(exclude_labels, np.int64))

    # vector of kept labels and their flat indices
    labels_kept = labels_all[keep]
    kept_lin_idx = np.flatnonzero(mask_flat)[keep]   # indices in the full flat volume (unique 1:1)
    if labels_kept.size == 0:
        raise ValueError("After filtering, no voxel remains under mask & parcellation.")

    uniq = np.unique(labels_kept); uniq.sort()
    R = uniq.size
    if R < 2:
        raise ValueError(
            f"Only {R} ROI left after filtering. "
            "Check your parcellation (non-integer labels? wrong atlas? include/exclude too strict?)."
        )
    id2idx = {int(k): i for i, k in enumerate(uniq)}
    # ROI index (0..R-1) for each kept voxel (same order as kept_lin_idx)
    roi_index = np.array([id2idx[int(l)] for l in labels_kept], dtype=np.int64)

    # ----- Per-ROI accumulators -----
    XtX = np.zeros((R, p, p), dtype=np.float64)
    XTy = np.zeros((R, p),     dtype=np.float64)
    YTY = np.zeros(R,          dtype=np.float64)
    N_r = np.zeros(R,          dtype=np.int64)

    # ----- Stream subjects -----
    for i, (ap, bp) in enumerate(zip(A_paths, B_paths)):
        Ai = nib.load(ap).get_fdata(dtype=dtype)
        Bi = nib.load(bp).get_fdata(dtype=dtype)
        if Ai.shape != mask_img.shape or Bi.shape != mask_img.shape:
            raise ValueError("All A/B images must exactly match mask shape.")
        d = (Ai - Bi).ravel()[kept_lin_idx].astype(np.float64, copy=False)
        finite = np.isfinite(d)  # per-voxel finite mask
        if not finite.any():
            continue

        idx_i = roi_index[finite]   # ROI id per finite voxel
        d_i   = d[finite]
        sums = np.bincount(idx_i, weights=d_i, minlength=R)
        cnts = np.bincount(idx_i, minlength=R)
        valid_roi = cnts > 0
        if not np.any(valid_roi):
            continue

        means = np.zeros(R, dtype=np.float64)
        means[valid_roi] = sums[valid_roi] / cnts[valid_roi]

        xi = X[i, :]
        X_outer = np.outer(xi, xi)        # (p,p)
        vr = np.where(valid_roi)[0]       # ROIs that have valid mean for this subject
        XtX[vr] += X_outer                # broadcast add
        XTy[vr] += np.outer(means[vr], xi)  # (k,p)
        YTY[vr] += means[vr] ** 2
        N_r[vr] += 1

    # ----- Solve per ROI -----
    beta   = np.full((R, p), np.nan, dtype=np.float64)
    sigma2 = np.full(R,     np.nan, dtype=np.float64)
    dof_r  = np.maximum(N_r - p, 0)
    XtX_inv_list = [None] * R

    for r in range(R):
        if N_r[r] <= p:
            continue
        try:
            XtX_inv = np.linalg.inv(XtX[r])
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX[r], rcond=1e-12)
        XtX_inv_list[r] = XtX_inv
        br = XtX_inv @ XTy[r]
        SSEr = YTY[r] - br @ XTy[r]
        beta[r] = br
        sigma2[r] = max(SSEr / max(dof_r[r], 1), var_floor)

    # ----- Helpers -----
    def _paint(vec: np.ndarray) -> nib.Nifti1Image:
        """Write an ROI vector back to 3D using kept_lin_idx and roi_index (parcel-constant)."""
        out = np.zeros(mask_img.shape, dtype=dtype)
        flat = out.ravel()
        # per kept voxel value = vec[ roi_index ]
        flat[kept_lin_idx] = vec.astype(dtype, copy=False)[roi_index]
        return nib.Nifti1Image(out, mask_img.affine, mask_img.header)

    results = {
        "names": names,
        "roi_ids": uniq.astype(int),
        "beta": beta,                    # (R, p)
        "sigma2": sigma2,                # (R,)
        "N_eff_per_roi": N_r,            # (R,)
        "dof_per_roi": dof_r,            # (R,)
        "contrasts": {}
    }

    if contrasts:
        os.makedirs(save_dir, exist_ok=True) if save_dir else None
        for name, c in contrasts.items():
            c = np.asarray(c, dtype=np.float64).reshape(-1)
            if c.size != p:
                raise ValueError(f"contrast '{name}' length {c.size} != {p}")
            eff = beta @ c                      # (R,)
            # per-ROI SE = sqrt(sigma2 * c'(XtX)^-1 c)
            se = np.full(R, np.nan, dtype=np.float64)
            for r in range(R):
                if N_r[r] <= p or XtX_inv_list[r] is None:
                    continue
                scale = float(c @ XtX_inv_list[r] @ c)
                se[r] = np.sqrt(max(sigma2[r] * scale, var_floor))
            t = eff / se

            eff_map = _paint(eff)
            t_map   = _paint(t)
            if save_dir:
                nib.save(eff_map, f"{save_dir}/{name}_effect_parc.nii")
                nib.save(t_map,   f"{save_dir}/{name}_t_parc.nii")

            results["contrasts"][name] = {
                "effect_roi": eff, "t_roi": t,
                "effect_map": eff_map, "t_map": t_map
            }

    return results

def interaction_glm_voxelwise(
    A1B1_paths: List[str],   # condition cell (A=1,B=1)
    A1B2_paths: List[str],   # (A=1,B=2)
    A2B1_paths: List[str],   # (A=2,B=1)
    A2B2_paths: List[str],   # (A=2,B=2)
    X: Union[np.ndarray, "pd.DataFrame"],  # (N x p), MUST include an intercept column of ones
    mask_path: str,
    *,
    contrasts: Optional[Dict[str, np.ndarray]] = None,  # over beta (length p)
    cell_weights: Optional[np.ndarray] = None,          # len 4 over [A1B1,A1B2,A2B1,A2B2]
    center_non_intercept: bool = True,
    intercept_first: bool = True,
    dtype = np.float32,
    var_floor: float = 1e-12,
    chunk_slices: int = 48,                 # number of z-slices per streaming chunk
    save_dir: Optional[str] = None,         # if set, will save NIfTI maps here
) -> Dict:
    
    """
    Streaming voxel-wise OLS for the 2x2 within-subject interaction:
      y_i = sum_k w_k * cell_k_i  (voxel-wise), then fit  y ~ 1 + covariates

    Returns
    -------
    results: dict
      - 'names': design column names
      - 'dof'  : degrees of freedom (N - p)
      - 'scale'[contrast]: scalar c'(X'X)^-1 c
      - 'effect_img'[contrast], 't_img'[contrast]: NIfTI images (voxel-wise)
    """
    
    # ---------- Design handling ----------
    if "pd" in str(type(X)):
        names = list(X.columns)
        X = X.values
    else:
        names = [f"x{i}" for i in range(X.shape[1])]
    X = np.asarray(X, dtype=np.float64)
    N, p = X.shape

    for lst in (A1B1_paths, A1B2_paths, A2B1_paths, A2B2_paths):
        assert len(lst) == N, "All condition lists must have N subjects (= rows of X)."

    if intercept_first:
        if not np.allclose(X[:, 0], 1.0):
            raise ValueError("First column must be intercept (all ones).")
    else:
        ones_cols = np.where(np.all(np.isclose(X, 1.0), axis=0))[0]
        if len(ones_cols) == 0:
            raise ValueError("Design must include an intercept column of all ones.")
        j0 = int(ones_cols[0])
        if j0 != 0:
            X[:, [0, j0]] = X[:, [j0, 0]]
            names[0], names[j0] = names[j0], names[0]

    if center_non_intercept and p > 1:
        X[:, 1:] -= X[:, 1:].mean(axis=0, keepdims=True)

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX, rcond=1e-12)
        print("[WARN] X'X not invertible; using pseudo-inverse. Check collinearity.")
    dof = max(N - p, 1)

    # ---------- Weights for AxB over 4 cells ----------
    if cell_weights is None:
        w = np.array([+1.0, -1.0, -1.0, +1.0], dtype=np.float64)
    else:
        w = np.asarray(cell_weights, dtype=np.float64).reshape(-1)
        if w.size != 4:
            raise ValueError("cell_weights must have length 4 (A1B1,A1B2,A2B1,A2B2).")

    # ---------- Mask & geometry ----------
    if isinstance(mask_path, str):
        mask_img = nib.load(mask_path)
    else:
        mask_img = mask_path

    mask = mask_img.get_fdata().astype(bool)
    shape = mask.shape
    Z = shape[-1]

    # Open images once (memory-mapped); we will slice per chunk
    imgs11 = [nib.load(p) for p in A1B1_paths]
    imgs12 = [nib.load(p) for p in A1B2_paths]
    imgs21 = [nib.load(p) for p in A2B1_paths]
    imgs22 = [nib.load(p) for p in A2B2_paths]

    # sanity: identical geometry
    ref_aff = imgs11[0].affine
    for im in imgs11[1:] + imgs12 + imgs21 + imgs22:
        if im.shape != shape:
            raise ValueError("All input images must match mask shape (pre-resample offline).")
        if not np.allclose(im.affine, ref_aff):
            raise ValueError("All input images must have identical affines as mask (pre-resample offline).")

    # ---------- Prepare outputs (per contrast) ----------
    if contrasts is None:
        c = np.zeros((1, p), dtype=np.float64); c[0, 0] = 1.0
        contrasts = {"intercept": c[0]}

    # allocate volumes to fill (effect/t images)
    effect_vols: Dict[str, np.ndarray] = {k: np.zeros(shape, dtype=dtype) for k in contrasts.keys()}
    t_vols: Dict[str, np.ndarray]      = {k: np.zeros(shape, dtype=dtype) for k in contrasts.keys()}
    scales: Dict[str, float]           = {k: float(np.asarray(v, np.float64) @ XtX_inv @ np.asarray(v, np.float64))
                                          for k, v in contrasts.items()}

    # ---------- Stream by z-chunks ----------
    for z0 in range(0, Z, max(1, int(chunk_slices))):
        z1 = min(Z, z0 + chunk_slices)
        mchunk = mask[..., z0:z1]
        nvox = int(mchunk.sum())
        if nvox == 0:
            continue

        XTy = np.zeros((p, nvox), dtype=np.float64)
        YTY = np.zeros(nvox, dtype=np.float64)

        for i in range(N):
            m11 = np.asarray(imgs11[i].dataobj[..., z0:z1], dtype=dtype)
            m12 = np.asarray(imgs12[i].dataobj[..., z0:z1], dtype=dtype)
            m21 = np.asarray(imgs21[i].dataobj[..., z0:z1], dtype=dtype)
            m22 = np.asarray(imgs22[i].dataobj[..., z0:z1], dtype=dtype)

            # AxB interaction composite
            y_chunk = (w[0]*m11 + w[1]*m12 + w[2]*m21 + w[3]*m22)

            # extract vector in mask
            yvec = y_chunk[mchunk].astype(np.float64, copy=False)
            YTY += yvec * yvec
            XTy += (X[i, :].reshape(-1, 1) * yvec.reshape(1, -1))

        beta  = XtX_inv @ XTy
        SSE   = YTY - np.sum(beta * XTy, axis=0)
        sigma2 = np.maximum(SSE / dof, var_floor)

        # =========== This part of codes may be unnecessary ===========
        for name, c in contrasts.items():
            c = np.asarray(c, dtype=np.float64).reshape(-1)
            eff = c @ beta
            se = np.sqrt(np.maximum(sigma2 * scales[name], var_floor))
            tvec = eff / np.maximum(se, np.sqrt(var_floor))

            slab_e = effect_vols[name][..., z0:z1]
            slab_t = t_vols[name][...,  z0:z1]

            # robustness check
            assert eff.shape[0] == nvox and tvec.shape[0] == nvox

            slab_e[mchunk] = eff.astype(dtype, copy=False)
            slab_t[mchunk] = tvec.astype(dtype, copy=False)
        # =============================================================

    # Wrap as NIfTI and optionally save
    effect_imgs = {}
    t_imgs = {}
    for name in contrasts.keys():
        eimg = nib.Nifti1Image(effect_vols[name], ref_aff, mask_img.header)
        timg = nib.Nifti1Image(t_vols[name],      ref_aff, mask_img.header)
        effect_imgs[name] = eimg
        t_imgs[name] = timg
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            nib.save(eimg, os.path.join(save_dir, f"{name}_effect_vox.nii"))
            nib.save(timg, os.path.join(save_dir, f"{name}_t_vox.nii"))

    return {"names": names, "dof": dof, "scale": scales,
            "effect_img": effect_imgs, "t_img": t_imgs}

def t_to_z_img(t_img, dof, sided: str = "two", p_min: float = 1e-300):
    """
    Convert a t-statistic image to a z-statistic image.

    Parameters
    ----------
    t_img : str or nib.Nifti1Image
        T-statistic map (NIfTI path or image).
    dof : int | float | str | nib.Nifti1Image
        Degrees of freedom. Can be a scalar (same for all voxels)
        or a NIfTI image with voxelwise DoF.
    sided : {"two", "one"}, default "two"
        Use a two-sided or one-sided conversion.
        - "two": p = 2 * sf(|t|, dof); z = sign(t) * isf(p/2)
        - "one": p = sf(t, dof);       z = isf(p) with t's sign retained
    p_min : float, default 1e-300
        Lower bound for p-values to avoid inf z-scores.

    Returns
    -------
    z_img : nib.Nifti1Image
        Z-statistic image aligned to t_img.
    """

    # Load inputs
    if isinstance(t_img, str):
        t_img = nib.load(t_img)

    t_data = np.asanyarray(t_img.get_fdata(dtype=np.float64))

    # Prepare DoF array matching t_data shape (or broadcastable)
    if isinstance(dof, (int, float)):
        dof_data = float(dof)
    else:
        dof_img = nib.load(dof)
        dof_data = np.asanyarray(dof_img.get_fdata(dtype=np.float64))
        if dof_data.shape != t_data.shape:
            raise ValueError("Voxelwise DoF image must match t_img shape.")

    # Build a mask of finite t and valid DoF
    mask = np.isfinite(t_data)
    if np.isscalar(dof_data):
        valid_dof = np.isfinite(dof_data) and (dof_data > 0)
    else:
        valid_dof = np.isfinite(dof_data) & (dof_data > 0)
        mask &= valid_dof

    z = np.full_like(t_data, np.nan, dtype=np.float64)
    if not np.any(mask):
        return new_img_like(t_img, z)

    t_vals = t_data[mask]
    dof_vals = dof_data if np.isscalar(dof_data) else dof_data[mask]

    # Compute p-values from t and DoF
    # sf(x) = 1 - cdf(x); robust in tails.
    if sided.lower().startswith("two"):
        p = 2.0 * tdist.sf(np.abs(t_vals), df=dof_vals)
        # Avoid zeros -> infinite z
        p = np.clip(p, p_min, 1.0)
        z_vals = norm.isf(p / 2.0)  # always positive
        # Restore the sign of the original t
        z_vals *= np.sign(t_vals)
    elif sided.lower().startswith("one"):
        # One-sided: keep sign consistent with t
        # For positive t: p = sf(t); for negative t: p ~ 1 - sf(|t|)
        # A simple choice: use the positive tail p and carry sign
        p = tdist.sf(t_vals, df=dof_vals)
        # For t < 0, sf(t) ~ 1 - cdf(t) ~ ~1, so cap for stability then restore sign
        p = np.clip(p, p_min, 1.0)
        z_vals = norm.isf(p)
        # Keep the sign of t (so negative t -> negative z)
        z_vals *= np.sign(t_vals)
    else:
        raise ValueError('`sided` must be "two" or "one".')

    z[mask] = z_vals

    # Return as a NIfTI aligned to the input T map
    return new_img_like(t_img, z)