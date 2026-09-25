#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 02:50:15 2025

    sorts of flexible functions to perform brain imaging analyses of surface data (.gii/cifti)

@author: dingrui
"""

# import modules
import os, re
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
from typing import List, Optional, Dict, Union, Tuple, Literal, Any

import mylib.stat.stats as Sts

ArrayLike = Union[np.ndarray, List[float]]

# -----------------------
# Helpers: SPM.mat reader
# -----------------------

def _spm_get_names_and_bcov(spm_mat_path: str) -> Tuple[List[str], np.ndarray]:
    """
    extract xX.name (regressor names) and xX.Bcov (parameter covariance / σ²) from SPM.mat，
    --> effect variance: Var(β_j) = ResMS * Bcov[j,j]
    """
    m = loadmat(spm_mat_path, squeeze_me=True, struct_as_record=False)
    SPM = m.get("SPM", None)
    if SPM is None:
        raise ValueError(f"Cannot find SPM in {spm_mat_path}")
    xX = getattr(SPM, "xX", None) if hasattr(SPM, "xX") else SPM["xX"]
    names = getattr(xX, "name", None) if hasattr(xX, "name") else xX["name"]
    Bcov = getattr(xX, "Bcov", None) if hasattr(xX, "Bcov") else xX["Bcov"]
    # names maybe object array or list
    if isinstance(names, (np.ndarray, list, tuple)):
        names = [str(n) for n in np.ravel(names).tolist()]
    else:
        names = [str(names)]
    Bcov = np.asarray(Bcov, dtype=np.float64)
    if Bcov.ndim != 2 or Bcov.shape[0] != Bcov.shape[1] or Bcov.shape[0] != len(names):
        raise ValueError("SPM.xX.Bcov shape mismatch with xX.name")
    return names, Bcov

def _match_regressor_index(
    names: List[str],
    cond: str,
    basis_idx: int = 1,
    exclude_modulators: bool = True
) -> int:
    """
    searching for column index matching given condition in SPM.xX.name
    --> 'Sn(1) cond*bf(1)', exclude modulator terms consisting of '^' if exclude_modulators=True
    """
    pat = re.compile(rf".*\b{re.escape(cond)}\b.*\*bf\({basis_idx}\)", re.IGNORECASE)
    candidates = [i for i, s in enumerate(names) if pat.search(s)]
    if exclude_modulators:
        candidates = [i for i in candidates if "^" not in names[i]]
    if len(candidates) == 0:
        pat2 = re.compile(rf".*\b{re.escape(cond)}\b.*", re.IGNORECASE)
        candidates = [i for i, s in enumerate(names) if pat2.search(s)]
        if exclude_modulators:
            candidates = [i for i in candidates if "^" not in names[i]]
    if len(candidates) == 0:
        raise ValueError(f"Cannot match regressor for condition '{cond}'. "
                         f"Check SPM.xX.name content.")
    return int(candidates[0])

# -----------------------
# Helpers: IO for surface
# -----------------------
def _as_cifti_maps_and_brain_axis(img):
    """
    Return (data_2d, brain_axis) where data_2d has shape (n_maps, n_brainordinates).
    Works for dscalar/dlabel/dtseries. Raises if no BrainModelAxis is found.
    """
    data = np.asanyarray(img.dataobj)
    ax0 = img.header.get_axis(0)
    ax1 = img.header.get_axis(1)
    # shape --> (n_maps, n_greyordinates)
    if hasattr(ax1, "iter_structures"):  # BrainModelAxis
        brain_axis = ax1
        if data.ndim == 1:
            data = data[np.newaxis, ...]
        elif data.ndim != 2:
            data = data.reshape((data.shape[0], -1))
        return data, brain_axis
    elif hasattr(ax0, "iter_structures"):
        brain_axis = ax0
        if data.ndim == 1:
            data = data[np.newaxis, ...]
        elif data.ndim != 2:
            data = data.reshape((data.shape[0], -1))
        return data.T, brain_axis
    else:
        raise ValueError("CIFTI image has no BrainModelAxis.")

def _cifti_take_hemi(data_2d, brain_axis, hemi: str, map_idx: int = 0, dtype=np.float32):
    """
    Slice one map from CIFTI 2D array by hemisphere ('lh' or 'rh').
    Returns a 1D vector of per-vertex values for that hemi.
    """
    hemi = hemi.lower()
    struct = "CIFTI_STRUCTURE_CORTEX_LEFT" if hemi == "lh" else "CIFTI_STRUCTURE_CORTEX_RIGHT"
    slc = None
    for name, s, _bm in brain_axis.iter_structures():
        if str(name) == struct:
            slc = s
            break
    if slc is None:
        # current hemi not in this CIFTI
        return np.array([], dtype=dtype)
    return np.asarray(data_2d[map_idx, slc], dtype=dtype).reshape(-1)

def _load_surf_metric(path: str, dtype=np.float32, hemi_hint: str = None, map_idx: int = 0) -> np.ndarray:
    """
    Load per-vertex metric (1D array). Supports:
      - GIFTI (.gii), FS (.mgh/.mgz), NumPy (.npy)
      - CIFTI (.dscalar.nii/.dtseries.nii): require hemi_hint ('lh'/'rh')
    For CIFTI, returns the selected hemi's vector from the specified map_idx.
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gii":
        img = nib.load(path)
        arr = img.darrays[0].data
        return np.asarray(arr, dtype=dtype).reshape(-1)
    elif ext in (".mgh", ".mgz"):
        img = nib.load(path)
        arr = np.asarray(img.get_fdata(), dtype=dtype)
        return arr.reshape(-1)
    elif ext == ".npy":
        arr = np.load(path)
        return np.asarray(arr, dtype=dtype).reshape(-1)
    elif ext == ".nii":
        # assume CIFTI2
        img = nib.load(path)
        if not isinstance(img, nib.cifti2.Cifti2Image):
            raise ValueError(f"{path} looks like NIfTI but not a CIFTI2 image.")
        if hemi_hint is None:
            raise ValueError("CIFTI metric requires hemi_hint='lh' or 'rh'.")
        data2d, brain_axis = _as_cifti_maps_and_brain_axis(img)
        return _cifti_take_hemi(data2d, brain_axis, hemi=hemi_hint, map_idx=map_idx, dtype=dtype)
    else:
        raise ValueError(f"Unsupported metric format: {path}")

def _load_surf_labels(path: str, hemi_hint: str = None, map_idx: int = 0) -> np.ndarray:
    """
    Load per-vertex integer labels. Supports:
      - GIFTI label/metric (.gii), FS .annot/.mgh/.mgz, NumPy .npy
      - CIFTI (.dlabel.nii or integer-valued .dscalar.nii): require hemi_hint
    Returns a 1D int array for the chosen hemisphere.
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gii":
        img = nib.load(path)
        arr = img.darrays[0].data
        return np.rint(np.asarray(arr)).astype(np.int64).reshape(-1)
    elif ext == ".annot":
        import nibabel.freesurfer.io as fsio
        labels, _, _ = fsio.read_annot(path)
        return labels.astype(np.int64).reshape(-1)
    elif ext in (".mgh", ".mgz"):
        img = nib.load(path)
        arr = np.asarray(img.get_fdata()).reshape(-1)
        return np.rint(arr).astype(np.int64)
    elif ext == ".npy":
        arr = np.load(path)
        return np.rint(np.asarray(arr)).astype(np.int64).reshape(-1)
    elif ext == ".nii":
        img = nib.load(path)
        if not isinstance(img, nib.cifti2.Cifti2Image):
            raise ValueError(f"{path} looks like NIfTI but not a CIFTI2 image.")
        if hemi_hint is None:
            raise ValueError("CIFTI labels require hemi_hint='lh' or 'rh'.")
        data2d, brain_axis = _as_cifti_maps_and_brain_axis(img)
        vec = _cifti_take_hemi(data2d, brain_axis, hemi=hemi_hint, map_idx=map_idx, dtype=np.float32)
        return np.rint(vec).astype(np.int64)  # round in case of accidental interpolation
    else:
        raise ValueError(f"Unsupported label format: {path}")

def _maybe_load_mask(path: Optional[str], n_vertices: int) -> np.ndarray:
    """
    Load a boolean vertex mask; if None -> all True.
    Accepts GIFTI/NPY; for GIFTI uses first darray.
    """
    if path is None:
        return np.ones(n_vertices, dtype=bool)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gii":
        arr = nib.load(path).darrays[0].data
        return np.asarray(arr, dtype=bool).reshape(-1)
    elif ext == ".npy":
        arr = np.load(path)
        return np.asarray(arr, dtype=bool).reshape(-1)
    else:
        raise ValueError(f"Unsupported mask format: {path}")

def _to_gifti_metric(data: np.ndarray,
                     hemi: Literal["lh","rh"],
                     name: Optional[str] = None) -> nib.GiftiImage:
    
    """Wrap a 1D per-vertex array into a minimal GIFTI metric image"""
    
    da = nib.gifti.GiftiDataArray(
        np.asarray(data, dtype=np.float32).reshape(-1),
        intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE']
    )
    
    img = nib.gifti.GiftiImage(darrays=[da])
    
    # Minimal metadata is enough for most viewers (WB/Freeview)
    img.meta = nib.gifti.GiftiMetaData.from_dict(
        {"AnatomicalStructurePrimary": "CortexLeft" if hemi == "lh" else "CortexRight"}
    )
    
    if name:
        da.meta = nib.gifti.GiftiMetaData.from_dict({"Name": name})
    
    return img

def threshold_gifti(in_gii: str,
                    low: float = -3.0,
                    high: float = 3.3,
                    return_mask: bool = False,
                    save_fpath_out_gii: Optional[str] = None,
                    save_fpath_mask: Optional[str] = None,
                    inplace: bool = False) -> Optional[np.ndarray]:
    """
    Load a GIFTI metric, set vertices with values inside [low, high] to 0,
    save a new GIFTI file, and optionally return/save a binary mask.

    Parameters
    ----------
    in_gii : str
        Path to input GIFTI metric (e.g. .func.gii).
    low : float
        Lower bound of interval (inclusive).
    high : float
        Upper bound of interval (inclusive).
    return_mask : bool
        If True, return a numpy uint8 mask where 1 indicates vertices that
        were set to 0 (i.e., values within [low, high]).
    save_fpath_out_gii : Optional[str]
        Path to output thresholded GIFTI.
    save_fpath_mask : Optional[str]
        If provided, save the binary mask as a GIFTI file at this path.
    inplace : bool
        If True, modify the loaded data array in-place (slightly faster / less memory).
        If False (default), operate on a copy.

    Returns
    -------
    mask : np.ndarray (uint8) or None
        If return_mask True, returns the binary mask array (shape: n_vertices).
        Otherwise returns None.
    """
    
    if isinstance(in_gii, str):
        g = nib.load(in_gii)
    else:
        g = in_gii

    # assume the first data array is the metric; if multiple arrays exist, handle explicitly
    if len(g.darrays) == 0:
        raise ValueError("Input GIFTI contains no data arrays.")
    data = g.darrays[0].data

    # make a working copy or operate in-place
    if inplace:
        arr = data
    else:
        arr = data.copy()

    # compute mask of vertices to set to zero (inclusive bounds)
    # NaNs will not satisfy the comparison and thus remain NaN in arr
    thr_mask = (arr >= low) & (arr <= high)

    # set those values to zero
    arr[thr_mask] = 0.0

    # build new GIFTI image to save (avoid mutating original object)
    new_g = nib.gifti.GiftiImage()
    # preserve intent and metadata if present in original (optional)
    # copy the metadata and intent from original array if available
    orig_da = g.darrays[0]
    new_da = nib.gifti.GiftiDataArray(data=arr.astype(np.float32),
                                     intent=orig_da.intent if hasattr(orig_da, "intent") else None,
                                     encoding=orig_da.encoding if hasattr(orig_da, "encoding") else "GZIP")
    new_g.darrays.append(new_da)
    if save_fpath_out_gii is not None:
        nib.save(new_g, save_fpath_out_gii)

    # prepare binary mask to return/save if requested
    if return_mask:
        # mask as uint8: 1 -> was set to zero (value inside [low, high]); 0 -> otherwise
        nonzero_mask = (arr != 0) & np.isfinite(arr) & (arr != np.nan)
        mask_uint8 = nonzero_mask.astype(np.uint8)

        mask_g = nib.gifti.GiftiImage()
        mask_da = nib.gifti.GiftiDataArray(data=mask_uint8,
                                          intent="NIFTI_INTENT_LABEL")
        mask_g.darrays.append(mask_da)
        
        return mask_g
        
        if save_fpath_mask is not None:
            nib.save(mask_g, save_fpath_mask)
    else:
        return new_g


# -------------------------------------------------------------
# Compute fixed effect across runs --> merge effect across runs
# -------------------------------------------------------------

def make_effvar_surf(
    conditions: List[str],
    # ResMS for each run（LH/RH）shared for all conditions
    sigma2_lh: List[Union[str, np.ndarray]],
    sigma2_rh: List[Union[str, np.ndarray]],
    # fullpath of SPM.mat（for calculating diag[(X'X)^-1]）
    spm_mat_paths: List[str],
    *,
    hemi_loader = _load_surf_labels,  # (path, hemi_hint='lh'/'rh')-> np.ndarray
    basis_idx: int = 1,               # default: bf(1) of cond
    dtype=np.float32
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    calculate effect-variance surface maps（LH/RH）for each condition of each run
    --> Var(β_cond) = ResMS (per-vertex) * Bcov[ii], i is the col index matching condition

    out[cond] = { 'lh_var': [arr_run1, ...], 'rh_var': [arr_run1, ...] }
    """
    n_runs = len(spm_mat_paths)
    if len(sigma2_lh) != n_runs or len(sigma2_rh) != n_runs:
        raise ValueError("sigma2_lh/rh length must equal #runs = len(spm_mat_paths).")

    # load all sigma2_* (ResMS map)
    sig2_lh = [(hemi_loader(p, hemi_hint="lh") if isinstance(p, str) else np.asarray(p, dtype=np.float64))
               for p in sigma2_lh ]
    sig2_rh = [(hemi_loader(p, hemi_hint="rh") if isinstance(p, str) else np.asarray(p, dtype=np.float64))
               for p in sigma2_rh ]
    Vlh, Vrh = sig2_lh[0].shape[0], sig2_rh[0].shape[0]
    for s in sig2_lh + sig2_rh:
        if s.ndim != 1: raise ValueError("sigma2 arrays must be 1D per-vertex.")
    # read (names, Bcov) for each run and find diag of each condition
    diag_per_run_per_cond: Dict[str, List[float]] = {c: [] for c in conditions}
    for r, spm_path in enumerate(spm_mat_paths):
        names, Bcov = _spm_get_names_and_bcov(spm_path)
        for cond in conditions:
            j = _match_regressor_index(names, cond, basis_idx=basis_idx, exclude_modulators=True)
            d = float(Bcov[j, j])  # [(X'X)^-1]_{jj}
            diag_per_run_per_cond[cond].append(d)

    # calculate effect variance maps for each condition
    out: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for cond in conditions:
        lh_vars, rh_vars = [], []
        for r in range(n_runs):
            d = diag_per_run_per_cond[cond][r]
            v_lh = (sig2_lh[r] * d).astype(dtype, copy=False)
            v_rh = (sig2_rh[r] * d).astype(dtype, copy=False)
            if v_lh.shape[0] != Vlh or v_rh.shape[0] != Vrh:
                raise ValueError(f"Run {r}: variance length mismatch.")
            lh_vars.append(v_lh); rh_vars.append(v_rh)
        out[cond] = {"lh_var": lh_vars, "rh_var": rh_vars}
    return out

# -----------------------
# Helpers: effects merger
# -----------------------

def _fe_one_hemi(
    eff_list: List[np.ndarray],   # length=K, each (V,)
    var_list: List[np.ndarray],   # length=K, each (V,)
    eps: float = 1e-12,
    dtype=np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    K = len(eff_list)
    if K != len(var_list):
        raise ValueError("eff_list and var_list must have same length.")
    E = np.vstack([np.asarray(e, dtype=np.float64).reshape(1, -1) for e in eff_list])   # (K,V)
    Vv = np.vstack([np.asarray(v, dtype=np.float64).reshape(1, -1) for v in var_list])  # (K,V)
    valid = np.isfinite(E) & np.isfinite(Vv) & (Vv > eps)
    W = np.zeros_like(Vv); W[valid] = 1.0 / Vv[valid]
    Wsum = np.sum(W, axis=0)                                  # (V,)
    ok = Wsum > 0
    beta_fe = np.zeros(E.shape[1], dtype=np.float64)
    var_fe  = np.full(E.shape[1], np.nan, dtype=np.float64)
    if np.any(ok):
        num = np.sum(W[:, ok] * E[:, ok], axis=0)
        beta_fe[ok] = num / Wsum[ok]
        var_fe[ok]  = 1.0 / Wsum[ok]
    return beta_fe.astype(dtype, copy=False), var_fe.astype(dtype, copy=False)

def compute_fixed_eff_surf(
    # single condition as input: list of per-run β and Var(β) for each hemi
    eff_lh: List[np.ndarray],
    eff_rh: List[np.ndarray],
    var_lh: List[np.ndarray],
    var_rh: List[np.ndarray],
    *,
    eps: float = 1e-12,
    dtype=np.float32
) -> Dict[str, np.ndarray]:
    """
    Compute fixed effect for a single condition merged across runs
     - This function takes a single condition as input
    """
    beta_lh, var_lh_fe = _fe_one_hemi(eff_lh, var_lh, eps=eps, dtype=dtype)
    beta_rh, var_rh_fe = _fe_one_hemi(eff_rh, var_rh, eps=eps, dtype=dtype)
    z_lh = np.zeros_like(beta_lh); z_rh = np.zeros_like(beta_rh)
    m_lh = np.isfinite(var_lh_fe) & (var_lh_fe > eps)
    m_rh = np.isfinite(var_rh_fe) & (var_rh_fe > eps)
    z_lh[m_lh] = beta_lh[m_lh] / np.sqrt(var_lh_fe[m_lh])
    z_rh[m_rh] = beta_rh[m_rh] / np.sqrt(var_rh_fe[m_rh])
    return {"lh_beta": beta_lh, "rh_beta": beta_rh,
            "lh_var": var_lh_fe, "rh_var": var_rh_fe,
            "lh_z": z_lh, "rh_z": z_rh}

def compute_fixed_eff_surf_multi(
    # multi-conditions as input: using the output of `make_effvar_surf` + β
    eff_by_cond: Dict[str, Dict[str, List[np.ndarray]]],  # {'cond': {'lh_eff':[...], 'rh_eff':[...]} }
    var_by_cond: Dict[str, Dict[str, List[np.ndarray]]],  # {'cond': {'lh_var':[...], 'rh_var':[...]} }
    *,
    eps: float = 1e-12,
    dtype=np.float32
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute fixed effect for multiple conditions merged across runs
     - This function takes multiple conditions as input
    """
    out = {}
    for cond in eff_by_cond.keys():
        res = compute_fixed_eff_surf(
            eff_lh=eff_by_cond[cond]["lh_eff"],
            eff_rh=eff_by_cond[cond]["rh_eff"],
            var_lh=var_by_cond[cond]["lh_var"],
            var_rh=var_by_cond[cond]["rh_var"],
            eps=eps, dtype=dtype
        )
        out[cond] = res
    return out

# ------------------------------------------
# Surface ROI-wise robust OLS for Δ = A - B 
# ------------------------------------------
def glm_AmB_surf_roi(
    A_lh_paths: List[str],
    A_rh_paths: List[str],
    B_lh_paths: List[str],
    B_rh_paths: List[str],
    X: Union[np.ndarray, "pd.DataFrame"],         # (N, p), col0 must be ones
    parc_lh_path: str,                            # LH labels (per-vertex integers; 0=bg)
    parc_rh_path: str,                            # RH labels
    *,
    mask_lh_path: Optional[str] = None,           # optional vertex masks (True=keep)
    mask_rh_path: Optional[str] = None,
    include_labels_lh: Optional[List[int]] = None,
    exclude_labels_lh: Optional[List[int]] = None,
    include_labels_rh: Optional[List[int]] = None,
    exclude_labels_rh: Optional[List[int]] = None,
    contrasts: Optional[Dict[str, np.ndarray]] = None,  # name -> (p,)
    dtype=np.float32,
    var_floor: float = 1e-12,
    save_dir: Optional[str] = None,               # if set, write GIFTI maps
) -> Dict:
    
    """
    Robust surface ROI-wise OLS for Δ = A - B with per-ROI missing-data handling.
    - Inputs are per-vertex maps for left/right hemispheres.
    - Each ROI gets its own XtX/XTy/YTY/df (only subjects with at least one finite vertex contribute).

    Returns
    -------
    {
      'names': list[str],
      'roi_table': [{'hemi':'lh','label':int,'n_vert':int}, ...],  # length R
      'beta': (R, p),
      'sigma2': (R,),
      'N_eff_per_roi': (R,),
      'dof_per_roi': (R,),
      'contrasts': {
         name: {
           'effect_roi': (R,),
           't_roi': (R,),
           'lh_effect': GiftiImage, 'rh_effect': GiftiImage,
           'lh_t': GiftiImage,      'rh_t': GiftiImage
         }, ...
      }
    }
    """
    
    # ---------- Design ----------
    if "pd" in str(type(X)):
        names = list(X.columns)
        X = X.values.copy().astype(np.float64)
    else:
        names = [f"x{i}" for i in range(X.shape[1])]
        X = np.asarray(X, dtype=np.float64)
    N, p = X.shape
    if not np.allclose(X[:, 0], 1.0):
        raise ValueError("First column of X must be an intercept (all ones).")
    if not (len(A_lh_paths) == len(B_lh_paths) == len(A_rh_paths) == len(B_rh_paths) == N):
        raise ValueError("All path lists must have the same length as rows of X.")

    # ---------- Parcellations & masks ----------
    lab_lh = _load_surf_labels(parc_lh_path, hemi_hint="lh", map_idx=0)
    lab_rh = _load_surf_labels(parc_rh_path, hemi_hint="rh", map_idx=0)

    m_lh = _maybe_load_mask(mask_lh_path, lab_lh.size)
    m_rh = _maybe_load_mask(mask_rh_path, lab_rh.size)

    # keep vertices: mask & label != 0 & include/exclude
    keep_lh = m_lh & (lab_lh != 0)
    keep_rh = m_rh & (lab_rh != 0)
    if include_labels_lh is not None:
        keep_lh &= np.isin(lab_lh, np.asarray(include_labels_lh, np.int64))
    if exclude_labels_lh is not None:
        keep_lh &= ~np.isin(lab_lh, np.asarray(exclude_labels_lh, np.int64))
    if include_labels_rh is not None:
        keep_rh &= np.isin(lab_rh, np.asarray(include_labels_rh, np.int64))
    if exclude_labels_rh is not None:
        keep_rh &= ~np.isin(lab_rh, np.asarray(exclude_labels_rh, np.int64))

    # unique ROI keys: ('lh', label) and ('rh', label)
    uniq_lh = np.unique(lab_lh[keep_lh]); uniq_lh.sort()
    uniq_rh = np.unique(lab_rh[keep_rh]); uniq_rh.sort()
    # Build global ROI keys
    roi_keys: List[Tuple[str, int]] = [('lh', int(k)) for k in uniq_lh] + [('rh', int(k)) for k in uniq_rh]
    if len(roi_keys) < 2:
        raise ValueError("Not enough ROIs after filtering; check masks/parcellations.")
    R = len(roi_keys)
    key_to_idx = {k: i for i, k in enumerate(roi_keys)}

    # Map each kept vertex -> global ROI index
    # LH
    kept_idx_lh = np.flatnonzero(keep_lh)
    roi_index_lh = np.array([key_to_idx[('lh', int(l))] for l in lab_lh[kept_idx_lh]], dtype=np.int64)
    # RH
    kept_idx_rh = np.flatnonzero(keep_rh)
    roi_index_rh = np.array([key_to_idx[('rh', int(l))] for l in lab_rh[kept_idx_rh]], dtype=np.int64)

    # ROI meta (counts)
    roi_counts = np.zeros(R, dtype=np.int64)
    # count vertices per ROI (LH + RH)
    roi_counts += np.bincount(roi_index_lh, minlength=R)
    roi_counts += np.bincount(roi_index_rh, minlength=R)

    roi_table = []
    for k in roi_keys:
        hemi, lab = k
        n_vert = (roi_index_lh.size if hemi == 'lh' else roi_index_rh.size)  # placeholder; fix below
        if hemi == 'lh':
            n_vert = (lab_lh[kept_idx_lh] == lab).sum()
        else:
            n_vert = (lab_rh[kept_idx_rh] == lab).sum()
        roi_table.append({'hemi': hemi, 'label': int(lab), 'n_vert': int(n_vert)})

    # ---------- Per-ROI accumulators ----------
    XtX = np.zeros((R, p, p), dtype=np.float64)
    XTy = np.zeros((R, p),    dtype=np.float64)
    YTY = np.zeros(R,         dtype=np.float64)
    N_r = np.zeros(R,         dtype=np.int64)

    # ---------- Stream subjects ----------
    for i in range(N):
        # load A/B for both hemispheres
        A_lh = _load_surf_metric(A_lh_paths[i], hemi_hint="lh", map_idx=0)  # 第 0 个 map
        A_rh = _load_surf_metric(A_rh_paths[i], hemi_hint="rh", map_idx=0)
        B_lh = _load_surf_metric(B_lh_paths[i], hemi_hint="lh", map_idx=0)
        B_rh = _load_surf_metric(B_rh_paths[i], hemi_hint="rh", map_idx=0)


        if A_lh.shape[0] != lab_lh.size or B_lh.shape[0] != lab_lh.size:
            raise ValueError("LH metric length does not match LH parcellation.")
        if A_rh.shape[0] != lab_rh.size or B_rh.shape[0] != lab_rh.size:
            raise ValueError("RH metric length does not match RH parcellation.")

        # Δ per hemisphere, restricted to kept vertices
        d_lh = (A_lh - B_lh)[kept_idx_lh].astype(np.float64, copy=False)
        d_rh = (A_rh - B_rh)[kept_idx_rh].astype(np.float64, copy=False)

        # Only finite vertices contribute
        fin_lh = np.isfinite(d_lh)
        fin_rh = np.isfinite(d_rh)

        # Per-ROI sums & counts across both hemispheres
        sums = np.zeros(R, dtype=np.float64)
        cnts = np.zeros(R, dtype=np.int64)
        if fin_lh.any():
            sums += np.bincount(roi_index_lh[fin_lh], weights=d_lh[fin_lh], minlength=R)
            cnts += np.bincount(roi_index_lh[fin_lh], minlength=R)
        if fin_rh.any():
            sums += np.bincount(roi_index_rh[fin_rh], weights=d_rh[fin_rh], minlength=R)
            cnts += np.bincount(roi_index_rh[fin_rh], minlength=R)

        valid_roi = cnts > 0
        if not np.any(valid_roi):
            continue

        means = np.zeros(R, dtype=np.float64)
        means[valid_roi] = sums[valid_roi] / cnts[valid_roi]

        xi = X[i, :]
        X_outer = np.outer(xi, xi)  # (p,p)
        vr = np.where(valid_roi)[0]
        XtX[vr] += X_outer
        # Note: XTy[r,:] accumulates X[i,:] * y_i_roi ; shape (R,p)
        XTy[vr] += np.outer(means[vr], xi)  # (k,p)
        YTY[vr] += means[vr] ** 2
        N_r[vr] += 1

    # ---------- Solve per ROI ----------
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

    # ---------- Painter: ROI vector -> per-vertex arrays ----------
    def _paint_hemi(vec_roi: np.ndarray, hemi: str) -> np.ndarray:
        """
        Map an ROI vector (length R) back to per-vertex array for one hemisphere.
        """
        if hemi == 'lh':
            out = np.zeros(lab_lh.size, dtype=dtype)
            if kept_idx_lh.size > 0:
                out_kept = vec_roi[roi_index_lh].astype(dtype, copy=False)
                out[kept_idx_lh] = out_kept
            return out
        else:
            out = np.zeros(lab_rh.size, dtype=dtype)
            if kept_idx_rh.size > 0:
                out_kept = vec_roi[roi_index_rh].astype(dtype, copy=False)
                out[kept_idx_rh] = out_kept
            return out

    # ---------- Pack results ----------
    results: Dict[str, Union[dict, list, np.ndarray]] = {
        "names": names,
        "roi_table": roi_table,                  # list of dicts: hemi/label/n_vert
        "beta": beta,                            # (R,p)
        "sigma2": sigma2,                        # (R,)
        "N_eff_per_roi": N_r,                    # (R,)
        "dof_per_roi": dof_r,                    # (R,)
        "contrasts": {}
    }

    if contrasts:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        for name, c in contrasts.items():
            c = np.asarray(c, dtype=np.float64).reshape(-1)
            if c.size != p:
                raise ValueError(f"contrast '{name}' length {c.size} != {p}")
            eff = beta @ c  # (R,)
            # per-ROI SE = sqrt(sigma2 * c'(XtX)^-1 c)
            se = np.full(R, np.nan, dtype=np.float64)
            for r in range(R):
                if N_r[r] <= p or XtX_inv_list[r] is None:
                    continue
                scale = float(c @ XtX_inv_list[r] @ c)
                se[r] = np.sqrt(max(sigma2[r] * scale, var_floor))
            t = eff / se

            # paint back to vertices
            eff_lh = _paint_hemi(eff, 'lh'); eff_rh = _paint_hemi(eff, 'rh')
            t_lh   = _paint_hemi(t,   'lh'); t_rh   = _paint_hemi(t,   'rh')

            # GIFTI images
            gifti_eff_lh = _to_gifti_metric(eff_lh, hemi='lh', name='effect size(beta) map')
            gifti_eff_rh = _to_gifti_metric(eff_rh, hemi='rh', name='effect size(beta) map')
            gifti_t_lh   = _to_gifti_metric(t_lh,   hemi='lh', name=f'one-sapmle t map for beta of {name}')
            gifti_t_rh   = _to_gifti_metric(t_rh,   hemi='rh', name=f'one-sapmle t map for beta of {name}')

            if save_dir:
                nib.save(gifti_eff_lh, os.path.join(save_dir, f"{name}_effect_lh.func.gii"))
                nib.save(gifti_eff_rh, os.path.join(save_dir, f"{name}_effect_rh.func.gii"))
                nib.save(gifti_t_lh,   os.path.join(save_dir, f"{name}_t_lh.func.gii"))
                nib.save(gifti_t_rh,   os.path.join(save_dir, f"{name}_t_rh.func.gii"))

            results["contrasts"][name] = {
                "effect_roi": eff,         "t_roi": t,
                "lh_effect": gifti_eff_lh, "rh_effect": gifti_eff_rh,
                "lh_t": gifti_t_lh,        "rh_t": gifti_t_rh
            }

    return results

# ------------------------------------------
# Surface ROI-wise robust GAM for Δ = A - B
# ------------------------------------------
    
def gam_AmB_surf_roi(
    A_lh_paths: List[str],
    A_rh_paths: List[str],
    B_lh_paths: List[str],
    B_rh_paths: List[str],
    X: pd.DataFrame,
    parc_lh, parc_rh,
    *,
    hemi_loader_metric = None,  # callable(path, hemi_hint)->1d array
    mask_lh: Optional[np.ndarray] = None,
    mask_rh: Optional[np.ndarray] = None,
    include_labels_lh: Optional[List[int]] = None,
    include_labels_rh: Optional[List[int]] = None,
    gam_cfg = Sts.GAMConfig(),
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    
    """
    ROI GAM + statistics wrapper that:
      - computes for each ROI: full GAM fit, reduced model (linear-only), RSS/edf
      - nested ANOVA (F/p) and partial R^2; FDR across ROIs
      - returns smooth (zero-mean), derivative, p/q, edf, partial R2, age_of_max_change
    """
        
    # ---- return age at which the ROI developmental trajectory changes maximally  ----
    def age_of_max_change(grid: np.ndarray, derivative: np.ndarray) -> float:
        g = np.asarray(grid, dtype=float)
        d = np.asarray(derivative, dtype=float)
        ok = np.isfinite(g) & np.isfinite(d)
        if not np.any(ok):
            return np.nan
        # return grid value corresponding to argmax |deriv|
        idx = int(np.nanargmax(np.abs(d[ok])))
        return float(g[ok][idx])
    
    # ---- input checks ----
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be pandas.DataFrame with column names, including gam_cfg.smooth_var")
    if gam_cfg.smooth_var not in X.columns:
        raise ValueError(f"{gam_cfg.smooth_var} not in X columns")
    N = len(X)
    if not (len(A_lh_paths)==len(B_lh_paths)==len(A_rh_paths)==len(B_rh_paths)==N):
        raise ValueError("A/B path lists length must equal rows of X")
    
    # --- parcellation image/data and mask arrays ---
    lab_lh = _load_surf_labels(parc_lh, hemi_hint="lh", map_idx=0)
    lab_rh = _load_surf_labels(parc_rh, hemi_hint="rh", map_idx=0)
    
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

    # ---- build Y matrix: ROI x subject means of Delta ----
    Y = np.full((R, N), np.nan, dtype=np.float64)
    # helper metric loader (user must supply one that supports CIFTI/GIFTI; else assume paths are arrays)
    def _load_metric(p, dtype, hemi_hint):
        if isinstance(p, str):
            if hemi_loader_metric is None:
                raise ValueError("hemi_loader_metric required to read CIFTI/GIFTI metric paths")
            return hemi_loader_metric(p, dtype, hemi_hint)
        else:
            arr = np.asarray(p)
            return arr.reshape(-1)

    for i in range(N):
        A_lh = _load_metric(A_lh_paths[i], np.float32, "lh")
        B_lh = _load_metric(B_lh_paths[i], np.float32, "lh")
        A_rh = _load_metric(A_rh_paths[i], np.float32, "rh")
        B_rh = _load_metric(B_rh_paths[i], np.float32, "rh")
        d_lh = (A_lh - B_lh)
        d_rh = (A_rh - B_rh)
        # LH
        if kept_idx_lh.size > 0:
            vals = d_lh[kept_idx_lh].astype(np.float64, copy=False)
            fin = np.isfinite(vals)
            if fin.any():
                sums = np.bincount(roi_index_lh[fin], weights=vals[fin], minlength=R)
                cnts = np.bincount(roi_index_lh[fin], minlength=R)
                means = np.divide(sums, cnts, out=np.full(R, np.nan), where=cnts>0)
                # store only lh ROIs
                lh_mask = np.zeros(R, dtype=bool) 
                lh_mask[[key_to_idx[('lh', int(l))] for l in uniq_lh]] = True
                Y[lh_mask, i] = means[lh_mask]
        # RH
        if kept_idx_rh.size > 0:
            vals = d_rh[kept_idx_rh].astype(np.float64, copy=False)
            fin = np.isfinite(vals)
            if fin.any():
                sums = np.bincount(roi_index_rh[fin], weights=vals[fin], minlength=R)
                cnts = np.bincount(roi_index_rh[fin], minlength=R)
                means = np.divide(sums, cnts, out=np.full(R, np.nan), where=cnts>0)
                rh_mask = np.zeros(R, dtype=bool)
                rh_mask[[key_to_idx[('rh', int(l))] for l in uniq_rh]] = True
                Y[rh_mask, i] = means[rh_mask]

    # ---- build age grid ----
    smooth_col = gam_cfg.smooth_var
    age = np.asarray(X[smooth_col], dtype=np.float64)
    if gam_cfg.grid_kind == "quantile":
        qs = np.linspace(0.01, 0.99, gam_cfg.grid_size)
        grid = np.unique(np.quantile(age[np.isfinite(age)], qs))
    else:
        grid = np.linspace(np.nanmin(age), np.nanmax(age), gam_cfg.grid_size)
    G = grid.size

    # ---- prepare storage ----
    smooth_zero = np.full((R, G), np.nan, dtype=np.float32)
    derivative = np.full((R, G), np.nan, dtype=np.float32)
    partial_r2 = np.full(R, np.nan, dtype=np.float32)
    p_age = np.full(R, np.nan, dtype=np.float32)
    F_age = np.full(R, np.nan, dtype=np.float32)
    df1_arr = np.full(R, np.nan, dtype=np.float32)
    df2_arr = np.full(R, np.nan, dtype=np.float32)
    edf_arr = np.full(R, np.nan, dtype=np.float32)
    edf_red_arr = np.full(R, np.nan, dtype=np.float32)
    n_eff = np.zeros(R, dtype=int)
    age_max_change = np.full(R, np.nan, dtype=np.float32)

    # prepare linear covariates matrix
    linear_cols = [c for c in X.columns if c != smooth_col]
    X_lin_all = np.asarray(X[linear_cols], dtype=np.float64) if len(linear_cols)>0 else None
    Xs_all = age.reshape(-1,1)

    lam_grid_vals = 10.0 ** np.linspace(*gam_cfg.lam_grid)

    # ---- ROI loop ----
    for r in range(R):
        y = Y[r, :].copy()
        ok = np.isfinite(y) & np.isfinite(age)
        if X_lin_all is not None:
            ok &= np.all(np.isfinite(X_lin_all), axis=1)
        idx = np.where(ok)[0]
        n_eff[r] = idx.size
        # skip small n
        if idx.size < max(12, gam_cfg.k_splines + 2):
            continue

        ysub = y[idx]
        Xs = Xs_all[idx, :]
        Xlin = X_lin_all[idx, :] if X_lin_all is not None else None

        # Fit full/reduced with helper
        rss_f, edf_f, rss_r, edf_r, model_full = Sts.fit_gam_full_and_reduced(
            ysub, Xs, Xlin,
            k_splines=gam_cfg.k_splines,
            lam_grid=lam_grid_vals,
            max_iter=1000,
            verbose=gam_cfg.verbose
        )

        # nested ANOVA
        F, p, df1, df2, pr2 = Sts.nested_anova_gaussian(
            rss_full=rss_f, rss_reduced=rss_r,
            n_obs=idx.size, 
            edf_full=edf_f, edf_reduced=edf_r)
        
        F_age[r] = F; p_age[r] = p; partial_r2[r] = pr2; df1_arr[r]=df1; df2_arr[r]=df2
        edf_arr[r] = edf_f; edf_red_arr[r] = edf_r

        # build grid with covariates at mean
        if Xlin is not None and Xlin.size > 0:
            lin_mean = np.nanmean(Xlin, axis=0, keepdims=True)  # shape (1, p_lin)
            Zgrid = np.hstack([grid.reshape(-1,1), np.repeat(lin_mean, grid.size, axis=0)])
        else:
            Zgrid = grid.reshape(-1,1)

        fgrid = model_full.predict(Zgrid).astype(np.float64).reshape(-1)
        if gam_cfg.center_smooth:
            f0 = fgrid - np.nanmean(fgrid)
        else:
            f0 = fgrid
            
        smooth_zero[r, :] = f0.astype(np.float32, copy=False)

        # derivative numeric
        df = np.full(G, np.nan, dtype=np.float64)
        if G >= 3:
            df[1:-1] = (f0[2:] - f0[:-2]) / (grid[2:] - grid[:-2])
            df[0] = (f0[1] - f0[0]) / (grid[1] - grid[0])
            df[-1] = (f0[-1] - f0[-2]) / (grid[-1] - grid[-2])
            
        derivative[r, :] = df.astype(np.float32, copy=False)

        # age of max change
        age_max_change[r] = age_of_max_change(grid, df)

        # optional save per ROI diagnostics
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"roi{r:04d}_smooth.npy"), f0)
            np.save(os.path.join(save_dir, f"roi{r:04d}_deriv.npy"), df)

    # ---- FDR across ROIs for age effect ----
    fdr = Sts.fdr_bh(p_age, alpha=0.05)
    qvals = fdr["qvals"]; reject = fdr["reject"]

    results = {
        "roi_table": roi_table,
        "grid": grid,
        "smooth_zero_mean": smooth_zero,   # (R, G)
        "derivative": derivative,          # (R, G)
        "F_age": F_age, "p_age": p_age, "q_age": qvals, "reject_age_fdr": reject,
        "partial_r2": partial_r2,
        "edf_full": edf_arr, "edf_reduced": edf_red_arr,
        "n_eff": n_eff,
        "age_of_max_change": age_max_change
    }
    return results
