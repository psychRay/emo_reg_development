#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 01:13:14 2026

@author: dingrui
"""

"""
Identify four brain systems of emotion regulation following Bo et al. (2024),
the logic details please ref to Fig. 1b. This will produce results directly 
related to Fig. 4 in the manuscript.

Input options:
1. NIfTI images: .nii or .nii.gz
2. Surface GIFTI images: paired left/right .func.gii files

Required conditions:
- Look neutral
- Look negative
- Regulate negative

Outputs:
- t maps for Emotion generation and Reappraisal
- log10(BF10) maps for both contrasts
- binary masks for:
    1. Reappraisal only
    2. Common appraisal
    3. Non-modifiable emotion generation
    4. Modifiable emotion generation
"""

# from __future__ import annotations

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from pathlib import Path
from scipy.special import roots_genlaguerre
from scipy import ndimage
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_glass_brain

from typing import Callable, Iterable, Union, Sequence
ArrayLike = Union[np.ndarray, Sequence[float]]
SummaryFunc = Callable[[np.ndarray], float]

#%% NECESSARY FUNCTIONS
# -----------------------------
# Bayes factor: Rouder JZS t-test
# -----------------------------

def compute_group_mask_fast(
    img_files,
    out_file=None,
    threshold=1.0,
    use_abs=True,
    exclude_nan=True,
    dtype=np.uint16,
):
    """
    Fast group mask computation for a list of NIfTI images.

    Parameters
    ----------
    img_files : list of str or Path
        Input NIfTI image paths. All images must have the same shape and affine.

    out_file : str or Path, optional
        Output group mask filename.

    threshold : float, default=1.0
        Proportion of subjects required for a voxel to be included.
        - 1.0 means intersection mask: voxel must be valid in all images.
        - 0.8 means voxel must be valid in at least 80% images.

    use_abs : bool, default=True
        If True, valid voxels are defined as abs(data) > 0.
        If False, valid voxels are defined as data != 0.

    exclude_nan : bool, default=True
        If True, NaN and Inf voxels are treated as invalid.

    dtype : numpy dtype, default=np.uint16
        dtype for counting valid voxels.

    Returns
    -------
    mask_img : nibabel.Nifti1Image
        Binary group mask image.
    """

    img_files = [Path(f) for f in img_files]
    if len(img_files) == 0:
        raise ValueError("No input images were provided.")

    ref_img = nib.load(str(img_files[0]))
    ref_shape = ref_img.shape[:3]
    ref_affine = ref_img.affine
    ref_header = ref_img.header.copy()

    count = np.zeros(ref_shape, dtype=dtype)

    for f in img_files:
        img = nib.load(str(f))

        if img.shape[:3] != ref_shape:
            raise ValueError(f"Shape mismatch: {f} has shape {img.shape}, expected {ref_shape}")

        if not np.allclose(img.affine, ref_affine):
            raise ValueError(f"Affine mismatch: {f}")

        data = np.asanyarray(img.dataobj)

        if data.ndim == 4:
            if data.shape[3] != 1:
                raise ValueError(f"4D image with more than one volume is not supported: {f}")
            data = data[..., 0]

        if use_abs:
            valid = np.abs(data) > 0
        else:
            valid = data != 0

        if exclude_nan:
            valid &= np.isfinite(data)

        count += valid.astype(dtype)

    min_count = int(np.ceil(threshold * len(img_files)))
    group_mask = count >= min_count

    ref_header.set_data_dtype(np.uint8)
    mask_img = nib.Nifti1Image(group_mask.astype(np.uint8), ref_affine, ref_header)

    if out_file is not None:
        nib.save(mask_img, str(out_file))

    return mask_img


def log10_bf10_from_t(
    t: np.ndarray,
    n: int,
    r: float = np.sqrt(2) / 2,
    n_quad: int = 64,
    chunk_size: int = 20000,
) -> np.ndarray:
    """
    Compute log10(BF10) from one-sample/group-level t statistics using the
    Rouder et al. JZS Bayes factor approximation.

    BF10 > 10 means strong evidence for the alternative.
    BF10 < 0.1 means strong evidence for the null.

    Parameters
    ----------
    t : array
        t statistics, shape = (n_features,)
    n : int
        Sample size or effective sample size.
    r : float
        Cauchy prior scale. Bo et al. used r = 0.707.
    n_quad : int
        Number of generalized Laguerre quadrature nodes.
    chunk_size : int
        Chunk size for memory-efficient computation.

    Returns
    -------
    log10_bf10 : array
        log10(BF10), same shape as t.
    """
    t = np.asarray(t, dtype=np.float64)
    out = np.full(t.shape, np.nan, dtype=np.float64)

    df = n - 1
    if df <= 0:
        raise ValueError("n must be at least 2.")

    valid = np.isfinite(t)
    t_valid = t[valid]

    # Transform the JZS integral using h = 1 / (2g).
    # Integral becomes 1/sqrt(pi) * ∫ h^(-1/2) exp(-h) f(h) dh.
    nodes, weights = roots_genlaguerre(n_quad, alpha=-0.5)
    g = 1.0 / (2.0 * nodes)
    a = 1.0 + n * g * (r ** 2)

    log_sqrt_pi = 0.5 * np.log(np.pi)

    for start in range(0, t_valid.size, chunk_size):
        stop = min(start + chunk_size, t_valid.size)
        tt = t_valid[start:stop]
        tt2 = tt[:, None] ** 2

        # Denominator of BF01, therefore numerator of BF10
        integrand = (
            a[None, :] ** (-0.5)
            * (1.0 + tt2 / (a[None, :] * df)) ** (-(df + 1.0) / 2.0)
        )
        denom = np.sum(weights[None, :] * integrand, axis=1) / np.sqrt(np.pi)

        # Numerator of BF01
        null_like = (1.0 + (tt ** 2) / df) ** (-(df + 1.0) / 2.0)

        # BF10 = denom / null_like
        log_bf10 = np.log(denom) - np.log(null_like)
        out_valid = log_bf10 / np.log(10.0)
        out[np.where(valid)[0][start:stop]] = out_valid

    return out


# -----------------------------
# Second-level GLM
# -----------------------------

def second_level_t(
    Y: np.ndarray,
    design_matrix: np.ndarray | None = None,
    contrast: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Fit mass-univariate second-level GLM.

    Parameters
    ----------
    Y : array, shape = (n_subjects, n_features)
        Subject-level contrast data.
    design_matrix : array, optional
        Second-level design matrix. If None, intercept-only model is used.
        For covariate-adjusted group effects, include intercept + covariates.
    contrast : array, optional
        Contrast vector for second-level GLM.
        If None, tests the intercept.

    Returns
    -------
    effect : array
        Contrast estimate.
    t : array
        t statistic.
    df : int
        Residual degrees of freedom.
    """
    Y = np.asarray(Y, dtype=np.float64)
    n_sub = Y.shape[0]

    if design_matrix is None:
        X = np.ones((n_sub, 1), dtype=np.float64)
    else:
        X = np.asarray(design_matrix, dtype=np.float64)
        if X.shape[0] != n_sub:
            raise ValueError("design_matrix rows must match number of subjects.")

    if contrast is None:
        c = np.zeros(X.shape[1], dtype=np.float64)
        c[0] = 1.0
    else:
        c = np.asarray(contrast, dtype=np.float64)
        if c.shape[0] != X.shape[1]:
            raise ValueError("contrast length must match design_matrix columns.")

    pinv_X = np.linalg.pinv(X)
    beta = pinv_X @ Y
    fitted = X @ beta
    resid = Y - fitted

    rank = np.linalg.matrix_rank(X)
    df = n_sub - rank
    if df <= 0:
        raise ValueError("Not enough degrees of freedom for second-level GLM.")

    XtX_inv = np.linalg.pinv(X.T @ X)
    c_var = float(c @ XtX_inv @ c.T)

    sigma2 = np.sum(resid ** 2, axis=0) / df
    effect = c @ beta
    se = np.sqrt(sigma2 * c_var)

    t = np.full(effect.shape, np.nan, dtype=np.float64)
    ok = se > 0
    t[ok] = effect[ok] / se[ok]

    return effect, t, df


# -----------------------------
# Image I/O helpers
# -----------------------------

class NiftiBackend:
    def __init__(self, mask_img=None, smoothing_fwhm=None):
        self.masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=smoothing_fwhm)
        self.fitted = False

    def fit_transform(self, imgs: list[str]) -> np.ndarray:
        data = self.masker.fit_transform(imgs)
        self.fitted = True
        return data

    def transform(self, imgs) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("NiftiBackend must be fit first.")
        return self.masker.transform(imgs)

    def save_vector(self, vec: np.ndarray, filename: str):
        if not self.fitted:
            raise RuntimeError("NiftiBackend must be fit before saving vectors.")

        vec = np.asarray(vec).reshape(-1)

        expected_n = self.masker.n_elements_
        if vec.size != expected_n:
            raise ValueError(
                f"Cannot save vector with wrong size. "
                f"Expected {expected_n}, got {vec.size}. "
                f"Filename: {filename}"
            )

        if not filename.endswith(".nii") and not filename.endswith(".nii.gz"):
            filename = filename + ".nii.gz"

        img = self.masker.inverse_transform(vec)
        img.to_filename(filename)


class GiftiSurfaceBackend:
    """
    Handles paired left/right .func.gii files.

    Each subject image should be represented as:
        (left_func_gii, right_func_gii)
    """

    def __init__(self):
        self.n_left = None
        self.n_right = None
        self.left_template = None
        self.right_template = None

    @staticmethod
    def _load_pair(pair: tuple[str, str]) -> np.ndarray:
        left_img = nib.load(pair[0])
        right_img = nib.load(pair[1])
        left = np.asarray(left_img.darrays[0].data, dtype=np.float64)
        right = np.asarray(right_img.darrays[0].data, dtype=np.float64)
        return np.concatenate([left, right])

    def fit_transform(self, imgs: list[tuple[str, str]]) -> np.ndarray:
        self.left_template = nib.load(imgs[0][0])
        self.right_template = nib.load(imgs[0][1])
        self.n_left = self.left_template.darrays[0].data.size
        self.n_right = self.right_template.darrays[0].data.size
        return np.vstack([self._load_pair(p) for p in imgs])

    transform = fit_transform

    def save_vector(self, vec: np.ndarray, filename_prefix: str):
        vec = np.asarray(vec)
        left_data = vec[: self.n_left].astype(np.float32)
        right_data = vec[self.n_left :].astype(np.float32)

        left_img = nib.gifti.GiftiImage()
        right_img = nib.gifti.GiftiImage()

        left_img.add_gifti_data_array(nib.gifti.GiftiDataArray(left_data))
        right_img.add_gifti_data_array(nib.gifti.GiftiDataArray(right_data))

        nib.save(left_img, filename_prefix + ".L.func.gii")
        nib.save(right_img, filename_prefix + ".R.func.gii")


def make_backend(example_img, mask_img=None):
    """
    Automatically choose NIfTI or paired GIFTI backend.
    """
    if isinstance(example_img, tuple):
        return GiftiSurfaceBackend()
    return NiftiBackend(mask_img=mask_img)


# -----------------------------
# Optional cluster filtering for NIfTI masks
# -----------------------------

def filter_small_clusters_nifti(
    backend: NiftiBackend,
    mask_vec: np.ndarray,
    min_cluster_size: int = 15,
) -> np.ndarray:
    """
    Remove small 3D clusters from a vectorized NIfTI mask.

    This version is robust to nilearn versions where masker.transform()
    returns either shape (1, n_voxels) or shape (n_voxels,).
    """
    mask_vec = np.asarray(mask_vec).reshape(-1).astype(np.float32)

    expected_n = backend.masker.n_elements_
    if mask_vec.size != expected_n:
        raise ValueError(
            f"mask_vec has wrong size before cluster filtering. "
            f"Expected {expected_n}, got {mask_vec.size}."
        )

    mask_img = backend.masker.inverse_transform(mask_vec)
    data = np.asarray(mask_img.get_fdata()) > 0

    labeled, n_lab = ndimage.label(data)
    out = np.zeros_like(data, dtype=bool)

    for lab in range(1, n_lab + 1):
        cluster = labeled == lab
        if int(cluster.sum()) >= min_cluster_size:
            out[cluster] = True

    filtered_img = nib.Nifti1Image(
        out.astype(np.float32),
        affine=mask_img.affine,
        header=mask_img.header,
    )

    filtered_vec = backend.masker.transform(filtered_img)

    # Important: do not use [0] here.
    filtered_vec = np.asarray(filtered_vec).reshape(-1).astype(bool)

    if filtered_vec.size != expected_n:
        raise ValueError(
            f"Filtered vector has wrong size after transform. "
            f"Expected {expected_n}, got {filtered_vec.size}."
        )

    return filtered_vec


# -----------------------------
# Main function
# -----------------------------

def identify_systems(
    look_neutral_imgs,
    look_negative_imgs,
    regulate_negative_imgs,
    out_dir: str,
    mask_img=None,
    design_matrix: np.ndarray | None = None,
    second_level_contrast: np.ndarray | None = None,
    bf_alt_log10: float = 1.0,
    bf_null_log10: float = -1.0,
    enforce_positive_condition: bool = True,
    min_cluster_size: int | None = 15,
):
    """
    Identify four systems following Bo et al. Fig. 1b.

    Parameters
    ----------
    look_neutral_imgs, look_negative_imgs, regulate_negative_imgs :
        For NIfTI:
            list of file paths.
        For paired GIFTI:
            list of (left_path, right_path) tuples.
    out_dir : str
        Output directory.
    mask_img : str or Niimg, optional
        NIfTI mask. Ignored for GIFTI.
    design_matrix : array, optional
        Second-level design matrix. If None, intercept-only model.
        Example: np.column_stack([np.ones(n), age, sex])
    second_level_contrast : array, optional
        GLM contrast. If None, tests the intercept.
        For age moderation, use contrast = [0, 1, 0] if age is column 2.
    bf_alt_log10 : float
        log10 BF10 threshold for alternative evidence. Default 1 = BF10 > 10.
    bf_null_log10 : float
        log10 BF10 threshold for null evidence. Default -1 = BF10 < 0.1.
    enforce_positive_condition : bool
        If True, applies Bo-style extra positivity constraints:
        - Reappraisal-only: Regulate negative mean > 0
        - Common / Non-modifiable / Modifiable: Look negative mean > 0
    min_cluster_size : int or None
        Remove NIfTI clusters smaller than this. Default follows Bo et al. style.
        Ignored for GIFTI.

    Returns
    -------
    results : dict
        Contains effects, t maps, log10 BF maps, and four binary masks as arrays.
    """
    os.makedirs(out_dir, exist_ok=True)

    n = len(look_neutral_imgs)
    if not (len(look_negative_imgs) == n and len(regulate_negative_imgs) == n):
        raise ValueError("All condition lists must have the same number of subjects.")

    backend = make_backend(look_neutral_imgs[0], mask_img=mask_img)

    Y_neu = backend.fit_transform(look_neutral_imgs)
    Y_neg = backend.transform(look_negative_imgs)
    Y_reg = backend.transform(regulate_negative_imgs)

    # Subject-level contrasts
    Y_emo = Y_neg - Y_neu      # Look negative - Look neutral
    Y_reapp = Y_reg - Y_neg    # Regulate negative - Look negative

    # Second-level t tests
    emo_eff, emo_t, emo_df = second_level_t(
        Y_emo,
        design_matrix=design_matrix,
        contrast=second_level_contrast,
    )
    reapp_eff, reapp_t, reapp_df = second_level_t(
        Y_reapp,
        design_matrix=design_matrix,
        contrast=second_level_contrast,
    )

    # For intercept-only models, this exactly follows the one-sample t-test BF.
    # For covariate-adjusted GLM tests, df + 1 is used as an effective sample size approximation.
    emo_log10_bf10 = log10_bf10_from_t(emo_t, n=emo_df + 1)
    reapp_log10_bf10 = log10_bf10_from_t(reapp_t, n=reapp_df + 1)

    # Evidence masks
    emo_pos_alt = (emo_log10_bf10 >= bf_alt_log10) & (emo_t > 0)
    reapp_pos_alt = (reapp_log10_bf10 >= bf_alt_log10) & (reapp_t > 0)
    reapp_neg_alt = (reapp_log10_bf10 >= bf_alt_log10) & (reapp_t < 0)

    emo_null = emo_log10_bf10 <= bf_null_log10
    reapp_null = reapp_log10_bf10 <= bf_null_log10

    # Bo et al. Fig. 1b systems
    reappraisal_only = reapp_pos_alt & emo_null
    common_appraisal = emo_pos_alt & reapp_pos_alt
    nonmodifiable_emotion = emo_pos_alt & reapp_null
    modifiable_emotion = emo_pos_alt & reapp_neg_alt

    if enforce_positive_condition:
        # Bo et al. added positive activation constraints in relevant conditions.
        reg_mean = np.nanmean(Y_reg, axis=0)
        neg_mean = np.nanmean(Y_neg, axis=0)

        reappraisal_only &= reg_mean > 0
        common_appraisal &= neg_mean > 0
        nonmodifiable_emotion &= neg_mean > 0
        modifiable_emotion &= neg_mean > 0

    # Optional 3D cluster filtering for NIfTI only
    if isinstance(backend, NiftiBackend) and min_cluster_size is not None:
        reappraisal_only = filter_small_clusters_nifti(
            backend, reappraisal_only, min_cluster_size
        )
        common_appraisal = filter_small_clusters_nifti(
            backend, common_appraisal, min_cluster_size
        )
        nonmodifiable_emotion = filter_small_clusters_nifti(
            backend, nonmodifiable_emotion, min_cluster_size
        )
        modifiable_emotion = filter_small_clusters_nifti(
            backend, modifiable_emotion, min_cluster_size
        )

    results = {
        "emotion_generation_effect": emo_eff,
        "emotion_generation_t": emo_t,
        "emotion_generation_log10_bf10": emo_log10_bf10,
        "reappraisal_effect": reapp_eff,
        "reappraisal_t": reapp_t,
        "reappraisal_log10_bf10": reapp_log10_bf10,
        "reappraisal_only": reappraisal_only.astype(np.uint8),
        "common_appraisal": common_appraisal.astype(np.uint8),
        "nonmodifiable_emotion": nonmodifiable_emotion.astype(np.uint8),
        "modifiable_emotion": modifiable_emotion.astype(np.uint8),
    }

    # Save outputs
    backend.save_vector(emo_t, os.path.join(out_dir, "emotion_generation_t"))
    backend.save_vector(reapp_t, os.path.join(out_dir, "reappraisal_t"))
    backend.save_vector(emo_log10_bf10, os.path.join(out_dir, "emotion_generation_log10BF10"))
    backend.save_vector(reapp_log10_bf10, os.path.join(out_dir, "reappraisal_log10BF10"))

    backend.save_vector(results["reappraisal_only"], os.path.join(out_dir, "mask_reappraisal_only"))
    backend.save_vector(results["common_appraisal"], os.path.join(out_dir, "mask_common_appraisal"))
    backend.save_vector(results["nonmodifiable_emotion"], os.path.join(out_dir, "mask_nonmodifiable_emotion"))
    backend.save_vector(results["modifiable_emotion"], os.path.join(out_dir, "mask_modifiable_emotion"))

    return results


def _load_brain_data_as_2d(img_file):
    """
    Load NIfTI, GIFTI, or CIFTI image as a 2D array.

    Returns
    -------
    data_2d : ndarray, shape (n_maps, n_features)
        Each row is one map/volume/time point, and each column is one voxel,
        vertex, or grayordinate.

    img_type : str
        One of {"nifti", "gifti", "cifti"}.

    original_shape : tuple
        Original data shape before flattening.
    """

    img_file = Path(img_file)
    img = nib.load(str(img_file))

    if isinstance(img, nib.Nifti1Image):
        data = np.asanyarray(img.dataobj)
        original_shape = data.shape

        if data.ndim == 3:
            data_2d = data.reshape(1, -1)
        elif data.ndim == 4:
            # Shape: X, Y, Z, T -> T, voxels
            data_2d = np.moveaxis(data, -1, 0).reshape(data.shape[-1], -1)
        else:
            raise ValueError(
                f"NIfTI image must be 3D or 4D, got shape {data.shape}: {img_file}"
            )

        return data_2d, "nifti", original_shape

    elif isinstance(img, nib.gifti.GiftiImage):
        arrays = [np.asarray(darray.data) for darray in img.darrays]

        if len(arrays) == 0:
            raise ValueError(f"GIFTI image contains no data arrays: {img_file}")

        # Most metric GIFTI files have one data array of shape (n_vertices,).
        # Some functional GIFTI files may contain multiple arrays.
        arrays = [arr.reshape(-1) for arr in arrays]
        data_2d = np.vstack(arrays)
        original_shape = tuple(arr.shape for arr in arrays)

        return data_2d, "gifti", original_shape

    elif isinstance(img, nib.cifti2.Cifti2Image):
        data = np.asanyarray(img.dataobj)
        original_shape = data.shape

        if data.ndim == 1:
            data_2d = data.reshape(1, -1)
        elif data.ndim == 2:
            # Typical CIFTI shape:
            # dscalar:  n_maps x n_grayordinates
            # dtseries: n_timepoints x n_grayordinates
            data_2d = data
        else:
            # Rare, but keep behavior explicit.
            data_2d = data.reshape(data.shape[0], -1)

        return data_2d, "cifti", original_shape

    else:
        raise TypeError(f"Unsupported image type for file: {img_file}")


def _load_mask_as_1d(mask_file, mask_index=0):
    """
    Load a NIfTI, GIFTI, or CIFTI mask as a 1D array.

    Parameters
    ----------
    mask_file : str or Path
        Mask image file.

    mask_index : int, default=0
        If the mask file contains multiple maps, use this index.

    Returns
    -------
    mask_1d : ndarray, shape (n_features,)
    mask_type : str
    original_shape : tuple
    """

    data_2d, mask_type, original_shape = _load_brain_data_as_2d(mask_file)

    if data_2d.shape[0] == 1:
        mask_1d = data_2d[0]
    else:
        if mask_index >= data_2d.shape[0]:
            raise IndexError(
                f"mask_index={mask_index} is out of range for mask with "
                f"{data_2d.shape[0]} maps: {mask_file}"
            )
        mask_1d = data_2d[mask_index]

    return mask_1d.reshape(-1), mask_type, original_shape


def _get_summary_func(summary):
    """
    Convert a summary name or callable into a function.

    Supported built-in names
    ------------------------
    mean, median, sum, std, var, min, max, abs_mean, abs_median,
    max_abs, pos_mean, neg_mean, nonzero_mean, count, count_nonzero,
    pXX, percentile_XX

    Examples
    --------
    p95, p5, percentile_90
    """

    if callable(summary):
        return summary

    summary = str(summary).lower()

    def _safe_mean(x):
        return np.nanmean(x) if x.size > 0 else np.nan

    def _safe_median(x):
        return np.nanmedian(x) if x.size > 0 else np.nan

    funcs = {
        "mean": lambda x: np.nanmean(x),
        "median": lambda x: np.nanmedian(x),
        "sum": lambda x: np.nansum(x),
        "std": lambda x: np.nanstd(x),
        "var": lambda x: np.nanvar(x),
        "min": lambda x: np.nanmin(x),
        "max": lambda x: np.nanmax(x),
        "abs_mean": lambda x: np.nanmean(np.abs(x)),
        "abs_median": lambda x: np.nanmedian(np.abs(x)),
        "max_abs": lambda x: np.nanmax(np.abs(x)),
        "count": lambda x: np.sum(np.isfinite(x)),
        "count_nonzero": lambda x: np.sum(np.isfinite(x) & (x != 0)),
        "pos_mean": lambda x: _safe_mean(x[x > 0]),
        "neg_mean": lambda x: _safe_mean(x[x < 0]),
        "pos_median": lambda x: _safe_median(x[x > 0]),
        "neg_median": lambda x: _safe_median(x[x < 0]),
        "nonzero_mean": lambda x: _safe_mean(x[x != 0]),
        "nonzero_median": lambda x: _safe_median(x[x != 0]),
    }

    if summary in funcs:
        return funcs[summary]

    if summary.startswith("p") and summary[1:].replace(".", "", 1).isdigit():
        q = float(summary[1:])
        return lambda x: np.nanpercentile(x, q)

    if summary.startswith("percentile_"):
        q = float(summary.replace("percentile_", ""))
        return lambda x: np.nanpercentile(x, q)

    raise ValueError(f"Unknown summary method: {summary}")


def extract_mask_summary_values(
    img_files,
    mask_files,
    summaries=("mean",),
    mask_threshold=0,
    mask_mode="positive",
    mask_index=0,
    ignore_nan=True,
    ignore_zero=False,
    image_names=None,
    mask_names=None,
    return_wide=False,
):
    """
    Extract summary values from a group of brain images within one or more masks.

    Parameters
    ----------
    img_files : list of str or Path
        Brain image files. Supported formats include NIfTI, GIFTI, and CIFTI.

    mask_files : list of str or Path
        Mask files. Each mask should have the same feature dimension as the images
        it is applied to.

    summaries : str, callable, or list of str/callable, default=("mean",)
        Summary methods to compute within each mask.

        Built-in summary names include:
        mean, median, sum, std, var, min, max,
        abs_mean, abs_median, max_abs,
        pos_mean, neg_mean, pos_median, neg_median,
        nonzero_mean, nonzero_median,
        count, count_nonzero,
        p95, p5, percentile_90, etc.

        A custom callable can also be supplied. It should accept a 1D array and
        return a scalar.

    mask_threshold : float, default=0
        Threshold used to binarize the mask.

    mask_mode : {"positive", "nonzero", "negative", "absolute"}, default="positive"
        Rule for selecting mask elements.

        - "positive": mask > mask_threshold
        - "nonzero": mask != 0
        - "negative": mask < mask_threshold
        - "absolute": abs(mask) > mask_threshold

    mask_index : int, default=0
        If a mask file contains multiple maps, use this map index.

    ignore_nan : bool, default=True
        If True, remove NaN/Inf values before computing summaries.

    ignore_zero : bool, default=False
        If True, remove zero-valued image elements before computing summaries.
        This can be useful when background zeros remain inside a broad mask.

    image_names : list of str, optional
        User-defined names for images. If None, filenames are used.

    mask_names : list of str, optional
        User-defined names for masks. If None, filenames are used.

    return_wide : bool, default=False
        If True, return a wide-format DataFrame.
        If False, return a long-format DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        Long-format columns:
        image, image_file, image_type, map_index, mask, mask_file,
        n_features, n_valid, summary, value

        Wide-format columns:
        image, image_file, image_type, map_index, mask, mask_file,
        n_features, n_valid, plus one column per summary.
    """

    img_files = [Path(f) for f in img_files]
    mask_files = [Path(f) for f in mask_files]

    if len(img_files) == 0:
        raise ValueError("img_files is empty.")

    if len(mask_files) == 0:
        raise ValueError("mask_files is empty.")

    if isinstance(summaries, (str, bytes)) or callable(summaries):
        summaries = [summaries]

    summary_funcs = {}
    for summary in summaries:
        if callable(summary):
            name = getattr(summary, "__name__", "custom_summary")
            summary_funcs[name] = summary
        else:
            summary_funcs[str(summary)] = _get_summary_func(summary)

    if image_names is None:
        image_names = [f.name for f in img_files]
    if mask_names is None:
        mask_names = [f.name for f in mask_files]

    if len(image_names) != len(img_files):
        raise ValueError("image_names must have the same length as img_files.")

    if len(mask_names) != len(mask_files):
        raise ValueError("mask_names must have the same length as mask_files.")

    mask_mode = mask_mode.lower()
    valid_mask_modes = {"positive", "nonzero", "negative", "absolute"}
    if mask_mode not in valid_mask_modes:
        raise ValueError(f"mask_mode must be one of {valid_mask_modes}.")

    loaded_masks = []

    for mask_file, mask_name in zip(mask_files, mask_names):
        mask_data, mask_type, mask_shape = _load_mask_as_1d(
            mask_file,
            mask_index=mask_index,
        )

        if mask_mode == "positive":
            mask_bool = mask_data > mask_threshold
        elif mask_mode == "nonzero":
            mask_bool = mask_data != 0
        elif mask_mode == "negative":
            mask_bool = mask_data < mask_threshold
        elif mask_mode == "absolute":
            mask_bool = np.abs(mask_data) > mask_threshold

        mask_bool = np.asarray(mask_bool, dtype=bool)

        loaded_masks.append(
            {
                "mask_name": mask_name,
                "mask_file": str(mask_file),
                "mask_type": mask_type,
                "mask_shape": mask_shape,
                "mask_bool": mask_bool,
            }
        )

    rows = []

    for img_file, image_name in zip(img_files, image_names):
        data_2d, image_type, image_shape = _load_brain_data_as_2d(img_file)

        for map_index in range(data_2d.shape[0]):
            data_1d = np.asarray(data_2d[map_index]).reshape(-1)

            for mask_info in loaded_masks:
                mask_bool = mask_info["mask_bool"]

                if mask_bool.shape[0] != data_1d.shape[0]:
                    raise ValueError(
                        "Feature dimension mismatch between image and mask:\n"
                        f"Image: {img_file}, map_index={map_index}, "
                        f"n_features={data_1d.shape[0]}, image_shape={image_shape}\n"
                        f"Mask: {mask_info['mask_file']}, "
                        f"n_features={mask_bool.shape[0]}, "
                        f"mask_shape={mask_info['mask_shape']}"
                    )

                values = data_1d[mask_bool]

                if ignore_nan:
                    values = values[np.isfinite(values)]

                if ignore_zero:
                    values = values[values != 0]

                base_row = {
                    "image": image_name,
                    "image_file": str(img_file),
                    "image_type": image_type,
                    "map_index": map_index,
                    "mask": mask_info["mask_name"],
                    "mask_file": mask_info["mask_file"],
                    "mask_type": mask_info["mask_type"],
                    "n_features": int(mask_bool.sum()),
                    "n_valid": int(values.size),
                }

                for summary_name, summary_func in summary_funcs.items():
                    if values.size == 0:
                        value = np.nan
                    else:
                        value = summary_func(values)

                    rows.append(
                        {
                            **base_row,
                            "summary": summary_name,
                            "value": float(value) if np.ndim(value) == 0 else value,
                        }
                    )

    df = pd.DataFrame(rows)

    if return_wide:
        index_cols = [
            "image",
            "image_file",
            "image_type",
            "map_index",
            "mask",
            "mask_file",
            "mask_type",
            "n_features",
            "n_valid",
        ]

        df = (
            df.pivot_table(
                index=index_cols,
                columns="summary",
                values="value",
                aggfunc="first",
            )
            .reset_index()
        )

        df.columns.name = None

    return df


def mediation_bootstrap(
    data,
    x,
    m,
    y,
    covariates=None,
    n_boot=5000,
    seed=123,
    ci=95,
):
    """
    Bootstrap mediation analysis using two OLS models.

    Model 1:
        M ~ X + covariates

    Model 2:
        Y ~ X + M + covariates

    Indirect effect:
        a * b

    Parameters
    ----------
    data : pandas.DataFrame
        Input data.

    x : str
        Predictor / exposure variable.

    m : str
        Mediator variable.

    y : str
        Outcome variable.

    covariates : list of str, optional
        Covariates included in both mediator and outcome models.

    n_boot : int, default=5000
        Number of bootstrap samples.

    seed : int, default=123
        Random seed.

    ci : float, default=95
        Confidence interval width.

    Returns
    -------
    result : dict
        Mediation results including point estimates and bootstrap CI.
    """

    if covariates is None:
        covariates = []

    cols = [x, m, y] + covariates
    df = data[cols].dropna().copy()

    cov_str = " + ".join(covariates)

    if cov_str:
        med_formula = f"{m} ~ {x} + {cov_str}"
        out_formula = f"{y} ~ {x} + {m} + {cov_str}"
    else:
        med_formula = f"{m} ~ {x}"
        out_formula = f"{y} ~ {x} + {m}"

    med_fit = smf.ols(med_formula, data=df).fit()
    out_fit = smf.ols(out_formula, data=df).fit()

    a = med_fit.params[x]
    b = out_fit.params[m]
    c_prime = out_fit.params[x]

    total_fit = smf.ols(
        f"{y} ~ {x}" + (f" + {cov_str}" if cov_str else ""),
        data=df,
    ).fit()

    c_total = total_fit.params[x]
    indirect = a * b
    direct = c_prime
    total = c_total
    prop_mediated = indirect / total if total != 0 else np.nan

    rng = np.random.default_rng(seed)
    boot_indirect = []
    boot_direct = []
    boot_total = []

    n = len(df)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_df = df.iloc[idx]

        try:
            boot_med = smf.ols(med_formula, data=boot_df).fit()
            boot_out = smf.ols(out_formula, data=boot_df).fit()
            boot_total_model = smf.ols(
                f"{y} ~ {x}" + (f" + {cov_str}" if cov_str else ""),
                data=boot_df,
            ).fit()

            boot_a = boot_med.params[x]
            boot_b = boot_out.params[m]
            boot_indirect.append(boot_a * boot_b)
            boot_direct.append(boot_out.params[x])
            boot_total.append(boot_total_model.params[x])

        except Exception:
            continue

    boot_indirect = np.asarray(boot_indirect)
    boot_direct = np.asarray(boot_direct)
    boot_total = np.asarray(boot_total)

    alpha = (100 - ci) / 2

    result = {
        "n": n,
        "a": a,
        "b": b,
        "indirect": indirect,
        "direct": direct,
        "total": total,
        "proportion_mediated": prop_mediated,
        "indirect_ci_low": np.percentile(boot_indirect, alpha),
        "indirect_ci_high": np.percentile(boot_indirect, 100 - alpha),
        "direct_ci_low": np.percentile(boot_direct, alpha),
        "direct_ci_high": np.percentile(boot_direct, 100 - alpha),
        "total_ci_low": np.percentile(boot_total, alpha),
        "total_ci_high": np.percentile(boot_total, 100 - alpha),
        "boot_indirect": boot_indirect,
        "boot_direct": boot_direct,
        "boot_total": boot_total,
        "mediator_model": med_fit,
        "outcome_model": out_fit,
        "total_model": total_fit,
    }

    return result

#%% PREPARE DATA INPUT
# Fig. 4 group-level system classification and subject-level mask summaries.
if __name__ == "__main__":
    # data directories and subjects info
    dir_data_fmri = '/public/home/dingrui/BIDS_DATA'
    dir_data_behv = '/public/home/dingrui/fmri_analysis/data/beh'
    dir_data_save = '/public/home/dingrui/fmri_analysis/data/fmri'

    sub_info = pd.read_csv(
        os.path.join(dir_data_behv, 'participants_in_tfmri_demographics.csv'),
        sep=',', index_col=False
    )

    sub_ls_ER = np.loadtxt(os.path.join(dir_data_behv, 'sub_list_valid_4_ER.txt'), dtype=str)
    sub_ls_TG = np.loadtxt(os.path.join(dir_data_behv, 'sub_list_valid_4_TG.txt'), dtype=str)

    subs_rm_ls_ER = np.loadtxt(
        os.path.join(
            '/public/home/dingrui/fmri_analysis/data/fmri', 
            'subs_removed_from_2ndlevel.txt'), 
        dtype=str).tolist()

    sub_ls_test = sub_ls_ER.tolist()

    domain_ls = ['emo', 'soc']

    # -------------------------------------
    # beta images into GLM 2nd level model
    # -------------------------------------
    # valid subjects list
    subs_rm_ls = subs_rm_ls_ER.copy()
    sub_ls_test = [sub for sub in sub_ls_test if sub not in subs_rm_ls]

    # 1st level beta images of each condition
    betaImg_rpsl_ls = [os.path.join(dir_data_save, 
                                    f'res_1stlevel/{sub}', 
                                    'effectSize-reg_neg_Observe_across_runs.nii.gz')
                       for sub in sub_ls_test if sub not in subs_rm_ls]

    betaImg_lkng_ls = [os.path.join(dir_data_save, 
                                    f'res_1stlevel/{sub}', 
                                    'effectSize-pas_neg_Observe_across_runs.nii.gz')
                       for sub in sub_ls_test if sub not in subs_rm_ls]
    betaImg_lknt_ls = [os.path.join(dir_data_save, 
                                    f'res_1stlevel/{sub}', 
                                    'effectSize-pas_neu_Observe_across_runs.nii.gz')
                       for sub in sub_ls_test if sub not in subs_rm_ls]

    # --------------------------------------
    # design matrix for GLM 2nd level model
    # -------------------------------------- 
    # subs_info of the selected valid subjects:
    sub_idx = sub_info['sub_id'].isin(sub_ls_test)
    df_sub_cov = sub_info[['sub_id', 'age', 'gender', 'site_id']].loc[sub_idx]

    subs_age = []
    subs_sex = []
    subs_site = []

    for sub in sub_ls_test:
        sub_age = df_sub_cov['age'][df_sub_cov['sub_id']==sub]
        sub_sex = df_sub_cov['gender'][df_sub_cov['sub_id']==sub]
        sub_sit = df_sub_cov['site_id'][df_sub_cov['sub_id']==sub]
        
        subs_age.append(sub_age)
        subs_sex.append(sub_sex)
        subs_site.append(sub_sit)

    age_z  = (np.array(subs_age) - np.mean(subs_age))/np.std(subs_age)
    gender = np.array(subs_sex)
    sites  = np.array(subs_site)

    X = pd.DataFrame({
        'intercept': 1.0,
        # 'age_z': age_z[:, 0],
        'gender': gender[:, 0],
        'site': sites[:, 0],
        })

    #%% SYSTEM IDENTIFICATION
    # 
    output_dir = '/public/home/dingrui/fmri_analysis/res/res_bayes'
    mask95 = compute_group_mask_fast(betaImg_rpsl_ls+betaImg_lkng_ls+betaImg_lknt_ls, threshold=0.95)
    contrast = np.array([1, 0, 0])  # adjusted group mean

    #
    res_systems = identify_systems(
        look_neutral_imgs=betaImg_lknt_ls,
        look_negative_imgs=betaImg_lkng_ls,
        regulate_negative_imgs=betaImg_rpsl_ls,
        out_dir=output_dir,
        mask_img=mask95,
        design_matrix=X.values,
        second_level_contrast=contrast,
    )

    #%% QUICK VIEW OF SYSTEMS REGIONS
    # 
    rpsl_only = os.path.join(output_dir, 'mask_reappraisal_only.nii.gz')
    rpsl_comm = os.path.join(output_dir, 'mask_common_appraisal.nii.gz')
    emot_modify = os.path.join(output_dir, 'mask_modifiable_emotion.nii.gz')
    emot_nonmod = os.path.join(output_dir, 'mask_nonmodifiable_emotion.nii.gz')

    fig, axes = plt.subplots(2,2, figsize=(8, 3))

    for sys_mask, mask_name, ax in zip(
            [rpsl_only, rpsl_comm, emot_modify, emot_nonmod],
            ['reappraisal only', 'common reappraisal', 'modifiable emotion', 'non-modifiable emotion'],
            [axes[0,0], axes[0,1], axes[1,0], axes[1,1]],
    ):
        disp = plot_glass_brain(
            sys_mask, threshold=0.01,
            display_mode='lzry', 
            cmap='autumn_r', colorbar=False,
            axes=ax,
            annotate=False,
        )
        
        disp.title(mask_name, size=8, color='white', bgcolor='k')
        
    #%% EXTRACT STATS VALUE FROM INDIVIDUAL STATS IMAGES (1)
    #
    # masks of brain systems of emotion regulation
    mask_files = [rpsl_only, rpsl_comm, emot_modify, emot_nonmod]

    dict_stats = {}
    for img_files, img_name in zip(
            [betaImg_rpsl_ls, betaImg_lkng_ls, betaImg_lknt_ls],
            ['rpsl', 'lkng', 'lknt'],
    ): 
        
        df_stats = extract_mask_summary_values(
            img_files=img_files,
            mask_files=mask_files,
            summaries=("mean", "median", "std"),
            mask_threshold=0,
            mask_mode="positive",
            ignore_nan=True,
            ignore_zero=False,
            return_wide=True,
        )
        
        dict_stats[img_name] = df_stats

    for img_name, df in dict_stats.items():
        fpath_csv = os.path.join(output_dir, f'betaVals_in_ERsystems_cond-{img_name}_subjectLevel.csv')
        df.to_csv(fpath_csv, index=False)

    #%% EXTRACT STATS VALUE FROM INDIVIDUAL STATS IMAGES (2)
    #
    # masks of age-related brain systems of emotion regulation
    # here only three valid masks, since the common appraisal has no overlapping regions with age-effect mask
    mask_names = [
        'agePos_rpsl_only',
        'ageNeg_rpsl_only',
        'agePos_rpsl_comn',
        'ageNeg_rpsl_comn',
        'ageNeg_mody_emot',
        'agePos_nmdy_emot',
        'ageNeg_nmdy_emot',
    ]

    masks_overlap = [os.path.join(output_dir, f'mask-{mask_name}.nii.gz') for mask_name in mask_names]

    dict_stats_ovlpMask = {}
    for img_files, img_name in zip(
            [betaImg_rpsl_ls, betaImg_lkng_ls, betaImg_lknt_ls],
            ['rpsl', 'lkng', 'lknt'],
    ): 
        
        df_stats = extract_mask_summary_values(
            img_files=img_files,
            mask_files=masks_overlap,
            summaries=("mean", "median", "std"),
            mask_threshold=0,
            mask_mode="positive",
            ignore_nan=True,
            ignore_zero=False,
            return_wide=True,
        )
        
        dict_stats_ovlpMask[img_name] = df_stats
        
    for img_name, df in dict_stats_ovlpMask.items():
        fpath_csv = os.path.join(output_dir, f'betaVals_in_age_ERsystems_cond-{img_name}_subjectLevel.csv')
        df.to_csv(fpath_csv, index=False)

    #%%
