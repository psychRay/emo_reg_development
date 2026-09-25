#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 04:22:09 2025

@author: dingrui
"""

#%%
import warnings
warnings.filterwarnings('ignore', message='.*deprecated.*')

import os
import chardet as chd
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy import stats
from scipy import sparse
from scipy.stats import t as t_dist
from scipy.sparse.csgraph import connected_components
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from matplotlib import cm
from matplotlib.colors import Normalize
from typing import List, Optional, Union, Dict, Tuple, Sequence
from neuromaps.parcellate import Parcellater

SingleGii = Union[str, Path]
BiHemiGii = Tuple[SingleGii, SingleGii]
GiiInput = Union[SingleGii, BiHemiGii]
ArrayLike = Union[np.ndarray, list, tuple]
ImageLike = Union[
    str,
    Path,
    np.ndarray,
    nib.Nifti1Image,
    nib.Nifti2Image,
    nib.GiftiImage,
    nib.Cifti2Image,
]

import mylib.brain.surface as surf
import mylib.brain.corr as corr
import mylib.brain.image as img
import mylib.stat.stats as Sts
import mylib.plotting.brain_plot as bplot

from surfplot import Plot

#%% NECESSARY FUNCTIONS

def lsdir(dir_path, keyword=None):
    file_ls = os.listdir(dir_path)
    if keyword is not None:
        return [file for file in file_ls if keyword in file]
    else:
        return file_ls


def encoding_detect(filename):
    with open(filename, 'rb') as f:
        content = f.read(10000)
        res_chd = chd.detect(content)
        encoding= res_chd['encoding']
        return encoding
    

def read_imgs_from_bids(
        dir_data_fmri,
        domain,         # 'emo'/'soc'
        sub_ls,         # the subject list 
        fname_img,      # the full name of the brain image of interest
        mod = 'func',   # modality of brain image --> fmri\mri ...
        ver = 'v1',     # the version of processed brain image
        space = 'fsLR', # the brain space brain images co-registered to
        headm_method = 'taskFriston',   # the algorithm used to addressing head motion
        return_valid_subs = False,      # whether check brain image of subject exists and 
                                        # return valid subjects list
):
    
    """
    A simple function to read brain images' fullpath into a list
    """
    
    img_ls_valid = []
    sub_ls_valid = []
    for sub in sub_ls:
        file_img_sub = os.path.join(
            dir_data_fmri,
            f'mri_{domain}_fmri_1stlevel_20250822/{ver}',
            f'sub-{sub}',
            f'{mod}',
            f'{domain}_fmri_1stlevel',
            f'sub-{sub}_task-{domain.upper()}_space-{space}_desc-{headm_method}_custom-smooth_first-level',
            f'{fname_img}')
        
        if return_valid_subs:
            # whether brain image file of current subject exists
            if os.path.exists(file_img_sub):
                img_ls_valid.append(file_img_sub)
                sub_ls_valid.append(sub)
        else:
            img_ls_valid.append(file_img_sub)
    
    return img_ls_valid, sub_ls_valid


def group_subjects_by_age(
    subject_ids: Sequence[Union[str, int]],
    ages: Sequence[Union[int, float]],
    age_ranges: Sequence[Tuple[float, float]],
    *,
    include_unassigned: bool = False
) -> Dict[Tuple[float, float], List[Union[str, int]]]:
    """
    Group subjects into age bins using left-closed, right-open intervals.

    Parameters
    ----------
    subject_ids : sequence of str or int
        Subject identifiers.
    
    ages : sequence of int or float
        Age values corresponding to each subject.
    
    age_ranges : sequence of tuple
        Age range list, e.g., [(6, 7), (7, 8), (8, 9)].
        Each tuple is interpreted as [left, right), meaning:
        age >= left and age < right.
    
    include_unassigned : bool, optional
        If True, return an additional key "unassigned" containing subjects
        whose ages do not fall into any provided age range.

    Returns
    -------
    grouped_subjects : dict
        Dictionary where keys are age range tuples and values are lists of
        subject IDs in that age group.
    """

    if len(subject_ids) != len(ages):
        raise ValueError(
            f"`subject_ids` and `ages` must have the same length. "
            f"Got {len(subject_ids)} subject IDs and {len(ages)} ages."
        )

    grouped_subjects = {age_range: [] for age_range in age_ranges}
    unassigned = []

    for subj_id, age in zip(subject_ids, ages):
        if age is None or np.isnan(age):
            unassigned.append(subj_id)
            continue

        assigned = False

        for left, right in age_ranges:
            if left >= right:
                raise ValueError(
                    f"Invalid age range {left, right}: "
                    "left boundary must be smaller than right boundary."
                )

            if left <= age < right:
                grouped_subjects[(left, right)].append(subj_id)
                assigned = True
                break

        if not assigned:
            unassigned.append(subj_id)

    if include_unassigned:
        grouped_subjects["unassigned"] = unassigned

    return grouped_subjects


def find_extremes(arr):
    """find maximal negative and minimal positive value in a given array"""
    negatives = arr[arr < 0]
    positives = arr[arr > 0]
    
    max_negative = np.max(negatives) if len(negatives) > 0 else np.nan
    min_positive = np.min(positives) if len(positives) > 0 else np.nan
    
    return max_negative, min_positive


def _load_one_gifti_vector(gii_file: Union[str, Path]) -> Tuple[np.ndarray, nib.GiftiImage]:
    """
    Load a single .func.gii file and return a 1D data vector.

    Assumption:
        Usually a beta .func.gii contains one data array.
        If multiple data arrays exist, they are concatenated.
    """
    gii_file = Path(gii_file)
    img = nib.load(str(gii_file))

    if not isinstance(img, nib.gifti.gifti.GiftiImage):
        raise TypeError(f"{gii_file} is not a valid GiftiImage.")

    arrays = [np.asarray(darray.data, dtype=np.float64).ravel() for darray in img.darrays]

    if len(arrays) == 0:
        raise ValueError(f"{gii_file} contains no data arrays.")

    data = np.concatenate(arrays)

    return data, img


def _load_surface_data(
    files: Sequence[GiiInput],
    both_hemispheres: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Load surface data.

    If both_hemispheres=True:
        each element of files must be a tuple:
            (left_hemi_file, right_hemi_file)

        Output data shape:
            n_images x (n_left_vertices + n_right_vertices)

    If both_hemispheres=False:
        each element of files is a single .func.gii file.

        Output data shape:
            n_images x n_vertices

    Returns
    -------
    data : np.ndarray
        Shape = images x vertices.

    ref_info : dict
        Reference images and split indices for saving results.
    """
    if len(files) == 0:
        raise ValueError("No .func.gii files were provided.")

    all_data = []

    if both_hemispheres:
        first_left, first_right = files[0]

        left_ref_data, left_ref_img = _load_one_gifti_vector(first_left)
        right_ref_data, right_ref_img = _load_one_gifti_vector(first_right)

        n_left = left_ref_data.shape[0]
        n_right = right_ref_data.shape[0]
        n_total = n_left + n_right

        all_data.append(np.r_[left_ref_data, right_ref_data])

        for i, pair in enumerate(files[1:], start=1):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(
                    "When both_hemispheres=True, each element must be "
                    "(left_hemi_file, right_hemi_file)."
                )

            left_file, right_file = pair
            left_data, _ = _load_one_gifti_vector(left_file)
            right_data, _ = _load_one_gifti_vector(right_file)

            if left_data.shape[0] != n_left:
                raise ValueError(
                    f"Left hemi file at index {i} has {left_data.shape[0]} vertices, "
                    f"but expected {n_left}."
                )

            if right_data.shape[0] != n_right:
                raise ValueError(
                    f"Right hemi file at index {i} has {right_data.shape[0]} vertices, "
                    f"but expected {n_right}."
                )

            all_data.append(np.r_[left_data, right_data])

        ref_info = {
            "both_hemispheres": True,
            "left_ref_img": left_ref_img,
            "right_ref_img": right_ref_img,
            "n_left": n_left,
            "n_right": n_right,
            "n_total": n_total,
        }

    else:
        first_data, ref_img = _load_one_gifti_vector(files[0])
        n_vertices = first_data.shape[0]

        all_data.append(first_data)

        for i, f in enumerate(files[1:], start=1):
            data, _ = _load_one_gifti_vector(f)

            if data.shape[0] != n_vertices:
                raise ValueError(
                    f"File at index {i} has {data.shape[0]} vertices, "
                    f"but expected {n_vertices}."
                )

            all_data.append(data)

        ref_info = {
            "both_hemispheres": False,
            "ref_img": ref_img,
            "n_vertices": n_vertices,
        }

    return np.vstack(all_data), ref_info


def _save_one_gifti_map(
    values: np.ndarray,
    reference_img: nib.GiftiImage,
    output_file: Union[str, Path],
    intent: str = "NIFTI_INTENT_NONE",
) -> Path:
    """
    Save one 1D vector as a .func.gii file.
    """
    output_file = Path(output_file)
    values = np.asarray(values, dtype=np.float32).ravel()

    out_img = nib.GiftiImage()
    out_img.meta = reference_img.meta
    out_img.labeltable = reference_img.labeltable

    darray = nib.gifti.GiftiDataArray(
        data=values,
        intent=intent,
        datatype="NIFTI_TYPE_FLOAT32"
    )

    out_img.add_gifti_data_array(darray)
    nib.save(out_img, str(output_file))

    return output_file


def _save_surface_map(
    values: np.ndarray,
    ref_info: Dict[str, object],
    output_dir: Union[str, Path],
    output_prefix: str,
    map_name: str,
    intent: str,
) -> Dict[str, Path]:
    """
    Save map either as one file or as left/right hemisphere files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    values = np.asarray(values).ravel()

    if ref_info["both_hemispheres"]:
        n_left = ref_info["n_left"]
        n_right = ref_info["n_right"]

        left_values = values[:n_left]
        right_values = values[n_left:n_left + n_right]

        left_file = output_dir / f"{output_prefix}_{map_name}_hemi-L.func.gii"
        right_file = output_dir / f"{output_prefix}_{map_name}_hemi-R.func.gii"

        _save_one_gifti_map(
            left_values,
            ref_info["left_ref_img"],
            left_file,
            intent=intent,
        )

        _save_one_gifti_map(
            right_values,
            ref_info["right_ref_img"],
            right_file,
            intent=intent,
        )

        return {
            "left": left_file,
            "right": right_file,
        }

    else:
        output_file = output_dir / f"{output_prefix}_{map_name}.func.gii"

        _save_one_gifti_map(
            values,
            ref_info["ref_img"],
            output_file,
            intent=intent,
        )

        return {
            "single": output_file,
        }


def second_level_surface_ttest_bihemi(
    group1_files: Sequence[GiiInput],
    group2_files: Sequence[GiiInput],
    design_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    test_type: str = "paired",
    contrast: Optional[np.ndarray] = None,
    output_dir: Union[str, Path] = ".",
    output_prefix: str = "second_level",
    both_hemispheres: bool = True,
    save_maps: bool = True,
    two_sided: bool = True,
) -> Dict[str, object]:
    """
    Vertex-wise second-level GLM/t-test for surface .func.gii data.

    Supports two common input formats:

    1. Bi-hemisphere mode, recommended:
        group1_files = [
            ("sub-01_L_condA.func.gii", "sub-01_R_condA.func.gii"),
            ("sub-02_L_condA.func.gii", "sub-02_R_condA.func.gii"),
        ]

        group2_files = [
            ("sub-01_L_condB.func.gii", "sub-01_R_condB.func.gii"),
            ("sub-02_L_condB.func.gii", "sub-02_R_condB.func.gii"),
        ]

    2. Single-file mode:
        group1_files = ["sub-01_condA.func.gii", ...]
        group2_files = ["sub-01_condB.func.gii", ...]

    Parameters
    ----------
    group1_files, group2_files :
        For paired test, ordered by subject.
        For independent test, group1 and group2 are independent samples.

    design_matrix :
        For paired test:
            rows = subjects/pairs.
            model is fit to group2 - group1.

        For independent test:
            rows = all subjects, ordered as group1 first, group2 second.

    test_type :
        "paired" or "independent".

    contrast :
        Contrast vector. If None:
            paired: test intercept.
            independent: test second column, assumed group effect.

    both_hemispheres :
        If True, each observation is a tuple of left and right .func.gii files.
        If False, each observation is one .func.gii file.

    Returns
    -------
    dict
        Contains t_map, p_map, z_map, effect_map, df, design_matrix, contrast,
        valid_mask, output_files.
    """
    test_type = test_type.lower()

    if test_type not in {"paired", "independent"}:
        raise ValueError("test_type must be either 'paired' or 'independent'.")

    data1, ref_info = _load_surface_data(
        group1_files,
        both_hemispheres=both_hemispheres,
    )

    data2, ref_info2 = _load_surface_data(
        group2_files,
        both_hemispheres=both_hemispheres,
    )

    n1, n_vertices = data1.shape
    n2, n_vertices2 = data2.shape

    if n_vertices != n_vertices2:
        raise ValueError(
            f"group1 and group2 have different total vertices: "
            f"{n_vertices} vs {n_vertices2}."
        )

    if test_type == "paired" and n1 != n2:
        raise ValueError(
            "For paired t-test, group1_files and group2_files must have the same length "
            "and must be ordered by subject pair."
        )

    if both_hemispheres:
        if ref_info["n_left"] != ref_info2["n_left"] or ref_info["n_right"] != ref_info2["n_right"]:
            raise ValueError("group1 and group2 have inconsistent left/right hemisphere vertex counts.")

    # ------------------------------------------------------------------
    # Build dependent variable and second-level design matrix.
    # ------------------------------------------------------------------
    if test_type == "paired":
        y = data2 - data1

        if design_matrix is None:
            X = np.ones((n1, 1), dtype=np.float64)
            design_df = pd.DataFrame(X, columns=["intercept"])
        else:
            if isinstance(design_matrix, pd.DataFrame):
                design_df = design_matrix.copy()
                X = design_df.to_numpy(dtype=np.float64)
            else:
                X = np.asarray(design_matrix, dtype=np.float64)
                design_df = pd.DataFrame(
                    X,
                    columns=[f"regressor_{i}" for i in range(X.shape[1])]
                )

            if X.shape[0] != n1:
                raise ValueError(
                    f"For paired test, design_matrix must have {n1} rows, "
                    f"but got {X.shape[0]}."
                )

            if not np.any(np.all(np.isclose(X, 1.0), axis=0)):
                X = sm.add_constant(X, has_constant="add")
                design_df = pd.DataFrame(
                    X,
                    columns=["intercept"] + list(design_df.columns)
                )

        if contrast is None:
            contrast = np.zeros(X.shape[1])
            contrast[0] = 1.0

    else:
        y = np.vstack([data1, data2])

        if design_matrix is None:
            group_indicator = np.r_[np.zeros(n1), np.ones(n2)]
            X = np.column_stack([
                np.ones(n1 + n2),
                group_indicator,
            ])
            design_df = pd.DataFrame(
                X,
                columns=["intercept", "group2_minus_group1"]
            )
        else:
            if isinstance(design_matrix, pd.DataFrame):
                design_df = design_matrix.copy()
                X = design_df.to_numpy(dtype=np.float64)
            else:
                X = np.asarray(design_matrix, dtype=np.float64)
                design_df = pd.DataFrame(
                    X,
                    columns=[f"regressor_{i}" for i in range(X.shape[1])]
                )

            if X.shape[0] != n1 + n2:
                raise ValueError(
                    f"For independent test, design_matrix must have {n1 + n2} rows, "
                    f"but got {X.shape[0]}."
                )

            if not np.any(np.all(np.isclose(X, 1.0), axis=0)):
                X = sm.add_constant(X, has_constant="add")
                design_df = pd.DataFrame(
                    X,
                    columns=["intercept"] + list(design_df.columns)
                )

        if contrast is None:
            if X.shape[1] < 2:
                raise ValueError(
                    "For independent test, contrast is required if the design matrix "
                    "does not contain a group column."
                )

            contrast = np.zeros(X.shape[1])
            contrast[1] = 1.0

    contrast = np.asarray(contrast, dtype=np.float64).ravel()

    if contrast.shape[0] != X.shape[1]:
        raise ValueError(
            f"Contrast length {contrast.shape[0]} does not match design matrix "
            f"with {X.shape[1]} columns."
        )

    # ------------------------------------------------------------------
    # Efficient vertex-wise OLS GLM.
    # ------------------------------------------------------------------
    valid_mask = np.all(np.isfinite(y), axis=0)

    t_map = np.full(n_vertices, np.nan, dtype=np.float64)
    p_map = np.full(n_vertices, np.nan, dtype=np.float64)
    z_map = np.full(n_vertices, np.nan, dtype=np.float64)
    effect_map = np.full(n_vertices, np.nan, dtype=np.float64)

    y_valid = y[:, valid_mask]

    rank_x = np.linalg.matrix_rank(X)
    df = X.shape[0] - rank_x

    if df <= 0:
        raise ValueError(
            f"Degrees of freedom is {df}. The design matrix is likely over-parameterized."
        )

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta_hat = xtx_inv @ X.T @ y_valid

    fitted = X @ beta_hat
    residuals = y_valid - fitted
    rss = np.sum(residuals ** 2, axis=0)
    sigma2 = rss / df

    contrast_variance_scalar = contrast @ xtx_inv @ contrast.T

    if contrast_variance_scalar <= 0:
        raise ValueError(
            "The contrast has non-positive variance. Check the design matrix and contrast."
        )

    effect_valid = contrast @ beta_hat
    se_valid = np.sqrt(sigma2 * contrast_variance_scalar)
    t_valid = effect_valid / se_valid

    if two_sided:
        p_valid = 2.0 * stats.t.sf(np.abs(t_valid), df=df)
        z_valid = stats.norm.isf(p_valid / 2.0) * np.sign(t_valid)
    else:
        p_valid = stats.t.sf(t_valid, df=df)
        z_valid = stats.norm.isf(p_valid) * np.sign(t_valid)

    t_map[valid_mask] = t_valid
    p_map[valid_mask] = p_valid
    z_map[valid_mask] = z_valid
    effect_map[valid_mask] = effect_valid

    output_files = {}

    if save_maps:
        output_files["t_map"] = _save_surface_map(
            t_map,
            ref_info,
            output_dir,
            output_prefix,
            "tmap",
            intent="NIFTI_INTENT_TTEST",
        )

        output_files["p_map"] = _save_surface_map(
            p_map,
            ref_info,
            output_dir,
            output_prefix,
            "pmap",
            intent="NIFTI_INTENT_PVAL",
        )

        output_files["z_map"] = _save_surface_map(
            z_map,
            ref_info,
            output_dir,
            output_prefix,
            "zmap",
            intent="NIFTI_INTENT_ZSCORE",
        )

        output_files["effect_map"] = _save_surface_map(
            effect_map,
            ref_info,
            output_dir,
            output_prefix,
            "effect",
            intent="NIFTI_INTENT_ESTIMATE",
        )

    results = {
        "t_map": t_map,
        "p_map": p_map,
        "z_map": z_map,
        "effect_map": effect_map,
        "df": df,
        "design_matrix": design_df,
        "contrast": contrast,
        "valid_mask": valid_mask,
        "output_files": output_files,
        "test_type": test_type,
        "n_group1": n1,
        "n_group2": n2,
        "both_hemispheres": both_hemispheres,
        "ref_info": ref_info,
    }

    return results


def make_contrast_from_column(design_df, column_name):
    """
    Create a contrast vector that tests one column of the final design matrix.

    Parameters
    ----------
    design_df : pandas.DataFrame
        The final design matrix returned by results["design_matrix"].
    column_name : str
        Column to test.

    Returns
    -------
    contrast : np.ndarray
        Contrast vector.
    """
    if column_name not in design_df.columns:
        raise ValueError(
            f"{column_name} is not in design matrix. "
            f"Available columns are: {list(design_df.columns)}"
        )

    contrast = np.zeros(len(design_df.columns), dtype=float)
    contrast[design_df.columns.get_loc(column_name)] = 1.0

    return contrast


def _signed_z_from_p_and_t(
    p_values: np.ndarray,
    t_values: np.ndarray,
    two_sided: bool = True,
) -> np.ndarray:
    """
    Convert corrected p-values to signed z-values.

    For two-sided tests:
        z = sign(t) * norm.isf(p / 2)

    For one-sided tests:
        z = sign(t) * norm.isf(p)
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    t_values = np.asarray(t_values, dtype=np.float64)

    z_values = np.full_like(p_values, np.nan, dtype=np.float64)

    valid = np.isfinite(p_values) & np.isfinite(t_values)
    p_safe = np.clip(p_values[valid], np.finfo(float).tiny, 1.0)

    if two_sided:
        z_values[valid] = stats.norm.isf(p_safe / 2.0) * np.sign(t_values[valid])
    else:
        z_values[valid] = stats.norm.isf(p_safe) * np.sign(t_values[valid])

    return z_values


def _compute_p_from_t(
    t_map: np.ndarray,
    df: float,
    two_sided: bool = True,
) -> np.ndarray:
    """
    Compute p-values from a t-map.
    """
    t_map = np.asarray(t_map, dtype=np.float64)
    p_map = np.full_like(t_map, np.nan, dtype=np.float64)

    valid = np.isfinite(t_map)

    if two_sided:
        p_map[valid] = 2.0 * stats.t.sf(np.abs(t_map[valid]), df=df)
    else:
        p_map[valid] = stats.t.sf(t_map[valid], df=df)

    return p_map


def t_to_z_array(
    t_values: ArrayLike,
    df: float,
    two_tailed: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Convert t-values to signed z-values.

    Parameters
    ----------
    t_values : array-like
        Input t-value array.

    df : float
        Degrees of freedom of the t statistic.

    two_tailed : bool, default=True
        If True, convert using two-tailed p-values.
        If False, convert using one-tailed p-values while preserving sign.

    dtype : np.dtype, default=np.float32
        Output data type.

    Returns
    -------
    z_values : np.ndarray
        Signed z-value array.
    """

    if df <= 0:
        raise ValueError("`df` must be positive.")

    t_values = np.asarray(t_values, dtype=np.float64)
    z_values = np.full_like(t_values, np.nan, dtype=np.float64)

    valid = np.isfinite(t_values)

    if not np.any(valid):
        return z_values.astype(dtype)

    abs_t = np.abs(t_values[valid])

    if two_tailed:
        p = 2.0 * t_dist.sf(abs_t, df)
        z_abs = stats.norm.isf(p / 2.0)
    else:
        p = t_dist.sf(abs_t, df)
        z_abs = stats.norm.isf(p)

    z_values[valid] = np.sign(t_values[valid]) * z_abs

    return z_values.astype(dtype)


def t_to_z_image(
    img: ImageLike,
    df: float,
    output_file: Optional[Union[str, Path]] = None,
    two_tailed: bool = True,
    dtype: np.dtype = np.float32,
):
    """
    Convert a t-value image to a signed z-value image.

    Supported input formats
    -----------------------
    1. numpy array
    2. NIfTI image or NIfTI file path: .nii, .nii.gz
    3. GIFTI image or GIFTI file path: .gii, .func.gii, .shape.gii
    4. CIFTI image or CIFTI file path: .nii with CIFTI intent, e.g. .dtseries.nii, .dscalar.nii

    Parameters
    ----------
    img : str, Path, np.ndarray, or nibabel image
        Input t-value image or array.

    df : float
        Degrees of freedom of the t statistic.
        For a one-sample second-level t test, df is usually n - 1.
        For a two-sample t test, df is usually n1 + n2 - 2 unless Welch correction is used.

    output_file : str or Path, optional
        If provided, save the z-value image to this path.

    two_tailed : bool, default=True
        If True, use two-tailed t-to-z conversion.
        If False, use one-tailed conversion while preserving sign.

    dtype : np.dtype, default=np.float32
        Output data type.

    Returns
    -------
    z_img_or_array
        If input is array, returns a z-value array.
        If input is image or file path, returns a nibabel image object.
    """

    input_is_path = isinstance(img, (str, Path))

    if input_is_path:
        img = nib.load(str(img))

    # Case 1: numpy array
    if isinstance(img, np.ndarray):
        z_data = t_to_z_array(
            img,
            df=df,
            two_tailed=two_tailed,
            dtype=dtype,
        )

        if output_file is not None:
            np.save(str(output_file), z_data)

        return z_data

    # Case 2: NIfTI image
    if isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)):
        data = img.get_fdata(dtype=np.float64)

        z_data = t_to_z_array(
            data,
            df=df,
            two_tailed=two_tailed,
            dtype=dtype,
        )

        header = img.header.copy()
        header.set_data_dtype(dtype)

        z_img = img.__class__(
            z_data,
            affine=img.affine,
            header=header,
        )

        if output_file is not None:
            nib.save(z_img, str(output_file))

        return z_img

    # Case 3: GIFTI image
    if isinstance(img, nib.GiftiImage):
        z_darrays = []

        for darray in img.darrays:
            data = np.asarray(darray.data, dtype=np.float64)

            z_data = t_to_z_array(
                data,
                df=df,
                two_tailed=two_tailed,
                dtype=dtype,
            )

            new_darray = nib.gifti.GiftiDataArray(
                data=z_data.astype(dtype),
                intent=darray.intent,
                datatype=nib.nifti1.data_type_codes.code[np.dtype(dtype)],
                encoding=darray.encoding,
                endian=darray.endian,
                coordsys=darray.coordsys,
                meta=darray.meta,
            )
            z_darrays.append(new_darray)

        z_img = nib.GiftiImage(
            header=img.header,
            extra=img.extra,
            file_map=img.file_map,
            meta=img.meta,
            labeltable=img.labeltable,
            darrays=z_darrays,
        )

        if output_file is not None:
            nib.save(z_img, str(output_file))

        return z_img

    # Case 4: CIFTI image
    if isinstance(img, nib.Cifti2Image):
        data = np.asarray(img.get_fdata(dtype=np.float64))

        z_data = t_to_z_array(
            data,
            df=df,
            two_tailed=two_tailed,
            dtype=dtype,
        )

        z_img = nib.Cifti2Image(
            dataobj=z_data.astype(dtype),
            header=img.header,
            nifti_header=img.nifti_header,
            file_map=img.file_map,
        )

        if output_file is not None:
            nib.save(z_img, str(output_file))

        return z_img

    raise TypeError(
        "Unsupported input type. "
        "Input must be a numpy array, file path, NIfTI image, GIFTI image, or CIFTI image."
    )
    
    
def _find_surface_clusters(
    supra_threshold_mask: np.ndarray,
    adjacency: sparse.spmatrix,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find connected clusters on a surface.

    Parameters
    ----------
    supra_threshold_mask : np.ndarray
        Boolean array, shape = n_vertices.
        True means the vertex is supra-threshold.

    adjacency : scipy sparse matrix
        Surface adjacency matrix, shape = n_vertices x n_vertices.

    Returns
    -------
    cluster_labels : np.ndarray
        Integer labels, shape = n_vertices.
        0 means not in any cluster.
        1, 2, ... are cluster labels.

    cluster_sizes : np.ndarray
        Cluster sizes in number of vertices.
        Length = n_clusters.
    """
    supra_threshold_mask = np.asarray(supra_threshold_mask, dtype=bool)

    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix.")

    if adjacency.shape[0] != supra_threshold_mask.shape[0]:
        raise ValueError(
            f"adjacency has {adjacency.shape[0]} vertices, "
            f"but mask has {supra_threshold_mask.shape[0]} vertices."
        )

    cluster_labels = np.zeros(supra_threshold_mask.shape[0], dtype=np.int32)

    supra_indices = np.where(supra_threshold_mask)[0]

    if supra_indices.size == 0:
        return cluster_labels, np.array([], dtype=np.int32)

    sub_adj = adjacency[supra_indices, :][:, supra_indices]

    n_clusters, labels_sub = connected_components(
        sub_adj,
        directed=False,
        return_labels=True,
    )

    cluster_sizes = np.bincount(labels_sub, minlength=n_clusters).astype(np.int32)

    for k in range(n_clusters):
        cluster_labels[supra_indices[labels_sub == k]] = k + 1

    return cluster_labels, cluster_sizes


def correct_surface_statistics(
    results: Optional[Dict[str, object]] = None,
    t_map: Optional[np.ndarray] = None,
    p_map: Optional[np.ndarray] = None,
    df: Optional[float] = None,
    ref_info: Optional[Dict[str, object]] = None,
    valid_mask: Optional[np.ndarray] = None,
    method: str = "fdr_bh",
    alpha: float = 0.05,
    two_sided: bool = True,
    output_dir: Union[str, Path] = ".",
    output_prefix: str = "second_level_corrected",
    save_maps: bool = True,
    null_t_maps: Optional[np.ndarray] = None,
    adjacency: Optional[sparse.spmatrix] = None,
    cluster_forming_p: float = 0.001,
) -> Dict[str, object]:
    """
    Correct vertex-wise surface statistical maps for multiple comparisons.

    This function is designed to work with the output of
    second_level_surface_ttest_bihemi(), but it can also be used directly
    with t_map / p_map / ref_info.

    Supported correction methods
    ----------------------------
    Vertex-wise p-value correction:
        "fdr_bh"       : Benjamini-Hochberg FDR
        "fdr_by"       : Benjamini-Yekutieli FDR
        "bonferroni"   : Bonferroni FWE
        "sidak"        : Sidak FWE
        "holm"         : Holm-Bonferroni FWE
        "holm-sidak"   : Holm-Sidak FWE

    Permutation-based correction:
        "max_t"        : vertex-wise max-T FWE correction
                         requires null_t_maps

        "cluster_size" : cluster-extent FWE correction
                         requires null_t_maps and adjacency

    Parameters
    ----------
    results : dict, optional
        Output from second_level_surface_ttest_bihemi().
        If provided, the function will read:
            results["t_map"]
            results["p_map"]
            results["df"]
            results["ref_info"]
            results["valid_mask"]

    t_map : np.ndarray, optional
        Observed t-map, shape = n_vertices.

    p_map : np.ndarray, optional
        Observed uncorrected p-map, shape = n_vertices.
        If None, t_map and df are required to compute p-values.

    df : float, optional
        Degrees of freedom. Required if p_map is None or if cluster threshold
        must be computed from t-distribution.

    ref_info : dict, optional
        Reference information returned by the bi-hemisphere loader.
        Required if save_maps=True.

    valid_mask : np.ndarray, optional
        Boolean mask defining vertices included in correction.
        If None, finite p-values and finite t-values are used.

    method : str
        Correction method.

    alpha : float
        Significance level.

    two_sided : bool
        Whether the original test is two-sided.

    output_dir : str or Path
        Output directory.

    output_prefix : str
        Prefix for saved files.

    save_maps : bool
        Whether to save corrected maps as .func.gii.

    null_t_maps : np.ndarray, optional
        Null t maps for permutation-based correction.
        Shape = n_permutations x n_vertices.

        For "max_t":
            corrected p at each vertex is:
                P(max |T_null| >= |T_observed|)

        For "cluster_size":
            null maps are thresholded with the same cluster-forming threshold,
            and the maximum cluster size per permutation is used as the null
            distribution.

    adjacency : scipy sparse matrix, optional
        Surface adjacency matrix for both hemispheres concatenated together.
        Required for "cluster_size".

    cluster_forming_p : float
        Cluster-forming threshold for cluster-size FWE correction.
        Common values are 0.001 or 0.01.

    Returns
    -------
    corrected_results : dict
        Contains:
            method
            alpha
            p_uncorrected_map
            p_corrected_map
            q_map
            z_corrected_map
            reject_map
            threshold
            output_files

        For cluster_size:
            also contains:
                cluster_labels
                cluster_sizes
                cluster_p_values
                cluster_p_corrected_map
    """

    if results is not None:
        if t_map is None and "t_map" in results:
            t_map = results["t_map"]

        if p_map is None and "p_map" in results:
            p_map = results["p_map"]

        if df is None and "df" in results:
            df = results["df"]

        if ref_info is None and "ref_info" in results:
            ref_info = results["ref_info"]

        if valid_mask is None and "valid_mask" in results:
            valid_mask = results["valid_mask"]

    if t_map is None and p_map is None:
        raise ValueError("At least one of t_map or p_map must be provided.")

    if t_map is not None:
        t_map = np.asarray(t_map, dtype=np.float64).ravel()

    if p_map is not None:
        p_map = np.asarray(p_map, dtype=np.float64).ravel()

    if p_map is None:
        if t_map is None or df is None:
            raise ValueError("If p_map is None, both t_map and df are required.")
        p_map = _compute_p_from_t(t_map, df=df, two_sided=two_sided)

    if t_map is None:
        t_map = np.full_like(p_map, np.nan, dtype=np.float64)

    if t_map.shape[0] != p_map.shape[0]:
        raise ValueError(
            f"t_map and p_map have different sizes: "
            f"{t_map.shape[0]} vs {p_map.shape[0]}."
        )

    n_vertices = p_map.shape[0]

    if valid_mask is None:
        valid_mask = np.isfinite(p_map)
        if t_map is not None:
            valid_mask = valid_mask & np.isfinite(t_map)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool).ravel()

    if valid_mask.shape[0] != n_vertices:
        raise ValueError(
            f"valid_mask has {valid_mask.shape[0]} vertices, "
            f"but p_map has {n_vertices} vertices."
        )

    method = method.lower()

    allowed_methods = {
        "fdr_bh",
        "fdr_by",
        "bonferroni",
        "sidak",
        "holm",
        "holm-sidak",
        "max_t",
        "cluster_size",
    }

    if method not in allowed_methods:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Available methods are: {sorted(allowed_methods)}"
        )

    p_corrected_map = np.full(n_vertices, np.nan, dtype=np.float64)
    q_map = np.full(n_vertices, np.nan, dtype=np.float64)
    reject_map = np.zeros(n_vertices, dtype=bool)
    z_corrected_map = np.full(n_vertices, np.nan, dtype=np.float64)

    threshold = alpha

    correction_mask = (
    valid_mask
    & np.isfinite(p_map)
    & (p_map >= 0.0)
    & (p_map <= 1.0)
    )
    
    if t_map is not None:
        correction_mask = correction_mask & np.isfinite(t_map)
    
    n_tested = int(np.sum(correction_mask))
    
    if n_tested == 0:
        raise ValueError(
            "No valid p-values are available for multiple-comparison correction. "
            "Check whether p_map contains NaN/inf values."
        )
    
    p_valid = p_map[correction_mask]
    
    if np.any(~np.isfinite(p_valid)):
        raise ValueError("p_valid still contains NaN or inf values.")
    
    if np.any((p_valid < 0.0) | (p_valid > 1.0)):
        raise ValueError("p_valid contains values outside [0, 1].")

    # ------------------------------------------------------------------
    # 1. Standard vertex-wise p-value corrections
    # ------------------------------------------------------------------
    if method in {
    "fdr_bh",
    "fdr_by",
    "bonferroni",
    "sidak",
    "holm",
    "holm-sidak",
    }:
        reject_valid, p_corr_valid, _, _ = multipletests(
            p_valid,
            alpha=alpha,
            method=method,
        )
    
        p_corrected_map[correction_mask] = p_corr_valid
        q_map[correction_mask] = p_corr_valid
        reject_map[correction_mask] = reject_valid
    
        z_corrected_map = _signed_z_from_p_and_t(
            p_corrected_map,
            t_map,
            two_sided=two_sided,
        )

    # ------------------------------------------------------------------
    # 2. Permutation max-T FWE correction
    # ------------------------------------------------------------------
    elif method == "max_t":
        if null_t_maps is None:
            raise ValueError("method='max_t' requires null_t_maps.")

        null_t_maps = np.asarray(null_t_maps, dtype=np.float64)

        if null_t_maps.ndim != 2:
            raise ValueError("null_t_maps must have shape n_permutations x n_vertices.")

        if null_t_maps.shape[1] != n_vertices:
            raise ValueError(
                f"null_t_maps has {null_t_maps.shape[1]} vertices, "
                f"but observed map has {n_vertices} vertices."
            )

        if two_sided:
            null_max_stat = np.nanmax(np.abs(null_t_maps[:, valid_mask]), axis=1)
            observed_stat = np.abs(t_map[valid_mask])
        else:
            null_max_stat = np.nanmax(null_t_maps[:, valid_mask], axis=1)
            observed_stat = t_map[valid_mask]

        n_perm = null_max_stat.shape[0]

        p_corr_valid = np.array([
            (np.sum(null_max_stat >= obs) + 1.0) / (n_perm + 1.0)
            for obs in observed_stat
        ])

        reject_valid = p_corr_valid <= alpha

        p_corrected_map[valid_mask] = p_corr_valid
        q_map[valid_mask] = p_corr_valid
        reject_map[valid_mask] = reject_valid

        z_corrected_map = _signed_z_from_p_and_t(
            p_corrected_map,
            t_map,
            two_sided=two_sided,
        )

    # ------------------------------------------------------------------
    # 3. Cluster-size FWE correction
    # ------------------------------------------------------------------
    elif method == "cluster_size":
        if null_t_maps is None:
            raise ValueError("method='cluster_size' requires null_t_maps.")

        if adjacency is None:
            raise ValueError("method='cluster_size' requires adjacency.")

        if df is None:
            raise ValueError("method='cluster_size' requires df.")

        null_t_maps = np.asarray(null_t_maps, dtype=np.float64)

        if null_t_maps.ndim != 2:
            raise ValueError("null_t_maps must have shape n_permutations x n_vertices.")

        if null_t_maps.shape[1] != n_vertices:
            raise ValueError(
                f"null_t_maps has {null_t_maps.shape[1]} vertices, "
                f"but observed map has {n_vertices} vertices."
            )

        if adjacency.shape[0] != n_vertices or adjacency.shape[1] != n_vertices:
            raise ValueError(
                f"adjacency must have shape {n_vertices} x {n_vertices}."
            )

        if two_sided:
            t_cluster_threshold = stats.t.isf(cluster_forming_p / 2.0, df=df)
            observed_supra = valid_mask & (np.abs(t_map) >= t_cluster_threshold)
        else:
            t_cluster_threshold = stats.t.isf(cluster_forming_p, df=df)
            observed_supra = valid_mask & (t_map >= t_cluster_threshold)

        cluster_labels, cluster_sizes = _find_surface_clusters(
            observed_supra,
            adjacency,
        )

        n_perm = null_t_maps.shape[0]
        null_max_cluster_sizes = np.zeros(n_perm, dtype=np.float64)

        for i in range(n_perm):
            null_t = null_t_maps[i]

            if two_sided:
                null_supra = valid_mask & (np.abs(null_t) >= t_cluster_threshold)
            else:
                null_supra = valid_mask & (null_t >= t_cluster_threshold)

            _, null_cluster_sizes = _find_surface_clusters(
                null_supra,
                adjacency,
            )

            if null_cluster_sizes.size > 0:
                null_max_cluster_sizes[i] = np.max(null_cluster_sizes)
            else:
                null_max_cluster_sizes[i] = 0.0

        cluster_p_values = np.ones(cluster_sizes.shape[0], dtype=np.float64)
        cluster_p_corrected_map = np.full(n_vertices, np.nan, dtype=np.float64)

        for k, size in enumerate(cluster_sizes, start=1):
            cluster_p = (
                np.sum(null_max_cluster_sizes >= size) + 1.0
            ) / (
                n_perm + 1.0
            )

            cluster_p_values[k - 1] = cluster_p

            in_cluster = cluster_labels == k
            cluster_p_corrected_map[in_cluster] = cluster_p

            if cluster_p <= alpha:
                reject_map[in_cluster] = True

        p_corrected_map = cluster_p_corrected_map.copy()
        q_map = cluster_p_corrected_map.copy()

        z_corrected_map = _signed_z_from_p_and_t(
            p_corrected_map,
            t_map,
            two_sided=two_sided,
        )

        threshold = {
            "cluster_forming_p": cluster_forming_p,
            "cluster_forming_t": t_cluster_threshold,
            "alpha_cluster_fwe": alpha,
        }

    output_files = {}

    if save_maps:
        if ref_info is None:
            raise ValueError("ref_info is required when save_maps=True.")

        output_files["p_corrected_map"] = _save_surface_map(
            p_corrected_map,
            ref_info,
            output_dir,
            output_prefix,
            f"{method}_pcorr",
            intent="NIFTI_INTENT_PVAL",
        )

        output_files["q_map"] = _save_surface_map(
            q_map,
            ref_info,
            output_dir,
            output_prefix,
            f"{method}_qmap",
            intent="NIFTI_INTENT_PVAL",
        )

        output_files["z_corrected_map"] = _save_surface_map(
            z_corrected_map,
            ref_info,
            output_dir,
            output_prefix,
            f"{method}_zcorr",
            intent="NIFTI_INTENT_ZSCORE",
        )

        output_files["reject_map"] = _save_surface_map(
            reject_map.astype(np.float32),
            ref_info,
            output_dir,
            output_prefix,
            f"{method}_reject_alpha-{alpha:g}",
            intent="NIFTI_INTENT_NONE",
        )

        if method == "cluster_size":
            output_files["cluster_labels"] = _save_surface_map(
                cluster_labels.astype(np.float32),
                ref_info,
                output_dir,
                output_prefix,
                f"{method}_cluster_labels",
                intent="NIFTI_INTENT_LABEL",
            )

    corrected_results = {
        "method": method,
        "alpha": alpha,
        "p_uncorrected_map": p_map,
        "p_corrected_map": p_corrected_map,
        "q_map": q_map,
        "z_corrected_map": z_corrected_map,
        "reject_map": reject_map,
        "threshold": threshold,
        "valid_mask": valid_mask,
        "output_files": output_files,
    }

    if method == "cluster_size":
        corrected_results.update({
            "cluster_labels": cluster_labels,
            "cluster_sizes": cluster_sizes,
            "cluster_p_values": cluster_p_values,
            "cluster_p_corrected_map": cluster_p_corrected_map,
            "cluster_forming_p": cluster_forming_p,
            "null_max_cluster_sizes": null_max_cluster_sizes,
        })

    if method == "max_t":
        corrected_results.update({
            "null_max_t": null_max_stat,
        })

    return corrected_results

if __name__ == "__main__":
    # Figure 3 surface and ROI analyses use the study participant maps.
    #%% DATA DIR AND SUBJECT INFO
    # ---------------------------------------------------------------------------------------
    # data directories
    dir_data_fmri = '/public/home/dingrui/BIDS_DATA'
    dir_data_behv = '/public/home/dingrui/fmri_analysis/data/beh'
    dir_data_save = '/public/home/dingrui/fmri_analysis/data/fmri'
    dir_data_fmri_surf = '/public/home/dingrui/fmri_analysis/data/fmri/res_1stlevel_surf'
    
    # sub info
    sub_info = pd.read_csv(
        os.path.join(dir_data_behv, 'participants_in_tfmri_demographics.csv'),
        sep=',', index_col=False
    )
    
    # qualified subject lists
    sub_ls_ER = np.loadtxt(os.path.join(dir_data_behv, 'sub_list_valid_4_ER.txt'), dtype=str)
    sub_ls_TG = np.loadtxt(os.path.join(dir_data_behv, 'sub_list_valid_4_TG.txt'), dtype=str)
    sub_ls_surf = lsdir(dir_data_fmri_surf)
    sub_ls_surf_ER = list(set(sub_ls_surf) & set(sub_ls_ER.tolist()))
    sub_ls_rm_ER = np.loadtxt(
        os.path.join(
            '/public/home/dingrui/fmri_analysis/data/fmri', 
            'subs_removed_from_2ndlevel.txt'), 
        dtype=str).tolist()
    
    # qc file
    df_qc = pd.read_csv(
        os.path.join(
            '/public/home/dingrui/fmri_analysis/data/fmri/QC_manul_fmri', 
            'qc_emo_OR_soc_(subs_meanFD)_meanFD-0.2mm.txt'), 
        sep='\t')
    
    # subjects finally into stats
    sub_ls_test_ER = [sub for sub in sub_ls_surf_ER if sub not in sub_ls_rm_ER]
    # sub_ls_test_ER = list(set(sub_ls_test_ER) & set(list(df_qc.sub_id)))
    
    domain_ls = ['emo', 'soc']
    
    #%% PREPARE DATA INPUT
    # ------------------------------------------------------
    # surf images of experimental conditions of interests
    # ------------------------------------------------------
    
    # use 'soc' or 'emo' or 'emo+soc'
    use_img = 'emo+soc'      
    
    fname_Limg_rpsl = ['beta_0001_hemi-L.func.gii', 'reg_neg_Observe_beta_lh.func.gii']
    fname_Rimg_rpsl = ['beta_0001_hemi-R.func.gii', 'reg_neg_Observe_beta_rh.func.gii']
    fname_Limg_lkng = ['beta_0002_hemi-L.func.gii', 'pas_neg_Observe_beta_lh.func.gii']
    fname_Rimg_lkng = ['beta_0002_hemi-R.func.gii', 'pas_neg_Observe_beta_rh.func.gii']
    fname_Limg_lknt = ['beta_0003_hemi-L.func.gii', 'pas_neu_Observe_beta_lh.func.gii']
    fname_Rimg_lknt = ['beta_0003_hemi-R.func.gii', 'pas_neu_Observe_beta_rh.func.gii']
    
    if use_img in ['soc', 'emo']:
        gii_L_ls_rpsl, _ = read_imgs_from_bids(dir_data_fmri, use_img, sub_ls_test_ER, fname_Limg_rpsl[0])
        gii_R_ls_rpsl, _ = read_imgs_from_bids(dir_data_fmri, use_img, sub_ls_test_ER, fname_Rimg_rpsl[0])
        gii_L_ls_lkng, _ = read_imgs_from_bids(dir_data_fmri, use_img, sub_ls_test_ER, fname_Limg_lkng[0])
        gii_R_ls_lkng, _ = read_imgs_from_bids(dir_data_fmri, use_img, sub_ls_test_ER, fname_Rimg_lkng[0])
        gii_L_ls_lknt, _ = read_imgs_from_bids(dir_data_fmri, use_img, sub_ls_test_ER, fname_Limg_lknt[0])
        gii_R_ls_lknt, _ = read_imgs_from_bids(dir_data_fmri, use_img, sub_ls_test_ER, fname_Rimg_lknt[0])
    else:
        gii_L_ls_rpsl = [os.path.join(dir_data_fmri_surf, sub, fname_Limg_rpsl[1]) for sub in sub_ls_test_ER]
        gii_R_ls_rpsl = [os.path.join(dir_data_fmri_surf, sub, fname_Rimg_rpsl[1]) for sub in sub_ls_test_ER]
        gii_L_ls_lkng = [os.path.join(dir_data_fmri_surf, sub, fname_Limg_lkng[1]) for sub in sub_ls_test_ER]
        gii_R_ls_lkng = [os.path.join(dir_data_fmri_surf, sub, fname_Rimg_lkng[1]) for sub in sub_ls_test_ER]
        gii_L_ls_lknt = [os.path.join(dir_data_fmri_surf, sub, fname_Limg_lknt[1]) for sub in sub_ls_test_ER]
        gii_R_ls_lknt = [os.path.join(dir_data_fmri_surf, sub, fname_Rimg_lknt[1]) for sub in sub_ls_test_ER]
    
    # parcellation image to parcellate surf images of brain activity
    lh_parc_fs = '/public/home/dingrui/tools/masks_atlas/lh.Schaefer2018_200Parcels_7Networks_order.annot'
    rh_parc_fs = '/public/home/dingrui/tools/masks_atlas/rh.Schaefer2018_200Parcels_7Networks_order.annot'
    parc_fsLR  = '/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_order.dscalar.nii'
    
    # -----------------------------
    # important covariates into X
    # -----------------------------
    
    # subs_info of the selected subjects:
    sub_idx = sub_info['sub_id'].isin(sub_ls_test_ER)
    df_sub_age = sub_info[['sub_id', 'age', 'gender', 'site_id']].loc[sub_idx]
    
    subs_age = []
    for sub in sub_ls_test_ER:
        sub_age = df_sub_age['age'][df_sub_age['sub_id']==sub].values[0]
        subs_age.append(sub_age)
    
    # grouping subjects into several subgroups/bins:
    age_ranges = [
        (6,8),   (8,9),   (9,10),  (10,11), (11,12), (12,13), 
        (13,14), (14,15), (15,16), (16,17), (17,18), (18,19),
    ]
    
    age_bin = group_subjects_by_age(sub_ls_test_ER, subs_age, age_ranges)
    age_bin[(6,19)] = sub_ls_test_ER
    
    # generate 2nd level design matrix for each age group/bin:
    dict_X = {}
    for age_range, subs_bin in age_bin.items():
        subs_age_bin = []
        subs_sex = []
        subs_site = []
        # subs_meanFD = []
        
        for sub in subs_bin:
            sub_age = df_sub_age['age'][df_sub_age['sub_id']==sub].values[0]
            sub_sex = df_sub_age['gender'][df_sub_age['sub_id']==sub].values[0]
            sub_sit = df_sub_age['site_id'][df_sub_age['sub_id']==sub].values[0]
            sub_mFD = df_qc['mean_FD'][df_qc['sub_id']==sub].values
            
            subs_age_bin.append(sub_age)
            subs_sex.append(sub_sex)
            subs_site.append(sub_sit)
            # subs_meanFD.append(sub_mFD)
        
        age_z  = (np.array(subs_age_bin) - np.mean(subs_age_bin))/np.std(subs_age_bin)
        gender = np.array(subs_sex)
        sites  = np.array(subs_site)
        # meanFD = np.array(subs_meanFD)
        
        X = pd.DataFrame({
            'intercept': 1.0,
            'age': subs_age_bin,
            'age_z': age_z,
            'gender': gender,
            # 'meanFD': meanFD[:, 0],
            'site': sites,
            })
        
        dict_X[age_range] = X
    
    #%% GLM ON EFFECT OF INTERESTS (VERTEX LV)
    #
    cond_rpsl_ls = []
    cond_lkng_ls = []
    
    for i in range(len(gii_L_ls_rpsl)):
        
        rpsl_in_sub = (gii_L_ls_rpsl[i], gii_R_ls_rpsl[i])
        lkng_in_sub = (gii_L_ls_lkng[i], gii_R_ls_lkng[i])
        
        cond_rpsl_ls.append(rpsl_in_sub)
        cond_lkng_ls.append(lkng_in_sub)
        
    #
    results = second_level_surface_ttest_bihemi(
        group1_files=cond_lkng_ls,
        group2_files=cond_rpsl_ls,
        design_matrix=X,
        test_type="paired",
        contrast=np.array([0, 1, 0, 0, 0]),
        save_maps=False,
        output_dir=None,
        output_prefix='2ndlv',
        both_hemispheres=True,
    )
    
    res_corr = correct_surface_statistics(
        results=results,
        method="fdr_bh",
        alpha=0.05,
        save_maps=False,
        output_dir=None,
        output_prefix=None,
    )
    
    #%% GLM ON EFFECT OF INTERESTS (PARC/ROI LV)
    
    # NOTE: glm performed at each age group/bin
    
    # results collector
    res_rpsl_lkng = {}
    res_lkng_lknt = {}
    
    # glm analysis
    for age_range, X in dict_X.items():
        print(f'\n GLM analysis for age group/bin: {age_range}, n_subs={len(age_bin[age_range])}')
        # brain images list of current age group/bin
        #--------------------------------------------
        # little helper
        def select_imgs(target_subIDs, source_imgs):
            imgs_sel = []
            for sub_id in target_subIDs:
                for gii_img in source_imgs:
                    if sub_id in gii_img:
                        imgs_sel.append(gii_img)
            return imgs_sel
        #--------------------------------------------
        gii_L_rpsl_bin = select_imgs(age_bin[age_range], gii_L_ls_rpsl)
        gii_R_rpsl_bin = select_imgs(age_bin[age_range], gii_R_ls_rpsl)
        gii_L_lkng_bin = select_imgs(age_bin[age_range], gii_L_ls_lkng)
        gii_R_lkng_bin = select_imgs(age_bin[age_range], gii_R_ls_lkng)
        gii_L_lknt_bin = select_imgs(age_bin[age_range], gii_L_ls_lknt)
        gii_R_lknt_bin = select_imgs(age_bin[age_range], gii_R_ls_lknt)
        
        # contrasts vector of current age group/bin
        contrasts = {
            "adj_mean_Delta":  [1.0] + [0.0]*(X.shape[1]-1),
            "age_on_Delta":    [0.0, 1.0] + [0.0]*(X.shape[1]-2),
            "gender_on_Delta": [0.0, 0.0, 1.0] + [0.0]*(X.shape[1]-3)
        }
        
        print('\n\t fitting GLM ...')
        # fit glm model
        res_eff_rpsl_lkng = surf.glm_AmB_surf_roi(
            A_lh_paths=gii_L_rpsl_bin, 
            A_rh_paths=gii_R_rpsl_bin,
            B_lh_paths=gii_L_lkng_bin, 
            B_rh_paths=gii_R_lkng_bin,
            X=X.values,
            contrasts=contrasts,
            parc_lh_path=parc_fsLR,
            parc_rh_path=parc_fsLR,
            save_dir=None, 
        )
        
        res_eff_lkng_lknt = surf.glm_AmB_surf_roi(
            A_lh_paths=gii_L_lkng_bin, 
            A_rh_paths=gii_R_lkng_bin,
            B_lh_paths=gii_L_lknt_bin, 
            B_rh_paths=gii_R_lknt_bin,
            X=X.values,
            contrasts=contrasts,
            parc_lh_path=parc_fsLR,
            parc_rh_path=parc_fsLR,
            save_dir=None, 
        )
        
        print('\t multiple comparison correction ...')
        # multiple comparisons correction
        for con in ['age_on_Delta', 'adj_mean_Delta']:
            res_eff_rpsl_lkng = corr.multi_compare_correction_flex(
                results=res_eff_rpsl_lkng,
                p_key=f"{con}",                           # prefix for outputs
                p_path=None,                              # not used: no p in results
                t_path=("contrasts", f"{con}", "t_roi"),
                dof_path="dof_per_roi",                   # or ("dof_per_roi",) or another path
                methods=("fdr_bh",),
                alpha=0.05,
                two_sided=True,
                in_place=True,
            )
            
            res_eff_lkng_lknt = corr.multi_compare_correction_flex(
                results=res_eff_lkng_lknt,
                p_key=f"{con}",                           # prefix for outputs
                p_path=None,                              # not used: no p in results
                t_path=("contrasts", f"{con}", "t_roi"),
                dof_path="dof_per_roi",                   # or ("dof_per_roi",) or another path
                methods=("fdr_bh",),
                alpha=0.05,
                two_sided=True,
                in_place=True,
            )
        
        print('\t results collecting ...')
        # res into dict
        res_rpsl_lkng[age_range] = res_eff_rpsl_lkng
        res_lkng_lknt[age_range] = res_eff_lkng_lknt
        print(f'\t Done for current age group/bin: {age_range}')
        
    
    #%% GAM ON AGE EFFECT OF INTERESTS
    # ---------------------------------------------------------------------------------------
    
    # X = pd.DataFrame({
    #         'age': np.array(subs_age)[:, 0],
    #         'gender': gender[:, 0],
    #         'meanFD': meanFD[:, 0],
    #         'site': sites[:, 0],
    #     })
    
    gam_cfg = Sts.GAMConfig(
        smooth_var="age", 
        k_splines=5, 
        lam_grid=[-5, 5, 12],
        lam_vals=None,
        center_smooth=True,
        grid_size=101, 
        grid_kind='quantile',
        n_boot=1000,
        bootstrap_deriv_se=False, 
        verbose=False)
    
    results = surf.gam_AmB_surf_roi(
        A_lh_paths=gii_L_ls_rpsl, 
        A_rh_paths=gii_R_ls_rpsl, 
        B_lh_paths=gii_L_ls_lkng, 
        B_rh_paths=gii_R_ls_lkng,
        X=dict_X[(6, 19)][['age', 'gender', 'site']], 
        parc_lh=parc_fsLR, parc_rh=parc_fsLR,
        hemi_loader_metric=surf._load_surf_metric,
        gam_cfg=gam_cfg,
        save_dir=None)
    
    
    
    #%% BRAIN VISUALIZATION
    # ---------------------------------------------------------------------------------------
    # surface files for underlay
    surf_fslr_LH = '/public/home/dingrui/tools/masks_atlas/fsaverage.L.inflated.32k_fs_LR.surf.gii'
    surf_fslr_RH = '/public/home/dingrui/tools/masks_atlas/fsaverage.R.inflated.32k_fs_LR.surf.gii'
    
    # extract effect maps of interests
    target_eff = 'adj_mean_Delta'
    gii_lh_eff = res_eff_rpsl_lkng['contrasts'][target_eff]['lh_t']
    gii_rh_eff = res_eff_rpsl_lkng['contrasts'][target_eff]['rh_t']
    
    vmin = np.min(np.vstack([gii_lh_eff.darrays[0].data, gii_rh_eff.darrays[0].data]).T)
    vmax = np.max(np.vstack([gii_lh_eff.darrays[0].data, gii_rh_eff.darrays[0].data]).T)
    
    # region mask for statistical significance (survive correction)
    t_roi_fdr = res_eff_rpsl_lkng['contrasts'][target_eff]['t_roi'][res_eff_rpsl_lkng[f'{target_eff}_rej_bh']]
    low_thresh, high_thresh = find_extremes(t_roi_fdr)
    
    sig_mask_lh_eff = surf.threshold_gifti(gii_lh_eff, round(low_thresh, 4), round(high_thresh, 4), return_mask=True)
    sig_mask_rh_eff = surf.threshold_gifti(gii_rh_eff, round(low_thresh, 4), round(high_thresh, 4), return_mask=True)
    
    # generate custom colormap with cut point of two colors anchored to 0
    cmap = img.make_center_anchored_div_cmap(
        vmin, vmax, 
        vcenter=0.0, cmap="coolwarm",
        samples_per_side=256, mode="segmented")
    
    # project vertex gii into surface of fsLR space
    p = Plot(surf_fslr_LH, surf_fslr_RH, brightness=0.2)
    
    p.add_layer(
        {'left':  gii_lh_eff,
         'right': gii_rh_eff},
        color_range=(vmin, vmax),
        cbar=False, cmap=cmap, as_outline=False
    )
    p.add_layer(
        {'left':  sig_mask_lh_eff,
         'right': sig_mask_rh_eff}, 
        cbar=False, cmap='hot_r', alpha=1, as_outline=True)
    
    fig = p.build()
    fig.suptitle('age effect on (lkng - lknt) (n=964)', x=0.52, y=0.92)
    
    # manually create a mappable matching your layer and attach a manual colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    smp = cm.ScalarMappable(norm=norm, cmap=cmap)
    smp.set_array([])
    
    cbar = fig.colorbar(smp, ax=fig.axes, location='bottom', ticks=[vmin, 0, vmax], shrink=0.35, pad=0.02, format='%.1f')
    cbar.ax.axvline(0, linestyle='-', linewidth=0.7, color='k', zorder=3)
    
    plt.show()
    
    #%% quick brain visualization of statistical maps
    #
    # surface files for underlay
    surf_fslr_LH = '/public/home/dingrui/tools/masks_atlas/fsaverage.L.inflated.32k_fs_LR.surf.gii'
    surf_fslr_RH = '/public/home/dingrui/tools/masks_atlas/fsaverage.R.inflated.32k_fs_LR.surf.gii'
    
    p = Plot(surf_fslr_LH, surf_fslr_RH, brightness=0.2)
    
    p.add_layer(
        data={'left':  sig_mask_lh_eff,
              'right': sig_mask_rh_eff},
        # color_range=(vmin, vmax),
        cbar=True, cmap='coolwarm', as_outline=False
    )
    
    fig = p.build()
    fig.show()
    
    #%% save results into local
    # save directory
    dir_res = '/public/home/dingrui/fmri_analysis/res/res_glm/roi_level/age_groups'
    # parcellation gifti images for brain data projection back to surface
    fpath_parc_atlas = [
        '/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii',
        '/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii'
    ]
    
    parcellater = Parcellater(tuple(fpath_parc_atlas), space='fsLR').fit()
    
    # effect of interest
    target_eff = 'adj_mean_Delta'
    
    for age_range, res in res_rpsl_lkng.items():
        brainData_thresh = res['contrasts'][target_eff]['t_roi']*res[f'{target_eff}_rej_bh']
        # project brain data back to brain surface
        gii_imgs_thresh = parcellater.inverse_transform(brainData_thresh)
        
        # save brain images into local
        # thresholded images:
        gii_imgs_thresh[0].to_filename(
            os.path.join(
                dir_res, 
                f'brainAct-surf_cont-rpsl-lkng_ageGroup-{age_range[0]}-{age_range[1]}_FDR-05.L.func.gii'
            )
        )
        gii_imgs_thresh[1].to_filename(
            os.path.join(
                dir_res, 
                f'brainAct-surf_cont-rpsl-lkng_ageGroup-{age_range[0]}-{age_range[1]}_FDR-05.R.func.gii'
            )
        )
        
        # unthresholded brain images:
        gii_lh_eff = res['contrasts'][target_eff]['lh_t']
        gii_rh_eff = res['contrasts'][target_eff]['rh_t']
        gii_lh_eff.to_filename(
            os.path.join(
                dir_res, 
                f'brainAct-surf_cont-rpsl-lkng_ageGroup-{age_range[0]}-{age_range[1]}_unthresh.L.func.gii'
            )
        )
        gii_rh_eff.to_filename(
            os.path.join(
                dir_res, 
                f'brainAct-surf_cont-rpsl-lkng_ageGroup-{age_range[0]}-{age_range[1]}_unthresh.R.func.gii'
            )
        )
    
    #%% TESTING CELL
    # ---------------------------------------------------------------------------------------
    # Visual check: a horizontal gradient from vmin to vmax
    vmin = -4.86
    vmax = 6.96
    cmap, norm = img.make_diverging_cmap(-4.86, 6.96, vcenter=0, cmap='bwr')
    data = np.linspace(vmin, vmax, 600, dtype=float)[None, :]  # shape (1, N)
    
    fig, ax = plt.subplots(figsize=(8, 1.2), dpi=160)
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks([])
    ax.set_xticks([0, int((0 - vmin) / (vmax - vmin) * (data.shape[1] - 1)), data.shape[1]-1])
    ax.set_xticklabels([f"{vmin}", "0", f"{vmax}"])
    ax.set_title("Diverging colormap centered at 0 (TwoSlopeNorm)")
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.25, pad=0.35)
    cbar.set_label("value")
    plt.show()
    #%%
