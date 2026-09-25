#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 07:05:54 2026

@author: dingrui
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Sequence, Optional, Union, Callable, Dict, Any

from neuromaps.datasets import fetch_fslr, fetch_atlas
from neuromaps.images import load_nifti, load_gifti, construct_shape_gii
from neuromaps.transforms import mni152_to_fslr
from neuromaps.parcellate import Parcellater
from neuromaps.stats import compare_images, permtest_metric
from surfplot import Plot
from nilearn.image import resample_to_img, math_img
from scipy.spatial.distance import cosine

from neuro_utils.parcellation import _get_cifti_data_matrix
from neuro_utils.wb_func import cifti_to_gifti

#%% NECESSARY FUNCTIONS
def mni152_to_fslr_mask_medial(img, density="32k", method="linear", data_dir=None):
    """
    Project MNI152 volume to fsLR surface and mask medial wall vertices.

    Parameters
    ----------
    img : str or os.PathLike or niimg_like
        Input MNI152 volumetric image.
    density : {"32k", "164k"}
        fsLR output density.
    method : {"linear", "nearest"}
        Interpolation method.
    data_dir : str or None
        Optional neuromaps data directory.

    Returns
    -------
    out : tuple of nib.GiftiImage
        Left and right hemisphere GIFTI images with medial wall set to NaN.
    """
    lh, rh = mni152_to_fslr(
        img,
        fslr_density=density,
        method=method
    )

    fslr = fetch_atlas("fsLR", density, data_dir=data_dir)
    medial_lh = nib.load(fslr["medial"].L).agg_data()
    medial_rh = nib.load(fslr["medial"].R).agg_data()

    data_lh = lh.agg_data().astype(float)
    data_rh = rh.agg_data().astype(float)

    # In neuromaps atlas, medial wall vertices have value 1
    data_lh[medial_lh == 0] = np.nan
    data_rh[medial_rh == 0] = np.nan

    lh_masked = construct_shape_gii(data_lh)
    rh_masked = construct_shape_gii(data_rh)

    return lh_masked, rh_masked


ImageLike = Union[
    str,
    os.PathLike,
    nib.Nifti1Image,
    nib.GiftiImage,
    nib.Cifti2Image,
    tuple,
    list,
    np.ndarray,
]


def _infer_image_format(img: ImageLike) -> str:
    """
    Infer image format from path, nibabel image object, tuple/list, or ndarray.

    Returns
    -------
    fmt : str
        One of {'nifti', 'gifti', 'cifti', 'array'}.
    """
    if isinstance(img, nib.Cifti2Image):
        return "cifti"

    if isinstance(img, nib.GiftiImage):
        return "gifti"

    if isinstance(img, nib.Nifti1Image):
        return "nifti"

    if isinstance(img, np.ndarray):
        return "array"

    if isinstance(img, (tuple, list)):
        if len(img) == 0:
            raise ValueError("Empty tuple/list is not a valid brain image.")

        sub_formats = [_infer_image_format(x) for x in img]

        if len(set(sub_formats)) != 1:
            raise ValueError(
                f"Mixed image formats within tuple/list image: {sub_formats}"
            )

        return sub_formats[0]

    if isinstance(img, (str, os.PathLike)):
        path = str(img)

        if path.endswith((".nii", ".nii.gz")):
            # CIFTI commonly uses .dscalar.nii, .dtseries.nii, .dlabel.nii, etc.
            cifti_suffixes = (
                ".dscalar.nii",
                ".dtseries.nii",
                ".dlabel.nii",
                ".ptseries.nii",
                ".pscalar.nii",
                ".pconn.nii",
                ".dconn.nii",
            )
            if path.endswith(cifti_suffixes):
                return "cifti"
            return "nifti"

        if path.endswith((".gii", ".func.gii", ".shape.gii", ".label.gii")):
            return "gifti"

        raise ValueError(f"Cannot infer image format from file path: {path}")

    raise TypeError(f"Unsupported image type: {type(img)}")


def _check_uniform_format(images: Sequence[ImageLike]) -> str:
    """
    Check whether all images have the same inferred format.
    """
    if len(images) < 2:
        raise ValueError("At least two images are required.")

    formats = [_infer_image_format(img) for img in images]

    if len(set(formats)) != 1:
        raise ValueError(
            "All input brain images must have the same format. "
            f"Detected formats: {formats}"
        )

    return formats[0]


def _get_pair_nulls(
    nulls: Optional[Union[Sequence[np.ndarray], Dict[Any, np.ndarray]]],
    i: int,
    label_i: Any,
):
    """
    Get null array corresponding to the first image in compare_images(img_i, img_j).

    In neuromaps, nulls must correspond to the first positional argument.
    """
    if nulls is None:
        return None

    if isinstance(nulls, dict):
        if label_i not in nulls:
            raise KeyError(f"No nulls found for image label: {label_i}")
        return nulls[label_i]

    if isinstance(nulls, (list, tuple)):
        if len(nulls) <= i:
            raise ValueError(
                "When nulls is a list/tuple, it must have the same length as images."
            )
        return nulls[i]

    raise TypeError(
        "nulls must be None, a list/tuple of null arrays, or a dict mapping labels to null arrays."
    )


def pairwise_compare_images(
    images: Sequence[ImageLike],
    labels: Optional[Sequence[str]] = None,
    metric: Union[str, Callable] = "pearsonr",
    ignore_zero: bool = True,
    nulls: Optional[Union[Sequence[np.ndarray], Dict[Any, np.ndarray]]] = None,
    nan_policy: str = "omit",
    diagonal: float = 1.0,
    return_dataframe: bool = True,
    p_diagonal: float = 0.0,
):
    """
    Compute pairwise image similarity matrix using neuromaps.stats.compare_images.

    Parameters
    ----------
    images : sequence
        A sequence of brain images. Supported forms include:
        - NIfTI paths or nib.Nifti1Image
        - GIFTI paths / nib.GiftiImage / tuple of left-right GIFTI images
        - CIFTI paths or nib.Cifti2Image
        - array-like data

        All images must have the same inferred format.

    labels : sequence of str, optional
        Labels for rows/columns of the output matrices. If None, labels will be
        generated as image_0, image_1, ...

    metric : {'pearsonr', 'spearmanr'} or callable, default='pearsonr'
        Similarity metric passed to neuromaps.stats.compare_images.

    ignore_zero : bool, default=True
        Whether to ignore zero values during comparison. This follows the
        neuromaps compare_images behavior.

    nulls : list, tuple, dict, or None, default=None
        Null data for significance testing.

        Recommended format:
            nulls = [nulls_for_img0, nulls_for_img1, ...]

        or:
            nulls = {'map1': nulls_for_map1, 'map2': nulls_for_map2, ...}

        Important:
        For pair (i, j), this function calls:
            compare_images(images[i], images[j], nulls=nulls[i])

        Therefore nulls[i] must correspond to images[i].

    nan_policy : {'propagate', 'raise', 'omit'}, default='omit'
        How to handle NaN values.

    diagonal : float, default=1.0
        Diagonal value of the correlation/similarity matrix.

    return_dataframe : bool, default=True
        If True, return pandas DataFrame objects.
        If False, return numpy arrays.

    p_diagonal : float, default=0.0
        Diagonal value of the p-value matrix when nulls are provided.

    Returns
    -------
    r_matrix : pandas.DataFrame or numpy.ndarray
        Pairwise similarity / correlation matrix.

    p_matrix : pandas.DataFrame or numpy.ndarray, optional
        Pairwise p-value matrix. Returned only when nulls is not None.

    info : dict
        Basic information, including inferred image format and metric.
    """
    images = list(images)
    n_images = len(images)

    image_format = _check_uniform_format(images)

    if labels is None:
        labels = [f"image_{i}" for i in range(n_images)]
    else:
        labels = list(labels)

    if len(labels) != n_images:
        raise ValueError("labels must have the same length as images.")

    if nulls is not None and isinstance(nulls, (list, tuple)):
        if len(nulls) != n_images:
            raise ValueError(
                "When nulls is a list/tuple, it must have the same length as images."
            )

    r_mat = np.full((n_images, n_images), np.nan, dtype=float)
    np.fill_diagonal(r_mat, diagonal)

    if nulls is not None:
        p_mat = np.full((n_images, n_images), np.nan, dtype=float)
        np.fill_diagonal(p_mat, p_diagonal)
    else:
        p_mat = None

    for i in range(n_images):
        for j in range(i + 1, n_images):
            if nulls is None:
                r = compare_images(
                    images[i],
                    images[j],
                    metric=metric,
                    ignore_zero=ignore_zero,
                    nulls=None,
                    nan_policy=nan_policy,
                )
                
                if isinstance(r, tuple) and np.nan in r:
                    r_mat[i, j] = 0
                    r_mat[j, i] = 0
                else:
                    r_mat[i, j] = r
                    r_mat[j, i] = r

            else:
                pair_nulls = _get_pair_nulls(nulls, i, labels[i])

                r, p = compare_images(
                    images[i],
                    images[j],
                    metric=metric,
                    ignore_zero=ignore_zero,
                    nulls=pair_nulls,
                    nan_policy=nan_policy,
                )

                r_mat[i, j] = r
                r_mat[j, i] = r

                # This p-value is based on nulls for images[i].
                # We mirror it to keep a symmetric p-value matrix.
                p_mat[i, j] = p
                p_mat[j, i] = p

    info = {
        "n_images": n_images,
        "image_format": image_format,
        "metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "callable"),
        "ignore_zero": ignore_zero,
        "nan_policy": nan_policy,
        "p_values": nulls is not None,
    }

    if return_dataframe:
        r_mat = pd.DataFrame(r_mat, index=labels, columns=labels)

        if p_mat is not None:
            p_mat = pd.DataFrame(p_mat, index=labels, columns=labels)

    if nulls is None:
        return r_mat, info

    return r_mat, p_mat, info
