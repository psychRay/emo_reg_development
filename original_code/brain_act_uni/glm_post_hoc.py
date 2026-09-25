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

if __name__ == "__main__":
    # Figure 3 and Figure 5 image comparisons use participant and reference brain maps.
    #%%
    # source and target images
    fpath_srcImg = '/public/home/qinshaozheng/fmri_task/resources/annotation_files/zMap-emoReg.nii.gz'
    fpath_trgImg = '/public/home/qinshaozheng/fmri_task/ml_signature/res/mlRes-enc_unthresh_desc-brainImg_task-classify_rpsl_lkng.dscalar.nii'
    
    srcImg = list(mni152_to_fslr(load_nifti(fpath_srcImg), fslr_density='32k'))
    trgImg = [cifti_to_gifti(fpath_trgImg)['L_img'], cifti_to_gifti(fpath_trgImg)['R_img']]
    
    # visualize two images for a quick view
    surfaces = fetch_fslr()
    lh, rh = surfaces['veryinflated']
    
    for layer in [srcImg, trgImg]:
        surf = Plot(surf_lh=lh, surf_rh=rh, brightness=0.5)
        surf.add_layer(
            {'left': layer[0], 'right': layer[1]}, 
            cmap='YlOrRd_r',
            zero_transparent=False,
        )
        
        fig = surf.build()
        fig.show()
    
    #%%
    # parcellate image
    fpath_parc_atlas = [
        '/public/home/qinshaozheng/fmri_task/resources/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii',
        '/public/home/qinshaozheng/fmri_task/resources/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii'
    ]
    
    parcellater = Parcellater(tuple(fpath_parc_atlas), space='fsLR').fit()
    parcs_srcImg = parcellater.transform(srcImg, space='fsLR')
    parcs_trgImg = parcellater.transform(trgImg, space='fsLR')
    
    # project parcellated brain data back into brain surface
    srcImg_parcs = parcellater.inverse_transform(parcs_srcImg)
    trgImg_parcs = parcellater.inverse_transform(parcs_trgImg*(parcs_trgImg<0))
    
    # visualize projected brain image for a quick view
    surf = Plot(surf_lh=lh, surf_rh=rh, brightness=0.5)
    surf.add_layer(
        {'left': srcImg_parcs[0], 'right': srcImg_parcs[1]}, 
        cmap='YlOrRd_r',
        zero_transparent=False,
    )
    
    fig = surf.build()
    fig.show()
    
    #%% SPATIAL CORRELATIONS AMONG A GROUP OF IMAGES(1)
    # pairwise spatial correlation given a group of brain images
    # whether plot heatmap of correlations
    draw_heat = False
    # directories of brain images
    dir_annot = '/public/home/qinshaozheng/fmri_task/resources/annotation_files'
    dir_sourc = '/public/home/qinshaozheng/fmri_task/brain_glm/res_glm'
    
    # annotation filenames
    names_annot = [
        'reappraisal_FDR_0.05.nii.gz',
        'reappraisal_comp1.nii',
        'reappraisal_comp2.nii',
        'negative_emotion_FDR_0.01.nii.gz',
        'emoGen_comp1.nii',
        'emoGen_comp2.nii',
        'faces_FDR_0.01.nii.gz',
    ]
    
    # stats map filenames
    names_sourc = [
        'brainAct-volume_cont-rpsl-lkng_unthresh.nii.gz',
        'brainAct-volume_cont-lkng-lknt_unthresh.nii.gz'
    ]
    
    # labels of all included images
    imgs_labels = [
        'emoReg',
        'emoReg_comp1',
        'emoReg_comp2',
        'negEmo',
        'emoGen_comp1',
        'emoGen_comp2',
        'face',
        'RPSL',
        'LKNG',
    ]
    
    # pair-wise spatial correlations
    imgs_annot = [nib.load(os.path.join(dir_annot, name)) for name in names_annot]
    imgs_sourc = [nib.load(os.path.join(dir_sourc, name)) for name in names_sourc]
    
    imgs_rsamp_annot = [resample_to_img(img, imgs_sourc[0], interpolation='continuous') for img in imgs_annot]
    imgs_alrdy = [math_img('img*(img>2.81)', img=img_sourc) for img_sourc in imgs_sourc]
    
    imgs_ls = imgs_rsamp_annot + imgs_alrdy
    
    r_mat_1, info = pairwise_compare_images(
        images=imgs_ls,
        labels=imgs_labels,
        metric=cosine,
        diagonal=0,
        ignore_zero=True,
    )
    
    r_mat1_cos = 1-r_mat_1
    
    #---------------------------------------
    # plot heatmap for pair-wise correlation
    #---------------------------------------
    if draw_heat:
        df_heat = r_mat1_cos[['RPSL', 'LKNG']].iloc[0:6, :]
        sns.heatmap(
            df_heat,
            robust=True, annot=True,
            cmap='viridis', linecolor='k', linewidths=1.5,
            square=True,
        )
    
    #%% SPATIAL CORRELATIONS AMONG A GROUP OF IMAGES(2)
    # pairwise spatial correlation given a group of brain images
    # whether plot heatmap of correlations
    draw_line = False
    # directories of brain images
    dir_annot = '/public/home/qinshaozheng/fmri_task/resources/annotation_files'
    dir_sourc = '/public/home/qinshaozheng/fmri_task/brain_glm/res_glm/age_groups'
    
    # annotation filenames
    names_annot = [
        'reappraisal_FDR_0.05.nii.gz',
        'reappraisal_comp1.nii',
        'reappraisal_comp2.nii',
        'negative_emotion_FDR_0.01.nii.gz',
        'emoGen_comp1.nii',
        'emoGen_comp2.nii',
    ]
    
    # stats map filenames
    age_range = [
        (6,8),   (8,9),   (9,10),  (10,11), (11,12), (12,13),
        (13,14), (14,15), (15,16), (16,17), (17,18), (18,19),
    ]
    
    names_sourc = [
        os.path.join(
            dir_sourc,
            f'brainAct-volume_stats-z_cont-rpsl-lkng_ageGroup-{age[0]}-{age[1]}_unthresh.nii',
        )
        for age in age_range
    ]
    
    # labels of all included images
    imgs_labels = [
        'emoReg',
        'emoReg_comp1',
        'emoReg_comp2',
        'negEmo',
        'emoGen_comp1',
        'emoGen_comp2',
        'age 6-8', 'age 8-9', 'age 9-10', 'age 10-11', 'age 11-12', 'age 12-13',
        'age 13-14', 'age 14-15', 'age 15-16', 'age 16-17', 'age 17-18', 'age 18-19',
    ]
    
    # pair-wise spatial correlations
    imgs_annot = [nib.load(os.path.join(dir_annot, name)) for name in names_annot]
    imgs_sourc = [nib.load(os.path.join(dir_sourc, name)) for name in names_sourc]
    
    imgs_rsamp = [resample_to_img(img, imgs_sourc[0], interpolation='continuous') for img in imgs_annot]
    imgs_alrdy = [math_img('img*(img>3.29)', img=img_sourc) for img_sourc in imgs_sourc]
    
    imgs_ls = imgs_rsamp + imgs_alrdy
    
    r_mat_2, info = pairwise_compare_images(
        images=imgs_ls,
        labels=imgs_labels,
        metric=cosine,
        diagonal=0,
        ignore_zero=True,
    )
    
    r_mat2_cos = 1 - r_mat_2
    
    # lineplot for sequential correlations
    if draw_line:
        df_line = r_mat2_cos[['emoReg', 'emoReg_comp2']].iloc[6:18]
        
        fig, ax = plt.subplots(figsize=(3.5, 3))
        ax = sns.lineplot(
            df_line, 
            palette={'emoReg': '#9B359B', 'emoReg_comp2': '#CE636D'},
            linewidth=2, markers=True, dashes=False, 
            ax=ax, legend=False
        )
        
        ax.set_xticks(list(range(0,12)))
        ax.set_xticklabels([
            '6-8', '8-9', '9-10', '10-11', '11-12', '12-13',
            '13-14', '14-15', '15-16', '16-17', '17-18', '18-19'
        ])
        
        sns.despine()
        # 
        ax.axvline(x=4, color='gray', linestyle='--', linewidth=1, alpha=0.8)
        plt.xlabel('age groups/bins')
        plt.ylabel('similarity btw spatial patterns')
        plt.xticks(rotation=90, ha='center', fontsize=10)
    
    
    #%% 
    # compare two brain images using based on null models
    spatial_r = compare_images(
        src=srcImg_parcs,
        trg=trgImg_parcs,
        metric='spearmanr',
    )
    
    #%%
    #%%
