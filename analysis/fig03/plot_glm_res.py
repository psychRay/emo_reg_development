#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:13:20 2026

For Fig. 3e calculation and plotting; 
For Fig. 4c plotting;
For Fig. 4d calculation;


@author: dingrui
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from pathlib import Path
from nilearn.image import math_img, resample_to_img
from nilearn.plotting import plot_glass_brain
from neuromaps.transforms import mni152_to_fslr

from neuro_utils.parcellation import brain_to_parcel, parcel_to_brain, parcel_mapper
from neuro_utils.display import make_subcortex_display_mask_cifti
from neuro_utils.wb_func import dscalar_to_border, cifti_to_gifti, gifti_to_cifti
from neuro_utils.network import compute_network_contribution, _load_brain_like_data, summarize_brain_maps_by_network

# %% NECESSARY FUNCTIONS
#
def keep_values_in_nifti(
    img,
    values_to_keep,
    out_file=None,
    dtype=None
):
    """
    Set voxels whose values are not in `values_to_keep` to 0.

    Parameters
    ----------
    img : str, Path, or nibabel image
        Input NIfTI image or file path.

    values_to_keep : array-like
        Values to keep. Voxels not equal to any of these values will be set to 0.

    out_file : str or Path, optional
        Output NIfTI file path. If None, the image is not saved.

    dtype : numpy dtype, optional
        Output data dtype. If None, use the original data dtype when possible.

    Returns
    -------
    out_img : nibabel.Nifti1Image
        Modified NIfTI image.
    """

    # Load image if a file path is provided
    if isinstance(img, (str, Path)):
        img = nib.load(str(img))

    data = img.get_fdata()
    values_to_keep = np.asarray(values_to_keep)

    # Create mask: True for voxels whose value is in values_to_keep
    keep_mask = np.isin(data, values_to_keep)

    # Set all other voxels to 0
    out_data = np.where(keep_mask, data, 0)

    if dtype is not None:
        out_data = out_data.astype(dtype)

    out_img = nib.Nifti1Image(
        out_data,
        affine=img.affine,
        header=img.header.copy()
    )

    # Update header dtype if requested
    if dtype is not None:
        out_img.set_data_dtype(dtype)

    if out_file is not None:
        nib.save(out_img, str(out_file))

    return out_img


def plot_network_distributed_score(
    df,
    network_name_col=None,
    network_order=None,
    age_order=None,
    network_colors=None,
    multiply_by=100,
    title="Network distributed score",
    xlabel="Age group",
    ylabel="Distributed score",
    figsize=(10, 5),
    rotation=90,
    legend=True,
    legend_loc="upper left",
    legend_bbox=(1.02, 1.0),
    ax=None,
    save_path=None,
    dpi=300,
):
    """
    Plot network distributed scores across age groups as a stacked bar chart.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data matrix.

        Supported formats:

        1. Network names as index:
           index = network names
           columns = age groups

        2. Network names as a column:
           one column contains network names, e.g. "network_name"
           other columns contain age groups

    network_name_col : str, optional
        Column name containing network names.
        If None, the function assumes network names are stored in df.index.

    network_order : list, optional
        Order of networks from bottom to top in the stacked bar.
        If None, use the order in the input data.

    age_order : list, optional
        Order of age groups on the x-axis.
        If None, use all columns except `network_name_col` when network names
        are stored in a column; otherwise use df.columns.

    network_colors : dict or list, optional
        Colors for networks.
        If dict, keys should be network names.
        If list, length should match number of networks.
        If None, matplotlib default color cycle is used.

    multiply_by : float or None, default=100
        If values are proportions, use multiply_by=100.
        If values are already percentages or raw scores, use multiply_by=None.

    title, xlabel, ylabel : str
        Plot labels.

    figsize : tuple
        Figure size.

    rotation : int
        Rotation angle for x tick labels.

    legend : bool
        Whether to show legend.

    legend_loc : str
        Legend location.

    legend_bbox : tuple
        Legend bbox_to_anchor setting.

    ax : matplotlib.axes.Axes, optional
        Existing axes object. If None, create a new figure and axes.

    save_path : str, optional
        If provided, save the figure to this path.

    dpi : int
        Resolution for saved figure.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")

    plot_df = df.copy()

    # Case 1: network names are stored in a regular column
    if network_name_col is not None:
        if network_name_col not in plot_df.columns:
            raise ValueError(
                f"`network_name_col` = {network_name_col!r} was not found in df.columns."
            )

        if plot_df[network_name_col].duplicated().any():
            duplicated = plot_df.loc[
                plot_df[network_name_col].duplicated(), network_name_col
            ].tolist()
            raise ValueError(
                f"Duplicated network names found in `{network_name_col}`: {duplicated}"
            )

        plot_df = plot_df.set_index(network_name_col)

    # Case 2: network names are already stored in index
    else:
        if plot_df.index.name is None:
            plot_df.index.name = "network_name"

    # Determine network order
    if network_order is None:
        network_order = list(plot_df.index)
    else:
        missing_networks = set(network_order) - set(plot_df.index)
        if missing_networks:
            raise ValueError(f"Networks not found in data: {missing_networks}")

    # Determine age order
    if age_order is None:
        age_order = list(plot_df.columns)
    else:
        missing_ages = set(age_order) - set(plot_df.columns)
        if missing_ages:
            raise ValueError(f"Age groups not found in data columns: {missing_ages}")

    plot_df = plot_df.loc[network_order, age_order]

    # Convert all selected values to numeric
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")

    if plot_df.isna().any().any():
        bad_locations = np.where(plot_df.isna())
        bad_rows = plot_df.index[bad_locations[0]].tolist()
        bad_cols = plot_df.columns[bad_locations[1]].tolist()
        raise ValueError(
            "Selected data contain non-numeric or NaN values. "
            f"Examples: {list(zip(bad_rows[:5], bad_cols[:5]))}"
        )

    if multiply_by is not None:
        plot_df = plot_df * multiply_by

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(len(age_order))
    bottom = np.zeros(len(age_order), dtype=float)

    for i, network in enumerate(network_order):
        values = plot_df.loc[network].values.astype(float)

        if network_colors is None:
            color = None
        elif isinstance(network_colors, dict):
            color = network_colors.get(network, None)
        else:
            if len(network_colors) != len(network_order):
                raise ValueError(
                    "`network_colors` as a list must have the same length as `network_order`."
                )
            color = network_colors[i]

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=network,
            color=color,
            edgecolor="black",
            linewidth=0.3,
        )

        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(age_order, rotation=rotation)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if legend:
        ax.legend(
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            frameon=False,
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_bilateral_bar(
    left_values,
    right_values,
    y_labels=None,
    figsize=(7, 5),
    bar_height=0.7,
    left_label="Left",
    right_label="Right",
    xlabel="Value",
    ylabel=None,
    xaxis_position="bottom",
    left_colors=None,
    right_colors=None,
    show_yticklabels=False,
    show_values=False,
    value_format="{:.2f}",
    value_offset=None,
    value_text_kwargs=None,
    left_bar_kwargs=None,
    right_bar_kwargs=None,
    ax=None,
):
    """
    Plot two one-dimensional arrays as bilateral horizontal bar plots.

    Parameters
    ----------
    left_values : array-like
        Values plotted on the left side of the shared y-axis.

    right_values : array-like
        Values plotted on the right side of the shared y-axis.

    y_labels : list-like, optional
        Labels for the shared y-axis. Used only when show_yticklabels=True.

    figsize : tuple
        Figure size.

    bar_height : float
        Height of each horizontal bar.

    left_label : str
        Legend label for the left-side bars.

    right_label : str
        Legend label for the right-side bars.

    xlabel : str
        X-axis label.

    ylabel : str, optional
        Y-axis label.

    xaxis_position : {"bottom", "top"}
        Position of the x-axis.

    left_colors : str or list-like, optional
        Color or colors for the left-side bars.

    right_colors : str or list-like, optional
        Color or colors for the right-side bars.

    show_yticklabels : bool
        Whether to show y-axis tick labels.

    show_values : bool
        Whether to show value labels at the outer edge of each bar.

    value_format : str
        Format string for value labels, e.g., "{:.2f}" or "{:.3f}".

    value_offset : float, optional
        Offset between bar edge and value label. If None, it is determined
        automatically from the data range.

    value_text_kwargs : dict, optional
        Keyword arguments for value text labels.

    left_bar_kwargs : dict, optional
        Additional keyword arguments for left-side bars.

    right_bar_kwargs : dict, optional
        Additional keyword arguments for right-side bars.

    ax : matplotlib.axes.Axes, optional
        Existing axes object.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """

    left_values = np.asarray(left_values, dtype=float).ravel()
    right_values = np.asarray(right_values, dtype=float).ravel()

    if left_values.ndim != 1 or right_values.ndim != 1:
        raise ValueError("left_values and right_values must be one-dimensional.")

    if len(left_values) != len(right_values):
        raise ValueError("left_values and right_values must have the same length.")

    if xaxis_position not in ["bottom", "top"]:
        raise ValueError("xaxis_position must be either 'bottom' or 'top'.")

    n = len(left_values)
    y_pos = np.arange(n)

    if y_labels is None:
        y_labels = [str(i) for i in range(n)]

    if len(y_labels) != n:
        raise ValueError("y_labels must have the same length as input arrays.")

    left_bar_kwargs = left_bar_kwargs or {}
    right_bar_kwargs = right_bar_kwargs or {}
    value_text_kwargs = value_text_kwargs or {}

    default_left_kwargs = dict(alpha=0.8, label=left_label)
    default_right_kwargs = dict(alpha=0.8, label=right_label)

    if left_colors is not None:
        default_left_kwargs["color"] = left_colors

    if right_colors is not None:
        default_right_kwargs["color"] = right_colors

    default_left_kwargs.update(left_bar_kwargs)
    default_right_kwargs.update(right_bar_kwargs)

    default_value_text_kwargs = dict(
        fontsize=10,
        va="center"
    )
    default_value_text_kwargs.update(value_text_kwargs)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Left-side bars are plotted as negative values.
    ax.barh(
        y_pos,
        -left_values,
        height=bar_height,
        **default_left_kwargs
    )

    # Right-side bars are plotted as positive values.
    ax.barh(
        y_pos,
        right_values,
        height=bar_height,
        **default_right_kwargs
    )

    # Determine x-axis range.
    max_value = max(np.max(left_values), np.max(right_values))

    if value_offset is None:
        value_offset = max_value * 0.03 if max_value > 0 else 0.03

    # Add value labels.
    if show_values:
        for i, value in enumerate(left_values):
            ax.text(
                -value - value_offset,
                y_pos[i],
                value_format.format(value),
                ha="right", rotation=90,
                **default_value_text_kwargs
            )

        for i, value in enumerate(right_values):
            ax.text(
                value + value_offset,
                y_pos[i],
                value_format.format(value),
                ha="left", rotation=270,
                **default_value_text_kwargs
            )

    # Move the y-axis spine to the center.
    ax.spines["left"].set_position(("data", 0))
    ax.spines["left"].set_visible(True)

    # Control x-axis position.
    if xaxis_position == "bottom":
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_label_position("bottom")

        ax.spines["bottom"].set_visible(True)
        ax.spines["top"].set_visible(False)

    else:
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        ax.spines["top"].set_visible(True)
        ax.spines["bottom"].set_visible(False)

    # Hide unnecessary spines.
    ax.spines["right"].set_visible(False)

    ax.set_yticks(y_pos)

    if show_yticklabels:
        ax.set_yticklabels(y_labels)
    else:
        ax.set_yticklabels([])

    ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Make x-axis symmetric and leave room for value labels.
    xlim = max_value + value_offset * 4
    ax.set_xlim(-xlim, xlim)

    # Display x tick labels as absolute values.
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x):g}" for x in xticks])
    
    if left_label is not None or right_label is not None:
        ax.legend(frameon=False)

    return fig, ax
