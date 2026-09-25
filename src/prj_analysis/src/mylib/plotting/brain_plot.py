#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 03:40:42 2025

    sorts of tools to visualize brain-related contents

@author: dingrui
"""

# modules
import warnings
warnings.filterwarnings('ignore', message='.*deprecated.*')

import os
from typing import Union, Dict, Optional

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# functions
def plot_network_boxplot(
    df: pd.DataFrame,
    value_col: str = "voxel_val",
    label_col: str = "network_label",
    order_by: Optional[str] = "median",  # {'median','mean','alphabetical', None}
    orient: Optional[str] = "v",         # {'v'--> vertical box, 'h' --> horizontal box}
    percentiles = None, log_scale = False,
    figsize = (10, 6),
    # ---- coloring ----
    palette: Optional[Union[str, Dict[str, str]]] = "tab20",
    lighten: bool = True,
    lighten_factor: float = 0.15,
    # ---- appearance ----
    showfliers: bool = False, box_width = 0.5,
    linewidth: float = 0.8,
    median_linewidth: float = 1.2,
    # ---- points control ----
    show_points: bool = False,
    points_layer: Optional[str] = "below",  # 'below' | 'above' | None
    points_alpha: float = 0.35,
    points_size: float = 1.5,
    points_max_n: int = 50_000,
    # ---- axes properties ----
    xlabel = None, ylabel = None,
    num_xticks: int = None, num_yticks: int = None,
):
    """
    Draw a boxplot grouped by `label_col`, with optional per-category coloring and points.
    
        
        - df: a pd.dataframe with two columns representing voxel/vertex values and network label 
              assigned to current voxel/vertex, which can be generated via function `image_to_network_df` 
              within brain module;
        - points_layer: 'below' draws points underneath the boxes (i.e., boxes on top);
    
    """
    # ---- order on x-axis ----
    if order_by in ("median", "mean"):
        if log_scale:
            df = df[df[value_col] > 0]
        stat = df.groupby(label_col)[value_col].agg(order_by).sort_values()
        order = stat.index.tolist()
    elif order_by == "alphabetical":
        order = sorted(df[label_col].unique().tolist())
    else:
        order = list(pd.unique(df[label_col]))

    # ---- palette to dict {label -> color} ----
    def _lighten(color, factor):
        import matplotlib.colors as mcolors
        rgb = np.array(mcolors.to_rgb(color))
        return tuple((1 - factor) * rgb + factor * 1.0)

    labels = order
    if isinstance(palette, dict):
        pal_map = {lab: palette.get(lab) for lab in labels}
        miss = [lab for lab, col in pal_map.items() if col is None]
        if miss:
            cyc = sns.color_palette(n_colors=len(miss))
            for lab, col in zip(miss, cyc): pal_map[lab] = col
    else:
        n = max(len(labels), 3)
        pal_seq = sns.color_palette(palette or "tab20", n_colors=n) \
                  if not isinstance(palette, (list, tuple)) else list(palette)
        if len(pal_seq) < len(labels):
            pal_seq += sns.color_palette(n_colors=len(labels) - len(pal_seq))
        pal_map = {lab: pal_seq[i] for i, lab in enumerate(labels)}

    if lighten:
        pal_map = {lab: _lighten(col, lighten_factor) for lab, col in pal_map.items()}

    # ---- plotting ----
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # helper: draw points with same palette
    def _draw_points(z=1):
        n = len(df)
        df_plot = df.sample(points_max_n, random_state=42) if n > points_max_n else df
        sns.stripplot(
            data=df_plot,
            x=label_col if orient=='v' else value_col, 
            y=value_col if orient=='v' else label_col, 
            order=order,
            hue=label_col, palette=pal_map, dodge=False,
            alpha=points_alpha, size=points_size, legend=False, zorder=z
        )

    # 1) points below -> draw points first (lower zorder), then boxes on top (higher zorder)
    if show_points and points_layer == "below":
        _draw_points(z=1)

    # 2) boxes (ensure they sit above points)
    box = sns.boxplot(
        data=df,
        x=label_col if orient=='v' else value_col, 
        y=value_col if orient=='v' else label_col, 
        order=order, orient=orient, whis=percentiles,
        log_scale=log_scale,
        palette=pal_map, showfliers=showfliers, 
        width=box_width, showcaps=False, zorder=2,
        boxprops={'linewidth': linewidth, 'edgecolor': 'k'},
        medianprops={'linewidth': median_linewidth}
    )
    
    # tick properties
    if orient == 'h' and num_xticks is not None:
        box.xaxis.set_major_locator(ticker.MaxNLocator(nbins=num_xticks))
    if orient == 'v' and num_yticks is not None:
        box.yaxis.set_major_locator(ticker.MaxNLocator(nbins=num_yticks))
    
    if log_scale:
        box.xaxis.set_minor_locator(ticker.NullLocator())

    # 3) points above -> draw after boxes (higher zorder)
    if show_points and points_layer == "above":
        _draw_points(z=3)
    
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    ax.set_title("Voxel/Vertex/ROI values by network")
    # ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if orient=='v':
        plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
    return ax, pal_map