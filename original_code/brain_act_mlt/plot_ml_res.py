#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 22:42:46 2026

@author: dingrui
"""

# %%
import os
import json
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from tqdm import tqdm
from brokenaxes import brokenaxes
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from scipy.stats import chi2
from pygam import LinearGAM, s, f
from nilearn.image import math_img
from neuromaps.transforms import mni152_to_fslr

from neuro_utils.parcellation import brain_to_parcel, parcel_to_brain, parcel_mapper
from neuro_utils.display import make_subcortex_display_mask_cifti
from neuro_utils.wb_func import dscalar_to_border
from neuro_utils.network import compute_network_contribution, _load_brain_like_data, summarize_brain_maps_by_network

# %% define functions
#
def boxplot_with_points(
    df,
    points,
    figsize=None,
    ax=None,
    box_kwargs=None,
    point_kwargs=None,
    line_kwargs=None,
    spine_visible=None,
    x_tick_rotation=0,
    xticklabels=None,
    xlabel=None,
    ylabel=None,
    title=None,
    show_line=False,
    y_break=None,
    break_ratios=(1, 3),
    break_hspace=0.05,
    break_diag_kwargs=None,
):
    """
    Plot boxplots for each column in a DataFrame and overlay one point per column.
    Optionally use a broken y-axis when boxplots and points are far apart.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Each column will be treated as one group and plotted as one box.

    points : list, tuple, numpy.ndarray, or pandas.Series
        Values to overlay on the boxplots. The length must equal the number of columns in df.
        The order of points should match the order of df.columns.

    ax : matplotlib.axes.Axes, optional
        Existing matplotlib Axes object. Only used when y_break is None.

    box_kwargs : dict, optional
        Keyword arguments passed to seaborn.boxplot.

    point_kwargs : dict, optional
        Keyword arguments passed to ax.scatter.

    line_kwargs : dict, optional
        Keyword arguments passed to ax.plot when show_line=True.

    spine_visible : dict, optional
        Controls the visibility of axes spines.
        Keys can include "top", "right", "bottom", and "left".

    x_tick_rotation : int or float, default=0
        Rotation angle for x-axis tick labels.

    xlabel : str, optional
        X-axis label.

    ylabel : str, optional
        Y-axis label.

    title : str, optional
        Plot title.

    show_line : bool, default=False
        Whether to connect the overlaid points with a line.

    y_break : tuple or None, default=None
        If not None, should be:
            ((lower_min, lower_max), (upper_min, upper_max))
        Example:
            y_break=((0.46, 0.54), (0.67, 0.71))
        This creates a broken y-axis.

    break_ratios : tuple, default=(1, 3)
        Height ratios for the upper and lower axes when y_break is used.

    break_hspace : float, default=0.05
        Vertical spacing between the two axes when y_break is used.

    Returns
    -------
    ax or (ax_top, ax_bottom)
        If y_break is None, returns one Axes.
        If y_break is not None, returns (ax_top, ax_bottom).
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    points = np.asarray(points)

    if len(points) != df.shape[1]:
        raise ValueError(
            f"Length of points must equal the number of DataFrame columns. "
            f"Got len(points)={len(points)}, but df has {df.shape[1]} columns."
        )

    if box_kwargs is None:
        box_kwargs = {}

    if point_kwargs is None:
        point_kwargs = {}

    if line_kwargs is None:
        line_kwargs = {}
        
    if break_diag_kwargs is None:
        break_diag_kwargs = {}

    if spine_visible is None:
        spine_visible = {
            "top": False,
            "right": False,
            "bottom": True,
            "left": True,
        }

    default_box_kwargs = {
        "width": 0.5,
        "showfliers": False,
    }

    default_point_kwargs = {
        "s": 70,
        "color": "red",
        "edgecolor": "black",
        "linewidth": 0.8,
        "zorder": 10,
    }

    default_line_kwargs = {
        "color": "red",
        "linewidth": 1.5,
        "alpha": 0.8,
        "zorder": 9,
    }
    
    default_break_diag_kwargs = {
        "d": 0.008,
        "tilt": 45,
        "linewidth": 1.2,
        "color": "k",
    }

    default_box_kwargs.update(box_kwargs)
    default_point_kwargs.update(point_kwargs)
    default_line_kwargs.update(line_kwargs)
    default_break_diag_kwargs.update(break_diag_kwargs)

    long_df = df.melt(var_name="group", value_name="value")
    column_order = list(df.columns)
    x_positions = np.arange(len(column_order))

    def _draw_main_content(current_ax):
        """Draw boxplots and overlaid points on a given axis."""
        sns.boxplot(
            data=long_df,
            x="group",
            y="value",
            order=column_order,
            ax=current_ax,
            **default_box_kwargs,
        )

        if show_line:
            current_ax.plot(
                x_positions,
                points,
                **default_line_kwargs,
            )

        current_ax.scatter(
            x_positions,
            points,
            **default_point_kwargs,
        )

    def _apply_spine_visibility(current_ax, spine_dict):
        """Apply spine visibility settings."""
        for spine_name, visible in spine_dict.items():
            if spine_name not in current_ax.spines:
                raise ValueError(
                    f"Invalid spine name: {spine_name}. "
                    f"Valid spine names are: {list(current_ax.spines.keys())}"
                )
            current_ax.spines[spine_name].set_visible(visible)

    if y_break is None:
        
        if figsize:
            _, ax = plt.subplots(figsize=figsize)
        else:
            _, ax = plt.subplots(figsize=(max(6, df.shape[1] * 0.8), 4))

        _draw_main_content(ax)

        ax.set_xticks(x_positions)
        if xticklabels:
            ax.set_xticklabels(xticklabels, rotation=x_tick_rotation)
        else:
            ax.set_xticklabels(column_order, rotation=x_tick_rotation)

        _apply_spine_visibility(ax, spine_visible)

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        return ax

    else:
        if ax is not None:
            raise ValueError("When y_break is used, ax should be None.")

        (lower_ylim, upper_ylim) = y_break
        
        if figsize:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure(figsize=(max(6, df.shape[1] * 0.8), 5))
    
        bax = brokenaxes(
            ylims=(lower_ylim, upper_ylim),
            hspace=break_hspace,
            height_ratios=break_ratios,
            despine=False,
            fig=fig,
            d=default_break_diag_kwargs.get("d", 0.008),
            tilt=default_break_diag_kwargs.get("tilt", 45),
        )
    
        # brokenaxes internally contains multiple matplotlib Axes.
        # We draw the same seaborn boxplot and overlaid points on each sub-axis.
        ax_top = bax.axs[0]
        ax_bottom = bax.axs[1]
        
        sns.boxplot(
            data=long_df,
            x="group",
            y="value",
            order=column_order,
            ax=ax_bottom,
            **default_box_kwargs,
        ).set_ylabel(None)
        
        ax_bottom.set_xlim(-0.5, len(column_order)-0.5)
        
        ax_top.scatter(
            x_positions,
            points,
            **default_point_kwargs,
        )

        if show_line:
            ax_top.plot(
                x_positions,
                points,
                **default_line_kwargs,
            )
        
        ax_top.set_xlim(-0.5, len(column_order)-0.5)
    
        # xticks properties of ax_bottom
        ax_bottom.set_xticks(x_positions)
        
        if xticklabels:
            ax_bottom.set_xticklabels(xticklabels, rotation=x_tick_rotation)
        else:
            ax_bottom.set_xticklabels(column_order, rotation=x_tick_rotation)
        
        # set ax spines
        _apply_spine_visibility(ax_bottom, spine_visible)
        _apply_spine_visibility(
            ax_top, 
            {"top": False,
             "right": False,
             "bottom": False,
             "left": True}
        )
    
        if xlabel is not None:
            ax_bottom.set_xlabel(xlabel)
    
        if ylabel is not None:
            bax.set_ylabel(ylabel)
    
        if title is not None:
            bax.axs[0].set_title(title)
    
        # brokenaxes creates diagonal marks automatically.
        # The following modifies the style of diagonal marks if accessible.
        diag_handles = getattr(bax, "diag_handles", [])
        diag_handles_left = [diag_handles[i] for i in [0, 2]]
        diag_handles_right = [diag_handles[i] for i in [1, 3]]
        
        for diag_handle in diag_handles_left:
            diag_handle.set_linewidth(default_break_diag_kwargs.get("linewidth", 1.2))
            diag_handle.set_color(default_break_diag_kwargs.get("color", "k"))
        
        for diag_handle in diag_handles_right:
            diag_handle.set_linewidth(0)
    
        return bax, long_df
    

def plot_ml_roc_curves(
    y_true,
    y_scores,
    positive_label=1,
    model_names=None,
    fold_ids=None,
    plot_folds=False,
    plot_pooled=True,
    plot_mean=True,
    mean_grid_size=101,
    title="ROC curve",
    figsize=(5, 5),
    ax=None,
    alpha_folds=0.25,
    linewidth_pooled=2.0, linecolor_pooled='b',
    linewidth_mean=2.0, linecolor_fold=None,
    zorder_fold=1,
    zorder_band=2,
    zorder_pooled=4,
    zorder_mean=5,
):
    """
    Plot ROC curve(s) for one or multiple ML models, optionally with fold-wise ROC curves.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True binary labels.

    y_scores : array-like, list/tuple, or dict
        Continuous prediction scores.

        Option 1: single model
            y_scores = array-like, shape (n_samples,)

        Option 2: multiple models
            y_scores = {
                "model_1": score_array_1,
                "model_2": score_array_2,
                ...
            }

        Option 3: list/tuple of score arrays
            y_scores = [score_array_1, score_array_2, ...]
            model_names should be provided or generated automatically.

    positive_label : int, str, or float
        Label treated as the positive class.

    model_names : list[str] or None
        Optional model names when y_scores is a list/tuple of score arrays.
        Ignored if y_scores is a dict.

    fold_ids : array-like or None, shape (n_samples,)
        Fold labels for each sample.
        If provided and plot_folds=True, fold-wise ROC curves are drawn.

    plot_folds : bool
        Whether to plot ROC curve for each fold.

    plot_pooled : bool
        Whether to plot pooled OOF ROC using all samples together.

    plot_mean : bool
        Whether to plot mean ROC across folds.
        Requires fold_ids.

    mean_grid_size : int
        Number of FPR grid points for interpolating fold-wise ROC curves.

    title : str
        Figure title.

    figsize : tuple
        Figure size.

    ax : matplotlib.axes.Axes or None
        Existing axis. If None, a new figure and axis are created.

    alpha_folds : float
        Transparency for fold-wise ROC curves.

    linewidth_pooled : float
        Line width for pooled ROC.

    linewidth_mean : float
        Line width for mean ROC.

    Returns
    -------
    result : dict
        {
            "fig": matplotlib Figure,
            "ax": matplotlib Axes,
            "roc_table": pandas DataFrame,
            "auc_table": pandas DataFrame,
            "mean_roc_table": pandas DataFrame or None
        }
    """
    y_true = np.asarray(y_true).ravel()

    if y_true.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")

    y_bin = (y_true == positive_label).astype(int)

    if len(np.unique(y_bin)) != 2:
        raise ValueError(
            "After binarization, y_true must contain both positive and negative classes. "
            f"Check positive_label={positive_label}."
        )

    # Normalize y_scores to dict
    if isinstance(y_scores, dict):
        score_dict = y_scores
    elif isinstance(y_scores, (list, tuple)) and len(y_scores) > 0 and np.asarray(y_scores[0]).ndim > 0:
        if model_names is None:
            model_names = [f"model_{i + 1}" for i in range(len(y_scores))]
        if len(model_names) != len(y_scores):
            raise ValueError("model_names must have the same length as y_scores.")
        score_dict = dict(zip(model_names, y_scores))
    else:
        score_dict = {"model": y_scores}

    if fold_ids is not None:
        fold_ids = np.asarray(fold_ids).ravel()
        if len(fold_ids) != len(y_bin):
            raise ValueError(
                f"len(fold_ids)={len(fold_ids)} does not match len(y_true)={len(y_bin)}."
            )

    if plot_mean and fold_ids is None:
        raise ValueError("plot_mean=True requires fold_ids.")

    if plot_folds and fold_ids is None:
        raise ValueError("plot_folds=True requires fold_ids.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    roc_rows = []
    auc_rows = []
    mean_rows = []

    mean_fpr_grid = np.linspace(0.0, 1.0, mean_grid_size)

    for model_name, score in score_dict.items():
        score = np.asarray(score).ravel()

        if len(score) != len(y_bin):
            raise ValueError(
                f"Score length mismatch for model '{model_name}': "
                f"len(score)={len(score)}, len(y_true)={len(y_bin)}."
            )

        # --------------------------------------------------
        # 1. Pooled ROC across all out-of-fold predictions
        # --------------------------------------------------
        pooled_auc = np.nan

        if plot_pooled:
            fpr, tpr, thresholds = roc_curve(y_bin, score)
            pooled_auc = roc_auc_score(y_bin, score)

            label = f"{model_name} pooled AUC={pooled_auc:.3f}"

            ax.plot(
                fpr,
                tpr,
                color=linecolor_pooled if not isinstance(linecolor_pooled, dict) else linecolor_pooled[model_name],
                linewidth=linewidth_pooled,
                label=label,
                zorder=zorder_pooled,
            )

            for fp, tp, th in zip(fpr, tpr, thresholds):
                roc_rows.append({
                    "model": model_name,
                    "curve_type": "pooled",
                    "fold": "pooled",
                    "fpr": fp,
                    "tpr": tp,
                    "threshold": th,
                    "auc": pooled_auc,
                })

            auc_rows.append({
                "model": model_name,
                "curve_type": "pooled",
                "fold": "pooled",
                "auc": pooled_auc,
                "n_samples": len(y_bin),
            })

        # --------------------------------------------------
        # 2. Fold-wise ROC curves
        # --------------------------------------------------
        fold_tprs_interp = []
        fold_aucs = []

        if fold_ids is not None:
            for fold in np.unique(fold_ids):
                idx = fold_ids == fold
                y_fold = y_bin[idx]
                score_fold = score[idx]

                # ROC is undefined if one class is absent in this fold
                if len(np.unique(y_fold)) < 2:
                    auc_rows.append({
                        "model": model_name,
                        "curve_type": "fold",
                        "fold": fold,
                        "auc": np.nan,
                        "n_samples": int(np.sum(idx)),
                        "note": "Skipped: only one class present in this fold.",
                    })
                    continue

                fpr_fold, tpr_fold, thresholds_fold = roc_curve(y_fold, score_fold)
                auc_fold = roc_auc_score(y_fold, score_fold)

                fold_aucs.append(auc_fold)

                tpr_interp = np.interp(mean_fpr_grid, fpr_fold, tpr_fold)
                tpr_interp[0] = 0.0
                tpr_interp[-1] = 1.0
                fold_tprs_interp.append(tpr_interp)

                if plot_folds:
                    if linecolor_fold is not None:
                        ax.plot(
                            fpr_fold,
                            tpr_fold,
                            alpha=alpha_folds,
                            linewidth=1.0, color=linecolor_fold,
                            zorder=zorder_fold,
                        )
                    else:
                        ax.plot(
                            fpr_fold,
                            tpr_fold,
                            alpha=alpha_folds,
                            linewidth=1.0,
                            zorder=zorder_fold,
                        )

                for fp, tp, th in zip(fpr_fold, tpr_fold, thresholds_fold):
                    roc_rows.append({
                        "model": model_name,
                        "curve_type": "fold",
                        "fold": fold,
                        "fpr": fp,
                        "tpr": tp,
                        "threshold": th,
                        "auc": auc_fold,
                    })

                auc_rows.append({
                    "model": model_name,
                    "curve_type": "fold",
                    "fold": fold,
                    "auc": auc_fold,
                    "n_samples": int(np.sum(idx)),
                })

        # --------------------------------------------------
        # 3. Mean ROC across folds
        # --------------------------------------------------
        if plot_mean and len(fold_tprs_interp) > 0:
            fold_tprs_interp = np.asarray(fold_tprs_interp)

            mean_tpr = np.mean(fold_tprs_interp, axis=0)
            std_tpr = np.std(fold_tprs_interp, axis=0, ddof=1) if fold_tprs_interp.shape[0] > 1 else np.zeros_like(mean_tpr)

            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs, ddof=1) if len(fold_aucs) > 1 else 0.0

            ax.plot(
                mean_fpr_grid,
                mean_tpr,
                linewidth=linewidth_mean,
                linestyle="--", 
                label=f"{model_name} mean AUC={mean_auc:.3f}±{std_auc:.3f}",
                zorder=zorder_mean,
            )

            tpr_upper = np.minimum(mean_tpr + std_tpr, 1.0)
            tpr_lower = np.maximum(mean_tpr - std_tpr, 0.0)

            ax.fill_between(
                mean_fpr_grid,
                tpr_lower,
                tpr_upper,
                alpha=0.10,
                zorder=zorder_band,
            )

            for fp, tp, lo, hi in zip(mean_fpr_grid, mean_tpr, tpr_lower, tpr_upper):
                mean_rows.append({
                    "model": model_name,
                    "fpr": fp,
                    "mean_tpr": tp,
                    "tpr_lower": lo,
                    "tpr_upper": hi,
                    "mean_auc": mean_auc,
                    "std_auc": std_auc,
                    "n_valid_folds": len(fold_aucs),
                })

            auc_rows.append({
                "model": model_name,
                "curve_type": "mean",
                "fold": "mean",
                "auc": mean_auc,
                "auc_std": std_auc,
                "n_valid_folds": len(fold_aucs),
            })

    ax.plot([0, 1], [0, 1], linestyle="--", color='gray', linewidth=1.0, label="Chance")
    ax.set_xlabel("1 - Specificity / False Positive Rate")
    ax.set_ylabel("Sensitivity / True Positive Rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=8)

    roc_table = pd.DataFrame(roc_rows)
    auc_table = pd.DataFrame(auc_rows)
    mean_roc_table = pd.DataFrame(mean_rows) if len(mean_rows) > 0 else None

    return {
        "fig": fig,
        "ax": ax,
        "roc_table": roc_table,
        "auc_table": auc_table,
        "mean_roc_table": mean_roc_table,
    }


def plot_confusion_matrix(
    y_true,
    y_pred=None,
    y_score=None,
    threshold=0.0,
    positive_label=1,
    negative_label=0,
    labels=None,
    display_labels=None,
    normalize=None,
    title="Confusion matrix",
    cmap="Blues",
    ax=None,
    figsize=(5, 5),
    values_format=None,
    show_colorbar=True,
    show_accuracy=True,

    # Cell text options
    show_values=True,
    cell_text_mode="auto",
    cell_text_kwargs=None,
    diagonal_text_kwargs=None,
    offdiagonal_text_kwargs=None,

    # Cell border options
    show_cell_borders=True,
    cell_border_color="black",
    cell_border_width=0.8,

    # Tick/label options
    xlabel="Predicted label",
    ylabel="True label",
    tick_label_rotation=0,

    # Optional custom callbacks
    cell_text_func=None,
    cell_style_func=None,
):
    """
    Plot a customizable confusion matrix.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True labels.

    y_pred : array-like or None
        Predicted class labels. If None, y_score must be provided.

    y_score : array-like or None
        Continuous prediction scores. Used only when y_pred is None.
        For binary classification:
            y_pred = positive_label if y_score >= threshold else negative_label

    threshold : float
        Decision threshold for y_score.

    positive_label, negative_label : int, str, or float
        Labels assigned from y_score when y_pred is not provided.

    labels : list or None
        Label order used to compute confusion matrix.

    display_labels : list or None
        Labels shown on the plot axes.

    normalize : {None, "true", "pred", "all"}
        Normalization mode passed to sklearn.metrics.confusion_matrix.

    title : str
        Plot title.

    cmap : str
        Matplotlib colormap.

    ax : matplotlib.axes.Axes or None
        Existing axis. If None, a new figure and axis are created.

    figsize : tuple
        Figure size.

    values_format : str or None
        Format for displayed values.
        Examples:
            "d" for counts
            ".2f" for normalized values

    show_colorbar : bool
        Whether to show colorbar.

    show_accuracy : bool
        Whether to include accuracy in title.

    show_values : bool
        Whether to show text inside cells.

    cell_text_mode : {"auto", "count", "normalized", "count_and_normalized"}
        How to display cell text.

        "auto":
            If normalize is None, show raw counts.
            If normalize is not None, show normalized values.

        "count":
            Show raw counts.

        "normalized":
            Show normalized values.

        "count_and_normalized":
            Show both raw counts and normalized values.

    cell_text_kwargs : dict or None
        Default text style for all cells.
        Example:
            {"fontsize": 12, "fontweight": "bold"}

    diagonal_text_kwargs : dict or None
        Extra/override text style for diagonal cells.

    offdiagonal_text_kwargs : dict or None
        Extra/override text style for off-diagonal cells.

    show_cell_borders : bool
        Whether to draw borders around cells.

    cell_border_color : str
        Cell border color.

    cell_border_width : float
        Cell border line width.

    xlabel, ylabel : str
        Axis labels.

    tick_label_rotation : float
        Rotation angle for x-axis tick labels.

    cell_text_func : callable or None
        Optional custom function for cell text.
        Signature:
            cell_text_func(i, j, count_value, display_value) -> str

    cell_style_func : callable or None
        Optional custom function for cell-specific style.
        Signature:
            cell_style_func(i, j, count_value, display_value) -> dict

        Example return:
            {"color": "white", "fontsize": 14, "fontweight": "bold"}

    Returns
    -------
    result : dict
        {
            "fig": matplotlib Figure,
            "ax": matplotlib Axes,
            "confusion_matrix": ndarray,
            "confusion_matrix_counts": ndarray,
            "confusion_table": pandas DataFrame,
            "accuracy": float,
            "y_pred": ndarray
        }
    """
    y_true = np.asarray(y_true).ravel()

    if y_pred is None:
        if y_score is None:
            raise ValueError("Either y_pred or y_score must be provided.")

        y_score = np.asarray(y_score).ravel()

        if len(y_score) != len(y_true):
            raise ValueError(
                f"len(y_score)={len(y_score)} does not match len(y_true)={len(y_true)}."
            )

        y_pred = np.where(y_score >= threshold, positive_label, negative_label)
    else:
        y_pred = np.asarray(y_pred).ravel()

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"len(y_pred)={len(y_pred)} does not match len(y_true)={len(y_true)}."
        )

    if labels is None:
        labels = list(np.unique(np.concatenate([y_true, y_pred])))

    if display_labels is None:
        display_labels = labels

    cm_counts = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize=None,
    )

    cm_display = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize=normalize,
    )

    acc = accuracy_score(y_true, y_pred)

    if values_format is None:
        values_format = ".2f" if normalize is not None else "d"

    if cell_text_mode not in ["auto", "count", "normalized", "count_and_normalized"]:
        raise ValueError(
            "cell_text_mode must be one of "
            "{'auto', 'count', 'normalized', 'count_and_normalized'}."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)

    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n_classes = len(labels)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=display_labels,
        yticklabels=display_labels,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=tick_label_rotation,
        ha="right" if tick_label_rotation != 0 else "center",
        rotation_mode="anchor",
    )

    if show_accuracy:
        ax.set_title(f"{title}\nAccuracy = {acc:.3f}")
    else:
        ax.set_title(title)

    # Draw cell borders using minor ticks
    if show_cell_borders:
        ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
        ax.grid(
            which="minor",
            color=cell_border_color,
            linestyle="-",
            linewidth=cell_border_width,
        )
        ax.tick_params(which="minor", bottom=False, left=False)

    # Default text styles
    base_text_kwargs = {
        "ha": "center",
        "va": "center",
        "fontsize": 12,
    }

    if cell_text_kwargs is not None:
        base_text_kwargs.update(cell_text_kwargs)

    diag_kwargs = diagonal_text_kwargs or {}
    offdiag_kwargs = offdiagonal_text_kwargs or {}

    # Automatic text color based on background intensity
    threshold_color = cm_display.max() / 2.0 if cm_display.size > 0 else 0.0

    if show_values:
        for i in range(n_classes):
            for j in range(n_classes):
                count_value = cm_counts[i, j]
                display_value = cm_display[i, j]

                if cell_text_func is not None:
                    text = cell_text_func(i, j, count_value, display_value)
                else:
                    if cell_text_mode == "auto":
                        if normalize is None:
                            text = format(count_value, "d")
                        else:
                            text = format(display_value, values_format)

                    elif cell_text_mode == "count":
                        text = format(count_value, "d")

                    elif cell_text_mode == "normalized":
                        text = format(display_value, values_format)

                    else:
                        text = (
                            f"{count_value:d}\n"
                            f"({display_value:.2f})"
                        )

                text_kwargs = dict(base_text_kwargs)

                # Default automatic text color unless user explicitly sets it
                if "color" not in text_kwargs:
                    text_kwargs["color"] = (
                        "white" if display_value > threshold_color else "black"
                    )

                if i == j:
                    text_kwargs.update(diag_kwargs)
                else:
                    text_kwargs.update(offdiag_kwargs)

                if cell_style_func is not None:
                    custom_kwargs = cell_style_func(
                        i,
                        j,
                        count_value,
                        display_value,
                    )
                    if custom_kwargs is not None:
                        text_kwargs.update(custom_kwargs)

                ax.text(j, i, text, **text_kwargs)

    confusion_table = pd.DataFrame(
        cm_display,
        index=[f"True_{x}" for x in display_labels],
        columns=[f"Pred_{x}" for x in display_labels],
    )

    return {
        "fig": fig,
        "ax": ax,
        "confusion_matrix": cm_display,
        "confusion_matrix_counts": cm_counts,
        "confusion_table": confusion_table,
        "accuracy": acc,
        "y_pred": y_pred,
    }


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


def plot_hist_with_vline(
    values,
    vline_value, histtype='stepfilled',
    bins=30,
    hist_kws=None,
    vline_kws=None,
    ax=None,
    figsize=(6, 4),
):
    """
    Plot a histogram for a one-dimensional array and add a vertical line.

    Parameters
    ----------
    values : array-like
        One-dimensional input data.
    vline_value : float
        The x-axis value at which to draw the vertical line.
    bins : int or sequence, default=30
        Number of histogram bins or bin edges.
    hist_kws : dict, optional
        Keyword arguments passed to ax.hist().
        Example: {"color": "gray", "alpha": 0.6, "edgecolor": "black"}
    vline_kws : dict, optional
        Keyword arguments passed to ax.axvline().
        Example: {"color": "red", "linestyle": "--", "linewidth": 2}
    ax : matplotlib.axes.Axes, optional
        Existing Axes object. If None, a new figure and axes are created.
    figsize : tuple, default=(6, 4)
        Figure size used when ax is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.
    ax : matplotlib.axes.Axes
        Matplotlib axes object.
    """
    values = np.asarray(values).ravel()

    # Remove NaN and infinite values
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("`values` contains no finite observations.")

    if hist_kws is None:
        hist_kws = {}

    if vline_kws is None:
        vline_kws = {}

    default_hist_kws = {
        "color": "gray",
        "alpha": 0.7,
        "histtype": histtype,
        "edgecolor": "black",
    }

    default_vline_kws = {
        "color": "red",
        "linestyle": "--",
        "linewidth": 2,
    }

    default_hist_kws.update(hist_kws)
    default_vline_kws.update(vline_kws)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.hist(values, bins=bins, **default_hist_kws)
    ax.axvline(vline_value, **default_vline_kws)

    return fig, ax


def wald_stat_for_term(gam, term_idx):
    """
    Compute an approximate Wald statistic for one pyGAM term.
    """
    slices = gam.terms.get_coef_indices(term_idx)
    
    beta = gam.coef_[slices]
    cov = gam.statistics_['cov'][np.ix_(slices, slices)]
    
    W = beta.T @ np.linalg.pinv(cov) @ beta
    
    # A rough choice: use the number of coefficients as df
    df = len(slices)
    p = chi2.sf(W, df)
    
    return W, p, beta, df


def permutation_test_within_sex_age_effect(
    X,
    y,
    target_sex=1,
    n_perm=1000,
    n_grid=100,
    statistic="variance",
    n_splines=4,
    lam=np.logspace(-3, 3, 15),
    use_obs_perm=False,
    random_state=123,
):
    """
    Permutation test for whether age predicts y within one sex group.
    
    Permuted statistics: 'variance', 'range', 'mean_abs_deviation', 'max_abs_deviation'
                          or 'r2'
    
    This tests whether the trajectory within the target sex is stronger
    than expected if age and y were unrelated within that sex.
    """
    rng = np.random.default_rng(random_state)

    sex = X[:, 1]
    mask = sex == target_sex

    X_group = X[mask].copy()
    y_group = y[mask].copy()

    age_grid = np.linspace(X_group[:, 0].min(), X_group[:, 0].max(), n_grid)
    sex_grid = np.full_like(age_grid, target_sex)

    X_grid = np.column_stack([
        age_grid,
        sex_grid
    ])

    def compute_group_statistic(
            X_fit, y_fit, 
            n_splines=4, gridsearch=True, lam=np.logspace(-3, 3, 15)
        ):
        if gridsearch:
            gam = LinearGAM(
                s(0, n_splines=n_splines)
            ).gridsearch(X_fit[:, [0]], y_fit, lam=lam, progress=False)
        else:
            gam = LinearGAM(
                s(0, n_splines=n_splines),
                lam=lam,
            ).fit(X_fit[:, [0]], y_fit)

        pred = gam.predict(age_grid[:, None])

        pred_centered = pred - pred.mean()

        if statistic == "variance":
            T = np.mean(pred_centered ** 2)
        
        elif statistic == "range":
            T = pred.max() - pred.min()
        
        elif statistic == "mean_abs_deviation":
            T = np.mean(np.abs(pred_centered))
        
        elif statistic == "max_abs_deviation":
            T = np.max(np.abs(pred_centered))
        
        elif statistic == "r2":
            sse = np.sum((y_fit - gam.predict(X_fit[:, [0]])) ** 2)
            sst = np.sum((y_fit - np.mean(y_fit)) ** 2)
            T = 1 - sse / sst
        else:
            raise ValueError("statistic must be 'variance' or 'range'")

        return T, pred, gam

    T_obs, pred_obs, gam = compute_group_statistic(X_group, y_group, n_splines)
    chosen_lam = gam.lam

    T_perm = []

    for _ in tqdm(
            range(n_perm),
            desc='Permutation analysis for significance of within-sex age trajectory',
    ):
        X_perm = X_group.copy()
        X_perm[:, 0] = rng.permutation(X_group[:, 0])

        if use_obs_perm:
            T_b, _, _ = compute_group_statistic(
                X_perm, y_group, 
                n_splines, gridsearch=False, lam=chosen_lam
            )
        else:
            T_b, _, _ = compute_group_statistic(X_perm, y_group, n_splines)
            
        T_perm.append(T_b)

    T_perm = np.asarray(T_perm)

    p_perm = (np.sum(T_perm >= T_obs) + 1) / (len(T_perm) + 1)

    return {
        "T_obs": T_obs,
        "T_perm": T_perm,
        "p_perm": p_perm,
        "pred_obs": pred_obs,
        "age_grid": age_grid,
    }

if __name__ == "__main__":
    # %% DATA PREPARATION
    #
    dir_res = '/public/home/qinshaozheng/fmri_task/ml_signature/res'

    # samples for ML training
    samples = np.loadtxt(os.path.join(dir_res, '..', 'roiData', 'samples_labels_in_one_sheet.csv'), delimiter=',')

    # results data of ML training/testing for classification btw rpsl and lkng
    fpath_mlTrain = os.path.join(
        dir_res, 'ml_res-model-multiple_task-classify_rpsl_lkng.pkl')

    with open(fpath_mlTrain, 'rb') as r:
        res_ml_rpsl_lkng = pickle.load(r)

    # results data of bootstrapped decoding weights and encoding activtaions
    fpath_boot_dec = os.path.join(
        dir_res, 'mlRes-decodingWeight_desc-sumTab_task-classify_rpsl_lkng.csv')
    fpath_boot_enc = os.path.join(
        dir_res, 'mlRes-encodingActivation_desc-sumTab_task-classify_rpsl_lkng.csv')

    mlRes_dict = {}
    mlRes_labl = ['boot_dec', 'boot_enc']

    for fpath, label in zip(
            [fpath_boot_dec, fpath_boot_enc],
            mlRes_labl,
    ):
        mlRes_dict[label] = pd.read_csv(fpath, sep=',')


    # parcellation image
    parcel_atlas = os.path.join(
        '/public/home/qinshaozheng/fmri_task/resources/masks_atlas',
        'Schaefer2018_200Parcels_7Networks_order.dlabel.nii',
    )

    # %% PLOT ROC CURVE OF ML MODEL (RPSL vs LKNG)
    #
    y = samples[:, -2]
    site_ids = samples[:, -1]

    score_dict = {
        "linear_svc": res_ml_rpsl_lkng['observed_results']["model_results"]["linear_svc"]["oof_score"],
        "logistic_l2": res_ml_rpsl_lkng['observed_results']["model_results"]["logistic_l2"]["oof_score"],
        "rbf_svc": res_ml_rpsl_lkng['observed_results']["model_results"]["rbf_svc"]["oof_score"],
    }

    linecolor_pooled = {
        "linear_svc": '#279957',
        "logistic_l2": '#280437',
        "rbf_svc": '#C9C61E',
    }

    roc_res = plot_ml_roc_curves(
        y_true=y,
        y_scores=res_ml_rpsl_lkng['observed_results']["model_results"]["logistic_l2"]["oof_score"],
        positive_label=1,
        fold_ids=site_ids,
        plot_folds=True, linecolor_fold='gray', alpha_folds=0.7,
        plot_pooled=True, linecolor_pooled='r', linewidth_pooled=2,
        plot_mean=False,
        figsize=(4,4),
        title=None
    )

    roc_res['ax'].spines['right'].set_visible(False)
    roc_res['ax'].spines['top'].set_visible(False)
    roc_res["auc_table"]

    # %% PLOT CONFUSION MATRIX
    #
    cm_res = plot_confusion_matrix(
        y_true=y,
        y_score=res_ml_rpsl_lkng['observed_results']["model_results"]["logistic_l2"]["oof_score"],
        threshold=0.5,
        positive_label=1,
        negative_label=0,
        labels=[0, 1],
        display_labels=["B", "A"],
        show_values=False,
        normalize=None,
        title=None,
        cmap='gray', 
        show_cell_borders=True, cell_border_color='w', cell_border_width=5,
    )

    cm_res['ax'].spines[:].set_visible(False)

    # %% PROJECT BOOTSTRAPPED DECODING/ENCODING RESULTS TO BRAIN SURFACE
    #
    # parcel values of decoding/encoding weights
    boot_dec_unthresh = mlRes_dict['boot_dec']['full_weight'].values
    boot_enc_unthresh = mlRes_dict['boot_enc']['final_encoding_weight'].values

    boot_dec_thresh = mlRes_dict['boot_dec']['thresholded_weight'].values
    boot_enc_thresh = mlRes_dict['boot_enc']['fdr_thresholded_weight'].values

    #
    for parcel_data, fname in zip(
            [boot_dec_unthresh, boot_enc_unthresh, boot_dec_thresh, boot_enc_thresh],
            ['dec_unthresh', 'enc_unthresh', 'dec_thresh', 'enc_thresh'],
    ):

        out_img, _ = parcel_to_brain(
            parcel_values=parcel_data[0:200],
            parcellation=parcel_atlas,
            parcel_ids=None,
            background=0,
            fill_value=np.nan,
        )

        out_img.to_filename(
            os.path.join(
                dir_res,
                f'mlRes-{fname}_desc-brainImg_task-classify_rpsl_lkng.dscalar.nii'
            )
        )

    # generate a subcortex medial wall for display
    mask_img, info = make_subcortex_display_mask_cifti(
        template_cifti=parcel_atlas,
        output_file="fsLR_32k_subcortex_black_overlay.dscalar.nii",
        cortex_value=np.nan,
        subcortex_value=1.0,
        include_cerebellum=True,
    )

    mask_img.to_filename(os.path.join(
        dir_res, 'subcortex_medial_wall.dscalar.nii'))

    # %% GENERATE .boder FILE FOR THRESHOLDED DECODING/ENCODING MAPS
    #
    # dcalar images for decoding/encoding weights
    boot_dec_dscalar = os.path.join(
        dir_res, 'mlRes-dec_thresh_desc-brainImg_task-classify_rpsl_lkng.dscalar.nii')
    boot_enc_dscalar = os.path.join(
        dir_res, 'mlRes-enc_thresh_desc-brainImg_task-classify_rpsl_lkng.dscalar.nii')

    # surface image
    surf_l = '/public/home/qinshaozheng/fmri_task/resources/HCP_WB_Tutorial_1.5_Pr_kN3mg/Q1-Q6_R440.L.very_inflated.32k_fs_LR.surf.gii'
    surf_r = '/public/home/qinshaozheng/fmri_task/resources/HCP_WB_Tutorial_1.5_Pr_kN3mg/Q1-Q6_R440.R.very_inflated.32k_fs_LR.surf.gii'

    # transform dscalar image to border file
    for dscalar, fname in zip(
            [boot_dec_dscalar, boot_enc_dscalar],
            ['dec_thresh', 'enc_thresh'],
    ):

        result = dscalar_to_border(
            dscalar=dscalar,
            left_surface=surf_l,
            right_surface=surf_r,
            out_prefix=os.path.join(
                dir_res, f'mlRes-{fname}_desc-brainImg_task-classify_rpsl_lkng'),
            mode="expr",
            expression="(x > 0 || x < 0)",
            verbose=False,
        )

    # %% PLOT PERMUTATION RESULTS OF ML MODEL TRAINING AND TESTING
    #
    # data
    model_name = list(res_ml_rpsl_lkng['model_permutation_results'].keys())

    df_perm_score = pd.DataFrame(
        {
            f'perm_score_{model_name[0]}': res_ml_rpsl_lkng['model_permutation_results'][model_name[0]]['permutation_scores'],
            f'perm_score_{model_name[1]}': res_ml_rpsl_lkng['model_permutation_results'][model_name[1]]['permutation_scores'],
            f'perm_score_{model_name[2]}': res_ml_rpsl_lkng['model_permutation_results'][model_name[2]]['permutation_scores'],
        }
    )

    df_obsr_table = res_ml_rpsl_lkng['observed_summary']
    ls_obsr_score = [
        df_obsr_table['roc_auc'][df_obsr_table['model_name'] == model_name[0]].values[0],
        df_obsr_table['roc_auc'][df_obsr_table['model_name'] == model_name[1]].values[0],
        df_obsr_table['roc_auc'][df_obsr_table['model_name'] == model_name[2]].values[0],
    ]

    # plot
    ax, longDf = boxplot_with_points(
        df_perm_score,
        ls_obsr_score,
        figsize=(3, 3.7),
        box_kwargs={
            # "palette": "Set2",
            "width": 0.35,
            "showfliers": False,
            "linewidth": 1.5,
            "linecolor": '#adadad',
            "boxprops": {'facecolor': '#adadad'},
            "medianprops": {'linewidth': 2, 'color': 'white'},
            "showcaps": False,
            "whis": 3,
            "fill": True,
        },
        point_kwargs={
            "s": 70,
            "color": "red",
            "marker": "o",
            "zorder": 20,
        },
        y_break=((0.46, 0.54), (0.67, 0.71)),
        show_line=False,
        xticklabels=['logistic L2', 'linear SVC', 'rbf SVC'],
        xlabel='ML models',
        ylabel="ROC-AUC",
        title="Permutation results of ML models",
        x_tick_rotation=None,
    )

    plt.show()

    # %% COMPUTE NETWORK DISTRIBUTION WITHIN BRAIN ACTIVATION OF EACH AGE GROUP
    # stats map filenames
    dir_glmres = '/public/home/qinshaozheng/fmri_task/brain_glm/res_glm/age_groups'
    age_range = [
        (6,8),   (8,9),   (9,10),  (10,11), (11,12), (12,13),
        (13,14), (14,15), (15,16), (16,17), (17,18), (18,19),
    ]

    stats_imgs_ls = [
        os.path.join(
            dir_glmres,
            f'brainAct-volume_stats-z_cont-rpsl-lkng_ageGroup-{age[0]}-{age[1]}_unthresh.nii',
        )
        for age in age_range
    ]

    stats_imgs_thresh = [math_img('img*(img>2.81)', img=stats_img) for stats_img in stats_imgs_ls]

    # stack stats images as array of shape (n_maps, n_vertex)
    gii_stats_ls = [mni152_to_fslr(img, fslr_density='32k') for img in stats_imgs_thresh]
    gii_lh_rh_ls = []
    for gii_tuple in gii_stats_ls:
        lh = gii_tuple[0].darrays[0].data
        rh = gii_tuple[1].darrays[0].data
        gii_lh_rh = np.concatenate([lh, rh])
    
        gii_lh_rh_ls.append(gii_lh_rh)
    
    gii_stack = np.vstack(gii_lh_rh_ls)

    # image names
    image_names = [
        'age 6-8',   'age 8-9',   'age 9-10',  'age 10-11', 'age 11-12', 'age 12-13',
        'age 13-14', 'age 14-15', 'age 15-16', 'age 16-17', 'age 17-18', 'age 18-19',
    ]

    # atlas
    yeo_atlas = os.path.join(
        '/public/home/qinshaozheng/fmri_task/resources/masks_atlas/yeo2011',
        'Yeo2011_7Networks_N1000.dlabel.nii',
    )

    network_names = {
        1: 'visual', 
        2: 'som/motor', 
        3: 'dAttn', 
        4: 'vAttn', 
        5: 'limbic', 
        6: 'control', 
        7: 'default'
    }

    # compute network distribution in each given map from age groups/bins
    network_dist = compute_network_contribution(
        brain_maps=gii_stack,
        atlas=yeo_atlas,
        map_names=image_names,
        network_names=network_names,
        positive_only=False,
        cal_relative_contribution=True,
    )

    # ---------------
    # plotting block
    # ---------------
    # convert long df to wide df for convinient visualization
    target_index = 'positive_weight_contribution_pct_global'
    long_df = network_dist['network_metrics'][['map', 'network_name', target_index]]
    col_order = long_df['map'].unique()
    row_order = long_df['network_name'].unique()

    wide_df = long_df.pivot_table(
        index='network_name',           # row index
        columns='map',                  # column index
        values=target_index,            # value column
        aggfunc='first'
    )

    wide_df = wide_df.reindex(index=row_order, columns=col_order)

    # plot network distibution
    df_plot_netDist = wide_df.copy()
    df_plot_netDist.columns = [
        col.replace("age ", "") for col in df_plot_netDist.columns
    ]

    network_colors = {
        "visual": "#781286",
        "som/motor": "#4682b4",
        "dAttn": "#4a9b3c",
        "vAttn": "#c43afa",
        "limbic": "#dcf8a4",
        "control": "#e69422",
        "default": "#cd3e4e",
    }

    fig, ax = plot_network_distributed_score(
        df=df_plot_netDist,
        network_order=list(network_names.values()),
        network_colors=network_colors,
        multiply_by=1,
        title="Network contribution to ER",
        xlabel="Age groups/bins",
        ylabel="Network contribution score",
        legend=False,
        figsize=(3.2, 3.5),
        rotation=90,
    )

    plt.xticks(fontsize=11)

    # %% COMPUTE NETWORK DISTRIBUTION WITHIN BRAIN DECODING/ENCODING MAPS
    #
    save_res = False

    # brain images for decoding/encoding weights
    boot_dec_dscalar = os.path.join(
        dir_res, 'mlRes-dec_thresh_desc-brainImg_task-classify_rpsl_lkng.dscalar.nii')
    boot_enc_dscalar = os.path.join(
        dir_res, 'mlRes-enc_thresh_desc-brainImg_task-classify_rpsl_lkng.dscalar.nii')

    data_maps = [_load_brain_like_data(fpath) for fpath in [boot_dec_dscalar, boot_enc_dscalar]]

    # atlas
    yeo_atlas = os.path.join(
        '/public/home/qinshaozheng/fmri_task/resources/masks_atlas/yeo2011',
        'Yeo2011_7Networks_N1000.dlabel.nii',
    )

    network_names = {
        1: 'visual', 
        2: 'som/motor', 
        3: 'dAttn', 
        4: 'vAttn', 
        5: 'limbic', 
        6: 'control', 
        7: 'default'
    }

    # compute network distribution in each given map
    # this step performed for positive and negative weights separately
    # to match the directions of our visualizations of these maps, switch postive and 
    # negative weights when necessary 
    res_network_dist = {}
    for brain_data, map_name in zip(
            data_maps,
            ['Decoding weights', 'Encoding activations']
    ):
        brain_data_sep = [brain_data * (brain_data>0), brain_data * (brain_data<0)]

        network_dist = compute_network_contribution(
            brain_maps=brain_data_sep,
            atlas=yeo_atlas,
            map_names=['negWeights', 'posWeights'],
            network_names=network_names,
            positive_only=False,
            cal_relative_contribution=False,
        )
    
        res_network_dist[map_name] = network_dist
    
    # save res into local
    if save_res:
        for map_name, res in res_network_dist.items():
            fname = 'decWeights' if 'Decoding' in map_name else 'encActivations'
        
            res['network_metrics'].to_csv(
                os.path.join(dir_res, f'mlRes-{fname}_desc-networkDist_task-classify_rpsl_lkng.csv'),
                index=False,
            )

    # %% PLOTTING NETWORK DISTRIBUTION
    #
    map_name = 'Encoding activations'
    netDist_index = 'dice_coefficient_binary_network'
    netDist_posWeights = res_network_dist[map_name]['network_metrics'][netDist_index].values[7:14]
    netDist_negWeights = res_network_dist[map_name]['network_metrics'][netDist_index].values[0:7]

    labels = network_names

    colors_left = ["#781286", "#4682b4", "#4a9b3c", "#c43afa", "#dcf8a4", "#e69422", "#cd3e4e"]
    colors_right = ["#781286", "#4682b4", "#4a9b3c", "#c43afa", "#dcf8a4", "#e69422", "#cd3e4e"]

    fig, ax = plot_bilateral_bar(
        netDist_posWeights[::-1],
        netDist_negWeights[::-1],
        xaxis_position="bottom",
        xlabel="cosine similarity",
        left_colors=colors_left[::-1],
        right_colors=colors_right[::-1],
        show_yticklabels=False,
        show_values=False,
        value_format="{:.2f}",
        value_text_kwargs=dict(fontsize=9),
        figsize=(4, 4.5),
        left_label=None, right_label=None,
    )

    xticks = [-0.6, -0.4, -0.2, 0, 0.2, 0.4]
    ax.set_xlim([-0.6, 0.4])
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x):g}" for x in xticks])
    ax.set_yticks([])
    ax.spines['left'].set_color('white')
    ax.spines['left'].set_linewidth(3)

    plt.show()

    # %% PLOTTING AGE-ML_SIGNATURE_RESPONSE TRAJECTORIES
    #
    # GAM results
    dir_res_data = '/public/home/qinshaozheng/fmri_task/ml_signature/res'

    df_gam = pd.read_csv(os.path.join(dir_res_data, 'ml_posthoc-pattern_expression-shortDF.csv'))

    fpath_gam_res = os.path.join(dir_res_data, 'GAMres-summary_ER-patternExpression_with_age.pkl')
    fpath_local_TP = os.path.join(dir_res_data, 'GAMres-turningPoints_localPermStats_ER-patternExpression_with_age.json')

    with open(fpath_gam_res, 'rb') as r:
        res_gam = pickle.load(r)

    with open(fpath_local_TP, 'rb') as r:
        res_localTP = json.load(r) 

    # trajectory setting
    SHOW_TRAJ_CI = True
    SHOW_RAW_DATA = True
    SHOW_TP = False
    SHOW_TP_VERTICAL_LINE = True
    SAVE_FIG = False

    # decoding expression: ['sig_res_rpsl', 'sig_res_lkng', 'sig_res_lkng_rpsl']
    # encoding expression: ['enc_exp_rpsl', 'enc_exp_lkng', 'enc_exp_lkng_rpsl']
    plot_cols = ['sig_res_lkng_rpsl']

    # -----------------------
    # plotting block
    # -----------------------
    # formal plotting
    n_cols = len(plot_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3), sharey=False)

    if n_cols == 1:
        axes = [axes]

    default_colors = ["#f28f33", "#1f77b4", "#2ca02c", "#9467bd"]
    sex_labels_str = ['male', 'female']

    for ax, col in zip(axes, plot_cols):

        # ------------------------------------------------------------
        # Bottom x-axis: model trajectory
        # ------------------------------------------------------------
        for si, slabel in enumerate(sex_labels_str):
            color = default_colors[si % len(default_colors)]

            traj = res_gam[col]["traj"][slabel]

            # trajectory line
            ax.plot(
                traj["age"],
                traj["mean"],
                color=color,
                linewidth=3,
                label=slabel,
            )

            # trajectory 95% CI
            if SHOW_TRAJ_CI:
                ax.fill_between(
                    traj["age"],
                    traj["ci_low"],
                    traj["ci_high"],
                    color=color,
                    alpha=0.30
                )
        
            if SHOW_RAW_DATA:
            
                slabel_bin = 0 if slabel=='male' else 1
                tmp = df_gam[df_gam["sex"] == slabel_bin]
    
                ax.scatter(
                    tmp["age"],
                    tmp[col],
                    color=color,
                    alpha=0.3,
                    s=10,
                    edgecolors='gray'
                )
        
            # turning points
            if SHOW_TP:
                tp_results = res_localTP[col][slabel]
                for tp in tp_results:
                    tp_age = tp["fixed_tp_age"]
                    tp_type = tp["type"]
    
                    p_for_sig = tp["perm_p_value"]
    
                    is_significant = (not np.isnan(p_for_sig)) and (p_for_sig < 0.05)
    
                    y_tp = np.interp(tp_age, traj["age"], traj["mean"])
    
                    # Marker shape by TP type
                    marker = "D" 
    
                    # Optional vertical line
                    if SHOW_TP_VERTICAL_LINE:
                        ax.axvline(
                            tp_age,
                            color=color,
                            linestyle="--",
                            linewidth=1.4,
                            alpha=0.7
                        )
    
                    # Significant -> filled marker
                    if is_significant:
                        ax.scatter(
                            tp_age,
                            y_tp,
                            marker=marker,
                            s=70,
                            color=color,
                            edgecolor="black",
                            linewidth=2,
                            zorder=6
                        )

        ax.set_title(col)
        ax.set_xlabel("Age")
        ax.spines.left.set_visible(True)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.yaxis.set_tick_params(which="major", left=False)

    axes[0].spines.left.set_visible(True)
    axes[0].legend(frameon=False)

    if 'sig_res' in plot_cols[0]:
        axes[0].set_ylabel("Decoding expression (Signature response)")
        figlabel = 'sigResponse'
    else:
        axes[0].set_ylabel("Encoding expression (Haufe transformation)")
        figlabel = 'encExpression'

    # save fig as svg into local
    if SAVE_FIG:
        dir_beh_data = '/public/home/qinshaozheng/fmri_task/ml_signature/res'
        fpath_fig = os.path.join(
            dir_beh_data, 
            f'GAMfig-trajectory_ER-{figlabel}_with_age.svg',
        )
        plt.savefig(fpath_fig, bbox_inches='tight', transparent=False)

    plt.tight_layout()
    plt.show()

    # %% PLOTTING
    #
    figsize = (3, 4.3)
    fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=figsize,
            sharex=False,
            constrained_layout=True
        )

    # 
    col = 'sig_res_lkng_rpsl'
    d_colors = ['#f2b780', '#b3d8f2']
    default_colors = ['#e58f39', '#2472ad']
    for si, slabel in enumerate(sex_labels_str):
        color = default_colors[si % len(default_colors)]
    
        # ---------- Upper subplot ----------
        traj = res_gam[col]["traj"][slabel]
    
        # trajectory line
        axes[0].plot(
            traj["age"],
            traj["mean"],
            color=color,
            linewidth=2,
            label=slabel,
        )

        # trajectory 95% CI
        axes[0].fill_between(
            traj["age"],
            traj["ci_low"],
            traj["ci_high"],
            color=color,
            alpha=0.30,
            linewidths=0,
        )
    
        # ---------- Upper + lower subplots ----------
        # scatter
        slabel_bin = 0 if slabel=='male' else 1
        tmp = df_gam[df_gam["sex"] == slabel_bin]
    
        for ax in axes:
            ax.scatter(
                tmp["age"],
                tmp[col],
                marker='.',
                color=d_colors[slabel_bin], linewidths=0.5,
                alpha=0.65,
                s=50,
                edgecolors='k'
            )

            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
    
        axes[0].set_xticks([])
        axes[0].spines.bottom.set_visible(False)
        axes[0].xaxis.set_tick_params(which="major", bottom=False)


        # ---------- Lower subplot ----------
        # linearly fitted line
        coef = np.polyfit(tmp["age"], tmp[col], deg=1)
        x_grid = np.linspace(np.min(tmp["age"]), np.max(tmp["age"]), 200)
        y_grid_fit = np.polyval(coef, x_grid)
    
        # Residuals
        y_fit_obs = np.polyval(coef, tmp["age"])
        residuals = tmp[col] - y_fit_obs
    
        # Degrees of freedom
        n = len(tmp[col])
        p = 2  # intercept + slope
        dof = n - p
    
        # Residual standard error
        residual_std_error = np.sqrt(np.sum(residuals ** 2) / dof)
    
        # Mean and variance of x
        x_mean = np.mean(tmp["age"])
        sxx = np.sum((tmp["age"] - x_mean) ** 2)
    
        # Standard error of the fitted mean response
        fit_se = residual_std_error * np.sqrt(
            1 / n + (x_grid - x_mean) ** 2 / sxx
        )
    
        # 95% CI
        # For small samples, t distribution is more appropriate than 1.96
        from scipy.stats import t
    
        t_value = t.ppf(0.975, dof)
    
        ci_lower = y_grid_fit - t_value * fit_se
        ci_upper = y_grid_fit + t_value * fit_se


        axes[1].plot(
            x_grid, 
            y_grid_fit,
            color=color,
            linewidth=2,
            label=slabel,
        )
    
        # Plot 95% CI band
        axes[1].fill_between(
            x_grid,
            ci_lower,
            ci_upper,
            color=color,
            alpha=0.3,
            linewidth=0
        )

    axes[1].set_xticks([6, 9, 12, 16, 19])
    axes[0].set_yticks([-0.2, 0, 0.2, 0.4, 0.5])
    axes[1].set_yticks([-0.2, 0, 0.2, 0.4, 0.5])
    axes[1].set_xlabel('Age')
    axes[0].set_ylabel('Signature response')
    axes[1].set_ylabel('Signature response')
    axes[1].legend(frameon=False)

    xmin, xmax = axes[0].get_xlim()
    ymin, ymax = axes[0].get_ylim()

    x_gap = (xmax - xmin) * 0.05
    y_gap = (ymax - ymin) * 0.05

    for ax in axes:
        ax.spines['bottom'].set_bounds(6, 19)
        ax.spines['left'].set_bounds(-0.2, 0.5)

    # %% TESTING WHETHER WITHIN SEX AGE-ML_SIGNATURE_RESPONSE TRAJECTORIES IS NON-RANDOM
    #
    stats_sex_traj = permutation_test_within_sex_age_effect(
        X=df_gam[['age_z', 'sex']].values,
        y=df_gam['sig_res_lkng_rpsl'].values,
        target_sex=0,
        n_perm=5000,
        n_grid=200,
        statistic="mean_abs_deviation",
        use_obs_perm=True,
        random_state=123,
    )

    # %% PLOTTING PERMUTATION RESULTS OF LAST STEP
    #
    stats_sex_traj_M = stats_sex_traj.copy()

    perm_vals = stats_sex_traj_M['T_perm']
    obsr_val  = stats_sex_traj_M['T_obs']

    fig, ax = plot_hist_with_vline(
        perm_vals,
        obsr_val,
        figsize=(4, 4),
        bins=30,
        hist_kws={
            "color": '#e58f39',
            "alpha": 0.6,
            "edgecolor": 'white',
            "linewidth": 0,
        },
        vline_kws={
            "color": "r",
            "linestyle": "--",
            "linewidth": 2.5,
            # "label": "Observed value",
        },
    )

    ax.set_yticks([])
    ax.set_xlabel(None)
    # ax.set_ylabel("Frequency")
    ax.yaxis.set_tick_params(which="major", left=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.legend()

    plt.show()

    # %%
