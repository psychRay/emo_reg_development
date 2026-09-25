#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 04:42:34 2025

@author: dingrui
"""

#%% 
import os
import warnings
warnings.filterwarnings('ignore', message='.*deprecated.*')

import pandas as pd
pd.set_option('display.max_columns', 10)
pd.set_option('expand_frame_repr', False)

import numpy as np
import pingouin as pg

import chardet as chd
import matplotlib.pyplot as plt

from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from numpy.random import default_rng
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from patsy import dmatrix, dmatrices, build_design_matrices
from scipy.stats import chi2, ttest_rel, norm, t, gaussian_kde

# from tqdm import tqdm
from itertools import combinations


#%% Functions

def lsdir(dir_path, keyword):
    file_ls = os.listdir(dir_path)
    return [file for file in file_ls if keyword in file]

def encoding_detect(filename):
    with open(filename, 'rb') as f:
        content = f.read(10000)
        res_chd = chd.detect(content)
        encoding= res_chd['encoding']
        return encoding

def wald_block(X, col_names):
    L = np.zeros((len(col_names), X.shape[1]))
    for i, name in enumerate(col_names):
        L[i, X.columns.get_loc(name)] = 1.0
    
    diff = np.asarray(L @ params)
    V    = np.asarray(L @ cov @ L.T)
    W    = float(diff.T @ np.linalg.inv(V) @ diff)
    p    = 1 - chi2.cdf(W, df=L.shape[0])
    
    return {'Wald_chi2': W, 'p_val': p, 'df_block': L.shape[0]}

def to_age_c(df, a, col_age='age'):
    """Map a raw age to the centered scale used in the model"""
    return a - df[col_age].mean()

def X_from_new(new_df, design_info):
    new_df = new_df.copy()
    #
    Xnew = build_design_matrices([design_info], new_df)[0]
    Xnew = pd.DataFrame(Xnew, columns=design_info.column_names, index=new_df.index)
    # align with fixed effect
    Xnew = Xnew.reindex(columns=fe_names, fill_value=0.0)
    return Xnew

def predict_curve(df_data, cond, gender, ages, fe_names, fe_params, fe_cov):
    """
    Predict marginal mean and 95% CI for a given (Condition, Gender) across ages.
    Returns a DataFrame with columns: age, mean, lower, upper, se.
    """
    new = pd.DataFrame({
        "cognition": cond,
        "gender": gender,
        "age_c": [to_age_c(df_data, a, col_age='age') for a in ages],
    })
    Xnew = X_from_new(new, design_info)
    Xn = Xnew.to_numpy()
    mu = (Xnew @ fe_params).to_numpy().ravel()
    var = np.einsum("ij,jk,ik->i", Xn, fe_cov.to_numpy(), Xn)  # Var = x Σ x^T
    se = np.sqrt(np.maximum(var, 0.0))
    return pd.DataFrame({
        "age": ages, "mean": mu,
        "lower": mu - 1.96 * se,
        "upper": mu + 1.96 * se,
        "se": se
    })

def contrast_diff(df_data, design_info, condA, condB, gender, ages, site_region, fe_params, fe_cov):
    """
    Compute (A - B) contrast along ages with CI and z/p (Wald normal approx).
    Returns a DataFrame with columns: age, diff, se, z, p, lower, upper.
    """
    newA = pd.DataFrame({
            "cognition": condA, 
            "gender": gender, 
            "age_c": [to_age_c(df_data, a) for a in ages],
            "site_region": site_region
            })
    newB = pd.DataFrame({
            "cognition": condB, 
            "gender": gender, 
            "age_c": [to_age_c(df_data, a) for a in ages],
            "site_region": site_region
            })
    XA = X_from_new(newA, design_info).to_numpy()
    XB = X_from_new(newB, design_info).to_numpy()
    L  = XA - XB
    est = (L @ fe_params.to_numpy()).ravel()
    var = np.einsum("ij,jk,ik->i", L, fe_cov.to_numpy(), L)
    se  = np.sqrt(np.maximum(var, 0.0))
    z   = np.divide(est, se, out=np.full_like(est, np.nan), where=se > 0)
    p   = np.where(np.isfinite(z), 2 * (1 - norm.cdf(np.abs(z))), np.nan)
    return pd.DataFrame({
        "age": ages, "diff": est, "se": se, "z": z, "p": p,
        "lower": est - 1.96 * se, "upper": est + 1.96 * se
    })

def plot_age_moderation_2(
        diff_F, diff_M,
        diff_F_label, diff_M_label,
        condA_label, condB_label, 
        title="Age moderation of condition effect"):
    """
    Plot High−Low contrast vs. age with 95% CI for two groups (e.g., Female/Male).
    diff_F / diff_M must have columns: 'age', 'diff', 'lower', 'upper'.
    """

    # ensure numeric arrays (avoid dtype issues in fill_between)
    xF = np.asarray(diff_F["age"],   dtype=float)
    mF = np.asarray(diff_F["diff"],  dtype=float)
    lF = np.asarray(diff_F["lower"], dtype=float)
    uF = np.asarray(diff_F["upper"], dtype=float)

    xM = np.asarray(diff_M["age"],   dtype=float)
    mM = np.asarray(diff_M["diff"],  dtype=float)
    lM = np.asarray(diff_M["lower"], dtype=float)
    uM = np.asarray(diff_M["upper"], dtype=float)

    fig, ax = plt.subplots()
    ax.plot(xF, mF, label=diff_F_label)
    ax.fill_between(xF, lF, uF, alpha=0.25)

    ax.plot(xM, mM, label=diff_M_label)
    ax.fill_between(xM, lM, uM, alpha=0.25)

    ax.axhline(0, linestyle="--", color='r', linewidth=1)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(f"{condA_label} vs. {condB_label} (estimated effect)")
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.show()

def johnson_neyman_pointwise(diff_df: pd.DataFrame, alpha: float = 0.05):
    """
    Compute pointwise Johnson–Neyman significance regions for a contrast curve.
    diff_df must have columns: 'age', 'diff', 'se'.
    Returns:
      intervals: list of (start_age, end_age) where |z| > z_{1-alpha/2}
      boundaries: list of boundary ages (roots where |diff|-crit*se crosses 0)
      crit: critical z value used
    """
    ages = np.asarray(diff_df["age"], dtype=float)
    diff = np.asarray(diff_df["diff"], dtype=float)
    se   = np.asarray(diff_df["se"], dtype=float)
    crit = norm.ppf(1 - alpha/2)

    # h(a) = |diff(a)| - crit * se(a); h>0 means significant
    h = np.abs(diff) - crit * se
    sig = h > 0

    intervals = []
    boundaries = []

    # scan for runs of True and locate boundary by linear interpolation on h
    i = 0
    n = len(ages)
    while i < n:
        if not sig[i]:
            i += 1
            continue
        # start of a significant run
        start_idx = i
        i += 1
        while i < n and sig[i]:
            i += 1
        end_idx = i - 1

        # find left boundary (if run does not start at first grid point)
        if start_idx > 0:
            a1, a2 = ages[start_idx-1], ages[start_idx]
            h1, h2 = h[start_idx-1], h[start_idx]
            # linear interpolation for h=0
            a_left = a1 + (0 - h1) * (a2 - a1) / (h2 - h1)
        else:
            a_left = ages[start_idx]

        # find right boundary
        if end_idx < n - 1:
            a1, a2 = ages[end_idx], ages[end_idx+1]
            h1, h2 = h[end_idx], h[end_idx+1]
            a_right = a1 + (0 - h1) * (a2 - a1) / (h2 - h1)
        else:
            a_right = ages[end_idx]

        intervals.append((float(a_left), float(a_right)))
        boundaries.extend([float(a_left), float(a_right)])

    return {"intervals": intervals, "boundaries": boundaries, "crit": float(crit)}

def build_L(df_data, condA, condB, gender, ages, design_info):
    """
    Build L for (condA − condB) at given gender and ages(range)
    """
    newA = pd.DataFrame({"cognition": condA, "gender": gender,
                         "age_c": [to_age_c(df_data, a) for a in ages]})
    newB = pd.DataFrame({"cognition": condB, "gender": gender,
                         "age_c": [to_age_c(df_data, a) for a in ages]})
    XA = X_from_new(newA, design_info).to_numpy()
    XB = X_from_new(newB, design_info).to_numpy()
    return XA - XB  # (m x p)

def johnson_neyman_simultaneous(L: np.ndarray,
                                fe_params: np.ndarray,
                                fe_cov: np.ndarray,
                                alpha: float = 0.05,
                                n_draws: int = 2000,
                                seed: int = 0):
    """
    Simultaneous J–N via parametric bootstrap (supremum of standardized deviations).
    Inputs:
      L:         (m x p) contrast matrix along the age grid (each row = L(age))
      fe_params: (p,   ) fixed-effect coefficients (beta-hat)
      fe_cov:    (p x p) covariance of fixed effects (Sigma-hat)
    Returns:
      dict with:
        z: pointwise z(a) = (L beta_hat)/se(a)
        c_alpha: simultaneous critical value (>= z_{1-alpha/2})
        mask: boolean array where |z| > c_alpha (simultaneous significant set)
    """
    # Pointwise pieces
    est = L @ fe_params                  # (m,)
    var = np.einsum("ij,jk,ik->i", L, fe_cov, L)  # (m,)
    se  = np.sqrt(np.maximum(var, 1e-12))
    z   = est / se

    # Parametric bootstrap of the max standardized fluctuation
    rng = default_rng(seed)
    # Draw MVN deviations of beta: (n_draws x p)
    dev = rng.multivariate_normal(mean=np.zeros_like(fe_params),
                                  cov=fe_cov,
                                  size=n_draws)
    # For each draw, standardize along grid: (n_draws x m)
    std_dev = (dev @ L.T) / se  # broadcasting: each row vector / se
    T = np.max(np.abs(std_dev), axis=1)  # sup |.| over ages
    c_alpha = float(np.quantile(T, 1 - alpha))

    mask = np.abs(z) > c_alpha
    return {"z": z, "se": se, "c_alpha": c_alpha, "mask": mask}

def plot_condition_boxplots(
    df: pd.DataFrame,
    condition_col: str = "Condition",
    score_col: str = "Score",
    subject_col: str = "Subject",
    # --- data handling ---
    condition_order: list | None = None,
    # --- between-subject grouping (e.g., Gender) ---
    group_col: str | None = None,
    group_order: list | None = None,
    # --- layout ---
    horizontal: bool = False,
    title: str = "Observed distribution per condition",
    show_counts: bool = True,
    # --- box appearance ---
    box_colors: dict | list | tuple | str | None = None,
    box_width: float = 0.2,
    showcaps: bool = False, showfliers: bool = True, 
    outlier_color: str | dict | None = "#444444",
    outlier_marker: str = "o",
    outlier_size: float = 4.0,
    # --- points ---
    add_points: bool = False,
    point_style: str = "swarm",          # "swarm" or "jitter"
    point_palette: dict | list | tuple | str | None = None,
    point_marker: str | dict = "o",
    point_size: float = 20.0,
    point_alpha: float = 0.7,
    point_edgecolor: str | dict | None = "white",
    point_edgewidth: float = 0.6,
    point_jitter: float = 0.12,
    point_jitter_main_axis: float = 0.0,
    swarm_min_dist_px: float = 6.0,
    point_seed: int | None = 42,
    points_layer: str = "front",         # "front" | "behind"  
    points_position: str = "overlay",    # "overlay" | "under" 
    points_under_pad: float = 0.06,      # dist to box
):
    
    """
    Boxplot visualization of condition-wise descriptive stats of data
    (also can add group factor for grouped visualization, e.g. gender)
    """
    
    rng = np.random.default_rng(point_seed)

    # prepaare data 
    tmp = df[[subject_col, condition_col, score_col]].copy()
    used_score_col = score_col

    if condition_order is None:
        if pd.api.types.is_categorical_dtype(df[condition_col]):
            order = list(df[condition_col].cat.categories)
        else:
            order = sorted(tmp[condition_col].dropna().unique().tolist())
    else:
        order = condition_order.copy()

    if group_col is not None:
        tmp = df[[subject_col, condition_col, group_col, score_col]].copy()
        if group_order is None:
            group_order = sorted(tmp[group_col].dropna().unique().tolist())
        else:
            group_order = [g for g in group_order if g in set(tmp[group_col])]
        if len(group_order) == 0:
            group_col = None

    def _color_for_group(g, palette, fallback=("#1f77b4", "#d62728", "#2ca02c", "#9467bd")):
        if isinstance(palette, dict):
            return palette.get(g, fallback[0])
        elif isinstance(palette, (list, tuple)):
            idx = group_order.index(g) % len(palette)
            return palette[idx]
        elif isinstance(palette, str):
            return palette
        else:
            idx = group_order.index(g) % len(fallback)
            return fallback[idx]

    def _box_color(g):
        pal = box_colors if box_colors is not None else None
        return _color_for_group(g, pal)

    def _point_color(g):
        pal = point_palette if point_palette is not None else (box_colors if box_colors is not None else None)
        return _color_for_group(g, pal)

    def _point_marker_for(g):
        return point_marker.get(g, "o") if isinstance(point_marker, dict) else point_marker

    def _outlier_color_for(g):
        if isinstance(outlier_color, dict):
            return outlier_color.get(g, "#444444")
        elif isinstance(outlier_color, str) or outlier_color is None:
            return outlier_color or "#444444"
        return "#444444"

    base_pos = np.arange(len(order), dtype=float)
    gap = 0.13

    # figure
    if horizontal:
        fig_h = max(4.8, 0.7 * len(order) + 2.0)
        fig, ax = plt.subplots(figsize=(10.5, fig_h))
        ax.set_title(title)
        ax.set_xlabel(score_col)
    else:
        fig_w = max(6, 0.7 * len(order) + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 6.0))
        ax.set_title(title)
        ax.set_ylabel(score_col)

    # for 'swarm' layout:
    def _axis_pixels_per_unit(ax):
        bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans)
        w_px, h_px = bbox.width, bbox.height
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        px_per_x = w_px / max(1e-9, (x1 - x0))
        px_per_y = h_px / max(1e-9, (y1 - y0))
        return px_per_x, px_per_y

    def _swarm_offsets(values, base, max_width=point_jitter, min_dist_px=swarm_min_dist_px):
        vals = np.asarray(values, dtype=float)
        order_idx = np.argsort(vals)
        offsets = np.zeros_like(vals, dtype=float)
        px_x, px_y = _axis_pixels_per_unit(ax)
        main_thr = min_dist_px / (px_x if horizontal else px_y)
        orth_thr = min_dist_px / (px_y if horizontal else px_x)
        step = orth_thr
        placed = []
        for idx in order_idx:
            v = vals[idx]
            k = 0
            best = 0.0
            while True:
                cand = ((-1)**k)*((k+1)//2)*step if k>0 else 0.0
                if abs(cand) > point_jitter:
                    cand = np.sign(cand) * point_jitter
                ok = True
                for (vp, op) in placed:
                    if abs(v - vp) < main_thr and abs(cand - op) < orth_thr:
                        ok = False; break
                if ok: best = cand; break
                k += 1; 
                if k > 200: best = np.clip(cand, -point_jitter, point_jitter); break
            offsets[idx] = best
            placed.append((v, best))
        return offsets + base

    # for points under boxes:
    half = box_width / 2.0
    under_shift = half + points_under_pad
    def _shift_under(center):
        return center - under_shift if horizontal else center - under_shift

    # for box plotting:
    def _draw_boxes(data_lists, positions, color, label, showcaps=True):
        flierprops = dict(marker=outlier_marker, markersize=outlier_size,
                          markerfacecolor=_outlier_color_for(label),
                          markeredgecolor=_outlier_color_for(label),
                          alpha=0.9, zorder=3)
        bp = ax.boxplot(
            data_lists, vert=not horizontal, positions=positions, widths=box_width,
            manage_ticks=False, patch_artist=True, 
            showcaps=showcaps,
            showfliers=showfliers, flierprops=flierprops
        )
        for box in bp["boxes"]:
            box.set_facecolor(color); box.set_alpha(0.7)
            box.set_edgecolor(color); box.set_zorder(2.0)
            box.set_linewidth(0)
        for med in bp["medians"]:
            med.set_color('k'); med.set_linewidth(2.0); med.set_zorder(2.2)
        for whisk in bp["whiskers"]:
            whisk.set_color(color); whisk.set_zorder(2.0); whisk.set_linewidth(2.0)
        for cap in bp["caps"]:
            cap.set_color(color); cap.set_zorder(2.0); cap.set_linewidth(2.0)
        return plt.Line2D([], [], color=color, marker="s", linestyle="None", label=str(label))

    # for adding data points:
    def _scatter_points(vals, pos_center, color, marker, label):
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0: return

        # optional main-axis jitter
        if point_jitter_main_axis and point_jitter_main_axis > 0:
            spread = np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)
            noise = (point_jitter_main_axis * (spread if np.isfinite(spread) and spread>0 else 1.0))
            vals = vals + rng.normal(0, noise, size=len(vals))

        # overlay or under
        center = _shift_under(pos_center) if points_position == "under" else pos_center
        if point_style.lower() == "swarm":
            pos = _swarm_offsets(vals, base=center, max_width=point_jitter, min_dist_px=swarm_min_dist_px)
        else:
            pos = center + rng.uniform(-point_jitter, point_jitter, size=len(vals))

        # front or behind
        z = 0.8 if points_layer == "behind" else 2.6

        # colors
        ec = (point_edgecolor.get(label, "white") if isinstance(point_edgecolor, dict)
              else (point_edgecolor if isinstance(point_edgecolor, str) else "white"))

        if horizontal:
            ax.scatter(vals, pos, s=point_size, alpha=point_alpha,
                       facecolors=color, edgecolors=ec, linewidths=point_edgewidth,
                       marker=marker, zorder=z)
        else:
            ax.scatter(pos, vals, s=point_size, alpha=point_alpha,
                       facecolors=color, edgecolors=ec, linewidths=point_edgewidth,
                       marker=marker, zorder=z)

    # Collect and plotting:
    legend_handles = []
    if group_col is None:
        data = [tmp.loc[tmp[condition_col]==cond, used_score_col].dropna().to_numpy() for cond in order]
        labels = [f"{cond} (n={len(d)})" if show_counts else str(cond) for cond, d in zip(order, data)]
        positions = base_pos

        # 'behind': add points then plot boxes, or plot boxes first then adding data points:
        if add_points and points_layer == "behind":
            for i, (cond, vals) in enumerate(zip(order, data)):
                color = _point_color("All")
                marker = _point_marker_for("All")
                _scatter_points(vals, positions[i], color, marker, label="All")

        handle = _draw_boxes(data, positions, color="#1f77b4", label="All", showcaps=showcaps)
        legend_handles.append(handle)

        if add_points and points_layer != "behind":
            for i, (cond, vals) in enumerate(zip(order, data)):
                color = _point_color("All")
                marker = _point_marker_for("All")
                _scatter_points(vals, positions[i], color, marker, label="All")

        if horizontal:
            ax.set_yticks(base_pos, labels=labels)
        else:
            ax.set_xticks(base_pos, labels=labels)

    else:
        data_by_group = {}
        for g in group_order:
            arrs = []
            for cond in order:
                vals = tmp.loc[(tmp[group_col]==g) & (tmp[condition_col]==cond), used_score_col].dropna().to_numpy()
                arrs.append(vals)
            data_by_group[g] = arrs

        for i, g in enumerate(group_order):
            color = _box_color(g)
            pos = base_pos + (i - (len(group_order)-1)/2.0) * (2*gap)

            if add_points and points_layer == "behind":
                pcolor = _point_color(g); pmarker = _point_marker_for(g)
                for j, vals in enumerate(data_by_group[g]):
                    _scatter_points(vals, pos_center=pos[j], color=pcolor, marker=pmarker, label=g)

            handle = _draw_boxes(data_by_group[g], pos, color=color, label=g, showcaps=showcaps)
            legend_handles.append(handle)

            if add_points and points_layer != "behind":
                pcolor = _point_color(g); pmarker = _point_marker_for(g)
                for j, vals in enumerate(data_by_group[g]):
                    _scatter_points(vals, pos_center=pos[j], color=pcolor, marker=pmarker, label=g)

        # tick labels with totals
        lab = [f"{c} (n={int(sum(len(data_by_group[g][k]) for g in group_order))})" if show_counts else str(c)
               for k, c in enumerate(order)]
        if horizontal:
            ax.set_yticks(base_pos, labels=lab)
        else:
            ax.set_xticks(base_pos, labels=lab)
        ax.legend(handles=legend_handles, title=group_col, loc="best")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    return fig

def build_diffs_by_A(df, 
                    id_col="id", 
                    a_col="A", b_col="B", y_col="Y",
                    a_levels=None, b_levels=None, 
                    group_col=None, group_levels=None):

    """
    Build a dataframe for difference between two levels of B factor at different levels
    of A factors --> A modulation of B effect on dependent variable
    
    Returns:
      long_dB: long-format df with columns [id, A, dB] where dB = Y_{a,B1} - Y_{a,B2}
      wide_dB: wide-format df indexed by id with columns = A levels
    """
    
    d = df.copy()
    if a_levels is None:
        a_levels = list(pd.unique(d[a_col]))
    if b_levels is None:
        b_levels = list(pd.unique(d[b_col]))
        
    d[a_col] = pd.Categorical(d[a_col], categories=list(a_levels), ordered=True)
    d[b_col] = pd.Categorical(d[b_col], categories=list(b_levels), ordered=True)

    subj_group = None
    if group_col is not None:
        gtbl = d[[id_col, group_col]].drop_duplicates()
        if group_levels is None:
            group_levels = list(pd.unique(gtbl[group_col]))
        gtbl[group_col] = pd.Categorical(gtbl[group_col], categories=list(group_levels), ordered=True)
        subj_group = gtbl.set_index(id_col)[group_col]

    piv = d.pivot_table(index=id_col, columns=[a_col, b_col], values=y_col, aggfunc="mean")
    piv = piv.dropna()
    if piv.empty:
        raise ValueError("No subject has complete cells for all (A,B).")
        
    # compute dB at each A: d_a = Y_{a,B1} - Y_{a,B2}
    dB_cols = []
    for a in a_levels:
        d_a = piv[(a, b_levels[0])] - piv[(a, b_levels[1])]
        dB_cols.append(d_a.rename(a))
    wide_dB = pd.concat(dB_cols, axis=1)
    long_dB = wide_dB.reset_index().melt(id_vars=id_col, var_name=a_col, value_name="dB").dropna()

    if subj_group is not None:
        subj_group = subj_group.reindex(wide_dB.index)
        
    return wide_dB, long_dB, subj_group


def within_subject_ci_matrix(Z, err_type='se', alpha=0.05):
    
    """
    Compute within-subject SE/SD of a group of data with corresponding confidence intervals
    -----
        Z: a dataframe with each column representing one condition/type of data
        err_type: 'se' or 'sd'
        alpha: the range out of confidence interval
    """
    
    X = np.asarray(Z)
    n, k = X.shape
    subj_mean = X.mean(axis=1, keepdims=True)
    grand_mean = X.mean()
    normalized = X - subj_mean + grand_mean
    sd_norm = normalized.std(axis=0, ddof=1)
    if err_type=='se':
        se_norm = sd_norm / np.sqrt(n)
        morey = np.sqrt(k / (k - 1.0))
        adj_se = se_norm * morey
        tcrit = t.ppf(1 - alpha/2, df=n-1)
        ci_half = tcrit * adj_se
    else:
        ci_half = sd_norm
    # means at each level:
    means = X.mean(axis=0)
    
    return means, ci_half, n, normalized

def _get_cycle_colors(n):
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if len(prop_cycle) >= n:
        return prop_cycle[:n]
    reps = (n + len(prop_cycle) - 1) // len(prop_cycle)
    return (prop_cycle * reps)[:n]

def plot_dB_vs_A(
    df, id_col="id", a_col="A", b_col="B", y_col="Y",
    a_levels=None, b_levels=None,
    # mean marker controls
    marker_size=8, marker_facecolor='none', marker_edgecolor='tab:orange', marker_edgewidth=1.8,
    # connection line controls
    conn_color='tab:orange', conn_width=1.8, conn_style='-',
    # error bar controls
    err_type='se', capsize=8, err_width=1.6, alpha_ci=0.05,
    # subject data points (dot) under the line
    show_points=True, data4points='raw', points_size=14, points_alpha=0.35, points_color='0.6',
    # jitter controls (horizontal)
    points_jitter=True, points_jitter_width=0.08, points_jitter_seed=0,
    fig_size=(5,7),
    # output
    save_path=None
):  
    """
    Plot within-subject difference btw two levels of factor B against factor A levels
    """
    
    # prepare data for fig 
    wide_dB, long_dB, _ = build_diffs_by_A(df, id_col, a_col, b_col, y_col, a_levels, b_levels)
    if a_levels is None:
        a_levels = list(wide_dB.columns)
    if b_levels is None:
        b_levels = list(pd.unique(df_data_test['trust_level']))
        
    x = np.arange(len(a_levels))
    means, ci_half, n, normalized = within_subject_ci_matrix(wide_dB.values, err_type, alpha=alpha_ci)
    
    fig, ax = plt.subplots(figsize=fig_size)

    # Gray subject points with dot marker '.' under the line
    if show_points:
        if data4points=='raw':
            data_points = wide_dB.values
        elif data4points=='normalized':
            data_points = normalized
        rng = np.random.default_rng(points_jitter_seed)
        for i in range(wide_dB.shape[0]):
            xs = x.copy().astype(float)
            if points_jitter:
                xs = xs + rng.uniform(-points_jitter_width, points_jitter_width, size=len(x))
            ax.scatter(xs, data_points[i,:],
                       s=points_size, alpha=points_alpha, zorder=1,
                       c=points_color, marker='o', linewidths=1, edgecolors='gray')

    # Mean line + markers + error bars (with caps), above the points
    ax.errorbar(
        x, means, yerr=ci_half,
        marker='o', markersize=marker_size,
        markerfacecolor=marker_facecolor,
        markeredgecolor=marker_edgecolor, markeredgewidth=marker_edgewidth,
        linestyle=conn_style, linewidth=conn_width, color=conn_color,
        capsize=capsize, elinewidth=err_width,
        zorder=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in a_levels])
    ax.set_xlabel(a_col)
    ax.set_ylabel(f"effect of '{b_levels[0]} vs {b_levels[1]}' on {y_col}")
    ax.set_title(f"{a_col} moderation of effect of '{b_levels[0]} vs {b_levels[1]}'")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return save_path
    
    plt.show()
    return (means, ci_half, n)

def paired_diff_stats(x, y, alpha=0.05, n_boot=5000, random_state=0):
    
    """
    Compute CI of within-subject paired differences using bootstrapping method 
    """
    
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size == 0:
        return np.nan, (np.nan, np.nan), np.nan, 0
    diffs = x - y
    n = diffs.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha/2, 1-alpha/2])
    t, p = ttest_rel(x, y, nan_policy='omit')
    return float(diffs.mean()), (float(lo), float(hi)), float(p), int(n)

def holm_correction(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty_like(pvals)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        adj = max(adj, prev)
        adjusted[idx] = min(adj, 1.0)
        prev = adjusted[idx]
    return np.clip(adjusted, 0, 1)

def _half_violin_with_median(
    ax, x_center, values, side="+", *,
    width=0.8, facecolor=None, alpha=0.35, edgecolor=None, linewidth=1.0,
    show_median=True, median_linestyle='--', median_linewidth=2.0, median_color=None,
    n_points=200, zorder=2
):
    """
    Draw ONE half of a violin from x_center to either right ('+') or left ('-').
    The median line length is computed from KDE half-width at the sample median.
    Returns a dict with the median y and the drawn median x-extent.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    result = {"median": np.nan, "x0": np.nan, "x1": np.nan}

    if vals.size == 0:
        return result

    if vals.size < 2:
        # Not enough for KDE; draw a tiny tick around the single value
        med = float(np.median(vals))
        result.update(median=med, x0=x_center, x1=x_center)
        if show_median:
            ax.plot([x_center-0.02, x_center+0.02], [med, med],
                    color=median_color or edgecolor or facecolor, 
                    linestyle=median_linestyle, linewidth=median_linewidth, 
                    zorder=zorder+1)
        return result

    kde = gaussian_kde(vals)
    y_grid = np.linspace(np.min(vals), np.max(vals), n_points)
    dens = kde(y_grid)
    # Normalize density so the max half-width equals width/2
    scale = (width/2) / np.max(dens) if np.max(dens) > 0 else 0.0
    w = dens * scale

    # Coordinates for the filled half-violin
    if side == "+":
        x_fill = x_center + w
    else:
        x_fill = x_center - w

    ax.fill_betweenx(
        y_grid, x_center, x_fill,
        facecolor=facecolor, alpha=alpha, edgecolor=edgecolor or facecolor,
        linewidth=linewidth, zorder=zorder
    )

    # Median line from center to the half-width at the median
    med = float(np.median(vals))
    w_med = float(kde(med) * scale)  # half-width at the median via KDE
    if show_median:
        if side == "+":
            x0, x1 = x_center, x_center + w_med
        else:
            x0, x1 = x_center - w_med, x_center
        ax.plot([x0, x1], [med, med],
                color=median_color or edgecolor or facecolor,
                linestyle=median_linestyle, linewidth=median_linewidth, 
                zorder=zorder+1)
        result.update(median=med, x0=x0, x1=x1)
    else:
        result.update(median=med, x0=np.nan, x1=np.nan)

    return result


def draw_split_or_group_violins(
    ax, data_by_group, x_pos, group_order, colors, *,
    violin_width=0.8, violin_alpha=0.35, edgecolor=None, linewidth=1.0,
    show_median=True, median_linewidth=2.0, median_colors=None,
    n_points=200, zorder=2
):
    """
    Draw violins for one A level at position x_pos.
    - If len(group_order)==2 -> split violin (left=group_order[0], right=group_order[1]).
    - If len(group_order)==1 -> full violin (two halves from same data).
    - If len(group_order)>2 -> side-by-side skinny violins.

    Returns a dict {group_label: {"median": y, "x0": left_x, "x1": right_x}}.
    """
    results = {}
    ng = len(group_order)
    if ng == 0:
        return results

    # Helper to resolve median color
    def _mcol(g):
        if median_colors is not None and g in median_colors and median_colors[g] is not None:
            return median_colors[g]
        return colors[g]

    if ng == 1:
        g = group_order[0]
        vals = data_by_group[g]
        # left half
        resL = _half_violin_with_median(
            ax, x_pos, vals, side="-",
            width=violin_width, facecolor=colors[g], alpha=violin_alpha,
            edgecolor=edgecolor, linewidth=linewidth,
            show_median=show_median, median_linewidth=median_linewidth,
            median_color=_mcol(g), n_points=n_points, zorder=zorder
        )
        # right half
        resR = _half_violin_with_median(
            ax, x_pos, vals, side="+",
            width=violin_width, facecolor=colors[g], alpha=violin_alpha,
            edgecolor=edgecolor, linewidth=linewidth,
            show_median=show_median, median_linewidth=median_linewidth,
            median_color=_mcol(g), n_points=n_points, zorder=zorder
        )
        results[g] = {
            "median": resL["median"],
            "x0": resL["x0"], "x1": resR["x1"]
        }

    elif ng == 2:
        gL, gR = group_order[0], group_order[1]
        # left group -> left half
        resL = _half_violin_with_median(
            ax, x_pos, data_by_group[gL], side="-",
            width=violin_width, facecolor=colors[gL], alpha=violin_alpha,
            edgecolor=edgecolor, linewidth=linewidth,
            show_median=show_median, median_linewidth=median_linewidth,
            median_color=_mcol(gL), n_points=n_points, zorder=zorder
        )
        # right group -> right half
        resR = _half_violin_with_median(
            ax, x_pos, data_by_group[gR], side="+",
            width=violin_width, facecolor=colors[gR], alpha=violin_alpha,
            edgecolor=edgecolor, linewidth=linewidth,
            show_median=show_median, median_linewidth=median_linewidth,
            median_color=_mcol(gR), n_points=n_points, zorder=zorder
        )
        results[gL] = {"median": resL["median"], "x0": resL["x0"], "x1": resL["x1"]}
        results[gR] = {"median": resR["median"], "x0": resR["x0"], "x1": resR["x1"]}

    else:
        # >2 groups: draw skinny side-by-side violins
        offsets = np.linspace(-0.35, 0.35, ng)
        for k, g in enumerate(group_order):
            xg = x_pos + offsets[k]
            # full skinny violin for this group (two halves)
            resL = _half_violin_with_median(
                ax, xg, data_by_group[g], side="-",
                width=0.5, facecolor=colors[g], alpha=violin_alpha,
                edgecolor=edgecolor, linewidth=linewidth,
                show_median=show_median, median_linewidth=median_linewidth,
                median_color=_mcol(g), n_points=n_points, zorder=zorder
            )
            resR = _half_violin_with_median(
                ax, xg, data_by_group[g], side="+",
                width=0.5, facecolor=colors[g], alpha=violin_alpha,
                edgecolor=edgecolor, linewidth=linewidth,
                show_median=show_median, median_linewidth=median_linewidth,
                median_color=_mcol(g), n_points=n_points, zorder=zorder
            )
            results[g] = {"median": resL["median"], "x0": resL["x0"], "x1": resR["x1"]}

    return results

def draw_violin_medians(
    ax,
    wide_dB,                 # DataFrame: subjects x A-level columns (values are dB)
    subj_group,              # pd.Series (aligned to wide_dB.index) with group label per subject (or all None)
    a_levels,                # list of A levels in plotting order
    x_positions,             # numeric x for each A level (e.g., np.arange(len(a_levels)))
    group_levels_use,        # list of group labels in plotting order (len=1 -> single; 2 -> split; >2 -> side-by-side)
    color_map,               # dict {group_label -> color}; use {None: color} if no grouping
    violin_width=0.8,        # the width used to draw violins in your left panel
    # median line style & placement
    show_median=True,
    median_linestyle='--',
    median_linewidth=2.0,
    median_width_frac=0.45,  # single-group: half length = median_width_frac * violin_width
    median_half_shift=0.15,  # split (2 groups): center shift (in units of violin_width)
    median_half_len=0.12,    # split (2 groups): half length (in units of violin_width)
    median_colors=None,      # optional dict overriding color_map for medians, e.g. {'Male':'#1f77b4', 'Female':'#ff7f0e'}
    side_by_side_offsets=None # for >2 groups: optional array of x-offsets; default uses np.linspace(-0.35, 0.35, ng)
):
    """
    Draw median lines inside split/regular violins on `ax`.

    Behavior:
      - Single group: one horizontal line across the whole violin (centered at x_positions[j]).
      - Two groups (split): two short lines, one in each half, shifted away from center.
      - >2 groups: side-by-side skinny violins => one short line per group at its offset.

    Notes:
      - Colors: median_colors has priority; otherwise fall back to color_map.
      - zorder is set to 4 so medians sit above the filled violin.
    """
    if not show_median:
        return

    import numpy as np

    for j, a in enumerate(a_levels):
        # Gather data by group at this A level
        data_by_group = {}
        for g in group_levels_use:
            mask = (subj_group.values == g)
            vals = wide_dB.loc[mask, a].values
            if np.sum(mask) == 0 or vals.size == 0:
                continue
            data_by_group[g] = vals

        order_here = [g for g in group_levels_use if g in data_by_group]
        ng = len(order_here)
        x_center = x_positions[j]

        # color resolver
        def _mcol(g):
            if median_colors is not None and g in median_colors and median_colors[g] is not None:
                return median_colors[g]
            return color_map[g]

        if ng == 1:
            # Single-group: draw one horizontal line across the violin
            g = order_here[0]
            med = np.median(data_by_group[g])
            half = median_width_frac * violin_width
            ax.plot([x_center - half, x_center + half], [med, med],
                    linestyle=median_linestyle,
                    linewidth=median_linewidth, color=_mcol(g), zorder=4)

        elif ng == 2:
            # Split violin: draw two short lines, one per half
            gL, gR = order_here[0], order_here[1]
            medL = np.median(data_by_group[gL])
            medR = np.median(data_by_group[gR])

            # left half center
            cL = x_center - median_half_shift * violin_width
            ax.plot([cL - median_half_len*violin_width, cL + median_half_len*violin_width],
                    [medL, medL], 
                    linestyle=median_linestyle,
                    linewidth=median_linewidth, color=_mcol(gL), zorder=4)

            # right half center
            cR = x_center + median_half_shift * violin_width
            ax.plot([cR - median_half_len*violin_width, cR + median_half_len*violin_width],
                    [medR, medR], 
                    linestyle=median_linestyle,
                    linewidth=median_linewidth, color=_mcol(gR), zorder=4)

        else:
            # >2 groups: side-by-side configuration
            if (side_by_side_offsets is None) or (len(side_by_side_offsets) != ng):
                offsets = np.linspace(-0.35, 0.35, ng)
            else:
                offsets = side_by_side_offsets
            for k, g in enumerate(order_here):
                med = np.median(data_by_group[g])
                xg = x_center + offsets[k]
                half = 0.18  # pair with your skinny violin width (~0.5); tweak to fit your layout
                ax.plot([xg - half, xg + half], [med, med],
                        linestyle=median_linestyle,
                        linewidth=median_linewidth, color=_mcol(g), zorder=4)

def plot_ga_diffs(
    df, id_col="id", a_col="A", b_col="B", y_col="Y",
    a_levels=None, b_levels=None,
    group_col=None, group_levels=None, group_colors=None,
    data4points='normalized', point_marker='.', point_size=14, point_alpha=0.35,
    jitter=True, jitter_width=0.08, jitter_seed=0,
    connect_means=True, connect_linewidth=1.6, connect_linestyle='-',
    alpha=0.05, n_boot=5000, mcc='none', annotate_p=False, ygap=0.18,
    figsize=(10.5, 5.8), 
    return_CI=False,
    save_path="/mnt/data/dB_vs_A_GA_grouped.png"
):
    
    # prepare differece data for subsequent visualization
    wide_dB, long_dB, subj_group = build_diffs_by_A(
        df, id_col, a_col, b_col, y_col, 
        a_levels, b_levels, 
        group_col, group_levels
    )
    
    _, _, _, normalized = within_subject_ci_matrix(wide_dB.values)
    
    if a_levels is None:
        a_levels = list(wide_dB.columns)
    if b_levels is None:
        b_levels = list(pd.unique(df_data_test['trust_level']))
    
    x = np.arange(len(a_levels))

    if group_col is None:
        group_levels_use = [None]
        colors = _get_cycle_colors(1)
        color_map = {None: colors[0]}
        subj_group = pd.Series(index=wide_dB.index, data=[None]*wide_dB.shape[0])
    else:
        group_levels_use = list(group_levels)
        if group_colors is None:
            colors = _get_cycle_colors(len(group_levels_use))
            color_map = {g: colors[i] for i, g in enumerate(group_levels_use)}
        else:
            color_map = group_colors
    # -----------------------------------------------------------------------------------------------
    # formal visualization of Gardner–Altman plot
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 1.2], wspace=0.35)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    # subplot1 for raw data(points) of paired difference
    if data4points=='raw':
        data_points = wide_dB.values
    elif data4points=='normalized':
        data_points = normalized
    rng = np.random.default_rng(jitter_seed)
    for j, a in enumerate(a_levels):
        for g in group_levels_use:
            mask = (subj_group.values == g)
            vals = data_points[mask][:, j]
            if vals.size == 0: 
                continue
            xs = np.full(vals.size, x[j], dtype=float)
            if jitter:
                xs = xs + rng.uniform(-jitter_width, jitter_width, size=vals.size)
            axL.scatter(xs, vals, s=point_size, alpha=point_alpha,
                        marker=point_marker, linewidths=0, edgecolors='none', zorder=1,
                        c=color_map[g])

    if connect_means:
        for g in group_levels_use:
            mask = (subj_group.values == g)
            if np.sum(mask) == 0: 
                continue
            means_g = [wide_dB.loc[mask, a].mean() for a in a_levels]
            axL.plot(x, means_g, linestyle=connect_linestyle, linewidth=connect_linewidth,
                     marker='o', markersize=8, markerfacecolor='white', markeredgecolor='k',
                     color=color_map[g], zorder=3)

    axL.set_xticks(x)
    axL.set_xticklabels([str(a) for a in a_levels])
    axL.set_xlabel(a_col)
    axL.set_ylabel(f"'{b_levels[0]} - {b_levels[1]}' on {y_col}")
    axL.spines['top'].set_visible(False)
    axL.spines['right'].set_visible(False)

    # statistics of paired differences 
    pairs = list(combinations(range(len(a_levels)), 2))
    m = len(pairs) * len(group_levels_use)
    alpha_ci = alpha
    if mcc == 'bonferroni' and m > 0:
        alpha_ci = alpha / m
    
    rows = []
    p_collect = []
    for pi, (i, j) in enumerate(pairs):
        a_i, a_j = a_levels[i], a_levels[j]
        for gi, g in enumerate(group_levels_use):
            mask = (subj_group.values == g)
            xvec = wide_dB.loc[mask, a_i].values
            yvec = wide_dB.loc[mask, a_j].values
            md, (lo, hi), p, n = paired_diff_stats(
                xvec, yvec, 
                alpha=alpha_ci, n_boot=n_boot, 
                random_state=1000 + pi*10 + gi
            )
            label = f"{a_i} − {a_j} (n={n})" if g is None else f"{a_i} − {a_j} [{g}] (n={n})"
            rows.append({"pair": (a_i, a_j), "group": g, "md": md, "lo": lo, "hi": hi, "label": label, "color": color_map[g]})
            p_collect.append(p)

    p_corr = None
    if mcc == 'holm' and len(p_collect) > 0:
        p_corr = holm_correction(p_collect)
    
    # subplot2 for effect size of paired differences (95% CI)
    y_positions = []
    for idx, row in enumerate(rows):
        pair_index = pairs.index((a_levels.index(row["pair"][0]), a_levels.index(row["pair"][1])))
        if len(group_levels_use) == 1:
            y = pair_index
        else:
            g_idx = group_levels_use.index(row["group"])
            offset = (g_idx - (len(group_levels_use)-1)/2) * ygap
            y = pair_index + offset
        axR.errorbar(row["md"], y,
                     xerr=[[row["md"] - row["lo"]], [row["hi"] - row["md"]]],
                     fmt='o', markersize=5, elinewidth=1.6, capsize=0,
                     color=row["color"], zorder=2)
        y_positions.append(y)

    axR.axvline(0, linestyle='--', linewidth=1.0, color='gray', alpha=0.65, zorder=0)
    axR.set_yticks(range(len(pairs)))
    
    base_labels = [f"{a_levels[i]} − {a_levels[j]}" for (i, j) in pairs]
    axR.set_yticklabels(base_labels)
    
    label_x = 'Effect size of paired difference(95% CI)'
    # if mcc == 'bonferroni':
    #     label_x += f"[alpha_adj={alpha_ci:.3f}]"
    axR.set_xlabel(label_x)
    axR.set_title(f"Pairwise differences across {a_col} levels", fontsize=9)
    # axR.grid(True, axis='x', linestyle=':', linewidth=0.6, alpha=0.6)
    axR.spines['top'].set_visible(False)
    axR.spines['right'].set_visible(False)

    if mcc == 'holm' and p_corr is not None and len(p_corr) == len(rows):
        for idx, (y, row) in enumerate(zip(y_positions, rows)):
            axR.text(row["md"], y + 0.06, f"p={p_corr[idx]:.3f}", fontsize=9, ha='center', va='bottom',
                     color=row["color"], alpha=0.9)
    # -----------------------------------------------------------------------------------------------
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return save_path
    plt.show()
    
    if return_CI:
        return rows
    
def plot_ga_diffs_violin(
    df, id_col="id", a_col="A", b_col="B", y_col="Y",
    a_levels=None, b_levels=None,
    group_col=None, group_levels=None, group_colors=None,
    # Left panel (violins) options
    violin_width=0.8, violin_alpha=0.35, violin_edgecolor=None, violin_linewidth=1.0,
    draw_median=True, median_linewidth=2.0, median_colors=None,
    # connection lines/markers of group means
    connect_means=True, connect_linewidth=1.8, connect_linestyle='-',
    show_mean_markers=True, mean_markersize=7, mean_markerface='none', mean_markeredgewidth=1.6,
    # Right panel (pairwise differences) options
    alpha=0.05, n_boot=5000, mcc='none',  # 'none' | 'bonferroni' | 'holm-ci'
    ygap=0.18, annotate_p=False,  # annotate_p currently unused for 'holm-ci'; can be extended
    shade_rows=True, shade_colors=('0.90', '0.80'), band_pad=0.2,
    # layout & export
    figsize=(11, 6.0), save_path="/mnt/data/dB_vs_A_GA_splitviolin.png"
):
    
    """
    Left panel: split violin per A level (if exactly 2 groups) or regular violin/side-by-side for other cases.
    Right panel: all pairwise Aj−Ai differences (paired) with CI; CI can be MCC-adjusted.
    """
    
    # prepare differece data for subsequent visualization
    wide_dB, long_dB, subj_group = build_diffs_by_A(
        df, id_col, a_col, b_col, y_col, 
        a_levels, b_levels, 
        group_col, group_levels
    )
    
    if a_levels is None:
        a_levels = list(wide_dB.columns)
    if b_levels is None:
        b_levels = list(pd.unique(df_data_test['trust_level']))
        
    x = np.arange(len(a_levels))
    
    # Colors
    if group_col is None:
        group_levels_use = [None]
        colors_list = _get_cycle_colors(1)
        color_map = {None: colors_list[0]}
        subj_group = pd.Series(index=wide_dB.index, data=[None]*wide_dB.shape[0])
    else:
        group_levels_use = list(group_levels)
        if group_colors is None:
            colors_list = _get_cycle_colors(len(group_levels_use))
            color_map = {g: colors_list[i] for i, g in enumerate(group_levels_use)}
        else:
            color_map = group_colors

    # Figure & axes
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.2, 1], wspace=0.25)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # -------- LEFT: split/regular violins per A --------
    for j, a in enumerate(a_levels):
        data_by_group = {}
        for g in group_levels_use:
            mask = (subj_group.values == g)
            vals = wide_dB.loc[mask, a].values
            if np.sum(mask) == 0 or vals.size == 0:
                continue
            data_by_group[g] = vals
        order_here = [g for g in group_levels_use if g in data_by_group]
        _ = draw_split_or_group_violins(
            axL, data_by_group, x_pos=x[j], group_order=order_here, 
            violin_width=violin_width, violin_alpha=violin_alpha,
            colors=color_map, edgecolor=violin_edgecolor, linewidth=violin_linewidth,
            show_median=draw_median, median_linewidth=median_linewidth, median_colors=median_colors,
            n_points=200, zorder=2
        )

    # group-wise mean connection lines & markers
    for g in group_levels_use:
        mask = (subj_group.values == g)
        if np.sum(mask) == 0: 
            continue
        means_g = [wide_dB.loc[mask, a].mean() for a in a_levels]
        if connect_means:
            axL.plot(x, means_g, 
                     linestyle=connect_linestyle, linewidth=connect_linewidth,
                     color=color_map[g], zorder=3)
        if show_mean_markers:
            axL.plot(x, means_g, linestyle='none', marker='o', markersize=mean_markersize,
                     markerfacecolor=mean_markerface, markeredgewidth=mean_markeredgewidth,
                     markeredgecolor='k', color=color_map[g], zorder=3)

    axL.set_xticks(x)
    axL.set_xticklabels([str(a) for a in a_levels])
    axL.set_xlabel(a_col)
    axL.set_ylabel(f"'{b_levels[0]} - {b_levels[1]}' on {y_col}")
    title_left = f"Split violins by {group_col}" if (group_col is not None and len(group_levels_use)==2) else f"Violins by {a_col} level"
    axL.set_title(title_left, fontsize=10)
    axL.spines['top'].set_visible(False)
    axL.spines['right'].set_visible(False)

    # -------- RIGHT: pairwise Aj−Ai differences with (possibly) MCC-adjusted CI --------
    pairs = list(combinations(range(len(a_levels)), 2))  # (i,j) with i<j
    # define comparisons count for MCC
    m = len(pairs) * len(group_levels_use)
    # 1) prepare per-row alpha according to mcc
    def compute_alpha_per_row():
        if mcc == 'bonferroni':
            return np.full(m, alpha / m if m > 0 else alpha, dtype=float)
        elif mcc == 'holm-ci':
            # compute unadjusted p-values first to rank
            p_unadj = []
            tmp = []
            for (i, j) in pairs:
                for g in group_levels_use:
                    mask = (subj_group.values == g)
                    xvec = wide_dB.loc[mask, a_levels[j]].values
                    yvec = wide_dB.loc[mask, a_levels[i]].values
                    _, _, p, _ = paired_diff_stats(xvec, yvec, alpha=alpha, n_boot=n_boot, random_state=0)
                    p_unadj.append(p); tmp.append((i, j, g))
            order = np.argsort(p_unadj)
            alpha_seq = np.array([alpha / (m - r) for r in range(m)], dtype=float)
            alpha_per = np.empty(m, dtype=float); alpha_per[order] = alpha_seq
            return alpha_per
        else:
            return np.full(m, alpha, dtype=float)

    alpha_per_row = compute_alpha_per_row()

    # 2) compute stats & draw
    rows = []
    row_idx = 0
    for (i, j) in pairs:
        a_i, a_j = a_levels[i], a_levels[j]
        for g in group_levels_use:
            mask = (subj_group.values == g)
            xvec = wide_dB.loc[mask, a_j].values
            yvec = wide_dB.loc[mask, a_i].values
            md, (lo, hi), p, n = paired_diff_stats(
                xvec, yvec, 
                alpha=alpha_per_row[row_idx],
                n_boot=n_boot, random_state=1000 + row_idx)
            rows.append({
                "pair": (a_i, a_j), 
                "group": g, 
                "md": md, "lo": lo, "hi": hi, 
                "color": color_map[g], "n": n})
            row_idx += 1

    # Lay out y positions: one baseline row per pair; groups offset by ±ygap
    if shade_rows:
        ng = len(group_levels_use)
        # half-span of a "row" including group offsets; 
        #if only one group, this reduces to band_pad
        half = ((ng - 1) / 2.0) * ygap + band_pad
        # set range of y-axis to prevent auto-adaption of y-axis
        ymin_theory = -half
        ymax_theory = (len(pairs) - 1) + half
        axR.set_ylim(ymin_theory, ymax_theory) 
        # get length of y-axis
        length_y = ymax_theory - ymin_theory
        # make grid/bands behind data
        axR.set_axisbelow(True)
        for k in range(len(pairs)):
            y0 = length_y/len(pairs)*k-half
            y1 = length_y/len(pairs)*(k+1)-half
            # alternate colors across rows
            face = shade_colors[k % len(shade_colors)]
            axR.axhspan(y0, y1, facecolor=face, alpha=0.75,
                        edgecolor="none", zorder=-1, clip_on=False)
        
    y_positions = []
    for idx, row in enumerate(rows):
        pair_index = pairs.index((a_levels.index(row["pair"][0]), a_levels.index(row["pair"][1])))
        if len(group_levels_use) == 1:
            y = pair_index
        else:
            g_idx = group_levels_use.index(row["group"])
            offset = (g_idx - (len(group_levels_use)-1)/2) * ygap
            y = pair_index + offset
        axR.errorbar(row["md"], y,
                     xerr=[[row["md"] - row["lo"]], [row["hi"] - row["md"]]],
                     fmt='o', markersize=7, elinewidth=3, capsize=0,
                     color=row["color"], zorder=2)
        y_positions.append(y)

    axR.axvline(0, linestyle='--', linewidth=1.0, color='gray', alpha=0.8, zorder=0)
    axR.set_yticks(range(len(pairs)))
    axR.set_yticklabels([f"{a_levels[j]} − {a_levels[i]}" for (i, j) in pairs])
    
    xlabel = 'Effect size of paired difference(95% CI)'
    axR.set_xlabel(xlabel)
    
    titleR = f"Pairwise differences across {a_col} levels"
    axR.set_title(titleR, fontsize=10)
    axR.spines['top'].set_visible(False)
    axR.spines['right'].set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.show()
    
    return save_path

def plot_forest_simple_contrasts_enhanced(
    pts_F: pd.DataFrame,
    pts_M: pd.DataFrame,
    title: str = "Estimated difference (High − Low)",
    conA_label=None, conB_label=None,
    female_label: str = "Female",
    male_label: str = "Male",
    female_color: str = "#d62728",   # red-ish
    male_color: str = "#1f77b4",     # blue-ish
    band_colors: tuple = ("#eef3ff", "#fff0ea"),  # alternating horizontal bands per AGE GROUP
    figsize: tuple | None = None,
    jitter: float = 0.22,            # vertical offset within each age band
    square_size: float = 7.5,        # square marker size
):
    """
    Forest-style plot for simple contrasts at representative ages, with per-AGE-GROUP bands.
    - pts_F / pts_M must include columns: ['age','diff','lower','upper'] (one row per age).
    - We do an inner join on 'age'; each age row (band) shows two CIs (Female & Male).
    - Age groups are separated by alternating horizontal bands (axhspan), NOT per-sex lines.
    """

    # 1) Align ages by inner-join (require same ages for both groups)
    cols = ["age", "diff", "lower", "upper"]
    F = pts_F[cols].copy()
    M = pts_M[cols].copy()
    merged = F.merge(M, on="age", suffixes=("_F", "_M"))
    if merged.empty:
        raise ValueError("No overlapping ages between pts_F and pts_M.")
    merged = merged.sort_values("age").reset_index(drop=True)

    ages = np.asarray(merged["age"], dtype=float)
    n_age = len(ages)

    # 2) Auto figure size: elongate to fit ~13 age groups comfortably
    if figsize is None:
        # ~0.8 inch per age row + margins
        height = max(6.0, 0.8 * n_age + 2.0)
        figsize = (9.5, height)

    # 3) Prepare data arrays
    dF, loF, hiF = [np.asarray(merged[k], dtype=float) for k in ("diff_F", "lower_F", "upper_F")]
    dM, loM, hiM = [np.asarray(merged[k], dtype=float) for k in ("diff_M", "lower_M", "upper_M")]

    # y positions: one integer per AGE GROUP; jitter within the band for sex
    y  = np.arange(n_age, dtype=float)
    yF = y - jitter
    yM = y + jitter

    # x-limits with margin
    x_all = np.concatenate([loF, hiF, loM, hiM])
    xr = float(x_all.max() - x_all.min()) or 1.0
    xmin = float(x_all.min() - 0.08 * xr)
    xmax = float(x_all.max() + 0.08 * xr)

    # 4) Start plotting
    fig, ax = plt.subplots(figsize=figsize)

    # AGE-GROUP horizontal bands (each band spans the full width for one age)
    # Band spans from y-0.5 to y+0.5 so both Male/Female sit inside the same colored band.
    for i, yi in enumerate(y):
        ax.axhspan(yi - 0.5, yi + 0.5,
                   facecolor=band_colors[i % 2],
                   edgecolor="none",
                   alpha=0.65,
                   zorder=0)

    # Zero reference line
    ax.axvline(0.0, linestyle="--", color="0.35", linewidth=1.0, zorder=1)

    # CIs (as horizontal lines) drawn within each band
    ax.hlines(yF, loF, hiF, color=female_color, linewidth=1.8, zorder=2)
    ax.hlines(yM, loM, hiM, color=male_color,   linewidth=1.8, zorder=2)

    # Center squares
    ax.plot(dF, yF, marker="s", markersize=square_size, linestyle="None",
            markerfacecolor=female_color, markeredgecolor=female_color,
            label=female_label, zorder=3)
    ax.plot(dM, yM, marker="s", markersize=square_size, linestyle="None",
            markerfacecolor=male_color, markeredgecolor=male_color,
            label=male_label, zorder=3)

    # Y ticks: one per AGE GROUP (not per sex)
    def fmt_age(a):
        return f"{int(round(a))}y" if float(a).is_integer() else f"{a:g}y"
    ax.set_yticks(y, labels=[fmt_age(a) for a in ages])

    # Labels & title
    ax.set_xlabel(f"Estimated difference ({conA_label} − {conB_label})")
    ax.set_title(title)

    # Bounds & layout
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.7, n_age - 1 + 0.7)  # leave padding to see full bands
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
    
def plot_forest_simple_contrasts_horizontal(
    pts_F: pd.DataFrame,
    pts_M: pd.DataFrame,
    title: str = "Estimated difference (RPSL − LKNG)",
    female_label: str = "Female",
    male_label: str = "Male",
    female_color: str = "#d62728",   # red-ish
    male_color: str = "#1f77b4",     # blue-ish
    band_colors: tuple = ("#eef3ff", "#fff0ea"),  # alternating VERTICAL bands per AGE GROUP
    figsize: tuple | None = None,
    jitter: float = 0.22,            # horizontal offset within each age band
    square_size: float = 7.5,        # square marker size
    show_legend = False,
):
    """
    Forest-style plot for simple contrasts at representative ages (HORIZONTAL layout).

    - x-axis: age (representative ages)
    - y-axis: Estimated difference (High − Low)

    pts_F / pts_M must include columns: ['age','diff','lower','upper'] (one row per age).
    Each AGE GROUP is shown as a vertical band, within which Female & Male estimates
    are plotted with a small horizontal jitter.
    """

    # 1) Align ages by inner-join
    cols = ["age", "diff", "lower", "upper"]
    F = pts_F[cols].copy()
    M = pts_M[cols].copy()
    merged = F.merge(M, on="age", suffixes=("_F", "_M"))
    if merged.empty:
        raise ValueError("No overlapping ages between pts_F and pts_M.")
    merged = merged.sort_values("age").reset_index(drop=True)

    ages = np.asarray(merged["age"], dtype=float)
    n_age = len(ages)

    # 2) Auto figure size: widen to fit ~13 age groups
    if figsize is None:
        width = max(10.0, 0.8 * n_age + 2.0)
        figsize = (width, 6.0)

    # 3) Prepare arrays
    dF, loF, hiF = [np.asarray(merged[k], dtype=float)
                    for k in ("diff_F", "lower_F", "upper_F")]
    dM, loM, hiM = [np.asarray(merged[k], dtype=float)
                    for k in ("diff_M", "lower_M", "upper_M")]

    # x positions = AGE GROUPS; jitter within group for sex
    x  = np.arange(n_age, dtype=float)
    xF = x - jitter
    xM = x + jitter

    # y-limits with margin
    y_all = np.concatenate([loF, hiF, loM, hiM])
    yr = float(y_all.max() - y_all.min()) or 1.0
    ymin = float(y_all.min() - 0.08 * yr)
    ymax = float(y_all.max() + 0.08 * yr)

    # 4) Start plotting
    fig, ax = plt.subplots(figsize=figsize)

    # AGE-GROUP vertical bands (each band spans one age)
    for i, xi in enumerate(x):
        ax.axvspan(
            xi - 0.5, xi + 0.5,
            facecolor=band_colors[i % 2],
            edgecolor="none",
            alpha=0.65,
            zorder=0
        )

    # Zero reference line
    ax.axhline(0.0, linestyle="--", color="0.35", linewidth=1.0, zorder=1)

    # CIs (vertical lines now)
    ax.vlines(xF, loF, hiF, color=female_color, linewidth=2.5, zorder=2)
    ax.vlines(xM, loM, hiM, color=male_color,   linewidth=2.5, zorder=2)

    # Center squares
    ax.plot(
        xF, dF, marker="s", markersize=square_size, linestyle="None",
        markerfacecolor=female_color, markeredgecolor=female_color,
        label=female_label, zorder=3
    )
    ax.plot(
        xM, dM, marker="s", markersize=square_size, linestyle="None",
        markerfacecolor=male_color, markeredgecolor=male_color,
        label=male_label, zorder=3
    )

    # X ticks: one per AGE GROUP
    def fmt_age(a):
        return f"{int(round(a))}" if float(a).is_integer() else f"{a:g}"

    ax.set_xticks(x, labels=[fmt_age(a) for a in ages])

    # Labels & title
    ax.set_ylabel("Estimated difference (High − Low)")
    ax.set_xlabel("Age")
    ax.set_title(title)

    # Bounds & layout
    ax.set_xlim(-0.7, n_age - 1 + 0.7)
    y_ticks = np.arange(0, 1.2, 0.2) 
    ax.set_yticks(y_ticks)
    # ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    if show_legend:
        ax.legend(loc="best")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()


def set_pub_style(
    base_fontsize=10,
    font_family="DejaVu Sans",   # change to 'Arial' if your environment has it
    linewidth=1.8,
    tick_width=1.2,
    grid_alpha=0.25,
    dpi=300
):
    """Set a consistent, publication-grade Matplotlib style."""
    rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.size": base_fontsize,
        "font.family": font_family,
        "axes.titlesize": base_fontsize + 1,
        "axes.labelsize": base_fontsize,
        "axes.linewidth": linewidth,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": tick_width,
        "ytick.major.width": tick_width,
        "legend.frameon": False,
        "legend.fontsize": base_fontsize - 1,
        "axes.grid": False,  # we add grid selectively
    })
#%% Prepare Data Input

dir_beh_data = '/public/home/dingrui/fmri_analysis/data/beh'

sub_info = pd.read_csv(
    os.path.join(dir_beh_data, 'participants_in_tfmri_demographics.csv'), 
    sep=',', 
    usecols=['sub_id', 'gender', 'age', 'site_id', 'site_region']
)

data_subs_ER = os.path.join(dir_beh_data, 'data_4_anova_ER.csv')
data_subs_TG = os.path.join(dir_beh_data, 'data_4_anova_TG.csv')

df_data_ER = pd.read_csv(data_subs_ER, sep=',', index_col=False)
df_data_TG = pd.read_csv(data_subs_TG, sep=',', index_col=False)

sub_ls_ER = list(df_data_ER['sub_id'].unique())
sub_ls_TG = list(df_data_TG['sub_id'].unique())

# add site variable into df_data_ER/TG
for df_data in [df_data_ER, df_data_TG]:
    col_site_id = []
    col_site_region = []
    for sub in list(df_data['sub_id']):
        site_id = sub_info['site_id'][sub_info['sub_id']==sub].tolist()[0]
        site_region = sub_info['site_region'][sub_info['sub_id']==sub].tolist()[0]
        col_site_id.append(site_id)
        col_site_region.append(site_region)
        
    df_data['site_id'] = col_site_id
    df_data['site_region'] = col_site_region

#%% Stats Modeling for Experimental Effect

# 1) RM anova using statsmodels
aov1 = AnovaRM(data=df_data_ER, depvar='emot_rating', subject='sub_id', within=['cognition']).fit()

# Pivot to wide for paired comparisons (each column is a condition):
wide1 = df_data_ER.pivot(index='sub_id', columns='cognition', values='emot_rating')
df_data_ER['cognition'] = pd.Categorical(df_data_ER['cognition'], ordered=True)

levels = list(df_data_ER['cognition'].cat.categories)
pairs  = list(combinations(levels, 2))

results = []
for a, b in pairs:
    # Paired t-test within subjects (same subjects across conditions)
    tval, pval = ttest_rel(wide1[a], wide1[b], nan_policy="omit")
    
    # Cohen's dz for paired designs = mean(diff) / sd(diff)
    diff = (wide1[a] - wide1[b]).dropna()
    diff_mean = diff.mean()
    dz = diff.mean() / diff.std(ddof=1)
    results.append([a, b, diff_mean, tval, pval, dz])

posthoc_df = pd.DataFrame(results, columns=["CondA", "CondB", "diff", "t", "p_unc", "Cohen_dz"])

# Holm correction across the 3 pairwise tests here:
posthoc_df["p_holm"] = multipletests(posthoc_df["p_unc"], method="holm")[1]

# -----------------------------------------------------------------------------
# 2) RM anova using pingouin:
# ER  
aov1 = pg.mixed_anova(
    data=df_data_ER,
    dv='emot_rating', within='cognition', between='gender', 
    subject='sub_id', 
    effsize="np2",
)
    
# TG
df_data_test = df_data_TG.loc[df_data_TG.gender==0].copy()
aov2 = pg.rm_anova(
    data=df_data_test,
    dv='return_rate_mean', within=['cognition', 'trust_level'], 
    subject='sub_id', 
    effsize="np2",
)

aov2[['Source', 'SS', 'MS', 'F', 'p-unc', 'np2']]

## post-hoc comparisons for TG
#  goal: to examnie whether cognition enhances or reduces trust-level effect on 
#        subject's reciprocity

# (a) we can peform post-hoc test of cognition effect at each trust level:
simple_cog_given_tst = []
for tst in df_data_test['trust_level'].unique():
    tmp = df_data_test[df_data_test['trust_level'] == tst]
    out_post_hoc = pg.pairwise_ttests(
        dv='return_rate_mean',
        within='cognition',
        subject='sub_id',
        data=tmp,
        padjust='holm',
        effsize='hedges',
        return_desc=True
    )
    out_post_hoc['trust_level'] = tst
    simple_cog_given_tst.append(out_post_hoc)

df_posthoc_cog_at_tst = pd.concat(simple_cog_given_tst)
cols_interests = [
    'Contrast', 'A', 'B', 
    'mean(A)', 'mean(B)', 
    'Paired', 'T', 'p-unc', 'hedges', 
    'trust_level',
]
df_posthoc_cog_at_tst[cols_interests]

# (b) if interaction exists, we can also calculate difference between two levels of trust_level
#     per cognition, and directly statistically test cognition effect on such difference:
df_diff_B_given_A, _ = build_diffs_by_A(
    df_data_test, 
    id_col='sub_id', 
    a_col='cognition', b_col='trust_level',
    y_col='delta_eq_mean', 
    a_levels=('rpsl', 'lkng', 'lknt'), 
    b_levels=('highTrust', 'lowTrust'),
)

df_posthoc_tst_at_cog = pg.pairwise_ttests(
    dv='dB',
    within='cognition',
    subject='sub_id',
    data=df_diff_B_given_A,
    padjust='holm',
    effsize='hedges',
    return_desc=True
)

col_interests = [
    'Contrast', 'A', 'B', 
    'mean(A)', 'mean(B)', 
    'Paired', 'T', 'p-unc', 'hedges'
]
df_posthoc_tst_at_cog[col_interests]

# (c) visualization of cognition modulation of trust_level effect on TG indices:
# (c1) we can simply plot group mean of within-subject difference btw highT vs lowT,
#      along with data points or just directly plot interaction effect
plot_dB_vs_A(
    df_data_test, fig_size=(4,7),
    id_col='sub_id', a_col='cognition', b_col='trust_level', y_col='return_rate_mean',
    a_levels=('rpsl', 'lkng', 'lknt'),
    b_levels=('highTrust', 'lowTrust'),
    # marker properties of group mean 
    marker_size=10, marker_facecolor='white', 
    marker_edgecolor='k', marker_edgewidth=1.8,
    # connection line properties
    conn_color='k', conn_width=1.5, conn_style='-',
    # error bar properties
    err_type='se', capsize=3, err_width=1.6,
    # original data points of within-subject difference
    show_points=True, data4points='normalized', points_size=20, points_alpha=0.35,
    points_jitter_width=0.08,
)

# (c2) we also can visualize cognition modulatioin of trust_level effect using 
#      Gardner–Altman estimation plot
plot_ga_diffs(
    df_data_TG, 
    id_col='sub_id', a_col='cognition', b_col='trust_level', y_col='return_rate_mean',
    a_levels=('rpsl', 'lkng', 'lknt'),
    b_levels=('highTrust', 'lowTrust'),
    # grouping data by gender
    group_col="gender", group_levels=[0,1], group_colors={0:'#ff7f0e',1:'#1f77b4'},  # {'Male':'#1f77b4','Female':'#ff7f0e'}
    # left plot for raw data points of diff and means connection lines
    point_marker='.', point_size=25, point_alpha=0.3,
    jitter=True, jitter_width=0.2, jitter_seed=0,
    connect_means=True, connect_linewidth=1.6, connect_linestyle='--',
    # right plot for effect size (95% CI)
    alpha=0.05, n_boot=5000, mcc='bonferroni',  # 'none' / 'bonferroni' / 'holm'
    annotate_p=False,  
    ygap=0.22, figsize=(8, 4.5),
    return_CI=False, save_path=None
)

# (c3) last, we need to visualize the condition-wise data distribution using boxplot
plot_condition_boxplots(
    df_data_test,
    condition_col='trust_level', score_col='return_rate_mean', subject_col='sub_id',
    condition_order=['highTrust', 'lowTrust'],
    group_col='cognition', group_order=['rpsl', 'lkng', 'lknt'],
    add_points=True, point_style='jitter', points_layer='behind',
    point_size=10, point_jitter=0.05, 
    point_palette=['gray', 'gray'], point_edgecolor='gray',
    point_jitter_main_axis=0.1,
    box_colors={'rpsl':"#501d8a", 'lkng':"#1c8041", 'lknt':'#e55709'},
    show_counts=False,
)
# -----------------------------------------------------------------------------

#%% Stats Modeling with Mixed effect

df_data_ER['age_c'] = df_data_ER['age'] - df_data_ER['age'].mean()

# fixed effects: 
formula_l = 'emot_rating ~ C(cognition) * C(gender) * age_c + C(site_region)'
formula_s = 'emot_rating ~ C(cognition) * C(gender) * cr(age_c, df=3) + C(site_region)'
# random effects:
re_formula = '~C(cognition)'

#
model_linear = mixedlm(
    formula_l, 
    data=df_data_ER, 
    groups=df_data_ER['sub_id'],
    re_formula=re_formula,
)

model_spline = mixedlm(
    formula_s,
    data=df_data_ER,
    groups=df_data_ER['sub_id'],
    re_formula=re_formula,
)

model1 = model_linear.fit(reml=False, method='lbfgs')
model2 = model_spline.fit(reml=False, method='lbfgs')

#%% Site effect
y, X = dmatrices(formula_l, data=df_data_ER, return_type="dataframe")
params = model1.params.reindex(X.design_info.column_names)
cov    = model1.cov_params().loc[params.index, params.index]  

cols = X.columns.tolist()

site_cols = [c for c in cols if c.startswith("C(site_region)")]
site_effc = wald_block(X, site_cols)

#%% Joint Wald Test for Age Effects

#
formula_s_m = 'emot_rating ~ C(cognition) * C(gender, Treatment(reference=0)) * bs(age_c, df=4)'
formula_s_f = 'emot_rating ~ C(cognition) * C(gender, Treatment(reference=1)) * bs(age_c, df=4)'

age_effects_ls = []
for formula, gender in zip(
        [formula_s_m, formula_s_f],
        ['male', 'female']
):
    model = mixedlm(
        formula,
        data=df_data_ER,
        groups=df_data_ER['sub_id'],
        re_formula=re_formula,
    ).fit(reml=False, method='lbfgs')
    
    y, X = dmatrices(formula, data=df_data_ER, return_type="dataframe")
    
    params = model.params.reindex(X.design_info.column_names)
    cov    = model.cov_params().loc[params.index, params.index]  
    
    cols = X.columns.tolist()
    # age main effect:
    age_cols   = [c for c in cols if c.startswith("bs(age_c, df=4)[")]
    # age * condition:
    age_by_cond = [c for c in cols if c.startswith("C(cognition") and "bs(age_c, df=4)[" in c and "C(gender" not in c]
    # age * gender:
    age_by_gender = [c for c in cols if c.startswith("C(gender") and "bs(age_c, df=4)[" in c]
    # age * condition * gender:
    age_by_cond_by_gender = [c for c in cols if "C(cognition" in c and "C(gender" in c and "bs(age_c, df=4)[" in c]
    
    # Wald chi2 tests for above effects:
    res_wald = []
    for effect in [
            age_cols,
            age_by_cond,
            age_by_gender,
            age_by_cond_by_gender,
    ]:
        # wald_chi2, p_val, df_block = (wald_block(X, effect).values())
        res = wald_block(X, effect)
        res_wald.append(res)
    
    df_res_wald = pd.DataFrame(res_wald)
    df_res_wald.insert(
        loc=0, column='effect term',
        value=['age_main', 'age_by_cond', 'age_by_gender', 'age_by_cond_by_gender']
    )
    df_res_wald.insert(
        loc=0, column='gender',
        value=[gender]*4
    )
    
    age_effects_ls.append(df_res_wald)

df_wald_res = pd.concat(age_effects_ls)

#%%

# grid of age:
age_grid = np.linspace(df_data_ER['age'].min(), df_data_ER['age'].max(), 200)

# Extract fixed-effect pieces (names, params, covariance)
fe_names  = model2.model.exog_names                      # fixed-effect column names
fe_params = model2.fe_params.reindex(fe_names)           # align order explicitly
fe_cov    = model2.cov_params().loc[fe_names, fe_names]  # fixed-effect covariance

# Prepare RHS for patsy.dmatrix (dmatrix expects RHS only)
rhs = formula_s.split("~", 1)[1].strip()

X_train = dmatrix(rhs, data=df_data_ER, return_type="dataframe")
design_info = X_train.design_info

curve_F_rpsl = predict_curve(df_data_ER, 'rpsl', 1, age_grid, fe_names, fe_params, fe_cov)
curve_F_lkng = predict_curve(df_data_ER, 'lkng', 1, age_grid, fe_names, fe_params, fe_cov)

# Difference curve by subtraction (for visualization only)
diff_sub = curve_F_rpsl[['age']].copy()
diff_sub['diff']  = curve_F_rpsl['mean'].to_numpy() - curve_F_lkng['mean'].to_numpy()

#%% Plots

# boxplot for conditions comparison (experimental effects):
plot_condition_boxplots(
    df_data_ER,
    condition_col='cognition', score_col='emot_rating', subject_col='sub_id',
    group_col="gender", group_order=[0, 1],
    # figsize=(5, 6),
    add_points=True, point_style='jitter', points_layer='behind',
    point_size=10, point_jitter=0.08, 
    point_palette=['k', 'k'], point_edgecolor='k',
    point_jitter_main_axis=0.1,
    box_colors={0:"#f68724", 1:"#3278a7"},
    show_counts=False,
)

# age effect on cognition effect(rpsl vs lkng):
df_age_diff_m = contrast_diff(df_data_ER, design_info, 'rpsl', 'lkng', 0, age_grid, fe_params, fe_cov)
df_age_diff_f = contrast_diff(df_data_ER, design_info, 'rpsl', 'lkng', 1, age_grid, fe_params, fe_cov)

plot_age_moderation(
    df_age_diff_m, df_age_diff_f,
    condA_label='rpsl',
    condB_label='lkng',
)

# age effect on cognition effect(lknt vs lkng):
df_age_diff_m = contrast_diff(df_data_ER, design_info, 'lknt', 'lkng', 0, age_grid, fe_params, fe_cov)
df_age_diff_f = contrast_diff(df_data_ER, design_info, 'lknt', 'lkng', 1, age_grid, fe_params, fe_cov)

plot_age_moderation(
    df_age_diff_m, df_age_diff_f,
    condA_label='lknt',
    condB_label='lkng',
)

# forest plot for estimated effect/difference at representative age:
# 1) 'rpsl' vs 'lkng' 

rep_ages = [7, 8.0, 9, 10, 11, 12.0, 13, 14, 15, 16.0, 17, 18]

pts_F = contrast_diff(df_data_ER, design_info, 'rpsl', 'lkng', 1, rep_ages, 'south china', fe_params, fe_cov)
pts_M = contrast_diff(df_data_ER, design_info, 'rpsl', 'lkng', 0, rep_ages, 'south china', fe_params, fe_cov)

plot_forest_simple_contrasts_enhanced(
    pts_F, pts_M,
    title="(RPSL vs. LKNG) at representative ages",
    conA_label='RPSL', conB_label='LKNG',
    female_label="Female", male_label="Male",
    female_color='#3278a7', male_color='#f68724',
    band_colors=('#ecf0f1', '#bdc3c7'),
    figsize=(6, 5)
)

# 2) 'lknt' vs 'lkng'
# Extract fixed-effect pieces (names, params, covariance)
fe_names  = model1.model.exog_names                      # fixed-effect column names
fe_params = model1.fe_params.reindex(fe_names)           # align order explicitly
fe_cov    = model1.cov_params().loc[fe_names, fe_names]  # fixed-effect covariance

# Prepare RHS for patsy.dmatrix (dmatrix expects RHS only)
rhs = formula_l.split("~", 1)[1].strip()
X_train = dmatrix(rhs, data=df_data_ER, return_type="dataframe")
design_info = X_train.design_info

pts_F = contrast_diff(df_data_ER, design_info, 'lknt', 'lkng', 1, rep_ages, 'south china', fe_params, fe_cov)
pts_M = contrast_diff(df_data_ER, design_info, 'lknt', 'lkng', 0, rep_ages, 'south china', fe_params, fe_cov)

plot_forest_simple_contrasts_enhanced(
    pts_F, pts_M,
    title="(LKNT vs. LKNG) at representative ages",
    conA_label='LKNT', conB_label='LKNG',
    female_label="Female", male_label="Male",
    female_color='#3278a7', male_color='#f68724',
    band_colors=('#ecf0f1', '#bdc3c7'),
    figsize=(5, 8)
)

#%% Model Comparison between Linear and Non-linear Ones

# 1) likelihood ratio test(LRT)
# formulas for linear and non-linear age effect:
f_lin = "emot_rating ~ C(cognition, Treatment('lkng')) * C(gender, Treatment(0)) * age_c"
f_spl = "emot_rating ~ C(cognition, Treatment('lkng')) * C(gender, Treatment(0)) * bs(age_c, df=4)"
re_formula = '~C(cognition)'

# model fit:
fit_lin = mixedlm(
    f_lin, 
    data=df_data_ER, 
    groups=df_data_ER["sub_id"],
    re_formula=re_formula,
).fit(reml=False, method="lbfgs")

fit_spl = mixedlm(
    f_spl, 
    data=df_data_ER, 
    groups=df_data_ER["sub_id"],
    re_formula=re_formula,
).fit(reml=False, method="lbfgs")

# likelihood ratio:
ll_lin, ll_spl = fit_lin.llf, fit_spl.llf

# diff of degree of freedom(DoF) between number of parameters for fixed effects of two models:
df_lin, df_spl = len(fit_lin.fe_params), len(fit_spl.fe_params)
dof = df_spl - df_lin
LR  = 2*(ll_spl - ll_lin)
p   = chi2.sf(LR, dof)

print(f"LRT: 2Δℓ={LR:.3f}, df={dof}, p={p:.4g}")
print(f"AIC: linear={fit_lin.aic:.1f}, spline={fit_spl.aic:.1f}")
print(f"BIC: linear={fit_lin.bic:.1f}, spline={fit_spl.bic:.1f}")

# 2) joint Wald test on non-linear block of non-linear model to
#    directly statistically test whether age effect need non-linearity
#
# idea: decompose bs(age) basis into linear part and non-linear parts(R), and
#       then fit the model: emo_rating ~ C(cognition)*C(gender)*( age + R )

#
# original spline basis:
B = dmatrix("bs(age_c, df=4)", data=df_data_ER, return_type="dataframe")
B = pd.DataFrame(B, index=df_data_ER.index)

# decompose spline basis into linear and non-linear parts:
X_lin = np.c_[np.ones(len(df_data_ER)), df_data_ER["age_c"].values]   # intercept + age
Q, *_ = np.linalg.lstsq(X_lin, B.values, rcond=None)                  # coefficients for projection
B_hat = X_lin @ Q                                                     # fitted linear parts
R = B.values - B_hat                                                  # residual (nonlinear) basis
R = pd.DataFrame(
    R, 
    index=df_data_ER.index, 
    columns=[f"R{j}" for j in range(R.shape[1])],
)

# put unexplained non-linear parts into df_data:
df_aug = pd.concat([df_data_ER, R], axis=1)

# model fit:
rhs_R = " + ".join(R.columns)   # R*
f_aug = f"emot_rating ~ C(cognition, Treatment('lkng'))*C(gender, Treatment(0))*(age_c + {rhs_R})"
fit_aug = mixedlm(
    f_aug, 
    data=df_aug, 
    groups=df_aug["sub_id"],
    re_formula='~C(cognition)'
).fit(reml=False, method="lbfgs")

# ------------------------------------------------------------------------------------------------------
# joint Wald test on non-linear terms (R*):
fe_names  = fit_aug.model.exog_names
fe_params = fit_aug.fe_params.reindex(fe_names)
fe_cov    = fit_aug.cov_params().loc[fe_names, fe_names]

# select all columns(terms) consisting of 'R':
R_cols = [name for name in fe_names if any((f":{r}" in name) or (name.endswith(f"*{r}")) or (name==r) for r in R.columns)]

R_cols = [name for name in fe_names if any(r in name for r in R.columns)]

# Wald stats:
L = np.zeros((len(R_cols), len(fe_params)))
for i, col in enumerate(R_cols):
    L[i, fe_params.index.get_loc(col)] = 1.0

beta = fe_params.values
W = beta @ L.T
V = L @ fe_cov.values @ L.T

stat = float(W @ np.linalg.inv(V) @ W)
df_w = L.shape[0]
p_w  = chi2.sf(stat, df_w)
# ------------------------------------------------------------------------------------------------------

print(f"Wald nonlinearity block: chi2={stat:.3f}, df={df_w}, p={p_w:.4g}")

#%% 