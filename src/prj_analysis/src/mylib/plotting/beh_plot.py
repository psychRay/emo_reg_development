#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 01:11:43 2025

    sorts of useful tools to visualize behaviral contents

@author: dingrui
"""

# import necessary modules
import warnings
warnings.filterwarnings('ignore', message='.*deprecated.*')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mylib.behv.utils as U

from scipy.stats import gaussian_kde
from itertools import combinations

# define functions
def _get_cycle_colors(n):
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if len(prop_cycle) >= n:
        return prop_cycle[:n]
    reps = (n + len(prop_cycle) - 1) // len(prop_cycle)
    return (prop_cycle * reps)[:n]

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
    figsize = None, show_legend = None,
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
        if figsize is not None:
            fig_w = figsize[0]; fig_h = figsize[1]
        else:
            fig_w = 10.5
            fig_h = max(4.8, 0.7 * len(order) + 2.0)
        
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_title(title)
        ax.set_xlabel(score_col)
        
    else:
        if figsize is not None:
            fig_w = figsize[0]; fig_h = figsize[1]
        else:
            fig_w = max(6, 0.7 * len(order) + 2.0)
            fig_h = 6
        
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
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
        
        if show_legend:
            ax.legend(handles=legend_handles, title=group_col, loc="best")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_forest_simple_contrasts_enhanced(
    pts_F: pd.DataFrame,
    pts_M: pd.DataFrame,
    title: str = "Estimated difference (High − Low)",
    female_label: str = "Female",
    male_label: str = "Male",
    female_color: str = "#d62728",   # red-ish
    male_color: str = "#1f77b4",     # blue-ish
    grid_colors: tuple = ("#e8eef9", "#f9eee8"),  # alternating horizontal grid colors
    figsize: tuple | None = None,
    jitter: float = 0.18,
    square_size: float = 7.5,
):
    """
    Forest-style plot for simple contrasts at representative ages.
    - pts_F / pts_M must include columns: ['age', 'diff', 'lower', 'upper'].
    - Ages are merged (inner join). Each age row shows two CI bars (Female & Male),
      with square markers at the center estimate.
    - Alternating horizontal grid lines separate age groups.

    Parameters
    ----------
    pts_F, pts_M : DataFrame
        Must contain columns ['age','diff','lower','upper']. One row per age.
    title : str
        Plot title.
    female_label, male_label : str
        Legend labels.
    female_color, male_color : str
        Colors for the two groups (hex or matplotlib names).
    grid_colors : tuple(str, str)
        Two alternating colors for horizontal grid lines across age rows.
    figsize : (w,h) or None
        If None, height is auto-scaled by number of age groups.
    jitter : float
        Vertical offset so the two groups don’t overlap at the same age.
    square_size : float
        Marker size for the square centers.
    """

    # 1) Align ages by inner-join (require same age set for both groups)
    cols = ["age", "diff", "lower", "upper"]
    F = pts_F[cols].copy()
    M = pts_M[cols].copy()
    merged = F.merge(M, on="age", suffixes=("_F", "_M"))
    if merged.empty:
        raise ValueError("No overlapping ages between pts_F and pts_M.")
    merged = merged.sort_values("age").reset_index(drop=True)

    ages = np.asarray(merged["age"], dtype=float)
    n_age = len(ages)

    # 2) Figure size: elongate for many age groups (e.g., 13 ages)
    if figsize is None:
        # ~0.7 inches per age row + margins; tweak as you like
        height = max(6.0, 0.7 * n_age + 2.0)
        figsize = (9.0, height)

    # 3) Prepare data
    dF, loF, hiF = [np.asarray(merged[k], dtype=float) for k in ("diff_F", "lower_F", "upper_F")]
    dM, loM, hiM = [np.asarray(merged[k], dtype=float) for k in ("diff_M", "lower_M", "upper_M")]

    # y positions: one tick per age; jitter lines for each group
    y  = np.arange(n_age, dtype=float)
    yF = y - jitter
    yM = y + jitter

    # x-limits with margin
    x_all = np.concatenate([loF, hiF, loM, hiM])
    xr = float(x_all.max() - x_all.min())
    xmin = float(x_all.min() - 0.06 * xr)
    xmax = float(x_all.max() + 0.06 * xr)

    # 4) Draw
    fig, ax = plt.subplots(figsize=figsize)

    # alternating horizontal grid lines (behind data)
    for i, yi in enumerate(y):
        ax.hlines(
            yi, xmin, xmax,
            color=grid_colors[i % 2],
            linewidth=2.5,
            zorder=0,
            alpha=0.6
        )

    # zero line
    ax.axvline(0.0, linestyle="--", color="0.3", linewidth=1.0, zorder=1)

    # CIs (as horizontal lines)
    ax.hlines(yF, loF, hiF, color=female_color, linewidth=1.8, zorder=2)
    ax.hlines(yM, loM, hiM, color=male_color,   linewidth=1.8, zorder=2)

    # Centers (squares)
    ax.plot(dF, yF, marker="s", markersize=square_size, linestyle="None",
            markerfacecolor=female_color, markeredgecolor=female_color, label=female_label, zorder=3)
    ax.plot(dM, yM, marker="s", markersize=square_size, linestyle="None",
            markerfacecolor=male_color, markeredgecolor=male_color, label=male_label, zorder=3)

    # Y ticks & labels
    ax.set_yticks(y, labels=[f"{int(round(a))}y" if float(a).is_integer() else f"{a:g}y" for a in ages])

    # Labels & title
    ax.set_xlabel("Estimated difference (High − Low)")
    ax.set_title(title)

    # Bounds & layout
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.8, n_age - 1 + 0.8)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

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
    --> A moderation of B effect on dependent variable
    """
    
    # prepare data for fig 
    wide_dB, long_dB, _ = U.build_diffs_by_A(df, id_col, a_col, b_col, y_col, a_levels, b_levels)
    if a_levels is None:
        a_levels = list(wide_dB.columns)
    if b_levels is None:
        b_levels = list(pd.unique(df[b_col]))
        
    x = np.arange(len(a_levels))
    means, ci_half, n, normalized = U.within_subject_ci_matrix(wide_dB.values, err_type, alpha=alpha_ci)
    
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
    return (means, ci_half, n), fig

def _half_violin_with_median(
    ax, x_center, values, side="+", *,
    width=0.8, facecolor=None, alpha=0.35, edgecolor=None, linewidth=1.0,
    show_median=True, median_linestyle='--', median_linewidth=2.0, median_color=None,
    n_points=200, zorder=2
):
    """
    Plot ONE half of a violin from x_center to either right ('+') or left ('-').
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
    - If len(group_order)>2  -> side-by-side skinny violins.

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
    wide_dB,                 # DataFrame: subjects x A-level columns (eg. values are diff btw B levels at each A-level)
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
    
    """
    Gardner–Altman plot for factor A moderation of B effect on dependent variable
    
     - left panel:  within-subject difference btw two levels of B against A levels
     - right panel: 95% CI of estimated difference btw two levels of B grouped by A
     
    """
    
    # prepare differece data for subsequent visualization
    wide_dB, long_dB, subj_group = U.build_diffs_by_A(
        df, id_col, a_col, b_col, y_col, 
        a_levels, b_levels, 
        group_col, group_levels
    )
    
    _, _, _, normalized = U.within_subject_ci_matrix(wide_dB.values)
    
    if a_levels is None:
        a_levels = list(wide_dB.columns)
    if b_levels is None:
        b_levels = list(pd.unique(df[b_col]))
    
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
            md, (lo, hi), p, n = U.paired_diff_stats(
                xvec, yvec, 
                alpha=alpha_ci, n_boot=n_boot, 
                random_state=1000 + pi*10 + gi
            )
            label = f"{a_i} − {a_j} (n={n})" if g is None else f"{a_i} − {a_j} [{g}] (n={n})"
            rows.append({"pair": (a_i, a_j), "group": g, "md": md, "lo": lo, "hi": hi, "label": label, "color": color_map[g]})
            p_collect.append(p)

    p_corr = None
    if mcc == 'holm' and len(p_collect) > 0:
        p_corr = U.holm_correction(p_collect)
    
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
    return fig
    
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
    Gardner–Altman plot for factor A moderation of B effect on dependent variable (violin version)
    
     - left panel:  split violin per A level (if exactly 2 groups) or regular violin/side-by-side for other cases.
     - right panel: all pairwise Aj−Ai differences (paired) with CI; CI can be MCC-adjusted.
    """
    
    # prepare differece data for subsequent visualization
    wide_dB, long_dB, subj_group = U.build_diffs_by_A(
        df, id_col, a_col, b_col, y_col, 
        a_levels, b_levels, 
        group_col, group_levels
    )
    
    if a_levels is None:
        a_levels = list(wide_dB.columns)
    if b_levels is None:
        b_levels = list(pd.unique(df[b_col]))
        
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
                    _, _, p, _ = U.paired_diff_stats(xvec, yvec, alpha=alpha, n_boot=n_boot, random_state=0)
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
            md, (lo, hi), p, n = U.paired_diff_stats(
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
        return save_path
    
    plt.show()
    return fig
    
