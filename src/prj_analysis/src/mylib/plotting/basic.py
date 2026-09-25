#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 01:11:43 2025

    sorts of useful tools to visualize behaviral contents

@author: dingrui
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams



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
    
def get_golden_size(base_size, orientation='landscape'):
    """
    get gold ratio for figsize
    
    Parameters:
    -----------
    base_size : float
    orientation : str, 'landscape'/'portrait' 
    """
    phi = (1 + np.sqrt(5)) / 2
    
    if orientation == 'landscape':
        width = base_size
        height = width / phi
    else:  # portrait
        height = base_size
        width = height / phi
    
    return (width, height)

def _nice_axes(ax, which_grid=None):
    """Remove top/right spines, add light grid on x or y."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if which_grid is not None:
        if which_grid.lower() == "x":
            ax.xaxis.grid(True, alpha=0.25)
        elif which_grid.lower() == "y":
            ax.yaxis.grid(True, alpha=0.25)
        elif which_grid.lower() == "both":
            ax.grid(True, alpha=0.25)

def permutation_boxplot(
    vector,
    scalar,
    ax=None,
    figsize=(6.2, 3.6),
    *,
    # geometry & spacing
    box_width=0.45,        # box height (since the boxplot is horizontal)
    gap=0.35,              # vertical distance between box center and density baseline
    kde_height=0.5,        # maximal vertical amplitude of the density curve
    kde_points=400,        # sampling points for KDE curve
    # KDE bandwidth
    bandwidth=None,        # if SciPy is available, passed to gaussian_kde (e.g., 'scott', 'silverman', or float)
                           # if SciPy is NOT available and it's a positive float, used as bandwidth (in data units)
    # styles
    fill=True,             # fill area under the density curve
    fill_alpha=0.25,       # alpha for fill
    density_line_kwargs=None,  # e.g., {'lw':2}
    box_kwargs=None,           # forwarded to plt.boxplot (vert=False); showfliers is False by default here
    outlier_color='C3',        # color of the scalar "outlier-style" marker
    annotate=False,            # annotate numeric value of scalar
    xlabel=None,               # custom x-axis label
    title=None                 # optional title
):
    """
    Draw a HORIZONTAL boxplot of `vector` at y=0; overlay `scalar` in "outlier style";
    and draw a HORIZONTAL density curve for `vector` ABOVE the boxplot.

    Features:
    1) Horizontal boxplot and density; density is placed above the box.
    2) Optional fill under the density curve.
    3) `gap` controls the distance between boxplot and density baseline.
    4) Density x-range is strictly between vector's min and max.
    5) Only draw horizontal x-axis & xticklabels (hide y-axis & other spines).
    6) Customizable xlabel via `xlabel`.
    """
    # ----------------------------
    # Validate and clean data
    # ----------------------------
    v = np.asarray(vector, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        raise ValueError("`vector` must contain at least two finite numbers.")

    x_min, x_max = float(np.min(v)), float(np.max(v))
    if x_min == x_max:
        # degenerate case: add a tiny epsilon to avoid zero-length axis
        eps = 1e-9
        x_min -= eps
        x_max += eps

    # Fixed x-support for density: exactly [min, max]
    x = np.linspace(x_min, x_max, int(kde_points))

    # ----------------------------
    # Axes
    # ----------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # y-positions
    y_box = 0.0                   # center of the horizontal boxplot
    y_kde_base = y_box + box_width/2 + gap      # baseline of density (area fills upward from here)

    # ----------------------------
    # KDE (SciPy if available; NumPy fallback)
    # ----------------------------
    def _kde_numpy(grid, samples, bw=None):
        """
        Simple 1D Gaussian KDE using NumPy only.
        grid: (m,), samples: (n,), bw in data units (float) or None => Scott's rule.
        """
        samples = np.asarray(samples)
        n = samples.size
        s = np.std(samples, ddof=1)
        if not np.isfinite(s) or s == 0:
            s = 1e-3
        if bw is None:
            # Scott's rule in data units
            bw = s * np.power(n, -1.0/5.0)
        u = (grid[:, None] - samples[None, :]) / bw
        dens = np.exp(-0.5 * u * u) / (np.sqrt(2.0 * np.pi) * bw)
        dens = dens.mean(axis=1)
        return dens

    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(v, bw_method=bandwidth if bandwidth is not None else 'scott')
        dens = kde(x)
    except Exception:
        bw_np = bandwidth if (isinstance(bandwidth, (int, float)) and bandwidth > 0) else None
        dens = _kde_numpy(x, v, bw=bw_np)

    # Normalize density to have max vertical amplitude `kde_height`
    max_d = np.nanmax(dens) if np.isfinite(dens).any() else 1.0
    if not np.isfinite(max_d) or max_d <= 0:
        max_d = 1.0
    dens_scaled = (dens / max_d) * kde_height
    y_kde = y_kde_base + dens_scaled

    # ----------------------------
    # Horizontal boxplot
    # ----------------------------
    default_box = dict(
        vert=False,
        positions=[y_box],
        widths=box_width,
        patch_artist=True,
        showfliers=False,  # we draw the scalar ourselves
        boxprops=dict(facecolor='white', edgecolor='0.2', linewidth=1.3),
        medianprops=dict(color='0.1', linewidth=0),
        whiskerprops=dict(color='0.2', linewidth=1.2),
        capprops=dict(color='0.2', linewidth=1.2),
    )
    if box_kwargs:
        default_box.update(box_kwargs)
    ax.boxplot([v], showcaps=False, **default_box)

    # ----------------------------
    # Density curve (line + optional fill)
    # ----------------------------
    _line = dict(lw=2, alpha=0.95)
    if density_line_kwargs:
        _line.update(density_line_kwargs)
    # Draw fill first so the line stays on top
    if fill:
        ax.fill_between(x, y_kde_base, y_kde, color='gray', alpha=fill_alpha)
    # Draw the density line
    ax.plot(x, y_kde, color='gray', **_line)

    # ----------------------------
    # Scalar as "outlier-style" marker (on the box's y-level)
    # ----------------------------
    ax.plot(
        [scalar], [y_box],
        marker='o',
        markersize=8,
        markerfacecolor='r',
        markeredgecolor=outlier_color,
        markeredgewidth=1.6,
        zorder=5
    )
    if annotate:
        ax.annotate(f"{scalar:.3g}", (scalar, y_box),
                    xytext=(0, 8), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color=outlier_color)

    # ----------------------------
    # Axes cosmetics per spec
    # ----------------------------
    # draw horizontal x-axis + xticklabels
    for spine in ['top', 'left', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    # Hide y-axis ticks/labels entirely
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)

    # xlabel
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel("Value")

    if title:
        ax.set_title(title, pad=6)

    # Ensure y-limits accommodate both box and the entire density
    y_min = y_box - (box_width / 2) - 0.1
    # y_max = y_kde_base + kde_height + 0.1
    ax.set_ylim(y_min)
    
    plt.tight_layout()
    plt.show()
    
    return ax

def _kde_1d(x, grid, bw=None):
    """Simple Gaussian KDE without SciPy. bw in data units; Scott's rule if None."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    s = np.std(x, ddof=1)
    if not np.isfinite(s) or s == 0:
        s = 1e-3
    if bw is None:
        bw = s * n ** (-1/5)
    u = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * u * u) / (np.sqrt(2*np.pi) * bw)
    return dens.mean(axis=1)

def eye_plot_with_scatter(
    names,
    samples,                      # list of 1D arrays or 2D array (p x B)
    q_outer=(2.5, 97.5),
    q_inner=(25, 75),
    color_bpoint='r',
    linewidth_ci50=15,
    figsize=(7, 5),
    # density controls
    fill_density=False,
    max_half_thickness=0.5,      # max half thickness of "eye"
    grid_points=400,
    # scatter controls
    show_scatter=True,
    scatter_colors=("#280038", "#229B55", "#C6C100"),
    scatter_mode='density_jitter',# 'density_jitter' | 'jitter' | 'rug'
    jitter_scale=0.9,             # scale factor for jitter relative to max_half_thickness
    max_points=3000,              # per-variable subsample cap for scatter (None=all)
    s=6,                          # marker size
    alpha=0.35,                   # point alpha
    title='Eye plot with scatter (density-jitter)',
    xlabel='Coefficient (β)',
):
    """Eye plot (density fill + 50%/95% intervals + median) with optional scatter overlay."""
    # standardize input
    if isinstance(samples, np.ndarray):
        assert samples.ndim == 2
        boots = [samples[i, :] for i in range(samples.shape[0])]
    else:
        boots = [np.asarray(s, dtype=float) for s in samples]
    p = len(boots)
    assert len(names) == p
    
    if figsize is None:
        H = max(3.2, 0.48 * p + 1.0)
        figsize = (7.6, H)
        
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(p)

    # robust global x-limits
    all_vals = np.concatenate([b[np.isfinite(b)] for b in boots])
    xmin = np.nanpercentile(all_vals, 0.5)
    xmax = np.nanpercentile(all_vals, 99.5)

    xgrid = np.linspace(xmin, xmax, grid_points)

    for i, samps in enumerate(boots):
        x_i = samps[np.isfinite(samps)]
        if x_i.size == 0:
            continue

        # density for the "eye"
        dens = _kde_1d(x_i, xgrid, bw=None)
        dens = dens / (dens.max() if dens.max() > 0 else 1.0) * max_half_thickness
        if fill_density:
            ax.fill_between(xgrid, y[i]-dens, y[i]+dens, alpha=0.25, linewidth=0)

        # intervals + median
        q2p5, q97p5 = np.percentile(x_i, q_outer)
        q25, q75    = np.percentile(x_i, q_inner)
        med         = np.median(x_i)
        ax.hlines(y[i], q2p5, q97p5, lw=2, color=scatter_colors[i])
        ax.hlines(y[i], q25,  q75,   lw=linewidth_ci50, color=scatter_colors[i])
        ax.scatter([med], [y[i]], s=30, zorder=3, color=color_bpoint)

        # scatter overlay
        if show_scatter:
            # optional subsample for performance
            if (max_points is not None) and (x_i.size > max_points):
                idx = np.random.default_rng(0).choice(x_i.size, size=max_points, replace=False)
                xs = x_i[idx]
            else:
                xs = x_i

            if scatter_mode == 'rug':
                # draw small vertical ticks at y[i]
                ax.plot(xs, np.full_like(xs, y[i]), '|', ms=6, alpha=alpha)
            else:
                # compute per-point vertical jitter
                if scatter_mode == 'density_jitter':
                    # density-proportional jitter ("sina"): wider where density is higher
                    dens_full = _kde_1d(x_i, xgrid, bw=None)
                    dens_full = dens_full / (dens_full.max() if dens_full.max() > 0 else 1.0)
                    # map each point's x to density using interpolation
                    dens_at_x = np.interp(xs, xgrid, dens_full)
                    jitter_ampl = dens_at_x * (max_half_thickness * jitter_scale)
                else:
                    # constant jitter
                    jitter_ampl = np.full_like(xs, max_half_thickness * jitter_scale)

                # symmetric random jitter in [-ampl, +ampl]
                rng = np.random.default_rng(1+i)
                jitter = (rng.random(size=xs.size) * 2 - 1) * jitter_ampl
                ax.scatter(xs, y[i] + jitter, s=s, alpha=alpha, edgecolors='none', color=scatter_colors[i])

    # global cosmetics
    ax.axvline(0, lw=1, ls='--', color='gray', zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(np.array(names))
    ax.invert_yaxis()
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=6)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return ax