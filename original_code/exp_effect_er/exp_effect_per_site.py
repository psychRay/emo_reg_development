#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 08:58:10 2026

@author: dingrui
"""

import os
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rcParams

#%% FUNCTIONS
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

def posthoc_each_site(
        df, 
        dv="y", 
        subject="subject", 
        within="cognition", between="gender",
        site_col="site", 
        padjust="holm"
):
    out = []
    for site, d in df.groupby(site_col, observed=True):
        ph = pg.pairwise_tests(
            data=d,
            dv=dv,
            within=within,
            between=between,
            subject=subject,
            padjust=padjust,
            effsize="hedges",
            interaction=True,
            marginal=True,      # recommended in mixed design :contentReference[oaicite:3]{index=3}
            within_first=True   # within * between :contentReference[oaicite:4]{index=4}
        )
        ph.insert(0, site_col, site)
        out.append(ph)
    return pd.concat(out, ignore_index=True)

def cognition_contrast_effect_by_site(
    df,
    site_col="site",
    subject="subject",
    within="cognition",
    dv="y",
    level_a="cognition1",
    level_b="cognition2",
    ci_method="parametric",   # "parametric" (compute_esci) or "bootstrap"
    n_boot=5000,
    seed=42
):
    rows = []
    for site, d in df.groupby(site_col, observed=True):
        # Align paired observations by subject (and average if multiple rows per cell)
        wide = (d[d[within].isin([level_a, level_b])]
                .pivot_table(index=subject, columns=within, values=dv, aggfunc="mean"))
        if level_a not in wide.columns or level_b not in wide.columns:
            continue
        wide = wide[[level_a, level_b]].dropna()
        n = wide.shape[0]
        if n < 2:
            continue

        x = wide[level_a].to_numpy()
        y = wide[level_b].to_numpy()

        # Paired t-test (post-hoc test)
        t_res = pg.ttest(x, y, paired=True)  # :contentReference[oaicite:9]{index=9}
        tval = float(t_res.loc["T_test", "T"])
        dof  = float(t_res.loc["T_test", "dof"])
        p    = float(t_res.loc["T_test", "p_val"])

        # Effect size (Hedges g) :contentReference[oaicite:10]{index=10}
        g = float(pg.compute_effsize(x, y, paired=True, eftype="hedges"))

        # CI for effect size
        if ci_method == "parametric":
            ci_low, ci_high = pg.compute_esci(
                stat=g, nx=n, ny=n, paired=True, eftype="cohen",
                confidence=0.95, decimals=6
            )  # :contentReference[oaicite:11]{index=11}
        elif ci_method == "bootstrap":
            # bootstrap CI using a custom bivariate function :contentReference[oaicite:12]{index=12}
            def hedges_g(x_, y_):
                return pg.compute_effsize(x_, y_, paired=True, eftype="hedges")
            ci_low, ci_high = pg.compute_bootci(
                x, y, func=hedges_g, paired=True, n_boot=n_boot, seed=seed, method="percentile"
            )
        else:
            raise ValueError("ci_method must be 'parametric' or 'bootstrap'")

        rows.append({
            "site": site,
            "contrast": f"{level_a} - {level_b}",
            "n": n,
            "T": tval,
            "dof": dof,
            "p_unc": p,
            "g": g,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        })

    res = pd.DataFrame(rows)
    return res

def forest_plot(
    df_eff,
    label_col="site",
    effect_col="g",
    low_col="ci_low",
    high_col="ci_high",
    n_col="n",
    sort=True,
    title=None,
    xlabel="Effect sizes with 95% Confidence Intervals",
    ref_line=0.0,
    ref_line_color="tab:blue",
    grid_color="0.8",
    grid_ls=(0, (4, 4)),     # dashed like the example
    ci_lw=3,
    square_edgecolor="black",
    square_facecolor="black",
    size_range=(100, 200),    # marker area range (points^2) for squares

    # ---- pooled control ----
    pooled_label_value="Pooled",  # identify pooled row by label_col == pooled_label_value
    pooled_col=None,              # alternatively: a boolean column name, e.g., "is_pooled"
    pooled_at_bottom=True,        # move pooled row(s) to the bottom
    pooled_marker_size=250,       # marker area (points^2) for diamond
    pooled_edgecolor="black",
    pooled_facecolor="black",
    pooled_linewidth=0.8,

    xlim=None,
    figsize=None,
    y_label_pad=28,
):
    """
    Forest plot in the style of the provided example figure.
    - Squares encode per-site estimates; square AREA encodes sample size (n_col).
    - Pooled estimate(s) are drawn as diamond(s) without confidence interval.

    Notes
    -----
    - Pooled rows are identified either by:
        1) pooled_col: a boolean column (True means pooled), OR
        2) label_col == pooled_label_value
    """

    d = df_eff.copy()

    # Keep rows that have an effect value and a label.
    d = d.dropna(subset=[label_col, effect_col])

    # Identify pooled rows
    if pooled_col is not None and pooled_col in d.columns:
        pooled_mask = d[pooled_col].astype(bool).to_numpy()
    else:
        pooled_mask = (d[label_col].astype(str) == str(pooled_label_value)).to_numpy()

    d_pooled = d[pooled_mask].copy()
    d_main = d[~pooled_mask].copy()

    # For non-pooled rows, we still need CI to draw lines
    d_main = d_main.dropna(subset=[low_col, high_col])

    # Sorting
    if sort:
        d_main = d_main.sort_values(effect_col, ascending=True)

    # Put pooled at bottom (typical forest-plot style)
    if pooled_at_bottom:
        d_plot = np.concatenate([d_main.index.to_numpy(), d_pooled.index.to_numpy()])
        d = d.loc[d_plot].copy()
    else:
        # keep original order except for sorting d_main (pooled stays where it was)
        # rebuild d with d_main (sorted) and pooled in original relative order
        d = pd.concat([d_main, d_pooled], axis=0)

    # Recompute pooled mask on final d
    if pooled_col is not None and pooled_col in d.columns:
        pooled_mask = d[pooled_col].astype(bool).to_numpy()
    else:
        pooled_mask = (d[label_col].astype(str) == str(pooled_label_value)).to_numpy()

    y = np.arange(len(d))

    # ---- marker sizes for squares (area) mapped from n ----
    if n_col is not None and n_col in d.columns:
        n_all = d[n_col].astype(float).to_numpy()
        n_main = n_all[~pooled_mask]
        if n_main.size > 0:
            n_min, n_max = np.nanmin(n_main), np.nanmax(n_main)
            if np.isfinite(n_min) and np.isfinite(n_max) and (n_max - n_min) > 1e-12:
                s_main = size_range[0] + (n_main - n_min) / (n_max - n_min) * (size_range[1] - size_range[0])
            else:
                s_main = np.full(n_main.shape, np.mean(size_range), dtype=float)
        else:
            s_main = np.array([], dtype=float)
    else:
        s_main = np.full((~pooled_mask).sum(), np.mean(size_range), dtype=float)

    # ---- figure size ----
    if figsize is None:
        figsize = (10, max(3.5, 0.9 * len(d) + 1.5))
    fig, ax = plt.subplots(figsize=figsize)

    # ---- vertical gridlines (x-axis major grid) ----
    ax.set_axisbelow(True)
    ax.grid(axis="x", which="major", linestyle=grid_ls, color=grid_color, linewidth=1.0)

    # ---- CI lines for non-pooled rows only ----
    y_main = y[~pooled_mask]
    if y_main.size > 0:
        lows = d.loc[~pooled_mask, low_col].to_numpy(dtype=float)
        highs = d.loc[~pooled_mask, high_col].to_numpy(dtype=float)
        ax.hlines(y_main, lows, highs, color="black", linewidth=ci_lw)

    # ---- square markers for non-pooled ----
    effects_main = d.loc[~pooled_mask, effect_col].to_numpy(dtype=float)
    if effects_main.size > 0:
        ax.scatter(
            effects_main, y_main,
            s=s_main, marker="s",
            facecolor=square_facecolor,
            edgecolor=square_edgecolor,
            linewidth=0.8,
            zorder=3
        )

    # ---- diamond markers for pooled (NO CI) ----
    y_pooled = y[pooled_mask]
    effects_pooled = d.loc[pooled_mask, effect_col].to_numpy(dtype=float)
    if effects_pooled.size > 0:
        ax.scatter(
            effects_pooled, y_pooled,
            s=pooled_marker_size, marker="D",
            facecolor=pooled_facecolor,
            edgecolor=pooled_edgecolor,
            linewidth=pooled_linewidth,
            zorder=4
        )

    # ---- reference line at 0 (solid) ----
    ax.axvline(ref_line, color=ref_line_color, linewidth=2.0)

    # ---- y labels ----
    ax.set_yticks(y)
    ax.set_yticklabels(d[label_col].astype(str).tolist(), fontsize=18)
    ax.tick_params(axis="y", length=0, pad=y_label_pad)
    ax.invert_yaxis()

    # ---- x axis limits ----
    if xlim is None:
        # base xlim on non-pooled CI; fallback to effect values if CI missing
        xmin_candidates = []
        xmax_candidates = []
        if (~pooled_mask).sum() > 0:
            xmin_candidates.append(np.nanmin(d.loc[~pooled_mask, low_col].to_numpy(dtype=float)))
            xmax_candidates.append(np.nanmax(d.loc[~pooled_mask, high_col].to_numpy(dtype=float)))
        if d.shape[0] > 0:
            xmin_candidates.append(np.nanmin(d[effect_col].to_numpy(dtype=float)))
            xmax_candidates.append(np.nanmax(d[effect_col].to_numpy(dtype=float)))

        xmin = np.nanmin(xmin_candidates)
        xmax = np.nanmax(xmax_candidates)
        pad = 0.06 * (xmax - xmin) if xmax > xmin else 1.0
        xlim = (xmin - pad, xmax + pad)

    ax.set_xlim(*xlim)

    ax.tick_params(axis="x", labelsize=16, length=10, width=1.2)
    ax.set_xlabel(xlabel, fontsize=18)

    # ---- spines ----
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)

    if title:
        ax.set_title(title, fontsize=20, fontweight="bold", pad=18)

    fig.tight_layout()
    plt.show()
    
    return fig, ax

def plot_forest(
    data,
    estimate_col="estimate",
    lower_col="ci_lower",
    upper_col="ci_upper",
    label_col="type",
    group_col=None,
    label_order=None,
    group_order=None,
    show_ref_line=False,
    ref_line=1.0,
    colors=None,
    interval_height=0.22,
    point_size=90,
    xlim=None,
    xlabel=None,
    title=None,
    figsize=None,
    ax=None,
    legend=True,
):
    """
    Draw a forest plot from a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data. Must contain estimate, ci_lower, ci_upper, and label columns.
    estimate_col : str
        Column name for estimated values.
    lower_col : str
        Column name for lower confidence interval.
    upper_col : str
        Column name for upper confidence interval.
    label_col : str
        Column name for row labels, e.g., outcomes.
    group_col : str or None
        Column name for groups. If None, each row is plotted directly.
    label_order : list or None
        Explicit order of y-axis labels from top to bottom.
    group_order : list or None
        Explicit order of groups.
    ref_line : float
        Reference vertical line, commonly 0 or 1.
    colors : dict, list, or None
        Colors for groups. If dict, keys should be group names.
    interval_height : float
        Thickness of confidence interval bars.
    point_size : float
        Size of point markers.
    xlim : tuple or None
        X-axis limits.
    xlabel : str or None
        X-axis label.
    title : str or None
        Plot title.
    figsize : tuple or None
        Figure size.
    ax : matplotlib.axes.Axes or None
        Existing axes. If None, a new figure and axes are created.
    legend : bool
        Whether to show legend when group_col is provided.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    df = data.copy()

    required_cols = [estimate_col, lower_col, upper_col, label_col]
    if group_col is not None:
        required_cols.append(group_col)

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Drop rows with missing plotting values
    df = df.dropna(subset=[estimate_col, lower_col, upper_col, label_col])

    if df.empty:
        raise ValueError("No valid rows available for plotting.")

    # Define label order from top to bottom
    if label_order is None:
        label_order = list(pd.unique(df[label_col]))
    else:
        missing_labels = set(df[label_col]) - set(label_order)
        if missing_labels:
            raise ValueError(f"Some labels are not included in label_order: {missing_labels}")

    # Reverse numeric positions so the first label appears on top
    y_base = {label: i for i, label in enumerate(label_order[::-1])}

    if figsize is None:
        figsize = (8, max(3.5, 0.7 * len(label_order)))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if group_col is not None:
        if group_order is None:
            group_order = list(pd.unique(df[group_col]))
        else:
            missing_groups = set(df[group_col]) - set(group_order)
            if missing_groups:
                raise ValueError(f"Some groups are not included in group_order: {missing_groups}")

        n_groups = len(group_order)

        # Default color cycle
        if colors is None:
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            colors = {
                group: color_cycle[i % len(color_cycle)]
                for i, group in enumerate(group_order)
            }
        elif isinstance(colors, list):
            colors = {
                group: colors[i % len(colors)]
                for i, group in enumerate(group_order)
            }

        # Vertical offsets for grouped plotting
        if n_groups == 1:
            offsets = [0.0]
        else:
            offsets = np.linspace(
                -interval_height * 0.8,
                interval_height * 0.8,
                n_groups
            )

        group_offsets = dict(zip(group_order, offsets))

        for group in group_order:
            sub = df[df[group_col] == group]

            for _, row in sub.iterrows():
                label = row[label_col]
                y = y_base[label] + group_offsets[group]

                lower = row[lower_col]
                upper = row[upper_col]
                estimate = row[estimate_col]

                # Draw confidence interval as a thick horizontal bar
                ax.barh(
                    y=y,
                    width=upper - lower,
                    left=lower,
                    height=interval_height,
                    color=colors[group],
                    alpha=0.45,
                    edgecolor="none",
                    label=str(group) if _ == sub.index[0] else None,
                    zorder=2,
                )

                # Draw estimate point
                ax.scatter(
                    estimate,
                    y,
                    s=point_size,
                    color=colors[group],
                    edgecolor="red" if label=='Pooled' else "white",
                    linewidth=1.3,
                    zorder=3,
                )

    else:
        # One row equals one forest plot entry
        color = colors[0] if isinstance(colors, list) else "C0"

        for _, row in df.iterrows():
            label = row[label_col]
            y = y_base[label]

            lower = row[lower_col]
            upper = row[upper_col]
            estimate = row[estimate_col]

            ax.barh(
                y=y,
                width=upper - lower,
                left=lower,
                height=interval_height,
                color=color,
                alpha=0.45,
                edgecolor="none",
                zorder=2,
            )

            ax.scatter(
                estimate,
                y,
                s=point_size,
                color=color,
                edgecolor="red" if label=='Pooled' else "white",
                linewidth=1.3,
                zorder=3,
            )

    # Reference line
    if show_ref_line:
        ax.axvline(
            ref_line,
            color="black",
            linestyle="--",
            linewidth=1.6,
            zorder=1
        )

    # Axis settings
    ax.set_yticks(list(y_base.values()))
    ax.set_yticklabels(list(y_base.keys()))

    if xlim is not None:
        ax.set_xlim(xlim)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if title is not None:
        ax.set_title(title)

    # Grid and styling
    ax.grid(axis="x", color="#eeeeee", linewidth=1, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=5)

    if group_col is not None and legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            frameon=False,
            loc="best"
        )

    fig.tight_layout()
    return fig, ax

def edge_list_to_adjacency(
    edge_df,
    source_col="source",
    target_col="target",
    weight_col="weight",
    directed=False,
    fill_value=0,
    aggfunc="sum",
    sort_nodes=True,
):
    """
    Convert an edge list to an adjacency matrix.

    Parameters
    ----------
    edge_df : pandas.DataFrame
        Input edge list.
    source_col : str
        Column name for source nodes.
    target_col : str
        Column name for target nodes.
    weight_col : str or None
        Column name for edge weights. If None, all edges are assigned weight 1.
    directed : bool
        Whether the graph is directed.
    fill_value : int or float
        Value used when no edge exists.
    aggfunc : str or callable
        Aggregation function for duplicated edges, e.g., "sum", "mean", "max".
    sort_nodes : bool
        Whether to sort node labels.

    Returns
    -------
    adj_df : pandas.DataFrame
        Adjacency matrix.
    """

    df = edge_df.copy()

    if source_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"edge_df must contain '{source_col}' and '{target_col}' columns.")

    if weight_col is None:
        df["_weight"] = 1
        weight_col = "_weight"
    elif weight_col not in df.columns:
        raise ValueError(f"edge_df must contain '{weight_col}' column.")

    nodes = list(pd.unique(pd.concat([df[source_col], df[target_col]], ignore_index=True)))
    if sort_nodes:
        nodes = sorted(nodes)

    if not directed:
        # Canonicalize undirected edges:
        # A-B and B-A are treated as the same edge.
        edge_min = df[[source_col, target_col]].min(axis=1)
        edge_max = df[[source_col, target_col]].max(axis=1)

        df["_source"] = edge_min
        df["_target"] = edge_max

        grouped = (
            df
            .groupby(["_source", "_target"], as_index=False)[weight_col]
            .agg(aggfunc)
        )

        # Create both directions to make the adjacency matrix symmetric
        reversed_grouped = grouped.rename(
            columns={
                "_source": "_target",
                "_target": "_source"
            }
        )

        full_edges = pd.concat([grouped, reversed_grouped], ignore_index=True)

        adj_df = full_edges.pivot_table(
            index="_source",
            columns="_target",
            values=weight_col,
            aggfunc=aggfunc,
            fill_value=fill_value
        )

    else:
        adj_df = df.pivot_table(
            index=source_col,
            columns=target_col,
            values=weight_col,
            aggfunc=aggfunc,
            fill_value=fill_value
        )

    adj_df = adj_df.reindex(
        index=nodes,
        columns=nodes,
        fill_value=fill_value
    )

    # Optional: remove self-loop values from the diagonal
    # np.fill_diagonal(adj_df.values, fill_value)

    return adj_df

#%% DATA
# Prepare Data Input

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
    
#%% STAT

# mixed anova on whole dataset
aov = pg.mixed_anova(
    data=df_data_ER,
    dv="emot_rating",
    within="cognition",
    between="gender",
    subject="sub_id",
    correction="auto",   # auto sphericity check + GG correction when needed
    effsize="np2"
)

# mixed anova per site
ph_all_sites = posthoc_each_site(
    df=df_data_ER,
    dv="emot_rating", 
    subject="sub_id", 
    within="cognition", between="gender",
    site_col="site_region", 
    padjust="holm"
)

# calculate effect size and 95% CI
es_site = cognition_contrast_effect_by_site(
    df=df_data_ER,
    site_col="site_region",
    subject="sub_id",
    within="cognition",
    dv="emot_rating",
    level_a="rpsl",
    level_b="lkng",
    ci_method="parametric",   # "parametric" (compute_esci) or "bootstrap"
    n_boot=5000,
)

#%% FOREST PLOT

# fig properties
set_pub_style(
    base_fontsize=16,
    font_family="DejaVu Sans",   # change to 'Arial' if your environment has it
    linewidth=1.8,
    tick_width=1.2,
    grid_alpha=0.25,
    dpi=300
)

# forest plot
d = es_site.dropna(subset=["g", "ci_low", "ci_high"]).copy()
z = 1.96
d["se"] = (d["ci_high"] - d["ci_low"]) / (2 * z)
d["w"] = 1 / (d["se"] ** 2)

pooled = (d["w"] * d["g"]).sum() / d["w"].sum()
pooled_se = np.sqrt(1 / d["w"].sum())
pooled_low, pooled_high = pooled - z * pooled_se, pooled + z * pooled_se

pooled_row = pd.DataFrame([{
    "site": "Pooled",
    "g": pooled,
    "ci_low": pooled_low,
    "ci_high": pooled_high
}])

es_for_plot = pd.concat([es_site, pooled_row], ignore_index=True)

forest_plot(
    es_for_plot, 
    label_col="site", 
    effect_col="g", 
    low_col="ci_low", high_col="ci_high",
    sort=False, 
    figsize=(8, 7),
    title=None,
    pooled_edgecolor="red",
    pooled_facecolor="white",
    pooled_linewidth=3,
)

#%% FOREST PLOT 2
fig, ax = plot_forest(
    es_for_plot,
    estimate_col='g',
    lower_col='ci_low',
    upper_col='ci_high',
    label_col='site',
    interval_height=0.3,
    point_size=90,
    legend=False,
    colors=['#279F29'],
    xlim=(0.4, 1.6),
    xlabel="Hedges' g",
    figsize=(5.5, 4), 
)

ax.tick_params(axis='both', pad=10)
plt.xticks(
    ticks=[0.4, 0.4, 0.8, 1.2, 1.6],
    labels=[0.4, 0.4, 0.8, 1.2, 1.6]
)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#%% DISTRIBUTION OF EXPERIMENTAL DATA (emotion ratings)
fig, ax = plt.subplots(figsize=(3,3))
hist = sns.histplot(
    df_data_ER, stat='density', x='emot_rating', 
    hue='cognition', hue_order=['rpsl', 'lknt', 'lkng'], 
    palette={'rpsl': '#73C477', 'lknt': '#867cd8', 'lkng': '#E17F7E'},
    multiple='layer', discrete=False,
    bins=10, binwidth=0.07, element='bars', linewidth=0, alpha=1,
    kde=True, line_kws={'linewidth':3},
    ax=ax, legend=False,
)

sns.kdeplot(
    df_data_ER, x='emot_rating',
    hue='cognition', hue_order=['rpsl', 'lknt', 'lkng'],
    palette={'rpsl': 'green', 'lknt': 'purple', 'lkng': 'red'},
    ax=hist, legend=False,
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(
    ticks=[1, 2, 3, 4],
    labels=[1, 2, 3, 4]
)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#%% ANOVA OF AGES AMONG SITE/REGIONS

# data for anova and posthoc tests
df_age_by_region = df_data_ER.groupby(by='sub_id').agg(
    age=('age', 'mean'), site_region=('site_region', 'first')
    )

# stats
aov = pg.anova(data=df_age_by_region, dv='age', between='site_region', detailed=True)
posthoc = pg.pairwise_tests(
    data=df_age_by_region, dv='age', between='site_region', 
    parametric=True, padjust='fdr_bh', effsize='hedges'
)

# plot pairwsie p for posthoc ttets
df_adj_p = edge_list_to_adjacency(
    posthoc[['A', 'B', 'p_unc']],
    source_col='A',
    target_col='B',
    weight_col='p_unc',
    directed=False,
    fill_value=1,
)

mask = np.triu(np.ones_like(df_adj_p, dtype=bool))
fig, ax = plt.subplots(figsize=(3, 5))
sns.heatmap(
    df_adj_p, mask=~mask,
    robust=True,
    cmap='viridis_r', cbar=False,
    linecolor='white', linewidths=0,
    annot=False, square=False,
    xticklabels=False, yticklabels=False,
    ax=ax,
)

#%%
