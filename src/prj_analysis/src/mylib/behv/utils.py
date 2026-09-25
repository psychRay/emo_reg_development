#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 00:39:00 2025

@author: dingrui
"""

# import necessary modules
# import os
import warnings
warnings.filterwarnings('ignore', message='.*deprecated.*')

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import pingouin as pg

from numpy.random import default_rng
# from statsmodels.formula.api import mixedlm
# from statsmodels.stats.anova import AnovaRM
# from statsmodels.stats.multitest import multipletests
from patsy import dmatrix, dmatrices, build_design_matrices
from scipy.stats import chi2, ttest_rel, norm, t

# from tqdm import tqdm
# from itertools import combinations

# define functions
def wald_block(X, col_names, params, cov):
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

def X_from_new(new_df, design_info, fe_names):
    new_df = new_df.copy()
    #
    Xnew = build_design_matrices([design_info], new_df)[0]
    Xnew = pd.DataFrame(Xnew, columns=design_info.column_names, index=new_df.index)
    # align with fixed effect
    Xnew = Xnew.reindex(columns=fe_names, fill_value=0.0)
    return Xnew

def predict_curve(df_data, cond, gender, ages, design_info, fe_params, fe_cov):
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

def contrast_diff(df_data, design_info, condA, condB, gender, ages, fe_params, fe_cov):
    """
    Compute (A - B) contrast along ages with CI and z/p (Wald normal approx).
    Returns a DataFrame with columns: age, diff, se, z, p, lower, upper.
    """
    newA = pd.DataFrame({"cognition": condA, "gender": gender, "age_c": [to_age_c(df_data, a) for a in ages]})
    newB = pd.DataFrame({"cognition": condB, "gender": gender, "age_c": [to_age_c(df_data, a) for a in ages]})
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
    Build L for (High − Low) at given gender and ages
    """
    
    newA = pd.DataFrame({"cognition": condA, 
                         "gender": gender,
                         "age_c": [to_age_c(df_data, a) for a in ages]})
    newB = pd.DataFrame({"cognition": condB, 
                         "gender": gender,
                         "age_c": [to_age_c(df_data, a) for a in ages]})
    
    XA = X_from_new(newA, design_info).to_numpy()
    XB = X_from_new(newB, design_info).to_numpy()
    return XA - XB  # (m x p)

def johnson_neyman_simultaneous(
        L: np.ndarray,
        fe_params: np.ndarray,
        fe_cov: np.ndarray,
        alpha: float = 0.05,
        n_draws: int = 2000,
        seed: int = 0):
    
    """
    Simultaneous J–N via parametric bootstrap (supremum of standardized deviations)
    
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

