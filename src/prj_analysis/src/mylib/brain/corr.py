#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 04:16:24 2025

    sorts of tools for correction of inflation of false positive induced by multiple comparisons in 
    brain image analysis

@author: dingrui
"""

# import modules
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from scipy.stats import t as t_dist


# functions
def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not np.any(mask):
        return q
    p_valid = p[mask]
    m = p_valid.size
    order = np.argsort(p_valid)
    p_sorted = p_valid[order]
    ranks = np.arange(1, m+1)
    q_sorted = p_sorted * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    q_vals = np.empty_like(p_valid)
    q_vals[order] = q_sorted
    q[mask] = q_vals
    return q

def _bonferroni_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not np.any(mask):
        return out
    m = np.sum(mask)
    adj = p[mask] * m
    adj = np.minimum(adj, 1.0)
    out[mask] = adj
    return out

def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not np.any(mask):
        return out
    p_valid = p[mask]
    m = p_valid.size
    order = np.argsort(p_valid)
    p_sorted = p_valid[order]
    factors = (m - np.arange(0, m))
    adj_candidates = p_sorted * factors
    adj_cummax = np.maximum.accumulate(adj_candidates)
    adj_cummax = np.minimum(adj_cummax, 1.0)
    adj_unsorted = np.empty_like(adj_cummax)
    adj_unsorted[order] = adj_cummax
    out[mask] = adj_unsorted
    return out

# ---- small util to get nested keys from dict-like results ----
def _get_by_path(d: Dict[str, Any], path: Union[str, Iterable[str]]) -> Any:
    """
    Safe accessor: path can be a single key (str) or an iterable of nested keys.
    Returns None if any step missing.
    """
    if isinstance(path, str):
        return d.get(path, None)
    cur = d
    try:
        for k in path:
            if isinstance(cur, dict) and (k in cur):
                cur = cur[k]
            else:
                return None
        return cur
    except Exception:
        return None

# ---- main flexible multi-compare function ----
def multi_compare_correction_flex(
    results: Optional[Dict[str, Any]] = None,
    *,
    pvals: Optional[Sequence[float]] = None,
    p_key: str = "p_age",
    p_path: Optional[Union[str, Iterable[str]]] = None,
    t_path: Optional[Union[str, Iterable[str]]] = None,
    dof_path: Optional[Union[str, Iterable[str]]] = None,
    methods: Sequence[str] = ("fdr_bh", "bonferroni", "holm"),
    alpha: float = 0.05,
    two_sided: bool = True,
    in_place: bool = True
) -> Dict[str, Any]:
    """
    Flexible multiple-comparison correction.

    Parameters
    ----------
     - results : optional dict, If provided, we will read and (by default) write corrected fields into it.
     - pvals : optional array-like, Direct p-values to correct. If provided, used with highest priority.
     - p_key : str, Prefix for output fields (e.g., 'p_age'); used when writing keys back into `results`.
     - p_path : optional key or nested-key iterable to read existing p-values from `results` (if pvals not supplied).
     - t_path : optional key or nested-key iterable to read t-statistics from `results` (used to compute p if p unavailable).
     - dof_path : optional key or nested-key iterable to read degrees of freedom for p calculation (if needed).
     - methods : tuple/list of correction methods to compute (subset of 'fdr_bh','bonferroni','holm')
     - alpha : float, significance threshold for computing reject masks.
     - two_sided : bool, If computing p from t, whether to produce two-sided p (default True).
     - in_place : bool, If True and results provided, write corrected fields into results and return results.
                  Otherwise return a new dict with correction outputs.

    Returns
    -------
    out : dict
        The dict that contains original p (copied) and adjusted values/boolean rejects for requested methods.
    """
    if results is None and pvals is None:
        raise ValueError("Provide either results (with p/t entries) or pvals directly.")

    # determine p array
    if pvals is not None:
        p_arr = np.asarray(pvals, dtype=float)
    else:
        # try to read p from results via p_path
        if results is None:
            raise ValueError("results is None and pvals not provided.")
        p_from_res = None
        if p_path is not None:
            p_from_res = _get_by_path(results, p_path)
        # fallback: try p_key in top-level
        if p_from_res is None:
            p_from_res = results.get(p_key, None)
        if p_from_res is not None:
            p_arr = np.asarray(p_from_res, dtype=float)
        else:
            # need to compute p from t and df
            if t_path is None:
                raise KeyError("No p found and no t_path provided to compute p. Provide t_path and df_path.")
            t_arr = _get_by_path(results, t_path)
            if t_arr is None:
                raise KeyError(f"t not found at path {t_path} in results.")
            t_arr = np.asarray(t_arr, dtype=float)
            if dof_path is None:
                # try common keys
                df_cand = None
                for kname in ("dof_per_roi", "dof", "df", "N_eff_per_roi"):
                    df_cand = results.get(kname, None)
                    if df_cand is not None:
                        df_arr = np.asarray(df_cand, dtype=float)
                        break
                else:
                    raise KeyError("dof_path not provided and no common df key found in results.")
            else:
                df_cand = _get_by_path(results, dof_path)
                if df_cand is None:
                    raise KeyError(f"df not found at path {dof_path} in results.")
                df_arr = np.asarray(df_cand, dtype=float)

            # compute two-sided p
            if two_sided:
                p_arr = 2.0 * t_dist.sf(np.abs(t_arr), df_arr)
            else:
                # right tail
                p_arr = t_dist.sf(t_arr, df_arr)

    # ensure 1D vector
    p_arr = np.asarray(p_arr).ravel()
    out = {} if (results is None or not in_place) else results

    # store original p copy
    out[f"{p_key}_p"] = p_arr.copy()

    if "fdr_bh" in methods:
        q = _bh_adjust(p_arr)
        out[f"{p_key}_q_bh"] = q
        out[f"{p_key}_rej_bh"] = np.isfinite(q) & (q <= alpha)

    if "bonferroni" in methods:
        p_b = _bonferroni_adjust(p_arr)
        out[f"{p_key}_p_bonf"] = p_b
        out[f"{p_key}_rej_bonf"] = np.isfinite(p_b) & (p_b <= alpha)

    if "holm" in methods:
        p_h = _holm_adjust(p_arr)
        out[f"{p_key}_p_holm"] = p_h
        out[f"{p_key}_rej_holm"] = np.isfinite(p_h) & (p_h <= alpha)

    return out if (results is None or not in_place) else results

