#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:30:39 2026

@author: dingrui

GAM trajectory turning-point analysis for pyGAM 0.10.1.

This script supports:
1. Within-subject contrasts: rpsl-lkng and lkng-lknt
2. Gender-specific GAM trajectories
3. Posterior sampling of GAM coefficients
4. Derivative sign-change based turning points
5. KDE-based final turning point selection
6. Posterior median age, 95% CI, and sign-change posterior probability
7. Age-shuffle permutation test for observed final turning points

Required columns in the original long dataframe:
    sub_id, gender, age, cognition, dv

For contrast-level dataframe:
    sub_id, gender, age, contrast, dv_diff
"""

#%%
import io, os
import contextlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from pygam import LinearGAM, s, f
from brokenaxes import brokenaxes

try:
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
except ImportError as e:
    raise ImportError(
        "This script requires scipy. Please install it with: pip install scipy"
    ) from e


#%%
# =============================================================================
# 1. Data preparation/preprocessing
# =============================================================================

def make_within_subject_contrasts(
    df,
    dv_col="dv",
    subj_col="sub_id",
    age_col="age",
    gender_col="gender",
    cond_col="cognition",
    contrasts=(("rpsl", "lkng"), ("lkng", "lknt")),
    dropna_subjects=True,
):
    """
    Convert long-format within-subject data to contrast-level data.

    Input long dataframe:
        one row per subject-condition observation.

    Output dataframe:
        one row per subject-contrast observation.
        columns: sub_id, age, gender, contrast, dv_diff
    """
    required = {subj_col, age_col, gender_col, cond_col, dv_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    meta = (
        df[[subj_col, age_col, gender_col]]
        .drop_duplicates(subset=[subj_col])
        .set_index(subj_col)
    )

    wide = (
        df[[subj_col, cond_col, dv_col]]
        .pivot(index=subj_col, columns=cond_col, values=dv_col)
    )

    if dropna_subjects:
        needed_conds = sorted(set([x for pair in contrasts for x in pair]))
        wide = wide.dropna(subset=needed_conds)

    wide = wide.join(meta, how="inner")

    out = []
    for a, b in contrasts:
        if a not in wide.columns or b not in wide.columns:
            raise ValueError(f"Cannot compute contrast {a}-{b}: missing {a} or {b}")

        tmp = pd.DataFrame({
            "sub_id": wide.index,
            "age": np.asarray(wide[age_col], dtype=float),
            "gender": wide[gender_col].values,
            "contrast": f"{a}_minus_{b}",
            "dv_diff": np.asarray(wide[a], dtype=float) - np.asarray(wide[b], dtype=float),
        })
        out.append(tmp)

    return pd.concat(out, axis=0, ignore_index=True)


def _residualize_y_against_covariates(
    df,
    y_col,
    covariate_cols,
    add_mean=True,
    drop_first=True,
):
    """
    Residualize y against covariates using ordinary least squares.

    This is a wrapper-level workaround that allows the existing GAM+TP pipeline
    to handle covariate-adjusted outcomes without rewriting the GAM fitting
    function.

    Continuous covariates are used as numeric columns.
    Non-numeric covariates are dummy-coded.

    Returns
    -------
    y_adj : ndarray
        Residualized y. If add_mean=True, mean(y) is added back.
    """
    y = np.asarray(df[y_col], dtype=float)

    X_parts = []

    for col in covariate_cols:
        x = df[col]

        if pd.api.types.is_numeric_dtype(x):
            X_parts.append(pd.DataFrame({col: np.asarray(x, dtype=float)}))
        else:
            xd = pd.get_dummies(x, prefix=col, drop_first=drop_first)
            X_parts.append(xd)

    if len(X_parts) == 0:
        return y.copy()

    X = pd.concat(X_parts, axis=1)
    X = X.astype(float)

    # Add intercept
    X_mat = np.column_stack([
        np.ones(len(X)),
        X.values,
    ])

    mask = np.isfinite(y) & np.all(np.isfinite(X_mat), axis=1)

    if mask.sum() < X_mat.shape[1] + 2:
        raise ValueError(
            "Too few valid observations for covariate residualization."
        )

    beta, *_ = np.linalg.lstsq(X_mat[mask], y[mask], rcond=None)

    y_hat = X_mat @ beta
    resid = y - y_hat

    if add_mean:
        resid = resid + np.nanmean(y[mask])

    return resid


# =============================================================================
# 2. Gender coding and GAM fitting
# =============================================================================

def _silent_summary(gam):
    """
    Trigger pygam.statistics_ creation without printing summary output.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            gam.summary()
        except Exception:
            pass


def _factorize_gender(genders, fixed_levels=None):
    """
    Factorize gender into 0..K-1 codes without changing the user's intended mapping.

    Example:
        fixed_levels = [0, 1]
        means original gender value 0 is code 0 and original gender value 1 is code 1.

    This avoids automatic sorting or order flipping.
    """
    s_ = pd.Series(genders)

    if fixed_levels is None:
        # Preserve order of appearance if no fixed level order is given.
        levels = list(pd.unique(s_))
    else:
        levels = list(fixed_levels)

    cat = pd.Categorical(s_, categories=levels, ordered=True)
    codes = np.asarray(cat.codes, dtype=int)
    codes[codes < 0] = 0  # Unknown level -> baseline

    return codes, levels


def build_X_for_info(df_like, info, age_col="age", gender_col="gender"):
    """
    Build a design matrix matching a fitted GAM model.

    For smooth model:
        X = [age, gender_code, I_1, ..., I_{K-1}]
    where I_k indicates gender code k.
    """
    g_levels = info["gender_levels"]
    cat = pd.Categorical(df_like[gender_col], categories=g_levels, ordered=True)
    g_codes = np.asarray(cat.codes, dtype=int)
    g_codes[g_codes < 0] = 0

    age = np.asarray(df_like[age_col], dtype=float)

    cols = [age, g_codes]
    for k in range(1, len(g_levels)):
        cols.append((g_codes == k).astype(float))

    return np.column_stack(cols)


def fit_gam_age_by_gender_smooth(
    df_contrast,
    age_col="age",
    gender_col="gender",
    y_col="dv_diff",
    fixed_gender_levels=(0, 1),
    lam_grid=None,
    n_splines=25,
    verbose=False,
):
    """
    Fit the smooth gender-specific age GAM:

        y ~ s(age) + f(gender) + sum s(age, by=I_k)

    For binary gender with fixed_gender_levels=[0,1]:
        code 0 is the reference trajectory.
        code 1 gets a deviation smooth from the reference trajectory.

    In pyGAM 0.10.1, s(..., by=...) expects a column index, not a vector.
    """
    if lam_grid is None:
        lam_grid = np.logspace(-3, 3, 12)

    age = np.asarray(df_contrast[age_col], dtype=float)
    y = np.asarray(df_contrast[y_col], dtype=float)
    g_codes, g_levels = _factorize_gender(
        df_contrast[gender_col],
        fixed_levels=fixed_gender_levels
    )

    mask = np.isfinite(age) & np.isfinite(y) & np.isfinite(g_codes)
    age, y, g_codes = age[mask], y[mask], g_codes[mask]

    if len(y) < 8:
        raise ValueError(f"Too few observations after cleaning: n={len(y)}")

    K = len(g_levels)

    cols = [age, g_codes]
    indicator_cols = []

    for k in range(1, K):
        Ik = (g_codes == k).astype(float)
        indicator_cols.append(len(cols))
        cols.append(Ik)

    X = np.column_stack(cols)

    terms = s(0, n_splines=n_splines) + f(1)

    for by_col in indicator_cols:
        terms += s(0, by=by_col, n_splines=n_splines)

    gam = LinearGAM(terms)
    gam = gam.gridsearch(X, y, lam=lam_grid, progress=verbose)

    # _silent_summary(gam)

    info = {
        "kind": "smooth",
        "gender_levels": g_levels,
        "idx": {
            "age": 0,
            "gender": 1,
            "I_cols": indicator_cols,
        },
        "X": X,
        "y": y,
        "g_codes": g_codes,
        "fixed_gender_levels": list(fixed_gender_levels),
    }

    return gam, X, y, g_codes, info


# =============================================================================
# 3. Posterior sampling of GAM trajectories
# =============================================================================

def _safe_mvn_sample(mean, cov, n_draws, rng, max_tries=8):
    """
    Draw samples from a multivariate normal distribution with numerical jitter.
    """
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)

    p = len(mean)
    jitter = 0.0

    for i in range(max_tries):
        try:
            cov_j = cov + jitter * np.eye(p)
            return rng.multivariate_normal(mean, cov_j, size=n_draws)
        except np.linalg.LinAlgError:
            jitter = 10 ** (-10 + i)

    # Final fallback: diagonal covariance only.
    diag = np.clip(np.diag(cov), a_min=np.finfo(float).eps, a_max=None)
    return rng.normal(loc=mean, scale=np.sqrt(diag), size=(n_draws, p))


def sample_gam_coef_posterior(gam, n_draws=1000, seed=2025):
    """
    Approximate posterior sampling of GAM coefficients.

    pyGAM stores coefficient covariance in:
        gam.statistics_['cov']

    This is an approximate Bayesian / large-sample Gaussian posterior.
    """
    # _silent_summary(gam)

    stats = getattr(gam, "statistics_", {}) or {}
    if "cov" not in stats:
        raise KeyError(
            "gam.statistics_ does not contain 'cov'. "
            "Call gam.summary() first or check the fitted pyGAM object."
        )

    rng = np.random.default_rng(seed)
    coef = np.asarray(gam.coef_, dtype=float)
    cov = np.asarray(stats["cov"], dtype=float)

    return _safe_mvn_sample(coef, cov, n_draws=n_draws, rng=rng)


def posterior_trajectory_samples(
    gam,
    info,
    gender_value,
    age_grid,
    n_draws=1000,
    seed=2025,
):
    """
    Generate posterior samples of the age trajectory for one gender level.

    Returns:
        curves: array, shape = (n_draws, n_age_grid)
    """
    coef_draws = sample_gam_coef_posterior(gam, n_draws=n_draws, seed=seed)

    pred_df = pd.DataFrame({
        "age": age_grid,
        "gender": [gender_value] * len(age_grid),
    })

    Xg = build_X_for_info(pred_df, info, age_col="age", gender_col="gender")
    model_matrix = gam._modelmat(Xg)

    curves = model_matrix.dot(coef_draws.T).T
    curves = np.asarray(curves, dtype=float)

    return curves


# =============================================================================
# 4. Turning point extraction from posterior trajectories
# =============================================================================

def _fill_zero_sign(sign):
    """
    Replace zero signs by nearest non-zero signs to stabilize sign-change detection.
    """
    sign = np.asarray(sign, dtype=float).copy()

    if np.all(sign == 0):
        return sign

    # Forward fill
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1]

    # Backward fill
    for i in range(len(sign) - 2, -1, -1):
        if sign[i] == 0:
            sign[i] = sign[i + 1]

    return sign


def extract_turning_point_candidates(
    curves,
    age_grid,
    derivative_eps=None,
    edge_exclusion=0.25,
    keep_kind="both",
):
    """
    Extract turning point candidates from posterior trajectory samples.

    A turning point is defined as a zero-crossing of the first derivative:
        + to - : peak
        - to + : valley

    Parameters
    ----------
    curves : ndarray, shape = (n_draws, n_grid)
        Posterior trajectory samples.
    age_grid : ndarray, shape = (n_grid,)
        Age grid.
    derivative_eps : float or None
        Small derivative values are treated as zero.
        If None, an adaptive threshold is used.
    edge_exclusion : float
        Exclude candidates within this age distance from grid boundaries.
    keep_kind : {'both', 'peak', 'valley'}
        Which type of turning point candidates to keep.

    Returns
    -------
    candidates : DataFrame
        columns: draw_id, age, kind
    """
    curves = np.asarray(curves, dtype=float)
    age_grid = np.asarray(age_grid, dtype=float)

    n_draws, n_grid = curves.shape
    deriv = np.gradient(curves, age_grid, axis=1)

    if derivative_eps is None:
        scale = np.nanmedian(np.abs(deriv))
        derivative_eps = max(scale * 1e-5, 1e-10)

    rows = []
    age_min, age_max = age_grid.min(), age_grid.max()

    for d in range(n_draws):
        dd = deriv[d].copy()
        dd[np.abs(dd) < derivative_eps] = 0.0

        sign = np.sign(dd)
        sign = _fill_zero_sign(sign)

        if np.all(sign == 0):
            continue

        for i in range(n_grid - 1):
            if sign[i] * sign[i + 1] < 0:
                # Linear interpolation of derivative zero-crossing.
                d0, d1 = dd[i], dd[i + 1]
                a0, a1 = age_grid[i], age_grid[i + 1]

                if np.isfinite(d0) and np.isfinite(d1) and (d1 - d0) != 0:
                    age_tp = a0 - d0 * (a1 - a0) / (d1 - d0)
                else:
                    age_tp = 0.5 * (a0 + a1)

                if not (age_min + edge_exclusion <= age_tp <= age_max - edge_exclusion):
                    continue

                if sign[i] > 0 and sign[i + 1] < 0:
                    kind = "peak"
                elif sign[i] < 0 and sign[i + 1] > 0:
                    kind = "valley"
                else:
                    kind = "unknown"

                if keep_kind != "both" and kind != keep_kind:
                    continue

                rows.append({
                    "draw_id": d,
                    "age": float(age_tp),
                    "kind": kind,
                })

    return pd.DataFrame(rows)


# =============================================================================
# 5. KDE-based final turning point selection
# =============================================================================

def _find_kde_modes(
    ages,
    age_min,
    age_max,
    kde_grid_size=512,
    min_peak_distance=0.75,
    density_rel_height=0.05,
    bw_method=None,
):
    """
    Find KDE density modes of turning point candidate ages.
    """
    ages = np.asarray(ages, dtype=float)
    ages = ages[np.isfinite(ages)]

    if len(ages) < 3 or len(np.unique(ages)) < 3:
        return np.array([])

    kde = gaussian_kde(ages, bw_method=bw_method)
    x_eval = np.linspace(age_min, age_max, kde_grid_size)
    dens = kde(x_eval)

    if not np.any(np.isfinite(dens)):
        return np.array([])

    dens_max = np.nanmax(dens)
    if dens_max <= 0:
        return np.array([])

    grid_step = x_eval[1] - x_eval[0]
    min_dist_index = max(1, int(np.round(min_peak_distance / grid_step)))

    peaks, props = find_peaks(
        dens,
        height=dens_max * density_rel_height,
        distance=min_dist_index
    )

    if len(peaks) == 0:
        peaks = np.array([int(np.nanargmax(dens))])

    modes = x_eval[peaks]
    heights = dens[peaks]

    order = np.argsort(heights)[::-1]
    return modes[order]


def kde_finalize_turning_points(
    candidates,
    n_draws,
    age_grid,
    separate_kind=True,
    min_sign_prob=0.15,
    min_count=30,
    assign_window=0.75,
    min_peak_distance=0.75,
    density_rel_height=0.05,
    bw_method=None,
    ci=0.95,
):
    """
    Turn posterior turning point candidates into final KDE-based turning points.

    For each KDE mode:
        - assign nearby candidate TPs to the mode
        - retain at most one TP per posterior draw
        - compute posterior median age
        - compute 95% CI
        - compute sign-change posterior probability

    sign_change_prob:
        number of posterior draws that have a TP near this final mode / n_draws
    """
    if candidates is None or len(candidates) == 0:
        return pd.DataFrame()

    candidates = candidates.copy()
    age_min, age_max = float(np.min(age_grid)), float(np.max(age_grid))

    if separate_kind:
        groups = list(candidates.groupby("kind"))
    else:
        groups = [("mixed", candidates)]

    alpha = 1.0 - ci
    rows = []

    for kind_label, dfk in groups:
        if len(dfk) < min_count:
            continue

        modes = _find_kde_modes(
            dfk["age"].values,
            age_min=age_min,
            age_max=age_max,
            min_peak_distance=min_peak_distance,
            density_rel_height=density_rel_height,
            bw_method=bw_method,
        )

        if len(modes) == 0:
            continue

        for mode_age in modes:
            tmp = dfk.copy()
            tmp["dist_to_mode"] = np.abs(tmp["age"] - mode_age)
            tmp = tmp[tmp["dist_to_mode"] <= assign_window]

            if len(tmp) < min_count:
                continue

            # Keep one closest candidate per posterior draw.
            tmp = (
                tmp.sort_values("dist_to_mode")
                .drop_duplicates(subset=["draw_id"], keep="first")
            )

            n_draws_with_tp = tmp["draw_id"].nunique()
            sign_prob = n_draws_with_tp / float(n_draws)

            if sign_prob < min_sign_prob:
                continue

            ages = np.asarray(tmp["age"], dtype=float)

            if len(ages) < min_count:
                continue

            age_median = np.nanmedian(ages)
            age_ci_low = np.nanquantile(ages, alpha / 2)
            age_ci_high = np.nanquantile(ages, 1 - alpha / 2)

            if separate_kind:
                tp_kind = kind_label
            else:
                tp_kind = tmp["kind"].value_counts().idxmax()

            rows.append({
                "tp_kind": tp_kind,
                "kde_mode_age": float(mode_age),
                "posterior_median_age": float(age_median),
                "age_ci_low": float(age_ci_low),
                "age_ci_high": float(age_ci_high),
                "sign_change_prob": float(sign_prob),
                "n_candidates_assigned": int(len(tmp)),
                "n_draws_with_tp": int(n_draws_with_tp),
            })

    if len(rows) == 0:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(["posterior_median_age", "tp_kind"]).reset_index(drop=True)
    return out


def local_sign_change_probability(
    candidates,
    target_age,
    n_draws,
    window=0.75,
    tp_kind=None,
):
    """
    Compute local sign-change posterior probability around a target age.

    This is used for permutation testing of observed final TPs.
    """
    if candidates is None or len(candidates) == 0:
        return 0.0

    tmp = candidates.copy()
    tmp = tmp[np.abs(tmp["age"] - target_age) <= window]

    if tp_kind is not None and tp_kind != "mixed":
        tmp = tmp[tmp["kind"] == tp_kind]

    if len(tmp) == 0:
        return 0.0

    return tmp["draw_id"].nunique() / float(n_draws)


# =============================================================================
# 6. One-contrast turning point analysis
# =============================================================================

def analyze_turning_points_one_contrast(
    df_contrast,
    gender_col='gender',
    fixed_gender_levels=(0, 1),
    gender_label_map=None,
    lam_grid=None,
    n_splines=25,
    age_grid_size=300,
    n_draws=1000,
    seed=2025,
    separate_kind=True,
    min_sign_prob=0.15,
    min_count=30,
    assign_window=0.75,
    min_peak_distance=0.75,
    edge_exclusion=0.25,
    keep_kind="both",
):
    """
    Fit one contrast-level GAM and estimate final trajectory turning points
    for each gender-specific trajectory.

    Returns
    -------
    result : dict
        {
          'gam': fitted pyGAM object,
          'info': model info,
          'age_grid': common age grid,
          'curves': dict gender_value -> posterior curves,
          'candidates': dict gender_value -> candidate TP dataframe,
          'final_tps': dataframe of final TPs
        }
    """
    if gender_label_map is None:
        gender_label_map = {g: str(g) for g in fixed_gender_levels}

    if lam_grid is None:
        lam_grid = np.logspace(-3, 3, 12)

    gam, X, y, g_codes, info = fit_gam_age_by_gender_smooth(
        df_contrast,
        gender_col=gender_col,
        fixed_gender_levels=fixed_gender_levels,
        lam_grid=lam_grid,
        n_splines=n_splines,
        verbose=False,
    )

    age_min = float(np.nanmin(df_contrast["age"]))
    age_max = float(np.nanmax(df_contrast["age"]))
    age_grid = np.linspace(age_min, age_max, age_grid_size)

    curves_by_gender = {}
    cand_by_gender = {}
    final_list = []

    for j, gval in enumerate(info["gender_levels"]):
        curves = posterior_trajectory_samples(
            gam=gam,
            info=info,
            gender_value=gval,
            age_grid=age_grid,
            n_draws=n_draws,
            seed=seed + 1000 * (j + 1),
        )

        cands = extract_turning_point_candidates(
            curves=curves,
            age_grid=age_grid,
            edge_exclusion=edge_exclusion,
            keep_kind=keep_kind,
        )

        final_tps = kde_finalize_turning_points(
            candidates=cands,
            n_draws=n_draws,
            age_grid=age_grid,
            separate_kind=separate_kind,
            min_sign_prob=min_sign_prob,
            min_count=min_count,
            assign_window=assign_window,
            min_peak_distance=min_peak_distance,
        )

        if len(final_tps) > 0:
            final_tps.insert(0, "gender", gval)
            final_tps.insert(1, "gender_label", gender_label_map.get(gval, str(gval)))

        curves_by_gender[gval] = curves
        cand_by_gender[gval] = cands
        final_list.append(final_tps)

    if len(final_list) > 0:
        final_all = pd.concat(final_list, axis=0, ignore_index=True)
    else:
        final_all = pd.DataFrame()

    return {
        "gam": gam,
        "info": info,
        "age_grid": age_grid,
        "curves": curves_by_gender,
        "candidates": cand_by_gender,
        "final_tps": final_all,
    }


# =============================================================================
# 7. Age-shuffle permutation test
# =============================================================================
def _shuffle_age_at_subject_level(
    df,
    subj_col="sub_id",
    gender_col="gender",
    age_col="age",
    fixed_gender_levels=(0, 1),
    rng=None,
):
    """
    Shuffle age at the subject level within each gender group.

    This function assumes that each subject has a single age value
    within the current df_contrast. If a subject has multiple rows,
    all rows from that subject receive the same permuted age.

    This is appropriate for:
        - within-subject contrast-level data
        - retest/repeated rows where age is constant per subject
        - row-level repeated observations nested within subject

    Parameters
    ----------
    df : pandas.DataFrame
        Input contrast-level dataframe.

    subj_col : str
        Subject ID column.

    gender_col : str
        Gender column.

    age_col : str
        Age column.

    fixed_gender_levels : list or tuple
        Gender levels within which age should be shuffled.

    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    dperm : pandas.DataFrame
        Dataframe with subject-level permuted age.
    """

    if rng is None:
        rng = np.random.default_rng()

    dperm = df.copy()

    # One row per subject: retrieve subject-level metadata.
    subj_meta = (
        dperm[[subj_col, gender_col, age_col]]
        .drop_duplicates(subset=[subj_col])
        .copy()
    )

    # Check whether each subject has a unique age and gender.
    n_age_per_subj = dperm.groupby(subj_col)[age_col].nunique()
    n_gender_per_subj = dperm.groupby(subj_col)[gender_col].nunique()

    if (n_age_per_subj > 1).any():
        bad = n_age_per_subj[n_age_per_subj > 1].index.tolist()[:10]
        raise ValueError(
            "Some subjects have more than one age value in df_contrast. "
            "For true longitudinal/retest data with changing age, use a visit-level "
            "or subject-trajectory-level permutation instead. "
            f"Example problematic subjects: {bad}"
        )

    if (n_gender_per_subj > 1).any():
        bad = n_gender_per_subj[n_gender_per_subj > 1].index.tolist()[:10]
        raise ValueError(
            "Some subjects have more than one gender value in df_contrast. "
            f"Example problematic subjects: {bad}"
        )

    # Shuffle subject-level ages within gender.
    subj_to_perm_age = {}

    for gval in fixed_gender_levels:
        sub_g = subj_meta[subj_meta[gender_col] == gval].copy()

        if len(sub_g) <= 1:
            # Nothing to shuffle.
            for _, row in sub_g.iterrows():
                subj_to_perm_age[row[subj_col]] = row[age_col]
            continue

        original_ages = sub_g[age_col].to_numpy()
        permuted_ages = rng.permutation(original_ages)

        for sid, a_perm in zip(sub_g[subj_col].values, permuted_ages):
            subj_to_perm_age[sid] = a_perm

    # Apply subject-level permuted age to all rows of that subject.
    dperm[age_col] = dperm[subj_col].map(subj_to_perm_age).astype(float)

    return dperm


def permutation_test_observed_tps(
    df_contrast,
    observed_final_tps,
    fixed_gender_levels=(0, 1),
    lam_grid=None,
    n_splines=25,
    age_grid_size=300,
    n_draws_perm=300,
    n_perm=500,
    seed=2025,
    assign_window=0.75,
    edge_exclusion=0.25,
    keep_kind="both",
    subj_col="sub_id",
    gender_col="gender",
    age_col="age",
    shuffle_level="subject",
):
    """
    Age-shuffle permutation test for observed final turning points.

    This version supports subject-level age shuffling.

    Null
    ----
    The age-trajectory relationship is absent within gender strata.

    Permutation logic
    -----------------
    If shuffle_level == 'subject':
        Age is shuffled across subjects within each gender group.
        All rows belonging to the same subject receive the same permuted age.

    If shuffle_level == 'row':
        Age is shuffled across rows within each gender group.
        This should only be used when each row is an independent subject-level
        observation.

    Parameters
    ----------
    df_contrast : DataFrame
        Contrast-level dataframe.

    observed_final_tps : DataFrame
        Final TP table from the observed data.

    subj_col : str
        Subject ID column.

    shuffle_level : {'subject', 'row'}
        Level at which age is shuffled.

    Returns
    -------
    perm_summary : DataFrame
        observed_final_tps with added columns:
            perm_p
            perm_null_mean
            perm_null_95

    perm_null : dict
        key = observed TP row index
        value = null array of local sign-change probability
    """

    if lam_grid is None:
        lam_grid = np.logspace(-3, 3, 12)

    if observed_final_tps is None or len(observed_final_tps) == 0:
        return observed_final_tps, {}

    rng = np.random.default_rng(seed)

    obs = observed_final_tps.copy().reset_index(drop=True)

    # Store null distributions for each observed final TP.
    null = {i: [] for i in range(len(obs))}

    age_min = float(np.nanmin(df_contrast[age_col]))
    age_max = float(np.nanmax(df_contrast[age_col]))
    age_grid = np.linspace(age_min, age_max, age_grid_size)

    for b in tqdm(
            range(n_perm),
            desc='Permutation analysis for sign change probability of TP'
    ):
        # ------------------------------------------------------------
        # Key change: subject-level age shuffling
        # ------------------------------------------------------------
        if shuffle_level == "subject":
            dperm = _shuffle_age_at_subject_level(
                df=df_contrast,
                subj_col=subj_col,
                gender_col=gender_col,
                age_col=age_col,
                fixed_gender_levels=fixed_gender_levels,
                rng=rng,
            )

        elif shuffle_level == "row":
            dperm = df_contrast.copy()

            for gval in fixed_gender_levels:
                idx = dperm.index[dperm[gender_col] == gval].to_numpy()
                if len(idx) > 1:
                    dperm.loc[idx, age_col] = rng.permutation(
                        dperm.loc[idx, age_col].values
                    )

        else:
            raise ValueError("shuffle_level must be 'subject' or 'row'.")

        try:
            gam_p, Xp, yp, gcp, info_p = fit_gam_age_by_gender_smooth(
                dperm,
                gender_col=gender_col,
                fixed_gender_levels=fixed_gender_levels,
                lam_grid=lam_grid,
                n_splines=n_splines,
                verbose=False,
                age_col=age_col,
                y_col="dv_diff",
            )

        except Exception as e:
            warnings.warn(f"Permutation {b} failed during fitting: {e}")
            for i in range(len(obs)):
                null[i].append(np.nan)
            continue

        for gval in fixed_gender_levels:
            obs_g_idx = obs.index[obs["gender"] == gval].tolist()

            if len(obs_g_idx) == 0:
                continue

            try:
                curves_p = posterior_trajectory_samples(
                    gam=gam_p,
                    info=info_p,
                    gender_value=gval,
                    age_grid=age_grid,
                    n_draws=n_draws_perm,
                    seed=seed + 100000 + b * 10 + int(gval),
                )

                cands_p = extract_turning_point_candidates(
                    curves=curves_p,
                    age_grid=age_grid,
                    edge_exclusion=edge_exclusion,
                    keep_kind=keep_kind,
                )

                for i in obs_g_idx:
                    target_age = obs.loc[i, "posterior_median_age"]
                    tp_kind = obs.loc[i, "tp_kind"]

                    p_local = local_sign_change_probability(
                        candidates=cands_p,
                        target_age=target_age,
                        n_draws=n_draws_perm,
                        window=assign_window,
                        tp_kind=tp_kind,
                    )

                    null[i].append(p_local)

            except Exception as e:
                warnings.warn(f"Permutation {b} failed for gender={gval}: {e}")
                for i in obs_g_idx:
                    null[i].append(np.nan)

    # Summarize permutation null distributions.
    perm_p = []
    perm_mean = []
    perm_q95 = []

    for i in range(len(obs)):
        null_i = np.asarray(null[i], dtype=float)
        null_i = null_i[np.isfinite(null_i)]

        obs_stat = float(obs.loc[i, "sign_change_prob"])

        if len(null_i) == 0:
            perm_p.append(np.nan)
            perm_mean.append(np.nan)
            perm_q95.append(np.nan)
        else:
            pval = (1 + np.sum(null_i >= obs_stat)) / (len(null_i) + 1)
            perm_p.append(float(pval))
            perm_mean.append(float(np.nanmean(null_i)))
            perm_q95.append(float(np.nanquantile(null_i, 0.95)))

    obs["perm_p"] = perm_p
    obs["perm_null_mean"] = perm_mean
    obs["perm_null_95"] = perm_q95

    return obs, null


# =============================================================================
# 8. Simultaneous confidence band of derivative
# =============================================================================
def compute_derivative_draws(curves, age_grid):
    """
    Compute first-derivative draws from posterior trajectory draws.

    Parameters
    ----------
    curves : ndarray, shape = (n_draws, n_age_grid)
        Posterior trajectory samples.

    age_grid : ndarray, shape = (n_age_grid,)
        Age grid.

    Returns
    -------
    deriv : ndarray, shape = (n_draws, n_age_grid)
        Posterior first-derivative samples.
    """
    curves = np.asarray(curves, dtype=float)
    age_grid = np.asarray(age_grid, dtype=float)

    if curves.ndim != 2:
        raise ValueError("curves must be a 2D array: n_draws x n_age_grid")

    if curves.shape[1] != len(age_grid):
        raise ValueError("curves.shape[1] must match len(age_grid)")

    deriv = np.gradient(curves, age_grid, axis=1)
    
    return deriv


def _moving_average_1d(x, window=7):
    """
    Symmetric moving-average smoothing with reflect padding.
    Window must be a positive odd integer.
    """
    if window is None or window <= 1:
        return np.asarray(x, dtype=float)

    w = int(window)
    if w % 2 == 0:
        raise ValueError("window must be odd")

    x = np.asarray(x, dtype=float)
    pad = w // 2
    xpad = np.pad(x, pad, mode="reflect")
    kernel = np.ones(w) / w
    return np.convolve(xpad, kernel, mode="valid")


def simultaneous_band_from_draws(
    draws,
    alpha=0.05,
    center="median",
    smooth_sd_window=None,
    eps=1e-12,
):
    """
    Build a simultaneous confidence/credible band from posterior draws using
    the max-|t| method.

    Parameters
    ----------
    draws : ndarray, shape = (n_draws, n_grid)
        Posterior draws of a function, e.g., derivative draws.

    alpha : float, default=0.05
        Family-wise error level.
        alpha=0.05 gives a 95% simultaneous band.

    center : {'median', 'mean'}, default='median'
        Center curve.

    smooth_sd_window : int or None
        If not None, smooth pointwise SD before computing the max-|t| statistic.
        Use an odd integer, e.g., 7, 9, 11.
        The same smoothed SD is used for calibration and band construction.

    eps : float
        Lower bound for SD to avoid division by zero.

    Returns
    -------
    band : dict
        {
            'center': center_curve,
            'lo': lower_band,
            'hi': upper_band,
            'sd': pointwise_sd_used,
            'c': simultaneous_critical_value,
            'alpha': alpha
        }
    """
    draws = np.asarray(draws, dtype=float)

    if draws.ndim != 2:
        raise ValueError("draws must be a 2D array: n_draws x n_grid")

    if center == "median":
        mu = np.nanmedian(draws, axis=0)
    elif center == "mean":
        mu = np.nanmean(draws, axis=0)
    else:
        raise ValueError("center must be 'median' or 'mean'")

    sd = np.nanstd(draws, axis=0, ddof=1)
    sd = np.maximum(sd, eps)

    if smooth_sd_window is not None and smooth_sd_window > 1:
        sd = _moving_average_1d(sd, window=smooth_sd_window)
        sd = np.maximum(sd, eps)

    z = (draws - mu) / sd
    tmax = np.nanmax(np.abs(z), axis=1)
    c = float(np.nanquantile(tmax, 1 - alpha))

    lo = mu - c * sd
    hi = mu + c * sd

    return {
        "center": mu,
        "lo": lo,
        "hi": hi,
        "sd": sd,
        "c": c,
        "alpha": alpha,
    }


def compute_derivative_simultaneous_bands_for_result(
    result_one_contrast,
    alpha=0.05,
    center="median",
    smooth_sd_window=7,
):
    """
    Compute derivative simultaneous bands for each gender trajectory in one contrast.

    Parameters
    ----------
    result_one_contrast : dict
        One contrast result from:
            out["results_by_contrast"][contrast_name]

        Required fields:
            result_one_contrast["age_grid"]
            result_one_contrast["curves"]

    alpha : float
        Family-wise error level.

    center : {'median', 'mean'}
        Center of derivative draws.

    smooth_sd_window : int or None
        Optional SD smoothing window for simultaneous band construction.

    Returns
    -------
    deriv_results : dict
        deriv_results[gender_value] = {
            'age_grid': age_grid,
            'deriv_draws': derivative draws,
            'band': simultaneous band dict
        }
    """
    age_grid = np.asarray(result_one_contrast["age_grid"], dtype=float)
    curves_by_gender = result_one_contrast["curves"]

    deriv_results = {}

    for gval, curves in curves_by_gender.items():
        deriv_draws = compute_derivative_draws(curves, age_grid)

        band = simultaneous_band_from_draws(
            deriv_draws,
            alpha=alpha,
            center=center,
            smooth_sd_window=smooth_sd_window,
        )

        deriv_results[gval] = {
            "age_grid": age_grid,
            "deriv_draws": deriv_draws,
            "band": band,
        }

    return deriv_results


# =============================================================================
# 9. finite difference to construct derivative design matrix
# =============================================================================
def _clip_age_for_finite_difference(age_grid, eps, age_min, age_max):
    """
    Keep age +/- eps inside the observed age range.
    Uses forward/backward finite difference near boundaries.
    """
    age_grid = np.asarray(age_grid, dtype=float)

    age_minus = age_grid - eps
    age_plus = age_grid + eps

    # For boundary points, avoid going outside the fitting age range.
    age_minus = np.maximum(age_minus, age_min)
    age_plus = np.minimum(age_plus, age_max)

    denom = age_plus - age_minus

    if np.any(denom <= 0):
        raise ValueError("Invalid finite-difference denominator. Check eps and age range.")

    return age_minus, age_plus, denom


def finite_difference_derivative_matrix(
    gam,
    info,
    gender_value,
    age_grid,
    eps=1e-4,
    age_col="age",
    gender_col="gender",
):
    """
    Construct finite-difference derivative matrix D for one gender trajectory.

    The derivative is approximated as:

        f'(age) ≈ [f(age + eps) - f(age - eps)] / (2 * eps)

    In matrix form:

        D(age) = [X(age + eps) - X(age - eps)] / denom

    where X is the pyGAM model matrix.

    Parameters
    ----------
    gam : fitted pyGAM object
        Fitted LinearGAM.

    info : dict
        Model info returned by fit_gam_age_by_gender_smooth(...).

    gender_value : int, str, or category value
        Original gender value, e.g. 0 or 1.

    age_grid : ndarray
        Ages at which derivatives are evaluated.

    eps : float
        Finite-difference step size in age units.
        For age measured in years, values like 1e-4 to 1e-3 are usually fine.

    Returns
    -------
    D : ndarray or sparse matrix, shape = (n_age_grid, n_coef)
        Derivative design matrix.

    age_minus, age_plus : ndarray
        Actual ages used for finite difference.
    """
    age_grid = np.asarray(age_grid, dtype=float)

    # Restrict finite differences to observed fitting range.
    train_age = np.asarray(info["X"][:, info["idx"]["age"]], dtype=float)
    age_min = float(np.nanmin(train_age))
    age_max = float(np.nanmax(train_age))

    age_minus, age_plus, denom = _clip_age_for_finite_difference(
        age_grid=age_grid,
        eps=eps,
        age_min=age_min,
        age_max=age_max,
    )

    df_minus = pd.DataFrame({
        age_col: age_minus,
        gender_col: [gender_value] * len(age_grid),
    })

    df_plus = pd.DataFrame({
        age_col: age_plus,
        gender_col: [gender_value] * len(age_grid),
    })

    X_minus_raw = build_X_for_info(df_minus, info, age_col=age_col, gender_col=gender_col)
    X_plus_raw = build_X_for_info(df_plus, info, age_col=age_col, gender_col=gender_col)

    M_minus = gam._modelmat(X_minus_raw)
    M_plus = gam._modelmat(X_plus_raw)

    # denom is vector length n_grid; reshape for row-wise division.
    D = (M_plus - M_minus).multiply(1.0 / denom[:, None]) \
        if hasattr(M_plus - M_minus, "multiply") \
        else (M_plus - M_minus) / denom[:, None]

    return D, age_minus, age_plus


# =============================================================================
# 10. finite difference to compute simultaneous intervals of derivative
# =============================================================================
def finite_difference_derivative_draws(
    gam,
    info,
    gender_value,
    age_grid,
    n_draws=1000,
    seed=2025,
    eps=1e-4,
):
    """
    Compute posterior derivative draws using finite differences of the model matrix.

    Returns
    -------
    deriv_draws : ndarray, shape = (n_draws, n_age_grid)
        Posterior derivative curves.

    D : ndarray or sparse matrix
        Finite-difference derivative matrix.
    """
    coef_draws = sample_gam_coef_posterior(
        gam,
        n_draws=n_draws,
        seed=seed,
    )

    D, age_minus, age_plus = finite_difference_derivative_matrix(
        gam=gam,
        info=info,
        gender_value=gender_value,
        age_grid=age_grid,
        eps=eps,
    )

    # D shape: n_grid x n_coef
    # coef_draws shape: n_draws x n_coef
    deriv_draws = D.dot(coef_draws.T).T
    deriv_draws = np.asarray(deriv_draws, dtype=float)

    return deriv_draws, D


def simultaneous_band_from_derivative_draws(
    deriv_draws,
    alpha=0.05,
    center="median",
    smooth_sd_window=None,
    eps_sd=1e-12,
):
    """
    Construct simultaneous interval for derivative draws using max-|t|.

    Parameters
    ----------
    deriv_draws : ndarray, shape = (n_draws, n_age_grid)
        Posterior derivative curves.

    alpha : float
        alpha=0.05 gives a 95% simultaneous interval.

    center : {'median', 'mean'}
        Center derivative curve.

    smooth_sd_window : int or None
        Optional odd integer for smoothing the derivative SD curve.
        If used, the same smoothed SD is used both for max-|t| calibration
        and for interval construction.

    Returns
    -------
    band : dict
        {
            'center',
            'lo',
            'hi',
            'sd',
            'crit',
            'alpha',
            'level'
        }
    """
    deriv_draws = np.asarray(deriv_draws, dtype=float)

    if center == "median":
        mu = np.nanmedian(deriv_draws, axis=0)
    elif center == "mean":
        mu = np.nanmean(deriv_draws, axis=0)
    else:
        raise ValueError("center must be 'median' or 'mean'")

    sd = np.nanstd(deriv_draws, axis=0, ddof=1)
    sd = np.maximum(sd, eps_sd)

    if smooth_sd_window is not None and smooth_sd_window > 1:
        sd = _moving_average_1d(sd, window=smooth_sd_window)
        sd = np.maximum(sd, eps_sd)

    z = (deriv_draws - mu) / sd
    tmax = np.nanmax(np.abs(z), axis=1)
    crit = float(np.nanquantile(tmax, 1.0 - alpha))

    lo = mu - crit * sd
    hi = mu + crit * sd

    return {
        "center": mu,
        "lo": lo,
        "hi": hi,
        "sd": sd,
        "crit": crit,
        "alpha": alpha,
        "level": 1.0 - alpha,
    }


def compute_fd_derivative_simultaneous_bands_for_result(
    result_one_contrast,
    alpha=0.05,
    center="median",
    smooth_sd_window=7,
    n_draws=None,
    seed=2025,
    eps=1e-4,
):
    """
    Compute finite-difference derivative simultaneous intervals for each gender.

    This version follows the gratia-style logic more closely:
        derivative = finite difference of model predictions / model matrix.

    Parameters
    ----------
    result_one_contrast : dict
        One contrast result from out["results_by_contrast"][contrast_name].
        Must contain:
            'gam'
            'info'
            'age_grid'
            'curves'

    n_draws : int or None
        If None, use the number of posterior trajectory draws already stored in
        result_one_contrast['curves'][gender].
        If specified, draw a new set of posterior coefficients.

    eps : float
        Finite-difference step size.

    Returns
    -------
    deriv_results : dict
        deriv_results[gender_value] = {
            'age_grid',
            'deriv_draws',
            'band',
            'eps'
        }
    """
    gam = result_one_contrast["gam"]
    info = result_one_contrast["info"]
    age_grid = np.asarray(result_one_contrast["age_grid"], dtype=float)

    deriv_results = {}

    for j, gender_value in enumerate(info["gender_levels"]):
        if n_draws is None:
            # Match the number of original posterior trajectory draws.
            n_draws_g = result_one_contrast["curves"][gender_value].shape[0]
        else:
            n_draws_g = int(n_draws)

        deriv_draws, D = finite_difference_derivative_draws(
            gam=gam,
            info=info,
            gender_value=gender_value,
            age_grid=age_grid,
            n_draws=n_draws_g,
            seed=seed + 1000 * (j + 1),
            eps=eps,
        )

        band = simultaneous_band_from_derivative_draws(
            deriv_draws=deriv_draws,
            alpha=alpha,
            center=center,
            smooth_sd_window=smooth_sd_window,
        )

        deriv_results[gender_value] = {
            "age_grid": age_grid,
            "deriv_draws": deriv_draws,
            "band": band,
            "eps": eps,
            "D": D,
        }

    return deriv_results


# =============================================================================
# 11. Check signs of derivatives before and after TP
# =============================================================================
def _find_contiguous_true_segments(mask, age_grid):
    """
    Find contiguous True segments in a boolean mask.

    Returns
    -------
    segments : list of dict
        Each dict:
            {
                'start_idx',
                'end_idx',
                'start_age',
                'end_age',
                'n_points',
                'duration'
            }
    """
    mask = np.asarray(mask, dtype=bool)
    age_grid = np.asarray(age_grid, dtype=float)

    segments = []
    n = len(mask)

    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue

        start = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        end = i

        segments.append({
            "start_idx": int(start),
            "end_idx": int(end),
            "start_age": float(age_grid[start]),
            "end_age": float(age_grid[end]),
            "n_points": int(end - start + 1),
            "duration": float(age_grid[end] - age_grid[start]),
        })

        i += 1

    return segments


def _select_segment_nearest_tp(segments, tp_age, side):
    """
    Select the segment nearest to the target TP age.

    For pre side:
        choose the segment whose end_age is closest to and <= tp_age.
    For post side:
        choose the segment whose start_age is closest to and >= tp_age.
    """
    if len(segments) == 0:
        return None

    if side == "pre":
        valid = [seg for seg in segments if seg["end_age"] <= tp_age]
        if len(valid) == 0:
            return None
        return min(valid, key=lambda seg: abs(seg["end_age"] - tp_age))

    elif side == "post":
        valid = [seg for seg in segments if seg["start_age"] >= tp_age]
        if len(valid) == 0:
            return None
        return min(valid, key=lambda seg: abs(seg["start_age"] - tp_age))

    else:
        raise ValueError("side must be 'pre' or 'post'")


def test_one_tp_with_derivative_band(
    tp_row,
    deriv_result_for_gender,
    pre_window=1.5,
    post_window=1.5,
    min_segment_duration=0.25,
    min_segment_points=3,
):
    """
    Test whether one TP is supported by derivative simultaneous bands.

    Parameters
    ----------
    tp_row : pandas.Series
        One row from final TP table.
        Required fields:
            posterior_median_age
            tp_kind

    deriv_result_for_gender : dict
        One gender entry from compute_derivative_simultaneous_bands_for_result(...).

    pre_window : float
        Age window before TP to search for significant derivative sign.

    post_window : float
        Age window after TP to search for significant derivative sign.

    min_segment_duration : float
        Minimum duration of a significant derivative segment.

    min_segment_points : int
        Minimum number of age-grid points in a significant segment.

    Returns
    -------
    out : dict
        Derivative-band support summary for this TP.
    """
    age_grid = np.asarray(deriv_result_for_gender["age_grid"], dtype=float)
    band = deriv_result_for_gender["band"]

    lo = np.asarray(band["lo"], dtype=float)
    hi = np.asarray(band["hi"], dtype=float)
    center = np.asarray(band["center"], dtype=float)

    tp_age = float(tp_row["posterior_median_age"])
    tp_kind = str(tp_row["tp_kind"]).lower()

    pre_mask_range = (age_grid >= tp_age - pre_window) & (age_grid < tp_age)
    post_mask_range = (age_grid > tp_age) & (age_grid <= tp_age + post_window)

    if tp_kind == "peak":
        # Before peak: significantly positive derivative.
        pre_sig = pre_mask_range & (lo > 0)

        # After peak: significantly negative derivative.
        post_sig = post_mask_range & (hi < 0)

        pre_relation = "lower_band_above_0"
        post_relation = "upper_band_below_0"

    elif tp_kind == "valley":
        # Before valley: significantly negative derivative.
        pre_sig = pre_mask_range & (hi < 0)

        # After valley: significantly positive derivative.
        post_sig = post_mask_range & (lo > 0)

        pre_relation = "upper_band_below_0"
        post_relation = "lower_band_above_0"

    else:
        raise ValueError(f"Unknown tp_kind: {tp_kind}")

    pre_segments_all = _find_contiguous_true_segments(pre_sig, age_grid)
    post_segments_all = _find_contiguous_true_segments(post_sig, age_grid)

    # Apply minimum duration and point-count filters.
    pre_segments = [
        seg for seg in pre_segments_all
        if seg["duration"] >= min_segment_duration and seg["n_points"] >= min_segment_points
    ]
    post_segments = [
        seg for seg in post_segments_all
        if seg["duration"] >= min_segment_duration and seg["n_points"] >= min_segment_points
    ]

    pre_seg = _select_segment_nearest_tp(pre_segments, tp_age, side="pre")
    post_seg = _select_segment_nearest_tp(post_segments, tp_age, side="post")

    pre_support = pre_seg is not None
    post_support = post_seg is not None

    deriv_support = bool(pre_support and post_support)

    # Derivative center at TP, for descriptive purposes only.
    deriv_at_tp = float(np.interp(tp_age, age_grid, center))
    lo_at_tp = float(np.interp(tp_age, age_grid, lo))
    hi_at_tp = float(np.interp(tp_age, age_grid, hi))

    out = {
        "deriv_support": deriv_support,
        "pre_support": bool(pre_support),
        "post_support": bool(post_support),
        "pre_deriv_band_relation": pre_relation,
        "post_deriv_band_relation": post_relation,
        "deriv_center_at_tp": deriv_at_tp,
        "deriv_lo_at_tp": lo_at_tp,
        "deriv_hi_at_tp": hi_at_tp,
    }

    if pre_seg is not None:
        out.update({
            "pre_interval_start": pre_seg["start_age"],
            "pre_interval_end": pre_seg["end_age"],
            "pre_interval_duration": pre_seg["duration"],
            "pre_interval_n_points": pre_seg["n_points"],
        })
    else:
        out.update({
            "pre_interval_start": np.nan,
            "pre_interval_end": np.nan,
            "pre_interval_duration": np.nan,
            "pre_interval_n_points": 0,
        })

    if post_seg is not None:
        out.update({
            "post_interval_start": post_seg["start_age"],
            "post_interval_end": post_seg["end_age"],
            "post_interval_duration": post_seg["duration"],
            "post_interval_n_points": post_seg["n_points"],
        })
    else:
        out.update({
            "post_interval_start": np.nan,
            "post_interval_end": np.nan,
            "post_interval_duration": np.nan,
            "post_interval_n_points": 0,
        })

    return out


def add_derivative_band_support_to_final_tps(
    result_one_contrast,
    final_tps_one_contrast=None,
    alpha=0.05,
    center="median",
    smooth_sd_window=7,
    pre_window=1.5,
    post_window=1.5,
    min_segment_duration=0.25,
    min_segment_points=3,
):
    """
    Add derivative simultaneous-band support evidence to final TP table.

    Parameters
    ----------
    result_one_contrast : dict
        One contrast result from:
            out["results_by_contrast"][contrast_name]

    final_tps_one_contrast : DataFrame or None
        Final TP table.
        If None, this function uses:
            result_one_contrast["final_tps_with_perm"]
        if available, otherwise:
            result_one_contrast["final_tps"]

    alpha : float
        Family-wise error level for derivative simultaneous band.

    center : {'median', 'mean'}
        Center curve for derivative band.

    smooth_sd_window : int or None
        Optional smoothing window for derivative SD.

    pre_window, post_window : float
        Search windows around TP.

    min_segment_duration : float
        Minimum duration of significant derivative segment.

    min_segment_points : int
        Minimum number of grid points in significant derivative segment.

    Returns
    -------
    final_aug : DataFrame
        Final TP table with derivative-support columns added.

    deriv_results : dict
        Derivative band results by gender.
    """
    if final_tps_one_contrast is None:
        if "final_tps_with_perm" in result_one_contrast:
            final_tps_one_contrast = result_one_contrast["final_tps_with_perm"]
        else:
            final_tps_one_contrast = result_one_contrast["final_tps"]

    if final_tps_one_contrast is None or len(final_tps_one_contrast) == 0:
        return pd.DataFrame(), {}

    deriv_results = compute_derivative_simultaneous_bands_for_result(
        result_one_contrast,
        alpha=alpha,
        center=center,
        smooth_sd_window=smooth_sd_window,
    )

    final_aug = final_tps_one_contrast.copy().reset_index(drop=True)

    support_rows = []

    for _, row in final_aug.iterrows():
        gval = row["gender"]

        if gval not in deriv_results:
            support_rows.append({
                "deriv_support": False,
                "pre_support": False,
                "post_support": False,
                "pre_deriv_band_relation": None,
                "post_deriv_band_relation": None,
                "deriv_center_at_tp": np.nan,
                "deriv_lo_at_tp": np.nan,
                "deriv_hi_at_tp": np.nan,
                "pre_interval_start": np.nan,
                "pre_interval_end": np.nan,
                "pre_interval_duration": np.nan,
                "pre_interval_n_points": 0,
                "post_interval_start": np.nan,
                "post_interval_end": np.nan,
                "post_interval_duration": np.nan,
                "post_interval_n_points": 0,
            })
            continue

        support = test_one_tp_with_derivative_band(
            tp_row=row,
            deriv_result_for_gender=deriv_results[gval],
            pre_window=pre_window,
            post_window=post_window,
            min_segment_duration=min_segment_duration,
            min_segment_points=min_segment_points,
        )

        support_rows.append(support)

    support_df = pd.DataFrame(support_rows)

    final_aug = pd.concat([final_aug, support_df], axis=1)

    final_aug.attrs["derivative_band_alpha"] = alpha
    final_aug.attrs["derivative_band_center"] = center
    final_aug.attrs["derivative_band_smooth_sd_window"] = smooth_sd_window
    final_aug.attrs["pre_window"] = pre_window
    final_aug.attrs["post_window"] = post_window
    final_aug.attrs["min_segment_duration"] = min_segment_duration
    final_aug.attrs["min_segment_points"] = min_segment_points

    return final_aug, deriv_results


# =============================================================================
# 12. Full pipeline for two contrasts
# =============================================================================

def run_gam_turning_point_pipeline(
    df_long=None,
    df_contrasts=None,
    dv_col="dv",
    gender_col='gender',
    fixed_gender_levels=(0, 1),
    gender_label_map=None,
    contrasts=(("rpsl", "lkng"), ("lkng", "lknt")),
    lam_grid=None,
    n_splines=25,
    age_grid_size=300,
    n_draws=1000,
    n_perm=0,
    n_draws_perm=300,
    seed=2025,
    min_sign_prob=0.15,
    min_count=30,
    assign_window=0.75,
    min_peak_distance=0.75,
    edge_exclusion=0.25,
    separate_kind=True,
):
    """
    Full pipeline.

    You can provide either:
        df_long: original long dataframe with sub_id, gender, age, cognition, dv
    or:
        df_contrasts: contrast-level dataframe with sub_id, gender, age, contrast, dv_diff

    If n_perm > 0, age-shuffle permutation test is performed.
    """
    if lam_grid is None:
        lam_grid = np.logspace(-3, 3, 12)

    if gender_label_map is None:
        gender_label_map = {g: str(g) for g in fixed_gender_levels}

    if df_contrasts is None:
        if df_long is None:
            raise ValueError("Provide either df_long or df_contrasts.")
        df_contrasts = make_within_subject_contrasts(
            df_long,
            dv_col=dv_col,
            contrasts=contrasts,
        )

    all_final = []
    results = {}

    for cname, dsub in df_contrasts.groupby("contrast"):
        print(f"\n===== Analyzing contrast: {cname} =====")
        print(f"n subjects = {dsub['sub_id'].nunique()}, n rows = {len(dsub)}")

        res = analyze_turning_points_one_contrast(
            df_contrast=dsub,
            fixed_gender_levels=fixed_gender_levels,
            gender_col=gender_col,
            gender_label_map=gender_label_map,
            lam_grid=lam_grid,
            n_splines=n_splines,
            age_grid_size=age_grid_size,
            n_draws=n_draws,
            seed=seed,
            separate_kind=separate_kind,
            min_sign_prob=min_sign_prob,
            min_count=min_count,
            assign_window=assign_window,
            min_peak_distance=min_peak_distance,
            edge_exclusion=edge_exclusion,
        )

        final_tps = res["final_tps"].copy()

        if len(final_tps) > 0:
            final_tps.insert(0, "contrast", cname)

        if n_perm > 0 and len(final_tps) > 0:
            print(f"Running age-shuffle permutation test: n_perm={n_perm}")
            final_tps_perm, null = permutation_test_observed_tps(
                df_contrast=dsub,
                observed_final_tps=final_tps,
                gender_col=gender_col,
                fixed_gender_levels=fixed_gender_levels,
                lam_grid=lam_grid,
                n_splines=n_splines,
                age_grid_size=age_grid_size,
                n_draws_perm=n_draws_perm,
                n_perm=n_perm,
                seed=seed + 999,
                assign_window=assign_window,
                edge_exclusion=edge_exclusion,
            )
            final_tps = final_tps_perm
            res["perm_null"] = null

        results[cname] = res
        results[cname]["final_tps_with_perm"] = final_tps

        all_final.append(final_tps)

    if len(all_final) > 0:
        final_table = pd.concat(all_final, axis=0, ignore_index=True)
    else:
        final_table = pd.DataFrame()

    return {
        "df_contrasts": df_contrasts,
        "results_by_contrast": results,
        "final_table": final_table,
    }


def run_direct_outcome_gam_turning_point_pipeline(
    df,
    dv_col,
    age_col="age",
    subj_col=None,
    group_col=None,
    fixed_group_levels=None,
    group_label_map=None,
    outcome_name=None,
    covariate_cols=None,
    residualize_covariates=False,
    keep_original_columns=True,
    lam_grid=None,
    n_splines=25,
    verbose=True,
    **pipeline_kwargs,
):
    """
    Minimal wrapper for direct one-subject-per-row outcome data.

    It converts:
        dv -> dv_diff
        adds contrast = outcome_name
        optionally maps group_col -> gender

    Then it calls the existing run_gam_turning_point_pipeline(...).

    This version explicitly accepts lam_grid and n_splines and avoids duplicated
    keyword arguments.
    """

    d = df.copy()

    if outcome_name is None:
        outcome_name = f"direct_{dv_col}"

    if lam_grid is None:
        lam_grid = np.logspace(-3, 3, 12)

    # Avoid duplicate keyword passing.
    if "lam_grid" in pipeline_kwargs:
        raise ValueError(
            "lam_grid was passed both as an explicit argument and inside "
            "pipeline_kwargs. Please pass it only once."
        )

    if "n_splines" in pipeline_kwargs:
        raise ValueError(
            "n_splines was passed both as an explicit argument and inside "
            "pipeline_kwargs. Please pass it only once."
        )

    required = {dv_col, age_col}

    if subj_col is not None:
        required.add(subj_col)

    if group_col is not None:
        required.add(group_col)

    if covariate_cols is not None:
        required.update(covariate_cols)

    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Subject ID
    if subj_col is None:
        sub_id = np.arange(len(d))
    else:
        sub_id = d[subj_col].values

    # Group variable
    if group_col is None:
        # Dummy group for overall trajectory.
        group_values = np.zeros(len(d), dtype=int)

        if fixed_group_levels is None:
            fixed_group_levels = [0]

        if group_label_map is None:
            group_label_map = {0: "overall"}

    else:
        group_values = d[group_col].values

        if fixed_group_levels is None:
            # Preserve order of appearance; do not sort automatically.
            fixed_group_levels = list(pd.unique(pd.Series(group_values).dropna()))

        if group_label_map is None:
            group_label_map = {g: str(g) for g in fixed_group_levels}

    # Outcome
    y = np.asarray(d[dv_col], dtype=float)

    # Optional covariate residualization
    if covariate_cols is not None and residualize_covariates:
        y_used = _residualize_y_against_covariates(
            df=d,
            y_col=dv_col,
            covariate_cols=covariate_cols,
        )
    else:
        y_used = y

    # Build pseudo contrast-level dataframe.
    df_contrasts = pd.DataFrame({
        "sub_id": sub_id,
        "age": np.asarray(d[age_col], dtype=float),
        "gender": group_values,
        "contrast": outcome_name,
        "dv_diff": np.asarray(y_used, dtype=float),
    })

    if keep_original_columns:
        for col in d.columns:
            if col not in df_contrasts.columns:
                df_contrasts[col] = d[col].values

    # Drop invalid rows.
    mask = (
        np.isfinite(np.asarray(df_contrasts["age"], dtype=float)) &
        np.isfinite(np.asarray(df_contrasts["dv_diff"], dtype=float)) &
        pd.Series(df_contrasts["gender"]).notna().to_numpy()
    )

    df_contrasts = df_contrasts.loc[mask].reset_index(drop=True)

    if len(df_contrasts) < 8:
        raise ValueError(f"Too few valid rows after cleaning: n={len(df_contrasts)}")

    if verbose:
        print("Prepared direct outcome data for GAM TP pipeline")
        print(f"  outcome_name       : {outcome_name}")
        print(f"  n rows             : {len(df_contrasts)}")
        print(f"  group_col          : {group_col}")
        print(f"  fixed_group_levels : {fixed_group_levels}")
        print(f"  n_splines          : {n_splines}")
        print(f"  lam_grid length    : {len(lam_grid) if hasattr(lam_grid, '__len__') else 'scalar'}")
        print(f"  n_perm             : {pipeline_kwargs.get('n_perm', 0)}")
        print("Calling run_gam_turning_point_pipeline(...)")

    out = run_gam_turning_point_pipeline(
        df_contrasts=df_contrasts,
        fixed_gender_levels=fixed_group_levels,
        gender_label_map=group_label_map,
        lam_grid=lam_grid,
        n_splines=n_splines,
        **pipeline_kwargs,
    )

    out["df_direct_converted"] = df_contrasts
    out["direct_outcome_name"] = outcome_name
    out["direct_dv_col"] = dv_col
    out["direct_age_col"] = age_col
    out["direct_group_col"] = group_col

    return out


def add_derivative_band_support_to_pipeline_output(
    out,
    alpha=0.05,
    center="median",
    smooth_sd_window=7,
    pre_window=1.5,
    post_window=1.5,
    min_segment_duration=0.25,
    min_segment_points=3,
):
    """
    Add derivative-band support evidence to every contrast in the full pipeline output.

    Parameters
    ----------
    out : dict
        Output from run_gam_turning_point_pipeline(...)

    Returns
    -------
    out_aug : dict
        Same object as out, with derivative results added:
            out["results_by_contrast"][contrast]["derivative_results"]
            out["results_by_contrast"][contrast]["final_tps_with_deriv"]

        Also adds:
            out["final_table_with_deriv"]
    """
    all_tables = []

    for contrast_name, res in out["results_by_contrast"].items():
        if "final_tps_with_perm" in res:
            final_tps = res["final_tps_with_perm"]
        else:
            final_tps = res.get("final_tps", pd.DataFrame())

        final_aug, deriv_results = add_derivative_band_support_to_final_tps(
            result_one_contrast=res,
            final_tps_one_contrast=final_tps,
            alpha=alpha,
            center=center,
            smooth_sd_window=smooth_sd_window,
            pre_window=pre_window,
            post_window=post_window,
            min_segment_duration=min_segment_duration,
            min_segment_points=min_segment_points,
        )

        res["derivative_results"] = deriv_results
        res["final_tps_with_deriv"] = final_aug

        if final_aug is not None and len(final_aug) > 0:
            tmp = final_aug.copy()

            if "contrast" not in tmp.columns:
                tmp.insert(0, "contrast", contrast_name)

            all_tables.append(tmp)

    if len(all_tables) > 0:
        out["final_table_with_deriv"] = pd.concat(all_tables, axis=0, ignore_index=True)
    else:
        out["final_table_with_deriv"] = pd.DataFrame()

    return out


# =============================================================================
# 13. Optional plotting
# =============================================================================

def plot_posterior_trajectory_with_tps(
    result_one_contrast,
    final_tps_one_contrast,
    gender_label_map=None,
    title=None,
    alpha_band=0.25,
    trajectory_kwargs=None,
    band_kwargs=None,
    tp_line_kwargs=None,
    tp_marker_kwargs=None,
    tp_marker_shape="o",
    show_tp_lines=True,
    show_tp_markers=True,
    ax=None,
    figsize=(7.5, 5.2)
):
    """
    Plot posterior median trajectory and 95% pointwise posterior interval,
    with final turning points overlaid as vertical lines and/or markers.

    Parameters
    ----------
    result_one_contrast : dict
        Output for one contrast from analyze_turning_points_one_contrast(...)
        or from results_by_contrast[contrast_name].

    final_tps_one_contrast : pandas.DataFrame
        Final TP table for one contrast.
        Required columns include:
            gender, posterior_median_age, tp_kind

    gender_label_map : dict or None
        Mapping from original gender value to label.
        Example:
            {0: "male", 1: "female"}

    title : str or None
        Figure title.

    alpha_band : float
        Alpha value for posterior interval band.
        This is used only when band_kwargs does not specify alpha.

    trajectory_kwargs : dict or None
        Keyword arguments for trajectory lines.
        Example:
            trajectory_kwargs = dict(linewidth=2.5, linestyle="-", alpha=0.95)

    band_kwargs : dict or None
        Keyword arguments for posterior interval bands.
        Example:
            band_kwargs = dict(alpha=0.20)

    tp_line_kwargs : dict or None
        Keyword arguments for TP vertical lines.
        Example:
            tp_line_kwargs = dict(linestyle="--", linewidth=1.2, alpha=0.7)

    tp_marker_kwargs : dict or None
        Keyword arguments for TP markers.
        Example:
            tp_marker_kwargs = dict(markersize=8, markeredgewidth=1.0)

    tp_marker_shape : str or dict
        Marker shape for TP markers.

        If str:
            same marker is used for all TPs.
            Example:
                tp_marker_shape = "o"

        If dict:
            can specify marker by TP kind.
            Example:
                tp_marker_shape = {"peak": "^", "valley": "v"}

    show_tp_lines : bool
        Whether to draw vertical TP lines.

    show_tp_markers : bool
        Whether to draw TP markers at the trajectory-TP intersection.

    ax : matplotlib.axes.Axes or None
        Existing axes. If None, a new figure and axes are created.

    Returns
    -------
    fig, ax
    """

    info = result_one_contrast["info"]
    age_grid = result_one_contrast["age_grid"]
    curves_by_gender = result_one_contrast["curves"]

    if gender_label_map is None:
        gender_label_map = {g: str(g) for g in info["gender_levels"]}

    if trajectory_kwargs is None:
        trajectory_kwargs = {}
    if band_kwargs is None:
        band_kwargs = {}
    if tp_line_kwargs is None:
        tp_line_kwargs = {}
    if tp_marker_kwargs is None:
        tp_marker_kwargs = {}

    # Default styles; user-specified kwargs override these.
    traj_style_default = dict(linewidth=2.0, alpha=1.0)
    traj_style_default.update(trajectory_kwargs)

    band_style_default = dict(alpha=alpha_band)
    band_style_default.update(band_kwargs)

    tp_line_style_default = dict(linestyle="--", linewidth=1.0, alpha=0.7)
    tp_line_style_default.update(tp_line_kwargs)

    tp_marker_style_default = dict(
        markersize=7,
        markeredgewidth=1.0,
        linestyle="None",
        zorder=5,
    )
    tp_marker_style_default.update(tp_marker_kwargs)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Store median trajectories and line colors for TP marker placement.
    median_by_gender = {}
    color_by_gender = {}

    for gval, curves in curves_by_gender.items():
        med = np.nanmedian(curves, axis=0)
        lo = np.nanquantile(curves, 0.025, axis=0)
        hi = np.nanquantile(curves, 0.975, axis=0)

        median_by_gender[gval] = med

        label = gender_label_map.get(gval, str(gval))

        # Plot median trajectory.
        line, = ax.plot(
            age_grid,
            med,
            label=label,
            color='#F28213' if gval==0 else '#1B70AC',
            **traj_style_default
        )

        line_color = line.get_color()
        color_by_gender[gval] = line_color

        # Use the same color as the trajectory unless user explicitly sets color.
        band_style = band_style_default.copy()
        if "color" not in band_style:
            band_style["color"] = line_color

        ax.fill_between(
            age_grid,
            lo,
            hi,
            **band_style
        )

    # Plot TP vertical lines and TP markers.
    if final_tps_one_contrast is not None and len(final_tps_one_contrast) > 0:
        for _, row in final_tps_one_contrast.iterrows():
            gval = row["gender"]

            if gval not in median_by_gender:
                continue

            tp_age = float(row["posterior_median_age"])
            tp_kind = row["tp_kind"] if "tp_kind" in row else None

            med = median_by_gender[gval]
            y_tp = np.interp(tp_age, age_grid, med)

            line_color = color_by_gender.get(gval, None)

            if show_tp_lines:
                line_style = tp_line_style_default.copy()
                if "color" not in line_style and line_color is not None:
                    line_style["color"] = line_color
                ax.axvline(tp_age, **line_style)

            if show_tp_markers:
                marker_style = tp_marker_style_default.copy()

                if isinstance(tp_marker_shape, dict):
                    marker = tp_marker_shape.get(tp_kind, "o")
                else:
                    marker = tp_marker_shape

                marker_style["marker"] = marker

                if "color" not in marker_style and line_color is not None:
                    marker_style["color"] = line_color

                if "markerfacecolor" not in marker_style and line_color is not None:
                    marker_style["markerfacecolor"] = line_color

                if "markeredgecolor" not in marker_style:
                    marker_style["markeredgecolor"] = "black"

                ax.plot(
                    tp_age,
                    y_tp,
                    **marker_style
                )

    ax.axhline(0, linestyle="--", linewidth=1, color="k", alpha=0.5)
    ax.set_xlabel("Age")
    ax.set_ylabel("Contrast effect")

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    return fig, ax


def plot_tp_kde_from_pipeline(
    out,
    contrast_name,
    gender_value,
    tp_kind=None,
    gender_label_map=None,
    candidate_pad=0.25,
    bandwidth=None,
    age_grid_size=200,
    show_final_tps=True,
    candidate_hist=True,
    hist_alpha=0.20, hist_color='b',
    kde_kwargs=None,
    final_line_kwargs=None,
    final_ci_kwargs=None,
    ax=None,
):
    """
    Plot KDE of posterior turning-point candidates for one contrast and one gender.

    Parameters
    ----------
    out : dict
        Output from run_gam_turning_point_pipeline(...).

    contrast_name : str
        Example:
            "rpsl_minus_lkng"
            "lkng_minus_lknt"

    gender_value : int or str
        Original gender value, e.g., 0 or 1.

    tp_kind : {'peak', 'valley', None}
        If 'peak' or 'valley', plot KDE for that TP type only.
        If None, use all TP candidates.

    gender_label_map : dict or None
        Example:
            {0: "male", 1: "female"}

    bandwidth : str, float, callable, or None
        Passed to scipy.stats.gaussian_kde(bw_method=bandwidth).

    show_final_tps : bool
        Whether to overlay final TP median age and 95% CI.

    candidate_hist : bool
        Whether to show a light histogram behind KDE.

    kde_kwargs : dict or None
        Keyword arguments passed to ax.plot for KDE curve.

    final_line_kwargs : dict or None
        Keyword arguments for final TP vertical line.

    final_ci_kwargs : dict or None
        Keyword arguments for final TP CI span.

    ax : matplotlib.axes.Axes or None

    Returns
    -------
    fig, ax
    """
    if gender_label_map is None:
        gender_label_map = {}

    if kde_kwargs is None:
        kde_kwargs = {}
    if final_line_kwargs is None:
        final_line_kwargs = {}
    if final_ci_kwargs is None:
        final_ci_kwargs = {}

    kde_style = dict(linewidth=2.0)
    kde_style.update(kde_kwargs)

    final_line_style = dict(linestyle="--", linewidth=1.5, alpha=0.85)
    final_line_style.update(final_line_kwargs)

    final_ci_style = dict(alpha=0.15)
    final_ci_style.update(final_ci_kwargs)

    res = out["results_by_contrast"][contrast_name]

    # Candidate TP ages
    cands = res["candidates"][gender_value].copy()

    if tp_kind is not None:
        cands = cands[cands["kind"] == tp_kind].copy()

    if len(cands) < 3:
        raise ValueError(
            f"Too few TP candidates for KDE: contrast={contrast_name}, "
            f"gender={gender_value}, tp_kind={tp_kind}, n={len(cands)}"
        )

    ages = np.asarray(cands["age"], dtype=float)
    ages = ages[np.isfinite(ages)]

    age_min = float(np.nanmin(res["age_grid"]))
    age_max = float(np.nanmax(res["age_grid"]))

    x_min = max(age_min, float(np.nanmin(ages)) - candidate_pad)
    x_max = min(age_max, float(np.nanmax(ages)) + candidate_pad)
    
    x = np.linspace(x_min, x_max, age_grid_size)

    kde = gaussian_kde(ages, bw_method=bandwidth)
    y = kde(x)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
    else:
        fig = ax.figure

    if candidate_hist:
        ax.hist(
            ages,
            bins=30,
            density=True,
            alpha=hist_alpha,
            color=hist_color, edgecolor=hist_color,
        )

    ax.plot(x, y, **kde_style)

    # Final TPs
    if show_final_tps:
        final_tps = res.get("final_tps_with_perm", res.get("final_tps", pd.DataFrame()))
        final_sub = final_tps[final_tps["gender"] == gender_value].copy()

        if tp_kind is not None and "tp_kind" in final_sub.columns:
            final_sub = final_sub[final_sub["tp_kind"] == tp_kind].copy()

        for _, row in final_sub.iterrows():
            tp_age = float(row["posterior_median_age"])
            ci_low = float(row["age_ci_low"])
            ci_high = float(row["age_ci_high"])

            ax.axvspan(ci_low, ci_high, **final_ci_style)
            ax.axvline(tp_age, **final_line_style)

    # g_label = gender_label_map.get(gender_value, str(gender_value))
    kind_label = "all" if tp_kind is None else tp_kind

    ax.set_xlabel("Age")
    ax.set_ylabel("Density")
    ax.set_title(f"TP candidate KDE | {contrast_name} | {kind_label}")
    plt.tight_layout()

    return fig, ax


def plot_tp_permutation_hist_from_pipeline(
    out,
    contrast_name,
    tp_row_idx,
    bins=30, log=False,
    hist_kwargs=None,
    obs_line_kwargs=None,
    null_q_line=False,
    ax=None,
):
    """
    Plot permutation null distribution for one observed final TP.

    Parameters
    ----------
    out : dict
        Output from run_gam_turning_point_pipeline(...).

    contrast_name : str
        Contrast name, e.g., "rpsl_minus_lkng".

    tp_row_idx : int
        Row index in:
            out["results_by_contrast"][contrast_name]["final_tps_with_perm"].reset_index(drop=True)

        This is NOT necessarily the row index in out["final_table"] after concatenation.

    bins : int
        Number of histogram bins.

    hist_kwargs : dict or None
        Keyword arguments for ax.hist.

    obs_line_kwargs : dict or None
        Keyword arguments for observed statistic vertical line.

    null_q_line : bool
        Whether to draw the 95th percentile of permutation null.

    ax : matplotlib.axes.Axes or None

    Returns
    -------
    fig, ax
    """
    if hist_kwargs is None:
        hist_kwargs = {}
    if obs_line_kwargs is None:
        obs_line_kwargs = {}

    hist_style = dict(alpha=0.65, edgecolor="black")
    hist_style.update(hist_kwargs)

    obs_line_style = dict(color="red", linestyle="--", linewidth=2.0)
    obs_line_style.update(obs_line_kwargs)

    res = out["results_by_contrast"][contrast_name]

    if "perm_null" not in res:
        raise KeyError(
            "Permutation null distribution is not available. "
            "You need to run run_gam_turning_point_pipeline(..., n_perm > 0)."
        )

    final_tps = res["final_tps_with_perm"].reset_index(drop=True)

    if tp_row_idx >= len(final_tps):
        raise IndexError(
            f"tp_row_idx={tp_row_idx} is out of range. "
            f"This contrast has only {len(final_tps)} final TPs."
        )

    null = np.asarray(res["perm_null"][tp_row_idx], dtype=float)
    null = null[np.isfinite(null)]

    if len(null) == 0:
        raise ValueError("Permutation null distribution is empty or all NaN.")

    obs = float(final_tps.loc[tp_row_idx, "sign_change_prob"])
    perm_p = final_tps.loc[tp_row_idx, "perm_p"] if "perm_p" in final_tps.columns else np.nan
    null_q95 = np.nanquantile(null, 0.95)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
    else:
        fig = ax.figure

    ax.hist(null, bins=bins, log=log, **hist_style)

    ax.axvline(
        obs,
        label=f"Observed = {obs:.3f}",
        **obs_line_style,
    )

    if null_q_line:
        ax.axvline(
            null_q95,
            linestyle=":",
            linewidth=2.0,
            color="black",
            label=f"Null 95% = {null_q95:.3f}",
        )

    row = final_tps.loc[tp_row_idx]
    g = row["gender"]
    kind = row["tp_kind"]
    age = row["posterior_median_age"]

    ax.set_xlabel("Permutation local sign-change posterior probability")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Permutation test | {contrast_name} | row={tp_row_idx} | "
        f"gender={g}, {kind}, age={age:.2f}, p={perm_p:.4f}"
    )
    ax.legend(frameon=False)
    plt.tight_layout()

    return fig, ax


def plot_derivative_band_with_tps(
    result_one_contrast,
    final_tps_one_contrast,
    deriv_results=None,
    gender_label_map=None,
    alpha=0.05,
    center="median",
    smooth_sd_window=7,
    show_supported_intervals=True,
    trajectory_line_kwargs=None,
    band_kwargs=None,
    tp_line_kwargs=None,
    interval_kwargs=None,
    ax=None,
):
    """
    Plot derivative simultaneous bands and final TP positions.

    Parameters
    ----------
    result_one_contrast : dict
        One contrast result.

    final_tps_one_contrast : DataFrame
        Final TP table, preferably after derivative support columns were added.

    deriv_results : dict or None
        Output from compute_derivative_simultaneous_bands_for_result(...).
        If None, it is computed internally.

    gender_label_map : dict or None
        Mapping from gender value to display label.

    show_supported_intervals : bool
        Whether to highlight pre/post intervals that support the TP.

    Returns
    -------
    fig, ax
    """
    if trajectory_line_kwargs is None:
        trajectory_line_kwargs = {}
    if band_kwargs is None:
        band_kwargs = {}
    if tp_line_kwargs is None:
        tp_line_kwargs = {}
    if interval_kwargs is None:
        interval_kwargs = {}

    line_style = dict(linewidth=2.0)
    line_style.update(trajectory_line_kwargs)

    band_style = dict(alpha=0.20)
    band_style.update(band_kwargs)

    tp_line_style = dict(linestyle="--", linewidth=1.0, alpha=0.75)
    tp_line_style.update(tp_line_kwargs)

    interval_style = dict(alpha=0.12)
    interval_style.update(interval_kwargs)

    if deriv_results is None:
        deriv_results = compute_derivative_simultaneous_bands_for_result(
            result_one_contrast,
            alpha=alpha,
            center=center,
            smooth_sd_window=smooth_sd_window,
        )

    if gender_label_map is None:
        gender_label_map = {
            g: str(g) for g in result_one_contrast["info"]["gender_levels"]
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
    else:
        fig = ax.figure

    color_by_gender = {}

    for gval, dres in deriv_results.items():
        age_grid = dres["age_grid"]
        band = dres["band"]

        label = gender_label_map.get(gval, str(gval))

        line, = ax.plot(
            age_grid,
            band["center"],
            label=label,
            color='#F28213' if gval==0 else '#1B70AC',
            **line_style,
        )

        color = line.get_color()
        color_by_gender[gval] = color

        fill_style = band_style.copy()
        if "color" not in fill_style:
            fill_style["color"] = color

        ax.fill_between(
            age_grid,
            band["lo"],
            band["hi"],
            **fill_style,
        )

    if final_tps_one_contrast is not None and len(final_tps_one_contrast) > 0:
        for _, row in final_tps_one_contrast.iterrows():
            gval = row["gender"]
            tp_age = float(row["posterior_median_age"])
            color = color_by_gender.get(gval, "black")

            line_style2 = tp_line_style.copy()
            if "color" not in line_style2:
                line_style2["color"] = color

            ax.axvline(tp_age, **line_style2)

            if show_supported_intervals and "pre_interval_start" in row.index:
                int_style = interval_style.copy()
                if "color" not in int_style:
                    int_style["color"] = color

                if np.isfinite(row["pre_interval_start"]) and np.isfinite(row["pre_interval_end"]):
                    ax.axvspan(
                        row["pre_interval_start"],
                        row["pre_interval_end"],
                        **int_style,
                    )

                if np.isfinite(row["post_interval_start"]) and np.isfinite(row["post_interval_end"]):
                    ax.axvspan(
                        row["post_interval_start"],
                        row["post_interval_end"],
                        **int_style,
                    )

    ax.axhline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)

    ax.set_xlabel("Age")
    ax.set_ylabel("Developmental rate of effect")
    ax.set_title("First derivative of trajectory")
    # ax.legend(frameon=False, title="Gender")
    plt.tight_layout()

    return fig, ax


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

        return ax, long_df

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

#%%
#

if __name__ == "__main__":
    """
    Replace the synthetic data block with your real dataframe.

    Your real long dataframe should contain:
        sub_id, gender, age, cognition, dv

    gender coding example:
        0 = male
        1 = female
    """

    # -------------------------
    # Experiment data
    # -------------------------
    dir_data = '/public/home/dingrui/fmri_analysis/data/beh'
    fname_data = 'data_4_anova_ER.csv'
    fpath_data = os.path.join(dir_data, fname_data)
    fpath_behs = os.path.join(dir_data, 'beh_indices_mri_exp_ER_TG.csv')
    fpath_erqs = os.path.join(dir_data, 'er_profiles_new.csv')
    
    # emotion ratings:
    df_long = pd.read_csv(fpath_data, sep=',')
    
    # reaction time:
    df_behs = pd.read_csv(fpath_behs, sep=',')
    df_behs_sel = df_behs[['sub_id', 'age', 'sex', 'Efforts_ER', 'Feeling_lkng', 'Feeling_rpsl', 'Success_ER']].copy()

    subs_sel = np.isin(df_behs_sel, np.unique(df_long['sub_id']))
    df_rt = df_behs_sel[subs_sel]
    df_rt.loc[:, 'Efforts_ER'] = df_rt['Efforts_ER']/1000
    
    # erq:
    df_erq = pd.read_csv(fpath_erqs, sep=',').rename(columns={'subject': 'sub_id'})
    
    df_wide = {}
    df_wide['rt'] = df_rt.copy()
    df_wide['erq'] = df_erq.copy()

    # -------------------------
    # User-defined gender mapping
    # -------------------------
    fixed_gender_levels = [0, 1]
    gender_label_map = {
        0: "male",
        1: "female",
    }

    # -------------------------
    # Run full pipeline
    # -------------------------
    # emotion ratings:
    out1 = run_gam_turning_point_pipeline(
        df_long=df_long,
        dv_col="emot_rating",
        fixed_gender_levels=fixed_gender_levels,
        gender_label_map=gender_label_map,
        contrasts=[("rpsl", "lkng")],
        lam_grid=np.logspace(-3, 3, 10),
        n_splines=4,
        age_grid_size=200,
        n_draws=1000,
        n_perm=1000,             # Set>0, e.g., 500, to run permutation test.
        n_draws_perm=1000,
        seed=2025,
        min_sign_prob=0.25,
        min_count=50,
        assign_window=1,
        min_peak_distance=1,
        edge_exclusion=0.25,
        separate_kind=False,
    )

    final_table = out1["final_table"]
    print("\n===== Final turning points =====")
    print(final_table)
    
    # reaction time / reappraisal success:
    out2 = run_direct_outcome_gam_turning_point_pipeline(
        df=df_wide['erq'],
        dv_col="ERQ_CR",
        age_col="age",
        subj_col="sub_id",
        group_col="sex",
        fixed_group_levels=[0, 1],
        group_label_map={0: "male", 1: "female"},
        outcome_name="direct_dv",
        lam_grid=np.logspace(-3, 3, 10),
        n_splines=4,
        age_grid_size=200,
        n_draws=1000,
        n_perm=1000,
        n_draws_perm=1000,
        min_sign_prob=0.25,
        min_count=50,
        assign_window=1,
        min_peak_distance=1,
        edge_exclusion=0.25,
        separate_kind=False,
    )
    
    # derivative simultaneous confidence band
    out1 = add_derivative_band_support_to_pipeline_output(
        out1,
        alpha=0.05,
        center="median",
        smooth_sd_window=7,
        pre_window=1.5,
        post_window=1.5,
        min_segment_duration=0.25,
        min_segment_points=3,
    )

    final_deriv = out1["final_table_with_deriv"]
    
    print(final_deriv[[
        "contrast",
        "gender",
        "gender_label",
        "tp_kind",
        "posterior_median_age",
        "age_ci_low",
        "age_ci_high",
        "sign_change_prob",
        "deriv_support",
        "pre_support",
        "post_support",
        "pre_interval_start",
        "pre_interval_end",
        "post_interval_start",
        "post_interval_end",
    ]])

    # -------------------------
    # Optional plotting
    # -------------------------
    # trajectory
    out = out2.copy()
    for cname, res in out["results_by_contrast"].items():
        final_c = res["final_tps_with_perm"]
        fig, ax = plot_posterior_trajectory_with_tps(
            result_one_contrast=res,
            final_tps_one_contrast=final_c,
            gender_label_map=gender_label_map,
            figsize=(3.5, 3),
            title=None,
            trajectory_kwargs={
                'linewidth': 3,
                'linestyle': '-',
            },
            tp_line_kwargs={
                'linewidth': 1,
                'linestyle': '--',
            },
            tp_marker_shape='D',
            show_tp_lines=True,
            show_tp_markers=True,
        )
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(
            ticks=[8, 10, 12, 14, 16, 18],
            labels=[8, 10, 12, 14, 16, 18]
        )
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        # plt.ylim(3, 6)
        plt.show()
    
    # kde hist
    fig, ax = plt.subplots(figsize=(3,3))
    for g, c in zip([0,1], ['#F28213', '#1B70AC']): 
        fig, ax = plot_tp_kde_from_pipeline(
            out,
            contrast_name="rpsl_minus_lkng",
            gender_value=g, hist_color=c,
            tp_kind='peak',
            gender_label_map={0: "male", 1: "female"},
            ax=ax,
            show_final_tps=True,
            kde_kwargs={'color':c, 'linewidth':3},
            final_ci_kwargs={'color':c, 'alpha':0.0},
            final_line_kwargs={'color':c, 'linewidth':3, 'linestyle':'--'}
        )
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(3)
    ax.tick_params(axis='both', which='major', width=2, length=6)
    
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()
    
    # derivative curve (change rate)
    cname = "rpsl_minus_lkng"

    res = out["results_by_contrast"][cname]
    final_c = res["final_tps_with_deriv"]
    deriv_res = res["derivative_results"]
    
    fig, ax = plt.subplots(figsize=(3.5, 3))
    fig, ax = plot_derivative_band_with_tps(
        result_one_contrast=res,
        final_tps_one_contrast=final_c,
        deriv_results=deriv_res,
        gender_label_map={0: "male", 1: "female"},
        show_supported_intervals=False,
        alpha=0.05,
        trajectory_line_kwargs={
            'linewidth': 3,
            'linestyle': '-',
        },
        tp_line_kwargs={
            'linewidth': 1,
            'linestyle': '--'
        },
        ax=ax,
    )
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(
        ticks=[6, 8, 10, 12, 14, 16, 18],
        labels=[6, 8, 10, 12, 14, 16, 18]
    )
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
    
if __name__ == "__main__":
    # Figure 2b permutation diagnostic retained for standalone analysis.
    ax, longDf = boxplot_with_points(
        df_perm_null,
        ls_obsr_prob,
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
        y_break=((-0.01, 0.01), (0.7, 1)),
        show_line=False,
        xticklabels=['Male', 'Female'],
        xlabel=None,
        ylabel="Sign change probability",
        title="Permutation results of TP",
        x_tick_rotation=None,
    )
    
    plt.show()
