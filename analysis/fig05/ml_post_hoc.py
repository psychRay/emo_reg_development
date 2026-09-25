#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:50:59 2026

@author: dingrui
"""

#%% import necessary modules
import os
import pickle, json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from pygam import LinearGAM, s, f
from tqdm import tqdm


#%% define functions
def validate_multisite_paired_design(X, y, subject_ids, site_ids):
    """
    Validate the expected design:
    1. Binary classification.
    2. Each subject appears exactly twice.
    3. Each subject contributes one sample to each class.
    4. Each subject belongs to exactly one site.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    subject_ids : array-like, shape (n_samples,)
    site_ids : array-like, shape (n_samples,)

    Returns
    -------
    X, y, subject_ids, site_ids : numpy arrays
    """
    X = np.asarray(X)
    y = np.asarray(y)
    subject_ids = np.asarray(subject_ids)
    site_ids = np.asarray(site_ids)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")

    n = len(X)
    if not (len(y) == len(subject_ids) == len(site_ids) == n):
        raise ValueError("X, y, subject_ids, and site_ids must have the same length")

    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels, got classes={classes}")

    unique_subjects, subject_counts = np.unique(subject_ids, return_counts=True)
    if not np.all(subject_counts == 2):
        bad_subjects = unique_subjects[subject_counts != 2][:10]
        raise ValueError(
            "Each subject must contribute exactly 2 samples. "
            f"Examples of invalid subject IDs: {bad_subjects}"
        )

    for subj in unique_subjects:
        idx = subject_ids == subj
        y_subj = y[idx]
        site_subj = np.unique(site_ids[idx])

        if len(np.unique(y_subj)) != 2:
            raise ValueError(
                f"Subject {subj} does not contain one sample from each class. "
                f"Observed labels: {y_subj}"
            )

        if len(site_subj) != 1:
            raise ValueError(
                f"Subject {subj} appears in multiple sites: {site_subj}. "
                "Each subject must belong to exactly one site."
            )

    unique_sites = np.unique(site_ids)
    if len(unique_sites) < 2:
        raise ValueError("At least 2 sites are required for leave-one-site-out CV.")

    return X.astype(np.float64, copy=False), y, subject_ids, site_ids


def make_default_model_specs(
    selected_models=None,
    random_state=42,
):
    """
    Build a selectable model registry.

    Parameters
    ----------
    selected_models : list[str] or str or None
        Which models to include.
        - None: include all default models
        - str: include one model
        - list[str]: include selected models
    random_state : int

    Returns
    -------
    model_specs : dict
        Dict in the format required by
        nested_leave_one_site_out_compare_models().
    """
    all_model_specs = {
        # -------------------------
        # Linear models
        # -------------------------
        "logistic_l2": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=random_state
                ))
            ]),
            "param_grid": {
                "clf__C": np.logspace(-4, 4, 17)
            },
            "scoring": "roc_auc",
        },

        "linear_svc": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(
                    dual="auto",
                    max_iter=5000,
                    random_state=random_state
                ))
            ]),
            "param_grid": {
                "clf__C": np.logspace(-4, 4, 17)
            },
            "scoring": "roc_auc",
        },

        "ridge_classifier": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RidgeClassifier())
            ]),
            "param_grid": {
                "clf__alpha": np.logspace(-4, 4, 17)
            },
            "scoring": "roc_auc",
        },

        # -------------------------
        # Nonlinear models
        # -------------------------
        "rbf_svc": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=random_state
                ))
            ]),
            "param_grid": {
                "clf__C": np.logspace(-3, 3, 7),
                "clf__gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0],
            },
            "scoring": "roc_auc",
        },

        "random_forest": {
            "estimator": Pipeline([
                ("clf", RandomForestClassifier(
                    random_state=random_state,
                    n_jobs=-1
                ))
            ]),
            "param_grid": {
                "clf__n_estimators": [200, 500],
                "clf__max_depth": [None, 5, 10, 20],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features": ["sqrt", "log2", 0.2, 0.5],
            },
            "scoring": "roc_auc",
        },

        "hist_gbdt": {
            "estimator": Pipeline([
                ("clf", HistGradientBoostingClassifier(
                    random_state=random_state
                ))
            ]),
            "param_grid": {
                "clf__learning_rate": [0.01, 0.05, 0.1],
                "clf__max_iter": [100, 300],
                "clf__max_depth": [None, 3, 5, 10],
                "clf__min_samples_leaf": [20, 50],
                "clf__l2_regularization": [0.0, 0.01, 0.1],
            },
            "scoring": "roc_auc",
        },

        "mlp": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    max_iter=2000,
                    early_stopping=True,
                    random_state=random_state
                ))
            ]),
            "param_grid": {
                "clf__hidden_layer_sizes": [(64,), (128,), (128, 64)],
                "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                "clf__learning_rate_init": [1e-4, 1e-3, 1e-2],
            },
            "scoring": "roc_auc",
        },
    }

    if selected_models is None:
        return all_model_specs

    if isinstance(selected_models, str):
        selected_models = [selected_models]

    unknown_models = [m for m in selected_models if m not in all_model_specs]
    if len(unknown_models) > 0:
        raise ValueError(
            f"Unknown model(s): {unknown_models}. "
            f"Available models: {list(all_model_specs.keys())}"
        )

    return {m: all_model_specs[m] for m in selected_models}


def extract_signature_response(
    observed_results,
    model_name,
    sample_info=None,
    response_col="signature_response",
):
    """
    Extract out-of-fold signature response from existing nested LOSO results.

    Parameters
    ----------
    observed_results : dict
        Output of nested_leave_one_site_out_compare_models(...).
    model_name : str
        Model key in observed_results["model_results"].
    sample_info : pandas.DataFrame or None
        Optional sample-level metadata table with one row per sample.
    response_col : str
        Name of the output response column.

    Returns
    -------
    df : pandas.DataFrame
        Sample-level table containing signature response.
    """
    if model_name not in observed_results["model_results"]:
        raise KeyError(f"Model '{model_name}' not found in observed_results.")

    model_res = observed_results["model_results"][model_name]

    if "oof_score" not in model_res:
        raise KeyError(
            f"Model '{model_name}' does not contain 'oof_score'. "
            "Signature response cannot be extracted."
        )

    response = np.asarray(model_res["oof_score"], dtype=float)
    n_samples = len(response)

    if sample_info is None:
        df = pd.DataFrame({
            "sample_index": np.arange(n_samples),
            response_col: response,
        })
    else:
        sample_info = sample_info.reset_index(drop=True).copy()
        if len(sample_info) != n_samples:
            raise ValueError(
                f"len(sample_info)={len(sample_info)} does not match n_samples={n_samples}."
            )
        df = sample_info.copy()
        df[response_col] = response

    return df


def _get_model_spec(model_specs, model_name):
    if model_name not in model_specs:
        raise KeyError(f"Model '{model_name}' not found in model_specs.")
    return model_specs[model_name]


def _get_outer_site_order_from_results(observed_results, model_name):
    """
    Recover outer-fold held-out site order from existing results.
    This ensures fold i matches best_params_per_fold[i].
    """
    fold_df = observed_results["model_results"][model_name]["fold_metrics"].copy()
    if "held_out_site" not in fold_df.columns:
        raise KeyError("fold_metrics must contain 'held_out_site'.")
    return list(fold_df["held_out_site"].values)


def _find_final_linear_step(estimator):
    """
    Return the final classifier object.
    Supports plain estimator or Pipeline.
    """
    if hasattr(estimator, "named_steps"):
        last_key = list(estimator.named_steps.keys())[-1]
        return estimator.named_steps[last_key]
    return estimator


def _is_linear_coef_model(estimator):
    clf = _find_final_linear_step(estimator)
    return hasattr(clf, "coef_")


def _get_pipeline_transformer_and_clf(fitted_estimator):
    """
    For Pipeline:
        returns transformed design matrix via all steps except final estimator.
    For non-Pipeline:
        identity transform + estimator as clf.
    """
    if hasattr(fitted_estimator, "named_steps"):
        steps = list(fitted_estimator.named_steps.items())
        clf_name, clf = steps[-1]

        if len(steps) == 1:
            return None, clf

        transformer_names = [name for name, _ in steps[:-1]]
        return transformer_names, clf

    return None, fitted_estimator


def _transform_features_for_linear_weights(fitted_estimator, X):
    """
    Return the feature matrix that the final linear classifier actually sees.
    """
    if not hasattr(fitted_estimator, "named_steps"):
        return np.asarray(X)

    Xt = X
    steps = list(fitted_estimator.named_steps.items())
    for name, step in steps[:-1]:
        Xt = step.transform(Xt)
    return np.asarray(Xt)


def haufe_transform(X_train_transformed, weight_vector, ddof=1):
    """
    Compute Haufe-transformed encoding pattern.

    Parameters
    ----------
    X_train_transformed : ndarray, shape (n_samples, n_features)
        The feature matrix actually seen by the linear classifier.
    weight_vector : ndarray, shape (n_features,)
        Linear decoding weights.
    ddof : int
        Degrees of freedom for covariance/variance.

    Returns
    -------
    encoding_pattern : ndarray, shape (n_features,)
    """
    X_train_transformed = np.asarray(X_train_transformed, dtype=float)
    w = np.asarray(weight_vector, dtype=float).ravel()

    if X_train_transformed.ndim != 2:
        raise ValueError("X_train_transformed must be 2D.")
    if X_train_transformed.shape[1] != w.shape[0]:
        raise ValueError(
            f"Feature mismatch: X has {X_train_transformed.shape[1]} features, "
            f"but w has {w.shape[0]}."
        )

    s = X_train_transformed @ w
    s_var = np.var(s, ddof=ddof)

    if np.isclose(s_var, 0.0):
        raise ValueError("Variance of latent score is zero; Haufe transform is undefined.")

    cov_x = np.cov(X_train_transformed, rowvar=False, ddof=ddof)
    a = cov_x @ w / s_var
    return a


def compute_pattern_expression(X, pattern):
    """
    Sample-wise pattern expression:
        expression_i = x_i dot pattern

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    pattern : ndarray, shape (n_features,)

    Returns
    -------
    expr : ndarray, shape (n_samples,)
    """
    X = np.asarray(X, dtype=float)
    pattern = np.asarray(pattern, dtype=float).ravel()

    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[1] != pattern.shape[0]:
        raise ValueError(
            f"Feature mismatch: X has {X.shape[1]} features, "
            f"but pattern has {pattern.shape[0]}."
        )

    return X @ pattern


def reconstruct_pattern_expression_posthoc(
    X,
    y,
    subject_ids,
    site_ids,
    model_specs,
    observed_results,
    model_name,
    expression_type="encoding",
    sample_info=None,
    expression_col=None,
):
    """
    Post-hoc reconstruction of sample-wise pattern expression without modifying
    the original nested LOSO code.

    This function does NOT re-run nested CV tuning.
    It only re-fits each outer-fold final model using the already saved best params.

    Supported expression_type
    -------------------------
    - "decoder": use decoding weights
    - "encoding": use Haufe-transformed encoding pattern

    Notes
    -----
    - Only supported for linear estimators with coef_.
    - Requires:
        * observed_results["model_results"][model_name]["best_params_per_fold"]
        * observed_results["model_results"][model_name]["fold_metrics"]["held_out_site"]

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    subject_ids : ndarray
    site_ids : ndarray
    model_specs : dict
        Same registry used in the original nested run.
    observed_results : dict
        Output of nested_leave_one_site_out_compare_models(...).
    model_name : str
    expression_type : {"decoder", "encoding"}
    sample_info : DataFrame or None
    expression_col : str or None

    Returns
    -------
    out : dict
        {
            "sample_expression": DataFrame,
            "fold_patterns": list[dict]
        }
    """
    # Reuse your original validation helper if it exists
    X, y, subject_ids, site_ids = validate_multisite_paired_design(X, y, subject_ids, site_ids)

    spec = _get_model_spec(model_specs, model_name)
    base_estimator = spec["estimator"]

    if expression_type not in ["decoder", "encoding"]:
        raise ValueError("expression_type must be one of {'decoder', 'encoding'}.")

    best_params_per_fold = observed_results["model_results"][model_name]["best_params_per_fold"]
    held_out_site_order = _get_outer_site_order_from_results(observed_results, model_name)

    outer_cv = LeaveOneGroupOut()

    # Build mapping: held_out_site -> (train_idx, test_idx)
    split_map = {}
    for train_idx, test_idx in outer_cv.split(X, y, groups=site_ids):
        held_out_site = np.unique(site_ids[test_idx])
        if len(held_out_site) != 1:
            raise RuntimeError("Each outer test fold should contain exactly one site.")
        split_map[held_out_site[0]] = (train_idx, test_idx)

    if len(best_params_per_fold) != len(held_out_site_order):
        raise ValueError("Mismatch between best_params_per_fold and fold order length.")

    expr = np.full(X.shape[0], np.nan, dtype=float)
    fold_patterns = []

    for fold_idx, (held_out_site, best_params) in enumerate(
        zip(held_out_site_order, best_params_per_fold), start=1
    ):
        if held_out_site not in split_map:
            raise KeyError(f"Held-out site '{held_out_site}' not found in reconstructed split map.")

        train_idx, test_idx = split_map[held_out_site]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        estimator = clone(base_estimator)
        estimator.set_params(**best_params)
        estimator.fit(X_train, y_train)

        if not _is_linear_coef_model(estimator):
            raise TypeError(
                f"Model '{model_name}' is not a supported linear coef_ model. "
                "Pattern expression reconstruction is currently only supported for linear estimators."
            )

        clf = _find_final_linear_step(estimator)
        w = np.asarray(clf.coef_).ravel()

        X_train_t = _transform_features_for_linear_weights(estimator, X_train)
        X_test_t = _transform_features_for_linear_weights(estimator, X_test)

        if expression_type == "decoder":
            pattern = w
        else:
            pattern = haufe_transform(X_train_t, w)

        expr[test_idx] = compute_pattern_expression(X_test_t, pattern)

        fold_patterns.append({
            "fold": fold_idx,
            "held_out_site": held_out_site,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "best_params": best_params,
            "decoder_pattern": w,
            "encoding_pattern": haufe_transform(X_train_t, w),
        })

    if np.any(np.isnan(expr)):
        raise RuntimeError("Some samples were not assigned pattern expression values.")

    if expression_col is None:
        expression_col = f"{expression_type}_pattern_expression"

    if sample_info is None:
        df = pd.DataFrame({
            "sample_index": np.arange(X.shape[0]),
            expression_col: expr,
        })
    else:
        sample_info = sample_info.reset_index(drop=True).copy()
        if len(sample_info) != X.shape[0]:
            raise ValueError(
                f"len(sample_info)={len(sample_info)} does not match n_samples={X.shape[0]}."
            )
        df = sample_info.copy()
        df[expression_col] = expr

    return {
        "sample_expression": df,
        "fold_patterns": fold_patterns,
    }


def derive_signature_and_pattern_metrics(
    X,
    y,
    subject_ids,
    site_ids,
    observed_results,
    model_specs,
    model_name,
    sample_info=None,
    include_signature_response=True,
    include_decoder_expression=False,
    include_encoding_expression=True,
):
    """
    Unified helper to derive:
    - signature response (no re-fit)
    - decoder pattern expression (post-hoc fold-wise refit)
    - encoding pattern expression (post-hoc fold-wise refit)

    Returns
    -------
    out : dict
    """
    out = {}

    if include_signature_response:
        out["signature_response"] = extract_signature_response(
            observed_results=observed_results,
            model_name=model_name,
            sample_info=sample_info,
            response_col="signature_response",
        )

    if include_decoder_expression:
        out["decoder_expression"] = reconstruct_pattern_expression_posthoc(
            X=X,
            y=y,
            subject_ids=subject_ids,
            site_ids=site_ids,
            model_specs=model_specs,
            observed_results=observed_results,
            model_name=model_name,
            expression_type="decoder",
            sample_info=sample_info,
            expression_col="decoder_pattern_expression",
        )

    if include_encoding_expression:
        out["encoding_expression"] = reconstruct_pattern_expression_posthoc(
            X=X,
            y=y,
            subject_ids=subject_ids,
            site_ids=site_ids,
            model_specs=model_specs,
            observed_results=observed_results,
            model_name=model_name,
            expression_type="encoding",
            sample_info=sample_info,
            expression_col="encoding_pattern_expression",
        )

    return out


def _build_gam(model_type: str, n_splines: int, lam=None):
    """
    Build GAM according to the selected model type.
    If lam is None, do not pass it to LinearGAM.
    """
    if model_type == "common":
        terms = s(0, n_splines=n_splines) + f(1)

    elif model_type == "sex_modulated":
        terms = (
            s(0, n_splines=n_splines) +
            f(1) +
            s(0, by=1, n_splines=n_splines)
        )

    else:
        raise ValueError("model_type must be 'common' or 'sex_modulated'")

    if lam is None:
        return LinearGAM(terms)
    else:
        return LinearGAM(terms, lam=lam)


def _fit_gam_with_gridsearch(
        X: np.ndarray, 
        y: np.ndarray, 
        model_type: str, 
        N_SPLINES=4, lam=None):
    """
    Fit GAM with age smooth and tune lambda by grid search.
    """
    gam = _build_gam(model_type=model_type, n_splines=N_SPLINES, lam=None)
    gam.gridsearch(X, y, lam=lam, progress=False)
    
    return gam


def _cluster_bootstrap_subjects(df_in: pd.DataFrame, subject_col: str, rng: np.random.RandomState):
    """
    Cluster bootstrap: sample subjects with replacement and include all their rows.
    For cross-sectional data, this reduces to subject-level bootstrap.
    """
    subs = df_in[subject_col].unique()
    sampled_subs = rng.choice(subs, size=len(subs), replace=True)

    boot_parts = []
    for s_id in sampled_subs:
        boot_parts.append(df_in[df_in[subject_col] == s_id])
        
    return pd.concat(boot_parts, axis=0, ignore_index=True)


def _predict_curve(gam: LinearGAM, age_grid: np.ndarray, sex_code: int):
    """
    Return mean prediction for a given sex across age grid.
    """
    X_pred = np.zeros((len(age_grid), 2))
    X_pred[:, 0] = age_grid
    X_pred[:, 1] = sex_code
    
    y_mean = gam.predict(X_pred)
    return y_mean


def _compute_derivative(y: np.ndarray, x: np.ndarray):
    """
    Numerical first derivative based on finite differences.
    """
    return np.gradient(y, x)


def _find_zero_crossings(values: np.ndarray, x: np.ndarray, eps: float = 1e-6):
    """
    Find approximate zero-crossing locations by sign changes between adjacent points.
    Linear interpolation is used to estimate the crossing position.

    Parameters
    ----------
    values : array-like
        Derivative values evaluated on x.
    x : array-like
        Grid values.
    eps : float
        Small threshold to suppress tiny numerical fluctuations.

    Returns
    -------
    crossings : list of dict
        Each element contains:
        - x0: estimated crossing location
        - idx_left: left index
        - idx_right: right index
        - sign_left: sign on the left
        - sign_right: sign on the right
    """
    values = np.asarray(values).copy()
    x = np.asarray(x)

    # Threshold tiny values to zero
    values[np.abs(values) < eps] = 0.0

    crossings = []
    for i in range(len(values) - 1):
        v1, v2 = values[i], values[i + 1]

        # Case 1: direct sign flip
        if v1 * v2 < 0:
            # Linear interpolation for zero crossing
            x1, x2 = x[i], x[i + 1]
            x0 = x1 - v1 * (x2 - x1) / (v2 - v1)
            crossings.append({
                "x0": x0,
                "idx_left": i,
                "idx_right": i + 1,
                "sign_left": np.sign(v1),
                "sign_right": np.sign(v2)
            })

        # Case 2: one side exactly zero and neighboring signs differ
        elif v1 == 0 and i > 0:
            prev_sign = np.sign(values[i - 1])
            next_sign = np.sign(v2)
            if prev_sign != 0 and next_sign != 0 and prev_sign != next_sign:
                crossings.append({
                    "x0": x[i],
                    "idx_left": i,
                    "idx_right": i,
                    "sign_left": prev_sign,
                    "sign_right": next_sign
                })

    return crossings


def _classify_turning_points(first_derivative: np.ndarray, x: np.ndarray, eps: float = 1e-6):
    """
    Identify turning points from first-derivative sign changes.

    Returns
    -------
    turning_points : list of dict
        Each dict contains:
        - age
        - type: 'peak' or 'valley'
    """
    zero_crossings = _find_zero_crossings(first_derivative, x, eps=eps)

    turning_points = []
    for z in zero_crossings:
        left_sign = z["sign_left"]
        right_sign = z["sign_right"]

        if left_sign > 0 and right_sign < 0:
            tp_type = "peak"
        elif left_sign < 0 and right_sign > 0:
            tp_type = "valley"
        else:
            continue

        turning_points.append({
            "age": z["x0"],
            "type": tp_type
        })

    return turning_points


def _compute_local_sharpness(d1: np.ndarray, x: np.ndarray, age0: float, window_years: float):
    """
    Compute local sharpness as the integral-like sum of |slope|
    within a local age window around a candidate turning point.
    """
    mask = (x >= age0 - window_years) & (x <= age0 + window_years)
    if mask.sum() < 2:
        return 0.0

    return np.sum(np.abs(d1[mask]))


def _filter_turning_points_by_sharpness(
        tp_list, d1, x,
        mode="quantile",
        q=0.5,
        abs_threshold=0.05,
        window_years=1.0
):
    """
    Filter candidate turning points by local sharpness.
    """
    if len(tp_list) == 0:
        return []

    sharpness_vals = np.array([
        _compute_local_sharpness(d1, x, tp["age"], window_years)
        for tp in tp_list
    ])

    if mode == "quantile":
        threshold = np.quantile(sharpness_vals, q)
    elif mode == "absolute":
        threshold = abs_threshold
    else:
        raise ValueError("mode must be 'quantile' or 'absolute'")

    kept = []
    for tp, s_val in zip(tp_list, sharpness_vals):
        if s_val >= threshold:
            kept.append({
                "age": tp["age"],
                "type": tp["type"],
                "sharpness": s_val
            })

    return kept


def _merge_nearby_turning_points(tp_list, merge_window_years=1.0):
    """
    Merge nearby turning points within a posterior draw.
    Ages are averaged within each cluster.
    Type is assigned by majority vote; if tied, use the sharpest point's type.
    """
    if len(tp_list) == 0:
        return []

    tp_sorted = sorted(tp_list, key=lambda z: z["age"])
    clusters = [[tp_sorted[0]]]

    for tp in tp_sorted[1:]:
        if tp["age"] - clusters[-1][-1]["age"] <= merge_window_years:
            clusters[-1].append(tp)
        else:
            clusters.append([tp])

    merged = []
    for cl in clusters:
        ages = np.array([z["age"] for z in cl])
        sharpness = np.array([z["sharpness"] for z in cl])
        types = [z["type"] for z in cl]

        # Majority vote for type
        n_peak = sum(t == "peak" for t in types)
        n_valley = sum(t == "valley" for t in types)

        if n_peak > n_valley:
            final_type = "peak"
        elif n_valley > n_peak:
            final_type = "valley"
        else:
            final_type = cl[int(np.argmax(sharpness))]["type"]

        merged.append({
            "age": float(np.mean(ages)),
            "type": final_type,
            "sharpness": float(np.max(sharpness))
        })

    return merged


def _extract_final_turning_points_from_curve(
        y_curve: np.ndarray,
        age_grid: np.ndarray,
        slope_eps: float = 1e-6,
        sharpness_mode: str = "quantile",
        sharpness_q: float = 0.5,
        sharpness_abs: float = 0.05,
        sharpness_window_years: float = 1.0,
        merge_window_years: float = 1.0,
):
    """
    Full turning-point pipeline for a single trajectory:
    derivative sign change -> sharpness filtering -> nearby merging
    """
    d1 = np.gradient(y_curve, age_grid)

    # Step 1: candidate turning points
    tp_candidates = _classify_turning_points(
        d1, age_grid, eps=slope_eps
    )

    # Step 2: sharpness filter
    tp_filtered = _filter_turning_points_by_sharpness(
        tp_candidates,
        d1,
        age_grid,
        mode=sharpness_mode,
        q=sharpness_q,
        abs_threshold=sharpness_abs,
        window_years=sharpness_window_years
    )

    # Step 3: nearby-point merging
    tp_final = _merge_nearby_turning_points(
        tp_filtered,
        merge_window_years=merge_window_years
    )

    return tp_final


def _check_turning_point_support(all_final_tp,
                                 n_draws_with_tp,
                                 n_draws,
                                 min_total_points=15,
                                 min_draw_support=0.10):
    """
    QC check before final KDE.

    Returns
    -------
    qc : dict
        {
            "n_total_points": ...,
            "n_draws_with_tp": ...,
            "support_ratio": ...,
            "pass_qc": ...
        }
    """
    n_total_points = len(all_final_tp)
    support_ratio = (n_draws_with_tp / n_draws) if n_draws > 0 else 0.0

    pass_qc = (
        (n_total_points >= min_total_points) and
        (support_ratio >= min_draw_support)
    )

    return {
        "n_total_points": int(n_total_points),
        "n_draws_with_tp": int(n_draws_with_tp),
        "support_ratio": float(support_ratio),
        "pass_qc": bool(pass_qc)
    }


def _kde_major_turning_points(
        turning_ages,
        age_min,
        age_max,
        grid_n=1000,
        bw_method="scott",
        peak_prominence=0.01,
        min_support_prop=0.10,
):
    """
    Estimate KDE over turning-point ages and identify major peaks.

    Returns
    -------
    out : dict
        {
            "grid": age_grid_kde,
            "density": density_values,
            "peaks": [
                {
                    "age": peak_age,
                    "density": peak_density,
                    "support_prop": local_support_prop
                },
                ...
            ]
        }
    """
    turning_ages = np.asarray(turning_ages, dtype=float)

    if turning_ages.size < 2:
        return {
            "grid": None,
            "density": None,
            "peaks": []
        }

    kde_grid = np.linspace(age_min, age_max, grid_n)
    kde = gaussian_kde(turning_ages, bw_method=bw_method)
    density = kde(kde_grid)

    peak_idx, props = find_peaks(density, prominence=peak_prominence)

    peaks = []
    for idx in peak_idx:
        peak_age = kde_grid[idx]
        peak_density = density[idx]

        # Estimate local support proportion using a bandwidth-scale neighborhood
        # Here we use 5% of total age range as a simple local support window
        local_window = 0.05 * (age_max - age_min)
        local_support = np.mean(np.abs(turning_ages - peak_age) <= local_window)

        if local_support >= min_support_prop:
            peaks.append({
                "age": float(peak_age),
                "density": float(peak_density),
                "support_prop": float(local_support)
            })

    return {
        "grid": kde_grid,
        "density": density,
        "peaks": peaks
    }


def _run_kde_from_turning_points(
        all_final_tp,
        age_min,
        age_max,
        split_by_type=True,
        min_type_points=8,
        grid_n=1000,
        bw_method="scott",
        peak_prominence=0.01,
        min_support_prop=0.1,
):
    """
    Run final KDE either:
    - separately for peak / valley, or
    - pooled across all turning points

    Returns
    -------
    out : dict
        If split_by_type=True:
            {
                "mode": "split",
                "peak_ages": [...],
                "valley_ages": [...],
                "peak_kde": {...},
                "valley_kde": {...}
            }

        If split_by_type=False:
            {
                "mode": "pooled",
                "turning_ages": [...],
                "turning_kde": {...}
            }
    """
    if split_by_type:
        peak_ages = [z["age"] for z in all_final_tp if z["type"] == "peak"]
        valley_ages = [z["age"] for z in all_final_tp if z["type"] == "valley"]

        if len(peak_ages) >= min_type_points:
            peak_kde = _kde_major_turning_points(
                turning_ages=peak_ages,
                age_min=age_min,
                age_max=age_max,
                grid_n=grid_n,
                bw_method=bw_method,
                peak_prominence=peak_prominence,
                min_support_prop=min_support_prop,
            )
        else:
            peak_kde = {"grid": None, "density": None, "peaks": []}

        if len(valley_ages) >= min_type_points:
            valley_kde = _kde_major_turning_points(
                turning_ages=valley_ages,
                age_min=age_min,
                age_max=age_max,
                grid_n=grid_n,
                bw_method=bw_method,
                peak_prominence=peak_prominence
            )
        else:
            valley_kde = {"grid": None, "density": None, "peaks": []}

        return {
            "mode": "split",
            "peak_ages": peak_ages,
            "valley_ages": valley_ages,
            "peak_kde": peak_kde,
            "valley_kde": valley_kde
        }

    else:
        turning_ages = [z["age"] for z in all_final_tp]

        turning_kde = _kde_major_turning_points(
            turning_ages=turning_ages,
            age_min=age_min,
            age_max=age_max,
            grid_n=grid_n,
            bw_method=bw_method,
            peak_prominence=peak_prominence
        )

        return {
            "mode": "pooled",
            "turning_ages": turning_ages,
            "turning_kde": turning_kde
        }
    

def _compute_sign_change_probability(
        mu_draws: np.ndarray,
        age_grid: np.ndarray,
        tp_age: float,
        tp_mode: str = "any",
        window_years: float = 0.5,
        slope_eps: float = 1e-6
):
    """
    Compute sign-change posterior probability around a candidate turning point age.

    Parameters
    ----------
    mu_draws : array, shape (n_draws, n_grid)
        Posterior sampled trajectories.
    age_grid : array
        Age grid used for prediction.
    tp_age : float
        Final turning point age.
    tp_mode : str
        "peak", "valley", or "any"
    window_years : float
        Half-width of the local age window on each side of tp_age.
    slope_eps : float
        Small threshold for derivative sign.

    Returns
    -------
    prob : float
        Posterior probability of sign change around tp_age.
    """
    left_mask = (age_grid >= tp_age - window_years) & (age_grid < tp_age)
    right_mask = (age_grid > tp_age) & (age_grid <= tp_age + window_years)

    if left_mask.sum() < 2 or right_mask.sum() < 2:
        return np.nan

    count = 0
    valid = 0

    for draw in mu_draws:
        d1 = np.gradient(draw, age_grid)

        left_mean = np.mean(d1[left_mask])
        right_mean = np.mean(d1[right_mask])

        # Threshold tiny slopes to zero
        if abs(left_mean) < slope_eps:
            left_mean = 0.0
        if abs(right_mean) < slope_eps:
            right_mean = 0.0

        if left_mean == 0.0 or right_mean == 0.0:
            continue

        valid += 1

        if tp_mode == "peak":
            ok = (left_mean > 0) and (right_mean < 0)
        elif tp_mode == "valley":
            ok = (left_mean < 0) and (right_mean > 0)
        elif tp_mode == "any":
            ok = (left_mean * right_mean) < 0
        else:
            raise ValueError("tp_mode must be 'peak', 'valley', or 'any'")

        if ok:
            count += 1

    if valid == 0:
        return np.nan

    return count / valid


def _summarize_final_tp_from_posterior(
        all_final_tp,
        mu_draws: np.ndarray,
        age_grid: np.ndarray,
        final_tp_age: float,
        tp_mode: str = "any",
        match_window_years: float = 1.0,
        sign_window_years: float = 0.5,
        slope_eps: float = 1e-6
):
    """
    Summarize a final turning point using posterior evidence.

    Parameters
    ----------
    all_final_tp : list of dict
        All final turning points collected from posterior draws.
        Each element must contain at least 'age' and 'type'.
    mu_draws : array
        Posterior sampled trajectories.
    age_grid : array
        Age grid.
    final_tp_age : float
        Age of final turning point from KDE peak.
    tp_mode : str
        "peak", "valley", or "any"

    Returns
    -------
    out : dict
        {
            "posterior_median_age": ...,
            "ci_low": ...,
            "ci_high": ...,
            "sign_change_posterior_probability": ...,
            "n_matched_points": ...
        }
    """
    if tp_mode == "peak":
        matched = [
            z["age"] for z in all_final_tp
            if (z["type"] == "peak") and (abs(z["age"] - final_tp_age) <= match_window_years)
        ]
    elif tp_mode == "valley":
        matched = [
            z["age"] for z in all_final_tp
            if (z["type"] == "valley") and (abs(z["age"] - final_tp_age) <= match_window_years)
        ]
    elif tp_mode == "any":
        matched = [
            z["age"] for z in all_final_tp
            if abs(z["age"] - final_tp_age) <= match_window_years
        ]
    else:
        raise ValueError("tp_mode must be 'peak', 'valley', or 'any'")

    matched = np.asarray(matched, dtype=float)

    if matched.size == 0:
        median_age = np.nan
        ci_low = np.nan
        ci_high = np.nan
    else:
        median_age = float(np.median(matched))
        ci_low = float(np.percentile(matched, 2.5))
        ci_high = float(np.percentile(matched, 97.5))

    sign_prob = _compute_sign_change_probability(
        mu_draws=mu_draws,
        age_grid=age_grid,
        tp_age=final_tp_age,
        tp_mode=tp_mode,
        window_years=sign_window_years,
        slope_eps=slope_eps
    )

    return {
        "posterior_median_age": median_age,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "sign_change_posterior_probability": sign_prob,
        "n_matched_points": int(matched.size)
    }


def _get_fixed_tp_age(tp_summary_item, use_posterior_median=False):
    """
    Choose the fixed TP age for local permutation testing.
    """
    if use_posterior_median:
        return tp_summary_item["posterior_median_age"]
    else:
        return tp_summary_item["final_tp_age"]


def _empirical_one_sided_pvalue(observed, null_values):
    """
    Empirical one-sided permutation p-value:
    p = (1 + #null >= observed) / (1 + B)
    """
    null_values = np.asarray(null_values, dtype=float)
    return (1.0 + np.sum(null_values >= observed)) / (1.0 + len(null_values))


def _holm_correction(pvals):
    """
    Holm step-down correction.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def _bonferroni_correction(pvals):
    """
    Bonferroni correction.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    return np.minimum(pvals * m, 1.0)


def _fdr_bh_correction(pvals):
    """
    Benjamini-Hochberg FDR correction.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    adjusted = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        adj = ranked[i] * m / rank
        prev = min(prev, adj)
        adjusted[order[i]] = min(prev, 1.0)

    return adjusted


def _apply_multiple_correction(pvals, method=None):
    """
    Apply multiple-comparison correction to a list of p-values.
    """
    if method is None:
        return np.asarray(pvals, dtype=float)
    method = method.lower()

    if method == "bonferroni":
        return _bonferroni_correction(pvals)
    elif method == "holm":
        return _holm_correction(pvals)
    elif method == "fdr_bh":
        return _fdr_bh_correction(pvals)
    else:
        raise ValueError("Unknown correction method")


def _local_turning_point_permutation_test(
        gam_model_obs,
        X_obs, y_obs,
        sex_code, age_grid, age_grid_z,
        fixed_tp_age, tp_mode,
        n_draws, n_bootstraps, perm_n,
        rng,
        model_type, n_splines, lam_grid,
        sign_window_years,
        slope_eps
):
    """
    Run a local permutation test for one fixed turning point.

    Parameters
    ----------
    gam_model_obs : fitted GAM
        Observed-data fitted model.
    X_obs : array
        Observed design matrix.
    y_obs : array
        Observed response.
    sex_code : int
        Sex code for prediction.
    fixed_tp_age : float
        Fixed age at which local sign-change support is tested.
    tp_mode : str
        "peak", "valley", or "any"
    age_grid : array
        Prediction age grid.
    perm_n : int
        Number of permutations.

    Returns
    -------
    out : dict
        {
            "observed_sign_prob": ...,
            "null_sign_probs": ...,
            "p_value": ...
        }
    """
    # ---------------------------------
    # 1) observed sign-change probability
    # ---------------------------------
    X_pred_obs = np.zeros((len(age_grid_z), 2))
    X_pred_obs[:, 0] = age_grid_z
    X_pred_obs[:, 1] = sex_code

    mu_draws_obs = gam_model_obs.sample(
        X_obs,
        y_obs,
        quantity="mu",
        sample_at_X=X_pred_obs,
        n_draws=n_draws,
        n_bootstraps=n_bootstraps
    )

    observed_sign_prob = _compute_sign_change_probability(
        mu_draws=mu_draws_obs,
        age_grid=age_grid,
        tp_age=fixed_tp_age,
        tp_mode=tp_mode,
        window_years=sign_window_years,
        slope_eps=slope_eps,
    )

    # ---------------------------------
    # 2) null distribution under age shuffle
    # ---------------------------------
    null_sign_probs = []

    for _ in tqdm(
            range(perm_n),
            desc='\nPermutation testing for significance of existence of local turning point:'
        ):
        
        shuffled_age = rng.permutation(X_obs[:, 0])
        X_perm = np.column_stack([shuffled_age, X_obs[:, 1]])

        gam_perm = _build_gam(
            model_type=model_type,
            n_splines=n_splines,
            lam=None
        )
        
        gam_perm.gridsearch(X_perm, y_obs, lam=lam_grid, progress=False)

        X_pred_perm = np.zeros((len(age_grid_z), 2))
        X_pred_perm[:, 0] = age_grid_z
        X_pred_perm[:, 1] = sex_code

        mu_draws_perm = gam_perm.sample(
            X_perm,
            y_obs,
            quantity="mu",
            sample_at_X=X_pred_perm,
            n_draws=n_draws,
            n_bootstraps=n_bootstraps
        )

        perm_sign_prob = _compute_sign_change_probability(
            mu_draws=mu_draws_perm,
            age_grid=age_grid,
            tp_age=fixed_tp_age,
            tp_mode=tp_mode,
            window_years=sign_window_years,
            slope_eps=slope_eps
        )

        if np.isnan(perm_sign_prob):
            perm_sign_prob = 0.0

        null_sign_probs.append(perm_sign_prob)

    null_sign_probs = np.asarray(null_sign_probs, dtype=float)

    # ---------------------------------
    # 3) empirical p-value
    # ---------------------------------
    if np.isnan(observed_sign_prob):
        p_value = np.nan
    else:
        p_value = _empirical_one_sided_pvalue(observed_sign_prob, null_sign_probs)

    return {
        "observed_sign_prob": float(observed_sign_prob) if not np.isnan(observed_sign_prob) else np.nan,
        "null_sign_probs": null_sign_probs,
        "p_value": float(p_value) if not np.isnan(p_value) else np.nan
    }
