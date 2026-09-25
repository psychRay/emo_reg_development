#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 03:28:49 2026

@author: dingrui
"""

#%% import necessary modules
import os
import numpy as np
import pandas as pd
import json, pickle

from tqdm import tqdm
from scipy.stats import ttest_1samp

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    LeaveOneGroupOut,
    GroupKFold,
    GridSearchCV,
    permutation_test_score,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)


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


def build_logistic_pipeline(random_state=42):
    """
    Leakage-safe pipeline: all train-dependent transformations stay inside the Pipeline.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=5000,
                class_weight=None,
                random_state=random_state,
            )),
        ]
    )


def make_inner_site_cv(train_site_ids, max_inner_splits=5):
    """
    Build a site-aware inner CV splitter using only training sites.

    Strategy:
    - If number of training sites == 2: use LeaveOneGroupOut()
    - If number of training sites >= 3: use GroupKFold with up to max_inner_splits folds
    """
    n_train_sites = len(np.unique(train_site_ids))

    if n_train_sites < 2:
        raise ValueError(
            "Inner site-aware CV requires at least 2 distinct training sites."
        )

    if n_train_sites == 2:
        return LeaveOneGroupOut()

    n_splits = min(max_inner_splits, n_train_sites)
    return GroupKFold(n_splits=n_splits)


def _get_continuous_outputs(fitted_estimator, X, positive_label):
    """
    Extract continuous scores
    - If predict_proba exists: use positive-class probability
    - Else if decision_function exists: use decision values
    - Else: no ROC-AUC / log_loss
    
    Returns
    -------
    score_values : ndarray or None
        Continuous values usable for ROC-AUC.
    proba_values : ndarray or None
        Positive-class probabilities usable for log_loss.
    score_source : str or None
        'predict_proba', 'decision_function', or None
    """
    # Final classifier classes_ usually live on the last pipeline step
    if hasattr(fitted_estimator, "named_steps"):
        final_clf = fitted_estimator.named_steps[list(fitted_estimator.named_steps.keys())[-1]]
    else:
        final_clf = fitted_estimator

    if not hasattr(final_clf, "classes_"):
        return None, None, None

    classes_ = np.asarray(final_clf.classes_)
    if len(classes_) != 2:
        raise ValueError(f"Only binary classification is supported. Got classes={classes_}")

    pos_idx = np.where(classes_ == positive_label)[0]
    if len(pos_idx) != 1:
        raise ValueError(f"positive_label={positive_label} not found in classes_={classes_}")
    pos_idx = pos_idx[0]

    # 1) Probability path
    if hasattr(fitted_estimator, "predict_proba"):
        proba = fitted_estimator.predict_proba(X)[:, pos_idx]
        return proba, proba, "predict_proba"

    # 2) Decision-function path
    if hasattr(fitted_estimator, "decision_function"):
        decision = fitted_estimator.decision_function(X)
        decision = np.asarray(decision)

        if decision.ndim == 1:
            # For binary decision_function, sklearn conventions tie the sign
            # to class ordering. If positive_label is classes_[0], flip sign.
            score_values = decision.astype(float)
            if pos_idx == 0:
                score_values = -score_values
        else:
            score_values = decision[:, pos_idx].astype(float)

        return score_values, None, "decision_function"

    return None, None, None


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


def nested_leave_one_site_out_compare_models(
    X,
    y,
    subject_ids,
    site_ids,
    model_specs,
    default_scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
    max_inner_site_splits=5,
):
    """
    Run nested LOSO-site CV for multiple candidate models.

    Parameters
    ----------
    model_specs : dict
        Example:
        {
            "model_name": {
                "estimator": sklearn estimator or Pipeline,
                "param_grid": dict or list of dict,
                "scoring": "roc_auc"   # optional, overrides default_scoring
            },
            ...
        }

    Returns
    -------
    results : dict
        {
            "model_results": {
                model_name: {
                    "fold_metrics": DataFrame,
                    "overall_metrics": dict,
                    "oof_pred": ndarray,
                    "oof_score": ndarray,
                    "oof_proba": ndarray,
                    "best_params_per_fold": list,
                },
                ...
            },
            "summary_table": DataFrame
        }
    """
    
    X, y, subject_ids, site_ids = validate_multisite_paired_design(
        X, 
        y, 
        subject_ids, 
        site_ids
    )

    classes = np.unique(y)
    positive_label = classes[1]

    outer_cv = LeaveOneGroupOut()
    model_results = {}
    summary_rows = []

    for model_name, spec in model_specs.items():
        estimator = spec["estimator"]
        param_grid = spec.get("param_grid", {})
        scoring = spec.get("scoring", default_scoring)

        oof_pred = np.empty(len(y), dtype=y.dtype)
        oof_score = np.full(len(y), np.nan, dtype=float)
        oof_proba = np.full(len(y), np.nan, dtype=float)

        fold_rows = []
        best_params_per_fold = []

        for fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, y, groups=site_ids), start=1
        ):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            train_sites = site_ids[train_idx]
            held_out_site = np.unique(site_ids[test_idx])[0]

            inner_cv = make_inner_site_cv(
                train_site_ids=train_sites,
                max_inner_splits=max_inner_site_splits,
            )

            search = GridSearchCV(
                estimator=clone(estimator),
                param_grid=param_grid,
                scoring=scoring,
                cv=inner_cv,
                n_jobs=n_jobs,
                refit=True,
                return_train_score=False,
            )

            search.fit(X_train, y_train, groups=train_sites)
            best_model = search.best_estimator_

            y_pred = best_model.predict(X_test)
            score_values, proba_values, score_source = _get_continuous_outputs(
                best_model, X_test, positive_label=positive_label
            )

            oof_pred[test_idx] = y_pred
            if score_values is not None:
                oof_score[test_idx] = score_values
            if proba_values is not None:
                oof_proba[test_idx] = proba_values

            y_test_bin = (y_test == positive_label).astype(int)

            fold_result = {
                "model_name": model_name,
                "fold": fold,
                "held_out_site": held_out_site,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "y_true": y_test,
                "oof_pred": y_pred,
                "oof_score": score_values,
                "oof_proba": proba_values,
                "score_source": score_source,
                "best_params": search.best_params_,
                "inner_best_score": search.best_score_,
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "roc_auc": (
                    roc_auc_score(y_test_bin, score_values)
                    if score_values is not None else np.nan
                ),
                "log_loss": (
                    log_loss(y_test_bin, proba_values, labels=[0, 1])
                    if proba_values is not None else np.nan
                ),
            }

            fold_rows.append(fold_result)
            best_params_per_fold.append(search.best_params_)

        # Overall metrics
        oof_pred = np.asarray(oof_pred, dtype=y.dtype)
        y_bin = (y == positive_label).astype(int)

        overall = {
            "model_name": model_name,
            "n_sites": len(np.unique(site_ids)),
            "n_subjects": len(np.unique(subject_ids)),
            "accuracy": accuracy_score(y, oof_pred),
            "balanced_accuracy": balanced_accuracy_score(y, oof_pred),
            "roc_auc": (
                roc_auc_score(y_bin, oof_score)
                if not np.any(np.isnan(oof_score)) else np.nan
            ),
            "log_loss": (
                log_loss(y_bin, oof_proba, labels=[0, 1])
                if not np.any(np.isnan(oof_proba)) else np.nan
            ),
            "confusion_matrix": confusion_matrix(y, oof_pred, labels=classes),
        }

        model_results[model_name] = {
            "fold_metrics": pd.DataFrame(fold_rows),
            "overall_metrics": overall,
            "oof_pred": oof_pred,
            "oof_score": oof_score,
            "oof_proba": oof_proba,
            "best_params_per_fold": best_params_per_fold,
        }

        summary_rows.append({
            "model_name": model_name,
            "accuracy": overall["accuracy"],
            "balanced_accuracy": overall["balanced_accuracy"],
            "roc_auc": overall["roc_auc"],
            "log_loss": overall["log_loss"],
        })

    summary_table = pd.DataFrame(summary_rows).sort_values(
        by=["roc_auc", "balanced_accuracy", "accuracy"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return {
        "model_results": model_results,
        "summary_table": summary_table,
    }


def fit_final_multisite_models(
    X,
    y,
    subject_ids,
    site_ids,
    model_specs,
    default_scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
    force_logo_inner=True,
    max_inner_site_splits=5,
):
    
    X, y, subject_ids, site_ids = validate_multisite_paired_design(
        X, 
        y, 
        subject_ids, 
        site_ids
    )

    inner_cv = make_inner_site_cv(
        train_site_ids=site_ids,
        max_inner_splits=max_inner_site_splits,
    )

    fitted = {}

    for model_name, spec in model_specs.items():
        estimator = spec["estimator"]
        param_grid = spec.get("param_grid", {})
        scoring = spec.get("scoring", default_scoring)

        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
        )
        search.fit(X, y, groups=site_ids)

        fitted[model_name] = {
            "best_estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": pd.DataFrame(search.cv_results_),
        }

    return fitted


def permute_labels_within_subjects(y, subject_ids, random_state=None):
    """
    Permute labels only within each subject pair.

    Assumptions
    ----------
    - Each subject appears exactly twice.
    - The two samples correspond to two different conditions/classes.

    Mechanism
    ---------
    For each subject, with probability 0.5, swap the two labels.
    Otherwise keep them unchanged.

    Returns
    -------
    y_perm : ndarray, shape (n_samples,)
    """
    rng = np.random.RandomState(random_state)
    y = np.asarray(y).copy()
    subject_ids = np.asarray(subject_ids)

    y_perm = y.copy()

    unique_subjects, counts = np.unique(subject_ids, return_counts=True)
    if not np.all(counts == 2):
        raise ValueError("Each subject must appear exactly twice for within-subject permutation.")

    for subj in unique_subjects:
        idx = np.where(subject_ids == subj)[0]
        if len(idx) != 2:
            raise ValueError(f"Subject {subj} does not have exactly two samples.")

        # With probability 0.5, swap the two labels
        if rng.rand() < 0.5:
            y_perm[idx[0]], y_perm[idx[1]] = y_perm[idx[1]], y_perm[idx[0]]

    return y_perm


def multisite_permutation_test(
    X,
    y,
    subject_ids,
    site_ids,
    model_specs,
    n_permutations=1000,
    metric_for_pvalue="roc_auc",
    default_scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
    max_inner_site_splits=5,
    observed_results=None,
    return_full_observed_results=True,
):
    """
    Manual permutation test for multi-model nested LOSO-site CV.

    Parameters
    ----------
    observed_results : dict or None
        Optional precomputed output from
        nested_leave_one_site_out_compare_models(...).
        If provided, observed metrics will be read from it directly,
        and the observed model fitting will NOT be re-run.
    """
    rng = np.random.RandomState(random_state)

    # -------------------------
    # 1) Observed results
    # -------------------------
    if observed_results is None:
        observed_results = nested_leave_one_site_out_compare_models(
            X=X,
            y=y,
            subject_ids=subject_ids,
            site_ids=site_ids,
            model_specs=model_specs,
            default_scoring=default_scoring,
            random_state=random_state,
            n_jobs=n_jobs,
            max_inner_site_splits=max_inner_site_splits,
        )

    observed_summary = observed_results["summary_table"].copy()

    if metric_for_pvalue not in observed_summary.columns:
        raise ValueError(
            f"metric_for_pvalue='{metric_for_pvalue}' not found in observed summary table. "
            f"Available columns: {list(observed_summary.columns)}"
        )

    model_names = list(observed_results["model_results"].keys())

    observed_scores = {}
    for model_name in model_names:
        observed_scores[model_name] = observed_results["model_results"][model_name]["overall_metrics"][metric_for_pvalue]

    # -------------------------
    # 2) Permutation loop
    # -------------------------
    permutation_scores = {
        model_name: np.empty(n_permutations, dtype=float)
        for model_name in model_names
    }

    for i in tqdm(
            range(n_permutations),
            desc='Permutation test for significance of ML model metric:'
        ):
        y_perm = permute_labels_within_subjects(
            y=y,
            subject_ids=subject_ids,
            random_state=rng.randint(0, 2**31 - 1),
        )

        perm_results = nested_leave_one_site_out_compare_models(
            X=X,
            y=y_perm,
            subject_ids=subject_ids,
            site_ids=site_ids,
            model_specs=model_specs,
            default_scoring=default_scoring,
            random_state=random_state,
            n_jobs=n_jobs,
            max_inner_site_splits=max_inner_site_splits,
        )

        for model_name in model_names:
            permutation_scores[model_name][i] = (
                perm_results["model_results"][model_name]["overall_metrics"][metric_for_pvalue]
            )

    # -------------------------
    # 3) Empirical p-values
    # -------------------------
    model_permutation_results = {}
    summary_rows = []

    for model_name in model_names:
        obs = observed_scores[model_name]
        perm = permutation_scores[model_name]

        pvalue = (np.sum(perm >= obs) + 1.0) / (n_permutations + 1.0)

        model_permutation_results[model_name] = {
            "observed_score": obs,
            "permutation_scores": perm,
            "pvalue": pvalue,
        }

        summary_rows.append({
            "model_name": model_name,
            "metric_for_pvalue": metric_for_pvalue,
            "observed_score": obs,
            "perm_mean": np.mean(perm),
            "perm_std": np.std(perm, ddof=1),
            "perm_median": np.median(perm),
            "perm_q95": np.quantile(perm, 0.95),
            "pvalue": pvalue,
        })

    permutation_summary = pd.DataFrame(summary_rows).sort_values(
        by=["pvalue", "observed_score"],
        ascending=[True, False]
    ).reset_index(drop=True)

    out = {
        "observed_summary": observed_summary,
        "permutation_summary": permutation_summary,
        "model_permutation_results": model_permutation_results,
    }

    if return_full_observed_results:
        out["observed_results"] = observed_results

    return out


def fdr_bh(pvals, q=0.05):
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals : array-like
        P-values. Can be 1D or 2D.
    q : float
        Target FDR level.

    Returns
    -------
    reject : ndarray, bool
        Boolean mask of FDR-significant features.
    p_adj : ndarray
        FDR-adjusted p-values.
    """
    pvals = np.asarray(pvals, dtype=float)
    original_shape = pvals.shape
    p = pvals.ravel()

    if np.any(np.isnan(p)):
        raise ValueError("pvals contains NaN.")

    m = len(p)
    order = np.argsort(p)
    ranked_p = p[order]

    adj = ranked_p * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    p_adj = np.empty_like(adj)
    p_adj[order] = adj

    reject = p_adj <= q

    return reject.reshape(original_shape), p_adj.reshape(original_shape)


def get_final_estimator(fitted_estimator):
    """
    Return the final estimator from either a Pipeline or a plain estimator.
    """
    if hasattr(fitted_estimator, "named_steps"):
        last_step_name = list(fitted_estimator.named_steps.keys())[-1]
        return fitted_estimator.named_steps[last_step_name]
    return fitted_estimator


def extract_linear_weight_matrix(
    fitted_estimator,
    positive_label=None,
):
    """
    Extract linear decoding weights from a fitted linear classifier.

    Returns
    -------
    W : ndarray, shape (n_features, n_latent)
        Weight matrix. For binary classification, n_latent = 1.
    classes : ndarray or None
        Class labels from the final classifier, if available.
    """
    clf = get_final_estimator(fitted_estimator)

    if not hasattr(clf, "coef_"):
        raise TypeError(
            "The fitted estimator does not expose coef_. "
            "Fig.3a-style decoding weight maps require a linear model."
        )

    coef = np.asarray(clf.coef_, dtype=float)

    if coef.ndim == 1:
        W = coef.reshape(-1, 1)
    else:
        # sklearn coef_: usually (n_classes_or_1, n_features)
        # Haufe / pattern convention: (n_features, n_latent)
        W = coef.T

    classes = getattr(clf, "classes_", None)

    # For binary sklearn classifiers, coef_ usually points toward classes_[1].
    # If user wants the opposite class to be positive, flip sign.
    if positive_label is not None and classes is not None and W.shape[1] == 1:
        classes = np.asarray(classes)
        if len(classes) == 2:
            if positive_label == classes[0]:
                W = -W
            elif positive_label != classes[1]:
                raise ValueError(
                    f"positive_label={positive_label} not found in binary classes={classes}."
                )

    return W, classes


def convert_standardized_weights_to_original_space(
    fitted_estimator,
    W,
):
    """
    Convert coefficients from StandardScaler-transformed feature space
    back to the original feature scale.

    This only supports the common Pipeline:
        StandardScaler -> linear classifier

    If the pipeline contains PCA, feature selection, or other feature-changing
    transforms, do not use this conversion.
    """
    if not hasattr(fitted_estimator, "named_steps"):
        return W

    steps = fitted_estimator.named_steps

    if "scaler" not in steps:
        return W

    scaler = steps["scaler"]

    if not hasattr(scaler, "scale_"):
        return W

    scale = np.asarray(scaler.scale_, dtype=float)

    if scale.shape[0] != W.shape[0]:
        raise ValueError(
            "Cannot convert weights to original space because scaler.scale_ "
            "does not match the number of model features. This usually means "
            "the pipeline contains feature-changing steps such as PCA or feature selection."
        )

    return W / scale.reshape(-1, 1)


def make_bootstrap_indices(
    y,
    subject_ids=None,
    bootstrap_unit="subject",
    rng=None,
):
    """
    Generate bootstrap indices.

    Parameters
    ----------
    y : array-like
        Labels.
    subject_ids : array-like or None
        Subject IDs. Required if bootstrap_unit='subject'.
    bootstrap_unit : {"subject", "sample"}
        - "subject": resample subjects with replacement and keep all samples per subject.
        - "sample": resample samples with replacement.
    rng : np.random.RandomState

    Returns
    -------
    boot_idx : ndarray
        Sample indices for one bootstrap dataset.
    """
    y = np.asarray(y)

    if rng is None:
        rng = np.random.RandomState(None)

    if bootstrap_unit == "sample":
        return rng.choice(np.arange(len(y)), size=len(y), replace=True)

    if bootstrap_unit != "subject":
        raise ValueError("bootstrap_unit must be either 'subject' or 'sample'.")

    if subject_ids is None:
        raise ValueError("subject_ids must be provided when bootstrap_unit='subject'.")

    subject_ids = np.asarray(subject_ids)
    unique_subjects = np.unique(subject_ids)

    sampled_subjects = rng.choice(
        unique_subjects,
        size=len(unique_subjects),
        replace=True,
    )

    boot_idx = []
    for subj in sampled_subjects:
        boot_idx.extend(np.where(subject_ids == subj)[0])

    return np.asarray(boot_idx, dtype=int)


def bootstrap_two_sided_p_from_weights(W_boot, add_one=True):
    """
    Compute two-sided bootstrap sign p-values for each feature/latent dimension.

    Parameters
    ----------
    W_boot : ndarray, shape (n_bootstraps, n_features, n_latent)
        Bootstrapped decoding weights.
    add_one : bool
        If True, use a +1 correction to avoid zero p-values.

    Returns
    -------
    pvals : ndarray, shape (n_features, n_latent)
    """
    W_boot = np.asarray(W_boot, dtype=float)

    if W_boot.ndim != 3:
        raise ValueError("W_boot must have shape (n_bootstraps, n_features, n_latent).")

    n_boot = W_boot.shape[0]

    n_pos = np.sum(W_boot >= 0, axis=0)
    n_neg = np.sum(W_boot <= 0, axis=0)

    if add_one:
        pvals = 2.0 * (np.minimum(n_pos, n_neg) + 1.0) / (n_boot + 1.0)
    else:
        pvals = 2.0 * np.minimum(n_pos, n_neg) / n_boot

    return np.clip(pvals, 0, 1)


def compute_bootstrapped_decoding_map(
    X,
    y,
    model_spec,
    best_params=None,
    subject_ids=None,
    n_bootstraps=10000,
    q=0.05,
    unc_p_threshold=None,
    bootstrap_unit="subject",
    positive_label=None,
    weight_space="transformed",
    map_estimate="full",
    random_state=42,
    feature_names=None,
    verbose=True,
):
    """
    Compute a Jiang Fig.3a-style bootstrapped decoding weight map.

    This function implements the decoding-map part of Fig.3a:
        1. Fit a linear estimator.
        2. Bootstrap training samples/subjects with replacement.
        3. Refit the same model in each bootstrap sample.
        4. Extract linear decoding weights.
        5. Compute two-sided bootstrap sign p-values.
        6. Apply FDR correction.
        7. Return thresholded positive/negative decoding map.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Parcel-wise beta maps.
    y : ndarray, shape (n_samples,)
        Binary or multiclass labels.
    model_spec : dict
        One entry from model_specs, e.g. model_specs["linear_svc"].
        Must contain:
            - "estimator"
        Example:
            model_spec = model_specs["linear_svc"]
    best_params : dict or None
        Parameters to set before fitting, e.g. {"clf__C": 1.0}.
        If None, the estimator's current default parameters are used.
    subject_ids : array-like or None
        Subject IDs. Required for subject-level bootstrap.
    n_bootstraps : int
        Number of bootstrap refits. Jiang et al. used 10,000.
    q : float
        FDR threshold.
    unc_p_threshold : float or None
        Optional additional uncorrected p threshold.
    bootstrap_unit : {"subject", "sample"}
        Recommended: "subject" for paired/repeated-measures fMRI data.
    positive_label : label or None
        For binary classifiers, controls which class defines the positive direction.
    weight_space : {"transformed", "original"}
        - "transformed": use weights in the feature space seen by the classifier.
        - "original": convert StandardScaler-based weights back to original feature scale.
          Only valid for simple StandardScaler -> linear classifier pipelines.
    map_estimate : {"full", "bootstrap_mean"}
        Which weight estimate to threshold:
        - "full": full-data model weights, masked by bootstrap significance.
        - "bootstrap_mean": mean bootstrap weights, masked by bootstrap significance.
    random_state : int
    feature_names : list-like or None
        Optional feature names for output table.
    verbose : bool

    Returns
    -------
    result : dict
        {
            "full_weight": ndarray, shape (n_features, n_latent),
            "mean_boot_weight": ndarray, shape (n_features, n_latent),
            "p_unc": ndarray, shape (n_features, n_latent),
            "p_fdr": ndarray, shape (n_features, n_latent),
            "fdr_mask": ndarray, bool,
            "final_mask": ndarray, bool,
            "thresholded_map": ndarray, shape (n_features, n_latent),
            "weights_boot": ndarray, shape (n_bootstraps, n_features, n_latent),
            "classes": ndarray or None,
            "feature_table": pandas.DataFrame
        }
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")

    if len(y) != X.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    if bootstrap_unit == "subject":
        if subject_ids is None:
            raise ValueError("subject_ids is required for subject-level bootstrap.")
        subject_ids = np.asarray(subject_ids)
        if len(subject_ids) != X.shape[0]:
            raise ValueError("subject_ids must have the same length as y.")

    if weight_space not in ["transformed", "original"]:
        raise ValueError("weight_space must be 'transformed' or 'original'.")

    if map_estimate not in ["full", "bootstrap_mean"]:
        raise ValueError("map_estimate must be 'full' or 'bootstrap_mean'.")

    base_estimator = model_spec["estimator"]
    rng = np.random.RandomState(random_state)

    # -------------------------
    # 1) Fit full-data model
    # -------------------------
    full_est = clone(base_estimator)
    if best_params is not None:
        full_est.set_params(**best_params)
    full_est.fit(X, y)

    W_full, classes = extract_linear_weight_matrix(
        full_est,
        positive_label=positive_label,
    )

    if weight_space == "original":
        W_full = convert_standardized_weights_to_original_space(full_est, W_full)

    n_features, n_latent = W_full.shape

    # -------------------------
    # 2) Bootstrap refits
    # -------------------------
    W_boot = np.empty((n_bootstraps, n_features, n_latent), dtype=float)

    for b in tqdm(
            range(n_bootstraps),
            desc='\t- Bootstrapping analysis for significant decoding weights map'
    ):
        boot_idx = make_bootstrap_indices(
            y=y,
            subject_ids=subject_ids,
            bootstrap_unit=bootstrap_unit,
            rng=rng,
        )

        X_b = X[boot_idx]
        y_b = y[boot_idx]

        if len(np.unique(y_b)) < 2:
            raise RuntimeError(
                "A bootstrap sample contains fewer than two classes. "
                "Use subject-level bootstrap or check class balance."
            )

        est_b = clone(base_estimator)
        if best_params is not None:
            est_b.set_params(**best_params)

        est_b.fit(X_b, y_b)

        W_b, _ = extract_linear_weight_matrix(
            est_b,
            positive_label=positive_label,
        )

        if weight_space == "original":
            W_b = convert_standardized_weights_to_original_space(est_b, W_b)

        if W_b.shape != (n_features, n_latent):
            raise RuntimeError(
                f"Bootstrap weight shape mismatch at iteration {b}: "
                f"expected {(n_features, n_latent)}, got {W_b.shape}."
            )

        W_boot[b] = W_b

        if verbose and (b + 1) % max(1, n_bootstraps // 10) == 0:
            print(f"Bootstrap {b + 1}/{n_bootstraps} completed.")

    # -------------------------
    # 3) Bootstrap p-values + FDR
    # -------------------------
    p_unc = bootstrap_two_sided_p_from_weights(W_boot, add_one=True)

    fdr_mask, p_fdr = fdr_bh(p_unc, q=q)

    if unc_p_threshold is not None:
        final_mask = fdr_mask & (p_unc < unc_p_threshold)
    else:
        final_mask = fdr_mask

    W_mean = np.mean(W_boot, axis=0)

    if map_estimate == "full":
        W_display = W_full
    else:
        W_display = W_mean

    thresholded_map = np.where(final_mask, W_display, 0.0)

    # -------------------------
    # 4) Feature-level table
    # -------------------------
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    if len(feature_names) != n_features:
        raise ValueError("feature_names must have length n_features.")

    rows = []
    for k in range(n_latent):
        for j, fname in enumerate(feature_names):
            rows.append({
                "feature": fname,
                "latent_dim": k,
                "full_weight": W_full[j, k],
                "mean_boot_weight": W_mean[j, k],
                "p_unc": p_unc[j, k],
                "p_fdr": p_fdr[j, k],
                "fdr_significant": bool(fdr_mask[j, k]),
                "final_significant": bool(final_mask[j, k]),
                "thresholded_weight": thresholded_map[j, k],
                "sign": (
                    "positive" if thresholded_map[j, k] > 0
                    else "negative" if thresholded_map[j, k] < 0
                    else "zero"
                ),
            })

    feature_table = pd.DataFrame(rows)

    return {
        "full_weight": W_full,
        "mean_boot_weight": W_mean,
        "p_unc": p_unc,
        "p_fdr": p_fdr,
        "fdr_mask": fdr_mask,
        "final_mask": final_mask,
        "thresholded_map": thresholded_map,
        "weights_boot": W_boot,
        "classes": classes,
        "feature_table": feature_table,
        "settings": {
            "n_bootstraps": n_bootstraps,
            "q": q,
            "unc_p_threshold": unc_p_threshold,
            "bootstrap_unit": bootstrap_unit,
            "positive_label": positive_label,
            "weight_space": weight_space,
            "map_estimate": map_estimate,
            "random_state": random_state,
        },
    }


def haufe_transform_general(X_train_transformed, W, ddof=1, rcond=1e-12):
    """
    General Haufe transformation:

        A = cov(X) @ W @ cov(S)^(-1), where S = X @ W

    Parameters
    ----------
    X_train_transformed : ndarray, shape (n_samples, n_features)
        Feature matrix actually seen by the final linear classifier.
    W : ndarray, shape (n_features,) or (n_features, n_latent)
        Decoding weight vector or matrix.
    ddof : int
    rcond : float

    Returns
    -------
    A : ndarray, shape (n_features,) or (n_features, n_latent)
        Haufe-transformed encoding pattern.
    """
    X = np.asarray(X_train_transformed, dtype=float)
    W = np.asarray(W, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")

    input_w_was_1d = W.ndim == 1
    if input_w_was_1d:
        W = W.reshape(-1, 1)

    if W.ndim != 2:
        raise ValueError(f"W must be 1D or 2D, got shape={W.shape}")

    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"Feature mismatch: X has {X.shape[1]} features, "
            f"but W has shape {W.shape}."
        )

    S = X @ W

    cov_X = np.cov(X, rowvar=False, ddof=ddof)

    if S.shape[1] == 1:
        cov_S = np.array([[np.var(S[:, 0], ddof=ddof)]], dtype=float)
    else:
        cov_S = np.cov(S, rowvar=False, ddof=ddof)

    cov_S_inv = np.linalg.pinv(cov_S, rcond=rcond)
    A = cov_X @ W @ cov_S_inv

    if input_w_was_1d:
        return A[:, 0]

    return A


def _transform_features_before_final_estimator(fitted_estimator, X):
    """
    Apply all Pipeline steps before the final classifier.

    If estimator is not a Pipeline, return X unchanged.
    """
    X = np.asarray(X, dtype=float)

    if not hasattr(fitted_estimator, "named_steps"):
        return X

    Xt = X
    steps = list(fitted_estimator.named_steps.items())

    for name, step in steps[:-1]:
        Xt = step.transform(Xt)

    return np.asarray(Xt, dtype=float)


def _extract_W_from_fitted_linear_estimator(
    fitted_estimator,
    positive_label=None,
):
    """
    Extract sklearn linear classifier coef_ as W with shape:
        (n_features, n_latent)

    For binary classifiers, n_latent = 1.
    """
    if hasattr(fitted_estimator, "named_steps"):
        clf = fitted_estimator.named_steps[list(fitted_estimator.named_steps.keys())[-1]]
    else:
        clf = fitted_estimator

    if not hasattr(clf, "coef_"):
        raise TypeError(
            "The fitted estimator does not expose coef_. "
            "Fig.4-style reconstructed activation patterns require a linear model."
        )

    coef = np.asarray(clf.coef_, dtype=float)

    if coef.ndim == 1:
        W = coef.reshape(-1, 1)
    else:
        W = coef.T

    classes = getattr(clf, "classes_", None)
    
    # Binary direction control
    if positive_label is not None and classes is not None and W.shape[1] == 1:
        classes = np.asarray(classes)
        if len(classes) == 2:
            if positive_label == classes[0]:
                W = -W
            elif positive_label != classes[1]:
                raise ValueError(
                    f"positive_label={positive_label} not found in classes={classes}."
                )

    return W, classes


def fit_and_compute_haufe_map(
    X,
    y,
    estimator,
    best_params=None,
    positive_label=None,
):
    """
    Fit a fixed-parameter linear model and compute Haufe reconstructed activation map.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    estimator : sklearn estimator or Pipeline
    best_params : dict or None
        Fixed final hyperparameters.
    positive_label : label or None
        Positive class direction for binary classifiers.

    Returns
    -------
    result : dict
        {
            "fitted_estimator": fitted estimator,
            "W": decoding weight matrix,
            "A": Haufe reconstructed activation pattern,
            "classes": class labels
        }
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    est = clone(estimator)

    if best_params is not None:
        est.set_params(**best_params)

    est.fit(X, y)

    W, classes = _extract_W_from_fitted_linear_estimator(
        fitted_estimator=est,
        positive_label=positive_label,
    )

    X_t = _transform_features_before_final_estimator(est, X)

    A = haufe_transform_general(
        X_train_transformed=X_t,
        W=W,
    )

    return {
        "fitted_estimator": est,
        "W": W,
        "A": A,
        "classes": classes,
    }


def compute_reconstructed_activation_pattern(
    X,
    y,
    model_spec,
    best_params=None,
    subject_ids=None,
    n_bootstraps=10000,
    bootstrap_unit="subject",
    positive_label=None,
    random_state=42,
    feature_names=None,
    # BSR options
    compute_bsr=True,
    bsr_threshold=3.0,
    top_bsr_percent=None,
    # FDR options
    compute_fdr=True,
    fdr_q=0.05,
    fdr_method="ttest",
    unc_p_threshold=None,
    # Display options
    use_pattern_for_display="final",
    store_bootstrap_patterns=True,
    verbose=True,
):
    """
    Compute Haufe reconstructed activation pattern with both:
    1. Bootstrap Ratio (BSR) stability/effect-size assessment.
    2. Bootstrap-based FDR feature-wise inference.

    This function is intended for final-model interpretability, not for
    out-of-fold prediction. It fits the final model on all data, computes
    the Haufe-transformed reconstructed activation pattern, then performs
    bootstrap refits to estimate pattern stability and feature-wise inference.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Parcel-wise beta maps.

    y : ndarray, shape (n_samples,)
        Binary or multiclass labels.

    model_spec : dict
        One model specification from model_specs, e.g. model_specs["linear_svc"].
        Must contain key "estimator".

    best_params : dict or None
        Final hyperparameters, e.g. final_models[model_name]["best_params"].

    subject_ids : ndarray or None
        Subject IDs. Required if bootstrap_unit="subject".

    n_bootstraps : int
        Number of bootstrap refits.

    bootstrap_unit : {"subject", "sample"}
        "subject" is recommended for paired/repeated-measures fMRI data.

    positive_label : scalar or None
        Positive class direction for binary classifiers.

    random_state : int
        Random seed.

    feature_names : list-like or None
        Optional feature / parcel names.

    compute_bsr : bool
        Whether to compute bootstrap ratio:
            BSR = mean(bootstrap Haufe maps) / std(bootstrap Haufe maps)

    bsr_threshold : float or None
        Absolute BSR threshold. Common choices: 3.0 or 4.0.
        If None, BSR thresholding is skipped.

    top_bsr_percent : float or None
        If provided, retain the top X percent features by absolute BSR.
        Example: 5 means top 5%.

    compute_fdr : bool
        Whether to compute bootstrap-based FDR inference.

    fdr_q : float
        FDR threshold.

    fdr_method : {"ttest", "sign"}
        Method for bootstrap-based feature-wise p-values.

        "ttest":
            One-sample t-test of bootstrap Haufe values against zero.
            This mimics a conventional bootstrap-based group-level test,
            but should be interpreted cautiously because bootstrap samples
            are not independent subjects.

        "sign":
            Two-sided bootstrap sign-stability p-value:
                p = 2 * min(P(A_boot >= 0), P(A_boot <= 0))
            This is more conservative for directional stability.

    unc_p_threshold : float or None
        Optional additional uncorrected p-value threshold, e.g. 0.005.
        If provided:
            fdr_final_mask = fdr_mask & (p_unc < unc_p_threshold)

    use_pattern_for_display : {"final", "bootstrap_mean"}
        Which Haufe pattern to threshold for display:
        - "final": threshold the full-data final-model Haufe pattern.
        - "bootstrap_mean": threshold the mean bootstrap Haufe pattern.

    store_bootstrap_patterns : bool
        Whether to return all bootstrap Haufe maps.
        Set False to reduce memory usage.

    verbose : bool
        Whether to print bootstrap progress.

    Returns
    -------
    result : dict
        {
            "final_encoding_pattern": ndarray, shape (n_features, n_latent),
            "mean_boot_encoding_pattern": ndarray,
            "std_boot_encoding_pattern": ndarray,

            "bsr_map": ndarray or None,
            "bsr_mask": ndarray or None,
            "bsr_thresholded_pattern": ndarray or None,
            "top_bsr_mask": ndarray or None,
            "top_bsr_pattern": ndarray or None,

            "fdr_t_values": ndarray or None,
            "fdr_p_unc": ndarray or None,
            "fdr_p_fdr": ndarray or None,
            "fdr_mask": ndarray or None,
            "fdr_final_mask": ndarray or None,
            "fdr_thresholded_pattern": ndarray or None,

            "encoding_patterns_boot": ndarray or None,
            "classes": ndarray or None,
            "feature_table": DataFrame,
            "settings": dict
        }

    Notes
    -----
    BSR should be interpreted as a stability / effect-size index, not as a
    strict independent-sample t statistic.

    FDR results based on bootstrap t-tests are useful for comparison with
    prior work but should be interpreted cautiously in large samples because
    bootstrap iterations are not independent subjects.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")

    if len(y) != X.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    if "estimator" not in model_spec:
        raise KeyError("model_spec must contain key 'estimator'.")

    if bootstrap_unit not in ["subject", "sample"]:
        raise ValueError("bootstrap_unit must be either 'subject' or 'sample'.")

    if fdr_method not in ["ttest", "sign"]:
        raise ValueError("fdr_method must be either 'ttest' or 'sign'.")

    if bootstrap_unit == "subject":
        if subject_ids is None:
            raise ValueError("subject_ids is required when bootstrap_unit='subject'.")
        subject_ids = np.asarray(subject_ids)
        if len(subject_ids) != X.shape[0]:
            raise ValueError("subject_ids must have the same number of samples as X.")

    if use_pattern_for_display not in ["final", "bootstrap_mean"]:
        raise ValueError("use_pattern_for_display must be 'final' or 'bootstrap_mean'.")

    n_samples, n_features = X.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    if len(feature_names) != n_features:
        raise ValueError("feature_names must have length equal to n_features.")

    rng = np.random.RandomState(random_state)
    base_estimator = model_spec["estimator"]

    # --------------------------------------------------
    # 1. Fit final model on full data
    # --------------------------------------------------
    final_est = clone(base_estimator)

    if best_params is not None:
        final_est.set_params(**best_params)

    final_est.fit(X, y)

    W_final, classes = _extract_W_from_fitted_linear_estimator(
        fitted_estimator=final_est,
        positive_label=positive_label,
    )

    X_final_t = _transform_features_before_final_estimator(final_est, X)

    A_final = haufe_transform_general(
        X_train_transformed=X_final_t,
        W=W_final,
    )

    if A_final.ndim == 1:
        A_final = A_final.reshape(-1, 1)

    n_features_A, n_latent = A_final.shape

    if n_features_A != n_features:
        raise ValueError(
            "Feature dimension of the Haufe map does not match X. "
            "This may happen if the pipeline contains feature-changing steps "
            "such as PCA or feature selection."
        )

    # --------------------------------------------------
    # 2. Bootstrap refits and Haufe maps
    # --------------------------------------------------
    A_boot = np.empty((n_bootstraps, n_features, n_latent), dtype=float)

    for b in tqdm(
            range(n_bootstraps),
            desc='Bootstrapping analysis for significant/reliable encoding activation map',
    ):
        if bootstrap_unit == "subject":
            boot_idx = make_bootstrap_indices(y, subject_ids, 'subject', rng)
        else:
            boot_idx = make_bootstrap_indices(y, subject_ids, 'sample', rng)

        X_b = X[boot_idx]
        y_b = y[boot_idx]

        if len(np.unique(y_b)) < 2:
            raise RuntimeError(
                "A bootstrap sample contains fewer than two classes. "
                "Use subject-level bootstrap or check class balance."
            )

        est_b = clone(base_estimator)

        if best_params is not None:
            est_b.set_params(**best_params)

        est_b.fit(X_b, y_b)

        W_b, _ = _extract_W_from_fitted_linear_estimator(
            fitted_estimator=est_b,
            positive_label=positive_label,
        )

        X_b_t = _transform_features_before_final_estimator(est_b, X_b)

        A_b = haufe_transform_general(
            X_train_transformed=X_b_t,
            W=W_b,
        )

        if A_b.ndim == 1:
            A_b = A_b.reshape(-1, 1)

        if A_b.shape != A_final.shape:
            raise RuntimeError(
                f"Bootstrap Haufe map shape mismatch at iteration {b}: "
                f"expected {A_final.shape}, got {A_b.shape}."
            )

        A_boot[b] = A_b

        if verbose and (b + 1) % max(1, n_bootstraps // 10) == 0:
            print(f"Bootstrap {b + 1}/{n_bootstraps} completed.")

    # --------------------------------------------------
    # 3. Bootstrap summary
    # --------------------------------------------------
    A_boot_mean = np.mean(A_boot, axis=0)
    A_boot_std = np.std(A_boot, axis=0, ddof=1)

    if use_pattern_for_display == "final":
        display_pattern = A_final
    else:
        display_pattern = A_boot_mean

    # --------------------------------------------------
    # 4. BSR analysis
    # --------------------------------------------------
    if compute_bsr:
        eps = np.finfo(float).eps
        bsr_map = A_boot_mean / np.maximum(A_boot_std, eps)

        if bsr_threshold is not None:
            bsr_mask = np.abs(bsr_map) >= bsr_threshold
            bsr_thresholded_pattern = np.where(bsr_mask, display_pattern, 0.0)
        else:
            bsr_mask = np.full_like(bsr_map, False, dtype=bool)
            bsr_thresholded_pattern = None

        if top_bsr_percent is not None:
            if not (0 < top_bsr_percent <= 100):
                raise ValueError("top_bsr_percent must be in (0, 100].")

            abs_bsr = np.abs(bsr_map)
            top_bsr_cutoff = np.percentile(abs_bsr.ravel(), 100.0 - top_bsr_percent)
            top_bsr_mask = abs_bsr >= top_bsr_cutoff
            top_bsr_pattern = np.where(top_bsr_mask, display_pattern, 0.0)
        else:
            top_bsr_cutoff = None
            top_bsr_mask = None
            top_bsr_pattern = None
    else:
        bsr_map = None
        bsr_mask = None
        bsr_thresholded_pattern = None
        top_bsr_cutoff = None
        top_bsr_mask = None
        top_bsr_pattern = None

    # --------------------------------------------------
    # 5. FDR-based bootstrap inference
    # --------------------------------------------------
    if compute_fdr:
        if fdr_method == "ttest":
            fdr_t_values, fdr_p_unc = ttest_1samp(
                A_boot,
                popmean=0.0,
                axis=0,
                nan_policy="raise",
            )

            fdr_t_values = np.asarray(fdr_t_values, dtype=float)
            fdr_p_unc = np.asarray(fdr_p_unc, dtype=float)

        else:
            # Two-sided bootstrap sign-stability p-value
            n_pos = np.sum(A_boot >= 0, axis=0)
            n_neg = np.sum(A_boot <= 0, axis=0)

            fdr_p_unc = 2.0 * (np.minimum(n_pos, n_neg) + 1.0) / (n_bootstraps + 1.0)
            fdr_p_unc = np.clip(fdr_p_unc, 0.0, 1.0)
            fdr_t_values = None

        fdr_mask, fdr_p_fdr = fdr_bh(fdr_p_unc, q=fdr_q)

        if unc_p_threshold is not None:
            fdr_final_mask = fdr_mask & (fdr_p_unc < unc_p_threshold)
        else:
            fdr_final_mask = fdr_mask

        fdr_thresholded_pattern = np.where(fdr_final_mask, display_pattern, 0.0)

    else:
        fdr_t_values = None
        fdr_p_unc = None
        fdr_p_fdr = None
        fdr_mask = None
        fdr_final_mask = None
        fdr_thresholded_pattern = None

    # --------------------------------------------------
    # 6. Feature-level table
    # --------------------------------------------------
    rows = []

    for latent_idx in range(n_latent):
        for feat_idx, feat_name in enumerate(feature_names):
            row = {
                "feature": feat_name,
                "latent_dim": latent_idx,

                "final_encoding_weight": A_final[feat_idx, latent_idx],
                "mean_boot_encoding_weight": A_boot_mean[feat_idx, latent_idx],
                "std_boot_encoding_weight": A_boot_std[feat_idx, latent_idx],
            }

            # BSR columns
            if compute_bsr:
                row.update({
                    "bsr": bsr_map[feat_idx, latent_idx],
                    "abs_bsr": np.abs(bsr_map[feat_idx, latent_idx]),
                    "bsr_threshold": bsr_threshold,
                    "bsr_selected": (
                        bool(bsr_mask[feat_idx, latent_idx])
                        if bsr_mask is not None else False
                    ),
                    "bsr_thresholded_weight": (
                        bsr_thresholded_pattern[feat_idx, latent_idx]
                        if bsr_thresholded_pattern is not None else np.nan
                    ),
                    "top_bsr_percent": top_bsr_percent,
                    "top_bsr_cutoff": top_bsr_cutoff,
                    "top_bsr_selected": (
                        bool(top_bsr_mask[feat_idx, latent_idx])
                        if top_bsr_mask is not None else False
                    ),
                    "top_bsr_weight": (
                        top_bsr_pattern[feat_idx, latent_idx]
                        if top_bsr_pattern is not None else np.nan
                    ),
                })

                if bsr_thresholded_pattern is not None:
                    val_bsr = bsr_thresholded_pattern[feat_idx, latent_idx]
                    row["bsr_sign"] = (
                        "positive" if val_bsr > 0
                        else "negative" if val_bsr < 0
                        else "zero"
                    )
                else:
                    row["bsr_sign"] = "not_thresholded"

            # FDR columns
            if compute_fdr:
                row.update({
                    "fdr_method": fdr_method,
                    "fdr_q": fdr_q,
                    "fdr_t_value": (
                        fdr_t_values[feat_idx, latent_idx]
                        if fdr_t_values is not None else np.nan
                    ),
                    "fdr_p_unc": fdr_p_unc[feat_idx, latent_idx],
                    "fdr_p_fdr": fdr_p_fdr[feat_idx, latent_idx],
                    "fdr_selected": bool(fdr_mask[feat_idx, latent_idx]),
                    "fdr_final_selected": bool(fdr_final_mask[feat_idx, latent_idx]),
                    "unc_p_threshold": unc_p_threshold,
                    "fdr_thresholded_weight": fdr_thresholded_pattern[feat_idx, latent_idx],
                })

                val_fdr = fdr_thresholded_pattern[feat_idx, latent_idx]
                row["fdr_sign"] = (
                    "positive" if val_fdr > 0
                    else "negative" if val_fdr < 0
                    else "zero"
                )

            rows.append(row)

    feature_table = pd.DataFrame(rows)

    return {
        "final_encoding_pattern": A_final,
        "mean_boot_encoding_pattern": A_boot_mean,
        "std_boot_encoding_pattern": A_boot_std,

        "bsr_map": bsr_map,
        "bsr_mask": bsr_mask,
        "bsr_thresholded_pattern": bsr_thresholded_pattern,
        "top_bsr_mask": top_bsr_mask,
        "top_bsr_pattern": top_bsr_pattern,

        "fdr_t_values": fdr_t_values,
        "fdr_p_unc": fdr_p_unc,
        "fdr_p_fdr": fdr_p_fdr,
        "fdr_mask": fdr_mask,
        "fdr_final_mask": fdr_final_mask,
        "fdr_thresholded_pattern": fdr_thresholded_pattern,

        "encoding_patterns_boot": A_boot if store_bootstrap_patterns else None,
        "classes": classes,
        "feature_table": feature_table,
        "settings": {
            "n_bootstraps": n_bootstraps,
            "bootstrap_unit": bootstrap_unit,
            "positive_label": positive_label,
            "random_state": random_state,
            "best_params": best_params,

            "compute_bsr": compute_bsr,
            "bsr_threshold": bsr_threshold,
            "top_bsr_percent": top_bsr_percent,
            "top_bsr_cutoff": top_bsr_cutoff,

            "compute_fdr": compute_fdr,
            "fdr_q": fdr_q,
            "fdr_method": fdr_method,
            "unc_p_threshold": unc_p_threshold,

            "use_pattern_for_display": use_pattern_for_display,
            "store_bootstrap_patterns": store_bootstrap_patterns,
        },
    }


def compute_haufe_map_from_fitted_estimator(
    fitted_estimator,
    X,
    positive_label=None,
):
    """
    Compute Haufe reconstructed activation map from an already fitted estimator.

    Parameters
    ----------
    fitted_estimator : fitted sklearn estimator or Pipeline
        A fitted linear estimator or Pipeline.
    X : ndarray, shape (n_samples, n_features)
        Input feature matrix.
    positive_label : scalar or None
        Positive class direction for binary classifiers.

    Returns
    -------
    result : dict
        {
            "W": decoding weight matrix,
            "A": Haufe reconstructed activation pattern,
            "classes": class labels
        }
    """
    X = np.asarray(X, dtype=float)

    W, classes = _extract_W_from_fitted_linear_estimator(
        fitted_estimator=fitted_estimator,
        positive_label=positive_label,
    )

    X_t = _transform_features_before_final_estimator(fitted_estimator, X)

    A = haufe_transform_general(
        X_train_transformed=X_t,
        W=W,
    )

    if A.ndim == 1:
        A = A.reshape(-1, 1)

    return {
        "W": W,
        "A": A,
        "classes": classes,
    }


def compute_permutation_haufe_reconstructed_activation_map(
    X,
    y,
    subject_ids,
    model_spec=None,
    best_params=None,
    n_permutations=5000,
    q=0.05,
    positive_label=None,
    random_state=42,
    feature_names=None,
    store_null_maps=False,
    verbose=True,
    permute_refit_mode="fixed_params",
    model_specs=None,
    model_name=None,
    site_ids=None,
    default_scoring="roc_auc",
    n_jobs=-1,
    force_logo_inner=True,
    max_inner_site_splits=5,
):
    """
    Compute significance of Haufe reconstructed activation map using permutation.

    Two permutation-refit modes are supported:

    1. permute_refit_mode="fixed_params"
        - Use fixed best_params.
        - In each permutation, only refit estimator on X and y_null.
        - Faster.
        - Recommended default for computational efficiency.

    2. permute_refit_mode="retune_final"
        - In each permutation, call fit_final_multisite_models(...)
          with y_null.
        - This reselects hyperparameters under the permuted labels.
        - More computationally expensive.
        - More strictly matches a full final-model-building null.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Parcel-wise beta maps.
    y : ndarray, shape (n_samples,)
        Binary condition labels.
    subject_ids : ndarray, shape (n_samples,)
        Subject IDs. Each subject must appear exactly twice.
    model_spec : dict or None
        Required for permute_refit_mode="fixed_params".
        One model specification from model_specs, e.g. model_specs["linear_svc"].
    best_params : dict or None
        Fixed final hyperparameters for fixed_params mode.
    n_permutations : int
        Number of subject-level label-swap permutations.
    q : float
        FDR threshold.
    positive_label : scalar or None
        Positive class direction for binary classifiers.
    random_state : int
    feature_names : list-like or None
        Optional feature names.
    store_null_maps : bool
        If True, store all null Haufe maps.
    verbose : bool
    permute_refit_mode : {"fixed_params", "retune_final"}
        Whether to use fixed final parameters or retune final model in each permutation.
    model_specs : dict or None
        Required for permute_refit_mode="retune_final".
        Same format as used by fit_final_multisite_models(...).
    model_name : str or None
        Required for permute_refit_mode="retune_final".
    site_ids : ndarray or None
        Required for permute_refit_mode="retune_final".
    default_scoring : str
        Passed to fit_final_multisite_models in retune_final mode.
    n_jobs : int
        Passed to fit_final_multisite_models in retune_final mode.
    force_logo_inner : bool
        Passed to fit_final_multisite_models in retune_final mode.
    max_inner_site_splits : int
        Passed to fit_final_multisite_models in retune_final mode.

    Returns
    -------
    result : dict
        {
            "observed_A": ndarray,
            "p_unc": ndarray,
            "p_fdr": ndarray,
            "fdr_mask": ndarray,
            "thresholded_A": ndarray,
            "null_abs_ge_count": ndarray,
            "null_A": ndarray or None,
            "classes": ndarray or None,
            "feature_table": DataFrame,
            "permutation_best_params": list or None,
            "settings": dict
        }
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    subject_ids = np.asarray(subject_ids)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")

    if len(y) != X.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    if len(subject_ids) != X.shape[0]:
        raise ValueError("subject_ids must have the same number of samples as X.")

    if len(np.unique(y)) != 2:
        raise ValueError(
            "This subject-level label-swap permutation function is designed "
            "for binary paired classification."
        )

    if permute_refit_mode not in ["fixed_params", "retune_final"]:
        raise ValueError(
            "permute_refit_mode must be one of {'fixed_params', 'retune_final'}."
        )

    if permute_refit_mode == "fixed_params":
        if model_spec is None:
            raise ValueError("model_spec must be provided when permute_refit_mode='fixed_params'.")
        if "estimator" not in model_spec:
            raise KeyError("model_spec must contain key 'estimator'.")

    if permute_refit_mode == "retune_final":
        if model_specs is None:
            raise ValueError("model_specs must be provided when permute_refit_mode='retune_final'.")
        if model_name is None:
            raise ValueError("model_name must be provided when permute_refit_mode='retune_final'.")
        if site_ids is None:
            raise ValueError("site_ids must be provided when permute_refit_mode='retune_final'.")

        site_ids = np.asarray(site_ids)
        if len(site_ids) != X.shape[0]:
            raise ValueError("site_ids must have the same number of samples as X.")

        if model_name not in model_specs:
            raise KeyError(f"model_name='{model_name}' not found in model_specs.")

    n_samples, n_features = X.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    if len(feature_names) != n_features:
        raise ValueError("feature_names must have length equal to n_features.")

    rng = np.random.RandomState(random_state)

    # --------------------------------------------------------
    # 1. Observed reconstructed activation map
    # --------------------------------------------------------
    if permute_refit_mode == "fixed_params":
        obs = fit_and_compute_haufe_map(
            X=X,
            y=y,
            estimator=model_spec["estimator"],
            best_params=best_params,
            positive_label=positive_label,
        )

        observed_best_params = best_params

    else:
        observed_final = fit_final_multisite_models(
            X=X,
            y=y,
            subject_ids=subject_ids,
            site_ids=site_ids,
            model_specs={model_name: model_specs[model_name]},
            default_scoring=default_scoring,
            random_state=random_state,
            n_jobs=n_jobs,
            force_logo_inner=force_logo_inner,
            max_inner_site_splits=max_inner_site_splits,
        )

        observed_best_estimator = observed_final[model_name]["best_estimator"]
        observed_best_params = observed_final[model_name]["best_params"]

        obs = compute_haufe_map_from_fitted_estimator(
            fitted_estimator=observed_best_estimator,
            X=X,
            positive_label=positive_label,
        )

    A_obs = np.asarray(obs["A"], dtype=float)

    if A_obs.ndim == 1:
        A_obs = A_obs.reshape(-1, 1)

    n_features_A, n_latent = A_obs.shape

    if n_features_A != n_features:
        raise ValueError(
            "Observed Haufe map feature dimension does not match X. "
            "This may happen if the pipeline contains feature-changing steps "
            "such as PCA or feature selection."
        )

    abs_A_obs = np.abs(A_obs)

    null_abs_ge_count = np.zeros_like(A_obs, dtype=np.int64)

    if store_null_maps:
        null_A = np.empty(
            (n_permutations, n_features, n_latent),
            dtype=float,
        )
    else:
        null_A = None

    permutation_best_params = [] if permute_refit_mode == "retune_final" else None

    # --------------------------------------------------------
    # 2. Permutation loop
    # --------------------------------------------------------
    for k in tqdm(
            range(n_permutations),
            desc='\t- Permutation analysis for significant encoding activation map'
    ):
        y_null = permute_labels_within_subjects(
            y=y,
            subject_ids=subject_ids,
            random_state=rng.randint(0, 2**31 - 1),
        )

        if permute_refit_mode == "fixed_params":
            null_res = fit_and_compute_haufe_map(
                X=X,
                y=y_null,
                estimator=model_spec["estimator"],
                best_params=best_params,
                positive_label=positive_label,
            )

            A_null = np.asarray(null_res["A"], dtype=float)

        else:
            null_final = fit_final_multisite_models(
                X=X,
                y=y_null,
                subject_ids=subject_ids,
                site_ids=site_ids,
                model_specs={model_name: model_specs[model_name]},
                default_scoring=default_scoring,
                random_state=random_state,
                n_jobs=n_jobs,
                force_logo_inner=force_logo_inner,
                max_inner_site_splits=max_inner_site_splits,
            )

            null_best_estimator = null_final[model_name]["best_estimator"]
            null_best_params = null_final[model_name]["best_params"]
            permutation_best_params.append(null_best_params)

            null_res = compute_haufe_map_from_fitted_estimator(
                fitted_estimator=null_best_estimator,
                X=X,
                positive_label=positive_label,
            )

            A_null = np.asarray(null_res["A"], dtype=float)

        if A_null.ndim == 1:
            A_null = A_null.reshape(-1, 1)

        if A_null.shape != A_obs.shape:
            raise RuntimeError(
                f"Null Haufe map shape mismatch at permutation {k}: "
                f"expected {A_obs.shape}, got {A_null.shape}."
            )

        null_abs_ge_count += (np.abs(A_null) >= abs_A_obs)

        if store_null_maps:
            null_A[k] = A_null

        if verbose and (k + 1) % max(1, n_permutations // 10) == 0:
            print(f"Permutation {k + 1}/{n_permutations} completed.")

    # --------------------------------------------------------
    # 3. Empirical two-sided p-values with +1 correction
    # --------------------------------------------------------
    p_unc = (null_abs_ge_count + 1.0) / (n_permutations + 1.0)

    fdr_mask, p_fdr = fdr_bh(p_unc, q=q)

    thresholded_A = np.where(fdr_mask, A_obs, 0.0)

    # --------------------------------------------------------
    # 4. Feature table
    # --------------------------------------------------------
    rows = []

    for latent_idx in range(n_latent):
        for feat_idx, feat_name in enumerate(feature_names):
            val = thresholded_A[feat_idx, latent_idx]

            rows.append({
                "feature": feat_name,
                "latent_dim": latent_idx,
                "observed_A": A_obs[feat_idx, latent_idx],
                "abs_observed_A": abs_A_obs[feat_idx, latent_idx],
                "null_abs_ge_count": int(null_abs_ge_count[feat_idx, latent_idx]),
                "p_unc": p_unc[feat_idx, latent_idx],
                "p_fdr": p_fdr[feat_idx, latent_idx],
                "fdr_significant": bool(fdr_mask[feat_idx, latent_idx]),
                "thresholded_A": val,
                "sign": (
                    "positive" if val > 0
                    else "negative" if val < 0
                    else "zero"
                ),
            })

    feature_table = pd.DataFrame(rows)

    return {
        "observed_A": A_obs,
        "p_unc": p_unc,
        "p_fdr": p_fdr,
        "fdr_mask": fdr_mask,
        "thresholded_A": thresholded_A,
        "null_abs_ge_count": null_abs_ge_count,
        "null_A": null_A,
        "classes": obs["classes"],
        "feature_table": feature_table,
        "permutation_best_params": permutation_best_params,
        "observed_best_params": observed_best_params,
        "settings": {
            "n_permutations": n_permutations,
            "q": q,
            "positive_label": positive_label,
            "random_state": random_state,
            "store_null_maps": store_null_maps,
            "permute_refit_mode": permute_refit_mode,
            "default_scoring": default_scoring,
            "force_logo_inner": force_logo_inner,
            "max_inner_site_splits": max_inner_site_splits,
        },
    }

if __name__ == "__main__":
    #%% DATA PREPARATION
    #
    rng = np.random.default_rng(42)

    # ---- Data files
    cond_ls  = ['rpsl', 'lkng', 'lknt']
    dir_data = '/public/home/qinshaozheng/fmri_task/ml_signature/roiData'
    dir_save = '/public/home/qinshaozheng/fmri_task/ml_signature'
    fpath_parc = [os.path.join(dir_data, f'parc-cortex_subcortex_cond-{cond}.csv') for cond in cond_ls]
    fpath_subs = os.path.join(dir_data, 'subs_info.csv')

    subs_info = pd.read_csv(fpath_subs, sep=',')

    dict_parc = {}
    for cond, fpath in zip(cond_ls, fpath_parc):
        dict_parc[cond] = pd.read_csv(fpath, sep=',', index_col=False, header=None)

    n_subs = len(subs_info.sub_id)
    n_parc = 232
    n_site = 5
    target_site = [1, 2, 3, 4, 5]

    # two samples per subject: condition 1 vs condition 0
    X_C1 = np.asarray(dict_parc['rpsl'].iloc[:, 1:233])
    X_C2 = np.asarray(dict_parc['lkng'].iloc[:, 1:233])
    X_C3 = np.asarray(dict_parc['lknt'].iloc[:, 1:233])

    X = np.vstack([X_C1, X_C2])
    y = np.array([0]*n_subs + [1]*n_subs, dtype=int)    # 0 for rpsl/lknt, 1 for lkng

    # subject ids repeated twice
    sub_ids = np.arange(n_subs)
    groups  = np.vstack([sub_ids.reshape(-1, 1), 
                         sub_ids.reshape(-1, 1)]).reshape(-1)

    # sites 
    site_per_subj = np.asarray(subs_info['site'])
    site = np.vstack([site_per_subj.reshape(-1, 1), 
                      site_per_subj.reshape(-1, 1)])

    # sex per sample (binary)
    sex_per_subj = np.asarray(subs_info['gender'])
    sex = np.vstack([sex_per_subj.reshape(-1, 1), 
                     sex_per_subj.reshape(-1, 1)])

    # age for later analysis
    age_per_subj = np.asarray(subs_info['age_z'])
    age = np.vstack([age_per_subj.reshape(-1, 1), 
                     age_per_subj.reshape(-1, 1)])

    # ---- Build an "X_full" where last two columns are sex and site
    # ---- `ColumnTransformer` will know which columns to scale/encode
    X_full = np.c_[X, sex.reshape(-1), site.reshape(-1)]

    parcel_cols = list(range(n_parc))            # indices 0..n_parcels-1
    sex_col     = [n_parc]                       # index of sex column
    site_col    = [n_parc + 1]                   # index of site column

    #%% MAIN ANALYSIS
    #
    target_site = [1, 2, 3, 4, 5]
    idx_valid     = np.isin(site, target_site).any(axis=1)
    X_valid       = X[idx_valid]
    y_valid       = y[idx_valid]
    subIDs_valid  = groups[idx_valid]
    siteIDs_valid = site[idx_valid].reshape(-1)

    #
    model_specs = make_default_model_specs(['logistic_l2', 'linear_svc', 'rbf_svc'])

    results = nested_leave_one_site_out_compare_models(
        X=X_valid,
        y=y_valid,
        subject_ids=subIDs_valid,
        site_ids=siteIDs_valid,
        model_specs=model_specs,
        n_jobs=12,
        default_scoring="roc_auc",
        max_inner_site_splits=len(target_site),
    )

    print(results["summary_table"])

    #%% PERMUTATION TEST FOR SIGNIFICANCE
    #
    model_specs = make_default_model_specs(['logistic_l2', 'linear_svc', 'rbf_svc'])

    perm = multisite_permutation_test(
        X=X_valid,
        y=y_valid,
        subject_ids=subIDs_valid,
        site_ids=siteIDs_valid,
        model_specs=model_specs,
        n_permutations=1000,
        metric_for_pvalue="roc_auc",
        default_scoring="roc_auc",
        random_state=42,
        n_jobs=12,
        max_inner_site_splits=5,
        observed_results=None,
        return_full_observed_results=True,
    )

    print("Observed score:", perm["observed_score"])
    print("Permutation p-value:", perm["pvalue"])

    #%% BOOTSTRAPPED DECODING WEIGHTS MAP
    #
    # refit a final model to get best model params
    final_models = fit_final_multisite_models(
        X=X_valid,
        y=y_valid,
        subject_ids=subIDs_valid,
        site_ids=siteIDs_valid,
        model_specs=model_specs,
        default_scoring="roc_auc",
        random_state=42,
        n_jobs=10,
        force_logo_inner=True,
    )

    # bootstrapped analysis for significant decoding weights map
    boot_dec_map = compute_bootstrapped_decoding_map(
        X=X_valid,
        y=y_valid,
        model_spec=model_specs["logistic_l2"],
        best_params=final_models["logistic_l2"]["best_params"],
        subject_ids=subIDs_valid,
        n_bootstraps=5000,
        q=0.05,
        unc_p_threshold=None,
        bootstrap_unit="subject",
        positive_label=1,
        weight_space="transformed",
        map_estimate="full",
        random_state=42,
        feature_names=None,
        verbose=True,
    )

    #%% RECONSTRUCTED ENCODING ACTIVATION MAP(Haufe transformation) via BOOTSTRAPPING
    #
    enc_act_map_boot = compute_reconstructed_activation_pattern(
        X=X_valid,
        y=y_valid,
        model_spec=model_specs["logistic_l2"],
        best_params=final_models["logistic_l2"]["best_params"],
        subject_ids=subIDs_valid,
        n_bootstraps=5000,
        bootstrap_unit="subject",
        positive_label=1,
        random_state=42,
        feature_names=None,
        compute_bsr=True,
        bsr_threshold=3.0,
        top_bsr_percent=10,
        compute_fdr=True,
        fdr_q=0.05,
        fdr_method="sign",         # or "sign"
        unc_p_threshold=0.005,     # e.g. 0.005 if you want stricter FDR + unc. p
        use_pattern_for_display="final",
        store_bootstrap_patterns=True,
        verbose=True,
    )

    #%% RECONSTRUCTED ENCODING ACTIVATION MAP(Haufe transformation) via PERMUTATION
    #
    enc_act_map_perm = compute_permutation_haufe_reconstructed_activation_map(
        X=X_valid,
        y=y_valid,
        subject_ids=subIDs_valid,
        model_spec=model_specs["logistic_l2"],
        best_params=final_models["logistic_l2"]["best_params"],
        n_permutations=1000,
        q=0.05,
        positive_label=1,
        random_state=42,
        feature_names=None,
        store_null_maps=False,
        verbose=True,
    )

    #
    enc_act_map_perm = compute_permutation_haufe_reconstructed_activation_map(
        X=X_valid,
        y=y_valid,
        subject_ids=subIDs_valid,
        site_ids=siteIDs_valid,
        model_specs=model_specs,
        model_name="logistic_l2",
        n_permutations=1000,
        q=0.05,
        positive_label=1,
        random_state=42,
        feature_names=None,
        store_null_maps=False,
        verbose=True,
        permute_refit_mode="retune_final",
        default_scoring="roc_auc",
        n_jobs=10,
        force_logo_inner=True,
        max_inner_site_splits=5,
    )

    #%% SAVE RESULTS
    # 
    dir_save = '/public/home/qinshaozheng/fmri_task/ml_signature/res'

    # res of model training and testing
    fpath_res = os.path.join(dir_save, 'ml_res-model-multiple_task-classify_rpsl_lkng.pkl')

    with open(fpath_res, 'wb') as f:
        pickle.dump(perm, f)
    
    # res of bootstrapped decoding weights and encoding activation (Haufe transformation)
    fpath_res_dec = os.path.join(dir_save, 'mlRes-decodingWeight_task-classify_rpsl_lkng.json')
    fpath_res_enc = os.path.join(dir_save, 'mlRes-encodingActivation_task-classify_rpsl_lkng.json')
    fpath_res_dec_sumTab = os.path.join(dir_save, 'mlRes-decodingWeight_desc-sumTab_task-classify_rpsl_lkng.csv')
    fpath_res_enc_sumTab = os.path.join(dir_save, 'mlRes-encodingActivation_desc-sumTab_task-classify_rpsl_lkng.csv')

    def numpy_default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # save summary table of decoding weights and encoding activation
    save_dict_boot_dec = boot_dec_map.copy()
    save_dict_boot_enc = enc_act_map_boot.copy()
    sumTab_dec = save_dict_boot_dec.pop('feature_table')
    sumTab_enc = save_dict_boot_enc.pop('feature_table')

    for fpath, res in zip(
            [fpath_res_dec_sumTab, fpath_res_enc_sumTab],
            [sumTab_dec, sumTab_enc]
            ):
    
        res.to_csv(fpath, index=False)

    # save all results of bootstrapping analyses
    for fpath, res in zip(
            [fpath_res_dec, fpath_res_enc],
            [save_dict_boot_dec, save_dict_boot_enc],
            ):
    
        with open(fpath, 'w', encoding='utf-8') as file:
            json.dump(res, file, indent=4, default=numpy_default)


    #%%
    with open(fpath_res, 'rb') as r:
        res_ml_rpsl_lkng = pickle.load(r)
    
    #%% -----