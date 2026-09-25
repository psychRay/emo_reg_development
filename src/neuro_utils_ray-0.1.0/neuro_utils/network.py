#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:08:10 2026

@author: dingrui
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _load_brain_like_data(x, expected_kind="map"):
    """
    Load brain data from numpy array, nibabel image object, or file path.

    Supported file formats:
    - NIfTI: .nii, .nii.gz
    - GIFTI: .gii, .func.gii, .shape.gii, .label.gii
    - CIFTI: .dscalar.nii, .dtseries.nii, .pscalar.nii, .dlabel.nii, etc.

    Returns
    -------
    data : np.ndarray
        Loaded data.
    """
    if isinstance(x, (list, tuple)):
        loaded = [_load_brain_like_data(item, expected_kind=expected_kind) for item in x]
        loaded = [np.asarray(item) for item in loaded]

        # For a list of files/arrays, concatenate along the voxel dimension
        # when each item is a single brain vector. This is useful for lh/rh GIFTI.
        if all(item.ndim == 1 for item in loaded):
            return np.concatenate([item.ravel() for item in loaded])

        # Otherwise, treat each item as one or more maps and stack maps.
        flattened = []
        for item in loaded:
            flattened.append(_as_map_matrix(item))
        return np.vstack(flattened)

    if isinstance(x, np.ndarray):
        return x

    if isinstance(x, (str, os.PathLike, Path)):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError(
                "nibabel is required to load NIfTI/GIFTI/CIFTI files. "
                "Install it with: pip install nibabel"
            ) from exc

        img = nib.load(str(x))

        if img.__class__.__name__.lower().startswith("gifti"):
            arrays = [np.asarray(darray.data) for darray in img.darrays]
            if len(arrays) == 0:
                raise ValueError(f"No data arrays found in GIFTI file: {x}")
            if len(arrays) == 1:
                return arrays[0].ravel()
            return np.vstack([arr.ravel() for arr in arrays])

        return np.asarray(img.get_fdata())

    if hasattr(x, "get_fdata"):
        return np.asarray(x.get_fdata())

    if hasattr(x, "darrays"):
        arrays = [np.asarray(darray.data) for darray in x.darrays]
        if len(arrays) == 0:
            raise ValueError("No data arrays found in GIFTI image object.")
        if len(arrays) == 1:
            return arrays[0].ravel()
        return np.vstack([arr.ravel() for arr in arrays])

    raise TypeError(
        f"Unsupported {expected_kind} input type: {type(x)}. "
        "Use numpy array, nibabel image object, file path, or a list of these."
    )


def _as_map_matrix(data):
    """
    Convert loaded brain map data to shape (n_maps, n_voxels).

    Rules
    -----
    - 1D: one map -> (1, n_voxels)
    - 2D: assumed (n_maps, n_voxels)
    - 3D: one volumetric map -> (1, n_voxels)
    - 4D: NIfTI-like data, last dimension = n_maps -> (n_maps, n_voxels)
    """
    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        return data.reshape(1, -1)

    if data.ndim == 2:
        return data.reshape(data.shape[0], -1)

    if data.ndim == 3:
        return data.reshape(1, -1)

    if data.ndim == 4:
        n_maps = data.shape[-1]
        return np.moveaxis(data, -1, 0).reshape(n_maps, -1)

    raise ValueError(f"Unsupported brain map data dimension: {data.ndim}")


def _as_label_vector(data):
    """
    Convert atlas/network label data to shape (n_voxels,).

    If multiple label arrays are present, the first one is used.
    """
    data = np.asarray(data)

    if data.ndim == 1:
        return data.ravel()

    if data.ndim == 2:
        if data.shape[0] == 1:
            return data[0].ravel()
        if data.shape[1] == 1:
            return data[:, 0].ravel()
        return data[0].ravel()

    if data.ndim == 3:
        return data.ravel()

    if data.ndim == 4 and data.shape[-1] == 1:
        return data[..., 0].ravel()

    raise ValueError(
        "Atlas should be a 1D, 2D, 3D label image, or a 4D image with one volume."
    )


def _normalize_maps(maps, valid_mask, method=None, eps=1e-12):
    """
    Normalize each map within valid voxels.

    Parameters
    ----------
    maps : np.ndarray, shape (n_maps, n_voxels)
        Input map matrix.

    valid_mask : np.ndarray, shape (n_voxels,)
        Valid voxel mask.

    method : {None, "none", "zscore", "l2", "rank", "minmax"}
        Normalization method.

    Returns
    -------
    normalized : np.ndarray, shape (n_maps, n_voxels)
        Normalized maps.
    """
    if method is None or str(method).lower() == "none":
        return maps.copy()

    method = str(method).lower()
    normalized = maps.copy().astype(float)

    allowed = {"zscore", "l2", "rank", "minmax"}
    if method not in allowed:
        raise ValueError(
            f"Unknown normalization method: {method}. "
            f"Allowed options are None, 'zscore', 'l2', 'rank', and 'minmax'."
        )

    for i in range(normalized.shape[0]):
        vals = normalized[i, valid_mask].astype(float)

        if method == "zscore":
            mu = np.nanmean(vals)
            sd = np.nanstd(vals)
            normalized[i, valid_mask] = (vals - mu) / (sd + eps)

        elif method == "l2":
            norm = np.sqrt(np.nansum(vals ** 2))
            normalized[i, valid_mask] = vals / (norm + eps)

        elif method == "minmax":
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)
            normalized[i, valid_mask] = (vals - vmin) / (vmax - vmin + eps)

        elif method == "rank":
            # Rank-normalize to [0, 1] within each map.
            # This preserves spatial ordering but removes magnitude differences.
            order = np.argsort(vals)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(vals), dtype=float)
            if len(vals) > 1:
                ranks /= len(vals) - 1
            normalized[i, valid_mask] = ranks

    normalized[:, ~valid_mask] = 0.0
    return normalized


def _apply_significance_threshold_and_sign_rule(
    maps,
    valid_mask,
    significance_mask=None,
    threshold=None,
    positive_only=True,
):
    """
    Apply significance mask, threshold, and positive-only rule.
    """
    proc = maps.copy().astype(float)
    n_maps, n_voxels = proc.shape

    if significance_mask is not None:
        raw_sig = _load_brain_like_data(
            significance_mask,
            expected_kind="significance_mask",
        )
        sig = _as_map_matrix(raw_sig).astype(bool)

        if sig.shape[0] == 1 and n_maps > 1:
            sig = np.tile(sig, (n_maps, 1))

        if sig.shape != proc.shape:
            raise ValueError(
                f"significance_mask has shape {sig.shape}, but brain maps have "
                f"shape {proc.shape}. Use either one mask or one mask per map."
            )

        proc[~sig] = 0.0

    proc[:, ~valid_mask] = 0.0

    if threshold is None:
        threshold = 0.0

    if positive_only:
        proc[proc <= threshold] = 0.0
    else:
        proc[np.abs(proc) <= threshold] = 0.0

    return proc


def compute_network_contribution(
    brain_maps,
    atlas,
    map_names=None,
    network_names=None,
    brain_mask=None,
    significance_mask=None,
    threshold=None,
    positive_only=True,
    map_normalization=None,
    cal_relative_contribution=False,
    relative_normalization="l2",
    eps=1e-12,
):
    """
    Compute network contribution metrics for one or more brain maps.

    Key design
    ----------
    network_metrics:
        Uses the processed map controlled by map_normalization.
        By default, map_normalization=None, so single-map network metrics are
        computed on the original thresholded map.

    relative_contribution:
        Uses a separately processed map controlled by relative_normalization.
        By default, relative_normalization="l2", so winner-based voxel comparison
        is based on within-map L2-normalized maps.

    Parameters
    ----------
    brain_maps : array-like, nibabel image object, path, or list
        Brain map(s), typically reconstructed activation / encoding maps.
        Supported file inputs include NIfTI, GIFTI, and CIFTI.

    atlas : array-like, nibabel image object, or path
        Network / ROI label image in the same voxel/vertex/grayordinate space
        as brain_maps. Background should be 0 or NaN.

    map_names : list of str, optional
        Names of maps. If None, names are generated automatically.

    network_names : dict, optional
        Mapping from atlas integer labels to network names.

    brain_mask : array-like, nibabel image object, or path, optional
        Valid voxel/vertex/grayordinate mask.

    significance_mask : array-like, nibabel image object, path, or list, optional
        Boolean mask for significant voxels, e.g., FDR q < 0.05.
        Can be shape (n_voxels,) or (n_maps, n_voxels).

    threshold : float, optional
        Additional threshold after applying significance_mask.
        If positive_only=True, voxels <= threshold are removed.
        If positive_only=False, voxels with abs(value) <= threshold are removed.

    positive_only : bool, default=True
        If True, only positive voxels are retained.

    map_normalization : {None, "none", "zscore", "l2", "rank", "minmax"}, default=None
        Normalization applied before computing network_metrics.
        For Jiang-style single-map distribution, keep this as None.

    relative_normalization : {None, "none", "zscore", "l2", "rank", "minmax"}, default="l2"
        Normalization applied only before computing relative_contribution.
        Recommended values:
        - "l2": compare spatial patterns after equalizing map vector length.
        - "zscore": compare standardized voxel-wise spatial patterns.
        - "rank": compare relative spatial ordering, not raw magnitude.
        - None or "none": use raw processed maps; only recommended when maps
          are known to be directly comparable.

    eps : float, default=1e-12
        Small value to avoid division by zero.

    Returns
    -------
    results : dict
        results["network_metrics"] : pandas.DataFrame
            Single-map network distribution metrics.

        results["relative_contribution"] : pandas.DataFrame
            Winner-based relative contribution across maps.

        results["processed_maps"] : np.ndarray
            Maps used for network_metrics.

        results["relative_processed_maps"] : np.ndarray
            Maps used for relative_contribution.

        results["atlas_vector"] : np.ndarray
            Flattened atlas vector.
    """

    # ---------- Load data ----------
    raw_maps = _load_brain_like_data(brain_maps, expected_kind="brain_maps")
    maps = _as_map_matrix(raw_maps)

    n_maps, n_voxels = maps.shape

    raw_atlas = _load_brain_like_data(atlas, expected_kind="atlas")
    atlas_vec = _as_label_vector(raw_atlas)

    if atlas_vec.shape[0] != n_voxels:
        raise ValueError(
            f"Atlas has {atlas_vec.shape[0]} elements, but brain_maps has "
            f"{n_voxels} voxels/vertices/grayordinates. They must be in the same space."
        )

    if map_names is None:
        map_names = [f"map_{i + 1}" for i in range(n_maps)]

    if len(map_names) != n_maps:
        raise ValueError(
            f"map_names length is {len(map_names)}, but number of maps is {n_maps}."
        )

    if network_names is None:
        network_names = {}

    # ---------- Valid voxel mask ----------
    finite_mask = np.isfinite(atlas_vec)
    finite_mask &= np.all(np.isfinite(maps), axis=0)

    atlas_positive_mask = atlas_vec > 0

    if brain_mask is not None:
        raw_brain_mask = _load_brain_like_data(brain_mask, expected_kind="brain_mask")
        brain_mask_vec = _as_label_vector(raw_brain_mask).astype(bool)

        if brain_mask_vec.shape[0] != n_voxels:
            raise ValueError("brain_mask must match the number of brain map elements.")

        valid_mask = finite_mask & atlas_positive_mask & brain_mask_vec
    else:
        valid_mask = finite_mask & atlas_positive_mask

    if valid_mask.sum() == 0:
        raise ValueError("No valid voxels remain after applying atlas and brain_mask.")

    labels = np.array(sorted(np.unique(atlas_vec[valid_mask]).astype(int)))

    # ---------- Process maps for network_metrics ----------
    metric_maps = _normalize_maps(
        maps,
        valid_mask=valid_mask,
        method=map_normalization,
        eps=eps,
    )

    metric_proc = _apply_significance_threshold_and_sign_rule(
        metric_maps,
        valid_mask=valid_mask,
        significance_mask=significance_mask,
        threshold=threshold,
        positive_only=positive_only,
    )

    # ---------- Process maps for relative_contribution ----------
    relative_maps = _normalize_maps(
        maps,
        valid_mask=valid_mask,
        method=relative_normalization,
        eps=eps,
    )

    relative_proc = _apply_significance_threshold_and_sign_rule(
        relative_maps,
        valid_mask=valid_mask,
        significance_mask=significance_mask,
        threshold=threshold,
        positive_only=positive_only,
    )

    # ---------- Global denominators for network_metrics ----------
    global_abs_sum = np.sum(np.abs(metric_proc), axis=1) + eps
    global_pos_sum = np.sum(np.clip(metric_proc, 0, None), axis=1) + eps
    global_nonzero_count = np.sum(metric_proc != 0, axis=1) + eps
    global_l2 = np.sqrt(np.sum(metric_proc ** 2, axis=1)) + eps

    rows = []

    # ---------- Single-map network metrics ----------
    for m_idx, m_name in enumerate(map_names):
        this_map = metric_proc[m_idx]
        this_binary = this_map != 0

        this_binary_count = np.sum(this_binary)
        this_l2 = np.sqrt(np.sum(this_map ** 2)) + eps

        for label in labels:
            net_mask = valid_mask & (atlas_vec == label)
            net_vals = this_map[net_mask]

            n_net = int(net_mask.sum())
            n_nonzero = int(np.sum(net_vals != 0))
            n_pos = int(np.sum(net_vals > 0))
            n_neg = int(np.sum(net_vals < 0))

            sum_weight = float(np.sum(net_vals))
            sum_abs = float(np.sum(np.abs(net_vals)))
            sum_pos = float(np.sum(net_vals[net_vals > 0]))
            sum_neg = float(np.sum(net_vals[net_vals < 0]))

            mean_weight = float(np.mean(net_vals)) if n_net > 0 else np.nan
            mean_nonzero = (
                float(np.mean(net_vals[net_vals != 0]))
                if n_nonzero > 0
                else np.nan
            )
            mean_abs = float(np.mean(np.abs(net_vals))) if n_net > 0 else np.nan
            l2_norm = float(np.sqrt(np.sum(net_vals ** 2)))

            binary_net = net_mask.astype(float)

            cosine_binary = float(
                np.dot(np.abs(this_map), binary_net)
                / (this_l2 * (np.sqrt(np.sum(binary_net ** 2)) + eps))
            )

            dice_binary = float(
                2.0 * n_nonzero / (this_binary_count + n_net + eps)
            )

            rows.append(
                {
                    "map": m_name,
                    "network_label": int(label),
                    "network_name": network_names.get(int(label), str(int(label))),
                    "n_network_voxels": n_net,
                    "n_encoded_voxels": n_nonzero,
                    "n_positive_voxels": n_pos,
                    "n_negative_voxels": n_neg,
                    "voxel_pct_of_network": 100 * n_nonzero / max(n_net, 1),
                    "voxel_pct_of_global_encoded": (
                        100 * n_nonzero / global_nonzero_count[m_idx]
                    ),
                    "positive_voxel_pct_of_network": 100 * n_pos / max(n_net, 1),
                    "negative_voxel_pct_of_network": 100 * n_neg / max(n_net, 1),
                    "sum_weight": sum_weight,
                    "sum_abs_weight": sum_abs,
                    "sum_positive_weight": sum_pos,
                    "sum_negative_weight": sum_neg,
                    "mean_weight": mean_weight,
                    "mean_nonzero_weight": mean_nonzero,
                    "mean_abs_weight": mean_abs,
                    "l2_norm": l2_norm,
                    "cosine_similarity_binary_network": cosine_binary,
                    "dice_coefficient_binary_network": dice_binary,
                    "abs_weight_contribution_pct_global": (
                        100 * sum_abs / global_abs_sum[m_idx]
                    ),
                    "positive_weight_contribution_pct_global": (
                        100 * sum_pos / global_pos_sum[m_idx]
                    ),
                    "l2_contribution_pct_global": (
                        100 * l2_norm / global_l2[m_idx]
                    ),
                    "map_normalization": (
                        "none" if map_normalization is None else map_normalization
                    ),
                }
            )

    network_metrics = pd.DataFrame(rows)

    # ---------- Relative contribution across maps ----------
    relative_rows = []

    if n_maps >= 2 and cal_relative_contribution:
        positive_relative_proc = np.clip(relative_proc, 0, None)

        for label in labels:
            net_mask = valid_mask & (atlas_vec == label)
            net_values = positive_relative_proc[:, net_mask]

            n_net = int(net_mask.sum())
            encoded_any = np.any(net_values > 0, axis=0)
            n_encoded_any = int(np.sum(encoded_any))

            if n_net == 0:
                continue

            # Winner is defined by the largest positive normalized value.
            # Voxels with no positive value in any map are treated as unencoded.
            winners = np.argmax(net_values, axis=0)

            for m_idx, m_name in enumerate(map_names):
                win_mask = encoded_any & (winners == m_idx)
                n_win = int(np.sum(win_mask))

                relative_rows.append(
                    {
                        "map": m_name,
                        "network_label": int(label),
                        "network_name": network_names.get(int(label), str(int(label))),
                        "n_network_voxels": n_net,
                        "n_encoded_by_any_map": n_encoded_any,
                        "n_winner_voxels": n_win,
                        "winner_pct_of_network": 100 * n_win / max(n_net, 1),
                        "winner_pct_of_encoded_voxels": (
                            100 * n_win / n_encoded_any
                            if n_encoded_any > 0
                            else np.nan
                        ),
                        "relative_normalization": (
                            "none"
                            if relative_normalization is None
                            else relative_normalization
                        ),
                    }
                )

    relative_contribution = pd.DataFrame(relative_rows)

    return {
        "network_metrics": network_metrics,
        "relative_contribution": relative_contribution,
        "processed_maps": metric_proc,
        "relative_processed_maps": relative_proc,
        "atlas_vector": atlas_vec,
    }


def summarize_brain_maps_by_network(
    brain_maps,
    atlas,
    map_names=None,
    network_names=None,
    brain_mask=None,
    significance_mask=None,
    threshold=None,
    positive_only=False,
    map_normalization=None,
    summary_methods=None,
    percentiles=(25, 50, 75),
    include_zero=True,
    output_format="long",
    eps=1e-12,
):
    """
    Summarize one or more brain maps within each network/ROI.

    This function accepts brain maps in array, NIfTI, GIFTI, or CIFTI format,
    together with a network atlas in the same space, and computes network-wise
    summary values for each map.

    Parameters
    ----------
    brain_maps : array-like, nibabel image object, path, or list
        Brain map(s). Supported inputs are the same as in compute_network_contribution:
        NIfTI, GIFTI, CIFTI, or numpy arrays.

    atlas : array-like, nibabel image object, or path
        Network / ROI label image in the same voxel/vertex/grayordinate space
        as brain_maps. Background should usually be 0 or NaN.

    map_names : list of str, optional
        Names of maps. If None, names are generated automatically.

    network_names : dict, optional
        Mapping from atlas integer labels to network names.
        Example: {1: "Visual", 2: "Somatomotor", 7: "Default"}

    brain_mask : array-like, nibabel image object, or path, optional
        Valid voxel/vertex/grayordinate mask.

    significance_mask : array-like, nibabel image object, path, or list, optional
        Boolean significance mask, e.g., FDR q < 0.05.
        Can be shape (n_voxels,) or (n_maps, n_voxels).

    threshold : float, optional
        Additional threshold.
        If positive_only=True, voxels <= threshold are set to 0.
        If positive_only=False, voxels with abs(value) <= threshold are set to 0.

    positive_only : bool, default=False
        If True, only positive voxels are retained before summarizing.

    map_normalization : {None, "none", "zscore", "l2", "rank", "minmax"}, default=None
        Optional within-map normalization before thresholding and summarization.

    summary_methods : list of str or None
        Summary metrics to compute. If None, a default set is used.

        Supported methods:
        - "mean"
        - "median"
        - "std"
        - "var"
        - "min"
        - "max"
        - "sum"
        - "abs_mean"
        - "abs_median"
        - "abs_sum"
        - "positive_mean"
        - "positive_sum"
        - "negative_mean"
        - "negative_sum"
        - "nonzero_mean"
        - "nonzero_median"
        - "l2_norm"
        - "rms"
        - "n_voxels"
        - "n_nonzero"
        - "n_positive"
        - "n_negative"
        - "nonzero_pct"
        - "positive_pct"
        - "negative_pct"
        - "percentile"

    percentiles : tuple of int or float, default=(25, 50, 75)
        Percentiles to compute when "percentile" is included in summary_methods.

    include_zero : bool, default=True
        If True, summary statistics such as mean/median/std are computed across
        all voxels in the network, including zeros after thresholding.
        If False, these statistics are computed only across nonzero voxels.
        Count and percentage metrics are always computed relative to all network voxels.

    output_format : {"long", "wide"}, default="long"
        Output format.

        "long":
            One row per map-network pair, with summary metrics as columns.

        "wide":
            One row per map, columns are network_metric combinations.

    eps : float, default=1e-12
        Small value to avoid division by zero.

    Returns
    -------
    results : dict
        results["summary"] : pandas.DataFrame
            Network-wise summary table.

        results["processed_maps"] : np.ndarray
            Processed maps used for summarization, shape (n_maps, n_voxels).

        results["atlas_vector"] : np.ndarray
            Flattened atlas vector.

        results["valid_mask"] : np.ndarray
            Final valid voxel/vertex/grayordinate mask.
    """

    if summary_methods is None:
        summary_methods = [
            "mean",
            "median",
            "std",
            "sum",
            "abs_mean",
            "abs_sum",
            "positive_mean",
            "positive_sum",
            "negative_mean",
            "negative_sum",
            "l2_norm",
            "rms",
            "n_voxels",
            "n_nonzero",
            "n_positive",
            "n_negative",
            "nonzero_pct",
            "positive_pct",
            "negative_pct",
        ]

    supported_methods = {
        "mean",
        "median",
        "std",
        "var",
        "min",
        "max",
        "sum",
        "abs_mean",
        "abs_median",
        "abs_sum",
        "positive_mean",
        "positive_sum",
        "negative_mean",
        "negative_sum",
        "nonzero_mean",
        "nonzero_median",
        "l2_norm",
        "rms",
        "n_voxels",
        "n_nonzero",
        "n_positive",
        "n_negative",
        "nonzero_pct",
        "positive_pct",
        "negative_pct",
        "percentile",
    }

    unknown_methods = set(summary_methods) - supported_methods
    if unknown_methods:
        raise ValueError(
            f"Unsupported summary_methods: {sorted(unknown_methods)}. "
            f"Supported methods are: {sorted(supported_methods)}"
        )

    if output_format not in {"long", "wide"}:
        raise ValueError("output_format must be either 'long' or 'wide'.")

    if network_names is None:
        network_names = {}

    # ---------- Load and format brain maps ----------
    raw_maps = _load_brain_like_data(brain_maps, expected_kind="brain_maps")
    maps = _as_map_matrix(raw_maps)

    n_maps, n_voxels = maps.shape

    if map_names is None:
        map_names = [f"map_{i + 1}" for i in range(n_maps)]

    if len(map_names) != n_maps:
        raise ValueError(
            f"map_names length is {len(map_names)}, but number of maps is {n_maps}."
        )

    # ---------- Load and format atlas ----------
    raw_atlas = _load_brain_like_data(atlas, expected_kind="atlas")
    atlas_vec = _as_label_vector(raw_atlas)

    if atlas_vec.shape[0] != n_voxels:
        raise ValueError(
            f"Atlas has {atlas_vec.shape[0]} elements, but brain_maps has "
            f"{n_voxels} voxels/vertices/grayordinates. They must be in the same space."
        )

    # ---------- Define valid voxels ----------
    finite_mask = np.isfinite(atlas_vec)
    finite_mask &= np.all(np.isfinite(maps), axis=0)

    atlas_positive_mask = atlas_vec > 0

    if brain_mask is not None:
        raw_brain_mask = _load_brain_like_data(brain_mask, expected_kind="brain_mask")
        brain_mask_vec = _as_label_vector(raw_brain_mask).astype(bool)

        if brain_mask_vec.shape[0] != n_voxels:
            raise ValueError(
                "brain_mask must match the number of brain map elements."
            )

        valid_mask = finite_mask & atlas_positive_mask & brain_mask_vec
    else:
        valid_mask = finite_mask & atlas_positive_mask

    if valid_mask.sum() == 0:
        raise ValueError("No valid voxels remain after applying atlas and brain_mask.")

    labels = np.array(sorted(np.unique(atlas_vec[valid_mask]).astype(int)))

    # ---------- Optional normalization ----------
    normalized_maps = _normalize_maps(
        maps,
        valid_mask=valid_mask,
        method=map_normalization,
        eps=eps,
    )

    # ---------- Apply significance mask, threshold, and sign rule ----------
    proc = _apply_significance_threshold_and_sign_rule(
        normalized_maps,
        valid_mask=valid_mask,
        significance_mask=significance_mask,
        threshold=threshold,
        positive_only=positive_only,
    )

    rows = []

    for m_idx, m_name in enumerate(map_names):
        this_map = proc[m_idx]

        for label in labels:
            net_mask = valid_mask & (atlas_vec == label)
            raw_vals = this_map[net_mask]

            n_voxels = int(raw_vals.size)
            n_nonzero = int(np.sum(raw_vals != 0))
            n_positive = int(np.sum(raw_vals > 0))
            n_negative = int(np.sum(raw_vals < 0))

            if include_zero:
                vals = raw_vals
            else:
                vals = raw_vals[raw_vals != 0]

            # Use NaN for summaries that are undefined when no nonzero voxels exist.
            has_vals = vals.size > 0

            positive_vals = raw_vals[raw_vals > 0]
            negative_vals = raw_vals[raw_vals < 0]
            nonzero_vals = raw_vals[raw_vals != 0]

            row = {
                "map": m_name,
                "network_label": int(label),
                "network_name": network_names.get(int(label), str(int(label))),
            }

            if "n_voxels" in summary_methods:
                row["n_voxels"] = n_voxels

            if "n_nonzero" in summary_methods:
                row["n_nonzero"] = n_nonzero

            if "n_positive" in summary_methods:
                row["n_positive"] = n_positive

            if "n_negative" in summary_methods:
                row["n_negative"] = n_negative

            if "nonzero_pct" in summary_methods:
                row["nonzero_pct"] = 100 * n_nonzero / max(n_voxels, 1)

            if "positive_pct" in summary_methods:
                row["positive_pct"] = 100 * n_positive / max(n_voxels, 1)

            if "negative_pct" in summary_methods:
                row["negative_pct"] = 100 * n_negative / max(n_voxels, 1)

            if "mean" in summary_methods:
                row["mean"] = float(np.mean(vals)) if has_vals else np.nan

            if "median" in summary_methods:
                row["median"] = float(np.median(vals)) if has_vals else np.nan

            if "std" in summary_methods:
                row["std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan

            if "var" in summary_methods:
                row["var"] = float(np.var(vals, ddof=1)) if vals.size > 1 else np.nan

            if "min" in summary_methods:
                row["min"] = float(np.min(vals)) if has_vals else np.nan

            if "max" in summary_methods:
                row["max"] = float(np.max(vals)) if has_vals else np.nan

            if "sum" in summary_methods:
                row["sum"] = float(np.sum(vals)) if has_vals else np.nan

            if "abs_mean" in summary_methods:
                row["abs_mean"] = (
                    float(np.mean(np.abs(vals))) if has_vals else np.nan
                )

            if "abs_median" in summary_methods:
                row["abs_median"] = (
                    float(np.median(np.abs(vals))) if has_vals else np.nan
                )

            if "abs_sum" in summary_methods:
                row["abs_sum"] = (
                    float(np.sum(np.abs(vals))) if has_vals else np.nan
                )

            if "positive_mean" in summary_methods:
                row["positive_mean"] = (
                    float(np.mean(positive_vals)) if positive_vals.size > 0 else np.nan
                )

            if "positive_sum" in summary_methods:
                row["positive_sum"] = (
                    float(np.sum(positive_vals)) if positive_vals.size > 0 else 0.0
                )

            if "negative_mean" in summary_methods:
                row["negative_mean"] = (
                    float(np.mean(negative_vals)) if negative_vals.size > 0 else np.nan
                )

            if "negative_sum" in summary_methods:
                row["negative_sum"] = (
                    float(np.sum(negative_vals)) if negative_vals.size > 0 else 0.0
                )

            if "nonzero_mean" in summary_methods:
                row["nonzero_mean"] = (
                    float(np.mean(nonzero_vals)) if nonzero_vals.size > 0 else np.nan
                )

            if "nonzero_median" in summary_methods:
                row["nonzero_median"] = (
                    float(np.median(nonzero_vals)) if nonzero_vals.size > 0 else np.nan
                )

            if "l2_norm" in summary_methods:
                row["l2_norm"] = float(np.sqrt(np.sum(raw_vals ** 2)))

            if "rms" in summary_methods:
                row["rms"] = float(np.sqrt(np.mean(raw_vals ** 2))) if n_voxels > 0 else np.nan

            if "percentile" in summary_methods:
                for p in percentiles:
                    col = f"percentile_{p:g}"
                    row[col] = float(np.percentile(vals, p)) if has_vals else np.nan

            row["include_zero"] = include_zero
            row["map_normalization"] = "none" if map_normalization is None else map_normalization
            row["positive_only"] = positive_only

            rows.append(row)

    summary_df = pd.DataFrame(rows)

    if output_format == "wide":
        id_cols = ["map"]
        value_cols = [
            col for col in summary_df.columns
            if col not in {
                "map",
                "network_label",
                "network_name",
                "include_zero",
                "map_normalization",
                "positive_only",
            }
        ]

        wide_parts = []

        for _, row in summary_df.iterrows():
            prefix = str(row["network_name"])
            map_name = row["map"]

            wide_row = {"map": map_name}
            for col in value_cols:
                wide_row[f"{prefix}__{col}"] = row[col]

            wide_parts.append(wide_row)

        summary_df = (
            pd.DataFrame(wide_parts)
            .groupby("map", as_index=False)
            .first()
        )

    return {
        "summary": summary_df,
        "processed_maps": proc,
        "atlas_vector": atlas_vec,
        "valid_mask": valid_mask,
    }