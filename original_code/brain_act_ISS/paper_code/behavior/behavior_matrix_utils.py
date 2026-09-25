#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np


def save_npz_named(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(path), mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, arr in arrays.items():
            key = str(name).replace("/", "_").replace("\\", "_")
            bio = io.BytesIO()
            np.save(bio, np.asarray(arr), allow_pickle=False)
            zf.writestr(f"{key}.npy", bio.getvalue())


def pairwise_euclidean_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n_rows, n_feat = X.shape
    out = np.full((n_rows, n_rows), np.nan, dtype=np.float64)
    for i in range(n_rows):
        xi = X[i]
        for j in range(i, n_rows):
            xj = X[j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            n_valid = int(mask.sum())
            if n_valid == 0:
                continue
            diff = xi[mask] - xj[mask]
            scale = float(n_feat) / float(n_valid) if n_feat > 0 else 1.0
            val = np.sqrt(np.sum(diff * diff) * scale, dtype=np.float64)
            out[i, j] = val
            out[j, i] = val
    return out


def pairwise_cityblock_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n_rows, n_feat = X.shape
    out = np.full((n_rows, n_rows), np.nan, dtype=np.float64)
    for i in range(n_rows):
        xi = X[i]
        for j in range(i, n_rows):
            xj = X[j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            n_valid = int(mask.sum())
            if n_valid == 0:
                continue
            diff = np.abs(xi[mask] - xj[mask])
            scale = float(n_feat) / float(n_valid) if n_feat > 0 else 1.0
            val = np.sum(diff, dtype=np.float64) * scale
            out[i, j] = val
            out[j, i] = val
    return out


def pairwise_mahalanobis_distances(X: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    if n_features == 0:
        raise ValueError("Mahalanobis distance requires at least one feature")
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X = np.where(np.isfinite(X), X, col_mean.reshape(1, -1))
    cov = np.cov(X, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    cov = cov + ridge * np.eye(n_features, dtype=np.float64)
    inv_cov = np.linalg.pinv(cov)
    diff = X[:, None, :] - X[None, :, :]
    left = np.einsum("...i,ij->...j", diff, inv_cov)
    dist2 = np.einsum("...i,...i->...", left, diff)
    np.maximum(dist2, 0.0, out=dist2)
    return np.sqrt(dist2, dtype=np.float64)


def compute_difference_matrix(X: np.ndarray, method: str) -> np.ndarray:
    m = str(method).strip().lower()
    if m == "euclidean":
        return pairwise_euclidean_distances(X).astype(np.float32)
    if m in {"cityblock", "manhattan"}:
        return pairwise_cityblock_distances(X).astype(np.float32)
    if m == "mahalanobis":
        return pairwise_mahalanobis_distances(X).astype(np.float32)
    raise ValueError(f"Unknown difference method: {method}")
