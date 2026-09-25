#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_analysis_roi_isc_dev_models.py

Step 3:
For each stimulus type, test ROI subject-by-subject similarity against
developmental models. Models: M_nn, M_conv, and M_div. Tests are one-tailed
(positive) with model-wise FWER (max-T) correction.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t

try:
    from perm_null_io import make_metadata, save_perm_null_npz
except ImportError:  # pragma: no cover - package import fallback
    from .perm_null_io import make_metadata, save_perm_null_npz

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
MODEL_NAMES = ("M_nn", "M_conv", "M_div")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test ROI inter-subject similarity against developmental models (FDR only, or permutation FWER and FDR)")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--repr-prefix", type=str, default=None)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1, help="Number of ROI workers for permutation mode.")
    p.add_argument("--isc-method", type=str, default="mahalanobis", choices=("spearman", "pearson", "euclidean", "mahalanobis"))
    p.add_argument("--isc-prefix", type=str, default=None)
    p.add_argument("--assoc-method", type=str, default="spearman", choices=("pearson", "spearman"))
    p.add_argument(
        "--correction-mode",
        type=str,
        default="fdr_only",
        choices=("fdr_only", "perm_fwer_fdr"),
        help="fdr_only (default): parametric p-values and BH-FDR. perm_fwer_fdr: permutation p-values, model-wise max-T FWER, and BH-FDR.",
    )
    p.add_argument("--no-normalize-models", action="store_false", dest="normalize_models", default=True)
    p.add_argument("--no-fisher-z", action="store_false", dest="fisher_z", default=True)
    return p.parse_args()


_ROI_DEV_WORKER_CTX: dict = {}


def _init_roi_dev_worker(ctx: dict) -> None:
    global _ROI_DEV_WORKER_CTX
    _ROI_DEV_WORKER_CTX = ctx


def _chunk_indices(n_items: int, n_jobs: int) -> list[np.ndarray]:
    jobs = max(1, min(int(n_jobs), int(n_items)))
    chunks = [np.asarray(x, dtype=np.int32) for x in np.array_split(np.arange(int(n_items), dtype=np.int32), jobs)]
    return [x for x in chunks if int(x.size) > 0]


def has_repr_files(stim_dir: Path, repr_prefix: str) -> bool:
    required = (
        stim_dir / f"{repr_prefix}.npz",
        stim_dir / f"{repr_prefix}_subjects.csv",
        stim_dir / f"{repr_prefix}_rois.csv",
    )
    return all(p.exists() for p in required)


def detect_isc_prefix(stim_dir: Path) -> List[str]:
    # Auto-detect ISC outputs by filename convention inside one stimulus_type directory.
    prefixes = []
    for npy in sorted(stim_dir.glob("roi_isc_*_by_age.npy")):
        prefix = npy.stem
        if (stim_dir / f"{prefix}_subjects_sorted.csv").exists() and (stim_dir / f"{prefix}_rois.csv").exists():
            prefixes.append(prefix)
    return prefixes


def has_isc_files(stim_dir: Path, isc_prefix: str) -> bool:
    required = (
        stim_dir / f"{isc_prefix}.npy",
        stim_dir / f"{isc_prefix}_subjects_sorted.csv",
        stim_dir / f"{isc_prefix}_rois.csv",
    )
    return all(p.exists() for p in required)


def resolve_isc_prefix(stim_dir: Path, isc_prefix: Optional[str], isc_method: Optional[str]) -> str:
    if isc_prefix is not None:
        return str(isc_prefix)
    found = detect_isc_prefix(stim_dir)
    if isc_method is not None:
        # When isc_method is given, prefer the canonical prefix roi_isc_<method>_by_age.
        m = str(isc_method).strip().lower()
        expected = f"roi_isc_{m}_by_age"
        if expected in found:
            return expected
        raise FileNotFoundError(f"No product found for isc-method: {expected} (available: {found}）")
    if len(found) == 1:
        return found[0]
    if len(found) == 0:
        raise FileNotFoundError(f"{stim_dir} has no roi_isc_*_by_age.npy")
    raise ValueError(f"{stim_dir} contains more than one ISC prefix: {found}; set --isc-prefix or --isc-method")


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    # Benjamini–Hochberg FDR (returns q-values, monotone adjusted).
    p = np.asarray(pvals, dtype=np.float64).reshape(-1)
    q = np.full(p.shape, np.nan, dtype=np.float64)
    m = np.isfinite(p)
    if int(m.sum()) == 0:
        return q
    pv = p[m]
    n = int(pv.size)
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * float(n) / np.arange(1, n + 1, dtype=np.float64)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_valid = np.empty_like(pv)
    q_valid[order] = q_ranked
    q[m] = q_valid
    return q


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.isfinite(x)
    if int(m.sum()) < 10:
        return np.full_like(x, np.nan, dtype=np.float32)
    mu = float(x[m].mean())
    sd = float(x[m].std(ddof=0))
    if sd <= 0:
        return np.full_like(x, np.nan, dtype=np.float32)
    out = (x - mu) / sd
    out[~m] = np.nan
    return out.astype(np.float32)


def fisher_z(r: np.ndarray) -> np.ndarray:
    x = np.asarray(r, dtype=np.float32)
    m = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=np.float32)
    if not bool(m.any()):
        return out
    xc = np.clip(x[m], -0.999999, 0.999999)
    out[m] = np.arctanh(xc).astype(np.float32, copy=False)
    return out


def build_models(ages: np.ndarray, iu: np.ndarray, ju: np.ndarray, normalize: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Build pairwise developmental model vectors aligned with upper-tri indices (iu, ju).
    a = np.asarray(ages, dtype=np.float32).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("ages contain NaN or Inf")
    amax = float(a.max())
    ai = a[iu]
    aj = a[ju]
    nn = (amax - np.abs(ai - aj)).astype(np.float32)
    conv = np.minimum(ai, aj).astype(np.float32)
    div = (amax - 0.5 * (ai + aj)).astype(np.float32)
    if normalize:
        return zscore_1d(nn), zscore_1d(conv), zscore_1d(div)
    return nn, conv, div


def assoc(S_z: np.ndarray, m: np.ndarray, method: str) -> np.ndarray:
    # Compute association between each ROI's ISC vector (rows of S_z) and one model vector m.
    mv = np.asarray(m, dtype=np.float32).reshape(-1)
    mask_m = np.isfinite(mv)
    if int(mask_m.sum()) <= 1:
        return np.full((S_z.shape[0],), np.nan, dtype=np.float32)

    Sz = np.asarray(S_z, dtype=np.float32)
    mth = str(method).strip().lower()
    if mth == "pearson":
        joint = mask_m.reshape(1, -1) & np.isfinite(Sz)
        denom = joint.sum(axis=1).astype(np.float32, copy=False)
        prod = Sz * mv.reshape(1, -1)
        prod[~joint] = 0.0
        denom_safe = denom.copy()
        denom_safe[denom_safe == 0] = 1.0
        out = prod.sum(axis=1) / denom_safe
        out[denom <= 1] = np.nan
        return out.astype(np.float32, copy=False)

    if mth == "spearman":
        # Fast path: if all ROI rows are finite, pre-rank the model once and use matrix ops.
        out = np.full((Sz.shape[0],), np.nan, dtype=np.float32)
        valid_rows = np.isfinite(Sz).all(axis=1)

        if bool(valid_rows.any()):
            mr = np.full_like(mv, np.nan, dtype=np.float32)
            mr[mask_m] = zscore_1d(rankdata(mv[mask_m]).astype(np.float32, copy=False))
            Sz_fast = Sz[valid_rows]
            joint = mask_m.reshape(1, -1) & np.isfinite(Sz_fast)
            denom = joint.sum(axis=1).astype(np.float32, copy=False)
            prod = Sz_fast * mr.reshape(1, -1)
            prod[~joint] = 0.0
            denom_safe = denom.copy()
            denom_safe[denom_safe == 0] = 1.0
            out_fast = prod.sum(axis=1) / denom_safe
            out_fast[denom <= 1] = np.nan
            out[valid_rows] = out_fast.astype(np.float32, copy=False)

        invalid_rows = ~valid_rows
        if bool(invalid_rows.any()):
            idx_invalid = np.where(invalid_rows)[0]
            for i in idx_invalid.tolist():
                x = Sz[i].reshape(-1)
                joint_1d = mask_m & np.isfinite(x)
                if int(joint_1d.sum()) < 3:
                    continue
                xr = rankdata(x[joint_1d]).astype(np.float32, copy=False)
                mr = rankdata(mv[joint_1d]).astype(np.float32, copy=False)
                xz = zscore_1d(xr)
                mz = zscore_1d(mr)
                ok = np.isfinite(xz) & np.isfinite(mz)
                if int(ok.sum()) < 3:
                    continue
                out[i] = float((xz[ok] * mz[ok]).mean())

        return out

    raise ValueError(f"Unknown assoc-method: {method}")


def _permute_roi_dev_chunk(indices: np.ndarray) -> dict:
    ctx = _ROI_DEV_WORKER_CTX
    idx = np.asarray(indices, dtype=np.int32)
    s_chunk = np.asarray(ctx["S_z"][idx], dtype=np.float32)
    obs_chunk = np.asarray(ctx["r_obs_by_model"][:, idx], dtype=np.float32)
    n_perm = int(ctx["n_perm"])
    n_sub = int(ctx["n_sub"])
    n_chunk = int(idx.size)
    c_raw = np.zeros((len(MODEL_NAMES), n_chunk), dtype=np.int64)
    perm_r = np.full((len(MODEL_NAMES), n_chunk, n_perm), np.nan, dtype=np.float32)
    max_stats = np.full((len(MODEL_NAMES), n_perm), -np.inf, dtype=np.float32)
    rng = np.random.default_rng(int(ctx["seed"]))

    for pi in range(n_perm):
        ages_p = ctx["ages"][rng.permutation(n_sub)]
        m_perm = build_models(ages_p, ctx["iu"], ctx["ju"], normalize=bool(ctx["normalize_models"]))
        for mi, mv in enumerate(m_perm):
            rr = assoc(s_chunk, mv, method=str(ctx["assoc_method"]))
            perm_r[mi, :, pi] = rr.astype(np.float32, copy=False)
            obs = obs_chunk[mi]
            valid = np.isfinite(rr) & np.isfinite(obs)
            c_raw[mi, valid] += rr[valid] >= obs[valid]
            if np.any(valid):
                max_stats[mi, pi] = float(np.max(rr[valid]))

    return {"indices": idx, "c_raw": c_raw, "perm_r": perm_r, "max_stats": max_stats}


def run_one(
    stim_dir: Path,
    n_perm: int,
    seed: int,
    n_jobs: int,
    isc_prefix: str,
    normalize_models: bool,
    fisher_z_enabled: bool,
    assoc_method: str,
    correction_mode: str,
) -> pd.DataFrame:
    isc = np.load(stim_dir / f"{isc_prefix}.npy")
    sub_df = pd.read_csv(stim_dir / f"{isc_prefix}_subjects_sorted.csv")
    roi_df = pd.read_csv(stim_dir / f"{isc_prefix}_rois.csv")
    ages = sub_df["age"].astype(float).to_numpy()
    subjects = sub_df["subject"].astype(str).tolist()
    rois = roi_df["roi"].astype(str).tolist()

    n_sub = len(subjects)
    iu, ju = np.triu_indices(n_sub, k=1)
    n_pairs = int(iu.size)

    # Precompute z-scored ISC vectors per ROI (optionally Fisher-Z first).
    S_z = []
    assoc_m = str(assoc_method).strip().lower()
    for i in range(len(rois)):
        v = isc[i][iu, ju]
        if bool(fisher_z_enabled):
            v = fisher_z(v)
        if assoc_m == "spearman":
            mask_v = np.isfinite(v)
            vr = np.full_like(v, np.nan, dtype=np.float32)
            vr[mask_v] = rankdata(v[mask_v]).astype(np.float32, copy=False)
            S_z.append(zscore_1d(vr))
        else:
            S_z.append(zscore_1d(v))
    S_z = np.stack(S_z, axis=0).astype(np.float32)

    model_names = MODEL_NAMES
    m_obs = build_models(ages, iu, ju, normalize=bool(normalize_models))

    obs = []
    tests = []
    for mi, mname in enumerate(model_names):
        rr = assoc(S_z, m_obs[mi], method=str(assoc_method))
        for ri, roi in enumerate(rois):
            tests.append((roi, mname))
            obs.append(float(rr[ri]))
    r_obs = np.asarray(obs, dtype=np.float32)
    r_obs_by_model = r_obs.reshape(len(model_names), len(rois))

    n_rois = len(rois)
    n_tests = int(r_obs.size)
    p_raw = np.full(n_tests, np.nan, dtype=np.float64)
    p_fwer = np.full(n_tests, np.nan, dtype=np.float64)
    mode = str(correction_mode).strip().lower()
    perm_r = np.full((len(model_names), n_rois, int(n_perm) if int(n_perm) > 0 else 0), np.nan, dtype=np.float32)
    perm_max_abs_r = np.full((len(model_names), int(n_perm) if int(n_perm) > 0 else 0), np.nan, dtype=np.float32)
    if mode == "perm_fwer_fdr":
        c_raw_by_model = np.zeros((len(model_names), n_rois), dtype=np.int64)
        max_by_model = np.full((len(model_names), int(n_perm)), -np.inf, dtype=np.float32)
        jobs = max(1, min(int(n_jobs), n_rois))
        chunks = _chunk_indices(n_rois, jobs)
        worker_ctx = {
            "S_z": S_z,
            "r_obs_by_model": r_obs_by_model,
            "ages": np.asarray(ages, dtype=np.float32),
            "iu": iu,
            "ju": ju,
            "n_sub": int(n_sub),
            "n_perm": int(n_perm),
            "seed": int(seed),
            "normalize_models": bool(normalize_models),
            "assoc_method": str(assoc_method),
        }
        if jobs == 1:
            _init_roi_dev_worker(worker_ctx)
            results = [_permute_roi_dev_chunk(chunks[0])]
        else:
            print(f"[parallel] {stim_dir.name}: running {n_rois} ROI tasks in {len(chunks)} chunks with n_jobs={jobs}")
            results = []
            with ProcessPoolExecutor(max_workers=jobs, initializer=_init_roi_dev_worker, initargs=(worker_ctx,)) as ex:
                futures = [ex.submit(_permute_roi_dev_chunk, chunk) for chunk in chunks]
                for fut in as_completed(futures):
                    results.append(fut.result())
        for result in results:
            idx = np.asarray(result["indices"], dtype=np.int32)
            c_raw_by_model[:, idx] = np.asarray(result["c_raw"], dtype=np.int64)
            perm_r[:, idx, :] = np.asarray(result["perm_r"], dtype=np.float32)
            max_by_model = np.maximum(max_by_model, np.asarray(result["max_stats"], dtype=np.float32))
        ok = np.isfinite(r_obs)
        p_raw_by_model = np.full((len(model_names), n_rois), np.nan, dtype=np.float64)
        p_fwer_by_model = np.full((len(model_names), n_rois), np.nan, dtype=np.float64)
        ok_model = np.isfinite(r_obs_by_model)
        denom = float(int(n_perm) + 1)
        p_raw_by_model[ok_model] = (c_raw_by_model[ok_model] + 1.0) / denom
        for mi in range(len(model_names)):
            valid_max = np.isfinite(max_by_model[mi])
            perm_max_abs_r[mi] = max_by_model[mi]
            for ri in range(n_rois):
                obs_val = float(r_obs_by_model[mi, ri])
                if np.isfinite(obs_val):
                    p_fwer_by_model[mi, ri] = (float(np.sum(max_by_model[mi, valid_max] >= obs_val)) + 1.0) / denom
        p_raw = p_raw_by_model.reshape(-1)
        p_fwer = p_fwer_by_model.reshape(-1)
    elif mode == "fdr_only":
        # Fast path: treat r_obs as correlation coefficient and use one-tailed parametric p-values.
        ok = np.isfinite(r_obs)
        if n_pairs <= 2:
            raise ValueError("Too few subject pairs (n_pairs <= 2) for the parametric fdr_only test")
        dfree = float(n_pairs - 2)
        r = np.clip(r_obs[ok].astype(np.float64), -0.999999, 0.999999)
        t_stat = r * np.sqrt(dfree / np.maximum(1.0 - r * r, 1e-12))
        # Use log survival function to avoid underflow to exact 0 when n_pairs is huge.
        log_p = t.logsf(t_stat, df=dfree)
        min_log = float(np.log(np.finfo(np.float64).tiny))
        p_raw[ok] = np.exp(np.maximum(log_p, min_log))
    else:
        raise ValueError(f"Unsupported correction_mode: {correction_mode}")

    rows = []
    for i, (roi, model) in enumerate(tests):
        rows.append({
            "roi": roi,
            "model": model,
            "r_obs": float(r_obs[i]),
            "p_perm_one_tailed": float(p_raw[i]),
            "p_fwer_model_wise": float(p_fwer[i]),
            "n_subjects": int(n_sub),
            "n_pairs": int(n_pairs),
            "n_perm": int(n_perm),
            "seed": int(seed),
            "correction_mode": str(mode),
            "normalize_models": bool(normalize_models),
            "isc_prefix": str(isc_prefix),
            "assoc_method": str(assoc_method),
            "fisher_z": bool(fisher_z_enabled),
        })

    out_df = pd.DataFrame(rows).sort_values(["model", "roi"])
    out_df["p_fdr_bh_model_wise"] = np.nan
    for _, idx in out_df.groupby("model").groups.items():
        out_df.loc[idx, "p_fdr_bh_model_wise"] = bh_fdr(out_df.loc[idx, "p_perm_one_tailed"].to_numpy(dtype=float))
    out_df["p_fdr_bh_global"] = bh_fdr(out_df["p_perm_one_tailed"].to_numpy(dtype=float))
    out_prefix = "roi_isc_dev_models_perm_fwer"
    out_df.to_csv(stim_dir / f"{out_prefix}.csv", index=False)
    sub_df[["subject", "age"]].to_csv(stim_dir / f"{out_prefix}_subjects_sorted.csv", index=False)
    pd.DataFrame({"roi": rois}).to_csv(stim_dir / f"{out_prefix}_rois.csv", index=False)
    metadata = make_metadata(
        script_name=Path(__file__).name,
        result_type="roi_isc_dev_models_perm_null",
        input_files={
            "isc": str(stim_dir / f"{isc_prefix}.npy"),
            "subjects_sorted": str(stim_dir / f"{isc_prefix}_subjects_sorted.csv"),
            "rois": str(stim_dir / f"{isc_prefix}_rois.csv"),
        },
        output_csv=str(stim_dir / f"{out_prefix}.csv"),
        seed=int(seed),
        n_perm=int(n_perm),
        models=list(model_names),
        extra={
            "stimulus_dir": str(stim_dir),
            "isc_prefix": str(isc_prefix),
            "assoc_method": str(assoc_method),
            "correction_mode": str(mode),
            "normalize_models": bool(normalize_models),
            "fisher_z": bool(fisher_z_enabled),
        },
    )
    if mode == "perm_fwer_fdr" and int(n_perm) > 0:
        save_perm_null_npz(
            stim_dir / "roi_isc_dev_models_perm_null.npz",
            metadata=metadata,
            obs_r=r_obs.reshape(len(model_names), n_rois).astype(np.float32),
            perm_r=perm_r.astype(np.float32),
            perm_max_abs_r=perm_max_abs_r.astype(np.float32),
            models=np.asarray(model_names, dtype=str),
            rois=np.asarray(rois, dtype=str),
            subjects_sorted=np.asarray(subjects, dtype=str),
            ages_sorted=ages.astype(np.float32),
            p_perm_one_tailed=p_raw.reshape(len(model_names), n_rois).astype(np.float64),
            p_fwer_model_wise=p_fwer.reshape(len(model_names), n_rois).astype(np.float64),
        )
    return out_df


def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
    n_perm: int,
    seed: int,
    n_jobs: int,
    isc_prefix: Optional[str],
    isc_method: Optional[str],
    normalize_models: bool,
    fisher_z_enabled: bool,
    assoc_method: str,
    repr_prefix: Optional[str],
    correction_mode: str,
) -> None:
    by_stim = matrix_dir / str(stimulus_dir_name)
    stim_dirs = sorted([p for p in by_stim.iterdir() if p.is_dir()])
    if not stim_dirs:
        raise FileNotFoundError(f"{by_stim} has no condition directory")

    summary = []
    for d in stim_dirs:
        if repr_prefix is not None and not has_repr_files(d, repr_prefix=str(repr_prefix)):
            print(f"[SKIP] {d.name}: missing inputs for {repr_prefix}; skipped.")
            continue
        try:
            prefix = resolve_isc_prefix(d, isc_prefix=isc_prefix, isc_method=isc_method)
        except (FileNotFoundError, ValueError) as e:
            print(f"[SKIP] {d.name}: {e}; skipped.")
            continue
        if not has_isc_files(d, isc_prefix=str(prefix)):
            print(f"[SKIP] {d.name}: missing inputs for {prefix}; skipped.")
            continue
        fisher_z_eff = bool(fisher_z_enabled)
        pfx = str(prefix).strip().lower()
        if "roi_isc_euclidean_by_age" in pfx or "roi_isc_mahalanobis_by_age" in pfx:
            fisher_z_eff = False

        out = run_one(
            d,
            n_perm=int(n_perm),
            seed=int(seed),
            n_jobs=int(n_jobs),
            isc_prefix=str(prefix),
            normalize_models=bool(normalize_models),
            fisher_z_enabled=bool(fisher_z_eff),
            assoc_method=str(assoc_method),
            correction_mode=str(correction_mode),
        )
        sig = out[(out["r_obs"] > 0) & (out["p_fwer_model_wise"] <= 0.05)]
        sig_fdr = out[(out["r_obs"] > 0) & (out["p_fdr_bh_model_wise"] <= 0.05)]
        summary.append(
            {
                "stimulus_type": d.name,
                "n_rows": int(out.shape[0]),
                "correction_mode": str(correction_mode),
                "n_sig_pos_fwer": int(sig.shape[0]),
                "n_sig_pos_fdr_model_wise": int(sig_fdr.shape[0]),
            }
        )

    pd.DataFrame(summary).sort_values("stimulus_type").to_csv(matrix_dir / "roi_isc_dev_models_perm_fwer_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    by_stim = matrix_dir / str(args.stimulus_dir_name)
    stim_dirs = sorted([p for p in by_stim.iterdir() if p.is_dir()]) if by_stim.exists() else []
    if not stim_dirs:
        raise FileNotFoundError(f"{by_stim} has no condition directory")
    run(
        matrix_dir,
        str(args.stimulus_dir_name),
        int(args.n_perm),
        int(args.seed),
        int(args.n_jobs),
        isc_prefix=args.isc_prefix,
        isc_method=args.isc_method,
        normalize_models=bool(args.normalize_models),
        fisher_z_enabled=bool(args.fisher_z),
        assoc_method=str(args.assoc_method),
        repr_prefix=args.repr_prefix,
        correction_mode=str(args.correction_mode),
    )


if __name__ == "__main__":
    main()
