#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t

if __package__ in {None, "", "behavior"}:
    THIS_DIR = Path(__file__).resolve().parent
    EMO_DIR = THIS_DIR.parent
    if str(EMO_DIR) not in sys.path:
        sys.path.insert(0, str(EMO_DIR))

    from joint_analysis_roi_isc_dev_models import bh_fdr, build_models, fisher_z, zscore_1d  # noqa: E402
    from perm_null_io import make_metadata, save_perm_null_npz  # noqa: E402
else:
    from ..joint_analysis_roi_isc_dev_models import bh_fdr, build_models, fisher_z, zscore_1d  # noqa: E402
    from ..perm_null_io import make_metadata, save_perm_null_npz  # noqa: E402


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
MODEL_NAMES = ("M_nn", "M_conv", "M_div")
OUT_PREFIX = "behavior_isc_dev_models_perm_fwer"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate behavior ISC with three age models")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus", choices=("by_stimulus", "by_emotion"))
    p.add_argument("--behavior-isc-method", type=str, default="mahalanobis", choices=("spearman", "pearson", "euclidean", "mahalanobis"))
    p.add_argument("--behavior-isc-prefix", type=str, default=None)
    p.add_argument("--assoc-method", type=str, default="spearman", choices=("pearson", "spearman"))
    p.add_argument("--tail", type=str, default="positive", choices=("positive", "two_sided"))
    p.add_argument("--correction-mode", type=str, default="perm_fwer_fdr", choices=("fdr_only", "perm_fwer_fdr"))
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1, help="Number of permutation workers.")
    p.add_argument("--n-bootstrap", type=int, default=5000, help="Number of bootstrap resamples saved to bootstrap_rhos.npz. Use 0 to disable.")
    p.add_argument("--bootstrap-seed", type=int, default=202405)
    p.add_argument("--fisher-z-behavior", type=str, default=None)
    p.add_argument("--no-normalize-models", action="store_false", dest="normalize_models", default=True)
    return p.parse_args()


_BEHAV_DEV_WORKER_CTX: dict = {}
_BEHAV_BOOT_WORKER_CTX: dict = {}


def _init_beh_dev_worker(ctx: dict) -> None:
    global _BEHAV_DEV_WORKER_CTX
    _BEHAV_DEV_WORKER_CTX = ctx


def _init_beh_boot_worker(ctx: dict) -> None:
    global _BEHAV_BOOT_WORKER_CTX
    _BEHAV_BOOT_WORKER_CTX = ctx


def _chunk_perm_ranges(n_perm: int, n_jobs: int) -> list[tuple[int, int]]:
    jobs = max(1, min(int(n_jobs), int(n_perm)))
    bounds = np.linspace(0, int(n_perm), jobs + 1, dtype=int)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(jobs) if int(bounds[i + 1]) > int(bounds[i])]


def _policy_flag(method: str, override: Optional[str]) -> bool:
    if override is not None:
        return str(override).strip().lower() == "true"
    return str(method).strip().lower() in {"pearson", "spearman"}


def _resolve_prefix(prefix: Optional[str], method: str) -> str:
    return str(prefix) if prefix is not None else f"behavior_isc_{str(method).strip().lower()}_by_age"


def has_behavior_files(stim_dir: Path, behavior_isc_prefix: str) -> bool:
    required = (
        stim_dir / f"{behavior_isc_prefix}.npy",
        stim_dir / f"{behavior_isc_prefix}_subjects_sorted.csv",
    )
    return all(p.exists() for p in required)


def _assoc_one(x: np.ndarray, m: np.ndarray, method: str) -> float:
    xv = np.asarray(x, dtype=np.float32).reshape(-1)
    mv = np.asarray(m, dtype=np.float32).reshape(-1)
    ok = np.isfinite(xv) & np.isfinite(mv)
    if int(ok.sum()) < 3:
        return float("nan")

    if str(method).strip().lower() == "spearman":
        xr = rankdata(xv[ok]).astype(np.float32, copy=False)
        mr = rankdata(mv[ok]).astype(np.float32, copy=False)
        xz = zscore_1d(xr)
        mz = zscore_1d(mr)
    else:
        xz = zscore_1d(xv[ok])
        mz = zscore_1d(mv[ok])
    keep = np.isfinite(xz) & np.isfinite(mz)
    if int(keep.sum()) < 3:
        return float("nan")
    return float(np.mean(xz[keep] * mz[keep]))


def _p_from_counts(c: np.ndarray, n_perm: int, valid: np.ndarray) -> np.ndarray:
    p = np.full(c.shape, np.nan, dtype=np.float64)
    p[valid] = (c[valid].astype(float) + 1.0) / (float(n_perm) + 1.0)
    return p


def _permute_behavior_dev_range(bounds: tuple[int, int]) -> dict:
    start, stop = int(bounds[0]), int(bounds[1])
    ctx = _BEHAV_DEV_WORKER_CTX
    n_sub = int(ctx["n_sub"])
    n_local = max(0, stop - start)
    c_raw = np.zeros(len(MODEL_NAMES), dtype=np.int64)
    c_fwer = np.zeros(len(MODEL_NAMES), dtype=np.int64)
    perm_rhos = np.full((len(MODEL_NAMES), n_local), np.nan, dtype=np.float32)
    rng = np.random.default_rng(int(ctx["seed"]))
    for _ in range(start):
        rng.permutation(n_sub)
    for local_i in range(n_local):
        ages_p = ctx["ages"][rng.permutation(n_sub)]
        m_perm = build_models(ages_p, ctx["iu"], ctx["ju"], normalize=bool(ctx["normalize_models"]))
        r_perm = np.asarray([_assoc_one(ctx["beh_vec"], m, method=str(ctx["assoc_method"])) for m in m_perm], dtype=np.float32)
        perm_rhos[:, local_i] = r_perm
        valid = np.isfinite(r_perm) & ctx["valid_obs"]
        if str(ctx["tail_mode"]) == "two_sided":
            c_raw[valid] += np.abs(r_perm[valid]) >= np.abs(ctx["r_obs"][valid])
            if np.any(valid):
                max_stat = float(np.max(np.abs(r_perm[valid])))
                c_fwer[ctx["valid_obs"]] += max_stat >= np.abs(ctx["r_obs"][ctx["valid_obs"]])
        else:
            c_raw[valid] += r_perm[valid] >= ctx["r_obs"][valid]
            if np.any(valid):
                max_stat = float(np.max(r_perm[valid]))
                c_fwer[ctx["valid_obs"]] += max_stat >= ctx["r_obs"][ctx["valid_obs"]]
    return {"start": start, "stop": stop, "c_raw": c_raw, "c_fwer": c_fwer, "perm_rhos": perm_rhos}


def _bootstrap_behavior_dev_range(bounds: tuple[int, int]) -> dict:
    start, stop = int(bounds[0]), int(bounds[1])
    ctx = _BEHAV_BOOT_WORKER_CTX
    n_sub = int(ctx["n_sub"])
    n_local = max(0, stop - start)
    boot_rhos = np.full((len(MODEL_NAMES), n_local), np.nan, dtype=np.float32)
    iu, ju = np.triu_indices(n_sub, k=1)
    rng = np.random.default_rng(int(ctx["bootstrap_seed"]))
    for _ in range(start):
        rng.integers(0, n_sub, size=n_sub)
    for local_i in range(n_local):
        sample_idx = rng.integers(0, n_sub, size=n_sub)
        bmat = ctx["behavior_isc"][np.ix_(sample_idx, sample_idx)]
        ages_b = ctx["ages"][sample_idx]
        beh_vec = bmat[iu, ju].astype(np.float32, copy=False)
        same_subject = sample_idx[iu] == sample_idx[ju]
        beh_vec = beh_vec.copy()
        beh_vec[same_subject] = np.nan
        if bool(ctx["fisher_z_behavior"]):
            beh_vec = fisher_z(beh_vec)
        models = build_models(ages_b, iu, ju, normalize=bool(ctx["normalize_models"]))
        for mi, mv in enumerate(models):
            mv_eff = np.asarray(mv, dtype=np.float32).copy()
            mv_eff[same_subject] = np.nan
            boot_rhos[mi, local_i] = _assoc_one(beh_vec, mv_eff, method=str(ctx["assoc_method"]))
    return {"start": start, "stop": stop, "boot_rhos": boot_rhos}


def _compute_bootstrap_rhos(
    *,
    behavior_isc: np.ndarray,
    ages: np.ndarray,
    assoc_method: str,
    normalize_models: bool,
    fisher_z_behavior: bool,
    n_bootstrap: int,
    bootstrap_seed: int,
    n_jobs: int,
    stim_name: str,
) -> np.ndarray:
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap <= 0:
        return np.full((len(MODEL_NAMES), 0), np.nan, dtype=np.float32)
    n_sub = int(np.asarray(ages).size)
    jobs = max(1, min(int(n_jobs), n_bootstrap))
    ranges = _chunk_perm_ranges(n_bootstrap, jobs)
    worker_ctx = {
        "behavior_isc": np.asarray(behavior_isc, dtype=np.float32),
        "ages": np.asarray(ages, dtype=np.float32),
        "n_sub": int(n_sub),
        "bootstrap_seed": int(bootstrap_seed),
        "normalize_models": bool(normalize_models),
        "fisher_z_behavior": bool(fisher_z_behavior),
        "assoc_method": str(assoc_method),
    }
    if jobs == 1:
        _init_beh_boot_worker(worker_ctx)
        results = [_bootstrap_behavior_dev_range(ranges[0])]
    else:
        print(f"[bootstrap] {stim_name}: running {n_bootstrap} bootstrap resamples in {len(ranges)} chunks with n_jobs={jobs}")
        results = []
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_beh_boot_worker, initargs=(worker_ctx,)) as ex:
            futures = [ex.submit(_bootstrap_behavior_dev_range, r) for r in ranges]
            for fut in as_completed(futures):
                results.append(fut.result())
    boot_rhos = np.full((len(MODEL_NAMES), n_bootstrap), np.nan, dtype=np.float32)
    for result in results:
        start = int(result["start"])
        stop = int(result["stop"])
        boot_rhos[:, start:stop] = np.asarray(result["boot_rhos"], dtype=np.float32)
    return boot_rhos


def run_one(
    stim_dir: Path,
    behavior_isc_prefix: str,
    behavior_isc_method: str,
    fisher_z_behavior: bool,
    assoc_method: str,
    tail: str,
    correction_mode: str,
    normalize_models: bool,
    n_perm: int,
    seed: int,
    n_jobs: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    behavior_isc = np.load(stim_dir / f"{behavior_isc_prefix}.npy")
    sub_df = pd.read_csv(stim_dir / f"{behavior_isc_prefix}_subjects_sorted.csv")
    subjects = sub_df["subject"].astype(str).tolist()
    ages = sub_df["age"].astype(float).to_numpy()

    n_sub = len(subjects)
    iu, ju = np.triu_indices(n_sub, k=1)
    n_pairs = int(iu.size)
    beh_vec = np.asarray(behavior_isc, dtype=np.float32)[iu, ju]
    if bool(fisher_z_behavior):
        beh_vec = fisher_z(beh_vec)

    models = build_models(ages, iu, ju, normalize=bool(normalize_models))
    r_obs = np.asarray([_assoc_one(beh_vec, m, method=str(assoc_method)) for m in models], dtype=np.float32)

    tail_mode = str(tail).strip().lower()
    mode = str(correction_mode).strip().lower()
    p_raw = np.full(len(MODEL_NAMES), np.nan, dtype=np.float64)
    p_fwer = np.full(len(MODEL_NAMES), np.nan, dtype=np.float64)
    valid_obs = np.isfinite(r_obs)
    perm_rhos = np.full((len(MODEL_NAMES), int(n_perm) if int(n_perm) > 0 else 0), np.nan, dtype=np.float32)

    if mode == "perm_fwer_fdr":
        c_raw = np.zeros(len(MODEL_NAMES), dtype=np.int64)
        c_fwer = np.zeros(len(MODEL_NAMES), dtype=np.int64)
        jobs = max(1, min(int(n_jobs), int(n_perm)))
        ranges = _chunk_perm_ranges(int(n_perm), jobs)
        worker_ctx = {
            "beh_vec": beh_vec,
            "ages": np.asarray(ages, dtype=np.float32),
            "iu": iu,
            "ju": ju,
            "n_sub": int(n_sub),
            "seed": int(seed),
            "normalize_models": bool(normalize_models),
            "assoc_method": str(assoc_method),
            "tail_mode": str(tail_mode),
            "r_obs": r_obs,
            "valid_obs": valid_obs,
        }
        if jobs == 1:
            _init_beh_dev_worker(worker_ctx)
            results = [_permute_behavior_dev_range(ranges[0])]
        else:
            print(f"[parallel] {stim_dir.name}: running {int(n_perm)} permutations in {len(ranges)} chunks with n_jobs={jobs}")
            results = []
            with ProcessPoolExecutor(max_workers=jobs, initializer=_init_beh_dev_worker, initargs=(worker_ctx,)) as ex:
                futures = [ex.submit(_permute_behavior_dev_range, r) for r in ranges]
                for fut in as_completed(futures):
                    results.append(fut.result())
        for result in results:
            start = int(result["start"])
            stop = int(result["stop"])
            c_raw += np.asarray(result["c_raw"], dtype=np.int64)
            c_fwer += np.asarray(result["c_fwer"], dtype=np.int64)
            perm_rhos[:, start:stop] = np.asarray(result["perm_rhos"], dtype=np.float32)
        p_raw = _p_from_counts(c_raw, int(n_perm), valid_obs)
        p_fwer = _p_from_counts(c_fwer, int(n_perm), valid_obs)
    elif mode == "fdr_only":
        if n_pairs <= 2:
            raise ValueError("Too few subject pairs for parametric p-values")
        dfree = float(n_pairs - 2)
        r = np.clip(r_obs[valid_obs].astype(np.float64), -0.999999, 0.999999)
        t_stat = r * np.sqrt(dfree / np.maximum(1.0 - r * r, 1e-12))
        if tail_mode == "two_sided":
            p_raw[valid_obs] = 2.0 * t.sf(np.abs(t_stat), df=dfree)
        else:
            p_raw[valid_obs] = t.sf(t_stat, df=dfree)
    else:
        raise ValueError(f"Unsupported correction_mode: {correction_mode}")

    out = pd.DataFrame(
        {
            "model": list(MODEL_NAMES),
            "r_obs": r_obs.astype(float),
            "p_perm": p_raw.astype(float),
            "p_fwer_global": p_fwer.astype(float),
            "p_fdr_bh_global": bh_fdr(p_raw),
            "n_subjects": int(n_sub),
            "n_pairs": int(n_pairs),
            "n_perm": int(n_perm),
            "seed": int(seed),
            "n_jobs": int(n_jobs),
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(bootstrap_seed),
            "tail": str(tail_mode),
            "correction_mode": str(mode),
            "behavior_isc_prefix": str(behavior_isc_prefix),
            "behavior_isc_method": str(behavior_isc_method),
            "fisher_z_behavior": bool(fisher_z_behavior),
            "assoc_method": str(assoc_method),
            "normalize_models": bool(normalize_models),
        }
    ).sort_values("model")
    out.to_csv(stim_dir / f"{OUT_PREFIX}.csv", index=False)
    sub_df[["subject", "age"]].to_csv(stim_dir / f"{OUT_PREFIX}_subjects_sorted.csv", index=False)
    metadata = make_metadata(
        script_name=Path(__file__).name,
        result_type="behavior_isc_dev_models_perm_null",
        input_files={
            "behavior_isc": str(stim_dir / f"{behavior_isc_prefix}.npy"),
            "subjects_sorted": str(stim_dir / f"{behavior_isc_prefix}_subjects_sorted.csv"),
        },
        output_csv=str(stim_dir / f"{OUT_PREFIX}.csv"),
        seed=int(seed),
        n_perm=int(n_perm),
        models=list(MODEL_NAMES),
        extra={
            "stimulus_dir": str(stim_dir),
            "behavior_isc_prefix": str(behavior_isc_prefix),
            "behavior_isc_method": str(behavior_isc_method),
            "input_kind": "behavior_isc_matrix",
            "assoc_method": str(assoc_method),
            "tail": str(tail_mode),
            "correction_mode": str(mode),
            "normalize_models": bool(normalize_models),
            "fisher_z_behavior": bool(fisher_z_behavior),
        },
    )
    if mode == "perm_fwer_fdr" and int(n_perm) > 0:
        save_perm_null_npz(
            stim_dir / "behavior_isc_dev_models_perm_null.npz",
            metadata=metadata,
            obs_rho=r_obs.astype(np.float32),
            perm_rhos=perm_rhos.astype(np.float32),
            models=np.asarray(MODEL_NAMES, dtype=str),
            p_perm=p_raw.astype(np.float64),
            p_fwer=p_fwer.astype(np.float64),
            p_fdr_bh=bh_fdr(p_raw).astype(np.float64),
            subjects=np.asarray(subjects, dtype=str),
            ages=ages.astype(np.float32),
        )
    if int(n_bootstrap) > 0:
        boot_rhos = _compute_bootstrap_rhos(
            behavior_isc=behavior_isc,
            ages=ages,
            assoc_method=str(assoc_method),
            normalize_models=bool(normalize_models),
            fisher_z_behavior=bool(fisher_z_behavior),
            n_bootstrap=int(n_bootstrap),
            bootstrap_seed=int(bootstrap_seed),
            n_jobs=int(n_jobs),
            stim_name=stim_dir.name,
        )
        np.savez_compressed(
            stim_dir / "bootstrap_rhos.npz",
            models=np.asarray(MODEL_NAMES, dtype=str),
            bootstrap_rhos=boot_rhos.astype(np.float32),
            M_nn_bootstrap=boot_rhos[0].astype(np.float32),
            M_conv_bootstrap=boot_rhos[1].astype(np.float32),
            M_div_bootstrap=boot_rhos[2].astype(np.float32),
            subjects=np.asarray(subjects, dtype=str),
            ages=ages.astype(np.float32),
            behavior_isc_prefix=np.asarray(str(behavior_isc_prefix)),
            behavior_isc_method=np.asarray(str(behavior_isc_method)),
            assoc_method=np.asarray(str(assoc_method)),
            fisher_z_behavior=np.asarray(bool(fisher_z_behavior)),
            normalize_models=np.asarray(bool(normalize_models)),
            bootstrap_seed=np.asarray(int(bootstrap_seed), dtype=np.int64),
            n_bootstrap=np.asarray(int(n_bootstrap), dtype=np.int64),
        )
    return out


def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
    behavior_isc_method: str,
    behavior_isc_prefix: Optional[str],
    assoc_method: str,
    tail: str,
    correction_mode: str,
    n_perm: int,
    seed: int,
    n_jobs: int,
    n_bootstrap: int,
    bootstrap_seed: int,
    fisher_z_behavior: Optional[str],
    normalize_models: bool,
) -> None:
    by_stim = Path(matrix_dir) / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"Cannot find directory: {by_stim}")

    prefix = _resolve_prefix(behavior_isc_prefix, method=str(behavior_isc_method))
    fisher_z_eff = _policy_flag(method=str(behavior_isc_method), override=fisher_z_behavior)
    summary = []
    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        if not has_behavior_files(stim_dir, behavior_isc_prefix=str(prefix)):
            print(f"[SKIP] {stim_dir.name}: missing behavior ISC files for {prefix}")
            continue
        out = run_one(
            stim_dir=stim_dir,
            behavior_isc_prefix=str(prefix),
            behavior_isc_method=str(behavior_isc_method),
            fisher_z_behavior=bool(fisher_z_eff),
            assoc_method=str(assoc_method),
            tail=str(tail),
            correction_mode=str(correction_mode),
            normalize_models=bool(normalize_models),
            n_perm=int(n_perm),
            seed=int(seed),
            n_jobs=int(n_jobs),
            n_bootstrap=int(n_bootstrap),
            bootstrap_seed=int(bootstrap_seed),
        )
        row = {"stimulus_type": stim_dir.name, "n_rows": int(out.shape[0])}
        row["n_bootstrap"] = int(n_bootstrap)
        row["bootstrap_seed"] = int(bootstrap_seed)
        for model in MODEL_NAMES:
            sub = out[out["model"].astype(str) == str(model)]
            if not sub.empty:
                row[f"{model}_r_obs"] = float(sub.iloc[0]["r_obs"])
                row[f"{model}_p_fdr_bh_global"] = float(sub.iloc[0]["p_fdr_bh_global"])
                row[f"{model}_p_fwer_global"] = float(sub.iloc[0]["p_fwer_global"])
        summary.append(row)

    summary_name = f"{OUT_PREFIX}_{str(stimulus_dir_name)}_summary.csv"
    pd.DataFrame(summary).sort_values("stimulus_type").to_csv(Path(matrix_dir) / summary_name, index=False)


def main() -> None:
    args = parse_args()
    run(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        behavior_isc_method=str(args.behavior_isc_method),
        behavior_isc_prefix=args.behavior_isc_prefix,
        assoc_method=str(args.assoc_method),
        tail=str(args.tail),
        correction_mode=str(args.correction_mode),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        n_jobs=int(args.n_jobs),
        n_bootstrap=int(args.n_bootstrap),
        bootstrap_seed=int(args.bootstrap_seed),
        fisher_z_behavior=args.fisher_z_behavior,
        normalize_models=bool(args.normalize_models),
    )


if __name__ == "__main__":
    main()
