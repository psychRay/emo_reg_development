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

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from joint_analysis_roi_isc_dev_models import (  # noqa: E402
    assoc,
    bh_fdr,
    build_models,
    fisher_z,
    resolve_isc_prefix,
    zscore_1d,
)
from perm_null_io import make_metadata, save_perm_null_npz  # noqa: E402


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
DEFAULT_SCHAEFER_MAPPING_NAME = "schaefer2018_200parcels_7networks_order_mapping.csv"
OUT_PREFIX = "roi_isc_delta_dev_models_perm_fwer"
MODEL_NAMES = ("M_nn", "M_conv", "M_div")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Developmental model analysis for brain ISC delta: "
            "ROI_ISC(Reappraisal) - ROI_ISC(Passive_Emo) vs M_nn/M_conv/M_div."
        )
    )
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_emotion")
    p.add_argument("--reappraisal-name", type=str, default="Reappraisal")
    p.add_argument("--passive-name", type=str, default="Passive_Emo")
    p.add_argument("--out-condition-name", type=str, default=None)
    p.add_argument("--isc-method", type=str, default="mahalanobis", choices=("spearman", "pearson", "euclidean", "mahalanobis"))
    p.add_argument("--isc-prefix", type=str, default=None)
    p.add_argument("--assoc-method", type=str, default="spearman", choices=("pearson", "spearman"))
    p.add_argument("--correction-mode", type=str, default="fdr_only", choices=("fdr_only", "perm_fwer_fdr"))
    p.add_argument("--tail", type=str, default="positive", choices=("positive", "two-sided"))
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1, help="Number of ROI workers for permutation mode.")
    p.add_argument("--no-normalize-models", action="store_false", dest="normalize_models", default=True)
    p.add_argument("--no-fisher-z", action="store_false", dest="fisher_z", default=True)
    p.add_argument("--schaefer-mapping-file", type=Path, default=None)
    return p.parse_args()


def _read_subjects(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "subject" not in df.columns:
        raise ValueError(f"{path} must contain a subject column")
    return df


def _age_maps(*dfs: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    seen: dict[str, float] = {}
    for df in dfs:
        if "age" not in df.columns:
            continue
        for sub, age in zip(df["subject"].astype(str), pd.to_numeric(df["age"], errors="coerce")):
            if not np.isfinite(age) or float(age) >= 998.0:
                continue
            sub = str(sub)
            age = float(age)
            if sub in seen and abs(float(seen[sub]) - age) > 1e-5:
                raise ValueError(f"Age differs across subject CSVs for {sub}: {seen[sub]} vs {age}")
            seen[sub] = age
            out[sub] = age
    return out


def _check_brain(name: str, arr: np.ndarray, n_rois: int, n_sub: int) -> None:
    if arr.ndim != 3 or arr.shape[0] != n_rois or arr.shape[1:] != (n_sub, n_sub):
        raise ValueError(f"{name} shape {arr.shape} does not match expected ({n_rois}, {n_sub}, {n_sub})")


def _load_aligned_delta(
    reapp_dir: Path,
    passive_dir: Path,
    isc_prefix: str,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    reapp = np.load(reapp_dir / f"{isc_prefix}.npy")
    passive = np.load(passive_dir / f"{isc_prefix}.npy")
    rois_reapp = pd.read_csv(reapp_dir / f"{isc_prefix}_rois.csv")["roi"].astype(str).tolist()
    rois_passive = pd.read_csv(passive_dir / f"{isc_prefix}_rois.csv")["roi"].astype(str).tolist()
    if rois_reapp != rois_passive:
        raise ValueError("Reappraisal and Passive_Emo ROI order differ")

    sub_reapp_df = _read_subjects(reapp_dir / f"{isc_prefix}_subjects_sorted.csv")
    sub_passive_df = _read_subjects(passive_dir / f"{isc_prefix}_subjects_sorted.csv")
    sub_reapp = sub_reapp_df["subject"].astype(str).tolist()
    sub_passive = sub_passive_df["subject"].astype(str).tolist()
    for label, subs in {"Reappraisal": sub_reapp, "Passive_Emo": sub_passive}.items():
        if len(set(subs)) != len(subs):
            raise ValueError(f"Duplicate subjects in {label} subject file")

    _check_brain("Reappraisal ISC", reapp, len(rois_reapp), len(sub_reapp))
    _check_brain("Passive_Emo ISC", passive, len(rois_passive), len(sub_passive))

    age_map = _age_maps(sub_reapp_df, sub_passive_df)
    passive_set = set(sub_passive)
    subjects = [s for s in sub_reapp if s in passive_set and np.isfinite(age_map.get(s, np.nan))]
    if len(subjects) < 6:
        raise ValueError(f"Fewer than 6 common subjects with valid age: n={len(subjects)}")

    idx_reapp = {s: i for i, s in enumerate(sub_reapp)}
    idx_passive = {s: i for i, s in enumerate(sub_passive)}
    ir = [idx_reapp[s] for s in subjects]
    ip = [idx_passive[s] for s in subjects]
    delta = (
        np.asarray(reapp, dtype=np.float32)[:, ir, :][:, :, ir]
        - np.asarray(passive, dtype=np.float32)[:, ip, :][:, :, ip]
    )
    ages = np.asarray([age_map[s] for s in subjects], dtype=np.float32)
    return delta, subjects, ages, rois_reapp


def _roi_aliases(roi: str) -> set[str]:
    r = str(roi)
    aliases = {r}
    if "_" not in r:
        return aliases
    hemi, num = r.split("_", 1)
    if hemi not in {"L", "R"}:
        return aliases
    try:
        idx = int(num)
    except ValueError:
        return aliases
    if hemi == "R":
        aliases.add(f"R_{idx - 100}" if idx > 100 else f"R_{idx + 100}")
    return aliases


def _load_schaefer_mapping(mapping_file: Optional[Path], matrix_dir: Path) -> dict[str, dict]:
    path = Path(mapping_file) if mapping_file is not None else Path(matrix_dir) / DEFAULT_SCHAEFER_MAPPING_NAME
    if not path.exists():
        print(f"[mapping] Schaefer mapping file not found, output will omit source_name: {path}")
        return {}
    df = pd.read_csv(path)
    if "roi" not in df.columns or "source_name" not in df.columns:
        raise ValueError(f"{path} must contain roi and source_name columns")
    keep = [c for c in ("roi", "roi_schaefer_style", "source_name") if c in df.columns]
    out: dict[str, dict] = {}
    for _, row in df[keep].iterrows():
        rec = {"source_name": row["source_name"]}
        for col in ("roi", "roi_schaefer_style"):
            if col in row.index and pd.notna(row[col]):
                for alias in _roi_aliases(str(row[col])):
                    out.setdefault(alias, rec)
    print(f"[mapping] Loaded Schaefer mapping: {path}")
    return out


def _add_source_name(out_df: pd.DataFrame, mapping: dict[str, dict]) -> pd.DataFrame:
    if not mapping:
        return out_df
    records = []
    missing = []
    for roi in out_df["roi"].astype(str):
        rec = None
        for alias in _roi_aliases(str(roi)):
            if alias in mapping:
                rec = mapping[alias]
                break
        if rec is None:
            missing.append(str(roi))
            rec = {}
        records.append(rec)
    map_df = pd.DataFrame(records)
    out = pd.concat([out_df.reset_index(drop=True), map_df.reset_index(drop=True)], axis=1)
    if missing:
        print(f"[mapping] Warning: {len(set(missing))} ROI labels not found in Schaefer mapping")
    return out


def _prepare_delta_vectors(
    delta: np.ndarray,
    iu: np.ndarray,
    ju: np.ndarray,
    fisher_z_enabled: bool,
    assoc_method: str,
) -> np.ndarray:
    from scipy.stats import rankdata

    rows = []
    method = str(assoc_method).strip().lower()
    for ri in range(delta.shape[0]):
        v = delta[ri][iu, ju].astype(np.float32)
        if bool(fisher_z_enabled):
            v = fisher_z(v)
        if method == "spearman":
            mask = np.isfinite(v)
            vr = np.full_like(v, np.nan, dtype=np.float32)
            if int(mask.sum()) > 0:
                vr[mask] = rankdata(v[mask]).astype(np.float32, copy=False)
            rows.append(zscore_1d(vr))
        else:
            rows.append(zscore_1d(v))
    return np.stack(rows, axis=0).astype(np.float32)


_DELTA_DEV_WORKER_CTX: dict = {}


def _init_delta_dev_worker(ctx: dict) -> None:
    global _DELTA_DEV_WORKER_CTX
    _DELTA_DEV_WORKER_CTX = ctx


def _chunk_indices(n_items: int, n_jobs: int) -> list[np.ndarray]:
    jobs = max(1, min(int(n_jobs), int(n_items)))
    chunks = [np.asarray(x, dtype=np.int32) for x in np.array_split(np.arange(int(n_items), dtype=np.int32), jobs)]
    return [x for x in chunks if int(x.size) > 0]


def _permute_delta_dev_chunk(indices: np.ndarray) -> dict:
    ctx = _DELTA_DEV_WORKER_CTX
    idx = np.asarray(indices, dtype=np.int32)
    s_chunk = np.asarray(ctx["s_z"][idx], dtype=np.float32)
    obs_chunk = np.asarray(ctx["r_obs_by_model"][:, idx], dtype=np.float32)
    n_perm = int(ctx["n_perm"])
    n_sub = int(ctx["n_sub"])
    n_chunk = int(idx.size)
    c_raw = np.zeros((len(MODEL_NAMES), n_chunk), dtype=np.int64)
    max_stats = np.full((len(MODEL_NAMES), n_perm), -np.inf, dtype=np.float64)
    perm_r = np.full((len(MODEL_NAMES), n_chunk, n_perm), np.nan, dtype=np.float32)
    rng = np.random.default_rng(int(ctx["seed"]))
    tail = str(ctx.get("tail", "positive")).strip().lower()

    for pi in range(n_perm):
        ages_p = ctx["ages"][rng.permutation(n_sub)]
        m_perm = build_models(
            ages_p,
            ctx["iu"],
            ctx["ju"],
            normalize=bool(ctx["normalize_models"]),
        )
        for mi, mv in enumerate(m_perm):
            rr = assoc(s_chunk, mv, method=str(ctx["assoc_method"]))
            perm_r[mi, :, pi] = rr
            obs = obs_chunk[mi]
            valid = np.isfinite(rr) & np.isfinite(obs)
            if tail == "two-sided":
                c_raw[mi, valid] += np.abs(rr[valid]) >= np.abs(obs[valid])
                stat = np.abs(rr[valid])
            else:
                c_raw[mi, valid] += rr[valid] >= obs[valid]
                stat = rr[valid]
            if np.any(valid):
                max_stats[mi, pi] = float(np.max(stat))

    return {"indices": idx, "c_raw": c_raw, "max_stats": max_stats, "perm_r": perm_r}


def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
    reappraisal_name: str,
    passive_name: str,
    out_condition_name: Optional[str],
    isc_method: str,
    isc_prefix: Optional[str],
    assoc_method: str,
    correction_mode: str,
    tail: str,
    n_perm: int,
    seed: int,
    n_jobs: int,
    normalize_models: bool,
    fisher_z_enabled: bool,
    schaefer_mapping_file: Optional[Path],
) -> pd.DataFrame:
    matrix_dir = Path(matrix_dir)
    by_stim = matrix_dir / str(stimulus_dir_name)
    reapp_dir = by_stim / str(reappraisal_name)
    passive_dir = by_stim / str(passive_name)
    if not reapp_dir.exists():
        raise FileNotFoundError(f"Cannot find Reappraisal directory: {reapp_dir}")
    if not passive_dir.exists():
        raise FileNotFoundError(f"Cannot find Passive directory: {passive_dir}")

    prefix = resolve_isc_prefix(reapp_dir, isc_prefix=isc_prefix, isc_method=str(isc_method))
    passive_prefix = resolve_isc_prefix(passive_dir, isc_prefix=isc_prefix, isc_method=str(isc_method))
    if str(prefix) != str(passive_prefix):
        raise ValueError(f"Reappraisal/passive ISC prefixes differ: {prefix} vs {passive_prefix}")

    fisher_z_eff = bool(fisher_z_enabled)
    pfx = str(prefix).strip().lower()
    if "roi_isc_euclidean_by_age" in pfx or "roi_isc_mahalanobis_by_age" in pfx:
        fisher_z_eff = False

    delta, subjects, ages, rois = _load_aligned_delta(
        reapp_dir=reapp_dir,
        passive_dir=passive_dir,
        isc_prefix=str(prefix),
    )
    n_sub = len(subjects)
    iu, ju = np.triu_indices(n_sub, k=1)
    n_pairs = int(iu.size)
    s_z = _prepare_delta_vectors(
        delta=delta,
        iu=iu,
        ju=ju,
        fisher_z_enabled=bool(fisher_z_eff),
        assoc_method=str(assoc_method),
    )
    m_obs = build_models(ages, iu, ju, normalize=bool(normalize_models))

    n_rois = len(rois)
    r_obs_by_model = np.stack(
        [assoc(s_z, m_obs[mi], method=str(assoc_method)) for mi in range(len(MODEL_NAMES))],
        axis=0,
    ).astype(np.float32)
    p_raw_by_model = np.full(r_obs_by_model.shape, np.nan, dtype=np.float64)
    p_fwer_by_model = np.full(r_obs_by_model.shape, np.nan, dtype=np.float64)

    mode = str(correction_mode).strip().lower()
    tail_mode = str(tail).strip().lower()
    if tail_mode not in {"positive", "two-sided"}:
        raise ValueError(f"Unsupported tail: {tail}")
    if mode == "perm_fwer_fdr":
        jobs = max(1, min(int(n_jobs), int(n_rois)))
        chunks = _chunk_indices(n_rois, jobs)
        c_raw_by_model = np.zeros(r_obs_by_model.shape, dtype=np.int64)
        max_by_model = np.full((len(MODEL_NAMES), int(n_perm)), -np.inf, dtype=np.float64)
        perm_r_by_model = np.full((len(MODEL_NAMES), n_rois, int(n_perm)), np.nan, dtype=np.float32)
        worker_ctx = {
            "s_z": s_z,
            "r_obs_by_model": r_obs_by_model,
            "ages": np.asarray(ages, dtype=np.float32),
            "iu": iu,
            "ju": ju,
            "n_sub": int(n_sub),
            "n_perm": int(n_perm),
            "seed": int(seed),
            "normalize_models": bool(normalize_models),
            "assoc_method": str(assoc_method),
            "tail": str(tail_mode),
        }
        if jobs == 1:
            _init_delta_dev_worker(worker_ctx)
            results = [_permute_delta_dev_chunk(chunks[0])]
        else:
            print(f"[parallel] delta dev models: running {n_rois} ROI tasks in {len(chunks)} chunks with n_jobs={jobs}")
            results = []
            with ProcessPoolExecutor(max_workers=jobs, initializer=_init_delta_dev_worker, initargs=(worker_ctx,)) as ex:
                futures = [ex.submit(_permute_delta_dev_chunk, chunk) for chunk in chunks]
                for fut in as_completed(futures):
                    results.append(fut.result())

        for result in results:
            idx = np.asarray(result["indices"], dtype=np.int32)
            c_raw_by_model[:, idx] = np.asarray(result["c_raw"], dtype=np.int64)
            max_by_model = np.maximum(max_by_model, np.asarray(result["max_stats"], dtype=np.float64))
            perm_r_by_model[:, idx, :] = np.asarray(result["perm_r"], dtype=np.float32)

        denom = float(int(n_perm) + 1)
        ok = np.isfinite(r_obs_by_model)
        p_raw_by_model[ok] = (c_raw_by_model[ok] + 1.0) / denom
        for mi in range(len(MODEL_NAMES)):
            valid_max = np.isfinite(max_by_model[mi])
            for ri in range(n_rois):
                obs = float(r_obs_by_model[mi, ri])
                if np.isfinite(obs):
                    obs_stat = abs(obs) if tail_mode == "two-sided" else obs
                    p_fwer_by_model[mi, ri] = (float(np.sum(max_by_model[mi, valid_max] >= obs_stat)) + 1.0) / denom
    elif mode == "fdr_only":
        max_by_model = np.full((len(MODEL_NAMES), int(n_perm) if int(n_perm) > 0 else 0), np.nan, dtype=np.float64)
        perm_r_by_model = np.full((len(MODEL_NAMES), n_rois, int(n_perm) if int(n_perm) > 0 else 0), np.nan, dtype=np.float32)
        from scipy.stats import t

        ok = np.isfinite(r_obs_by_model)
        if n_pairs <= 2:
            raise ValueError("Too few subject pairs for fdr_only parametric p-values")
        dfree = float(n_pairs - 2)
        r = np.clip(r_obs_by_model[ok].astype(np.float64), -0.999999, 0.999999)
        t_stat = r * np.sqrt(dfree / np.maximum(1.0 - r * r, 1e-12))
        if tail_mode == "two-sided":
            log_p = np.log(2.0) + t.logsf(np.abs(t_stat), df=dfree)
        else:
            log_p = t.logsf(t_stat, df=dfree)
        min_log = float(np.log(np.finfo(np.float64).tiny))
        p_raw_by_model[ok] = np.minimum(1.0, np.exp(np.maximum(log_p, min_log)))
    else:
        raise ValueError(f"Unsupported correction_mode: {correction_mode}")

    rows = []
    for mi, model in enumerate(MODEL_NAMES):
        for ri, roi in enumerate(rois):
            rows.append(
                {
                    "roi": str(roi),
                    "model": str(model),
                    "r_obs": float(r_obs_by_model[mi, ri]),
                    "p_value": float(p_raw_by_model[mi, ri]),
                    "p_perm_one_tailed": float(p_raw_by_model[mi, ri]),
                    "p_fwer_model_wise": float(p_fwer_by_model[mi, ri]),
                    "tail": str(tail_mode),
                    "n_subjects": int(n_sub),
                    "n_pairs": int(n_pairs),
                    "n_perm": int(n_perm),
                    "seed": int(seed),
                    "n_jobs": int(n_jobs),
                    "correction_mode": str(mode),
                    "normalize_models": bool(normalize_models),
                    "isc_prefix": str(prefix),
                    "assoc_method": str(assoc_method),
                    "fisher_z": bool(fisher_z_eff),
                    "delta_name": f"{reappraisal_name}_minus_{passive_name}",
                    "reappraisal_name": str(reappraisal_name),
                    "passive_name": str(passive_name),
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["model", "roi"]).reset_index(drop=True)
    out_df["p_fdr_bh_model_wise"] = np.nan
    for _, idx in out_df.groupby("model").groups.items():
        out_df.loc[idx, "p_fdr_bh_model_wise"] = bh_fdr(out_df.loc[idx, "p_value"].to_numpy(dtype=float))
    out_df["p_fdr_bh_global"] = bh_fdr(out_df["p_value"].to_numpy(dtype=float))
    mapping = _load_schaefer_mapping(schaefer_mapping_file, matrix_dir=matrix_dir)
    out_df = _add_source_name(out_df, mapping)

    out_name = str(out_condition_name) if out_condition_name is not None else f"{reappraisal_name}_minus_{passive_name}"
    out_dir = by_stim / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{OUT_PREFIX}.csv"
    out_df.to_csv(out_csv, index=False)
    pd.DataFrame({"subject": subjects, "age": ages}).to_csv(out_dir / f"{OUT_PREFIX}_subjects_sorted.csv", index=False)
    pd.DataFrame({"roi": rois}).to_csv(out_dir / f"{OUT_PREFIX}_rois.csv", index=False)
    metadata = make_metadata(
        script_name=Path(__file__).name,
        result_type="roi_isc_delta_dev_models_perm_null",
        input_files={
            "reappraisal_dir": str(reapp_dir),
            "passive_dir": str(passive_dir),
            "isc_prefix": str(prefix),
        },
        output_csv=str(out_csv),
        seed=int(seed),
        n_perm=int(n_perm),
        models=list(MODEL_NAMES),
        extra={
            "stimulus_dir_name": str(stimulus_dir_name),
            "delta_name": str(out_name),
            "reappraisal_name": str(reappraisal_name),
            "passive_name": str(passive_name),
            "assoc_method": str(assoc_method),
            "correction_mode": str(mode),
            "tail": str(tail_mode),
            "normalize_models": bool(normalize_models),
            "fisher_z": bool(fisher_z_eff),
        },
    )
    if mode == "perm_fwer_fdr" and int(n_perm) > 0:
        save_perm_null_npz(
            out_dir / "roi_isc_delta_dev_models_perm_null.npz",
            metadata=metadata,
            obs_r=r_obs_by_model.astype(np.float32),
            perm_r=perm_r_by_model.astype(np.float32),
            perm_max_abs_r=np.asarray(max_by_model, dtype=np.float32),
            models=np.asarray(MODEL_NAMES, dtype=str),
            rois=np.asarray(rois, dtype=str),
            conditions=np.asarray([str(reappraisal_name), str(passive_name)], dtype=str),
            delta_name=np.asarray(str(out_name)),
            subjects_sorted=np.asarray(subjects, dtype=str),
            ages_sorted=np.asarray(ages, dtype=np.float32),
            p_perm_one_tailed=p_raw_by_model.astype(np.float64),
            p_fwer_model_wise=p_fwer_by_model.astype(np.float64),
        )

    summary_rows = []
    for model in MODEL_NAMES:
        sub = out_df[out_df["model"].astype(str) == str(model)]
        summary_rows.append(
            {
                "delta_name": out_name,
                "model": model,
                "n_rows": int(sub.shape[0]),
                "n_sig_pos_fwer": int(((sub["r_obs"] > 0) & (sub["p_fwer_model_wise"] <= 0.05)).sum()),
                "n_sig_neg_fwer": int(((sub["r_obs"] < 0) & (sub["p_fwer_model_wise"] <= 0.05)).sum()),
                "n_sig_pos_fdr_model_wise": int(((sub["r_obs"] > 0) & (sub["p_fdr_bh_model_wise"] <= 0.05)).sum()),
                "n_sig_neg_fdr_model_wise": int(((sub["r_obs"] < 0) & (sub["p_fdr_bh_model_wise"] <= 0.05)).sum()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / f"{OUT_PREFIX}_summary.csv", index=False)
    print(f"Saved: {out_csv}")
    return out_df


def main() -> None:
    args = parse_args()
    run(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        reappraisal_name=str(args.reappraisal_name),
        passive_name=str(args.passive_name),
        out_condition_name=args.out_condition_name,
        isc_method=str(args.isc_method),
        isc_prefix=args.isc_prefix,
        assoc_method=str(args.assoc_method),
        correction_mode=str(args.correction_mode),
        tail=str(args.tail),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        n_jobs=int(args.n_jobs),
        normalize_models=bool(args.normalize_models),
        fisher_z_enabled=bool(args.fisher_z),
        schaefer_mapping_file=args.schaefer_mapping_file,
    )


if __name__ == "__main__":
    main()
