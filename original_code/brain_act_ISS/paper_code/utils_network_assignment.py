#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils_network_assignment.py

Schaefer 200 7Networks ROI → network mapping & Tian S2 32 subcortical ROI → structure mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

DEFAULT_DISTANCE_THRESHOLD_MM: float = 20.0

SCHAEFER_7NETWORKS: Dict[str, str] = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Salience/Ventral Attention",
    "Limbic": "Limbic",
    "Cont": "Frontoparietal Control",
    "Default": "Default Mode Network",
}

_LEFT_NETWORK_RANGES: List[tuple[str, int, int]] = [
    ("Vis", 1, 13),
    ("SomMot", 14, 30),
    ("DorsAttn", 31, 43),
    ("SalVentAttn", 44, 52),
    ("Limbic", 53, 59),
    ("Cont", 60, 72),
    ("Default", 73, 100),
]

_RIGHT_NETWORK_RANGES: List[tuple[str, int, int]] = [
    ("Vis", 1, 13),
    ("SomMot", 14, 30),
    ("DorsAttn", 31, 43),
    ("SalVentAttn", 44, 53),
    ("Limbic", 54, 59),
    ("Cont", 60, 72),
    ("Default", 73, 100),
]


def get_schaefer200_network_labels() -> pd.DataFrame:
    rows: list[dict] = []
    for hemi, prefix, ranges in [
        ("L", "L", _LEFT_NETWORK_RANGES),
        ("R", "R", _RIGHT_NETWORK_RANGES),
    ]:
        idx_to_net: Dict[int, str] = {}
        for net, start, end in ranges:
            for i in range(start, end + 1):
                idx_to_net[i] = net
        for i in range(1, 101):
            roi = f"{prefix}_{i}"
            net = idx_to_net[i]
            rows.append(
                {
                    "roi": roi,
                    "roi_id": i,
                    "hemi": hemi,
                    "network": net,
                    "network_full": SCHAEFER_7NETWORKS[net],
                }
            )
    return pd.DataFrame(rows)


TIAN_S2_LABELS: Dict[str, str] = {
    "V_1": "HIP-head",
    "V_2": "HIP-body",
    "V_3": "AMY",
    "V_4": "THA-anterior",
    "V_5": "THA-posterior",
    "V_6": "THA-lateral",
    "V_7": "THA-medial",
    "V_8": "CAU-head",
    "V_9": "CAU-body",
    "V_10": "PUT-anterior",
    "V_11": "PUT-posterior",
    "V_12": "NAc",
    "V_13": "GP-external",
    "V_14": "GP-internal",
    "V_15": "THA-ventral",
    "V_16": "THA-intralaminar",
    "V_17": "HIP-head",
    "V_18": "HIP-body",
    "V_19": "AMY",
    "V_20": "THA-anterior",
    "V_21": "THA-posterior",
    "V_22": "THA-lateral",
    "V_23": "THA-medial",
    "V_24": "CAU-head",
    "V_25": "CAU-body",
    "V_26": "PUT-anterior",
    "V_27": "PUT-posterior",
    "V_28": "NAc",
    "V_29": "GP-external",
    "V_30": "GP-internal",
    "V_31": "THA-ventral",
    "V_32": "THA-intralaminar",
}

_STRUCTURE_TO_GROUP: Dict[str, str] = {
    "HIP-head": "Hippocampus",
    "HIP-body": "Hippocampus",
    "AMY": "Amygdala",
    "THA-anterior": "Thalamus",
    "THA-posterior": "Thalamus",
    "THA-lateral": "Thalamus",
    "THA-medial": "Thalamus",
    "THA-ventral": "Thalamus",
    "THA-intralaminar": "Thalamus",
    "CAU-head": "Caudate",
    "CAU-body": "Caudate",
    "PUT-anterior": "Putamen",
    "PUT-posterior": "Putamen",
    "NAc": "NAc",
    "GP-external": "GP",
    "GP-internal": "GP",
}


def get_tian_s2_labels() -> pd.DataFrame:
    rows: list[dict] = []
    for vid in range(1, 33):
        roi = f"V_{vid}"
        structure = TIAN_S2_LABELS[roi]
        hemi = "L" if vid <= 16 else "R"
        rows.append(
            {
                "roi": roi,
                "structure": structure,
                "hemi": hemi,
                "structure_group": _STRUCTURE_TO_GROUP[structure],
            }
        )
    return pd.DataFrame(rows)


def get_all_roi_info(roi_set: int = 232) -> pd.DataFrame:
    cortical = get_schaefer200_network_labels()
    cortical_out = pd.DataFrame(
        {
            "roi": cortical["roi"],
            "region_type": "cortical",
            "network_or_structure": cortical["network"],
            "hemi": cortical["hemi"],
        }
    )

    if roi_set == 200:
        return cortical_out.reset_index(drop=True)

    subcortical = get_tian_s2_labels()
    subcortical_out = pd.DataFrame(
        {
            "roi": subcortical["roi"],
            "region_type": "subcortical",
            "network_or_structure": subcortical["structure_group"],
            "hemi": subcortical["hemi"],
        }
    )
    return pd.concat([cortical_out, subcortical_out], ignore_index=True)


def get_roi_network_map(roi_set: int = 232) -> dict[str, str]:
    df = get_all_roi_info(roi_set=roi_set)
    return dict(zip(df["roi"], df["network_or_structure"]))


def get_sensorimotor_association_order() -> list[str]:
    return ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


SCAN_REGIONS: Dict[str, tuple[float, float, float]] = {
    "IE_superior_L": (-19.0, -34.0, 59.0),
    "IE_superior_R": (20.0, -31.0, 58.0),
    "IE_middle_L": (-38.0, -18.0, 44.0),
    "IE_middle_R": (40.0, -15.0, 43.0),
    "IE_inferior_L": (-54.0, -3.0, 14.0),
    "IE_inferior_R": (56.0, -1.0, 16.0),
    "SMA_L": (-5.0, -7.0, 52.0),
    "SMA_R": (5.0, -5.0, 49.0),
    "preSMA_dACC_L": (-7.0, 1.0, 36.0),
    "preSMA_dACC_R": (6.0, 3.0, 36.0),
    "Putamen_post_L": (-28.0, -5.0, -1.0),
    "Putamen_post_R": (28.0, -9.0, 3.0),
    "Thalamus_CM_L": (-10.0, -21.0, 2.0),
    "Thalamus_CM_R": (12.0, -20.0, 3.0),
}

SCAN_FUNCTIONAL_GROUPS: Dict[str, List[str]] = {
    "inter_effector": [
        "IE_superior_L", "IE_superior_R",
        "IE_middle_L", "IE_middle_R",
        "IE_inferior_L", "IE_inferior_R",
    ],
    "medial_motor": [
        "SMA_L", "SMA_R",
        "preSMA_dACC_L", "preSMA_dACC_R",
    ],
    "subcortical": [
        "Putamen_post_L", "Putamen_post_R",
        "Thalamus_CM_L", "Thalamus_CM_R",
    ],
}

_SCAN_SUBCORTICAL_ROIS: Dict[str, List[str]] = {
    "Putamen_post_L": ["V_11"],
    "Putamen_post_R": ["V_27"],
    "Thalamus_CM_L": ["V_7", "V_16"],
    "Thalamus_CM_R": ["V_23", "V_32"],
}


def _schaefer200_centroids_nilearn() -> Dict[str, tuple[float, float, float]]:
    try:
        from nilearn import datasets, image
    except ImportError:
        raise ImportError("nilearn is required: pip install nilearn")

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=200, yeo_networks=7, resolution_mm=1
    )
    atlas_img = image.load_img(atlas["maps"])
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine

    centroids: Dict[str, tuple[float, float, float]] = {}
    for parcel_id in range(1, 201):
        voxels = np.argwhere(atlas_data == parcel_id)
        if len(voxels) == 0:
            continue
        centroid_vox = voxels.mean(axis=0)
        mni = affine @ np.append(centroid_vox, 1.0)
        if parcel_id <= 100:
            roi = f"L_{parcel_id}"
        else:
            # Keep consistent with Schaefer ROI naming used elsewhere in this repo: R_1..R_100
            roi = f"R_{parcel_id - 100}"
        centroids[roi] = (float(mni[0]), float(mni[1]), float(mni[2]))
    return centroids


def _schaefer200_centroids_from_nifti(atlas_nifti: Union[str, Path]) -> Dict[str, tuple[float, float, float]]:
    """
    Compute Schaefer200 centroids from a local NIfTI atlas.

    Assumptions:
    - Atlas voxel values are integer labels 1..200.
    - Labels 1..100 correspond to left hemisphere parcels (L_1..L_100),
      and 101..200 correspond to right hemisphere parcels (R_1..R_100).
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required for local NIfTI atlas: pip install nibabel")

    atlas_img = nib.load(str(atlas_nifti))
    atlas_data = np.asarray(atlas_img.get_fdata())
    affine = np.asarray(atlas_img.affine, dtype=np.float64)

    centroids: Dict[str, tuple[float, float, float]] = {}
    for parcel_id in range(1, 201):
        voxels = np.argwhere(atlas_data == float(parcel_id))
        if int(voxels.shape[0]) == 0:
            continue
        centroid_vox = voxels.mean(axis=0)
        mni = affine @ np.append(centroid_vox, 1.0)
        roi = f"L_{parcel_id}" if parcel_id <= 100 else f"R_{parcel_id - 100}"
        centroids[roi] = (float(mni[0]), float(mni[1]), float(mni[2]))
    return centroids


def _load_centroids_from_csv(path: Union[str, Path]) -> Dict[str, tuple[float, float, float]]:
    """
    Load centroids from a CSV file.

    Supported schemas:
    - columns: roi, x, y, z
    - columns: roi, mni_x, mni_y, mni_z
    """
    df = pd.read_csv(Path(path))
    if "roi" not in df.columns:
        raise ValueError(f"Centroid CSV missing required column 'roi': {path}")

    if {"x", "y", "z"}.issubset(df.columns):
        x_col, y_col, z_col = "x", "y", "z"
    elif {"mni_x", "mni_y", "mni_z"}.issubset(df.columns):
        x_col, y_col, z_col = "mni_x", "mni_y", "mni_z"
    else:
        raise ValueError(
            f"Centroid CSV must include either (x,y,z) or (mni_x,mni_y,mni_z): {path}"
        )

    centroids: Dict[str, tuple[float, float, float]] = {}
    for _, row in df.iterrows():
        roi = str(row["roi"])
        centroids[roi] = (float(row[x_col]), float(row[y_col]), float(row[z_col]))
    return centroids


def _load_centroids_from_npy(path: Union[str, Path]) -> Dict[str, tuple[float, float, float]]:
    """
    Load centroids from a NPY file.

    Supported formats:
    - dict-like object saved via np.save(..., allow_pickle=True) containing {roi: (x,y,z)}
    - numpy array of shape (200, 3) or (N, 3) where N>=200:
      0..99 -> L_1..L_100, 100..199 -> R_1..R_100
    """
    arr = np.load(str(path), allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        obj = arr.item()
        if isinstance(obj, dict):
            centroids: Dict[str, tuple[float, float, float]] = {}
            for k, v in obj.items():
                vv = np.asarray(v, dtype=np.float64).reshape(-1)
                if vv.size != 3:
                    raise ValueError(f"Invalid centroid for key {k}: expected 3 values")
                centroids[str(k)] = (float(vv[0]), float(vv[1]), float(vv[2]))
            return centroids

    arr_f = np.asarray(arr, dtype=np.float64)
    if arr_f.ndim != 2 or arr_f.shape[1] != 3:
        raise ValueError(f"Unsupported centroid NPY format (expect dict or Nx3 array): {path}")
    if arr_f.shape[0] < 200:
        raise ValueError(f"Centroid NPY array must have at least 200 rows: {path}")

    centroids = {}
    for i in range(200):
        x, y, z = arr_f[i, 0], arr_f[i, 1], arr_f[i, 2]
        roi = f"L_{i + 1}" if i < 100 else f"R_{i - 99}"
        centroids[roi] = (float(x), float(y), float(z))
    return centroids


def load_schaefer200_centroids(
    atlas_source: str = "nilearn",
    atlas_nifti: Optional[Union[str, Path]] = None,
    centroids_file: Optional[Union[str, Path]] = None,
) -> Dict[str, tuple[float, float, float]]:
    """
    Unified entry for Schaefer200 centroid loading.

    Parameters
    - atlas_source:
      - "nilearn": download via nilearn (may require internet)
      - "nifti": compute from local NIfTI atlas (requires nibabel)
      - "centroids": load from centroid CSV/NPY
      - "auto": prefer centroids_file > atlas_nifti > nilearn
    - atlas_nifti: local Schaefer2018 200 atlas NIfTI path (nii/nii.gz)
    - centroids_file: centroid cache file (csv or npy)
    """
    src = str(atlas_source).lower().strip()
    if src == "auto":
        if centroids_file is not None:
            src = "centroids"
        elif atlas_nifti is not None:
            src = "nifti"
        else:
            src = "nilearn"

    if src == "centroids":
        if centroids_file is None:
            raise ValueError("atlas_source='centroids' requires centroids_file")
        p = Path(centroids_file)
        if p.suffix.lower() == ".csv":
            return _load_centroids_from_csv(p)
        if p.suffix.lower() == ".npy":
            return _load_centroids_from_npy(p)
        raise ValueError(f"Unsupported centroids_file extension (expect .csv/.npy): {p}")

    if src in ("nifti", "nii"):
        if atlas_nifti is None:
            raise ValueError("atlas_source='nifti' requires atlas_nifti")
        return _schaefer200_centroids_from_nifti(atlas_nifti)

    if src == "nilearn":
        return _schaefer200_centroids_nilearn()

    raise ValueError(f"Unsupported atlas_source: {atlas_source}")


def get_scan_roi_mapping(
    atlas_source: str = "nilearn",
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD_MM,
    atlas_nifti: Optional[Union[str, Path]] = None,
    centroids_file: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    centroids = load_schaefer200_centroids(
        atlas_source=atlas_source,
        atlas_nifti=atlas_nifti,
        centroids_file=centroids_file,
    )

    roi_info = get_all_roi_info(roi_set=232)
    roi_net_map = get_roi_network_map(roi_set=232)

    rows: list[dict] = []
    for scan_name, (sx, sy, sz) in SCAN_REGIONS.items():
        best_dist = float("inf")
        best_roi = ""
        for roi, (cx, cy, cz) in centroids.items():
            d = float(np.sqrt((cx - sx) ** 2 + (cy - sy) ** 2 + (cz - sz) ** 2))
            if d < best_dist:
                best_dist = d
                best_roi = roi

        if scan_name in _SCAN_SUBCORTICAL_ROIS:
            sub_rois = _SCAN_SUBCORTICAL_ROIS[scan_name]
            for sr in sub_rois:
                rows.append({
                    "scan_region": scan_name,
                    "nearest_roi": sr,
                    "distance_mm": 0.0,
                    "roi_network": roi_net_map.get(sr, ""),
                    "is_cortical": False,
                    "within_threshold": True,
                })

        rows.append({
            "scan_region": scan_name,
            "nearest_roi": best_roi,
            "distance_mm": round(best_dist, 1),
            "roi_network": roi_net_map.get(best_roi, ""),
            "is_cortical": True,
            "within_threshold": best_dist <= distance_threshold,
        })

    return pd.DataFrame(rows)


def get_scan_cortical_rois(
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD_MM,
    atlas_source: str = "nilearn",
    atlas_nifti: Optional[Union[str, Path]] = None,
    centroids_file: Optional[Union[str, Path]] = None,
) -> List[str]:
    df = get_scan_roi_mapping(
        atlas_source=atlas_source,
        distance_threshold=distance_threshold,
        atlas_nifti=atlas_nifti,
        centroids_file=centroids_file,
    )
    df = df[(df["is_cortical"]) & (df["within_threshold"])].copy()
    return sorted(set(df["nearest_roi"].tolist()))


def get_scan_subcortical_rois() -> List[str]:
    rois: List[str] = []
    for sub_list in _SCAN_SUBCORTICAL_ROIS.values():
        rois.extend(sub_list)
    return sorted(set(rois))


def get_scan_all_rois(
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD_MM,
    atlas_source: str = "nilearn",
    atlas_nifti: Optional[Union[str, Path]] = None,
    centroids_file: Optional[Union[str, Path]] = None,
) -> List[str]:
    cortical = get_scan_cortical_rois(
        distance_threshold=distance_threshold,
        atlas_source=atlas_source,
        atlas_nifti=atlas_nifti,
        centroids_file=centroids_file,
    )
    return sorted(set(cortical + get_scan_subcortical_rois()))


def get_scan_mapping_quality_report(
    atlas_source: str = "nilearn",
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD_MM,
    atlas_nifti: Optional[Union[str, Path]] = None,
    centroids_file: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a QC report for SCAN->ROI mapping.

    Returns
    - detail_df: one row per mapping (SCAN region -> nearest ROI; includes fixed subcortical rows)
    - summary_df: distribution summary (overall + by scan_group + cortical-only)
    """
    mapping_df = get_scan_roi_mapping(
        atlas_source=atlas_source,
        distance_threshold=distance_threshold,
        atlas_nifti=atlas_nifti,
        centroids_file=centroids_file,
    ).copy()

    region_to_group: Dict[str, str] = {}
    for grp, regions in SCAN_FUNCTIONAL_GROUPS.items():
        for r in regions:
            region_to_group[str(r)] = str(grp)

    mapping_df["scan_group"] = mapping_df["scan_region"].map(lambda x: region_to_group.get(str(x), "unknown"))
    mapping_df["distance_threshold_mm"] = float(distance_threshold)
    mapping_df["atlas_source"] = str(atlas_source)
    mapping_df["atlas_nifti"] = str(atlas_nifti) if atlas_nifti is not None else ""
    mapping_df["centroids_file"] = str(centroids_file) if centroids_file is not None else ""

    def _summarize(df: pd.DataFrame, label: str) -> Dict[str, object]:
        dists = df["distance_mm"].to_numpy(dtype=np.float64)
        within = df["within_threshold"].to_numpy(dtype=bool)
        dists_ok = dists[np.isfinite(dists)]
        out: Dict[str, object] = {
            "scope": label,
            "n_rows": int(df.shape[0]),
            "n_within_threshold": int(within.sum()),
            "pct_within_threshold": float(within.mean() * 100.0) if df.shape[0] > 0 else float("nan"),
            "distance_mean_mm": float(np.nanmean(dists_ok)) if dists_ok.size else float("nan"),
            "distance_median_mm": float(np.nanmedian(dists_ok)) if dists_ok.size else float("nan"),
            "distance_p95_mm": float(np.nanpercentile(dists_ok, 95)) if dists_ok.size else float("nan"),
            "distance_max_mm": float(np.nanmax(dists_ok)) if dists_ok.size else float("nan"),
        }
        return out

    summary_rows: List[Dict[str, object]] = []
    summary_rows.append(_summarize(mapping_df, "all_rows"))
    cortical_df = mapping_df[mapping_df["is_cortical"]].copy()
    summary_rows.append(_summarize(cortical_df, "cortical_only"))

    for grp in sorted(mapping_df["scan_group"].unique().tolist()):
        df_g = mapping_df[mapping_df["scan_group"] == grp].copy()
        summary_rows.append(_summarize(df_g, f"scan_group:{grp}"))
        df_g_c = df_g[df_g["is_cortical"]].copy()
        if not df_g_c.empty:
            summary_rows.append(_summarize(df_g_c, f"scan_group:{grp}:cortical_only"))

    summary_df = pd.DataFrame(summary_rows)
    return mapping_df, summary_df


def _parse_cli_args() -> "argparse.Namespace":
    import argparse

    p = argparse.ArgumentParser(
        description="Write a SCAN-to-ROI mapping QC report (offline atlas or centroids supported) to scan_mapping_qc.csv"
    )
    p.add_argument("--matrix-dir", type=Path, default=None, help="If set, write by default to matrix_dir/gradient_analysis/")
    p.add_argument("--out", type=Path, default=None, help="Output CSV. Default: gradient_analysis/scan_mapping_qc.csv or ./scan_mapping_qc.csv")
    p.add_argument("--out-summary", type=Path, default=None, help="Optional summary CSV. Default: scan_mapping_qc_summary.csv")
    p.add_argument(
        "--atlas-source",
        type=str,
        default="nilearn",
        choices=["nilearn", "nifti", "centroids", "auto"],
        help="Schaefer-200 centroid source: nilearn (download), nifti (local), centroids (CSV/NPY), or auto (prefer local)",
    )
    p.add_argument("--atlas-nifti", type=Path, default=None, help="Local Schaefer-200 atlas NIfTI (.nii or .nii.gz)")
    p.add_argument("--centroids-file", type=Path, default=None, help="Local centroid cache (.csv or .npy)")
    p.add_argument("--distance-threshold", type=float, default=float(DEFAULT_DISTANCE_THRESHOLD_MM))
    return p.parse_args()


def main() -> None:
    args = _parse_cli_args()

    out_dir = Path(".")
    if args.matrix_dir is not None:
        out_dir = Path(args.matrix_dir) / "gradient_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out) if args.out is not None else (out_dir / "scan_mapping_qc.csv")
    summary_path = Path(args.out_summary) if args.out_summary is not None else (out_dir / "scan_mapping_qc_summary.csv")

    detail_df, summary_df = get_scan_mapping_quality_report(
        atlas_source=str(args.atlas_source),
        distance_threshold=float(args.distance_threshold),
        atlas_nifti=args.atlas_nifti,
        centroids_file=args.centroids_file,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(out_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Minimal stdout summary for quick checks in pipelines.
    cortical_only = summary_df[summary_df["scope"] == "cortical_only"]
    if not cortical_only.empty:
        row0 = cortical_only.iloc[0].to_dict()
        print(
            f"[scan_mapping_qc] cortical within_threshold: "
            f"{row0.get('n_within_threshold')}/{row0.get('n_rows')} "
            f"({row0.get('pct_within_threshold'):.1f}%), "
            f"mean={row0.get('distance_mean_mm'):.1f}mm, max={row0.get('distance_max_mm'):.1f}mm"
        )
    print(f"[scan_mapping_qc] detail saved: {out_path}")
    print(f"[scan_mapping_qc] summary saved: {summary_path}")


if __name__ == "__main__":
    main()
