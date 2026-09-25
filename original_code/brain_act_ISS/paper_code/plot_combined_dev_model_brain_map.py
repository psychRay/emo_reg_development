#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create combined three-model brain maps for Passive_Emo and Reappraisal."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _patch_numpy_for_legacy_nibabel() -> None:
    if not hasattr(np, "sctypes"):
        np.sctypes = {  # type: ignore[attr-defined]
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.bytes_, np.str_, np.void, np.object_],
        }
    if not hasattr(np, "maximum_sctype"):
        def _maximum_sctype(dtype):
            dt = np.dtype(dtype)
            if dt.kind == "f":
                return np.longdouble
            if dt.kind == "c":
                return np.clongdouble
            if dt.kind == "u":
                return np.uint64
            if dt.kind in {"i", "b"}:
                return np.int64
            return dt.type

        np.maximum_sctype = _maximum_sctype  # type: ignore[attr-defined]


_patch_numpy_for_legacy_nibabel()

if __package__ in {None, ""}:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from plotting_style import setup_publication_style  # noqa: E402
    from plot_brain_surface_vol import ATLAS_SURF_L, ATLAS_SURF_R, ATLAS_VOL  # noqa: E402
    try:
        from utils_network_assignment import get_tian_s2_labels  # noqa: E402
    except ModuleNotFoundError:
        get_tian_s2_labels = None  # type: ignore[assignment]
else:
    from .plotting_style import setup_publication_style
    from .plot_brain_surface_vol import ATLAS_SURF_L, ATLAS_SURF_R, ATLAS_VOL
    try:
        from .utils_network_assignment import get_tian_s2_labels
    except ModuleNotFoundError:
        get_tian_s2_labels = None  # type: ignore[assignment]


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
DEFAULT_SCHAEFER_MAPPING_NAME = "schaefer2018_200parcels_7networks_order_mapping.csv"
DEFAULT_FSLR_INFLATED_L = Path("/public/home/dingrui/tools/masks_atlas/fsaverage.L.inflated.32k_fs_LR.surf.gii")
DEFAULT_FSLR_INFLATED_R = Path("/public/home/dingrui/tools/masks_atlas/fsaverage.R.inflated.32k_fs_LR.surf.gii")
DEFAULT_FSLR_BG_L = Path("/public/home/dingrui/tools/masks_atlas/fsaverage.L.sulc.32k_fs_LR.shape.gii")
DEFAULT_FSLR_BG_R = Path("/public/home/dingrui/tools/masks_atlas/fsaverage.R.sulc.32k_fs_LR.shape.gii")
MODEL_BITS = {"M_nn": 1, "M_conv": 2, "M_div": 4}
INTERACTION_EFFECTS = ("brain_x_M_nn", "brain_x_M_conv", "brain_x_M_div")
LEGACY_INTERACTION_EFFECTS = {
    "brain_x_M_nn": "interaction_M_nn",
    "brain_x_M_conv": "interaction_M_conv",
    "brain_x_M_div": "interaction_M_div",
}
INTERACTION_EFFECT_TO_MODEL = {
    "brain_x_M_nn": "M_nn",
    "brain_x_M_conv": "M_conv",
    "brain_x_M_div": "M_div",
}
BIT_COLORS = {
    1: "#e74c3c",  # M_nn
    2: "#2ca25f",  # M_conv
    4: "#3182bd",  # M_div
    3: "#f1c40f",  # nn + conv
    5: "#9b59b6",  # nn + div
    6: "#00bcd4",  # conv + div
    7: "#f7f7f7",  # all
}
BIT_LABELS = {
    1: "M_nn",
    2: "M_conv",
    4: "M_div",
    3: "M_nn + M_conv",
    5: "M_nn + M_div",
    6: "M_conv + M_div",
    7: "M_nn + M_conv + M_div",
}
INTERACTION_BIT_LABELS = {
    1: "brain x M_nn",
    2: "brain x M_conv",
    4: "brain x M_div",
    3: "brain x M_nn + brain x M_conv",
    5: "brain x M_nn + brain x M_div",
    6: "brain x M_conv + brain x M_div",
    7: "brain x M_nn + brain x M_conv + brain x M_div",
}
MESH_VERTICES = {
    "fsaverage3": 642,
    "fsaverage4": 2562,
    "fsaverage5": 10242,
    "fsaverage6": 40962,
    "fslr32k": 32492,
}
TIAN_S2_LABELS = {
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
STRUCTURE_TO_GROUP = {
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine Passive_Emo/Reappraisal significant ROIs into three-model brain maps")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_emotion")
    p.add_argument("--conditions", nargs="+", default=["Passive_Emo", "Reappraisal"])
    p.add_argument("--models", nargs="+", default=["M_nn", "M_conv", "M_div"])
    p.add_argument("--result-file", type=str, default="roi_isc_dev_models_perm_fwer.csv")
    p.add_argument("--result-csv", type=Path, default=None, help="Optional joint regression CSV. When provided, only brain x age-model interaction effects are plotted.")
    p.add_argument("--input-mode", type=str, default="auto", choices=("auto", "dev_models", "interaction"))
    p.add_argument("--p-col", type=str, default="p_fdr_bh_model_wise")
    p.add_argument("--p-prefix", type=str, default="p_fdr_perm", help="P-value prefix for interaction mode, e.g. p_fdr_perm or p_fwer.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--positive-only",
        dest="positive_only",
        action="store_true",
        default=True,
        help="Keep only significant ROIs with positive r_obs/beta. This is the default.",
    )
    p.add_argument(
        "--include-negative",
        dest="positive_only",
        action="store_false",
        help="Include both positive and negative significant ROIs.",
    )
    p.add_argument("--schaefer-mapping-file", type=Path, default=None)
    p.add_argument("--atlas-surf-left", type=Path, default=None, help="Override Schaefer left surface label.gii (defaults to plot_brain_surface_vol.ATLAS_SURF_L).")
    p.add_argument("--atlas-surf-right", type=Path, default=None, help="Override Schaefer right surface label.gii.")
    p.add_argument("--atlas-volume", type=Path, default=None, help="Override Tian subcortex volume atlas nii.")
    p.add_argument("--surface-left", type=Path, default=DEFAULT_FSLR_INFLATED_L, help="Left fsLR surface geometry used to render exported .func.gii maps.")
    p.add_argument("--surface-right", type=Path, default=DEFAULT_FSLR_INFLATED_R, help="Right fsLR surface geometry used to render exported .func.gii maps.")
    p.add_argument("--surface-bg-left", type=Path, default=None, help="Optional left fsLR sulc/curv/shape GIFTI used as the gray surface background texture.")
    p.add_argument("--surface-bg-right", type=Path, default=None, help="Optional right fsLR sulc/curv/shape GIFTI used as the gray surface background texture.")
    p.add_argument(
        "--use-geometry-bg-fallback",
        dest="use_geometry_bg_fallback",
        action="store_true",
        default=True,
        help="If no sulc/curv/shape file is found, synthesize a low-contrast background texture from surface geometry. This is the default.",
    )
    p.add_argument(
        "--no-geometry-bg-fallback",
        dest="use_geometry_bg_fallback",
        action="store_false",
        help="Disable geometry-derived surface background fallback; may make the surface look flat gray.",
    )
    p.add_argument("--enhance-surface-bg", action="store_true", help="Increase contrast of sulc/curv background textures. Off by default because it can look much darker than Workbench/SurfIce.")
    p.add_argument("--surface-bg-contrast", type=float, default=1.4, help="Contrast strength used only with --enhance-surface-bg.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--maps-only", action="store_true", help="Export Figure 6c surface and volume map data without rendering the brain figure.")
    return p.parse_args()


def _load_significant(args: argparse.Namespace) -> pd.DataFrame:
    """Load FULL dev-model long data (all ROIs/models) with is_sig + value columns."""
    rows = []
    for cond in [str(c) for c in args.conditions]:
        path = Path(args.matrix_dir) / str(args.stimulus_dir_name) / cond / str(args.result_file)
        if not path.exists():
            print(f"[SKIP] missing result file: {path}")
            continue
        df = pd.read_csv(path)
        if str(args.p_col) not in df.columns:
            raise KeyError(f"{path} missing p column: {args.p_col}")
        d = df[df["model"].astype(str).isin(args.models)].copy()
        d["condition"] = cond
        d["value"] = pd.to_numeric(d.get("r_obs", np.nan), errors="coerce")
        p = pd.to_numeric(d[str(args.p_col)], errors="coerce")
        d["p_value"] = p
        sig = p < float(args.alpha)
        if bool(args.positive_only):
            sig = sig & (d["value"] > 0)
        d["is_sig"] = sig.fillna(False)
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _interaction_p_col(effect: str, prefix: str) -> str:
    return f"{prefix}_{effect}_model_wise"


def _interaction_value(row: pd.Series, stem: str, effect: str) -> float:
    for candidate in (effect, LEGACY_INTERACTION_EFFECTS.get(effect, effect)):
        col = f"{stem}_{candidate}"
        if col in row.index:
            return float(row.get(col, np.nan))
    return float("nan")


def _load_interaction_significant(args: argparse.Namespace) -> pd.DataFrame:
    """Load FULL interaction long data (all ROIs/models) with is_sig + value(beta) columns."""
    path = Path(args.result_csv) if args.result_csv is not None else Path(args.matrix_dir) / str(args.stimulus_dir_name) / "Reappraisal_minus_Passive_Emo" / str(args.result_file)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find interaction result CSV: {path}")
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        roi = str(row.get("roi", ""))
        if not roi:
            continue
        for effect in INTERACTION_EFFECTS:
            p_val = float("nan")
            p_col_used = ""
            for candidate in (effect, LEGACY_INTERACTION_EFFECTS.get(effect, effect)):
                p_col = _interaction_p_col(candidate, str(args.p_prefix))
                if p_col in row.index:
                    p_val = float(row.get(p_col, np.nan))
                    p_col_used = p_col
                    break
            beta = _interaction_value(row, "beta", effect)
            is_sig = bool(np.isfinite(p_val) and p_val < float(args.alpha))
            if bool(args.positive_only) and is_sig and ((not np.isfinite(beta)) or beta <= 0):
                is_sig = False
            out = {
                "condition": "interaction",
                "roi": roi,
                "model": INTERACTION_EFFECT_TO_MODEL[effect],
                "effect": effect,
                "value": beta,
                "is_sig": is_sig,
                "p_value": p_val,
                "p_col": p_col_used,
                "t": _interaction_value(row, "t", effect),
                "source_csv": str(path),
            }
            for col in ("roi_schaefer_style", "source_name", "network", "network_full", "parcel_name"):
                if col in row.index:
                    out[col] = row.get(col, "")
            rows.append(out)
    return pd.DataFrame(rows)


def _resolve_input_mode(args: argparse.Namespace) -> str:
    if str(args.input_mode) != "auto":
        return str(args.input_mode)
    if args.result_csv is not None:
        return "interaction"
    return "dev_models"


def _roi_code_map(sig: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for _, row in sig.iterrows():
        roi = str(row.get("roi", ""))
        model = str(row.get("model", ""))
        bit = MODEL_BITS.get(model, 0)
        if bit <= 0 or not roi:
            continue
        out[roi] = int(out.get(roi, 0) | bit)
    return out


def _roi_aliases(roi: str) -> set[str]:
    text = str(roi).strip()
    aliases = {text} if text else set()
    m = re.fullmatch(r"([LRV])_(\d+)", text)
    if not m:
        return aliases
    hemi, raw = m.group(1), int(m.group(2))
    if hemi == "R":
        aliases.add(f"R_{raw - 100}" if raw > 100 else f"R_{raw + 100}")
    return aliases


def _sig_roi_aliases(sig: pd.DataFrame) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for _, row in sig.iterrows():
        roi = str(row.get("roi", "")).strip()
        if not roi:
            continue
        aliases = set()
        for col in ("roi", "roi_schaefer_style", "source_name"):
            if col in row.index and pd.notna(row.get(col)):
                aliases |= _roi_aliases(str(row.get(col)))
                aliases.add(str(row.get(col)).strip())
        out.setdefault(roi, set()).update(x for x in aliases if x)
    return out


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip()).strip("_") or "condition"


def _strip_schaefer_prefix(name: str) -> str:
    return re.sub(r"^7Networks_", "", str(name).strip(), flags=re.IGNORECASE)


def _display_label_from_record(roi: str, record: dict | None) -> str:
    record = record or {}
    structure = str(record.get("structure", "") or "").strip()
    hemi = str(record.get("hemi", "") or "").strip()
    if structure and hemi:
        return f"{hemi}_{structure}"
    if structure:
        return structure
    source = str(record.get("source_name", "") or "").strip()
    if source:
        return _strip_schaefer_prefix(source)
    parcel = str(record.get("parcel_name", "") or "").strip()
    if parcel and hemi:
        return f"{hemi}_{parcel}"
    if parcel:
        return parcel
    network = str(record.get("network", "") or record.get("network_full", "") or "").strip()
    if network and hemi:
        return f"{hemi}_{network}"
    if network:
        return network
    return str(roi)


def _record_for_aliases(aliases: set[str], mapping: dict[str, dict]) -> dict:
    for alias in aliases:
        rec = mapping.get(alias)
        if rec:
            return rec
    return {}


def _row_mapping_record(row: pd.Series, mapping: dict[str, dict]) -> dict:
    record = {
        col: row.get(col, "")
        for col in ("source_name", "network", "network_full", "parcel_name", "roi_schaefer_style", "hemi", "structure", "structure_group")
        if col in row.index and pd.notna(row.get(col))
    }
    if record:
        return record
    aliases = set()
    for col in ("roi", "roi_schaefer_style", "source_name"):
        if col in row.index and pd.notna(row.get(col)):
            aliases |= _roi_aliases(str(row.get(col)))
            aliases.add(str(row.get(col)).strip())
    return _record_for_aliases({x for x in aliases if x}, mapping)


def _add_display_labels(sig: pd.DataFrame, mapping: dict[str, dict]) -> pd.DataFrame:
    out = sig.copy()
    if out.empty:
        return out
    labels = []
    for _, row in out.iterrows():
        roi = str(row.get("roi", ""))
        labels.append(_display_label_from_record(roi, _row_mapping_record(row, mapping)))
    out["display_label"] = labels
    return out


def _load_schaefer_mapping(args: argparse.Namespace) -> dict[str, dict]:
    path = Path(args.schaefer_mapping_file) if args.schaefer_mapping_file is not None else Path(args.matrix_dir) / DEFAULT_SCHAEFER_MAPPING_NAME
    out: dict[str, dict] = {}
    for _, row in _tian_s2_labels().iterrows():
        roi = str(row.get("roi", "")).strip()
        if not roi:
            continue
        out[roi] = {
            "region_type": "subcortical",
            "structure": str(row.get("structure", "")),
            "structure_group": str(row.get("structure_group", "")),
            "hemi": str(row.get("hemi", "")),
            "hemi_index": None,
            "schaefer_label_id": None,
        }
    if not path.exists():
        print(f"[mapping] Schaefer mapping not found, falling back to numeric ROI labels: {path}")
        return out
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        rec = {
            "hemi": str(row.get("hemi", "")),
            "hemi_index": int(row["hemi_index"]) if "hemi_index" in row.index and pd.notna(row.get("hemi_index")) else None,
            "schaefer_label_id": int(row["schaefer_label_id"]) if "schaefer_label_id" in row.index and pd.notna(row.get("schaefer_label_id")) else None,
        }
        for col in ("source_name", "network", "network_full", "parcel_name", "roi_schaefer_style"):
            if col in row.index and pd.notna(row.get(col)):
                rec[col] = str(row.get(col))
        for col in ("roi", "roi_schaefer_style", "source_name"):
            if col in row.index and pd.notna(row.get(col)):
                for alias in _roi_aliases(str(row.get(col))) | {str(row.get(col)).strip()}:
                    if alias:
                        out.setdefault(alias, rec)
    print(f"[mapping] Loaded Schaefer mapping: {path}")
    return out


def _fallback_tian_s2_labels() -> pd.DataFrame:
    rows = []
    for vid in range(1, 33):
        roi = f"V_{vid}"
        structure = TIAN_S2_LABELS[roi]
        rows.append(
            {
                "roi": roi,
                "structure": structure,
                "structure_group": STRUCTURE_TO_GROUP[structure],
                "hemi": "L" if vid <= 16 else "R",
            }
        )
    return pd.DataFrame(rows)


def _tian_s2_labels() -> pd.DataFrame:
    if get_tian_s2_labels is None:
        return _fallback_tian_s2_labels()
    return get_tian_s2_labels()


def _choose_label(labels: np.ndarray, candidates: list[int]) -> int | None:
    label_set = set(np.unique(np.asarray(labels, dtype=int)).astype(int).tolist())
    for cand in candidates:
        if int(cand) in label_set:
            return int(cand)
    return None


def _roi_to_space_label(aliases: set[str], mapping: dict[str, dict], surf_l: np.ndarray, surf_r: np.ndarray, vol_data: np.ndarray) -> tuple[str, int] | None:
    for alias in aliases:
        rec = mapping.get(alias)
        if rec:
            hemi = str(rec.get("hemi", ""))
            hemi_index = rec.get("hemi_index")
            schaefer_label_id = rec.get("schaefer_label_id")
            if hemi == "L" and hemi_index is not None:
                label = _choose_label(surf_l, [int(hemi_index), int(schaefer_label_id or hemi_index)])
                if label is not None:
                    return "L", label
            if hemi == "R" and hemi_index is not None:
                label = _choose_label(surf_r, [int(hemi_index), int(schaefer_label_id or hemi_index), int(hemi_index) + 100])
                if label is not None:
                    return "R", label

    for alias in aliases:
        m = re.fullmatch(r"([LRV])_(\d+)", str(alias).strip())
        if not m:
            continue
        hemi, raw = m.group(1), int(m.group(2))
        if hemi == "L":
            label = _choose_label(surf_l, [raw])
            if label is not None:
                return "L", label
        elif hemi == "R":
            candidates = [raw, raw - 100 if raw > 100 else raw + 100]
            label = _choose_label(surf_r, candidates)
            if label is not None:
                return "R", label
        else:
            label = _choose_label(vol_data, [raw])
            if label is not None:
                return "V", label
    return None


def _save_gifti(path: Path, arr: np.ndarray, intent: str = "NIFTI_INTENT_ESTIMATE") -> None:
    import nibabel as nib

    gi = nib.gifti.GiftiImage(
        darrays=[nib.gifti.GiftiDataArray(np.asarray(arr, dtype=np.float32), intent=nib.nifti1.intent_codes[intent])]
    )
    nib.save(gi, str(path))


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    text = str(hex_color).strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color}")
    return (
        int(text[0:2], 16) / 255.0,
        int(text[2:4], 16) / 255.0,
        int(text[4:6], 16) / 255.0,
        float(alpha),
    )


def _save_label_gifti(path: Path, arr: np.ndarray, label_names: dict[int, str]) -> None:
    """Save integer surface labels with a Workbench-readable label table."""
    import nibabel as nib

    data = np.rint(np.asarray(arr, dtype=float)).astype(np.int32)
    darray = nib.gifti.GiftiDataArray(data, intent=nib.nifti1.intent_codes["NIFTI_INTENT_LABEL"])
    gi = nib.gifti.GiftiImage(darrays=[darray])
    table = nib.gifti.GiftiLabelTable()

    bg = nib.gifti.GiftiLabel(key=0, red=0.0, green=0.0, blue=0.0, alpha=0.0)
    bg.label = "Background"
    table.labels.append(bg)
    for code in [1, 2, 4, 3, 5, 6, 7]:
        rgba = _hex_to_rgba(BIT_COLORS[int(code)], alpha=1.0)
        lab = nib.gifti.GiftiLabel(key=int(code), red=rgba[0], green=rgba[1], blue=rgba[2], alpha=rgba[3])
        lab.label = str(label_names.get(int(code), BIT_LABELS.get(int(code), str(code))))
        table.labels.append(lab)

    gi.labeltable = table
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(gi, str(path))


def _load_atlas(args: argparse.Namespace):
    """Load Schaefer surface labels + Tian volume atlas, honoring CLI overrides."""
    import nibabel as nib
    from nilearn import image

    surf_l_path = Path(args.atlas_surf_left) if args.atlas_surf_left is not None else Path(ATLAS_SURF_L)
    surf_r_path = Path(args.atlas_surf_right) if args.atlas_surf_right is not None else Path(ATLAS_SURF_R)
    vol_path = Path(args.atlas_volume) if args.atlas_volume is not None else Path(ATLAS_VOL)
    surf_l = nib.load(str(surf_l_path)).darrays[0].data.astype(int)
    surf_r = nib.load(str(surf_r_path)).darrays[0].data.astype(int)
    vol_img = image.load_img(str(vol_path))
    vol_data = np.rint(vol_img.get_fdata()).astype(int)
    return surf_l, surf_r, vol_img, vol_data


def _apply_code_to_mask(arr: np.ndarray, mask: np.ndarray, code: int) -> None:
    if not np.any(mask):
        return
    current = np.rint(np.asarray(arr[mask], dtype=float)).astype(np.int32)
    arr[mask] = np.bitwise_or(current, int(code)).astype(np.float32)


def _resolve_roi_targets(
    rois: list[str],
    roi_aliases: dict[str, set[str]],
    mapping: dict[str, dict],
    surf_l: np.ndarray,
    surf_r: np.ndarray,
    vol_data: np.ndarray,
) -> dict[str, tuple[str, int]]:
    """Resolve each ROI name to a (space, atlas_label) target once."""
    out: dict[str, tuple[str, int]] = {}
    for roi in rois:
        aliases = roi_aliases.get(str(roi), set()) | _roi_aliases(str(roi))
        resolved = _roi_to_space_label(aliases, mapping, surf_l=surf_l, surf_r=surf_r, vol_data=vol_data)
        if resolved is not None:
            out[str(roi)] = resolved
    return out


def _fill_value_maps(
    value_by_roi: dict[str, float],
    targets: dict[str, tuple[str, int]],
    surf_l: np.ndarray,
    surf_r: np.ndarray,
    vol_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write a continuous (or binary) value into surface/volume arrays per ROI."""
    out_l = np.zeros_like(surf_l, dtype=np.float32)
    out_r = np.zeros_like(surf_r, dtype=np.float32)
    out_v = np.zeros_like(vol_data, dtype=np.float32)
    for roi, value in value_by_roi.items():
        target = targets.get(str(roi))
        if target is None or not np.isfinite(value):
            continue
        space, label = target
        if space == "L":
            out_l[surf_l == int(label)] = float(value)
        elif space == "R":
            out_r[surf_r == int(label)] = float(value)
        elif space == "V":
            out_v[vol_data == int(label)] = float(value)
    return out_l, out_r, out_v


def _save_map_triplet(out_dir: Path, prefix: str, out_l: np.ndarray, out_r: np.ndarray, vol_img, out_v: np.ndarray, intent: str) -> None:
    from nilearn import image

    _save_gifti(out_dir / f"{prefix}_surf_L.func.gii", out_l, intent=intent)
    _save_gifti(out_dir / f"{prefix}_surf_R.func.gii", out_r, intent=intent)
    image.new_img_like(vol_img, out_v).to_filename(str(out_dir / f"{prefix}_volume.nii.gz"))


def _discrete_cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = [BIT_COLORS[i] for i in range(1, 8)]
    cmap = ListedColormap(colors, name="dev_model_overlap")
    norm = BoundaryNorm(np.arange(0.5, 8.5, 1.0), cmap.N)
    return cmap, norm


def _infer_surface_mesh(n_vertices: int) -> str:
    for mesh, expected in MESH_VERTICES.items():
        if int(n_vertices) == int(expected):
            return mesh
    supported = ", ".join(f"{k}={v}" for k, v in MESH_VERTICES.items())
    raise ValueError(f"Cannot infer surface mesh from {n_vertices} vertices. Supported: {supported}")


def _is_left_surface_name(name: str) -> bool:
    low = str(name).lower()
    return any(x in low for x in ["hemi-l", "_l.", ".l.", "-l.", "_lh", ".lh", "-lh", "_left", ".left"])


def _is_right_surface_name(name: str) -> bool:
    low = str(name).lower()
    return any(x in low for x in ["hemi-r", "_r.", ".r.", "-r.", "_rh", ".rh", "-rh", "_right", ".right"])


def _find_fslr_surface_geometry(search_dirs: list[Path]) -> tuple[Path | None, Path | None]:
    if DEFAULT_FSLR_INFLATED_L.exists() and DEFAULT_FSLR_INFLATED_R.exists():
        return DEFAULT_FSLR_INFLATED_L, DEFAULT_FSLR_INFLATED_R
    keywords = ("inflated", "very_inflated", "midthickness", "pial", "white")
    roots: list[Path] = []
    for search_dir in search_dirs:
        root = Path(search_dir)
        roots.append(root)
        for parent in root.parents:
            roots.append(parent)
            if len(roots) >= 10:
                break

    left: list[tuple[int, int, str, Path]] = []
    right: list[tuple[int, int, str, Path]] = []

    def _score(path: Path, root: Path) -> tuple[int, int, str, Path]:
        name = path.name.lower()
        if "very_inflated" in name:
            pri = 1
        elif "inflated" in name:
            pri = 0
        elif "midthickness" in name:
            pri = 2
        elif "pial" in name:
            pri = 3
        elif "white" in name:
            pri = 4
        else:
            pri = 9
        try:
            depth = len(path.relative_to(root).parts)
        except ValueError:
            depth = 99
        return pri, depth, str(path), path

    seen_roots: set[Path] = set()
    for root in roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)
        if not root.exists() or not root.is_dir():
            continue
        candidates = list(root.glob("*.gii"))
        for sub in ("surf", "surface", "surfaces", "mesh", "meshes", "fslr", "fs_lr", "conte69"):
            subdir = root / sub
            if subdir.exists() and subdir.is_dir():
                candidates.extend(subdir.rglob("*.gii"))
        for path in candidates:
            low = path.name.lower()
            if ".func." in low or ".label." in low:
                continue
            if not any(k in low for k in keywords):
                continue
            item = _score(path, root)
            if _is_left_surface_name(low):
                left.append(item)
            elif _is_right_surface_name(low):
                right.append(item)
    return (sorted(left)[0][3] if left else None, sorted(right)[0][3] if right else None)


def _surface_vertex_count(surface_path: Path) -> int:
    from nilearn.surface import load_surf_mesh

    mesh = load_surf_mesh(str(surface_path))
    coords = np.asarray(mesh.coordinates if hasattr(mesh, "coordinates") else mesh[0], dtype=float)
    return int(coords.shape[0])


def _load_gifti_scalar(path: Path) -> np.ndarray:
    import nibabel as nib

    img = nib.load(str(path))
    arrays = [np.asarray(arr.data).reshape(-1) for arr in getattr(img, "darrays", [])]
    if not arrays:
        raise ValueError(f"No data arrays in surface background file: {path}")
    return np.asarray(arrays[0], dtype=float).reshape(-1)


def _is_left_bg_name(name: str) -> bool:
    low = str(name).lower()
    return any(x in low for x in [".l.", "_l.", "-l.", ".lh.", "_lh.", "hemi-l", ".left.", "_left."])


def _is_right_bg_name(name: str) -> bool:
    low = str(name).lower()
    return any(x in low for x in [".r.", "_r.", "-r.", ".rh.", "_rh.", "hemi-r", ".right.", "_right."])


def _find_surface_backgrounds(search_dirs: list[Path], n_vertices: int, explicit_left: Path | None, explicit_right: Path | None) -> tuple[Path | None, Path | None]:
    if explicit_left is not None and explicit_right is not None and Path(explicit_left).exists() and Path(explicit_right).exists():
        return Path(explicit_left), Path(explicit_right)
    if DEFAULT_FSLR_BG_L.exists() and DEFAULT_FSLR_BG_R.exists():
        return DEFAULT_FSLR_BG_L, DEFAULT_FSLR_BG_R

    roots: list[Path] = []
    for search_dir in search_dirs:
        root = Path(search_dir)
        roots.append(root)
        for parent in root.parents:
            roots.append(parent)
            if len(roots) >= 10:
                break

    left: list[tuple[int, int, str, Path]] = []
    right: list[tuple[int, int, str, Path]] = []

    def _score(path: Path, root: Path) -> tuple[int, int, str, Path]:
        name = path.name.lower()
        if "sulc" in name:
            pri = 0
        elif "curv" in name or "curvature" in name:
            pri = 1
        elif "shape" in name:
            pri = 2
        else:
            pri = 9
        try:
            depth = len(path.relative_to(root).parts)
        except ValueError:
            depth = 99
        return pri, depth, str(path), path

    seen_roots: set[Path] = set()
    for root in roots:
        if root in seen_roots or not root.exists() or not root.is_dir():
            continue
        seen_roots.add(root)
        candidates = list(root.glob("*.gii"))
        for sub in ("surf", "surface", "surfaces", "mesh", "meshes", "fslr", "fs_lr", "conte69"):
            subdir = root / sub
            if subdir.exists() and subdir.is_dir():
                candidates.extend(subdir.rglob("*.gii"))
        for path in candidates:
            low = path.name.lower()
            if ".surf." in low or ".label." in low:
                continue
            if not any(k in low for k in ("sulc", "curv", "curvature", "shape")):
                continue
            try:
                if int(_load_gifti_scalar(path).size) != int(n_vertices):
                    continue
            except Exception:
                continue
            item = _score(path, root)
            if _is_left_bg_name(low):
                left.append(item)
            elif _is_right_bg_name(low):
                right.append(item)
    return (sorted(left)[0][3] if left else None, sorted(right)[0][3] if right else None)


def _load_surface_backgrounds(search_dir: Path, n_vertices: int, explicit_left: Path | None, explicit_right: Path | None) -> tuple[np.ndarray | None, np.ndarray | None, Path | None, Path | None]:
    left, right = _find_surface_backgrounds(
        [search_dir, Path(ATLAS_SURF_L).parent, Path(ATLAS_SURF_R).parent],
        n_vertices=n_vertices,
        explicit_left=explicit_left,
        explicit_right=explicit_right,
    )
    if left is None or right is None:
        print("[surface] no fsLR sulc/curv/shape background found; surface will be flat gray. Use --surface-bg-left/--surface-bg-right.")
        return None, None, left, right
    bg_l = _load_gifti_scalar(left)
    bg_r = _load_gifti_scalar(right)
    if int(bg_l.size) != int(n_vertices) or int(bg_r.size) != int(n_vertices):
        raise ValueError(
            f"Surface background vertex count mismatch: L={left} has {bg_l.size}, R={right} has {bg_r.size}, expected {n_vertices}"
        )
    print(f"[surface] using fsLR background texture: L={left}, R={right}")
    return bg_l, bg_r, left, right


def _validate_surface_alignment(l_array: np.ndarray, r_array: np.ndarray, surface_left: Path | None, surface_right: Path | None) -> None:
    if surface_left is None or surface_right is None:
        return
    if (not Path(surface_left).exists()) or (not Path(surface_right).exists()):
        return
    n_l = _surface_vertex_count(Path(surface_left))
    n_r = _surface_vertex_count(Path(surface_right))
    if int(n_l) != int(np.asarray(l_array).size):
        raise ValueError(
            f"Left surface geometry vertex count does not match exported left .func.gii data: "
            f"surface={surface_left} has {n_l}, data has {np.asarray(l_array).size}"
        )
    if int(n_r) != int(np.asarray(r_array).size):
        raise ValueError(
            f"Right surface geometry vertex count does not match exported right .func.gii data: "
            f"surface={surface_right} has {n_r}, data has {np.asarray(r_array).size}"
        )
    print(f"[surface] validated .func.gii data against geometry: L={surface_left}, R={surface_right}")


def _load_surface_meshes(n_vertices: int, search_dir: Path, surface_left: Path | None = None, surface_right: Path | None = None):
    from nilearn import datasets

    mesh = _infer_surface_mesh(int(n_vertices))
    if mesh != "fslr32k":
        fsavg = datasets.fetch_surf_fsaverage(mesh=mesh)
        return mesh, fsavg.infl_left, fsavg.infl_right, fsavg.sulc_left, fsavg.sulc_right

    if surface_left is not None and surface_right is not None and Path(surface_left).exists() and Path(surface_right).exists():
        print(f"[surface] using specified fsLR geometry: L={surface_left}, R={surface_right}")
        return mesh, str(surface_left), str(surface_right), None, None

    left, right = _find_fslr_surface_geometry([search_dir, Path(ATLAS_SURF_L).parent, Path(ATLAS_SURF_R).parent])
    if left is not None and right is not None:
        print(f"[surface] using fsLR inflated geometry: L={left}, R={right}")
        return mesh, str(left), str(right), None, None
    try:
        from neuromaps import datasets as nm_datasets

        fslr = nm_datasets.fetch_fslr(density="32k")
        surf = fslr.get("inflated", None) if isinstance(fslr, dict) else None
        if surf is None and isinstance(fslr, dict):
            surf = fslr.get("midthickness", None)
        if isinstance(surf, (list, tuple)) and len(surf) >= 2:
            return mesh, str(surf[0]), str(surf[1]), None, None
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Cannot render fsLR-32k cortical preview because no local fslr geometry was found "
            "and neuromaps.fetch_fslr failed. The .func.gii files were still exported."
        ) from exc
    raise RuntimeError("Cannot render fsLR-32k cortical preview because no L/R surface geometry was found.")


def _enhance_bg_contrast(bg: np.ndarray | None, strength: float = 1.2) -> np.ndarray | None:
    """Boost gyri/sulci contrast of a surface background.

    nilearn min-max normalizes bg_map internally, so a linear scale would not
    change appearance. We instead reshape the distribution with a symmetric
    S-curve around its median to deepen the dark/bright separation -> visible
    sulcal shading on an otherwise flat inflated surface."""
    if bg is None:
        return None
    # bg may be a path/str (e.g. fsaverage sulc map) - load it to a scalar array first.
    if isinstance(bg, (str, Path)):
        try:
            from nilearn.surface import load_surf_data

            bg = load_surf_data(str(bg))
        except Exception:  # noqa: BLE001
            return bg
    x = np.asarray(bg, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite) or float(np.nanstd(x[finite])) <= 0:
        return bg
    lo = float(np.nanpercentile(x[finite], 2))
    hi = float(np.nanpercentile(x[finite], 98))
    if hi <= lo:
        return bg
    u = np.clip((x - lo) / (hi - lo), 0.0, 1.0)  # 0..1
    # Logistic S-curve centered at 0.5 -> pushes values toward 0/1 (more contrast).
    s = 1.0 / (1.0 + np.exp(-float(strength) * (u - 0.5) * 2.0))
    s0 = 1.0 / (1.0 + np.exp(strength))
    s1 = 1.0 / (1.0 + np.exp(-strength))
    s = (s - s0) / (s1 - s0)
    out = np.array(x, dtype=float)
    out[finite] = s[finite]
    return out


def _surface_geometry_bg(surf_mesh) -> np.ndarray | None:
    try:
        from nilearn.surface import load_surf_mesh

        mesh = load_surf_mesh(surf_mesh)
        coords = np.asarray(mesh.coordinates if hasattr(mesh, "coordinates") else mesh[0], dtype=float)
        faces = np.asarray(mesh.faces if hasattr(mesh, "faces") else mesh[1], dtype=int)
        if coords.ndim != 2 or coords.shape[1] < 3:
            return None
        if faces.ndim != 2 or faces.shape[1] < 3:
            return None

        n_vertices = coords.shape[0]
        face_vertices = coords[faces[:, :3]]
        face_normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
        face_norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
        valid_faces = np.squeeze(face_norm > 0)
        if not np.any(valid_faces):
            return None
        face_normals[valid_faces] = face_normals[valid_faces] / face_norm[valid_faces]

        vertex_normals = np.zeros_like(coords, dtype=float)
        for col in range(3):
            np.add.at(vertex_normals, faces[:, col], face_normals)
        vertex_norm = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        valid_vertices = np.squeeze(vertex_norm > 0)
        if not np.any(valid_vertices):
            return None
        vertex_normals[valid_vertices] = vertex_normals[valid_vertices] / vertex_norm[valid_vertices]

        light = np.asarray([0.25, -0.35, 0.90], dtype=float)
        light = light / np.linalg.norm(light)
        bg = np.sum(vertex_normals * light[None, :], axis=1)

        edges = np.vstack(
            [
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
                faces[:, [1, 0]],
                faces[:, [2, 1]],
                faces[:, [0, 2]],
            ]
        )
        neigh_sum = np.zeros(n_vertices, dtype=float)
        neigh_count = np.zeros(n_vertices, dtype=float)
        np.add.at(neigh_count, edges[:, 0], 1.0)
        valid = neigh_count > 0
        if not np.any(valid):
            return None

        # Smooth enough to suppress triangle edges, but not so much that the
        # sulcal/gyri lighting is washed out.
        for _ in range(8):
            neigh_sum.fill(0.0)
            np.add.at(neigh_sum, edges[:, 0], bg[edges[:, 1]])
            bg[valid] = 0.55 * bg[valid] + 0.45 * (neigh_sum[valid] / neigh_count[valid])

        if not np.isfinite(bg).any() or float(np.nanstd(bg)) <= 0:
            bg = coords[:, 2]
        lo = float(np.nanpercentile(bg, 2))
        hi = float(np.nanpercentile(bg, 98))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        scaled = np.clip((bg - lo) / (hi - lo), 0.0, 1.0)
        return 0.08 + 0.34 * (1.0 - scaled)
    except Exception:
        return None


def _plot_roi_separation_contours(plotting, surf_mesh, code_map: np.ndarray, atlas_labels: np.ndarray, ax) -> None:
    """Draw a thin separation line around EACH colored parcel so adjacent ROIs
    sharing the same color stay visually distinct."""
    if not hasattr(plotting, "plot_surf_contours"):
        return
    code = np.asarray(code_map, dtype=float)
    atlas = np.asarray(atlas_labels, dtype=int)
    colored = code > 0
    if int(colored.sum()) == 0:
        return
    levels = sorted({int(x) for x in np.unique(atlas[colored]).tolist() if int(x) != 0})
    if not levels:
        return
    try:
        plotting.plot_surf_contours(
            surf_mesh,
            atlas,
            levels=levels,
            colors=["#3a3a3a"] * len(levels),
            linewidths=0.8,
            axes=ax,
            legend=False,
        )
    except Exception:
        return


def _draw_venn_legend(ax, present_codes: set[int]) -> None:
    """Draw a two-circle Venn legend: M_nn (red, left) and M_conv (green, right)
    with their overlap colored yellow (M_nn + M_conv)."""
    from matplotlib.patches import Circle

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    r = 0.26
    cy = 0.62
    cx_l, cx_r = 0.40, 0.60
    col_nn = BIT_COLORS[1]
    col_conv = BIT_COLORS[2]
    col_both = BIT_COLORS[3]

    # Base circles (semi-transparent so the overlap visually mixes), then an
    # explicit yellow lens patch on top for the intersection region.
    ax.add_patch(Circle((cx_l, cy), r, facecolor=col_nn, edgecolor="#2b2b2b", linewidth=1.0, alpha=0.92, zorder=2))
    ax.add_patch(Circle((cx_r, cy), r, facecolor=col_conv, edgecolor="#2b2b2b", linewidth=1.0, alpha=0.92, zorder=2))
    # Intersection lens approximated by a small circle centered between the two.
    if 3 in present_codes:
        ax.add_patch(
            Circle(((cx_l + cx_r) / 2.0, cy), r * 0.46, facecolor=col_both, edgecolor="none", alpha=1.0, zorder=3)
        )
    # Labels.
    ax.text(cx_l - r * 0.55, cy, "M_nn", ha="center", va="center", fontsize=10.0, color="#1a1a1a", zorder=4)
    ax.text(cx_r + r * 0.55, cy, "M_conv", ha="center", va="center", fontsize=10.0, color="#1a1a1a", zorder=4)
    if 3 in present_codes:
        ax.text(
            (cx_l + cx_r) / 2.0,
            cy - r - 0.12,
            "M_nn + M_conv",
            ha="center",
            va="center",
            fontsize=10.0,
            color="#1a1a1a",
            zorder=4,
        )


def _plot_combined_brain_figure(
    l_array: np.ndarray,
    r_array: np.ndarray,
    atlas_l: np.ndarray,
    atlas_r: np.ndarray,
    out_svg: Path,
    save_png: bool,
    bit_labels: dict[int, str],
    present_codes: list[int],
    title: str,
    surface_left: Path | None,
    surface_right: Path | None,
    surface_bg_left: Path | None,
    surface_bg_right: Path | None,
    use_geometry_bg_fallback: bool,
    enhance_surface_bg: bool,
    surface_bg_contrast: float,
) -> None:
    """Render cortical surface bitmask overlap map (4 views). Volume is NOT drawn (exported only)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from nilearn import plotting

    setup_publication_style()
    cmap, _ = _discrete_cmap()
    fig = plt.figure(figsize=(8.6, 6.0), constrained_layout=False)

    surface_rendered = False
    surface_error = ""
    try:
        import inspect

        _validate_surface_alignment(l_array, r_array, surface_left=surface_left, surface_right=surface_right)
        _, surf_l, surf_r, bg_l, bg_r = _load_surface_meshes(
            l_array.size,
            search_dir=out_svg.parent,
            surface_left=surface_left,
            surface_right=surface_right,
        )
        if bg_l is None or bg_r is None:
            found_bg_l, found_bg_r, _, _ = _load_surface_backgrounds(
                out_svg.parent,
                n_vertices=int(l_array.size),
                explicit_left=surface_bg_left,
                explicit_right=surface_bg_right,
            )
            if found_bg_l is not None and found_bg_r is not None:
                bg_l = found_bg_l
                bg_r = found_bg_r
        if (bg_l is None or bg_r is None) and bool(use_geometry_bg_fallback):
            bg_l = _surface_geometry_bg(surf_l)
            bg_r = _surface_geometry_bg(surf_r)
            if bg_l is not None and bg_r is not None:
                print("[surface] using smooth geometry-derived lighting texture as fallback background.")
        elif bg_l is None or bg_r is None:
            bg_l = None
            bg_r = None
            print("[surface] no fsLR sulc/curv/shape background used; relying on surface geometry shading only.")
        if bool(enhance_surface_bg):
            bg_l = _enhance_bg_contrast(bg_l, strength=float(surface_bg_contrast))
            bg_r = _enhance_bg_contrast(bg_r, strength=float(surface_bg_contrast))
        surf_extra = {}
        surf_params = inspect.signature(plotting.plot_surf_stat_map).parameters
        if "alpha" in surf_params:
            surf_extra["alpha"] = 1.0
        if "bg_on_data" in surf_params:
            surf_extra["bg_on_data"] = False
        if "darkness" in surf_params:
            surf_extra["darkness"] = 0.95 if bg_l is not None and bg_r is not None else 0.7
        view_specs = [
            ([0.175, 0.560, 0.330, 0.345], surf_l, l_array, atlas_l, "left", "lateral"),
            ([0.175, 0.265, 0.330, 0.345], surf_l, l_array, atlas_l, "left", "medial"),
            ([0.420, 0.560, 0.330, 0.345], surf_r, r_array, atlas_r, "right", "lateral"),
            ([0.420, 0.265, 0.330, 0.345], surf_r, r_array, atlas_r, "right", "medial"),
        ]
        for rect, mesh_geo, data_arr, atlas_arr, hemi, view in view_specs:
            ax = fig.add_axes(rect, projection="3d")
            plotting.plot_surf_stat_map(
                mesh_geo,
                data_arr,
                hemi=hemi,
                view=view,
                cmap=cmap,
                bg_map=bg_l if hemi == "left" else bg_r,
                threshold=0.5,
                vmin=0.5,
                vmax=7.5,
                **surf_extra,
                axes=ax,
                title="",
                colorbar=False,
            )
            _plot_roi_separation_contours(plotting, mesh_geo, data_arr, atlas_arr, ax)
        surface_rendered = True
    except Exception as exc:  # noqa: BLE001
        surface_rendered = False
        surface_error = str(exc)

    if not surface_rendered:
        ax = fig.add_axes([0.045, 0.265, 0.640, 0.610])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Cortical surface data exported as .func.gii\n"
            f"Preview not rendered here: {surface_error}",
            ha="center",
            va="center",
        )

    # Venn legend: M_nn, M_conv, and their overlap.
    legend_codes = [c for c in [1, 2, 4, 3, 5, 6, 7] if c in present_codes]
    present_set = set(int(c) for c in present_codes)
    venn_ok = bool(present_set) and present_set.issubset({1, 2, 3})
    if venn_ok:
        ax_leg = fig.add_axes([0.355, 0.005, 0.29, 0.20])
        _draw_venn_legend(ax_leg, present_set)
    else:
        ax_leg = fig.add_axes([0.10, 0.045, 0.80, 0.10])
        ax_leg.axis("off")
        handles = [Patch(facecolor=BIT_COLORS[code], edgecolor="black", label=bit_labels[code]) for code in legend_codes]
        if handles:
            ax_leg.legend(
                handles=handles,
                loc="center",
                ncol=len(handles),
                frameon=False,
                handlelength=1.6,
                handleheight=1.1,
                borderaxespad=0.0,
                columnspacing=1.6,
                handletextpad=0.6,
                fontsize=10.5,
            )
    fig.suptitle(title)
    # Save PNG only.
    if bool(save_png):
        out_png = out_svg.with_suffix(".png")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _fill_bitmask_maps(
    code_by_roi: dict[str, int],
    targets: dict[str, tuple[str, int]],
    surf_l: np.ndarray,
    surf_r: np.ndarray,
    vol_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Build combined bitmask surface/volume arrays (overlap codes 1-7) from per-ROI codes."""
    out_l = np.zeros_like(surf_l, dtype=np.float32)
    out_r = np.zeros_like(surf_r, dtype=np.float32)
    out_v = np.zeros_like(vol_data, dtype=np.float32)
    mapped_codes: dict[str, int] = {}
    for roi, code in code_by_roi.items():
        target = targets.get(str(roi))
        if target is None or int(code) <= 0:
            continue
        space, label = target
        if space == "L":
            _apply_code_to_mask(out_l, surf_l == int(label), int(code))
        elif space == "R":
            _apply_code_to_mask(out_r, surf_r == int(label), int(code))
        elif space == "V":
            _apply_code_to_mask(out_v, vol_data == int(label), int(code))
        mapped_codes[str(roi)] = int(code)
    return out_l, out_r, out_v, mapped_codes


def _cleanup_prefix_outputs(out_dir: Path, prefix: str) -> None:
    """Remove stale outputs for this condition before writing new filtered maps."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    suffixes = (".func.gii", ".label.gii", ".nii.gz", ".csv", ".png", ".svg")
    for path in out_dir.glob(f"{prefix}_*"):
        if path.is_file() and any(str(path).endswith(suffix) for suffix in suffixes):
            path.unlink()


def _render_outputs(
    sig: pd.DataFrame,
    prefix: str,
    title: str,
    out_dir: Path,
    args: argparse.Namespace,
    bit_labels: dict[int, str],
    mapping: dict[str, dict],
) -> None:
    if sig.empty:
        print(f"[SKIP] {prefix}: no significant ROI rows")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_prefix_outputs(out_dir, prefix)
    sig = _add_display_labels(sig, mapping)
    roi_aliases = _sig_roi_aliases(sig)

    surf_l, surf_r, vol_img, vol_data = _load_atlas(args)
    all_rois = [str(x) for x in pd.unique(sig["roi"].astype(str))]
    targets = _resolve_roi_targets(all_rois, roi_aliases, mapping, surf_l, surf_r, vol_data)

    models_present = [m for m in ("M_nn", "M_conv", "M_div") if m in set(sig["model"].astype(str))]
    audit_rows = []
    positive_only = bool(args.positive_only)
    # === Requirement 1+2: per-model nii, two versions ===
    for model in models_present:
        msig = sig[sig["model"].astype(str) == model]
        sig_rows = msig[msig["is_sig"].astype(bool)]
        # Version 1: FDR-significant only -> continuous r + binary
        val_sig = {str(r["roi"]): float(r["value"]) for _, r in sig_rows.iterrows()}
        bin_sig = {str(r["roi"]): 1.0 for _, r in sig_rows.iterrows()}
        l1, r1, v1 = _fill_value_maps(val_sig, targets, surf_l, surf_r, vol_data)
        _save_map_triplet(out_dir, f"{prefix}_{model}_sig_rvalue", l1, r1, vol_img, v1, intent="NIFTI_INTENT_ESTIMATE")
        l1b, r1b, v1b = _fill_value_maps(bin_sig, targets, surf_l, surf_r, vol_data)
        _save_map_triplet(out_dir, f"{prefix}_{model}_sig_binary", l1b, r1b, vol_img, v1b, intent="NIFTI_INTENT_LABEL")
        # Version 2: all ROIs -> continuous r
        val_all = {str(r["roi"]): float(r["value"]) for _, r in msig.iterrows()}
        l2, r2, v2 = _fill_value_maps(val_all, targets, surf_l, surf_r, vol_data)
        _save_map_triplet(out_dir, f"{prefix}_{model}_all_rvalue", l2, r2, vol_img, v2, intent="NIFTI_INTENT_ESTIMATE")
        for _, r in msig.iterrows():
            roi = str(r["roi"])
            tgt = targets.get(roi)
            audit_rows.append({
                "roi": roi,
                "display_label": str(r.get("display_label", roi)),
                "model": model,
                "value": float(r["value"]) if np.isfinite(r["value"]) else np.nan,
                "is_sig": bool(r["is_sig"]),
                "positive_only_filter": positive_only,
                "passes_positive_filter": bool((not positive_only) or (np.isfinite(r["value"]) and float(r["value"]) > 0)),
                "space": tgt[0] if tgt else "",
                "atlas_label": int(tgt[1]) if tgt else np.nan,
                "mapping_status": "mapped" if tgt else "unmatched",
            })

    # === Requirement 1: overlay the per-model significant results into a combined bitmask map ===
    sig_only = sig[sig["is_sig"].astype(bool)]
    if sig_only.empty:
        sig.to_csv(out_dir / f"{prefix}_all_rows.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(out_dir / f"{prefix}_roi_value_audit.csv", index=False)
        print(f"[SKIP] {prefix}: no significant ROI rows after positive_only={positive_only}")
        return
    code_by_roi = _roi_code_map(sig_only)
    comb_l, comb_r, comb_v, mapped_codes = _fill_bitmask_maps(code_by_roi, targets, surf_l, surf_r, vol_data)
    _save_map_triplet(out_dir, f"{prefix}_combined_sig_bitmask", comb_l, comb_r, vol_img, comb_v, intent="NIFTI_INTENT_LABEL")
    _save_label_gifti(out_dir / f"{prefix}_combined_sig_bitmask_surf_L.label.gii", comb_l, bit_labels)
    _save_label_gifti(out_dir / f"{prefix}_combined_sig_bitmask_surf_R.label.gii", comb_r, bit_labels)
    present_codes = sorted({int(c) for c in mapped_codes.values()})

    # Audit + code CSVs
    pd.DataFrame(audit_rows).to_csv(out_dir / f"{prefix}_roi_value_audit.csv", index=False)
    code_rows = []
    for roi, code in code_by_roi.items():
        aliases = roi_aliases.get(str(roi), set()) | _roi_aliases(str(roi))
        display_label = _display_label_from_record(str(roi), _record_for_aliases(aliases, mapping))
        code_rows.append({
            "roi": roi,
            "display_label": display_label,
            "model_code": int(code),
            "model_label": bit_labels.get(int(code), ""),
            "mapped": str(roi) in mapped_codes,
        })
    pd.DataFrame(code_rows).to_csv(out_dir / f"{prefix}_roi_model_codes.csv", index=False)
    sig.to_csv(out_dir / f"{prefix}_all_rows.csv", index=False)

    if not bool(args.maps_only):
        _plot_combined_brain_figure(
            comb_l,
            comb_r,
            atlas_l=surf_l,
            atlas_r=surf_r,
            out_svg=out_dir / f"{prefix}_brain_map.svg",
            save_png=not bool(args.no_png),
            bit_labels=bit_labels,
            present_codes=present_codes,
            title=title,
            surface_left=Path(args.surface_left) if args.surface_left is not None else None,
            surface_right=Path(args.surface_right) if args.surface_right is not None else None,
            surface_bg_left=Path(args.surface_bg_left) if args.surface_bg_left is not None else None,
            surface_bg_right=Path(args.surface_bg_right) if args.surface_bg_right is not None else None,
            use_geometry_bg_fallback=bool(args.use_geometry_bg_fallback),
            enhance_surface_bg=bool(args.enhance_surface_bg),
            surface_bg_contrast=float(args.surface_bg_contrast),
        )
    n_unmatched = int(sum(1 for r in code_rows if not r["mapped"]))
    if n_unmatched:
        print(f"[mapping] Warning: {prefix}: {n_unmatched} significant ROI(s) unmatched to atlas.")
    print(f"Saved {prefix}: per-model + combined surface/volume maps under {out_dir}")


def _cleanup_old_combined_dev_outputs(out_dir: Path) -> None:
    for path in Path(out_dir).glob("passive_reappraisal_combined_dev_models*"):
        if path.is_file():
            path.unlink()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.matrix_dir) / "figures" / str(args.stimulus_dir_name) / "combined_dev_model_brain_map"
    mode = _resolve_input_mode(args)
    mapping = _load_schaefer_mapping(args)
    if mode == "interaction":
        sig = _load_interaction_significant(args)
        prefix = "brain_age_interaction_combined_models"
        bit_labels = INTERACTION_BIT_LABELS
        title = "Brain x age-model interaction map"
        if sig.empty:
            raise ValueError("No significant ROI rows found for combined brain map.")
        _render_outputs(sig, prefix=prefix, title=title, out_dir=out_dir, args=args, bit_labels=bit_labels, mapping=mapping)
    else:
        sig = _load_significant(args)
        bit_labels = BIT_LABELS
        if sig.empty:
            raise ValueError("No significant ROI rows found for combined brain map.")
        _cleanup_old_combined_dev_outputs(out_dir)
        rendered = 0
        for condition in [str(c) for c in args.conditions]:
            sub = sig[sig["condition"].astype(str) == condition].copy()
            if sub.empty:
                print(f"[SKIP] {condition}: no significant ROI rows")
                continue
            prefix = f"{_safe_name(condition)}_dev_model"
            title = f"{condition} developmental-model brain map"
            _render_outputs(sub, prefix=prefix, title=title, out_dir=out_dir, args=args, bit_labels=bit_labels, mapping=mapping)
            rendered += 1
        if rendered == 0:
            raise ValueError("No condition had plottable significant ROI rows.")


if __name__ == "__main__":
    main()
