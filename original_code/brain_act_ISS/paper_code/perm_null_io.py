#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared IO helpers for permutation/null distribution files."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def make_metadata(
    *,
    script_name: str,
    result_type: str,
    input_files: Mapping[str, Any] | None = None,
    output_csv: str | Path | None = None,
    seed: int | None = None,
    n_perm: int | None = None,
    models: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "script_name": str(script_name),
        "result_type": str(result_type),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_files": dict(input_files or {}),
        "output_csv": "" if output_csv is None else str(output_csv),
        "seed": None if seed is None else int(seed),
        "n_perm": None if n_perm is None else int(n_perm),
        "models": [] if models is None else [str(x) for x in models],
    }
    if extra:
        meta.update(dict(extra))
    return meta


def save_perm_null_npz(path: str | Path, metadata: Mapping[str, Any] | None = None, **arrays: Any) -> Path:
    """Save arrays plus JSON metadata to a compressed NPZ and sidecar JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata or {})
    metadata_json = json.dumps(meta, ensure_ascii=False, sort_keys=True, default=_json_default)
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["metadata_json"] = np.asarray(metadata_json)
    np.savez_compressed(out, **payload)
    meta_path = out.with_name(f"{out.stem}_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return out


def load_perm_null_npz(path: str | Path, required: Sequence[str] | None = None) -> dict[str, Any]:
    """Load a null NPZ and validate required fields."""
    p = require_null_file(path)
    data = np.load(p, allow_pickle=False)
    out: dict[str, Any] = {k: data[k] for k in data.files}
    if "metadata_json" in out:
        raw = out["metadata_json"]
        meta_text = str(raw.item() if getattr(raw, "shape", ()) == () else raw)
        try:
            out["metadata"] = json.loads(meta_text)
        except json.JSONDecodeError:
            out["metadata"] = {}
    missing = [k for k in (required or []) if k not in out]
    if missing:
        raise KeyError(f"Null file {p} is missing required fields: {missing}")
    return out


def require_null_file(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Cannot find required permutation null file: {p}. "
            "Please rerun the corresponding analysis script after enabling/saving null distributions."
        )
    return p


def as_str_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([str(v) for v in values], dtype=object)

