#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_ORDER_INFO = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_order_info.txt")
DEFAULT_OUT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final/schaefer2018_200parcels_7networks_order_mapping.csv")

NETWORK_FULL = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Salience/Ventral Attention",
    "Limbic": "Limbic",
    "Cont": "Frontoparietal Control",
    "Default": "Default Mode Network",
}
NETWORK_ORDER = {
    "Vis": 1,
    "SomMot": 2,
    "DorsAttn": 3,
    "SalVentAttn": 4,
    "Limbic": 5,
    "Cont": 6,
    "Default": 7,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Parse Schaefer2018 *_order_info.txt and export mapping from project ROI labels "
            "(L_1..L_100, R_101..R_200 by default) to 7-network/parcel names."
        )
    )
    p.add_argument("--order-info", type=Path, default=DEFAULT_ORDER_INFO)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--right-label-mode",
        choices=("global", "hemi"),
        default="global",
        help="global: RH labels are R_101..R_200; hemi: RH labels are R_1..R_100.",
    )
    return p.parse_args()


def _parse_name(name: str) -> tuple[str, str, str]:
    parts = str(name).strip().split("_")
    if len(parts) < 4 or parts[0] != "7Networks" or parts[1] not in {"LH", "RH"}:
        raise ValueError(f"Unexpected Schaefer label name: {name}")
    hemi = "L" if parts[1] == "LH" else "R"
    network = parts[2]
    parcel_name = "_".join(parts[2:])
    return hemi, network, parcel_name


def parse_order_info(path: Path, right_label_mode: str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find order info file: {path}")

    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict] = []
    i = 0
    while i < len(raw_lines):
        name = raw_lines[i]
        if i + 1 >= len(raw_lines):
            raise ValueError(f"Missing numeric/color line after label name: {name}")
        color_line = raw_lines[i + 1]
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", color_line)
        if len(nums) < 5:
            raise ValueError(f"Unexpected numeric/color line for {name}: {color_line}")
        label_id = int(float(nums[0]))
        rgba = [int(float(x)) for x in nums[1:5]]
        hemi, network, parcel_name = _parse_name(name)

        if hemi == "L":
            hemi_index = label_id
            roi = f"L_{hemi_index}"
            roi_schaefer_style = roi
        else:
            hemi_index = label_id - 100 if label_id > 100 else label_id
            roi = f"R_{label_id}" if str(right_label_mode) == "global" else f"R_{hemi_index}"
            roi_schaefer_style = f"R_{hemi_index}"

        rows.append(
            {
                "matrix_order": len(rows) + 1,
                "roi": roi,
                "roi_schaefer_style": roi_schaefer_style,
                "schaefer_label_id": label_id,
                "hemi": hemi,
                "hemi_index": hemi_index,
                "source_name": name,
                "network_order": NETWORK_ORDER.get(network, ""),
                "network": network,
                "network_full": NETWORK_FULL.get(network, network),
                "parcel_name": parcel_name,
                "rgba_r": rgba[0],
                "rgba_g": rgba[1],
                "rgba_b": rgba[2],
                "rgba_a": rgba[3],
            }
        )
        i += 2

    if len(rows) != 200:
        raise ValueError(f"Expected 200 Schaefer parcels, parsed {len(rows)} rows from {path}")
    labels = [r["roi"] for r in rows]
    if len(set(labels)) != len(labels):
        dup = sorted({x for x in labels if labels.count(x) > 1})
        raise ValueError(f"Duplicated output ROI labels: {dup}")
    return rows


def main() -> None:
    args = parse_args()
    rows = parse_order_info(Path(args.order_info), right_label_mode=str(args.right_label_mode))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "matrix_order",
        "roi",
        "roi_schaefer_style",
        "schaefer_label_id",
        "hemi",
        "hemi_index",
        "source_name",
        "network_order",
        "network",
        "network_full",
        "parcel_name",
        "rgba_r",
        "rgba_g",
        "rgba_b",
        "rgba_a",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved mapping table: {out}")
    print(f"Rows: {len(rows)}")
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["network"])] = counts.get(str(row["network"]), 0) + 1
    for network, n in sorted(counts.items(), key=lambda kv: NETWORK_ORDER.get(kv[0], 999)):
        print(f"{network}: {n}")


if __name__ == "__main__":
    main()
