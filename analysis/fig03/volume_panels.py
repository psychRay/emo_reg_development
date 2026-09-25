"""Figure 3b, 3c, and 3e calculations from whole-brain contrast maps."""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.image import math_img, resample_to_img
from scipy.spatial.distance import cosine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEURO_UTILS = PROJECT_ROOT / "src" / "neuro_utils_ray-0.1.0"
if str(NEURO_UTILS) not in sys.path:
    sys.path.insert(0, str(NEURO_UTILS))
os.environ.setdefault("NEUROMAPS_DATA", str(PROJECT_ROOT / "resources" / "neuromaps_cache"))

from .glm_post_hoc import pairwise_compare_images  # noqa: E402
from .plot_glm_res import plot_network_distributed_score  # noqa: E402
from neuro_utils.network import (  # noqa: E402
    compute_network_contribution,
    summarize_brain_maps_by_network,
)
from neuromaps.transforms import mni152_to_fslr  # noqa: E402

AGE_BINS = (
    (6, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
    (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19),
)
NETWORK_NAMES = {
    1: "visual", 2: "som/motor", 3: "dAttn", 4: "vAttn",
    5: "limbic", 6: "control", 7: "default",
}
NETWORK_COLORS = {
    "visual": "#781286", "som/motor": "#4682b4", "dAttn": "#4a9b3c",
    "vAttn": "#c43afa", "limbic": "#dcf8a4", "control": "#e69422",
    "default": "#cd3e4e",
}
REFERENCE_FILES = (
    ("EmoReg (Neurosynth)", "reappraisal_FDR_0.05.nii.gz"),
    ("EmoReg_comp1", "reappraisal_comp1.nii"),
    ("EmoReg_comp2", "reappraisal_comp2.nii"),
    ("EmoRct (Neurosynth)", "negative_emotion_FDR_0.01.nii.gz"),
    ("EmoRct_comp1", "emoGen_comp1.nii"),
    ("EmoRct_comp2", "emoGen_comp2.nii"),
)


def _required(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required Figure 3 input is missing: {path}")
    return path


def _task_map(results_dir: Path, contrast: str) -> Path:
    return _required(results_dir / f"brainAct-volume_cont-{contrast}_unthresh.nii.gz")


def _task_fdr_map(results_dir: Path, contrast: str) -> Path:
    return _required(results_dir / f"brainAct-volume_cont-{contrast}_FDR-005.nii.gz")


def _age_map(results_dir: Path, age_bin: tuple[int, int]) -> Path:
    return _required(
        results_dir / "age_groups" /
        f"brainAct-volume_stats-z_cont-rpsl-lkng_ageGroup-{age_bin[0]}-{age_bin[1]}_unthresh.nii"
    )


def _references(reference_dir: Path) -> tuple[list[str], list[nib.spatialimages.SpatialImage]]:
    labels, images = [], []
    for label, filename in REFERENCE_FILES:
        path = _required(reference_dir / filename)
        labels.append(label)
        images.append(nib.load(str(path)))
    return labels, images


def _similarities(
    references: list[nib.spatialimages.SpatialImage],
    reference_labels: list[str],
    task_paths: list[Path],
    task_labels: list[str],
    *,
    threshold: float | None,
) -> pd.DataFrame:
    first_target = nib.load(str(task_paths[0]))
    resampled_references = [
        resample_to_img(image, first_target, interpolation="continuous")
        for image in references
    ]
    task_images = []
    for path in task_paths:
        image = nib.load(str(path))
        if threshold is not None:
            image = math_img(f"img * (img > {threshold})", img=image)
        task_images.append(image)
    distances, _ = pairwise_compare_images(
        images=resampled_references + task_images,
        labels=reference_labels + task_labels,
        metric=cosine,
        diagonal=0,
        ignore_zero=True,
    )
    similarities = 1 - distances
    return similarities.loc[reference_labels, task_labels]


def _save_fig3b(results_dir: Path, reference_dir: Path, output_dir: Path) -> None:
    reference_labels, references = _references(reference_dir)
    task_labels = ["Reappraise - Look", "Look - Neutral"]
    task_paths = [
        _task_map(results_dir, "rpsl-lkng"),
        _task_map(results_dir, "lkng-lknt"),
    ]
    matrix = _similarities(
        references, reference_labels, task_paths, task_labels, threshold=2.81,
    )
    matrix.to_csv(output_dir / "fig3b_spatial_cosine_similarity.csv")
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    sns.heatmap(
        matrix, robust=True, annot=True, cmap="viridis", linecolor="k",
        linewidths=1.5, square=True, annot_kws={"size": 8}, ax=ax,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3b_spatial_cosine_similarity.png", dpi=300)
    plt.close(fig)


def _save_fig3c(results_dir: Path, yeo_atlas: Path, output_dir: Path) -> None:
    from .plot_glm_res import plot_bilateral_bar

    contrasts = ["rpsl-lkng", "lkng-lknt"]
    map_names = ["Reappraise - Look", "Look - Neutral"]
    volume_maps = [nib.load(str(_task_fdr_map(results_dir, contrast))) for contrast in contrasts]
    surface_pairs = [mni152_to_fslr(image, fslr_density="32k") for image in volume_maps]
    surface_maps = np.vstack([
        np.concatenate([
            np.asarray(pair[0].darrays[0].data),
            np.asarray(pair[1].darrays[0].data),
        ])
        for pair in surface_pairs
    ])

    network_summary = summarize_brain_maps_by_network(
        brain_maps=surface_maps,
        atlas=str(yeo_atlas),
        map_names=map_names,
        network_names=NETWORK_NAMES,
        positive_only=True,
        include_zero=False,
        summary_methods=["mean"],
    )["summary"]
    network_metrics = compute_network_contribution(
        brain_maps=surface_maps,
        atlas=str(yeo_atlas),
        map_names=map_names,
        network_names=NETWORK_NAMES,
        positive_only=True,
        cal_relative_contribution=False,
    )["network_metrics"]
    network_summary.to_csv(output_dir / "fig3c_network_mean_t_values.csv", index=False)
    network_metrics.to_csv(output_dir / "fig3c_network_metrics.csv", index=False)

    mean_values = network_summary.pivot(index="network_name", columns="map", values="mean")
    mean_values = mean_values.reindex(list(NETWORK_NAMES.values()))
    cosine_values = network_metrics.pivot(
        index="network_name", columns="map", values="cosine_similarity_binary_network"
    ).reindex(list(NETWORK_NAMES.values()))
    palette = [NETWORK_COLORS[name] for name in NETWORK_NAMES.values()]

    fig = plt.figure(figsize=(8.4, 3.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.8], wspace=0.3)
    for index, map_name in enumerate(map_names):
        ax = fig.add_subplot(grid[0, index])
        vals = mean_values[map_name].to_numpy(dtype=float)
        wedges, _ = ax.pie(
            np.maximum(vals, 0), startangle=90, counterclock=False,
            colors=palette, wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 0.8},
        )
        for wedge, value, network_name in zip(wedges, vals, NETWORK_NAMES.values()):
            angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
            ax.text(0.77 * np.cos(angle), 0.77 * np.sin(angle), f"{value:.1f}",
                    ha="center", va="center", fontsize=7)
            ax.text(1.24 * np.cos(angle), 1.24 * np.sin(angle), network_name,
                    ha="center", va="center", fontsize=6)
        ax.set_title(map_name, fontsize=8, pad=5)
    ax = fig.add_subplot(grid[0, 2])
    plot_bilateral_bar(
        cosine_values[map_names[0]].to_numpy(), cosine_values[map_names[1]].to_numpy(),
        xaxis_position="bottom", xlabel="cosine similarity",
        left_colors=palette, right_colors=palette,
        show_yticklabels=False, show_values=False, figsize=(4, 4.5),
        left_label=None, right_label=None, ax=ax,
    )
    ax.set_yticks([])
    ax.set_xlabel("cosine similarity")
    ax.set_xlim([-0.6, 0.4])
    ticks = [-0.6, -0.4, -0.2, 0, 0.2, 0.4]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(value):g}" for value in ticks])
    legend_handles = [Patch(facecolor=NETWORK_COLORS[name], label=name)
                      for name in NETWORK_NAMES.values()]
    fig.legend(handles=legend_handles, loc="center right", bbox_to_anchor=(1.01, 0.5),
               frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3c_network_summary.png", dpi=300)
    plt.close(fig)


def _save_fig3e_left(results_dir: Path, reference_dir: Path, output_dir: Path) -> None:
    reference_labels, references = _references(reference_dir)
    age_labels = [f"{lo}-{hi}" for lo, hi in AGE_BINS]
    similarities = _similarities(
        references, reference_labels,
        [_age_map(results_dir, age_bin) for age_bin in AGE_BINS],
        age_labels, threshold=3.29,
    ).T
    selected = similarities[["EmoReg (Neurosynth)", "EmoReg_comp2"]].copy()
    selected.to_csv(output_dir / "fig3e_spatial_cosine_similarity_by_age.csv")
    fig, ax = plt.subplots(figsize=(3.5, 3))
    sns.lineplot(
        data=selected, palette={"EmoReg (Neurosynth)": "#9B359B", "EmoReg_comp2": "#CE636D"},
        linewidth=2, markers=True, dashes=False, legend=False, ax=ax,
    )
    ax.set_xticks(np.arange(len(AGE_BINS)))
    ax.set_xticklabels([f"{lo}-{hi}" for lo, hi in AGE_BINS])
    ax.axvline(x=4, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel("Age groups/bins")
    ax.set_ylabel("Spatial patterns' similarity")
    ax.tick_params(axis="x", rotation=90, labelsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3e_spatial_similarity_by_age.png", dpi=300)
    plt.close(fig)


def _save_fig3e_right(results_dir: Path, yeo_atlas: Path, output_dir: Path) -> None:
    age_labels = [f"age {lo}-{hi}" for lo, hi in AGE_BINS]
    volume_maps = [
        math_img("img * (img > 2.81)", img=nib.load(str(_age_map(results_dir, age_bin))))
        for age_bin in AGE_BINS
    ]
    surface_pairs = [mni152_to_fslr(image, fslr_density="32k") for image in volume_maps]
    surface_maps = np.vstack([
        np.concatenate([
            np.asarray(pair[0].darrays[0].data),
            np.asarray(pair[1].darrays[0].data),
        ])
        for pair in surface_pairs
    ])
    result = compute_network_contribution(
        brain_maps=surface_maps,
        atlas=str(yeo_atlas),
        map_names=age_labels,
        network_names=NETWORK_NAMES,
        positive_only=False,
        cal_relative_contribution=True,
    )
    result["network_metrics"].to_csv(output_dir / "fig3e_network_contribution_long.csv", index=False)
    target = "positive_weight_contribution_pct_global"
    wide = result["network_metrics"].pivot(
        index="network_name", columns="map", values=target,
    ).reindex(list(NETWORK_NAMES.values()), columns=age_labels)
    wide.columns = [label.replace("age ", "") for label in wide.columns]
    wide.to_csv(output_dir / "fig3e_network_contribution.csv")
    fig, ax = plot_network_distributed_score(
        df=wide,
        network_order=list(NETWORK_NAMES.values()),
        network_colors=NETWORK_COLORS,
        multiply_by=1,
        title="Network contribution to ER",
        xlabel="Age groups/bins",
        ylabel="Network contribution score",
        legend=False,
        figsize=(3.2, 3.5),
        rotation=90,
    )
    ax.tick_params(axis="x", labelsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3e_network_contribution_by_age.png", dpi=300)
    plt.close(fig)


def run_volume_panels(
    *, results_dir: Path, reference_dir: Path, yeo_atlas: Path, output_dir: Path,
) -> None:
    """Run Fig. 3b, 3c, and 3e from the original GLM result maps."""
    results_dir = Path(results_dir)
    reference_dir = Path(reference_dir)
    yeo_atlas = _required(Path(yeo_atlas))
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_fig3b(results_dir, reference_dir, output_dir)
    _save_fig3c(results_dir, yeo_atlas, output_dir)
    _save_fig3e_left(results_dir, reference_dir, output_dir)
    _save_fig3e_right(results_dir, yeo_atlas, output_dir)
    (output_dir / "fig3_volume_panels_metadata.json").write_text(json.dumps({
        "panels": ["3b", "3c", "3e-left", "3e-right"],
        "glm_results_dir": str(results_dir.resolve()),
        "reference_map_dir": str(reference_dir.resolve()),
        "yeo_atlas": str(yeo_atlas.resolve()),
        "fig3b_task_map_z_threshold": 2.81,
        "fig3e_left_age_map_z_threshold": 3.29,
        "fig3e_right_age_map_z_threshold": 2.81,
        "fig3c_mean": "positive mean from FDR-0.005 masked whole-sample maps",
        "fig3c_network_metric": "cosine_similarity_binary_network",
        "fig3e_right_network_metric": "positive_weight_contribution_pct_global",
        "age_bins": [f"{lo}-{hi}" for lo, hi in AGE_BINS],
    }, indent=2), encoding="utf-8")
