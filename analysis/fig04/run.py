"""Figure 4: identify emotion regulation systems and summarize behavior, age effects, and mediation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.image import resample_to_img

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .er_systems_identification import (  # noqa: E402
    compute_group_mask_fast,
    identify_systems,
    mediation_bootstrap,
)

SYSTEMS = {
    "reappraisal_only": ("Reappraisal-only", "#E83C27"),
    "common_appraisal": ("Common appraisal", "#F932EE"),
    "nonmodifiable_emotion": ("Nonmodifiable emotion", "#0D1DF3"),
    "modifiable_emotion": ("Modifiable emotion", "#119C65"),
}
NETWORKS = {1: "visual", 2: "som/motor", 3: "dAttn", 4: "vAttn",
            5: "limbic", 6: "control", 7: "default"}
NETWORK_COLORS = {"visual": "#781286", "som/motor": "#4682b4", "dAttn": "#4a9b3c",
                  "vAttn": "#c43afa", "limbic": "#dcf8a4", "control": "#e69422",
                  "default": "#cd3e4e"}
CONDITIONS = ("rpsl", "lkng", "lknt")


def make_toy_data(seed: int = 20260925, n_subjects: int = 160) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Make toy condition volumes plus age and mediator variables for pipeline checks."""
    rng = np.random.default_rng(seed)
    shape = (24, 24, 18)
    xx, yy, zz = np.indices(shape)
    centers = [(6, 7, 8), (17, 7, 8), (6, 17, 8), (17, 17, 8)]
    regions = [sum((axis - center) ** 2 for axis, center in zip((xx, yy, zz), c)) <= 9 for c in centers]
    age = np.linspace(7, 18, n_subjects) + rng.normal(0, 0.2, n_subjects)
    gender = rng.integers(0, 2, n_subjects)
    site = rng.integers(1, 5, n_subjects)
    success = 0.55 + 0.018 * (age - 12) + rng.normal(0, 0.09, n_subjects)
    manifest_rows = []
    maps = {condition: [] for condition in CONDITIONS}
    for i in range(n_subjects):
        baseline = rng.normal(0, 0.16, shape)
        neutral = baseline + rng.normal(0, 0.12, shape)
        negative = neutral.copy() + rng.normal(0, 0.12, shape)
        negative[regions[0]] = neutral[regions[0]] + rng.normal(0, 0.008, int(regions[0].sum()))
        negative[regions[1]] += 0.85 + rng.normal(0, 0.06)
        negative[regions[2]] += 0.85 + rng.normal(0, 0.06)
        negative[regions[3]] += 0.85 + rng.normal(0, 0.06)
        regulate = negative + rng.normal(0, 0.12, shape)
        regulate[regions[0]] += 0.85 + 0.10 * (age[i] - 12) + rng.normal(0, 0.06)
        regulate[regions[1]] += 0.85 + 0.07 * (age[i] - 12) + rng.normal(0, 0.06)
        regulate[regions[2]] -= 0.85 + rng.normal(0, 0.06)
        regulate[regions[3]] = negative[regions[3]] + rng.normal(0, 0.008, int(regions[3].sum()))
        for condition, volume in (("rpsl", regulate), ("lkng", negative), ("lknt", neutral)):
            maps[condition].append(volume.astype(np.float32))
        manifest_rows.append({"subject_id": f"toy-{i:03d}", "age": age[i], "gender": gender[i],
                              "site": site[i], "reappraisal_success": success[i]})
    return pd.DataFrame(manifest_rows), np.stack([maps[c] for c in CONDITIONS]), np.stack(regions).astype(np.uint8)


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(np.asarray(data), affine), str(path))


def _data_on_reference(image: nib.spatialimages.SpatialImage,
                       reference: nib.spatialimages.SpatialImage,
                       interpolation: str) -> np.ndarray:
    """Avoid resampling participant maps already stored on the system-mask grid."""
    if image.shape[:3] == reference.shape[:3] and np.allclose(image.affine, reference.affine):
        return np.squeeze(np.asarray(image.get_fdata()))
    return np.squeeze(resample_to_img(image, reference, interpolation=interpolation).get_fdata())


def _plot_behavior(manifest: pd.DataFrame, condition_paths: dict[str, list[Path]],
                   masks: dict[str, Path], output: Path, atlas_path: Path | None) -> None:
    """Fig. 4c: compute condition beta summaries, success associations, and Yeo network shares."""
    rows = []
    network_rows = []
    atlas_img = nib.load(str(atlas_path)) if atlas_path else None
    for system, mask_path in masks.items():
        mask_img = nib.load(str(mask_path))
        mask = np.squeeze(mask_img.get_fdata()) > 0
        system_values: dict[str, np.ndarray] = {}
        for condition in CONDITIONS:
            values = []
            for subject, image_path in zip(manifest["subject_id"], condition_paths[condition]):
                image = nib.load(str(image_path))
                beta = _data_on_reference(image, mask_img, "continuous")[mask]
                mean = float(np.nanmean(beta)) if beta.size else np.nan
                values.append(mean)
                rows.append({"subject_id": subject, "system": system, "condition": condition, "mean_beta": mean})
            system_values[condition] = np.asarray(values)
        delta = system_values["rpsl"] - system_values["lkng"]
        for sid, value, success in zip(manifest["subject_id"], delta, manifest["reappraisal_success"]):
            rows.append({"subject_id": sid, "system": system, "condition": "rpsl_minus_lkng",
                         "mean_beta": value, "reappraisal_success": success})
        if atlas_img is not None:
            atlas = _data_on_reference(atlas_img, mask_img, "nearest").astype(int)
            labels, counts = np.unique(atlas[mask & (atlas > 0)], return_counts=True)
            total = max(int(counts.sum()), 1)
            network_rows.extend({"system": system, "network_id": int(label),
                                 "network_name": NETWORKS.get(int(label), f"network-{label}"),
                                 "voxel_count": int(count),
                                 "fraction": float(count / total)} for label, count in zip(labels, counts))
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "fig4c_subject_system_beta.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for j, (system, (label, color)) in enumerate(SYSTEMS.items()):
        data = summary[summary.system == system]
        condition_data = data[data.condition.isin(CONDITIONS)]
        sns.barplot(data=condition_data, x="condition", y="mean_beta", color="#8497B0", width=0.6, ax=axes[0, j])
        axes[0, j].set_title(label, fontsize=9)
        axes[0, j].set_xlabel("")
        axes[0, j].tick_params(axis="x", labelrotation=30, labelsize=8)
        delta_data = data[data.condition == "rpsl_minus_lkng"].copy()
        delta_data["reappraisal_success"] = manifest["reappraisal_success"].to_numpy()
        sns.regplot(data=delta_data, x="mean_beta", y="reappraisal_success", color=color,
                    robust=True, ci=95, scatter_kws={"edgecolors": "k", "s": 20}, ax=axes[1, j])
        axes[1, j].set_xlabel("Regulate negative − Look negative", fontsize=8)
        axes[1, j].set_ylabel("Reappraisal success" if j == 0 else "", fontsize=8)
        axes[1, j].tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(output / "fig4c_behavior_preview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if network_rows:
        network = pd.DataFrame(network_rows)
        network.to_csv(output / "fig4c_network_distribution.csv", index=False)
        pivot = network.pivot(index="system", columns="network_id", values="fraction").fillna(0)
        fig, axes = plt.subplots(1, len(SYSTEMS), figsize=(10, 2.7))
        for ax, (system, (label, _)) in zip(axes, SYSTEMS.items()):
            system_share = network[network.system == system]
            if not system_share.empty:
                labels = system_share["network_name"]
                colors = [NETWORK_COLORS.get(name, "#999999") for name in labels]
                ax.pie(system_share["fraction"], labels=labels, colors=colors,
                       startangle=90, textprops={"fontsize": 7})
            ax.set_title(label, fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "fig4c_network_pies.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        ax = pivot.plot(kind="bar", stacked=True, figsize=(7, 3), colormap="tab10")
        ax.set_ylabel("Fraction of voxels")
        ax.set_xlabel("")
        ax.legend(title="Yeo network", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output / "fig4c_network_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()


def _save_age_overlap_masks(age_pos_path: Path, age_neg_path: Path,
                            masks: dict[str, Path], output: Path) -> None:
    """Fig. 4d: save intersections of FDR age masks and the four system masks."""
    for direction, age_path in (("positive", age_pos_path), ("negative", age_neg_path)):
        age_img = nib.load(str(age_path))
        age = np.squeeze(age_img.get_fdata()) > 0
        for system in SYSTEMS:
            sys_img = nib.load(str(masks[system]))
            sys_data = _data_on_reference(sys_img, age_img, "nearest") > 0
            overlap = (age & sys_data).astype(np.uint8)
            target = output / f"fig4d_age_{direction}_overlap_{system}.nii.gz"
            _write_nifti(target, overlap, age_img.affine)


def _run_mediation(manifest: pd.DataFrame, output: Path) -> None:
    """Fig. 4e: bootstrap the age → regional mediator → reappraisal success paths."""
    results = []
    for mediator in ("aMTG", "aSTG"):
        result = mediation_bootstrap(data=manifest, x="age", m=mediator, y="reappraisal_success",
                                     n_boot=1000, seed=123)
        results.append({"mediator": mediator, "n": result["n"], "a": result["a"], "b": result["b"],
                        "indirect": result["indirect"], "indirect_ci_low": result["indirect_ci_low"],
                        "indirect_ci_high": result["indirect_ci_high"], "direct": result["direct"],
                        "total": result["total"], "a_p": result["mediator_model"].pvalues["age"],
                        "b_p": result["outcome_model"].pvalues[mediator],
                        "direct_p": result["outcome_model"].pvalues["age"]})
    pd.DataFrame(results).to_csv(output / "fig4e_mediation.csv", index=False)


def run_figure_4(*, input_path: Path | None, age_pos_mask: Path | None,
                 age_neg_mask: Path | None, mediator_mask_dir: Path | None,
                 atlas_path: Path | None,
                 toy: bool, output_dir: Path) -> None:
    """Run Fig. 4a–e computations, with NIfTI maps retained for Connectome Workbench/MRIcroGL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if toy:
        manifest, toy_maps, toy_regions = make_toy_data()
        affine = np.eye(4)
        with tempfile.TemporaryDirectory(prefix="fig04-") as temp_name:
            temp = Path(temp_name)
            condition_paths = {condition: [] for condition in CONDITIONS}
            for i, row in manifest.iterrows():
                for j, condition in enumerate(CONDITIONS):
                    path = temp / f"sub-{i:03d}_{condition}.nii.gz"
                    _write_nifti(path, toy_maps[j, i], affine)
                    condition_paths[condition].append(path)
            age_pos_path = temp / "toy_age_positive_mask.nii.gz"
            age_neg_path = temp / "toy_age_negative_mask.nii.gz"
            mediator_mask_dir = temp / "mediator_masks"
            mediator_mask_dir.mkdir()
            _write_nifti(age_pos_path, toy_regions[0] | toy_regions[1], affine)
            _write_nifti(age_neg_path, toy_regions[2] | toy_regions[3], affine)
            _write_nifti(mediator_mask_dir / "aMTG.nii.gz", toy_regions[0], affine)
            _write_nifti(mediator_mask_dir / "aSTG.nii.gz", toy_regions[1], affine)
            atlas_path = temp / "toy_yeo_atlas.nii.gz"
            toy_atlas = np.zeros(toy_regions.shape[1:], dtype=np.int16)
            for i, region in enumerate(toy_regions):
                toy_atlas[region > 0] = i + 1
            _write_nifti(atlas_path, toy_atlas, affine)
            _run_panels(manifest, condition_paths, age_pos_path, age_neg_path,
                        mediator_mask_dir, atlas_path, output_dir, toy=True)
    else:
        if input_path is None:
            raise ValueError("Real-data mode requires --input participant manifest CSV")
        if age_pos_mask is None or age_neg_mask is None:
            raise ValueError("Real-data mode requires positive and negative FDR age-effect masks")
        if mediator_mask_dir is None:
            raise ValueError("Real-data mode requires --fig4-mediator-mask-dir containing aMTG.nii.gz and aSTG.nii.gz")
        manifest = pd.read_csv(input_path)
        _validate_manifest(manifest, input_path.parent)
        condition_paths = {condition: [Path(value) if Path(value).is_absolute() else input_path.parent / value
                                      for value in manifest[condition]] for condition in CONDITIONS}
        _run_panels(manifest, condition_paths, age_pos_mask, age_neg_mask,
                    mediator_mask_dir, atlas_path, output_dir, toy=False)


def _validate_manifest(manifest: pd.DataFrame, base: Path) -> None:
    required = {"subject_id", "age", "gender", "site", "reappraisal_success", *CONDITIONS}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Participant manifest is missing columns: {sorted(missing)}")
    if manifest["subject_id"].duplicated().any() or len(manifest) < 8:
        raise ValueError("Manifest needs at least eight participants with unique subject_id values")
    numeric = manifest[["age", "gender", "site", "reappraisal_success"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Manifest demographic, behavior, and mediator values must be finite")
    for condition in CONDITIONS:
        for value in manifest[condition]:
            image_path = Path(value)
            image_path = image_path if image_path.is_absolute() else base / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing {condition} NIfTI: {image_path}")


def _run_panels(manifest: pd.DataFrame, condition_paths: dict[str, list[Path]],
                age_pos_mask: Path, age_neg_mask: Path, mediator_mask_dir: Path,
                atlas_path: Path | None,
                output: Path, toy: bool) -> None:
    group_mask_path = output / "fig4_analysis_mask.nii.gz"
    compute_group_mask_fast(
        [str(path) for condition in CONDITIONS for path in condition_paths[condition]],
        out_file=group_mask_path,
        threshold=0.95,
    )
    design = np.column_stack([np.ones(len(manifest)), manifest["gender"], manifest["site"]])
    identify_systems(
        look_neutral_imgs=[str(path) for path in condition_paths["lknt"]],
        look_negative_imgs=[str(path) for path in condition_paths["lkng"]],
        regulate_negative_imgs=[str(path) for path in condition_paths["rpsl"]],
        out_dir=str(output), mask_img=str(group_mask_path), design_matrix=design,
        second_level_contrast=np.array([1, 0, 0]), bf_alt_log10=1.0, bf_null_log10=-1.0,
        enforce_positive_condition=True, min_cluster_size=15,
    )
    masks = {name: output / f"mask_{name}.nii.gz" for name in SYSTEMS}
    mediator_values = _extract_mediator_values(manifest, condition_paths, mediator_mask_dir)
    manifest = manifest.join(mediator_values.set_index("subject_id"), on="subject_id")
    mediator_values.to_csv(output / "fig4e_mediator_values.csv", index=False)
    _plot_behavior(manifest, condition_paths, masks, output, atlas_path)
    _save_age_overlap_masks(age_pos_mask, age_neg_mask, masks, output)
    _run_mediation(manifest, output)
    (output / "run_metadata.json").write_text(json.dumps({
        "figure": 4, "toy": toy, "computed_panels": ["4b_masks", "4c", "4d_overlap_masks", "4e_mediation"],
        "system_identification": "analysis.fig04.er_systems_identification.identify_systems",
        "bf_alt_log10": 1.0, "bf_null_log10": -1.0, "positive_condition_constraint": True,
        "minimum_cluster_size": 15, "second_level_design": ["intercept", "gender", "site"],
        "group_mask_coverage_threshold": 0.95,
        "age_mask_source": "user-supplied positive and negative FDR-significant masks" if not toy else "synthetic positive and negative toy masks",
        "mediator_values": "mean rpsl-minus-lkng beta within supplied aMTG and aSTG masks",
        "mediation_bootstrap": {"n_boot": 1000, "seed": 123},
        "brain_map_outputs": "Fig. 4b component masks for MRIcroGL/Surfice; Fig. 4d age-by-system overlap masks for MRIcroGL",
    }, indent=2), encoding="utf-8")


def _extract_mediator_values(manifest: pd.DataFrame,
                             condition_paths: dict[str, list[Path]],
                             mask_dir: Path) -> pd.DataFrame:
    """Fig. 4e: extract regional reappraisal-minus-look beta values from aMTG/aSTG masks."""
    rows = []
    for mediator in ("aMTG", "aSTG"):
        mask_path = mask_dir / f"{mediator}.nii.gz"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing Fig. 4e ROI mask: {mask_path}")
        mask_img = nib.load(str(mask_path))
        roi_mask = np.squeeze(mask_img.get_fdata()) > 0
        if not roi_mask.any():
            raise ValueError(f"Fig. 4e ROI mask is empty: {mask_path}")
        for i, subject in enumerate(manifest["subject_id"]):
            rpsl = _data_on_reference(nib.load(str(condition_paths["rpsl"][i])), mask_img, "continuous")
            lkng = _data_on_reference(nib.load(str(condition_paths["lkng"][i])), mask_img, "continuous")
            rows.append({"subject_id": subject, mediator: float(np.nanmean((rpsl - lkng)[roi_mask]))})
    return pd.DataFrame(rows).groupby("subject_id", as_index=False).first()
