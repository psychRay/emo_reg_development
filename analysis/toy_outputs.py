"""Keep only manuscript-facing results after a synthetic figure run."""

from __future__ import annotations

import shutil
from pathlib import Path


TOY_INTERMEDIATES = {
    1: ("run_metadata.json", "toy_fig01_input.csv"),
    2: ("run_metadata.json",),
    3: ("run_metadata.json", "labels_lh.npy", "labels_rh.npy", "toy_fig3_mask.nii.gz"),
    4: (
        "run_metadata.json", "fig4_analysis_mask.nii.gz",
        "emotion_generation_log10BF10.nii.gz", "emotion_generation_t.nii.gz",
        "reappraisal_log10BF10.nii.gz", "reappraisal_t.nii.gz",
        "fig4a_system_evidence_rules.png", "fig4b_system_masks_preview.png",
        "fig4d_age_overlap_preview.png", "fig4e_mediation_preview.png",
        "fig4e_mediator_values.csv", "fig4c_subject_system_beta.csv",
        "fig4a_system_voxel_counts.csv", "fig4c_behavior_preview.png",
    ),
    5: (
        "run_metadata.json", "fig5b_model_results.pkl",
        "fig5b_oof_predictions.csv", "fig5c_decoder_full_weight.dscalar.nii",
        "fig5d_signature_response.csv",
    ),
}


def _remove_child_directory(root: Path, name: str) -> None:
    root = root.resolve()
    target = (root / name).resolve()
    if target.parent != root:
        raise ValueError(f"Unexpected output directory: {target}")
    if target.is_dir():
        shutil.rmtree(target)


def finish_toy_outputs(figure: int, output_dir: Path) -> None:
    """Discard reproducible scratch products after all Figure outputs are saved."""
    output = Path(output_dir).resolve()
    if figure in TOY_INTERMEDIATES:
        for name in TOY_INTERMEDIATES[figure]:
            (output / name).unlink(missing_ok=True)
        if figure == 3:
            for path in output.glob("fig3a_*"):
                name = path.name
                final = (
                    name.endswith("_roi_statistics.csv")
                    or name.endswith("_vertex_statistics.csv")
                    or "FDR-005" in name
                    or "fdr_fdr_bh_reject_alpha-0.005" in name
                    or "_tmap_" in name
                )
                if path.is_file() and not final:
                    path.unlink()
        return
    if figure != 6:
        raise ValueError(f"Unknown toy figure: {figure}")

    work = output / "intermediate"
    joint = work / "by_emotion" / "Reappraisal_minus_Passive_Emo"
    result = joint / "roi_isc_behavior_delta_age_regression_joint.csv"
    if not result.is_file():
        raise FileNotFoundError(result)
    shutil.copy2(result, output / "fig6d" / "joint_regression.csv")
    selection = joint / "roi_isc_delta_dev_models_perm_fwer.csv"
    if selection.is_file():
        shutil.copy2(selection, output / "fig6d" / "roi_selection.csv")

    for condition in ("Passive_Emo", "Reappraisal"):
        source = work / "by_emotion" / condition / "behavior_isc_dev_models_perm_fwer.csv"
        if source.is_file():
            shutil.copy2(source, output / "fig6b" / f"{condition}_model_statistics.csv")

    for path in (output / "fig6c").iterdir():
        name = path.name
        keep = (
            name.endswith("_all_rows.csv")
            or name.endswith("_roi_model_codes.csv")
            or "_combined_sig_bitmask_surf_" in name and name.endswith(".label.gii")
            or name.endswith("_combined_sig_bitmask_volume.nii.gz")
            or any(f"_{model}_sig_rvalue_" in name for model in ("M_nn", "M_conv"))
        )
        if path.is_file() and not keep:
            path.unlink()

    brain_maps = output / "fig6d" / "brain_maps"
    for path in brain_maps.glob("*_mapping_check.csv"):
        path.unlink()
    coupling = output / "fig6d" / "coupling"
    (coupling / "subjects_age_sorted.csv").unlink(missing_ok=True)
    for path in coupling.rglob("*_age_sorted_conditional_coupling_matrix.npy"):
        path.unlink()
    _remove_child_directory(output, "intermediate")
