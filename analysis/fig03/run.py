"""Figure 3a and 3d: volume activation models and surface ROI age effects."""

from __future__ import annotations

import json
import sys
import tempfile
import shutil
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

import nibabel as nib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MYLIB = PROJECT_ROOT / "src" / "prj_analysis" / "src"
if str(MYLIB) not in sys.path:
    sys.path.insert(0, str(MYLIB))
NEURO_UTILS = PROJECT_ROOT / "src" / "neuro_utils_ray-0.1.0"
if str(NEURO_UTILS) not in sys.path:
    sys.path.insert(0, str(NEURO_UTILS))

from mylib.brain import surface, volume  # noqa: E402
from mylib.stat.stats import GAMConfig  # noqa: E402
from mylib.stat import stats as stat_utils  # noqa: E402
from neuro_utils.wb_func import dscalar_to_border, gifti_to_cifti  # noqa: E402

CONDITIONS = ("rpsl", "lkng", "lknt")
CONDITION_CONTRASTS = (("rpsl", "lkng"), ("lkng", "lknt"))
FDR_ALPHA_FIG3A = 0.005


def _write_metric(path: Path, values: np.ndarray, hemi: str) -> None:
    intent = nib.nifti1.intent_codes["NIFTI_INTENT_NONE"]
    data_array = nib.gifti.GiftiDataArray(np.asarray(values, dtype=np.float32), intent=intent)
    data_array.meta = nib.gifti.GiftiMetaData.from_dict({"Name": "Figure 3 statistical map"})
    image = nib.gifti.GiftiImage(darrays=[data_array])
    image.meta = nib.gifti.GiftiMetaData.from_dict({
        "AnatomicalStructurePrimary": "CortexLeft" if hemi == "lh" else "CortexRight"
    })
    nib.save(image, str(path))


def _design_matrix(metadata: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "intercept": np.ones(len(metadata)),
        "age": metadata["age"].to_numpy(dtype=float),
        "gender": metadata["gender"].to_numpy(dtype=float),
        "meanFD": metadata["meanFD"].to_numpy(dtype=float),
        "site_id": metadata["site_id"].to_numpy(dtype=float),
    })


def _read_design_tsv(input_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(input_path, sep="\t")
    required = {"sub_id", "age", "gender", "site_id", "meanFD"}
    if missing := required - set(metadata.columns):
        raise ValueError(f"Figure 3 TSV is missing design columns: {sorted(missing)}")
    metadata = metadata.loc[:, ["sub_id", "age", "gender", "site_id", "meanFD"]].copy()
    if metadata["sub_id"].isna().any() or metadata["sub_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Figure 3 TSV sub_id values must be present")
    if metadata["sub_id"].astype(str).duplicated().any():
        raise ValueError("Figure 3 TSV must contain one row per unique sub_id")
    for column in ("age", "gender", "site_id", "meanFD"):
        metadata[column] = pd.to_numeric(metadata[column], errors="raise")
    if not np.isfinite(metadata[["age", "gender", "site_id", "meanFD"]].to_numpy()).all():
        raise ValueError("Figure 3 design columns must contain only finite values")
    if set(metadata["gender"].unique()) != {0, 1}:
        raise ValueError("Figure 3 gender codes must be 0 and 1")
    return metadata


def _volume_paths(metadata: pd.DataFrame, volume_dir: Path) -> dict[str, list[Path]]:
    """Resolve first-level condition maps using the study's subject-folder layout."""
    beta_names = {"rpsl": "beta_0001.nii.gz", "lkng": "beta_0002.nii.gz",
                  "lknt": "beta_0003.nii.gz"}
    paths = {condition: [] for condition in CONDITIONS}
    for sub_id in metadata["sub_id"].astype(str):
        for condition in CONDITIONS:
            path = volume_dir / sub_id / beta_names[condition]
            if not path.is_file():
                raise FileNotFoundError(f"Missing {condition} volume beta map for {sub_id}: {path}")
            paths[condition].append(path)
    return paths


def _project_volume_image(image: nib.spatialimages.SpatialImage, output_prefix: Path) -> tuple[Path, Path]:
    from neuromaps.transforms import mni152_to_fslr

    projected = mni152_to_fslr(image, fslr_density="32k")
    paths = []
    for hemi, metric in zip(("lh", "rh"), projected):
        path = output_prefix.with_name(f"{output_prefix.name}_{hemi}.func.gii")
        nib.save(metric, str(path))
        paths.append(path)
    return paths[0], paths[1]


def _write_dscalar(lh_path: Path, rh_path: Path, output_path: Path,
                   template: Path, map_name: str) -> None:
    gifti_to_cifti(
        lh_gifti=lh_path, rh_gifti=rh_path, template_cifti=template,
        output_file=output_path, output_type="dscalar", scalar_names=[map_name],
        fill_value=np.nan,
    )


def _write_border_if_available(dscalar_path: Path, out_prefix: Path, *,
                               left_surface: Path, right_surface: Path,
                               workbench_command: str, expression: str) -> bool:
    command = shutil.which(workbench_command)
    if command is None and not Path(workbench_command).is_file():
        return False
    dscalar_to_border(
        dscalar=dscalar_path, out_prefix=out_prefix,
        left_surface=left_surface, right_surface=right_surface,
        hemi="both", mode="expr", expression=expression,
        class_name="dscalar_roi", placement=0.33,
        wb_command=command or workbench_command, keep_intermediates=False,
        verbose=False,
    )
    return True


def _surface_templates() -> tuple[Path, Path]:
    template_dir = PROJECT_ROOT / "resources" / "templates" / "fslr_32k"
    return (
        template_dir / "Q1-Q6_R440.L.very_inflated.32k_fs_LR.surf.gii",
        template_dir / "Q1-Q6_R440.R.very_inflated.32k_fs_LR.surf.gii",
    )


def _save_volume_tests(metadata: pd.DataFrame, paths: dict[str, list[Path]], mask_path: Path,
                       surface_atlas: Path, output_dir: Path,
                       workbench_command: str) -> bool:
    design = _design_matrix(metadata)
    contrast = {"adjusted_mean": np.array([1, 0, 0, 0, 0], dtype=float)}
    mask_img = nib.load(str(mask_path))
    mask = mask_img.get_fdata() != 0
    left_surface, right_surface = _surface_templates()
    borders_generated = True
    for condition_paths in paths.values():
        for path in condition_paths:
            image = nib.load(str(path))
            if image.shape != mask_img.shape or not np.allclose(image.affine, mask_img.affine):
                raise ValueError(f"Condition map and brain mask must share geometry: {path}")
    for condition_a, condition_b in CONDITION_CONTRASTS:
        name = f"{condition_a}_minus_{condition_b}"
        effect_path = output_dir / f"fig3a_{name}_effect.nii.gz"
        t_path = output_dir / f"fig3a_{name}_t.nii.gz"
        result = volume.glm_AmB_with_covariates(
            A_paths=[str(p) for p in paths[condition_a]],
            B_paths=[str(p) for p in paths[condition_b]],
            X=design,
            mask_path=str(mask_path),
            contrasts=contrast,
            out_effect_paths={"adjusted_mean": str(effect_path)},
            out_t_paths={"adjusted_mean": str(t_path)},
            out_beta_stack_path=None,
        )
        t_img = result["contrasts"]["adjusted_mean"]["t"]
        pvals = 2 * stats.t.sf(np.abs(t_img.get_fdata()[mask]), df=result["dof"])
        reject, qvals, _, _ = multipletests(pvals, alpha=FDR_ALPHA_FIG3A, method="fdr_bh")
        q_map = np.ones(mask.shape, dtype=np.float32)
        reject_map = np.zeros(mask.shape, dtype=np.uint8)
        q_map[mask] = qvals.astype(np.float32)
        reject_map[mask] = reject.astype(np.uint8)
        q_img = nib.Nifti1Image(q_map, mask_img.affine, mask_img.header)
        q_path = output_dir / f"fig3a_{name}_q_fdr.nii.gz"
        nib.save(q_img, str(q_path))
        nib.save(nib.Nifti1Image(reject_map, mask_img.affine, mask_img.header),
                 str(output_dir / f"fig3a_{name}_fdr_mask.nii.gz"))
        _project_volume_image(nib.load(str(effect_path)), output_dir / f"fig3a_{name}_effect")
        unthresh_pair = _project_volume_image(nib.load(str(t_path)), output_dir / f"fig3a_{name}_unthresh")
        _project_volume_image(q_img, output_dir / f"fig3a_{name}_q_fdr")
        significant_t = np.zeros(mask.shape, dtype=np.float32)
        significant_t[mask] = t_img.get_fdata()[mask] * reject.astype(np.float32)
        significant_img = nib.Nifti1Image(significant_t, mask_img.affine, mask_img.header)
        sig_path = output_dir / f"fig3a_{name}_FDR-005.nii.gz"
        nib.save(significant_img, str(sig_path))
        fdr_pair = _project_volume_image(significant_img, output_dir / f"fig3a_{name}_FDR-005")
        positive_t = np.maximum(significant_t, 0)
        positive_img = nib.Nifti1Image(positive_t, mask_img.affine, mask_img.header)
        positive_pair = _project_volume_image(positive_img, output_dir / f"fig3a_{name}_FDR-005_positive")
        unthresh_cifti = output_dir / f"fig3a_{name}_unthresh.dscalar.nii"
        fdr_cifti = output_dir / f"fig3a_{name}_FDR-005.dscalar.nii"
        positive_cifti = output_dir / f"fig3a_{name}_FDR-005_positive.dscalar.nii"
        _write_dscalar(*unthresh_pair, unthresh_cifti, surface_atlas, f"{name} unthresholded")
        _write_dscalar(*fdr_pair, fdr_cifti, surface_atlas, f"{name} FDR q<.005")
        _write_dscalar(*positive_pair, positive_cifti, surface_atlas, f"{name} FDR q<.005 positive")
        borders_generated &= _write_border_if_available(
            fdr_cifti, output_dir / f"fig3a_{name}_FDR-005",
            left_surface=left_surface, right_surface=right_surface,
            workbench_command=workbench_command, expression="(x != 0)",
        )
        borders_generated &= _write_border_if_available(
            positive_cifti, output_dir / f"fig3a_{name}_FDR-005_positive",
            left_surface=left_surface, right_surface=right_surface,
            workbench_command=workbench_command, expression="(x > 0)",
        )
    return borders_generated


def _save_volume_age_gam(metadata: pd.DataFrame, paths: dict[str, list[Path]],
                         volume_atlas: Path, surface_atlas: Path, output_dir: Path,
                         workbench_command: str) -> bool:
    """Fit the Fig. 3d ROI GAM in volume space and paint its parcel statistics to fsLR."""
    atlas_img = nib.load(str(volume_atlas))
    atlas = np.rint(atlas_img.get_fdata()).astype(np.int32)
    roi_ids = np.unique(atlas[atlas > 0])
    if roi_ids.size == 0:
        raise ValueError(f"Volume atlas contains no positive parcel labels: {volume_atlas}")
    roi_index = np.searchsorted(roi_ids, atlas[atlas > 0])
    roi_counts = np.bincount(roi_index, minlength=roi_ids.size).astype(float)
    Y = np.empty((roi_ids.size, len(metadata)), dtype=np.float64)
    for subject_idx, (a_path, b_path) in enumerate(zip(paths["rpsl"], paths["lkng"])):
        a_img, b_img = nib.load(str(a_path)), nib.load(str(b_path))
        if a_img.shape != atlas.shape or b_img.shape != atlas.shape:
            raise ValueError(f"Condition maps and volume atlas must have matching shapes: {a_path}, {b_path}")
        if not np.allclose(a_img.affine, atlas_img.affine) or not np.allclose(b_img.affine, atlas_img.affine):
            raise ValueError(f"Condition maps and volume atlas must share geometry: {a_path}, {b_path}")
        delta = (a_img.get_fdata(dtype=np.float32) - b_img.get_fdata(dtype=np.float32))[atlas > 0]
        sums = np.bincount(roi_index, weights=delta, minlength=roi_ids.size)
        Y[:, subject_idx] = sums / roi_counts

    X = metadata[["age", "gender", "meanFD", "site_id"]].astype(float)
    age = X["age"].to_numpy()
    Xlin_all = X[[col for col in X.columns if col != "age"]].to_numpy()
    lam_grid = 10.0 ** np.linspace(-5, 5, 12)
    partial_r2 = np.full(roi_ids.size, np.nan)
    F_age = np.full(roi_ids.size, np.nan)
    p_age = np.full(roi_ids.size, np.nan)
    age_max = np.full(roi_ids.size, np.nan)
    cfg = GAMConfig(smooth_var="age", k_splines=5, lam_grid=[-5, 5, 12],
                    center_smooth=True, grid_size=101, grid_kind="quantile",
                    n_boot=1000, bootstrap_deriv_se=False, verbose=False)
    grid = np.unique(np.quantile(age, np.linspace(0.01, 0.99, cfg.grid_size)))
    for roi_idx, y in enumerate(Y):
        rss_f, edf_f, rss_r, edf_r, model = stat_utils.fit_gam_full_and_reduced(
            y, age[:, None], Xlin_all, k_splines=cfg.k_splines,
            lam_grid=lam_grid, max_iter=1000, verbose=cfg.verbose)
        f_value, p_value, _, _, pr2 = stat_utils.nested_anova_gaussian(
            rss_full=rss_f, rss_reduced=rss_r, n_obs=len(y),
            edf_full=edf_f, edf_reduced=edf_r)
        partial_r2[roi_idx], F_age[roi_idx], p_age[roi_idx] = pr2, f_value, p_value
        xlin_mean = np.repeat(Xlin_all.mean(axis=0, keepdims=True), grid.size, axis=0)
        fitted = model.predict(np.column_stack([grid, xlin_mean]))
        smooth = fitted - np.nanmean(fitted) if cfg.center_smooth else fitted
        derivative = np.gradient(smooth, grid)
        age_max[roi_idx] = grid[np.nanargmax(np.abs(derivative))]
    fdr = stat_utils.fdr_bh(p_age, alpha=0.05)
    q_age, reject = fdr["qvals"], fdr["reject"]
    pd.DataFrame({"parcel": roi_ids, "partial_r2": partial_r2, "F_age": F_age,
                  "p_age": p_age, "q_age": q_age, "significant_fdr": reject,
                  "age_of_max_change": age_max}).to_csv(
                      output_dir / "fig3d_age_gam_roi_statistics.csv", index=False)

    labels = {hemi: surface._load_surf_labels(str(surface_atlas), hemi_hint=hemi)
              for hemi in ("lh", "rh")}
    for hemi in ("lh", "rh"):
        values = np.zeros(labels[hemi].size, dtype=np.float32)
        q_values = np.ones(labels[hemi].size, dtype=np.float32)
        significant = np.zeros(labels[hemi].size, dtype=np.float32)
        for idx, parcel in enumerate(roi_ids):
            vertices = labels[hemi] == parcel
            values[vertices] = partial_r2[idx]
            q_values[vertices] = q_age[idx]
            if reject[idx]:
                significant[vertices] = partial_r2[idx]
        _write_metric(output_dir / f"fig3d_partial_r2_{hemi}.func.gii", values, hemi)
        _write_metric(output_dir / f"fig3d_q_age_{hemi}.func.gii", q_values, hemi)
        _write_metric(output_dir / f"fig3d_partial_r2_fdr_{hemi}.func.gii", significant, hemi)
    partial_cifti = output_dir / "fig3d_partial_r2.dscalar.nii"
    q_cifti = output_dir / "fig3d_q_age.dscalar.nii"
    significant_cifti = output_dir / "fig3d_partial_r2_fdr.dscalar.nii"
    _write_dscalar(output_dir / "fig3d_partial_r2_lh.func.gii",
                   output_dir / "fig3d_partial_r2_rh.func.gii", partial_cifti,
                   surface_atlas, "Fig. 3d age GAM partial R-squared")
    _write_dscalar(output_dir / "fig3d_q_age_lh.func.gii",
                   output_dir / "fig3d_q_age_rh.func.gii", q_cifti,
                   surface_atlas, "Fig. 3d age GAM FDR q-value")
    _write_dscalar(output_dir / "fig3d_partial_r2_fdr_lh.func.gii",
                   output_dir / "fig3d_partial_r2_fdr_rh.func.gii", significant_cifti,
                   surface_atlas, "Fig. 3d significant age GAM partial R-squared")
    left_surface, right_surface = _surface_templates()
    return _write_border_if_available(
        significant_cifti, output_dir / "fig3d_partial_r2_fdr",
        left_surface=left_surface, right_surface=right_surface,
        workbench_command=workbench_command, expression="(x > 0)",
    )


def _make_toy_volume_inputs(directory: Path, output_dir: Path,
                            volume_atlas_path: Path) -> tuple[pd.DataFrame, dict[str, list[Path]], Path]:
    """Create small-sample condition maps on the supplied Schaefer volume grid."""
    rng = np.random.default_rng(20260925)
    n_subjects = 36
    metadata = pd.DataFrame({
        "sub_id": [f"toy-{i:03d}" for i in range(n_subjects)],
        "age": rng.uniform(6.2, 18.8, n_subjects),
        "gender": np.tile([0, 1], n_subjects // 2),
        "site_id": np.tile([1, 2, 3, 4], n_subjects // 4),
        "meanFD": rng.uniform(0.04, 0.24, n_subjects),
    })
    atlas_img = nib.load(str(volume_atlas_path))
    atlas = np.rint(atlas_img.get_fdata()).astype(np.int32)
    roi_ids = np.unique(atlas[atlas > 0])
    roi_code = np.zeros(int(atlas.max()) + 1, dtype=np.float32)
    roi_code[roi_ids] = rng.normal(0.3, 0.2, roi_ids.size)
    base_pattern = roi_code[np.clip(atlas, 0, roi_code.size - 1)]
    paths = {condition: [] for condition in CONDITIONS}
    for i, row in metadata.iterrows():
        subject_dir = directory / str(row.sub_id)
        subject_dir.mkdir(parents=True, exist_ok=True)
        baseline = base_pattern + rng.normal(0, 0.03, roi_code.size)[np.clip(atlas, 0, roi_code.size - 1)]
        age_effect = 0.2 * np.sin((row.age - 12.0) / 3.0 + roi_code[np.clip(atlas, 0, roi_code.size - 1)] * 5)
        condition_maps = {
            "rpsl": baseline + 0.35 * base_pattern + age_effect,
            "lkng": baseline,
            "lknt": baseline - 0.1 * base_pattern,
        }
        for condition, beta_index in zip(CONDITIONS, (1, 2, 3)):
            image_data = condition_maps[condition].astype(np.float32)
            path = subject_dir / f"beta_{beta_index:04d}.nii.gz"
            nib.save(nib.Nifti1Image(image_data, atlas_img.affine, atlas_img.header), str(path))
            paths[condition].append(path)
    mask_path = output_dir / "toy_fig3_mask.nii.gz"
    nib.save(nib.Nifti1Image((atlas > 0).astype(np.uint8), atlas_img.affine, atlas_img.header),
             str(mask_path))
    return metadata, paths, mask_path


def run_figure_3(*, input_path: Path | None, toy: bool, output_dir: Path,
                 schaefer_atlas: Path | None = None, volume_dir: Path | None = None,
                 brain_mask: Path | None = None, volume_atlas: Path | None = None,
                 workbench_command: str = "wb_command") -> None:
    """Compute Fig. 3a in volume space and Fig. 3d from volume-derived ROI differences."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which(workbench_command) is None and not Path(workbench_command).is_file():
        print("wb_command not found; GIFTI/CIFTI maps will be written, but Workbench border files will be skipped.")
    if not toy:
        if input_path is None or volume_dir is None or brain_mask is None or schaefer_atlas is None or volume_atlas is None:
            raise ValueError("Real Figure 3a/3d requires --input TSV, --fig3-volume-dir, --fig3-brain-mask, --fig3-volume-atlas, and --schaefer-atlas")
        metadata = _read_design_tsv(input_path)
        data_paths = _volume_paths(metadata, volume_dir)
        borders_3a = _save_volume_tests(metadata, data_paths, brain_mask,
                                        schaefer_atlas, output_dir, workbench_command)
        borders_3d = _save_volume_age_gam(metadata, data_paths, volume_atlas,
                                          schaefer_atlas, output_dir, workbench_command)
        (output_dir / "run_metadata.json").write_text(json.dumps({
            "figure": 3, "toy": False, "computed_panels": ["3a", "3d"],
            "analysis_space": "MNI152 volume; fsLR 32k surface projection for Workbench maps",
            "condition_maps": {"rpsl": "beta_0001.nii.gz", "lkng": "beta_0002.nii.gz", "lknt": "beta_0003.nii.gz"},
            "design_columns": ["intercept", "age", "gender", "meanFD", "site_id"],
            "activation_fdr_alpha": FDR_ALPHA_FIG3A,
            "workbench_borders_generated": bool(borders_3a and borders_3d),
            "gam_configuration": {"smooth_var": "age", "k_splines": 5,
                "lam_grid": [-5, 5, 12], "grid_size": 101,
                "grid_kind": "quantile", "n_boot": 1000,
                "bootstrap_deriv_se": False},
        }, indent=2), encoding="utf-8")
        return

    volume_atlas = volume_atlas or PROJECT_ROOT / "resources/atlases/schaefer2018/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
    schaefer_atlas = schaefer_atlas or PROJECT_ROOT / "resources/atlases/schaefer2018/Schaefer2018_200Parcels_7Networks_order.dlabel.nii"
    with tempfile.TemporaryDirectory(prefix="fig03-toy-", dir=output_dir) as temp_dir:
        metadata, data_paths, mask_path = _make_toy_volume_inputs(Path(temp_dir), output_dir, volume_atlas)
        borders_3a = _save_volume_tests(metadata, data_paths, mask_path,
                                        schaefer_atlas, output_dir, workbench_command)
        borders_3d = _save_volume_age_gam(metadata, data_paths, volume_atlas,
                                          schaefer_atlas, output_dir, workbench_command)
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "figure": 3,
        "toy": toy,
        "computed_panels": ["3a", "3d"],
        "activation_contrasts": [f"{a} - {b}" for a, b in CONDITION_CONTRASTS],
        "activation_fdr_alpha": FDR_ALPHA_FIG3A,
        "analysis_space": "MNI152 volume; fsLR 32k surface projection for Workbench maps",
        "design_columns": ["intercept", "age", "gender", "meanFD", "site_id"],
        "workbench_borders_generated": bool(borders_3a and borders_3d),
        "age_gam_source": "volume Schaefer ROI means; mylib.stat.stats GAM fitting and nested ANOVA",
        "gam_configuration": {
            "smooth_var": "age", "k_splines": 5, "lam_grid": [-5, 5, 12],
            "grid_size": 101, "grid_kind": "quantile", "n_boot": 1000,
            "bootstrap_deriv_se": False,
        },
    }, indent=2), encoding="utf-8")
