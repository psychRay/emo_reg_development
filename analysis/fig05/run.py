"""Figure 5 analyses and manuscript-linked outputs."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / "src" / "neuro_utils_ray-0.1.0"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from .ml_classification import (  # noqa: E402
    compute_bootstrapped_decoding_map,
    compute_reconstructed_activation_pattern,
    fit_final_multisite_models,
    make_default_model_specs,
    multisite_permutation_test,
    nested_leave_one_site_out_compare_models,
)
from .ml_post_hoc import (  # noqa: E402
    _cluster_bootstrap_subjects,
    _fit_gam_with_gridsearch,
    _predict_curve,
    extract_signature_response,
)
from .plot_ml_res import (  # noqa: E402
    boxplot_with_points,
    plot_bilateral_bar,
    plot_hist_with_vline,
    plot_ml_roc_curves,
    plot_network_distributed_score,
    permutation_test_within_sex_age_effect,
)
from neuro_utils.network import compute_network_contribution  # noqa: E402
from neuro_utils.parcellation import parcel_to_brain  # noqa: E402

CONDITIONS = ("rpsl", "lkng")
NETWORK_NAMES = {
    1: "visual", 2: "som/motor", 3: "dAttn", 4: "vAttn",
    5: "limbic", 6: "control", 7: "default",
}
NETWORK_COLORS = {
    "visual": "#781286", "som/motor": "#4682b4", "dAttn": "#4a9b3c",
    "vAttn": "#c43afa", "limbic": "#dcf8a4", "control": "#e69422",
    "default": "#cd3e4e",
}


def make_toy_data(seed: int = 20260925, n_subjects: int = 64, n_features: int = 232):
    """Create paired parcel responses for exercising the Fig. 5 analysis path."""
    rng = np.random.default_rng(seed)
    age = np.linspace(7, 18, n_subjects) + rng.normal(0, 0.25, n_subjects)
    gender = np.tile([0, 1], n_subjects // 2)
    site = np.arange(n_subjects) % 4 + 1
    age_z = (age - age.mean()) / age.std(ddof=1)
    shared = rng.normal(size=(n_subjects, n_features))
    rpsl = shared + rng.normal(scale=0.65, size=shared.shape)
    lkng = shared + rng.normal(scale=0.65, size=shared.shape)
    lkng[:, :8] += 0.45 + 0.07 * age_z[:, None]
    lknt = rng.normal(size=shared.shape)
    meta = pd.DataFrame({"sub_id": [f"toy-{i:03d}" for i in range(n_subjects)],
                         "age": age, "age_z": age_z, "gender": gender, "site": site})
    return meta, {"rpsl": rpsl, "lkng": lkng, "lknt": lknt}


def _load_real_data(data_dir: Path):
    meta_path = data_dir / "subs_info.csv"
    meta = pd.read_csv(meta_path)
    if "sub_id" not in meta.columns and "sub_ids" in meta.columns:
        meta = meta.rename(columns={"sub_ids": "sub_id"})
    required_meta = {"sub_id", "age", "age_z", "gender", "site"}
    missing = required_meta - set(meta.columns)
    if missing:
        raise ValueError(f"{meta_path} is missing columns: {sorted(missing)}")
    matrices = {}
    for condition in CONDITIONS:
        path = data_dir / f"parc-cortex_subcortex_cond-{condition}.csv"
        frame = pd.read_csv(path, header=None)
        if len(frame) != len(meta) or frame.shape[1] != 233:
            raise ValueError(f"{path} must have {len(meta)} rows, an ID column, and 232 parcel columns")
        if not np.array_equal(frame.iloc[:, 0].astype(str).to_numpy(), meta["sub_id"].astype(str).to_numpy()):
            raise ValueError(f"Subject order in {path} does not match {meta_path}")
        matrices[condition] = frame.iloc[:, 1:233].to_numpy(dtype=float)
    return meta, matrices


def _paired_design(meta: pd.DataFrame, matrices: dict[str, np.ndarray]):
    n = len(meta)
    X = np.vstack([matrices["rpsl"], matrices["lkng"]])
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    subjects = np.tile(meta["sub_id"].astype(str).to_numpy(), 2)
    sites = np.tile(meta["site"].to_numpy(), 2)
    if not np.isfinite(X).all():
        raise ValueError("Parcel response matrices contain non-finite values")
    return X, y, subjects, sites


def _reduce_model_grids(model_specs, toy: bool):
    if toy:
        for name, spec in model_specs.items():
            grid = spec["param_grid"]
            for key, values in list(grid.items()):
                if isinstance(values, (list, tuple, np.ndarray)):
                    grid[key] = list(values)[len(values) // 2:len(values) // 2 + 1]
    return model_specs


def _save_scalar_map(values: np.ndarray, atlas_path: Path, output_path: Path, name: str):
    image, _ = parcel_to_brain(
        parcel_values=np.asarray(values, dtype=float)[:200],
        parcellation=str(atlas_path), parcel_ids=None,
        background=0, fill_value=np.nan, output_format="cifti", scalar_names=[name],
    )
    image.to_filename(str(output_path))


def _plot_cv_results(y, sites, observed, perm, output: Path):
    scores = {name: result["oof_score"] for name, result in observed["model_results"].items()}
    roc = plot_ml_roc_curves(y_true=y, y_scores=scores, positive_label=1, fold_ids=sites,
                             plot_folds=False, plot_pooled=True, plot_mean=False,
                             title=None, figsize=(4, 4))
    roc["fig"].savefig(output / "fig5b_multimodel_roc.png", dpi=300, bbox_inches="tight")
    plt.close(roc["fig"])
    logistic = observed["model_results"]["logistic_l2"]["oof_score"]
    roc = plot_ml_roc_curves(y_true=y, y_scores=logistic, positive_label=1, fold_ids=sites,
                             plot_folds=True, plot_pooled=True, plot_mean=False,
                             title=None, figsize=(4, 4), linecolor_fold="gray",
                             alpha_folds=0.7, linecolor_pooled="#e83c27", linewidth_pooled=2)
    roc["fig"].savefig(output / "fig5b_logistic_roc.png", dpi=300, bbox_inches="tight")
    plt.close(roc["fig"])
    names = list(perm["model_permutation_results"])
    perm_scores = pd.DataFrame({f"perm_{name}": perm["model_permutation_results"][name]["permutation_scores"]
                                for name in names})
    observed_scores = [float(perm["observed_summary"].loc[
        perm["observed_summary"]["model_name"] == name, "roc_auc"].iloc[0]) for name in names]
    ax, _ = boxplot_with_points(
        perm_scores, observed_scores, figsize=(3, 3.7),
        box_kwargs={"width": 0.35, "showfliers": False, "linewidth": 1.5,
                    "linecolor": "#adadad", "boxprops": {"facecolor": "#adadad"},
                    "medianprops": {"linewidth": 2, "color": "white"},
                    "showcaps": False, "whis": 3, "fill": True},
        point_kwargs={"s": 70, "color": "red", "marker": "o", "zorder": 20},
        y_break=((0.46, 0.54), (0.67, 0.71)), show_line=False,
        xticklabels=["logistic L2", "linear SVC", "rbf SVC"], xlabel="ML models",
        ylabel="ROC-AUC", title="Permutation results of ML models")
    fig = plt.gcf()
    fig.savefig(output / "fig5b_model_permutation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _fit_signature_trajectory(meta, scores, toy: bool, output: Path):
    n = len(meta)
    curve_data = pd.DataFrame({"subject_id": meta["sub_id"].astype(str).to_numpy(),
                               "age": meta["age"].to_numpy(), "age_z": meta["age_z"].to_numpy(),
                               "sex": meta["gender"].to_numpy(),
                               "reappraise": scores[:n], "look": scores[n:]})
    curve_data["signature_response"] = curve_data["look"] - curve_data["reappraise"]
    X = curve_data[["age_z", "sex"]].to_numpy(dtype=float)
    y = curve_data["signature_response"].to_numpy(dtype=float)
    lam_grid = np.logspace(-3, 3, 15)
    grid_n = 200
    age_grid = np.linspace(curve_data["age"].min(), curve_data["age"].max(), grid_n)
    age_grid_z = (age_grid - curve_data["age"].mean()) / curve_data["age"].std(ddof=1)
    gam = _fit_gam_with_gridsearch(X, y, model_type="sex_modulated", N_SPLINES=4, lam=lam_grid)
    rng = np.random.RandomState(42)
    n_boot = 20 if toy else 1000
    curves = {sex: [] for sex in (0, 1)}
    for _ in range(n_boot):
        boot = _cluster_bootstrap_subjects(curve_data, "subject_id", rng)
        Xb = boot[["age_z", "sex"]].to_numpy(dtype=float)
        yb = boot["signature_response"].to_numpy(dtype=float)
        boot_gam = _fit_gam_with_gridsearch(Xb, yb, model_type="sex_modulated", N_SPLINES=4, lam=lam_grid)
        for sex in (0, 1):
            curves[sex].append(_predict_curve(boot_gam, age_grid_z, sex))
    fig, axes = plt.subplots(2, 1, figsize=(3, 4.3), sharex=False, constrained_layout=True)
    colors = {0: "#e58f39", 1: "#2472ad"}
    rows = []
    for sex in (0, 1):
        mean = _predict_curve(gam, age_grid_z, sex)
        boot_curves = np.asarray(curves[sex])
        low, high = np.nanpercentile(boot_curves, [2.5, 97.5], axis=0)
        label = "Male" if sex == 0 else "Female"
        axes[0].plot(age_grid, mean, color=colors[sex], lw=2, label=label)
        axes[0].fill_between(age_grid, low, high, color=colors[sex], alpha=0.3, linewidth=0)
        subset = curve_data[curve_data["sex"] == sex]
        axes[0].scatter(subset["age"], subset["signature_response"], marker=".",
                        color="#f2b780" if sex == 0 else "#b3d8f2", alpha=0.65,
                        s=35, edgecolors="k", linewidths=0.4)
        coef = np.polyfit(subset["age"], subset["signature_response"], deg=1)
        linear_x = np.linspace(subset["age"].min(), subset["age"].max(), grid_n)
        linear_y = np.polyval(coef, linear_x)
        residual = subset["signature_response"] - np.polyval(coef, subset["age"])
        dof = len(subset) - 2
        se_resid = np.sqrt(np.sum(residual ** 2) / dof)
        sxx = np.sum((subset["age"] - subset["age"].mean()) ** 2)
        from scipy.stats import t
        ci = t.ppf(0.975, dof) * se_resid * np.sqrt(
            1 / len(subset) + (linear_x - subset["age"].mean()) ** 2 / sxx)
        axes[1].scatter(subset["age"], subset["signature_response"], marker=".",
                        color="#f2b780" if sex == 0 else "#b3d8f2", alpha=0.65,
                        s=35, edgecolors="k", linewidths=0.4)
        axes[1].plot(linear_x, linear_y, color=colors[sex], lw=2, label=label)
        axes[1].fill_between(linear_x, linear_y - ci, linear_y + ci,
                             color=colors[sex], alpha=0.3, linewidth=0)
        rows.extend({"age": a, "sex": sex, "fit": f, "ci_low": lo, "ci_high": hi}
                    for a, f, lo, hi in zip(age_grid, mean, low, high))
    axes[0].set_xticks([])
    axes[0].spines["bottom"].set_visible(False)
    axes[0].set_ylabel("Signature response")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Signature response")
    axes[1].set_xticks([6, 9, 12, 16, 19])
    axes[1].legend(frameon=False)
    fig.savefig(output / "fig5d_age_signature_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output / "fig5d_age_signature_curves.csv", index=False)
    perm = permutation_test_within_sex_age_effect(
        X=X, y=y, target_sex=0, n_perm=10 if toy else 5000, n_grid=grid_n,
        statistic="mean_abs_deviation", n_splines=4, lam=lam_grid,
        use_obs_perm=True, random_state=123)
    hist = plot_hist_with_vline(perm["T_perm"], perm["T_obs"], figsize=(4, 4), bins=30,
                                hist_kws={"color": "#e58f39", "alpha": 0.6, "edgecolor": "white"})
    hist_fig = hist[0].figure if isinstance(hist, tuple) else hist.figure
    hist_fig.savefig(output / "fig5d_age_signature_permutation.png", dpi=300, bbox_inches="tight")
    plt.close(hist_fig)
    (output / "fig5d_permutation_stats.json").write_text(
        json.dumps({"statistic": float(perm["T_obs"]), "p_value": float(perm["p_perm"]),
                    "n_permutations": int(len(perm["T_perm"])), "target_sex": 0}, indent=2), encoding="utf-8")


def _network_outputs(encoding_pattern: np.ndarray, parcel_atlas_path: Path,
                     yeo_atlas_path: Path, output: Path):
    surface_map = output / "fig5e_haufe_pattern.dscalar.nii"
    _save_scalar_map(encoding_pattern, parcel_atlas_path, surface_map, "Haufe activation")
    atlas_img = nib.load(str(yeo_atlas_path))
    pattern_img = nib.load(str(surface_map))
    values = np.asarray(pattern_img.get_fdata()).reshape(-1)
    # Keep positive and negative Haufe weights separate, as in Fig. 5f.
    maps = np.stack([np.where(values < 0, values, 0), np.where(values > 0, values, 0)])
    dist = compute_network_contribution(
        brain_maps=maps, atlas=atlas_img, map_names=["negative_weights", "positive_weights"],
        network_names=NETWORK_NAMES, positive_only=False, cal_relative_contribution=False)
    metrics = dist["network_metrics"]
    metrics.to_csv(output / "fig5f_network_metrics.csv", index=False)
    positive = metrics.loc[metrics["map"] == "positive_weights"].set_index("network_name").reindex(NETWORK_NAMES.values())
    negative = metrics.loc[metrics["map"] == "negative_weights"].set_index("network_name").reindex(NETWORK_NAMES.values())
    fig, ax = plot_bilateral_bar(
        positive["cosine_similarity_binary_network"].fillna(0).to_numpy(),
        negative["cosine_similarity_binary_network"].fillna(0).to_numpy(),
        y_labels=list(NETWORK_NAMES.values()), figsize=(4, 4.5),
        xaxis_position="bottom", xlabel="Cosine similarity",
        left_colors=[NETWORK_COLORS[n] for n in NETWORK_NAMES.values()],
        right_colors=[NETWORK_COLORS[n] for n in NETWORK_NAMES.values()],
        show_yticklabels=False, show_values=False, left_label=None, right_label=None)
    fig.legend(handles=[Patch(facecolor=NETWORK_COLORS[name], label=name)
                        for name in NETWORK_NAMES.values()],
               frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.subplots_adjust(right=0.78)
    ax.set_xlim(-0.6, 0.4)
    ax.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4])
    fig.savefig(output / "fig5f_network_similarity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_figure_5(*, data_dir: Path | None, toy: bool, output_dir: Path,
                 atlas_path: Path | None = None, bootstraps: int | None = None,
                 permutations: int | None = None, n_jobs: int | None = None):
    """Run the Fig. 5 classifier, maps, age signature, and network analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if toy:
        meta, matrices = make_toy_data()
        n_boot = bootstraps or 5
        n_perm = permutations or 5
        jobs = n_jobs or 1
    else:
        if data_dir is None:
            raise ValueError("Real-data mode requires --fig5-data-dir")
        meta, matrices = _load_real_data(data_dir)
        n_boot = bootstraps or 5000
        n_perm = permutations or 1000
        jobs = n_jobs or 12
    atlas_path = atlas_path or ROOT / "resources/atlases/schaefer2018/Schaefer2018_200Parcels_7Networks_order.dlabel.nii"
    yeo_path = ROOT / "resources/atlases/yeo2011/Yeo2011_7Networks_N1000.dlabel.nii"
    X, y, subjects, sites = _paired_design(meta, matrices)
    model_specs = _reduce_model_grids(make_default_model_specs(
        ["logistic_l2", "linear_svc", "rbf_svc"], random_state=42), toy)
    observed = nested_leave_one_site_out_compare_models(
        X=X, y=y, subject_ids=subjects, site_ids=sites, model_specs=model_specs,
        default_scoring="roc_auc", random_state=42, n_jobs=jobs,
        max_inner_site_splits=min(5, len(np.unique(sites))))
    perm = multisite_permutation_test(
        X=X, y=y, subject_ids=subjects, site_ids=sites, model_specs=model_specs,
        n_permutations=n_perm, metric_for_pvalue="roc_auc", default_scoring="roc_auc",
        random_state=42, n_jobs=jobs, max_inner_site_splits=min(5, len(np.unique(sites))),
        observed_results=observed, return_full_observed_results=True)
    _plot_cv_results(y, sites, observed, perm, output_dir)
    pd.DataFrame({"subject_id": subjects, "site": sites, "condition_label": y,
                  "oof_score_logistic_l2": observed["model_results"]["logistic_l2"]["oof_score"]}).to_csv(
        output_dir / "fig5b_oof_predictions.csv", index=False)
    with (output_dir / "fig5b_model_results.pkl").open("wb") as f:
        pickle.dump(perm, f)
    pd.DataFrame(perm["observed_summary"]).to_csv(output_dir / "fig5b_model_auc.csv", index=False)
    final = fit_final_multisite_models(
        X=X, y=y, subject_ids=subjects, site_ids=sites, model_specs=model_specs,
        default_scoring="roc_auc", random_state=42, n_jobs=10 if not toy else 1,
        force_logo_inner=True, max_inner_site_splits=min(5, len(np.unique(sites))))
    decoder = compute_bootstrapped_decoding_map(
        X=X, y=y, model_spec=model_specs["logistic_l2"],
        best_params=final["logistic_l2"]["best_params"], subject_ids=subjects,
        n_bootstraps=n_boot, q=0.05, unc_p_threshold=None, bootstrap_unit="subject",
        positive_label=1, weight_space="transformed", map_estimate="full",
        random_state=42, feature_names=None, verbose=False)
    haufe = compute_reconstructed_activation_pattern(
        X=X, y=y, model_spec=model_specs["logistic_l2"],
        best_params=final["logistic_l2"]["best_params"], subject_ids=subjects,
        n_bootstraps=n_boot, bootstrap_unit="subject", positive_label=1,
        random_state=42, feature_names=None, compute_bsr=True, bsr_threshold=3.0,
        top_bsr_percent=10, compute_fdr=True, fdr_q=0.05, fdr_method="sign",
        unc_p_threshold=0.005, use_pattern_for_display="final",
        store_bootstrap_patterns=False, verbose=False)
    decoder["feature_table"].to_csv(output_dir / "fig5c_decoder_feature_table.csv", index=False)
    haufe["feature_table"].to_csv(output_dir / "fig5e_haufe_feature_table.csv", index=False)
    _save_scalar_map(np.asarray(decoder["full_weight"]).reshape(-1), atlas_path,
                     output_dir / "fig5c_decoder_full_weight.dscalar.nii", "Decoder weight")
    _save_scalar_map(np.asarray(decoder["thresholded_map"]).reshape(-1), atlas_path,
                     output_dir / "fig5c_decoder_fdr_weight.dscalar.nii", "FDR decoder weight")
    enc_pattern = np.asarray(haufe["final_encoding_pattern"]).reshape(-1)
    _save_scalar_map(enc_pattern, atlas_path, output_dir / "fig5e_haufe_pattern.dscalar.nii", "Haufe activation")
    signature = extract_signature_response(
        observed_results=observed, model_name="logistic_l2",
        sample_info=pd.DataFrame({"subject_id": subjects, "condition": np.r_[np.repeat("rpsl", len(meta)),
                                                                           np.repeat("lkng", len(meta))]}))
    signature.to_csv(output_dir / "fig5d_signature_response.csv", index=False)
    _fit_signature_trajectory(meta, signature["signature_response"].to_numpy(), toy, output_dir)
    _network_outputs(enc_pattern, atlas_path, yeo_path, output_dir)
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "figure": 5, "toy": toy, "models": list(model_specs),
        "nested_cv": "leave-one-site-out; ROC-AUC; random_state=42; max five inner site splits",
        "permutations": n_perm, "subject_bootstraps": n_boot,
        "decoder_fdr_q": 0.05, "haufe_fdr_q": 0.05, "haufe_uncorrected_p": 0.005,
        "gam": {"n_splines": 4, "lambda_grid": np.logspace(-3, 3, 15).tolist(),
                "subject_bootstraps": 20 if toy else 1000, "age_grid": 200},
        "surface_atlas": str(atlas_path), "network_atlas": str(yeo_path),
        "toy_results_are_manuscript_estimates": False,
    }, indent=2), encoding="utf-8")
