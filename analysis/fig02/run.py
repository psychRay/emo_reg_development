"""Figure 2b-e: developmental trajectories, derivative bands, and ERQ strategy use."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gam_age_on_task import (
    add_derivative_band_support_to_pipeline_output,
    plot_derivative_band_with_tps,
    plot_posterior_trajectory_with_tps,
    plot_tp_kde_from_pipeline,
    run_direct_outcome_gam_turning_point_pipeline,
    run_gam_turning_point_pipeline,
)

GENDER_LEVELS = [0, 1]
GENDER_LABELS = {0: "male", 1: "female"}
LAM_GRID = np.logspace(-3, 3, 10)
PIPELINE_PARAMETERS = {
    "age_grid_size": 200,
    "n_draws": 1000,
    "n_perm": 1000,
    "n_draws_perm": 1000,
    "min_sign_prob": 0.25,
    "min_count": 50,
    "assign_window": 1,
    "min_peak_distance": 1,
    "edge_exclusion": 0.25,
    "separate_kind": False,
}


def make_toy_inputs(seed: int = 20260925) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic long ratings and ERQ profiles for the Figure 2 pipeline."""
    rng = np.random.default_rng(seed)
    subjects = []
    for gender in GENDER_LEVELS:
        for index in range(120):
            age = float(np.clip(6.2 + index * 12.4 / 119 + rng.normal(0, 0.16), 6, 18.9))
            subjects.append((f"toy-{gender}-{index:03d}", age, gender))

    ratings: list[dict] = []
    profiles: list[dict] = []
    for sub_id, age, gender in subjects:
        person = rng.normal(0, 0.45)
        phase = age - (11.0 + 0.2 * gender)
        reappraisal_effect = 0.75 - 0.48 * np.cos(2 * np.pi * phase / 10.0)
        look = float(np.clip(2.25 + person + rng.normal(0, 0.4), 1, 4))
        reappraise = float(np.clip(look + reappraisal_effect + rng.normal(0, 0.45), 1, 4))
        neutral = float(np.clip(3.0 + person + rng.normal(0, 0.4), 1, 4))
        for cognition, value in (("rpsl", reappraise), ("lkng", look), ("lknt", neutral)):
            ratings.append({"sub_id": sub_id, "age": age, "gender": gender,
                            "cognition": cognition, "emot_rating": value})
        erq = 2.0 + 1.15 / (1 + np.exp(-(age - 12.0) / 2.3)) + 0.08 * gender
        profiles.append({"sub_id": sub_id, "age": age, "sex": gender,
                         "ERQ_CR": float(erq + rng.normal(0, 0.3))})
    return pd.DataFrame(ratings), pd.DataFrame(profiles)


def _read_inputs(input_path: Path | None, profiles_path: Path | None, toy: bool):
    if toy:
        return make_toy_inputs()
    if input_path is None or profiles_path is None:
        raise ValueError("Real-data mode requires both --input and --profiles CSV files")
    ratings = pd.read_csv(input_path)
    profiles = pd.read_csv(profiles_path)
    required_ratings = {"sub_id", "age", "gender", "cognition", "emot_rating"}
    profiles = profiles.rename(columns={"subject": "sub_id"})
    required_profiles = {"sub_id", "age", "sex", "ERQ_CR"}
    if missing := required_ratings - set(ratings.columns):
        raise ValueError(f"Behavior CSV is missing columns: {sorted(missing)}")
    if missing := required_profiles - set(profiles.columns):
        raise ValueError(f"Profile CSV is missing columns: {sorted(missing)}")
    return ratings, profiles


def _save_trajectory(output: Path, result: dict, contrast: str, filename: str) -> None:
    fit = result["results_by_contrast"][contrast]
    tps = fit["final_tps_with_perm"]
    fig, ax = plot_posterior_trajectory_with_tps(
        result_one_contrast=fit,
        final_tps_one_contrast=tps,
        gender_label_map=GENDER_LABELS,
        figsize=(3.5, 3),
        title=None,
        trajectory_kwargs={"linewidth": 3, "linestyle": "-"},
        tp_line_kwargs={"linewidth": 1, "linestyle": "--"},
        tp_marker_shape="D",
        show_tp_lines=True,
        show_tp_markers=True,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([8, 10, 12, 14, 16, 18])
    ax.tick_params(axis="both", labelsize=10)
    fig.savefig(output / filename, bbox_inches="tight")
    plt.close(fig)
    tps.to_csv(output / filename.replace(".svg", "_turning_points.csv"), index=False)


def _run_contrast(ratings: pd.DataFrame, pipeline_parameters: dict,
                  *, outcome: str = "emot_rating") -> dict:
    return run_gam_turning_point_pipeline(
        df_long=ratings,
        dv_col=outcome,
        fixed_gender_levels=GENDER_LEVELS,
        gender_label_map=GENDER_LABELS,
        contrasts=[("rpsl", "lkng")],
        lam_grid=LAM_GRID,
        n_splines=4,
        seed=2025,
        **pipeline_parameters,
    )


def run_figure_2(*, input_path: Path | None, profiles_path: Path | None,
                 toy: bool, output_dir: Path) -> None:
    """Run the Figure 2b-e computational panels"""
    output_dir.mkdir(parents=True, exist_ok=True)
    ratings, profiles = _read_inputs(input_path, profiles_path, toy)
    pipeline_parameters = dict(PIPELINE_PARAMETERS)
    if toy:
        pipeline_parameters["n_perm"] = 0

    # Figure 2b-c: reappraisal-minus-look trajectories and supported first derivatives.
    reappraisal = _run_contrast(ratings, pipeline_parameters)
    reappraisal = add_derivative_band_support_to_pipeline_output(
        reappraisal,
        alpha=0.05,
        center="median",
        smooth_sd_window=7,
        pre_window=1.5,
        post_window=1.5,
        min_segment_duration=0.25,
        min_segment_points=3,
    )
    contrast_name = "rpsl_minus_lkng"
    _save_trajectory(output_dir, reappraisal, contrast_name, "fig2b_reappraisal_success.svg")
    reappraisal["final_table"].to_csv(output_dir / "fig2b_turning_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(3, 3))
    for gender, color in zip(GENDER_LEVELS, ["#F28213", "#1B70AC"]):
        fig, ax = plot_tp_kde_from_pipeline(
            reappraisal,
            contrast_name=contrast_name,
            gender_value=gender,
            hist_color=color,
            tp_kind="peak",
            gender_label_map=GENDER_LABELS,
            ax=ax,
            show_final_tps=True,
            kde_kwargs={"color": color, "linewidth": 3},
            final_ci_kwargs={"color": color, "alpha": 0.0},
            final_line_kwargs={"color": color, "linewidth": 3, "linestyle": "--"},
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(3)
    ax.tick_params(axis="both", which="major", width=2, length=6, labelsize=22)
    fig.savefig(output_dir / "fig2b_turning_point_kde.svg", bbox_inches="tight")
    plt.close(fig)

    fit = reappraisal["results_by_contrast"][contrast_name]
    fig, ax = plt.subplots(figsize=(3.5, 3))
    fig, ax = plot_derivative_band_with_tps(
        result_one_contrast=fit,
        final_tps_one_contrast=fit["final_tps_with_deriv"],
        deriv_results=fit["derivative_results"],
        gender_label_map=GENDER_LABELS,
        show_supported_intervals=False,
        alpha=0.05,
        trajectory_line_kwargs={"linewidth": 3, "linestyle": "-"},
        tp_line_kwargs={"linewidth": 1, "linestyle": "--"},
        ax=ax,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([6, 8, 10, 12, 14, 16, 18])
    ax.tick_params(axis="both", labelsize=10)
    fig.savefig(output_dir / "fig2c_reappraisal_derivative.svg", bbox_inches="tight")
    plt.close(fig)
    fit["final_tps_with_deriv"].to_csv(output_dir / "fig2c_derivative_support.csv", index=False)

    # Figure 2d: normalize the reappraisal effect by passive-viewing ratings.
    paired = ratings.pivot(index="sub_id", columns="cognition", values="emot_rating")
    meta = ratings.drop_duplicates("sub_id").set_index("sub_id")[["age", "gender"]]
    relative_effect = ((paired["rpsl"] - paired["lkng"]) / paired["lkng"])
    relative = meta.join(relative_effect.rename("relative_success")).reset_index()
    fig2d = run_direct_outcome_gam_turning_point_pipeline(
        df=relative, dv_col="relative_success", age_col="age", subj_col="sub_id",
        group_col="gender", fixed_group_levels=GENDER_LEVELS,
        group_label_map=GENDER_LABELS, outcome_name="relative_success",
        lam_grid=LAM_GRID, n_splines=4, seed=2025, **pipeline_parameters,
    )
    _save_trajectory(output_dir, fig2d, "relative_success", "fig2d_relative_success.svg")

    # Figure 2e: ERQ cognitive-reappraisal strategy-use trajectory.
    fig2e = run_direct_outcome_gam_turning_point_pipeline(
        df=profiles, dv_col="ERQ_CR", age_col="age", subj_col="sub_id",
        group_col="sex", fixed_group_levels=GENDER_LEVELS,
        group_label_map=GENDER_LABELS, outcome_name="ERQ_CR",
        lam_grid=LAM_GRID, n_splines=4, seed=2025, **pipeline_parameters,
    )
    _save_trajectory(output_dir, fig2e, "ERQ_CR", "fig2e_ERQ_strategy_use.svg")

    (output_dir / "run_metadata.json").write_text(json.dumps({
        "figure": 2,
        "toy": toy,
        "panels": ["2b", "2c", "2d", "2e"],
        "gam_source": "analysis/fig02/gam_age_on_task.py",
        "pygam_version_required_by_source": "0.10.1",
        "lam_grid": LAM_GRID.tolist(),
        "n_splines": 4,
        "pipeline_parameters": pipeline_parameters,
        "relative_success_formula": "(reappraise - look) / look",
    }, indent=2), encoding="utf-8")
