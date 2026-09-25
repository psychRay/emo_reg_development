"""Figure 1b-d: participant age, emotion ratings, and site-level reappraisal effects."""

from __future__ import annotations

import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROJECT_LIB = Path(__file__).resolve().parents[2] / "src" / "prj_analysis" / "src"
if str(PROJECT_LIB) not in sys.path:
    sys.path.insert(0, str(PROJECT_LIB))
from mylib.plotting.beh_plot import plot_condition_boxplots

import pingouin as pg


CONDITIONS = ("rpsl", "lkng", "lknt")
AREAS = ("East China", "Middle China", "North China", "South China", "West China")
AREA_ALIASES = {
    "EC": "East China", "MC": "Middle China", "NC": "North China",
    "SC": "South China", "WC": "West China",
}
REQUIRED = {"sub_id", "age", "gender", "site_id", "site_region", "cognition", "emot_rating"}


def make_toy_data(seed: int = 20260925) -> pd.DataFrame:
    """Generate a balanced, synthetic five-area/eight-site repeated-measures sample."""
    rng = np.random.default_rng(seed)
    sites = (
        ("WC", "West China"), ("SC", "South China"),
        ("NC", "North China"), ("MC", "Middle China"),
        ("EC", "East China"), ("EC2", "East China"),
        ("MC2", "Middle China"), ("NC2", "North China"),
    )
    rows: list[dict] = []
    for site_index, (site_id, area) in enumerate(sites):
        for person in range(32):
            sub_id = f"toy-{site_index:02d}-{person:03d}"
            age = float(np.clip(6.2 + 12.2 * (person // 2) / 15 + rng.normal(0, 0.22), 6, 18.9))
            gender = person % 2
            baseline = rng.normal(0, 0.55)
            site_shift = 0.045 * (site_index - 3.5)
            rating = {
                "rpsl": 2.85 + baseline + site_shift + rng.normal(0, 0.65),
                "lkng": 1.88 + baseline + site_shift + rng.normal(0, 0.65),
                "lknt": 3.05 + baseline + site_shift + rng.normal(0, 0.65),
            }
            for condition in CONDITIONS:
                rows.append({
                    "sub_id": sub_id,
                    "age": round(age, 3),
                    "gender": gender,
                    "site_id": site_id,
                    "site_region": area,
                    "cognition": condition,
                    "emot_rating": float(np.clip(rating[condition], 1, 4)),
                })
    return pd.DataFrame(rows)


def validate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require complete paired ratings and constant metadata within each subject."""
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    df = df[list(REQUIRED)].copy()
    df["site_region"] = df["site_region"].replace(AREA_ALIASES)
    unknown_areas = set(df["site_region"]) - set(AREAS)
    if unknown_areas:
        raise ValueError(f"Unknown site_region values: {sorted(unknown_areas)}")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["emot_rating"] = pd.to_numeric(df["emot_rating"], errors="coerce")
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("Figure 1 input contains missing required values")
    if not np.isfinite(df["age"]).all():
        raise ValueError("Age must be numeric and finite")
    if not np.isfinite(df["emot_rating"]).all():
        raise ValueError("Emotion ratings must be numeric and finite")
    if set(df["cognition"].unique()) != set(CONDITIONS):
        raise ValueError(f"Conditions must be exactly {CONDITIONS}")
    counts = df.groupby(["sub_id", "cognition"], observed=True).size()
    if not counts.eq(1).all() or not df.groupby("sub_id", observed=True).size().eq(3).all():
        raise ValueError("Every subject needs exactly one observation per condition")
    metadata = ["age", "gender", "site_id", "site_region"]
    if not df.groupby("sub_id", observed=True)[metadata].nunique().eq(1).all().all():
        raise ValueError("Subject metadata differs across condition rows")
    people = df.drop_duplicates("sub_id")["sub_id age gender site_id site_region".split()].copy()
    if set(people["site_region"]) != set(AREAS):
        raise ValueError("Figure 1 expects participants from all five site regions")
    if people.groupby("site_id")["site_region"].nunique().gt(1).any():
        raise ValueError("A site_id cannot belong to multiple areas")
    if people["gender"].nunique() != 2:
        raise ValueError("Figure 1 mixed ANOVA expects exactly two gender groups")
    wide = df.pivot(index="sub_id", columns="cognition", values="emot_rating")
    return people, wide.loc[:, list(CONDITIONS)]


def age_statistics(people: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Figure 1b: sex and area age comparisons, one record per participant."""
    sexes = list(pd.unique(people["gender"]))
    x = people.loc[people.gender == sexes[0], "age"].to_numpy(dtype=float)
    y = people.loc[people.gender == sexes[1], "age"].to_numpy(dtype=float)
    sex_test = stats.ttest_ind(x, y, equal_var=False)
    area_aov = pg.anova(data=people, dv="age", between="site_region", detailed=True)
    area_stat = float(area_aov.loc[area_aov["Source"] == "site_region", "F"].iloc[0])
    area_p = float(area_aov.loc[area_aov["Source"] == "site_region", "p-unc"].iloc[0])
    overall = pd.DataFrame([
        {"comparison": "gender", "statistic": float(sex_test.statistic),
         "p_value": float(sex_test.pvalue), "n": len(people)},
        {"comparison": "site_region", "statistic": area_stat,
         "p_value": area_p, "n": len(people)},
    ])
    original = pg.pairwise_tests(data=people, dv="age", between="site_region",
                                 parametric=True, padjust="fdr_bh", effsize="hedges")
    pair_rows = []
    for a, b in combinations([area for area in AREAS if area in set(people.site_region)], 2):
        match = original.loc[((original.A == a) & (original.B == b)) |
                             ((original.A == b) & (original.B == a))]
        if len(match) != 1:
            raise ValueError("The Figure 1b area comparison is missing or duplicated")
        pair_rows.append({"area_a": a, "area_b": b, "t": float(match["T"].iloc[0]),
                          "p_unc": float(match["p-unc"].iloc[0]),
                          "p_fdr_bh": float(match["p-corr"].iloc[0])})
    pairwise = pd.DataFrame(pair_rows)
    return overall, pairwise


def mixed_anova(people: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Figure 1c: condition within subject and gender between subjects ANOVA."""
    long = wide.stack().rename("emot_rating").reset_index().rename(columns={"level_1": "cognition"})
    long = long.merge(people[["sub_id", "gender"]], on="sub_id", validate="many_to_one")
    result = pg.mixed_anova(data=long, dv="emot_rating", within="cognition",
                            between="gender", subject="sub_id",
                            correction="auto", effsize="np2")
    return result.rename(columns={"Source": "term", "SS": "ss", "DF1": "df1",
                                  "DF2": "df2", "p-unc": "p_value",
                                  "np2": "partial_eta2"})


def condition_posthoc(wide: pd.DataFrame) -> pd.DataFrame:
    """Figure 1c: paired t tests, Cohen dz, Holm across three contrasts."""
    rows = []
    for a, b in (("rpsl", "lkng"), ("lkng", "lknt"), ("rpsl", "lknt")):
        diff = (wide[a] - wide[b]).to_numpy(dtype=float)
        test = stats.ttest_rel(wide[a], wide[b])
        rows.append({"condition_a": a, "condition_b": b, "n": len(diff),
                     "mean_difference": float(diff.mean()), "t": float(test.statistic),
                     "df": len(diff) - 1, "p_unc": float(test.pvalue),
                     "cohen_dz": float(diff.mean() / diff.std(ddof=1))})
    result = pd.DataFrame(rows)
    result["p_holm"] = multipletests(result["p_unc"], method="holm")[1]
    return result


def site_effects(people: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Figure 1d: paired Hedges g, 95% CI, and inverse-variance pooled effect."""
    merged = people.set_index("sub_id").join(wide)
    from analysis.fig01.original_effects import cognition_contrast_effect_by_site

    long = (merged.reset_index()
            .melt(id_vars=["sub_id", "site_region"], value_vars=["rpsl", "lkng"],
                  var_name="cognition", value_name="emot_rating"))
    original = cognition_contrast_effect_by_site(
        long, site_col="site_region", subject="sub_id", within="cognition",
        dv="emot_rating", level_a="rpsl", level_b="lkng", ci_method="parametric"
    )
    result = original.rename(columns={"site": "site_region"})
    result["ci_se"] = (result.ci_high - result.ci_low) / (2 * 1.96)
    result["site_region"] = pd.Categorical(result.site_region, categories=AREAS, ordered=True)
    result = result.sort_values("site_region").reset_index(drop=True)
    result["site_region"] = result["site_region"].astype(str)
    pooled_se_by_area = (result["ci_high"] - result["ci_low"]).to_numpy(dtype=float) / (2 * 1.96)
    weights = 1 / np.square(pooled_se_by_area)
    pooled = float(np.average(result["g"], weights=weights))
    se = float(np.sqrt(1 / weights.sum()))
    result.loc[len(result)] = {"site_region": "Pooled", "n": int(result.n.sum()),
                               "g": pooled, "ci_low": pooled - 1.96 * se,
                               "ci_high": pooled + 1.96 * se, "ci_se": se}
    return result


def plot_fig1b(people: pd.DataFrame, pairwise: pd.DataFrame, output: Path) -> None:
    """Fig. 1b: show age distributions by gender and area with pairwise area p values."""
    fig = plt.figure(figsize=(14, 4.8))
    ax_hist = fig.add_axes([0.06, 0.16, 0.31, 0.73])
    ax_ridge = fig.add_axes([0.48, 0.16, 0.29, 0.73])
    ax_heat = fig.add_axes([0.83, 0.22, 0.14, 0.62])
    palette = {"Male": "#f68724", "Female": "#3278a7"}
    gender_names = people.gender.map({0: "Male", 1: "Female", "M": "Male", "F": "Female"})
    if gender_names.isna().any():
        raise ValueError("Fig. 1b requires gender coded as 0/1 or M/F")
    ages = people.age.to_numpy(dtype=float)
    bins = np.linspace(min(6, ages.min()), max(19, ages.max()), 26)
    left = people.loc[gender_names == "Male", "age"].to_numpy(dtype=float)
    right = people.loc[gender_names == "Female", "age"].to_numpy(dtype=float)
    # Fig. 1b displays 25-bin stacked densities with gender-specific KDE curves.
    h_m, _ = np.histogram(left, bins=bins, density=True)
    h_f, _ = np.histogram(right, bins=bins, density=True)
    width = np.diff(bins)
    ax_hist.bar(bins[:-1], h_m, width=width, align="edge", color=palette["Male"],
                alpha=.55, edgecolor="black", linewidth=.8, label="Male")
    ax_hist.bar(bins[:-1], h_f, width=width, bottom=h_m, align="edge",
                color=palette["Female"], alpha=.55, edgecolor="black",
                linewidth=.8, label="Female")
    grid = np.linspace(bins[0], bins[-1], 500)
    for label, sample in (("Male", left), ("Female", right)):
        if len(sample) > 2 and np.std(sample) > 0:
            ax_hist.plot(grid, stats.gaussian_kde(sample)(grid),
                         color=palette[label], linewidth=2.5)
    ax_hist.set_title("Age distributions by gender", fontweight="bold")
    ax_hist.set_xlabel("Age (years)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(frameon=False, loc="upper left")
    for i, area in enumerate(AREAS):
        area_mask = people.site_region == area
        for label in ("Male", "Female"):
            sample = people.loc[area_mask & (gender_names == label), "age"].to_numpy(dtype=float)
            if len(sample) > 2 and np.std(sample) > 0:
                density = stats.gaussian_kde(sample)(grid)
                density = density / density.max() * .7
                baseline = len(AREAS) - 1 - i
                ax_ridge.fill_between(grid, baseline, baseline + density,
                                      color=palette[label], alpha=.6)
                ax_ridge.plot(grid, baseline + density, color="0.35", linewidth=.8)
        ax_ridge.hlines(len(AREAS) - 1 - i, grid[0], grid[-1], color="0.55", linewidth=.7)
    ax_ridge.set_yticks(range(len(AREAS)), list(reversed(AREAS)))
    ax_ridge.set_xlim(grid[0], grid[-1])
    ax_ridge.set_ylim(-.1, len(AREAS) - .1 + .9)
    ax_ridge.set_title("Age distributions by gender per site/area", fontweight="bold")
    ax_ridge.set_xlabel("Age (years)")
    ax_ridge.tick_params(axis="y", length=0)
    matrix = np.ones((len(AREAS), len(AREAS)))
    for row in pairwise.itertuples(index=False):
        i, j = AREAS.index(row.area_a), AREAS.index(row.area_b)
        matrix[i, j] = matrix[j, i] = row.p_unc
    matrix = matrix[::-1, ::-1]
    heat = ax_heat.imshow(np.ma.masked_array(matrix, mask=np.tril(np.ones_like(matrix, dtype=bool), -1)),
                          vmin=0, vmax=1, cmap="viridis_r")
    abbreviations = ("WC", "SC", "NC", "MC", "EC")
    ax_heat.set_xticks(range(len(AREAS)), abbreviations, rotation=90)
    ax_heat.set_yticks(range(len(AREAS)), abbreviations)
    for i in range(len(AREAS)):
        for j in range(len(AREAS)):
            if j >= i:
                ax_heat.text(j, i, f"{matrix[i, j]:.2g}", ha="center", va="center",
                             fontsize=8, color="white" if matrix[i, j] > .45 else "black")
    fig.colorbar(heat, ax=ax_heat, label="Pairwise p", shrink=.65)
    ax_heat.set_title("Pairwise T tests", fontsize=10)
    for ax in (ax_hist, ax_ridge):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fig1c(df: pd.DataFrame, posthoc: pd.DataFrame, output: Path) -> None:
    """Fig. 1c: show condition ratings by gender and rating distributions."""
    source = df.copy()
    source["cognition"] = source.cognition.str.upper()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")
        fig, ax = plot_condition_boxplots(
            source, condition_col="cognition", score_col="emot_rating",
            subject_col="sub_id", condition_order=["RPSL", "LKNG", "LKNT"],
            group_col="gender", group_order=[0, 1], figsize=(6, 4.5),
            add_points=True, point_style="jitter", points_layer="behind",
            point_size=10, point_jitter=.08, point_palette=["gray", "gray"],
            point_edgecolor="k", point_jitter_main_axis=.1,
            box_width=.22, box_colors={0: "#f68724", 1: "#3278a7"},
            show_counts=False,
        )
    fig.set_size_inches(10, 4.5)
    ax.set_position([.08, .18, .49, .68])
    ax.set_title("")
    ax.set_xticks(range(3), ["Reappraise", "Look", "Neutral"])
    ax.set_ylabel("Emotion ratings")
    ax.set_ylim(.7, 4.55)
    for left, right, height, a, b in ((0, 1, 4.25, "rpsl", "lkng"),
                                       (1, 2, 4.45, "lkng", "lknt")):
        match = posthoc.loc[(posthoc.condition_a == a) & (posthoc.condition_b == b)]
        if len(match) == 1 and float(match.p_holm.iloc[0]) < .001:
            ax.plot([left, right], [height, height], color="0.2", linewidth=1)
            ax.text((left + right) / 2, height + .02, "***", ha="center", va="bottom", fontsize=13)
    ax.legend(handles=[Patch(facecolor="#f68724", label="Male"),
                       Patch(facecolor="#3278a7", label="Female")],
              frameon=False, loc="upper center", bbox_to_anchor=(.5, -.09), ncol=2)
    density_ax = fig.add_axes([.65, .42, .31, .4])
    grid = np.linspace(0.8, 4.2, 250)
    for condition, color in (("rpsl", "#73C477"), ("lkng", "#E17F7E"), ("lknt", "#867cd8")):
        ratings = df.loc[df.cognition == condition, "emot_rating"].to_numpy(dtype=float)
        density_ax.plot(grid, stats.gaussian_kde(ratings)(grid), color=color,
                        label={"rpsl": "Reappraise", "lkng": "Look", "lknt": "Neutral"}[condition])
    density_ax.set_xlabel("Emotion ratings")
    density_ax.set_ylabel("Density")
    density_ax.legend(frameon=False, fontsize=8)
    density_ax.spines["top"].set_visible(False)
    density_ax.spines["right"].set_visible(False)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fig1d(effects: pd.DataFrame, output: Path) -> None:
    """Figure 1d: area-specific and inverse-variance pooled Hedges g."""
    fig = plt.figure(figsize=(8, 4.5))
    sample_ax = fig.add_axes([.04, .16, .33, .71])
    ax = fig.add_axes([.41, .16, .55, .71])
    for i, row in effects.reset_index(drop=True).iterrows():
        ax.plot([row.ci_low, row.ci_high], [i, i], color="#82b78b", lw=10, alpha=0.7)
        ax.scatter([row.g], [i], s=95, color="white" if row.site_region == "Pooled" else "#17763d",
                   edgecolor="red" if row.site_region == "Pooled" else "white", zorder=3)
    ax.set_yticks(range(len(effects)))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(.4, max(1.6, float(effects.ci_high.max()) + .05))
    ax.set_xticks([.4, .8, 1.2, 1.6])
    ax.grid(axis="x", color="0.85", linestyle="--", linewidth=.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Effect sizes with 95% CIs (Hedge’s g)")
    ax.set_title("Effect size of reappraisal per site/area", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    area_rows = effects.loc[effects.site_region != "Pooled"]
    sample_ax.set_xlim(0, 100)
    sample_ax.set_ylim(len(effects) - .5, -.5)
    max_n = float(area_rows.n.max())
    for i, row in enumerate(area_rows.itertuples()):
        sample_ax.text(0, i, row.site_region, ha="left", va="center", fontsize=9)
        sample_ax.barh(i, 32 * row.n / max_n, left=65, height=.42,
                       color="#cbd9bd", edgecolor="0.3")
        sample_ax.text(95, i, str(int(row.n)), ha="right", va="center", fontsize=8)
    sample_ax.text(0, len(effects) - 1, "Pooled", ha="left", va="center",
                   fontsize=10, fontweight="bold")
    sample_ax.set_yticks([])
    sample_ax.set_xticks([])
    sample_ax.set_title("Sample sizes", fontsize=10, fontweight="bold", loc="right")
    for spine in sample_ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_figure_1(*, input_path: Path | None, toy: bool, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if toy:
        df = make_toy_data()
    else:
        if input_path is None:
            raise ValueError("An input CSV is required outside toy mode")
        df = pd.read_csv(input_path)
    people, wide = validate_data(df)
    age_overall, age_pairwise = age_statistics(people)
    anova = mixed_anova(people, wide)
    posthoc = condition_posthoc(wide)
    effects = site_effects(people, wide)

    # Fig. 1b: demographic comparisons and distribution plot.
    age_overall.to_csv(output_dir / "fig1b_age_overall.csv", index=False)
    age_pairwise.to_csv(output_dir / "fig1b_age_pairwise.csv", index=False)
    plot_fig1b(people, age_pairwise, output_dir / "fig1b_age_distribution.svg")

    # Fig. 1c: mixed ANOVA, paired contrasts, and rating distribution plot.
    anova.to_csv(output_dir / "fig1c_mixed_anova.csv", index=False)
    posthoc.to_csv(output_dir / "fig1c_condition_posthoc.csv", index=False)
    plot_fig1c(df, posthoc, output_dir / "fig1c_ratings.svg")

    # Fig. 1d: area-level and pooled paired effect sizes.
    effects.to_csv(output_dir / "fig1d_site_effects.csv", index=False)
    plot_fig1d(effects, output_dir / "fig1d_site_forest.svg")

    (output_dir / "run_metadata.json").write_text(json.dumps({
        "figure": 1,
        "toy": toy,
        "n_subjects": len(people),
        "n_sites": int(people.site_id.nunique()),
        "n_areas": int(people.site_region.nunique()),
        "conditions": list(CONDITIONS),
        "site_effect_ci": "paired parametric CI using Pingouin compute_esci formula",
        "statistics_backend": "pingouin",
        "seed": 20260925,
    }, indent=2), encoding="utf-8")
