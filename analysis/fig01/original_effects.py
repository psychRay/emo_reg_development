"""Paired emotion regulation effect sizes by site for Figure 1d."""

import pandas as pd
import pingouin as pg


def cognition_contrast_effect_by_site(
    df,
    site_col="site",
    subject="subject",
    within="cognition",
    dv="y",
    level_a="cognition1",
    level_b="cognition2",
    ci_method="parametric",   # "parametric" (compute_esci) or "bootstrap"
    n_boot=5000,
    seed=42
):
    rows = []
    for site, d in df.groupby(site_col, observed=True):
        # Align paired observations by subject (and average if multiple rows per cell)
        wide = (d[d[within].isin([level_a, level_b])]
                .pivot_table(index=subject, columns=within, values=dv, aggfunc="mean"))
        if level_a not in wide.columns or level_b not in wide.columns:
            continue
        wide = wide[[level_a, level_b]].dropna()
        n = wide.shape[0]
        if n < 2:
            continue

        x = wide[level_a].to_numpy()
        y = wide[level_b].to_numpy()

        # Paired t-test (post-hoc test)
        t_res = pg.ttest(x, y, paired=True)
        t_row = t_res.iloc[0]
        tval = float(t_row["T"])
        dof = float(t_row["dof"])
        p = float(t_row["p_val"] if "p_val" in t_row else t_row["p-val"])

        # Effect size (Hedges g)
        g = float(pg.compute_effsize(x, y, paired=True, eftype="hedges"))

        # CI for effect size
        if ci_method == "parametric":
            ci_low, ci_high = pg.compute_esci(
                stat=g, nx=n, ny=n, paired=True, eftype="cohen",
                confidence=0.95, decimals=6
            )
        elif ci_method == "bootstrap":
            # bootstrap CI using a custom bivariate function
            def hedges_g(x_, y_):
                return pg.compute_effsize(x_, y_, paired=True, eftype="hedges")
            ci_low, ci_high = pg.compute_bootci(
                x, y, func=hedges_g, paired=True, n_boot=n_boot, seed=seed, method="percentile"
            )
        else:
            raise ValueError("ci_method must be 'parametric' or 'bootstrap'")

        rows.append({
            "site": site,
            "contrast": f"{level_a} - {level_b}",
            "n": n,
            "T": tval,
            "dof": dof,
            "p_unc": p,
            "g": g,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        })

    res = pd.DataFrame(rows)
    return res
