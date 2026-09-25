# Figure 2

Run Python panels 2b-2e from the project root:

```bash
python run.py --figure 2 --toy
python run.py --figure 2 --input path/to/behavior.csv --profiles path/to/er_profiles.csv --output outputs/fig02/real
```

The behavior CSV requires `sub_id`, `gender`, `age`, `cognition`, and
`emot_rating`, with one row per participant and condition (`rpsl`, `lkng`,
`lknt`). The profiles CSV requires `sub_id` (or `subject`), `age`, `sex`, and
`ERQ_CR`. Provide analysis-ready participants. Toy mode reduces the age
permutation count; real mode uses the configured manuscript count.

Run panel 2f separately. Set input and output paths in the R scripts, then run:

```bash
Rscript analysis/fig02/mediation/gam_mediation_age_crosssectional.R
Rscript analysis/fig02/mediation/mediation_direct_effect_forest.R
```

Python outputs are saved under `outputs/fig02/{toy,real}/`.
