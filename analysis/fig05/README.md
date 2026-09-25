# Figure 5

Run the low-cost synthetic workflow check:

```bash
python run.py --figure 5 --toy
```

For participant data, supply a directory containing:

```text
fig05_data/
|-- parc-cortex_subcortex_cond-rpsl.csv
|-- parc-cortex_subcortex_cond-lkng.csv
`-- subs_info.csv
```

The two parcel CSVs are headerless, with participant ID in column 0 and 232
parcel values in columns 1-232. Their row IDs must match `subs_info.csv`, which
requires `sub_id` (or `sub_ids`), `age`, `age_z`, `gender`, and `site`.

```bash
python run.py --figure 5 --fig5-data-dir path/to/fig05_data --output outputs/fig05/real
```

Real-data mode uses nested leave-one-site-out ROC-AUC for logistic L2, linear
SVC, and RBF SVC; 1,000 classifier permutations; 5,000 subject bootstraps for
decoder and Haufe maps; and the configured sex-modulated GAM and age permutation
settings. To reduce runtime for local checks, override counts with
`--fig5-permutations`, `--fig5-bootstraps`, and `--fig5-n-jobs`.

The runner exports CSV summaries, PNG plots, and CIFTI maps under
`outputs/fig05/{toy,real}/`. Use Connectome Workbench to render the surface maps.
The Schaefer and Yeo atlases default to the copies under `resources/`.
