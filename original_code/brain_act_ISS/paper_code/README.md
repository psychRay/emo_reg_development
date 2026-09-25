# Figure 6

From the project root:

```bash
python run.py --figure 6 --toy
python run.py --figure 6 --fig6-config brain_act_ISS/paper_code/config_paper.json
```

Set the `paths` in `config_paper.json` for participant data. Relative paths are
resolved from the config file. Set `matrix_dir` to `intermediate/` under the
desired output directory. The Schaefer dlabel is split into the left and
right GIFTI labels required by the original scripts when those paths are empty.
Supply the Tian subcortical atlas and Schaefer order-info text for mapping and
plot steps.

Panel outputs are organized as `fig6b/`, `fig6c/`, and `fig6d/` beside
`intermediate/` for participant runs. Toy runs remove intermediate products
after saving the panel results. Figure 6c exports maps for Connectome
Workbench. Figure 6d exports brain-map data and generates its bar, heatmap,
and age-window plots.

Use `python brain_act_ISS/paper_code/run_paper.py --list` for step names and
`--fig6-steps` on the root command to run a subset. The LSS step is disabled in
the config when single-trial betas already exist.
