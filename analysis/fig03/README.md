# Figure 3

Run the low-cost pipeline check with generated toy data:

```bash
python run.py --figure 3 --toy
```

For panels 3a/3d, provide a tab-delimited design matrix with one row per participant and columns `sub_id`, `age`, `gender`, `site_id`, and `meanFD`. Organize each participant's condition maps under `<volume-dir>/<sub_id>/` as `beta_0001.nii.gz` (reappraise), `beta_0002.nii.gz` (look), and `beta_0003.nii.gz` (neutral). Supply an MNI volume mask and the Schaefer surface dlabel:

```bash
python run.py --figure 3 --input path/to/fig03_design.tsv --fig3-volume-dir path/to/first_level_volumes --fig3-brain-mask path/to/MNI_mask.nii.gz --schaefer-atlas resources/atlases/schaefer2018/Schaefer2018_200Parcels_7Networks_order.dlabel.nii --output outputs/fig03/real
```

The Schaefer MNI volume atlas is read from `resources/atlases/schaefer2018/` by default. Panels 3b/3c/3e use existing GLM result maps:

Panels 3a/3d write surface GIFTI metrics and CIFTI dense scalar maps. The
runner uses Connectome Workbench `wb_command` from `PATH` to create border
files with `neuro_utils`. If it is installed elsewhere, pass its executable
path with `--fig3-workbench`.

```bash
python run.py --figure 3 --glm-results-dir path/to/res_glm --output outputs/fig03/volume-panels
```
