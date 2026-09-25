# Figure 4

Run the toy pipeline from the project root:

```bash
python run.py --figure 4 --toy
```

Real-data mode requires a participant CSV and the age-effect and mediation ROI
masks:

```bash
python run.py --figure 4 --input path/to/fig04_participants.csv --fig4-age-positive-mask path/to/positive_age_fdr_mask.nii.gz --fig4-age-negative-mask path/to/negative_age_fdr_mask.nii.gz --fig4-mediator-mask-dir path/to/aMTG_aSTG_masks --output outputs/fig04/real
```

The participant CSV requires `subject_id`, `age`, `gender`, `site`,
`reappraisal_success`, and `rpsl`, `lkng`, `lknt` columns containing NIfTI beta
map paths. The mediator mask directory must contain `aMTG.nii.gz` and
`aSTG.nii.gz`. Age masks must already be thresholded using the manuscript FDR
procedure. The Yeo volume atlas defaults to `resources/atlases/yeo2011/`.

The runner exports the system and age-overlap masks for MRIcroGL/Surfice. Panel
4a and the mediation diagram in 4e are manually drawn. Outputs are saved under
`outputs/fig04/{toy,real}/`.
