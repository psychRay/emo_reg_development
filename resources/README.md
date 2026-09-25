# Resources and real-data inputs

Place shared atlases, reference maps, and surface templates under this folder.
The analysis runners use the following paths by default:

| Resource | Used by |
|---|---|
| `atlases/schaefer2018/Schaefer2018_200Parcels_7Networks_order.dlabel.nii` | Fig. 3 and Fig. 5 surface parcel mapping |
| `atlases/schaefer2018/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz` | Fig. 4 volume-space parcel mapping |
| `atlases/yeo2011/Yeo2011_7Networks_N1000.dlabel.nii` | Fig. 3, 4, and 5 surface network analysis |
| `atlases/yeo2011/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz` | Fig. 4 volume network analysis |
| `reference_maps/` | Fig. 3 task-map comparisons and Fig. 5 reference maps |
| `templates/fslr_32k/` | Surface display and Workbench exports |

Real participant data are provided separately. Figure 3 volume panels require
whole-brain and age-bin GLM maps. Figure 4 requires participant beta maps,
positive and negative FDR age masks, and a directory containing `aMTG.nii.gz`
and `aSTG.nii.gz`. Figure 5 requires the two parcel response CSVs and
`subs_info.csv` described in [its input guide](../analysis/fig05/README.md).

Neuromaps may fetch MNI-to-fsLR registration files on first use if they are not
cached locally. Final brain-map rendering uses Connectome Workbench,
MRIcroGL, or Surfice as specified in the figure guides.
