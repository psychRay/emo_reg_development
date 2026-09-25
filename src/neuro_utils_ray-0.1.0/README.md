# neuro-utils-ray

Brain-space mapping utilities used by the manuscript analysis runners. Install
the project environment from the repository root with `environment.yml`; this
package is imported from `src/neuro_utils_ray-0.1.0/` by the runners.

```python
from neuro_utils.parcellation import brain_to_parcel, parcel_to_brain

values, info = brain_to_parcel(
    brain_data="map.dscalar.nii",
    parcellation="atlas.dlabel.nii",
    background=0,
    reduction="mean",
)
image, parcel_ids = parcel_to_brain(
    parcel_values=values,
    parcellation="atlas.dlabel.nii",
    parcel_ids=info["parcel_ids"],
    background=0,
    fill_value=float("nan"),
)
image.to_filename("parcel_map.dscalar.nii")
```

Input maps and atlases must already share coordinate space, resolution, density,
and grayordinate ordering. The utilities do not register or resample images.
