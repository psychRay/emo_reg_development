from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.cifti2 import Cifti2Image
from nibabel.cifti2.cifti2_axes import BrainModelAxis, ScalarAxis


def make_subcortex_display_mask_cifti(
    template_cifti,
    output_file=None,
    cortex_value=np.nan,
    subcortex_value=1.0,
    include_cerebellum=True,
    mask_name="subcortex_display_mask",
    dtype=np.float32,
):
    """
    Create a CIFTI dscalar image for display-only subcortex masking.

    The output image has:
        cortical grayordinates      = cortex_value, usually NaN
        subcortical grayordinates   = subcortex_value, usually 1

    This is intended for visualization overlay, e.g., rendering subcortical
    grayordinates as black while keeping the main cortical statistical map intact.

    Parameters
    ----------
    template_cifti : str, Path, or nib.Cifti2Image
        A CIFTI image in the target grayordinate space, e.g. fsLR 32k
        .dscalar.nii, .dtseries.nii, or .dlabel.nii.

        The template must contain a BrainModelAxis. The output mask will use
        exactly the same BrainModelAxis.

    output_file : str, Path, or None
        If provided, save the generated CIFTI image to this path.

    cortex_value : float
        Value assigned to left and right cortical grayordinates.
        Default is NaN, which is usually best for overlay masks.

    subcortex_value : float
        Value assigned to non-cortical grayordinates.
        Default is 1.0.

    include_cerebellum : bool
        If True, cerebellum is treated as subcortical/non-cortical display mask.
        If False, cerebellum is assigned cortex_value instead.

    mask_name : str
        Scalar map name in the output CIFTI.

    dtype : numpy dtype
        Output data dtype.

    Returns
    -------
    mask_img : nib.Cifti2Image
        A dscalar-like CIFTI image with shape (1, n_grayordinates).

    info : dict
        Summary information, including included structures and counts.

    Notes
    -----
    This function does not modify the main statistical image. It creates a
    display-only overlay mask.

    CIFTI medial wall vertices are typically absent from the BrainModelAxis,
    so this function masks subcortical/non-cortical grayordinates, not true
    surface medial wall vertices.
    """
    if isinstance(template_cifti, (str, Path)):
        img = nib.load(str(template_cifti))
    else:
        img = template_cifti

    if not isinstance(img, nib.Cifti2Image):
        raise TypeError("template_cifti must be a CIFTI image or CIFTI filepath.")

    axes = [img.header.get_axis(i) for i in range(len(img.shape))]

    brain_axis_index = None
    brain_axis = None

    for i, axis in enumerate(axes):
        if isinstance(axis, BrainModelAxis):
            brain_axis_index = i
            brain_axis = axis
            break

    if brain_axis is None:
        raise ValueError("No BrainModelAxis found in template_cifti.")

    structure_names = np.asarray(brain_axis.name)

    cortex_structures = {
        "CIFTI_STRUCTURE_CORTEX_LEFT",
        "CIFTI_STRUCTURE_CORTEX_RIGHT",
    }

    cerebellum_structures = {
        "CIFTI_STRUCTURE_CEREBELLUM",
        "CIFTI_STRUCTURE_CEREBELLUM_LEFT",
        "CIFTI_STRUCTURE_CEREBELLUM_RIGHT",
    }

    is_cortex = np.isin(structure_names, list(cortex_structures))

    if include_cerebellum:
        is_subcortex = ~is_cortex
    else:
        is_cerebellum = np.isin(structure_names, list(cerebellum_structures))
        is_subcortex = (~is_cortex) & (~is_cerebellum)

    mask_data = np.full(
        (1, len(brain_axis)),
        cortex_value,
        dtype=dtype,
    )

    mask_data[0, is_subcortex] = subcortex_value

    scalar_axis = ScalarAxis([mask_name])

    mask_img = Cifti2Image(
        dataobj=mask_data,
        header=(scalar_axis, brain_axis),
    )

    unique_structures, counts = np.unique(structure_names, return_counts=True)

    info = {
        "brain_axis_index": brain_axis_index,
        "n_grayordinates": len(brain_axis),
        "n_cortex": int(np.sum(is_cortex)),
        "n_subcortex_masked": int(np.sum(is_subcortex)),
        "include_cerebellum": include_cerebellum,
        "structures": dict(zip(unique_structures.tolist(), counts.tolist())),
        "masked_structures": sorted(np.unique(structure_names[is_subcortex]).tolist()),
        "cortex_structures": sorted(np.unique(structure_names[is_cortex]).tolist()),
    }

    if output_file is not None:
        nib.save(mask_img, str(output_file))

    return mask_img, info