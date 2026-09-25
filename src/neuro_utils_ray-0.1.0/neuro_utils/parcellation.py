"""
CIFTI/GIFTI parcellation utilities.

This module provides two high-level functions:

- brain_to_parcel: convert CIFTI/GIFTI brain-space data to parcel-level values.
- parcel_to_brain: map parcel-level values back to CIFTI/GIFTI brain space.

The functions do not perform spatial registration or resampling. Brain data and
parcellation must already have matching space, density/resolution, and
vertex/grayordinate ordering.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
import nibabel as nib
from nibabel.cifti2 import Cifti2Image
from nibabel.cifti2.cifti2_axes import BrainModelAxis, ScalarAxis


Reduction = Literal["mean", "median", "sum"]
Mode = Literal["to_parcel", "to_brain"]
OutputFormat = Literal["cifti", "gifti"]


def _load_img(img: Any) -> Any:
    """Load image if a filepath is provided."""
    if isinstance(img, (str, os.PathLike)):
        return nib.load(str(img))
    return img


def _is_cifti(img: Any) -> bool:
    """Check whether an image is a CIFTI image."""
    img = _load_img(img)
    return isinstance(img, nib.Cifti2Image)


def _is_gifti(img: Any) -> bool:
    """Check whether an image is a GIFTI image."""
    img = _load_img(img)
    return isinstance(img, nib.GiftiImage)


def _as_tuple(x: Any) -> tuple[Any, ...]:
    """Convert single image/path or tuple/list into a tuple."""
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x,)


def _get_gifti_array(img: Any) -> np.ndarray:
    """
    Extract data array from a GIFTI image.

    For most .func.gii or .label.gii files, the first darray is sufficient.
    If there are multiple darrays, they are stacked as columns.
    """
    img = _load_img(img)
    if not isinstance(img, nib.GiftiImage):
        raise TypeError("Expected a GIFTI image or a GIFTI filepath.")

    arrays = [np.asarray(d.data) for d in img.darrays]

    if len(arrays) == 0:
        raise ValueError("The GIFTI image contains no data arrays.")

    if len(arrays) == 1:
        return arrays[0]

    return np.column_stack(arrays)


def _make_gifti_array(data: np.ndarray) -> nib.GiftiImage:
    """Create a simple GIFTI image from a 1D or 2D array."""
    data = np.asarray(data, dtype=np.float32)

    gii = nib.GiftiImage()

    if data.ndim == 1:
        gii.add_gifti_data_array(nib.gifti.GiftiDataArray(data))
    elif data.ndim == 2:
        for i in range(data.shape[1]):
            gii.add_gifti_data_array(nib.gifti.GiftiDataArray(data[:, i]))
    else:
        raise ValueError("GIFTI output data must be 1D or 2D.")

    return gii


def _get_cifti_axes(img: Any) -> list[Any]:
    """Return all CIFTI axes from a CIFTI image."""
    img = _load_img(img)
    header = img.header
    return [header.get_axis(i) for i in range(len(img.shape))]


def _find_brainmodel_axis(img: Any) -> tuple[int, BrainModelAxis]:
    """Find the BrainModelAxis and its dimension index in a CIFTI image."""
    axes = _get_cifti_axes(img)

    for i, axis in enumerate(axes):
        if isinstance(axis, BrainModelAxis):
            return i, axis

    raise ValueError("No BrainModelAxis found in the CIFTI image.")


def _move_cifti_brain_axis_to_last(data: np.ndarray, brain_axis_index: int) -> np.ndarray:
    """
    Move CIFTI brain model axis to the last dimension.

    Output shape is (..., n_grayordinates).
    """
    return np.moveaxis(data, brain_axis_index, -1)


def _get_cifti_label_vector(parcellation: Any) -> tuple[np.ndarray, BrainModelAxis]:
    """
    Extract a 1D label vector from a CIFTI dlabel-like image.

    Assumes one dimension is BrainModelAxis and the remaining dimension(s)
    contain label maps. If multiple label maps exist, the first one is used.
    """
    parc = _load_img(parcellation)

    if not isinstance(parc, nib.Cifti2Image):
        raise TypeError("Expected a CIFTI parcellation image or filepath.")

    brain_axis_index, brain_axis = _find_brainmodel_axis(parc)

    labels = np.asarray(parc.get_fdata())
    labels = _move_cifti_brain_axis_to_last(labels, brain_axis_index)

    if labels.ndim == 1:
        label_vec = labels
    else:
        # Use the first label map if multiple maps exist.
        label_vec = labels.reshape(-1, labels.shape[-1])[0]

    return label_vec.astype(float), brain_axis


def _get_cifti_data_matrix(data_img: Any) -> tuple[np.ndarray, BrainModelAxis, int, tuple[int, ...], list[Any]]:
    """
    Extract CIFTI data as a 2D matrix.

    Returns
    -------
    data_matrix : ndarray, shape (n_maps, n_grayordinates)
        For a single scalar map, n_maps = 1.
    brain_axis : BrainModelAxis
        CIFTI brain model axis.
    brain_axis_index : int
        Original dimension index of the BrainModelAxis.
    nonbrain_shape : tuple
        Original shape of all non-brain dimensions.
    axes : list
        Original CIFTI axes.
    """
    img = _load_img(data_img)

    if not isinstance(img, nib.Cifti2Image):
        raise TypeError("Expected a CIFTI data image or filepath.")

    axes = _get_cifti_axes(img)
    brain_axis_index, brain_axis = _find_brainmodel_axis(img)

    data = np.asarray(img.get_fdata())
    data = _move_cifti_brain_axis_to_last(data, brain_axis_index)

    n_grayordinates = data.shape[-1]
    nonbrain_shape = data.shape[:-1]

    if len(nonbrain_shape) == 0:
        data_matrix = data.reshape(1, n_grayordinates)
    else:
        data_matrix = data.reshape(-1, n_grayordinates)

    return data_matrix, brain_axis, brain_axis_index, nonbrain_shape, axes


def _valid_parcel_ids(labels: np.ndarray, background: int | float | None = 0) -> np.ndarray:
    """Return sorted parcel IDs excluding background and NaN."""
    labels = np.asarray(labels)

    valid = labels[~np.isnan(labels)]

    if background is not None:
        valid = valid[valid != background]

    parcel_ids = np.unique(valid).astype(int)
    parcel_ids = np.sort(parcel_ids)

    return parcel_ids


def _parcellate_matrix(
    data_matrix: np.ndarray,
    labels: np.ndarray,
    background: int | float | None = 0,
    ignore_nan: bool = True,
    reduction: Reduction = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parcellate a 2D data matrix.

    Parameters
    ----------
    data_matrix : ndarray, shape (n_maps, n_vertices_or_grayordinates)
    labels : ndarray, shape (n_vertices_or_grayordinates,)
    background : int, float, or None
        Label treated as background.
    ignore_nan : bool
        Whether to ignore NaN values during parcel aggregation.
    reduction : {"mean", "median", "sum"}
        Parcel-wise aggregation function.

    Returns
    -------
    parcel_values : ndarray, shape (n_maps, n_parcels)
    parcel_ids : ndarray, shape (n_parcels,)
    """
    data_matrix = np.asarray(data_matrix, dtype=float)
    labels = np.asarray(labels, dtype=float)

    if data_matrix.shape[-1] != labels.shape[0]:
        raise ValueError(
            f"Data length ({data_matrix.shape[-1]}) and label length "
            f"({labels.shape[0]}) do not match."
        )

    parcel_ids = _valid_parcel_ids(labels, background=background)
    parcel_values = np.full((data_matrix.shape[0], len(parcel_ids)), np.nan, dtype=float)

    for j, pid in enumerate(parcel_ids):
        mask = labels == pid
        values = data_matrix[:, mask]

        if reduction == "mean":
            if ignore_nan:
                parcel_values[:, j] = np.nanmean(values, axis=1)
            else:
                parcel_values[:, j] = np.mean(values, axis=1)
        elif reduction == "median":
            if ignore_nan:
                parcel_values[:, j] = np.nanmedian(values, axis=1)
            else:
                parcel_values[:, j] = np.median(values, axis=1)
        elif reduction == "sum":
            if ignore_nan:
                parcel_values[:, j] = np.nansum(values, axis=1)
            else:
                parcel_values[:, j] = np.sum(values, axis=1)
        else:
            raise ValueError("reduction must be one of {'mean', 'median', 'sum'}.")

    if parcel_values.shape[0] == 1:
        parcel_values = parcel_values[0]

    return parcel_values, parcel_ids


def _inverse_parcellate_vector(
    parcel_values: np.ndarray,
    labels: np.ndarray,
    parcel_ids: np.ndarray | None = None,
    background: int | float | None = 0,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map parcel-level values back to vertices/grayordinates.

    Parameters
    ----------
    parcel_values : ndarray, shape (n_parcels,) or (n_maps, n_parcels)
    labels : ndarray, shape (n_vertices_or_grayordinates,)
    parcel_ids : ndarray or None
        Parcel IDs corresponding to columns of parcel_values.
    background : int, float, or None
        Background label.
    fill_value : float
        Value assigned to background or unlabeled locations.

    Returns
    -------
    brain_data : ndarray
        Shape is (n_vertices_or_grayordinates,) for a single map,
        or (n_maps, n_vertices_or_grayordinates) for multiple maps.
    parcel_ids : ndarray
        Parcel IDs used for mapping.
    """
    labels = np.asarray(labels, dtype=float)
    values = np.asarray(parcel_values, dtype=float)

    if parcel_ids is None:
        parcel_ids = _valid_parcel_ids(labels, background=background)
    else:
        parcel_ids = np.asarray(parcel_ids).astype(int)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    if values.shape[1] != len(parcel_ids):
        raise ValueError(
            f"parcel_values has {values.shape[1]} parcels, but parcel_ids has "
            f"{len(parcel_ids)} parcels."
        )

    out = np.full((values.shape[0], labels.shape[0]), fill_value, dtype=float)

    for j, pid in enumerate(parcel_ids):
        out[:, labels == pid] = values[:, j][:, None]

    if out.shape[0] == 1:
        out = out[0]

    return out, parcel_ids


def brain_to_parcel(
    brain_data: Any,
    parcellation: Any,
    background: int | float | None = 0,
    ignore_nan: bool = True,
    reduction: Reduction = "mean",
    return_info: bool = True,
) -> tuple[np.ndarray, dict[str, Any]] | np.ndarray:
    """
    Convert CIFTI or GIFTI brain data into a parcel-level vector/matrix.

    Parameters
    ----------
    brain_data : str, nib.Cifti2Image, nib.GiftiImage, or tuple/list
        Brain data image. For GIFTI, use a tuple/list for left and right hemispheres.
    parcellation : str, nib.Cifti2Image, nib.GiftiImage, or tuple/list
        Parcellation image. For GIFTI, use a tuple/list for left and right hemispheres.
    background : int, float, or None
        Label treated as background. Default is 0.
    ignore_nan : bool
        Whether to ignore NaN values when aggregating parcel values.
    reduction : {"mean", "median", "sum"}
        Parcel-wise aggregation function.
    return_info : bool
        If True, return a dictionary containing parcel IDs and format metadata.

    Returns
    -------
    parcel_values : ndarray
        Shape is (n_parcels,) for one map, or (n_maps, n_parcels) for multiple maps.
    info : dict, optional
        Metadata needed for inverse mapping.
    """
    # CIFTI mode
    if _is_cifti(brain_data) and _is_cifti(parcellation):
        data_matrix, data_brain_axis, _, _, _ = _get_cifti_data_matrix(brain_data)
        label_vec, parc_brain_axis = _get_cifti_label_vector(parcellation)

        if len(data_brain_axis) != len(parc_brain_axis):
            raise ValueError("CIFTI data and parcellation have different numbers of grayordinates.")

        parcel_values, parcel_ids = _parcellate_matrix(
            data_matrix=data_matrix,
            labels=label_vec,
            background=background,
            ignore_nan=ignore_nan,
            reduction=reduction,
        )

        info = {
            "format": "cifti",
            "parcel_ids": parcel_ids,
            "background": background,
            "parcellation": parcellation,
        }

        return (parcel_values, info) if return_info else parcel_values

    # GIFTI mode
    data_imgs = _as_tuple(brain_data)
    parc_imgs = _as_tuple(parcellation)

    if len(data_imgs) != len(parc_imgs):
        raise ValueError(
            "For GIFTI inputs, brain_data and parcellation must have the same number of images."
        )

    if not all(_is_gifti(x) for x in data_imgs):
        raise TypeError("brain_data must be either CIFTI or GIFTI.")
    if not all(_is_gifti(x) for x in parc_imgs):
        raise TypeError("parcellation must be either CIFTI or GIFTI.")

    data_list = []
    label_list = []

    for d_img, p_img in zip(data_imgs, parc_imgs):
        d = _get_gifti_array(d_img)
        p = _get_gifti_array(p_img)

        if p.ndim > 1:
            p = p[:, 0]

        if d.ndim == 1:
            d = d.reshape(1, -1)
        elif d.ndim == 2:
            # Expected as vertices x maps; convert to maps x vertices.
            if d.shape[0] == p.shape[0]:
                d = d.T
            elif d.shape[1] == p.shape[0]:
                pass
            else:
                raise ValueError("GIFTI data shape does not match parcellation.")
        else:
            raise ValueError("GIFTI data must be 1D or 2D.")

        if d.shape[-1] != p.shape[0]:
            raise ValueError(
                f"GIFTI data length ({d.shape[-1]}) and parcellation length ({p.shape[0]}) do not match."
            )

        data_list.append(d)
        label_list.append(p)

    data_matrix = np.concatenate(data_list, axis=1)
    label_vec = np.concatenate(label_list, axis=0)

    parcel_values, parcel_ids = _parcellate_matrix(
        data_matrix=data_matrix,
        labels=label_vec,
        background=background,
        ignore_nan=ignore_nan,
        reduction=reduction,
    )

    hemi_lengths = [len(x) for x in label_list]

    info = {
        "format": "gifti",
        "parcel_ids": parcel_ids,
        "background": background,
        "parcellation": parcellation,
        "hemi_lengths": hemi_lengths,
    }

    return (parcel_values, info) if return_info else parcel_values


def parcel_to_brain(
    parcel_values: np.ndarray,
    parcellation: Any,
    parcel_ids: np.ndarray | None = None,
    background: int | float | None = 0,
    fill_value: float = np.nan,
    output_format: OutputFormat | None = None,
    scalar_names: list[str] | None = None,
) -> tuple[Cifti2Image | nib.GiftiImage | tuple[nib.GiftiImage, ...], np.ndarray]:
    """
    Map parcel-level vector/matrix back to CIFTI or GIFTI brain space.

    Parameters
    ----------
    parcel_values : ndarray
        Shape is (n_parcels,) or (n_maps, n_parcels).
    parcellation : str, nib.Cifti2Image, nib.GiftiImage, or tuple/list
        CIFTI or GIFTI parcellation.
    parcel_ids : ndarray or None
        Parcel IDs corresponding to parcel_values. If None, inferred from parcellation.
    background : int, float, or None
        Background label.
    fill_value : float
        Value assigned to background or unlabeled vertices/grayordinates.
    output_format : {"cifti", "gifti", None}
        If None, inferred from parcellation.
    scalar_names : list[str] or None
        Names for CIFTI scalar maps. If None, automatic names are generated.

    Returns
    -------
    img : nib.Cifti2Image, nib.GiftiImage, or tuple of nib.GiftiImage
        Brain-space image containing parcel values.
    used_parcel_ids : ndarray
        Parcel IDs used for mapping.
    """
    if output_format is None:
        if _is_cifti(parcellation):
            output_format = "cifti"
        else:
            output_format = "gifti"

    if output_format == "cifti":
        label_vec, brain_axis = _get_cifti_label_vector(parcellation)

        brain_data, used_parcel_ids = _inverse_parcellate_vector(
            parcel_values=parcel_values,
            labels=label_vec,
            parcel_ids=parcel_ids,
            background=background,
            fill_value=fill_value,
        )

        if brain_data.ndim == 1:
            brain_data = brain_data.reshape(1, -1)

        n_maps = brain_data.shape[0]

        if scalar_names is None:
            scalar_names = [f"map_{i + 1}" for i in range(n_maps)]

        if len(scalar_names) != n_maps:
            raise ValueError("Length of scalar_names must match number of maps.")

        scalar_axis = ScalarAxis(scalar_names)

        out = Cifti2Image(
            dataobj=brain_data.astype(np.float32),
            header=(scalar_axis, brain_axis),
        )

        return out, used_parcel_ids

    if output_format == "gifti":
        parc_imgs = _as_tuple(parcellation)

        if not all(_is_gifti(x) for x in parc_imgs):
            raise TypeError("GIFTI output requires GIFTI parcellation.")

        label_list = []

        for p_img in parc_imgs:
            p = _get_gifti_array(p_img)
            if p.ndim > 1:
                p = p[:, 0]
            label_list.append(p)

        label_vec = np.concatenate(label_list, axis=0)

        brain_data, used_parcel_ids = _inverse_parcellate_vector(
            parcel_values=parcel_values,
            labels=label_vec,
            parcel_ids=parcel_ids,
            background=background,
            fill_value=fill_value,
        )

        if brain_data.ndim == 1:
            brain_data = brain_data.reshape(1, -1)

        hemi_lengths = [len(x) for x in label_list]
        split_points = np.cumsum(hemi_lengths)[:-1]
        split_data = np.split(brain_data, split_points, axis=1)

        gifti_imgs = []
        for hemi_data in split_data:
            # Convert maps x vertices to vertices x maps for GIFTI writing.
            hemi_data = hemi_data.T
            gifti_imgs.append(_make_gifti_array(hemi_data))

        if len(gifti_imgs) == 1:
            return gifti_imgs[0], used_parcel_ids

        return tuple(gifti_imgs), used_parcel_ids

    raise ValueError("output_format must be one of {'cifti', 'gifti', None}.")


def parcel_mapper(
    mode: Mode,
    data: Any,
    parcellation: Any,
    parcel_ids: np.ndarray | None = None,
    background: int | float | None = 0,
    ignore_nan: bool = True,
    reduction: Reduction = "mean",
    fill_value: float = np.nan,
    scalar_names: list[str] | None = None,
    return_info: bool = True,
) -> Any:
    """
    Unified interface for brain-to-parcel and parcel-to-brain mapping.

    Parameters
    ----------
    mode : {"to_parcel", "to_brain"}
        Mapping direction.
    data : image-like or ndarray
        If mode == "to_parcel", this is brain data image.
        If mode == "to_brain", this is parcel-level vector/matrix.
    parcellation : image-like
        CIFTI or GIFTI parcellation.
    parcel_ids : ndarray or None
        Parcel ID ordering for inverse mapping.
    background : int, float, or None
        Background label.
    ignore_nan : bool
        Whether to ignore NaN during parcellation.
    reduction : {"mean", "median", "sum"}
        Parcel aggregation method.
    fill_value : float
        Value assigned to unlabeled brain locations during inverse mapping.
    scalar_names : list[str] or None
        CIFTI scalar map names for inverse mapping.
    return_info : bool
        Whether to return metadata in brain-to-parcel mode.

    Returns
    -------
    Result depends on mode.
    """
    if mode == "to_parcel":
        return brain_to_parcel(
            brain_data=data,
            parcellation=parcellation,
            background=background,
            ignore_nan=ignore_nan,
            reduction=reduction,
            return_info=return_info,
        )

    if mode == "to_brain":
        return parcel_to_brain(
            parcel_values=data,
            parcellation=parcellation,
            parcel_ids=parcel_ids,
            background=background,
            fill_value=fill_value,
            scalar_names=scalar_names,
        )

    raise ValueError("mode must be either 'to_parcel' or 'to_brain'.")
