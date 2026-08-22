"""Conversion between DataFrames and labelled arrays.

The orchestration here is shared by every DataFrame library: adapters only read
and write columns as numpy arrays.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from labelled_array._reshape import (
    factorize,
    flat_positions,
    natural_index,
    scatter,
    take_flat,
)
from labelled_array._utils import as_name_list, is_null, per_name
from labelled_array.adapters import DataFrameAdapter, get_adapter, resolve_adapter
from labelled_array.dimensions import Dimensions
from labelled_array.exceptions import DimensionError

__all__ = ["from_dataframe", "to_dataframe"]


def _scatter_column(
    name: str,
    values: np.ndarray,
    positions: np.ndarray,
    shape: tuple[int, ...],
    fill_value: Any,
    dtype: Any,
) -> np.ndarray:
    """Scatter one column, reporting which column and option caused a failure."""
    try:
        return scatter(values, positions, shape, fill_value=fill_value, dtype=dtype)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"cannot build variable {name!r} of dtype "
            f"{np.dtype(values.dtype if dtype is None else dtype)} "
            f"with fill_value={fill_value!r}: {error}"
        ) from error


def _natural_codes(
    factorized: Mapping[str, tuple[np.ndarray, np.ndarray]],
    dim_names: Sequence[str],
    natural_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return the natural index of each dimension being replaced by one.

    A natural index numbers the rows within each combination of every
    non-natural dimension plus the natural dimensions declared before it, so
    where a dimension sits in the axis order does not affect the result.
    """
    grouping = [factorized[name][1] for name in dim_names if name not in natural_names]
    indices = {}
    for name in natural_names:
        codes = factorized[name][1]
        indices[name] = natural_index([*grouping, codes])
        grouping.append(codes)
    return indices


def from_dataframe(
    cls: type,
    df: Any,
    cols: str | Sequence[str] | None = None,
    dims: str | Sequence[str] | None = None,
    *,
    natural: str | Sequence[str] | None = None,
    fill_value: Any = None,
    dtype: Any = None,
    mask: bool = True,
    attrs: Mapping[str, Any] | None = None,
    backend: Any = None,
    engine: str | DataFrameAdapter | None = None,
) -> Any:
    """Build a labelled array from a DataFrame. See :meth:`LabelledArray.from_dataframe`."""
    adapter = get_adapter(engine) if engine is not None else resolve_adapter(df)

    index_names = adapter.index_names(df)
    df = adapter.reset_index(df)
    available = adapter.column_names(df)

    if dims is None:
        dims = index_names
        if not dims:
            raise DimensionError(
                "dims must be given: the frame has no named index to take them from"
            )
    dim_names = as_name_list(dims)
    if cols is None:
        cols = [name for name in available if name not in dim_names]
    columns = as_name_list(cols)

    missing = [name for name in [*dim_names, *columns] if name not in available]
    if missing:
        raise KeyError(f"columns not in the frame: {missing}, available: {available}")
    overlap = sorted(set(dim_names) & set(columns))
    if overlap:
        raise DimensionError(f"columns cannot be both dimensions and variables: {overlap}")

    fill_values = per_name(fill_value, columns, "fill_value")
    dtypes = per_name(dtype, columns, "dtype")

    natural_names = as_name_list(natural)
    unknown_natural = [name for name in natural_names if name not in dim_names]
    if unknown_natural:
        raise DimensionError(f"natural names are not dimensions: {unknown_natural}")

    dim_values = {name: adapter.column_values(df, name) for name in dim_names}
    factorized = {name: factorize(values) for name, values in dim_values.items()}
    natural_codes = _natural_codes(factorized, dim_names, natural_names)

    labels, codes, natural_sources = {}, [], {}
    for name in dim_names:
        if name in natural_codes:
            dim_codes = natural_codes[name]
            dim_labels = np.arange(dim_codes.max() + 1 if dim_codes.size else 0)
            axis_name = f"{name}{cls._idx_suffix}"
            natural_sources[axis_name] = name
        else:
            axis_name = name
            dim_labels, dim_codes = factorized[name]
        labels[axis_name] = dim_labels
        codes.append(dim_codes)

    dimensions = Dimensions(labels, natural=natural_sources)
    positions = flat_positions(codes, dimensions, adapter.n_rows(df))

    data = {
        name: _scatter_column(
            name,
            adapter.column_values(df, name),
            positions,
            dimensions.shape,
            fill_value=fill_values[name],
            dtype=dtypes[name],
        )
        for name in columns
    }
    coords = {
        name: scatter(dim_values[name], positions, dimensions.shape) for name in natural_names
    }

    # Positions are unique, so the array is fully populated exactly when there
    # is one row per cell -- no need to build a mask to find that out.
    presence = None
    if mask and positions.size != dimensions.size:
        presence = np.zeros(dimensions.shape, dtype=bool)
        presence.reshape(-1)[positions] = True

    return cls(
        data=data,
        dims=dimensions,
        mask=presence,
        coords=coords,
        attrs=attrs,
        backend=backend,
    )


def to_dataframe(
    array: Any,
    engine: str | DataFrameAdapter | None = None,
    *,
    index: bool = True,
    dropna: bool = False,
    expand_natural: bool = True,
) -> Any:
    """Convert a labelled array to a DataFrame. See :meth:`LabelledArray.to_dataframe`."""
    adapter = get_adapter(engine)
    backend = array.backend

    positions = (
        np.arange(array.size, dtype=np.intp)
        if array.mask is None
        else np.flatnonzero(array.mask.reshape(-1))
    )

    values = {
        name: take_flat(backend.to_numpy(variable), positions)
        for name, variable in array.data.items()
    }

    if dropna and values:
        keep = None
        for column in values.values():
            null = is_null(column)
            keep = null if keep is None else np.logical_and(keep, null, out=keep)
        keep = ~keep
        positions = positions[keep]
        values = {name: column[keep] for name, column in values.items()}

    # A natural index dimension is written out as the values it stands in for,
    # so that coordinate is not repeated as a column of its own.
    expanded = (
        {name: _natural_source(array, name) for name in array.dims.names}
        if expand_natural
        else {}
    )

    columns: dict[str, np.ndarray] = {}
    index_names: list[str] = []
    if array.ndim:
        cells = np.unravel_index(positions, array.shape)
        for axis, name in enumerate(array.dims.names):
            source = expanded.get(name)
            if source is None:
                columns[name] = array.dims[name][cells[axis]]
                index_names.append(name)
            else:
                columns[source] = take_flat(array.coords[source], positions)
                index_names.append(source)

    for name, coord in array.coords.items():
        if name not in columns:
            columns[name] = take_flat(coord, positions)
    columns.update(values)

    return adapter.build_frame(columns, index_names=index_names, index=index)


def _natural_source(array: Any, dim: str) -> str | None:
    """Return the coordinate a natural index dimension stands in for, if any."""
    source = array.dims.source(dim)
    return source if source is not None and source in array.coords else None
