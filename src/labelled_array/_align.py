"""Moving arrays onto other label sets: reindex, align and concat.

A dense array is addressed by position, so combining two of them is a question
about their labels first: which cells the result spans, and where each operand's
values land in it. Everything here answers that question in labels and then
hands the gather to the backend in one pass, so a cell is only ever copied
once, however many dimensions were joined.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from labelled_array._reshape import default_fill_value, factorize
from labelled_array._utils import per_name
from labelled_array.dimensions import Dimensions, as_1d
from labelled_array.exceptions import DimensionError

__all__ = ["JOINS", "align", "concat", "reindex"]

#: How :func:`align` resolves a dimension whose labels differ between arrays.
JOINS = ("inner", "outer", "left", "right", "exact")


def _isin(labels: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Return a boolean array marking the entries of ``labels`` present in ``other``.

    Object arrays -- what every string column becomes -- are matched through a
    set of their python values, because :func:`numpy.isin` would compare them
    pairwise.
    """
    if labels.dtype.kind == "O" or other.dtype.kind == "O":
        known = set(other.tolist())
        return np.array([value in known for value in labels.tolist()], dtype=bool)
    return np.isin(labels, other)


def join_labels(name: str, label_sets: Sequence[np.ndarray], join: str) -> np.ndarray:
    """Return the labels a dimension takes when several arrays are combined.

    Parameters
    ----------
    name : str
        Dimension being joined, used in the error message.
    label_sets : sequence of numpy.ndarray
        The labels each array gives the dimension, in operand order.
    join : {'inner', 'outer', 'left', 'right', 'exact'}
        ``'inner'`` keeps the labels every array has, in the order the first
        array holds them; ``'outer'`` takes the union, sorted where the dtype
        allows it; ``'left'`` and ``'right'`` take the first or last array's
        labels as they are; ``'exact'`` refuses to join at all.

    Raises:
    ------
    ValueError
        If ``join`` is not one of the modes above.
    DimensionError
        If ``join`` is ``'exact'`` and the label sets are not identical.
    """
    if join not in JOINS:
        raise ValueError(f"join must be one of {JOINS}, got {join!r}")
    if join == "exact":
        raise DimensionError(
            f"dimension {name!r} has different labels in the operands, and join='exact'"
        )
    if join == "left":
        return label_sets[0]
    if join == "right":
        return label_sets[-1]
    if join == "outer":
        # factorize sorts where the dtype allows it and falls back to first-seen
        # order otherwise, which is how from_dataframe orders labels too.
        return factorize(np.concatenate([np.asarray(labels) for labels in label_sets]))[
            0
        ]

    keep = np.ones(len(label_sets[0]), dtype=bool)
    for other in label_sets[1:]:
        keep &= _isin(label_sets[0], other)
    return label_sets[0][keep]


def reindex(array: Any, indexers: Mapping[str, Any], fill_value: Any = None) -> Any:
    """Return ``array`` on new labels for the named dimensions.

    See :meth:`~labelled_array.LabelledArray.reindex`.
    """
    dims = array.dims
    dims._require(indexers)
    if not indexers:
        return array

    backend = array.backend
    positions: dict[int, np.ndarray] = {}
    valid: np.ndarray | None = None
    new_dims = dims
    for name, labels in indexers.items():
        wanted = as_1d(labels, name)
        found = dims.locate(name, wanted)
        axis = dims.axis(name)
        # Missing labels gather from position 0 and are overwritten with the
        # fill below, which keeps the gather itself a single pass.
        positions[axis] = np.where(found >= 0, found, 0)
        # An alternative dimension describes the entries of this one, so it can
        # be carried only when every new label is an old one -- a subset or a
        # permutation. A label the array did not have has no description, and
        # inventing a null one would give it a label that can never be selected.
        carried = (
            {alt: values[found] for alt, values in dims.alt_on(name).items()}
            if (found >= 0).all()
            else None
        )
        new_dims = new_dims.replace(name, wanted, alt=carried)
        if (found < 0).any():
            spread = [1] * dims.ndim
            spread[axis] = found.size
            present = (found >= 0).reshape(spread)
            valid = present if valid is None else valid & present

    cells = array._map_cells(lambda be, arr: be.take_multi(arr, positions))
    if valid is None:
        return array._replace(dims=new_dims, **cells)

    fills = per_name(fill_value, array.variables, "fill_value", "variables")
    cells["data"] = {
        name: backend.where(
            valid,
            value,
            default_fill_value(backend.dtype(value))
            if fills[name] is None
            else fills[name],
        )
        for name, value in cells["data"].items()
    }
    cells["coords"] = {
        name: np.where(valid, value, default_fill_value(value.dtype))
        for name, value in cells["coords"].items()
    }
    filled = np.broadcast_to(valid, new_dims.shape)
    cells["mask"] = filled.copy() if cells["mask"] is None else cells["mask"] & filled
    return array._replace(dims=new_dims, **cells)


def align(
    arrays: Sequence[Any],
    join: str = "inner",
    fill_value: Any = None,
    exclude: Sequence[str] = (),
) -> tuple:
    """Return the arrays reindexed onto one label set per shared dimension.

    ``exclude`` names dimensions to leave alone, which is what
    :func:`concat` needs for the dimension it is joining along.

    See :meth:`~labelled_array.LabelledArray.align`.
    """
    if join not in JOINS:
        raise ValueError(f"join must be one of {JOINS}, got {join!r}")
    arrays = list(arrays)
    if len(arrays) < 2:
        return tuple(arrays)

    targets: dict[str, np.ndarray] = {}
    excluded = set(exclude)
    for name in dict.fromkeys(name for array in arrays for name in array.dims.names):
        if name in excluded:
            continue
        label_sets = [array.dims[name] for array in arrays if name in array.dims]
        if len(label_sets) < 2:
            continue
        first = label_sets[0]
        if all(
            other.dtype == first.dtype and np.array_equal(other, first)
            for other in label_sets[1:]
        ):
            continue
        targets[name] = join_labels(name, label_sets, join)

    if not targets:
        return tuple(arrays)
    return tuple(
        reindex(
            array,
            {name: labels for name, labels in targets.items() if name in array.dims},
            fill_value=fill_value,
        )
        for array in arrays
    )


def _check_concat_inputs(arrays: list, dim: str) -> bool:
    """Validate the operands of :func:`concat` and report whether ``dim`` is new."""
    first = arrays[0]
    for other in arrays[1:]:
        if other.variables != first.variables:
            raise DimensionError(
                f"arrays to concat must hold the same variables: "
                f"{first.variables} vs {other.variables}"
            )
        if set(other.coords) != set(first.coords):
            raise DimensionError(
                f"arrays to concat must hold the same coordinates: "
                f"{sorted(first.coords)} vs {sorted(other.coords)}"
            )
        mine = {name: first.dims.alt_ref(name) for name in first.dims.alt}
        theirs = {name: other.dims.alt_ref(name) for name in other.dims.alt}
        if mine != theirs:
            raise DimensionError(
                f"arrays to concat must hold the same alternative dimensions, on the "
                f"same dimensions: {sorted(mine)} vs {sorted(theirs)}"
            )
        if other.backend.name != first.backend.name:
            raise DimensionError(
                f"arrays to concat must be on one backend: {first.backend.name!r} vs "
                f"{other.backend.name!r}; move one with to_backend() first"
            )
    holders = [dim in array.dims for array in arrays]
    if any(holders) and not all(holders):
        raise DimensionError(
            f"dimension {dim!r} is a dimension of some of the arrays but not all of them"
        )
    for array in arrays:
        if set(array.dims.names) != set(first.dims.names):
            raise DimensionError(
                f"arrays to concat must share their dimensions: "
                f"{first.dims.names} vs {array.dims.names}"
            )
    return not any(holders)


def _concat_alt(aligned: Sequence[Any], dim: str, is_new: bool) -> dict:
    """Return the alternative dimensions the joined array keeps.

    One tied to the dimension being joined is concatenated, the way that
    dimension's own labels are. One tied to any other dimension describes labels
    the operands now share, so they have to agree on it -- alignment puts them
    on one label set, but not on one description of it. An alternative dimension
    alignment has already dropped from an operand, because that operand gained
    labels it had no description for, is dropped here too.
    """
    first = aligned[0]
    kept = {}
    for name, values in first.dims.alt.items():
        others = [array.dims.alt.get(name) for array in aligned[1:]]
        if any(other is None for other in others):
            continue
        if not is_new and first.dims.alt_ref(name) == dim:
            kept[name] = np.concatenate([values, *others])
            continue
        disagreeing = next(
            (other for other in others if not np.array_equal(values, other)), None
        )
        if disagreeing is not None:
            raise DimensionError(
                f"arrays to concat disagree on alternative dimension {name!r}: "
                f"{values.tolist()} vs {disagreeing.tolist()}"
            )
        kept[name] = values
    return kept


def concat(
    arrays: Sequence[Any],
    dim: str,
    labels: Any = None,
    join: str = "outer",
    fill_value: Any = None,
) -> Any:
    """Join arrays along an existing or a new dimension.

    See :meth:`~labelled_array.LabelledArray.concat`.
    """
    arrays = list(arrays)
    if not arrays:
        raise ValueError("concat needs at least one array")
    if len(arrays) == 1 and dim in arrays[0].dims and labels is None:
        return arrays[0]

    is_new = _check_concat_inputs(arrays, dim)
    order = arrays[0].dims.names
    aligned = [
        array if array.dims.names == order else array.transpose(order)
        for array in arrays
    ]
    aligned = list(
        align(
            aligned, join=join, fill_value=fill_value, exclude=() if is_new else [dim]
        )
    )

    first = aligned[0]
    kept_alt = _concat_alt(aligned, dim, is_new)
    backend = first.backend
    axis = 0 if is_new else first.dims.axis(dim)
    shape = first.shape

    def joined(pieces: Sequence[Any], reshape: Any, concatenate: Any) -> Any:
        """Join one array per operand, giving them the new axis first if there is one."""
        if is_new:
            pieces = [reshape(piece, (1, *shape)) for piece in pieces]
        return concatenate(list(pieces), axis)

    data = {
        name: joined(
            [array[name] for array in aligned], backend.reshape, backend.concatenate
        )
        for name in first.variables
    }
    coords = {
        name: joined(
            [array.coords[name] for array in aligned], np.reshape, np.concatenate
        )
        for name in first.coords
    }

    mask = None
    if any(array.mask is not None for array in aligned):
        # An array with no mask is fully populated, which is what the join of
        # one that has a mask with one that does not has to say.
        masks = [
            np.ones(array.shape, dtype=bool) if array.mask is None else array.mask
            for array in aligned
        ]
        mask = joined(masks, np.reshape, np.concatenate)

    if is_new:
        natural = dict(first.dims.natural)
        if labels is None:
            labels = np.arange(len(aligned))
            natural[dim] = None
        else:
            labels = as_1d(labels, dim)
        if len(labels) != len(aligned):
            raise DimensionError(
                f"got {len(labels)} labels for {len(aligned)} arrays along new dimension {dim!r}"
            )
        alt = {
            name: (first.dims.alt_ref(name), values)
            for name, values in kept_alt.items()
        }
        dims = Dimensions({dim: labels, **dict(first.dims)}, natural=natural, alt=alt)
    else:
        length = sum(len(array.dims[dim]) for array in aligned)
        if labels is not None:
            joined = as_1d(labels, dim)
            if len(joined) != length:
                raise DimensionError(
                    f"got {len(joined)} labels for {length} entries along dimension {dim!r}"
                )
        elif first.dims.is_natural(dim):
            # A natural index counts positions, so joining two of them renumbers
            # rather than repeating 0, 1, 2 ... once per array.
            joined = np.arange(length)
        else:
            joined = np.concatenate([array.dims[dim] for array in aligned])
        # An operand may have lost an alternative dimension to alignment, and one
        # that no longer describes every operand cannot survive the join.
        base = first.dims.drop_alt(set(first.dims.alt) - set(kept_alt))
        on_dim = {
            name: values
            for name, values in kept_alt.items()
            if base.alt_ref(name) == dim
        }
        dims = base.replace(dim, joined, alt=on_dim)

    return first._replace(dims=dims, data=data, coords=coords, mask=mask)
