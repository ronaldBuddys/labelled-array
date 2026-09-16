"""The :class:`LabelledArray` container."""

from __future__ import annotations

import copy as copy_module
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from labelled_array import _align, _convert, _fill, _ops
from labelled_array._arithmetic import ArithmeticMixin
from labelled_array._utils import (
    array_repr,
    arrays_equal,
    as_name_list,
    is_null,
    per_name,
)
from labelled_array.backends import (
    DEFAULT_BACKEND,
    ArrayBackend,
    _resolve_or_none,
    get_backend,
)
from labelled_array.dimensions import Dimensions
from labelled_array.exceptions import DimensionError

__all__ = ["LabelledArray"]


class LabelledArray(ArithmeticMixin):
    """One or more equally shaped arrays sharing named, labelled dimensions.

    Every variable has the shape given by ``dims``, so a cell is addressed by one
    label per dimension. Dimension labels, the presence mask and the coordinates
    are always held as numpy arrays; only the variables live on the array
    backend, which may be numpy, JAX or torch.

    Parameters
    ----------
    data : mapping of str to array_like
        Variables, each of shape ``dims.shape``.
    dims : Dimensions or mapping or sequence of (str, array_like), optional
        Dimension names and labels, in axis order. Defaults to a natural index
        per axis of the first variable.
    mask : array_like of bool, optional
        Shape ``dims.shape``, True where a source row existed. Cells that are
        False are omitted by :meth:`to_dataframe`, which is what makes a round
        trip lossless for dtypes with no missing-value representation.
    coords : mapping of str to array_like, optional
        Auxiliary labels of shape ``dims.shape``, holding the values a natural
        index dimension stands in for.
    attrs : dict, optional
        Free-form metadata, carried through operations unchanged.
    backend : str or ArrayBackend, optional
        Array backend for the variables. Defaults to the backend of the first
        variable that a registered backend recognises, else numpy.

    Raises
    ------
    DimensionError
        If a variable, mask or coordinate does not match ``dims.shape``, or if a
        variable, coordinate and dimension name collide.

    Notes
    -----
    Variable, coordinate and dimension names share one namespace, because a
    DataFrame column can only carry one of each.

    Examples
    --------
    >>> import numpy as np
    >>> la = LabelledArray({"w": np.arange(6).reshape(2, 3)},
    ...                    dims={"a": [1, 2], "b": ["x", "y", "z"]})
    >>> la.shape
    (2, 3)
    >>> la["w"][0, 2]
    np.int64(2)
    """

    #: Backend used when none is given and no variable identifies one.
    _default_backend: str = DEFAULT_BACKEND
    #: Suffix given to a dimension replaced by a natural index.
    _idx_suffix: str = "_idx"
    #: Largest ``size`` for which ``repr`` prints an ``eval``-able constructor.
    _repr_full_threshold: int = 1000

    def __init__(
        self,
        data: Mapping[str, Any],
        dims: Dimensions | Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
        mask: Any = None,
        coords: Mapping[str, Any] | None = None,
        attrs: Mapping[str, Any] | None = None,
        backend: str | ArrayBackend | None = None,
        alt_dims: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(data, Mapping):
            raise TypeError(
                f"data must be a mapping of name to array, got {type(data).__name__}; "
                "use LabelledArray.from_array for a single unnamed array"
            )
        data = dict(data)
        for name in data:
            if not isinstance(name, str):
                raise TypeError(f"variable names must be str, got {name!r}")

        self._backend = self._resolve_backend(data, backend)
        self._data = {
            name: self._backend.asarray(value) for name, value in data.items()
        }
        self._dims = self._resolve_dims(dims, alt_dims)
        self._mask = _as_mask(mask, self._dims.shape)
        self._coords = {
            name: _as_coord(name, value, self._dims.shape)
            for name, value in (coords or {}).items()
        }
        self._attrs = dict(attrs or {})

        self._validate()

    # -- construction helpers ----------------------------------------------------

    def _resolve_backend(
        self, data: Mapping[str, Any], backend: str | ArrayBackend | None
    ) -> ArrayBackend:
        """Return the backend to hold the variables on."""
        if backend is not None:
            return get_backend(backend)
        for value in data.values():
            owner = _resolve_or_none(value)
            if owner is not None:
                return owner
        return get_backend(self._default_backend)

    def _resolve_dims(self, dims: Any, alt_dims: Any = None) -> Dimensions:
        """Return ``dims`` as :class:`Dimensions`, defaulting to a natural index.

        ``alt_dims`` is layered on whatever ``dims`` resolves to, so alternative
        dimensions can be given alongside a plain mapping, an existing
        :class:`Dimensions` that already carries some, or no ``dims`` at all.
        """
        if isinstance(dims, Dimensions):
            resolved = dims
        elif dims is not None:
            resolved = Dimensions(dims)
        elif not self._data:
            raise DimensionError("dims must be given when there are no variables")
        else:
            first = next(iter(self._data.values()))
            resolved = Dimensions.from_shape(self._backend.shape(first))
        return Dimensions(resolved, alt=alt_dims) if alt_dims else resolved

    def _validate(self) -> None:
        """Check shapes and names. Subclasses extend this to add requirements."""
        shape = self._dims.shape
        for name, value in self._data.items():
            if self._backend.shape(value) != shape:
                raise DimensionError(
                    f"variable {name!r} has shape {self._backend.shape(value)}, "
                    f"expected {shape} from dimensions {list(self._dims.names)}"
                )
        counts = Counter(
            [*self._dims.names, *self._dims.alt, *self._data, *self._coords]
        )
        clashes = sorted(name for name, count in counts.items() if count > 1)
        if clashes:
            raise DimensionError(
                "names must be unique across dimensions, alternative dimensions, "
                f"variables and coordinates: {clashes}"
            )

    @classmethod
    def from_array(
        cls,
        array: Any,
        dims: Any = None,
        name: str = "values",
        **kwargs: Any,
    ) -> LabelledArray:
        """Build a labelled array from a single unnamed array.

        Parameters
        ----------
        array : array_like
            The sole variable.
        dims : Dimensions or mapping, optional
            Dimensions; defaults to a natural index per axis.
        name : str, default 'values'
            Name to give the variable.
        **kwargs
            Passed to the constructor.
        """
        return cls({name: array}, dims, **kwargs)

    def _fields(self) -> dict[str, Any]:
        """Return the constructor arguments that reproduce this instance.

        A subclass taking extra constructor arguments extends this, and every
        derived object then carries them::

            def _fields(self):
                return {**super()._fields(), "currency": self.currency}
        """
        return {
            "data": self._data,
            "dims": self._dims,
            "mask": self._mask,
            "coords": self._coords,
            "attrs": self._attrs,
            "backend": self._backend,
        }

    def _replace(self, **changes: Any) -> LabelledArray:
        """Return a new instance of this class with the given fields substituted.

        Every method returning a derived object goes through here, so subclasses
        keep their own type, and their own state via :meth:`_fields`, without
        overriding anything.
        """
        return type(self)(**{**self._fields(), **changes})

    def _cells(self) -> dict[str, Any]:
        """Return the arrays indexed by the dimensions, as constructor fields."""
        return {
            "data": dict(self._data),
            "coords": dict(self._coords),
            "mask": self._mask,
        }

    def _map_cells(
        self, apply: Any, cells: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Apply ``apply(backend, array)`` to every array of the array's shape.

        The variables, the coordinates and the mask are all indexed by the same
        dimensions, so they must be taken from, reshaped and transposed
        together; only their backend differs, since coordinates and the mask are
        always numpy. Routing every shape-changing operation through here is
        what keeps the three from drifting apart.
        """
        cells = self._cells() if cells is None else cells
        numpy_backend = get_backend("numpy")
        mask = cells["mask"]
        return {
            "data": {
                name: apply(self._backend, value)
                for name, value in cells["data"].items()
            },
            "coords": {
                name: apply(numpy_backend, value)
                for name, value in cells["coords"].items()
            },
            "mask": None if mask is None else apply(numpy_backend, mask),
        }

    # -- properties --------------------------------------------------------------

    @property
    def dims(self) -> Dimensions:
        """Dimension names and labels."""
        return self._dims

    @property
    def data(self) -> Mapping[str, Any]:
        """Read-only view of the variables."""
        return MappingProxyType(self._data)

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        """Read-only view of the coordinates."""
        return MappingProxyType(self._coords)

    @property
    def attrs(self) -> dict[str, Any]:
        """Free-form metadata (mutable)."""
        return self._attrs

    @property
    def mask(self) -> np.ndarray | None:
        """Presence mask, or None when every cell is populated."""
        return self._mask

    @property
    def backend(self) -> ArrayBackend:
        """Backend holding the variables."""
        return self._backend

    @property
    def variables(self) -> tuple[str, ...]:
        """Variable names, in insertion order."""
        return tuple(self._data)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape shared by every variable."""
        return self._dims.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._dims.ndim

    @property
    def size(self) -> int:
        """Number of cells spanned by the dimensions."""
        return self._dims.size

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        """Numpy dtype of each variable."""
        return {name: self._backend.dtype(value) for name, value in self._data.items()}

    @property
    def values(self) -> Any:
        """The sole variable's array.

        Raises
        ------
        ValueError
            If there is not exactly one variable.
        """
        if len(self._data) != 1:
            raise ValueError(
                f"values is defined only for a single variable, this has {self.variables}"
            )
        return next(iter(self._data.values()))

    @property
    def density(self) -> float:
        """Fraction of cells that are populated, 1.0 when there is no mask."""
        if self._mask is None:
            return 1.0
        return float(self._mask.mean()) if self._mask.size else 0.0

    # -- derived instances -------------------------------------------------------

    def _take(
        self, positions: Mapping[str, np.ndarray], drop: bool = False
    ) -> LabelledArray:
        """Return the entries at integer ``positions`` along the named dimensions.

        Parameters
        ----------
        positions : mapping of str to ndarray of int
            Positions to keep, per dimension.
        drop : bool, default False
            Drop the selected dimensions that end up with a single entry.
        """
        dims = self._dims.subset(positions)
        axes = {self._dims.axis(name): indices for name, indices in positions.items()}
        cells = self._map_cells(lambda backend, array: backend.take_multi(array, axes))

        if drop:
            dropped = [name for name in positions if len(dims[name]) == 1]
            if dropped:
                dims = dims.drop(dropped)
                cells = self._map_cells(
                    lambda backend, array: backend.reshape(array, dims.shape), cells
                )
        return self._replace(dims=dims, **cells)

    def select(
        self,
        indexers: Mapping[str, Any] | None = None,
        drop: bool = False,
        missing: str = "raise",
        **kwargs: Any,
    ):
        """Select along dimensions, or alternative dimensions, by label.

        Parameters
        ----------
        indexers : mapping of str to scalar or array_like, optional
            Labels to keep, per dimension. Dimensions not named are kept whole.
            A name may also be an *alternative* dimension, in which case its
            labels resolve to positions along the dimension it is tied to; the
            two kinds mix freely in one call. Because alternative labels need
            not be unique, one of them can keep several entries.
        drop : bool, default False
            Drop the selected dimensions that end up with a single entry.
            Without it a scalar label yields a length-one dimension, so the rank
            of the result never depends on how many labels were asked for.
        missing : {'raise', 'ignore'}, default 'raise'
            Whether labels absent from a dimension raise
            :class:`LabelNotFoundError` or are skipped.
        **kwargs
            Alternative to ``indexers``, as keyword arguments.

        Returns
        -------
        LabelledArray
            An instance of this class holding the selection, with the
            dimensions ordered as the labels were requested.

        Raises
        ------
        DimensionError
            If a name is neither a dimension nor an alternative dimension, if
            the same label is requested twice, or if two names in one call
            select along the same dimension -- an alternative dimension and the
            dimension it is tied to, say, or two alternative dimensions on it.
            That is an intersection whose order neither request defines, so it
            is asked for as two chained calls instead.
        LabelNotFoundError
            If a label is absent and ``missing`` is ``'raise'``.

        Examples
        --------
        >>> import numpy as np
        >>> la = LabelledArray({"w": np.arange(6).reshape(2, 3)},
        ...                    {"a": [1, 2], "b": ["x", "y", "z"]},
        ...                    alt_dims={"g": ("b", ["odd", "even", "odd"])})
        >>> la.select({"a": 1}).shape
        (1, 3)
        >>> la.select({"a": 1}, drop=True).shape
        (3,)
        >>> la.select(b=["z", "x"]).dims["b"].tolist()
        ['z', 'x']
        >>> la.select(g="odd").dims["b"].tolist()
        ['x', 'z']

        See Also
        --------
        isel : select by integer position, which alternative dimensions have none.
        """
        return self._take(self._resolve_alt(indexers, kwargs, missing), drop=drop)

    def _resolve(
        self,
        indexers: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
        to_positions: Any,
    ) -> dict[str, np.ndarray]:
        """Merge and validate indexers, turning each into integer positions."""
        indexers = {**(indexers or {}), **kwargs}
        self._dims._require(indexers)
        return {name: to_positions(name, value) for name, value in indexers.items()}

    def _resolve_alt(
        self,
        indexers: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
        missing: str,
    ) -> dict[str, np.ndarray]:
        """Merge indexers, resolving alternative dimensions onto their own.

        Names are unique across dimensions, alternative dimensions, variables
        and coordinates, so a name needs no disambiguation: it is looked up as a
        dimension first and as an alternative dimension second. An alternative
        dimension resolves to positions along the dimension it is tied to, which
        is the key its positions land under.
        """
        indexers = {**(indexers or {}), **kwargs}
        alt = self._dims.alt
        if not alt.keys() & indexers.keys():
            self._dims._require(indexers)
            return {
                name: self._dims.get_indexer(name, labels, missing=missing)
                for name, labels in indexers.items()
            }

        positions: dict[str, np.ndarray] = {}
        claimed: dict[str, str] = {}
        for name, labels in indexers.items():
            if name in alt:
                ref = self._dims.alt_ref(name)
                found = self._dims.alt_indexer(name, labels, missing=missing)
            else:
                self._dims._require([name])
                ref, found = name, self._dims.get_indexer(name, labels, missing=missing)
            if ref in claimed:
                # Two selections along one axis is an intersection, and the order
                # of the result is not defined by either request. Chained calls
                # say which order is wanted, so ask for that rather than guess.
                raise DimensionError(
                    f"{name!r} and {claimed[ref]!r} both select along dimension "
                    f"{ref!r}; select on one of them and then the other"
                )
            claimed[ref] = name
            positions[ref] = found
        return positions

    def isel(
        self,
        indexers: Mapping[str, Any] | None = None,
        drop: bool = False,
        **kwargs: Any,
    ):
        """Select along dimensions by integer position.

        Parameters
        ----------
        indexers : mapping of str to int, slice or array_like of int, optional
            Positions to keep, per dimension. Dimensions not named are kept
            whole. Only real dimensions can be named: an alternative dimension
            has no axis of its own, so it has no positions to index by. Its
            labels are carried along with the dimension it is tied to.
        drop : bool, default False
            Drop the selected dimensions that end up with a single entry;
            otherwise a scalar position yields a length-one dimension.
        **kwargs
            Alternative to ``indexers``, as keyword arguments.

        Returns
        -------
        LabelledArray
            An instance of this class holding the selection.

        Examples
        --------
        >>> import numpy as np
        >>> la = LabelledArray({"w": np.arange(6).reshape(2, 3)})
        >>> la.isel(dim_1=0).shape
        (2, 1)
        >>> la.isel(dim_1=0, drop=True).shape
        (2,)
        """
        return self._take(
            self._resolve(indexers, kwargs, self._dims.get_positions), drop=drop
        )

    def rename(self, mapping: Mapping[str, str] | None = None, **kwargs: str):
        """Rename dimensions, variables or coordinates.

        Parameters
        ----------
        mapping : mapping of str to str, optional
            Current name to new name. Names are unique across dimensions,
            alternative dimensions, variables and coordinates, so no
            disambiguation is needed.
        **kwargs
            Alternative to ``mapping``, as keyword arguments.
        """
        mapping = {**(mapping or {}), **kwargs}
        alt = set(self._dims.alt)
        known = set(self._dims.names) | alt | set(self._data) | set(self._coords)
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise DimensionError(f"unknown names: {unknown}")
        # Coordinate renames are passed through too, so that a natural index
        # keeps pointing at the coordinate it stands in for.
        sources = set(self._dims.natural.values())
        dims = self._dims.rename(
            {
                k: v
                for k, v in mapping.items()
                if k in self._dims or k in alt or k in sources
            }
        )
        data = {mapping.get(name, name): value for name, value in self._data.items()}
        coords = {
            mapping.get(name, name): value for name, value in self._coords.items()
        }
        return self._replace(dims=dims, data=data, coords=coords)

    def transpose(self, *order: str | Sequence[str]):
        """Return a view with the dimensions reordered.

        Parameters
        ----------
        *order : str or sequence of str
            The dimension names in their new order, given either as separate
            arguments or as one sequence. Must be a permutation of the current
            names; empty reverses them.
        """
        if len(order) == 1 and not isinstance(order[0], str):
            names = list(order[0])
        elif order:
            names = list(order)
        else:
            names = list(reversed(self._dims.names))
        dims = self._dims.transpose(names)
        axes = [self._dims.axis(name) for name in names]
        return self._replace(
            dims=dims,
            **self._map_cells(lambda backend, array: backend.transpose(array, axes)),
        )

    def astype(self, dtype: Any):
        """Cast the variables.

        Parameters
        ----------
        dtype : dtype or mapping of str to dtype
            One dtype for every variable, or a dtype per named variable.
        """
        dtypes = per_name(dtype, self.variables, "dtype", "variables")
        data = {
            name: value
            if dtypes[name] is None
            else self._backend.astype(value, dtypes[name])
            for name, value in self._data.items()
        }
        return self._replace(data=data)

    def to_backend(self, backend: str | ArrayBackend):
        """Return the array held on another array backend.

        Raises
        ------
        BackendDtypeError
            If the target backend cannot hold one of the variable dtypes.
        """
        target = get_backend(backend)
        data = {}
        for name, value in self._data.items():
            target.check_dtype(self._backend.dtype(value), context=name)
            data[name] = target.asarray(self._backend.to_numpy(value))
        return self._replace(data=data, backend=target)

    def assign(self, variables: Mapping[str, Any] | None = None, **kwargs: Any):
        """Return the array with variables added or replaced."""
        variables = {**(variables or {}), **kwargs}
        return self._replace(data={**self._data, **variables})

    def drop_vars(self, names: str | Iterable[str]):
        """Return the array without the named variables."""
        names = set(as_name_list(names))
        unknown = sorted(names - set(self._data))
        if unknown:
            raise KeyError(f"unknown variables: {unknown}")
        return self._replace(
            data={
                name: value for name, value in self._data.items() if name not in names
            }
        )

    def copy(self, deep: bool = True):
        """Return a copy, duplicating the arrays when ``deep``."""
        if not deep:
            return self._replace()
        return self._replace(
            attrs=copy_module.deepcopy(self._attrs),
            **self._map_cells(lambda backend, array: backend.copy(array)),
        )

    # -- combining ---------------------------------------------------------------

    def reindex(
        self,
        indexers: Mapping[str, Any] | None = None,
        fill_value: Any = None,
        **kwargs: Any,
    ):
        """Return the array on a different label set for the named dimensions.

        Labels the array already has bring their cells with them; labels it does
        not are new, empty cells, filled with ``fill_value`` and marked absent in
        the mask. Labels the array has and the new set does not are dropped.

        Parameters
        ----------
        indexers : mapping of str to array_like, optional
            New labels, per dimension. Dimensions not named are unchanged.
        fill_value : scalar or mapping of str to scalar, optional
            Value for the new cells, per variable. Defaults to NaN, NaT, empty
            string, False, None or 0, whichever the dtype can hold -- the same
            choice :meth:`from_dataframe` makes.
        **kwargs
            Alternative to ``indexers``, as keyword arguments.

        Returns
        -------
        LabelledArray
            An instance of this class on the new labels.

        Raises
        ------
        DimensionError
            If a dimension is unknown, or the new labels are not unique.

        Examples
        --------
        >>> import numpy as np
        >>> la = LabelledArray({"w": np.arange(3.0)}, {"b": ["x", "y", "z"]})
        >>> wider = la.reindex(b=["x", "z", "q"])
        >>> wider["w"]
        array([ 0.,  2., nan])
        >>> wider.mask.tolist()
        [True, True, False]

        See Also
        --------
        align : put two arrays onto one label set.
        """
        return _align.reindex(
            self, {**(indexers or {}), **kwargs}, fill_value=fill_value
        )

    def reindex_like(self, other: LabelledArray, fill_value: Any = None):
        """Return the array on ``other``'s labels, for the dimensions they share.

        Parameters
        ----------
        other : LabelledArray
            Array whose labels to take.
        fill_value : scalar or mapping of str to scalar, optional
            Value for cells ``other`` has and this array does not.
        """
        shared = {
            name: other.dims[name] for name in other.dims.names if name in self._dims
        }
        return _align.reindex(self, shared, fill_value=fill_value)

    def align(
        self, *others: LabelledArray, join: str = "inner", fill_value: Any = None
    ):
        """Return this array and ``others`` on one label set per shared dimension.

        Dimensions no other array has are left alone, and no array is
        transposed: alignment is about labels, not about axis order.

        Parameters
        ----------
        *others : LabelledArray
            Arrays to align with this one.
        join : {'inner', 'outer', 'left', 'right', 'exact'}, default 'inner'
            How to resolve a dimension whose labels differ. ``'inner'`` keeps
            the labels every array has, in this array's order; ``'outer'`` takes
            the union; ``'left'`` and ``'right'`` take this array's or the last
            one's labels; ``'exact'`` raises unless they already match.
        fill_value : scalar or mapping of str to scalar, optional
            Value for cells an array gains, as in :meth:`reindex`.

        Returns
        -------
        tuple of LabelledArray
            This array first, then ``others``, all on the joined labels.

        Examples
        --------
        >>> import numpy as np
        >>> left = LabelledArray({"w": np.arange(3.0)}, {"b": ["x", "y", "z"]})
        >>> right = LabelledArray({"w": np.arange(2.0)}, {"b": ["y", "z"]})
        >>> a, b = left.align(right)
        >>> a.dims["b"].tolist()
        ['y', 'z']
        """
        return _align.align([self, *others], join=join, fill_value=fill_value)

    @classmethod
    def concat(
        cls,
        arrays: Sequence[LabelledArray],
        dim: str,
        labels: Any = None,
        join: str = "outer",
        fill_value: Any = None,
    ):
        """Join arrays along an existing or a new dimension.

        Parameters
        ----------
        arrays : sequence of LabelledArray
            Arrays to join. They must hold the same variables and coordinates,
            live on the same backend, and span the same dimensions.
        dim : str
            Dimension to join along. A dimension every array has is extended,
            and its labels must stay unique; a name none of them has becomes a
            new first axis, one entry per array.
        labels : array_like, optional
            Labels for ``dim`` in the result. Defaults to the joined labels of
            an existing dimension, or ``0 .. n-1`` for a new one. A natural
            index is always renumbered.
        join : {'inner', 'outer', 'left', 'right', 'exact'}, default 'outer'
            How to resolve the *other* dimensions, whose labels need not match.
        fill_value : scalar or mapping of str to scalar, optional
            Value for cells an array gains from that join.

        Returns
        -------
        LabelledArray
            An instance of the first array's class.

        Raises
        ------
        DimensionError
            If the arrays disagree on their variables, coordinates, backend or
            dimensions, if ``dim`` is a dimension of only some of them, or if
            the joined labels are not unique.

        Examples
        --------
        >>> import numpy as np
        >>> jan = LabelledArray({"w": np.arange(2.0)}, {"day": [1, 2]})
        >>> feb = LabelledArray({"w": np.arange(2.0)}, {"day": [3, 4]})
        >>> LabelledArray.concat([jan, feb], dim="day").dims["day"].tolist()
        [1, 2, 3, 4]
        >>> LabelledArray.concat([jan, feb], dim="month").shape  # 'day' is joined too
        (2, 4)
        """
        return _align.concat(
            arrays, dim, labels=labels, join=join, fill_value=fill_value
        )

    def fillna(self, value: Any):
        """Return the array with unpopulated and null cells filled.

        A cell is filled when the mask marks it absent or the value there is
        null, so the result holds data everywhere and the mask is dropped --
        which also means :meth:`to_dataframe` then emits every cell.

        Parameters
        ----------
        value : scalar or mapping of str to scalar
            Fill value, for every variable or per named variable. Variables a
            mapping leaves out keep their nulls, and the mask is then kept too.

        Examples
        --------
        >>> import numpy as np
        >>> la = LabelledArray({"w": np.array([1.0, np.nan])}, {"b": ["x", "y"]})
        >>> la.fillna(0.0)["w"].tolist()
        [1.0, 0.0]
        """
        values = per_name(value, self.variables, "value", "variables")
        absent = None if self._mask is None else ~self._mask
        data = {}
        for name, array in self._data.items():
            if values[name] is None:
                data[name] = array
                continue
            empty = is_null(self._backend.to_numpy(array))
            if absent is not None:
                empty = empty | absent
            data[name] = self._backend.where(~empty, array, values[name])
        filled_all = all(values[name] is not None for name in self.variables)
        return self._replace(data=data, mask=None if filled_all else self._mask)

    def ffill(self, dim: str, limit: int | None = None):
        """Return the array with the last present position carried forward along ``dim``.

        A cell the mask marks absent takes the value of the nearest present
        cell before it along ``dim``, in label order, and becomes present
        itself. Filled cells have to become present: were they left absent,
        :meth:`to_dataframe` would drop the very rows the fill was performed to
        create.

        The fill is over the mask, not over NaN. A masked-in NaN is a real
        observation of nothing and is left alone, as everywhere else here.

        Parameters
        ----------
        dim : str
            Dimension to carry along.
        limit : int, optional
            Carry at most this many positions past a present cell. Cells
            further than that stay absent. Unlimited by default.

        Returns
        -------
        LabelledArray
            An instance of this class with the same dimensions and labels.
            Cells before the first present position along ``dim`` are
            untouched, and stay absent at their fill value. An array with no
            mask is fully populated, so it comes back unchanged.

        Raises
        ------
        DimensionError
            If ``dim`` is not a dimension of this array, or is a natural
            index: the padding cells a natural axis holds were never rows, and
            filling them would give each the label of its source, so the
            result could not round-trip through :meth:`to_dataframe`.
        ValueError
            If ``limit`` is less than 1.

        Notes
        -----
        Variables, coordinates and the mask are gathered together, so a filled
        cell's coordinates name the cell its value came from rather than
        itself. For a real coordinate that is usually what you want -- it is
        what says how stale the carried value is -- but it does mean a
        coordinate varying along ``dim`` no longer matches that dimension's
        label at a filled cell. For the coordinate a natural index stands in
        for it would be the cell's whole identity, which is why a natural
        ``dim`` is refused.

        The age of a carried value needs no new concept: take the mask before
        the fill, derive the age from it, and store it as a coordinate, which
        costs nothing because the age varies along ``dim`` alone::

            present = curve.mask.any(axis=-1)
            pos = np.arange(len(dates))
            last = np.maximum.accumulate(np.where(present, pos, -1))
            age = np.where(last >= 0, dates - dates[last], np.timedelta64("NaT"))
            curve = curve.ffill("date")._replace(
                coords={**curve.coords, "fill_age": age[:, None]}
            )

        ``examples/09_ffill_and_staleness.py`` works this through end to end.

        Examples
        --------
        >>> import numpy as np
        >>> la = LabelledArray(
        ...     {"w": np.array([1.0, 0.0, 0.0])},
        ...     {"b": ["x", "y", "z"]},
        ...     mask=np.array([True, False, False]),
        ... )
        >>> la.ffill("b")["w"].tolist()
        [1.0, 1.0, 1.0]
        >>> la.ffill("b", limit=1).mask.tolist()
        [True, True, False]

        See Also
        --------
        fillna : fill every absent cell with one value and drop the mask.
        reindex : put the array on another label set, marking what is new absent.
        """
        return _fill.ffill(self, dim, limit)

    # -- reductions ----------------------------------------------------------------

    def reduce(self, op: str, dim: str | Sequence[str] | None = None):
        """Reduce every variable over the named dimensions.

        Only populated cells take part: those the mask marks absent are
        replaced with the reduction's identity first, so a sparse array reduces
        to what its rows say and nothing else. A NaN that is masked in is data,
        and propagates.

        Parameters
        ----------
        op : {'sum', 'mean', 'min', 'max'}
            Reduction to apply.
        dim : str or sequence of str, optional
            Dimensions to reduce over; defaults to all of them, which gives a
            zero-dimensional array.

        Returns
        -------
        LabelledArray
            An instance of this class without the reduced dimensions. Cells
            with no populated input are filled and marked absent in the mask,
            so an empty sum is absent rather than zero. Coordinates that vary
            along a reduced dimension are dropped, since there is no longer a
            cell for them to label.

        Raises
        ------
        TypeError
            If a variable's dtype cannot be reduced: strings and objects have
            no ordering the backends share, and dates cannot be summed.

        Examples
        --------
        >>> import numpy as np
        >>> la = LabelledArray({"w": np.arange(6.0).reshape(2, 3)},
        ...                    {"a": [1, 2], "b": ["x", "y", "z"]})
        >>> la.sum("b")["w"].tolist()
        [3.0, 12.0]
        >>> la.mean().values.tolist()
        2.5

        See Also
        --------
        groupby : reduce within groups of one dimension.
        rolling : reduce over trailing windows along one dimension.
        """
        return _ops.reduce(self, op, dim)

    def sum(self, dim: str | Sequence[str] | None = None):
        """Sum over the named dimensions. See :meth:`reduce`."""
        return _ops.reduce(self, "sum", dim)

    def mean(self, dim: str | Sequence[str] | None = None):
        """Average over the named dimensions, counting only populated cells.

        See :meth:`reduce`.
        """
        return _ops.reduce(self, "mean", dim)

    def min(self, dim: str | Sequence[str] | None = None):
        """Smallest value over the named dimensions. See :meth:`reduce`."""
        return _ops.reduce(self, "min", dim)

    def max(self, dim: str | Sequence[str] | None = None):
        """Largest value over the named dimensions. See :meth:`reduce`."""
        return _ops.reduce(self, "max", dim)

    def groupby(self, by: Any, dim: str | None = None):
        """Group one dimension, for reducing within each group.

        Parameters
        ----------
        by : str or array_like
            The group each label of the dimension belongs to. Given as a name,
            it is either an alternative dimension -- which is already one value
            per label of the dimension it describes -- or a coordinate that
            varies along exactly one dimension; either way that dimension is the
            one grouped. Given as an array, it holds one group value per label
            of ``dim``.
        dim : str, optional
            Dimension to group. Required when ``by`` is an array; checked
            against the alternative dimension or coordinate when ``by`` names
            one.

        Returns
        -------
        GroupBy
            Supporting ``sum``, ``mean``, ``min`` and ``max``, each returning
            an array whose grouping dimension is labelled by group value.

        Raises
        ------
        DimensionError
            If ``by`` names a dimension rather than an alternative dimension or
            a coordinate, if a coordinate varies along more than one dimension,
            or if an array of group values is the wrong length.

        Examples
        --------
        >>> import numpy as np
        >>> la = LabelledArray({"w": np.arange(4.0)}, {"asset": ["a", "b", "c", "d"]})
        >>> la.groupby(["tech", "tech", "banks", "banks"], dim="asset").sum()["w"].tolist()
        [5.0, 1.0]
        >>> tagged = la._replace(
        ...     dims=Dimensions(la.dims, alt={"sector": ("asset", ["t", "t", "b", "b"])})
        ... )
        >>> tagged.groupby("sector").sum()["w"].tolist()
        [5.0, 1.0]

        Notes
        -----
        The grouping dimension comes back labelled by group value, so an
        alternative dimension on it describes labels that are gone and is
        dropped.
        """
        return _ops.groupby(self, by, dim)

    def rolling(self, dim: str, window: int, min_periods: int | None = None):
        """Return trailing windows along one dimension, for reducing over.

        The window at each position covers that position and the ``window - 1``
        before it, so the result keeps this array's shape, labels and
        coordinates. The dimension this is meant for is a natural index -- the
        positions of a date within its group -- where neighbouring positions are
        neighbouring observations.

        Parameters
        ----------
        dim : str
            Dimension to roll along.
        window : int
            Number of positions in each window, between 1 and the length of the
            dimension.
        min_periods : int, optional
            Populated cells a window needs before it produces a value.
            Defaults to ``window``, so partial windows -- the first few
            positions, and any window overlapping absent cells -- are marked
            absent rather than being computed from less data than asked for.

        Returns
        -------
        Rolling
            Supporting ``sum``, ``mean``, ``min`` and ``max``.

        Examples
        --------
        >>> import numpy as np
        >>> la = LabelledArray({"w": np.arange(5.0)}, {"t": [0, 1, 2, 3, 4]})
        >>> la.rolling("t", 2).sum()["w"].tolist()
        [nan, 1.0, 3.0, 5.0, 7.0]
        """
        return _ops.rolling(self, dim, window, min_periods=min_periods)

    # -- DataFrame conversion ----------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        cols: str | Sequence[str] | None = None,
        dims: str | Sequence[str] | None = None,
        *,
        natural: str | Sequence[str] | None = None,
        alt: Mapping[str, str] | None = None,
        fill_value: Any = None,
        dtype: Any = None,
        mask: bool = True,
        attrs: Mapping[str, Any] | None = None,
        backend: str | ArrayBackend | None = None,
        engine: str | None = None,
    ) -> LabelledArray:
        """Build a labelled array from a pandas or polars DataFrame.

        Each dimension column contributes its sorted unique values as that
        dimension's labels, and every value column becomes a variable of the
        resulting dense array. Cells no row populates are recorded in the mask
        and filled with a placeholder the dtype can hold.

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            Source frame. A pandas (Multi)Index is moved into columns first.
        cols : str or sequence of str, optional
            Columns to turn into variables; defaults to every column that is not
            a dimension.
        dims : str or sequence of str, optional
            Columns identifying a cell, in axis order; defaults to the names of
            the frame's index levels, which polars frames never have.
        natural : str or sequence of str, optional
            Dimensions to replace with a natural index. Each named dimension
            becomes ``<name>_idx``, labelled by position within the other
            dimensions rather than by value, and its values are kept as a
            coordinate so the round trip restores the original column. This is
            what keeps a dimension whose labels are near-unique per group --
            dates, typically -- from blowing the array up.
        alt : mapping of str to str, optional
            Columns to fold in as *alternative* dimensions, each mapped to the
            dimension it describes. The column must say one thing per entry of
            that dimension, and is stored once per entry rather than once per
            cell -- which is what keeps a descriptive column from becoming a
            dimension of its own and leaving the array mostly empty. It cannot
            be tied to a dimension named in ``natural``, whose labels are
            positions rather than values.
        fill_value : scalar or mapping of str to scalar, optional
            Value for unpopulated cells, per variable. Defaults to NaN, NaT,
            empty string, False, None or 0, whichever the dtype can hold.
        dtype : dtype or mapping of str to dtype, optional
            Dtype per variable; defaults to the dtype of the column.
        mask : bool, default True
            Record which cells came from a row. Set False to save memory when
            the frame is known to be dense; the mask is dropped automatically
            when every cell is populated.
        attrs : mapping, optional
            Metadata to attach.
        backend : str or ArrayBackend, optional
            Array backend for the variables; defaults to numpy.
        engine : str, optional
            Force a DataFrame engine instead of detecting it from ``df``.

        Returns
        -------
        LabelledArray
            An instance of this class.

        Raises
        ------
        DuplicateIndexError
            If two rows map to the same cell, making the conversion ambiguous.
        KeyError
            If a named column is not in the frame.

        Notes
        -----
        Labels come from the values actually present, so a filtered frame never
        produces an array padded out to labels that no row uses.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"], "w": [1.0, 2.0, 3.0]})
        >>> la = LabelledArray.from_dataframe(df, cols="w", dims=["a", "b"])
        >>> la.shape
        (2, 2)
        >>> la.density
        0.75

        See Also
        --------
        to_dataframe : the inverse conversion.
        """
        return _convert.from_dataframe(
            cls,
            df,
            cols=cols,
            dims=dims,
            natural=natural,
            alt=alt,
            fill_value=fill_value,
            dtype=dtype,
            mask=mask,
            attrs=attrs,
            backend=backend,
            engine=engine,
        )

    def to_dataframe(
        self,
        engine: str | None = None,
        *,
        index: bool = True,
        dropna: bool = False,
        expand_natural: bool = True,
    ) -> Any:
        """Convert to a DataFrame with one row per populated cell.

        Parameters
        ----------
        engine : str, optional
            DataFrame engine, ``'pandas'`` (the default) or ``'polars'``.
        index : bool, default True
            Put the dimension columns into the frame's index. polars has no
            index and ignores this.
        dropna : bool, default False
            Also drop rows in which every variable is null. Unnecessary when a
            mask is present, which already excludes unpopulated cells.
        expand_natural : bool, default True
            Write a natural index dimension out as the values it stands in for,
            taken from its coordinate, instead of as the index itself.

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            Dimension columns first, then any remaining coordinates, then the
            variables.

        Notes
        -----
        Rows come out ordered by the dimension labels, so a round trip
        reproduces the source frame up to that ordering.

        See Also
        --------
        from_dataframe : the inverse conversion.
        """
        return _convert.to_dataframe(
            self, engine, index=index, dropna=dropna, expand_natural=expand_natural
        )

    # -- comparison --------------------------------------------------------------

    def equals(self, other: object) -> bool:
        """Whether dimensions, variables, coordinates and mask are all identical.

        Values are compared with nulls matching nulls, so unpopulated cells do
        not make two otherwise identical arrays differ. Attributes, variable
        dtypes and the array backend are ignored, so the same data built from
        pandas and from polars, or on numpy and on torch, compares equal.
        """
        if not isinstance(other, LabelledArray):
            return False
        if self._dims != other._dims or self.variables != other.variables:
            return False
        if (self._mask is None) != (other._mask is None):
            return False
        if self._mask is not None and not np.array_equal(self._mask, other._mask):
            return False
        if set(self._coords) != set(other._coords):
            return False
        if not all(
            arrays_equal(value, other._coords[name])
            for name, value in self._coords.items()
        ):
            return False
        return all(
            arrays_equal(
                self._backend.to_numpy(value),
                other._backend.to_numpy(other._data[name]),
            )
            for name, value in self._data.items()
        )

    # -- magic methods -----------------------------------------------------------

    def __getitem__(self, key: str | Sequence[str]) -> Any:
        """Return a variable, a subset of variables, or a label selection.

        A string returns that variable's array, a list of strings returns an
        array holding just those variables, and a mapping is passed to
        :meth:`select`.
        """
        if isinstance(key, str):
            try:
                return self._data[key]
            except KeyError:
                raise KeyError(
                    f"unknown variable {key!r}, expected one of {list(self._data)}"
                ) from None
        if isinstance(key, Mapping):
            return self.select(key)
        if isinstance(key, (list, tuple)) and all(
            isinstance(name, str) for name in key
        ):
            missing = [name for name in key if name not in self._data]
            if missing:
                raise KeyError(f"unknown variables: {missing}")
            return self._replace(data={name: self._data[name] for name in key})
        raise TypeError(
            f"index must be a variable name, a list of names, or a mapping of "
            f"dimension to labels, got {type(key).__name__}"
        )

    def __setitem__(self, key: str, value: Any) -> None:
        """Add or replace a variable in place.

        Validated exactly as :meth:`assign` is, so a subclass's own invariants
        hold however a variable is set.
        """
        self._data = self.assign({key: value})._data

    def __delitem__(self, key: str) -> None:
        """Remove a variable in place, validating what is left."""
        self._data = self.drop_vars(key)._data

    def __contains__(self, key: object) -> bool:
        """Whether ``key`` is a variable name."""
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        """Iterate over the variable names."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of variables (not cells; see :attr:`size` for those)."""
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        """Structural equality, as :meth:`equals`; not an element-wise comparison."""
        if not isinstance(other, LabelledArray):
            return NotImplemented
        return self.equals(other)

    __hash__ = None  # type: ignore[assignment]

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """Return the sole variable as a numpy array; requires one variable."""
        array = self._backend.to_numpy(self.values)
        if dtype is not None:
            array = array.astype(dtype)
        return array.copy() if copy else array

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo: dict | None = None):
        return self.copy(deep=True)

    def __getstate__(self) -> dict[str, Any]:
        """Return the pickled state, recording the backend by name."""
        return {
            "data": {
                name: self._backend.to_numpy(value)
                for name, value in self._data.items()
            },
            "dims": self._dims,
            "mask": self._mask,
            "coords": self._coords,
            "attrs": self._attrs,
            "backend": self._backend.name,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state)

    def __repr__(self) -> str:
        """Return an ``eval``-able constructor call when small enough.

        Evaluating it needs ``LabelledArray``, ``Dimensions`` and ``array``
        (i.e. :func:`numpy.array`) in scope. Above
        :attr:`_repr_full_threshold` cells the summary from :meth:`__str__` is
        returned instead, which is deliberately not valid Python rather than
        being a silently truncated constructor.
        """
        if self.size > self._repr_full_threshold:
            return f"{self}\n# too large for an eval-able repr; use to_repr(full=True)"
        return self.to_repr()

    def to_repr(self, full: bool = True) -> str:
        """Return an ``eval``-able constructor call.

        Parameters
        ----------
        full : bool, default True
            Print every element. When False, long arrays are summarised and the
            result is no longer ``eval``-able.
        """
        parts = [
            "data={"
            + ", ".join(
                f"{name!r}: {array_repr(self._backend.to_numpy(value), full=full)}"
                for name, value in self._data.items()
            )
            + "}",
            f"dims={self._dims.to_repr(full=full)}",
        ]
        if self._mask is not None:
            parts.append(f"mask={array_repr(self._mask, full=full)}")
        if self._coords:
            parts.append(
                "coords={"
                + ", ".join(
                    f"{name!r}: {array_repr(value, full=full)}"
                    for name, value in self._coords.items()
                )
                + "}"
            )
        if self._attrs:
            parts.append(f"attrs={self._attrs!r}")
        if self._backend.name != self._default_backend:
            parts.append(f"backend={self._backend.name!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __str__(self) -> str:
        """Return a human-readable summary of the dimensions and variables."""
        lines = [
            f"{type(self).__name__}: {len(self._data)} variable(s) over "
            f"{self.ndim} dimension(s), shape {self.shape}, backend {self._backend.name}"
        ]
        lines.append(str(self._dims))
        if self._data:
            width = max(len(name) for name in self._data)
            lines.append("Variables:")
            lines.extend(
                f"  {name:<{width}} {dtype}" for name, dtype in self.dtypes.items()
            )
        if self._coords:
            width = max(len(name) for name in self._coords)
            lines.append("Coordinates:")
            lines.extend(
                f"  {name:<{width}} {value.dtype}"
                for name, value in self._coords.items()
            )
        if self._mask is not None:
            populated = int(self._mask.sum())
            density = populated / self._mask.size if self._mask.size else 0.0
            lines.append(
                f"Mask: {populated} / {self._mask.size} cells populated ({density:.1%} density)"
            )
        if self._attrs:
            lines.append(f"Attrs: {self._attrs!r}")
        return "\n".join(lines)


def _as_mask(mask: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    """Return ``mask`` as a read-only boolean array of ``shape``, or None."""
    if mask is None:
        return None
    array = np.asarray(mask, dtype=bool)
    if array.shape != shape:
        raise DimensionError(f"mask has shape {array.shape}, expected {shape}")
    array = array.copy() if array.flags.writeable else array
    array.flags.writeable = False
    return array


def _as_coord(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Return coordinate ``value`` as a read-only numpy array of ``shape``.

    A coordinate that does not vary along every dimension may be given at any
    broadcastable shape -- ``(n, 1)`` against ``(n, m)``, or ``(m,)`` against
    the trailing axis. It is stored broadcast, so every consumer still reads a
    full-shape array, but the values behind it are held once.

    Notes
    -----
    The result is frozen through a fresh view rather than in place, so freezing
    it does not make the caller's own array read-only. It has to be frozen: a
    broadcast coordinate has a zero stride, and one assignment through it would
    silently update every cell sharing the underlying element.
    """
    array = np.asarray(value)
    if array.shape != shape:
        try:
            array = np.broadcast_to(array, shape)
        except ValueError as exc:
            raise DimensionError(
                f"coordinate {name!r} has shape {array.shape}, "
                f"which is not broadcastable to {shape}"
            ) from exc
    array = array.view()
    array.flags.writeable = False
    return array
