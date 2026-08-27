"""Ordered, validated dimension names and labels."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from labelled_array._utils import array_repr, as_name_list, duplicates, is_null
from labelled_array.exceptions import DimensionError, LabelNotFoundError

__all__ = ["Dimensions"]

#: Accepted values for the ``missing`` argument of :meth:`Dimensions.get_indexer`.
MISSING_MODES = ("raise", "ignore")


def as_1d(value: Any, name: str, what: str = "labels") -> np.ndarray:
    """Return ``value`` as a one-dimensional array, treating a scalar as one entry.

    Parameters
    ----------
    value : array_like
        Value to coerce.
    name : str
        Dimension the value belongs to, used in the error message.
    what : str, default 'labels'
        What the value is, used in the error message.
    """
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise DimensionError(
            f"{what} for dimension {name!r} must be one-dimensional, got shape {array.shape}"
        )
    return array


def _as_labels(name: str, labels: Any) -> np.ndarray:
    """Return ``labels`` as a read-only, non-null, unique, one-dimensional array."""
    array = as_1d(labels, name)
    if is_null(array).any():
        raise DimensionError(
            f"labels for dimension {name!r} must not be null: a null label can never "
            f"be selected, as it does not compare equal to itself"
        )
    repeated = duplicates(array)
    if repeated:
        raise DimensionError(
            f"labels for dimension {name!r} must be unique, duplicated: {repeated}"
        )
    if array.flags.writeable:
        array = array.copy()
        array.flags.writeable = False
    return array


def _as_natural(natural: Any, names: Sequence[str]) -> dict[str, str | None]:
    """Return the natural-index flags as a ``dimension -> source`` mapping.

    A plain iterable of names flags them as positional with no source.
    """
    if isinstance(natural, Mapping):
        flags = dict(natural)
    else:
        flags = dict.fromkeys(natural, None)
    unknown = sorted(set(flags) - set(names))
    if unknown:
        raise DimensionError(f"natural names are not dimensions: {unknown}")
    return flags


def _as_alt_labels(name: str, labels: Any, length: int) -> np.ndarray:
    """Return ``labels`` as read-only, non-null, one-dimensional labels of ``length``.

    Unlike :func:`_as_labels` duplicates are allowed, and are the point: an
    alternative dimension describes the entries of the dimension it is tied to,
    and several of those usually share a description.
    """
    array = as_1d(labels, name, "alternative labels")
    if is_null(array).any():
        raise DimensionError(
            f"labels for alternative dimension {name!r} must not be null: a null "
            f"label can never be selected, as it does not compare equal to itself"
        )
    if array.size != length:
        raise DimensionError(
            f"alternative dimension {name!r} has {array.size} labels, expected "
            f"{length} to match the dimension it refers to"
        )
    if array.flags.writeable:
        array = array.copy()
        array.flags.writeable = False
    return array


def _as_alt(
    alt: Mapping[str, Any] | None, labels: Mapping[str, np.ndarray]
) -> dict[str, tuple[str, np.ndarray]]:
    """Return the alternative dimensions as a ``name -> (ref_dim, labels)`` mapping.

    Each entry is given either as ``{"ref_dim": ..., "labels": ...}`` or as the
    terser ``(ref_dim, labels)`` pair.
    """
    resolved: dict[str, tuple[str, np.ndarray]] = {}
    for name, spec in (alt or {}).items():
        if not isinstance(name, str) or not name:
            raise DimensionError(
                f"alternative dimension names must be non-empty str, got {name!r}"
            )
        if name in labels:
            raise DimensionError(
                f"alternative dimension {name!r} is already a dimension"
            )
        if isinstance(spec, Mapping):
            absent = sorted({"ref_dim", "labels"} - set(spec))
            if absent:
                raise DimensionError(
                    f"alternative dimension {name!r} needs 'ref_dim' and 'labels', "
                    f"missing {absent}"
                )
            ref, values = spec["ref_dim"], spec["labels"]
        else:
            ref, values = spec
        if ref not in labels:
            raise DimensionError(
                f"alternative dimension {name!r} refers to unknown dimension {ref!r}, "
                f"expected one of {list(labels)}"
            )
        resolved[name] = (ref, _as_alt_labels(name, values, len(labels[ref])))
    return resolved


def _comparable(reference: np.ndarray, name: str, labels: Any) -> np.ndarray:
    """Return ``labels`` as a one-dimensional array comparable with ``reference``."""
    query = as_1d(labels, name, "labels selected")
    # datetime64/timedelta64 compare across units only once aligned; other
    # kinds are matched via their python equivalents, which avoids the silent
    # truncation of casting (e.g. '<U8' labels cast to '<U1').
    if reference.dtype.kind in "mM" and query.dtype != reference.dtype:
        with contextlib.suppress(TypeError, ValueError):
            query = query.astype(reference.dtype)
    return query


def _gathered(values: np.ndarray, positions: Any) -> np.ndarray:
    """Return ``values`` at ``positions``, read-only.

    Fancy indexing always copies, so the result owns its buffer and freezing it
    leaves the caller's array alone.
    """
    picked = values[np.asarray(positions, dtype=np.intp)]
    picked.flags.writeable = False
    return picked


class Dimensions(Mapping[str, np.ndarray]):
    """An ordered mapping of dimension name to its unique labels.

    Labels are always held as numpy arrays, independently of the array backend,
    and are read-only: a ``Dimensions`` instance is treated as immutable, and
    every method returning a modified version returns a new instance.

    Parameters
    ----------
    dims : Mapping or sequence of (str, array_like) or Dimensions
        Dimension names in axis order, each mapped to its labels.
    natural : iterable of str or mapping of str to str, optional
        Dimensions whose labels are a natural index (``0 .. n-1``) rather than
        values of their own. Given as a mapping, each value names the
        coordinate holding the values the index stands in for, which is what a
        DataFrame round trip writes back out; given as a plain iterable, the
        dimensions are simply positional.
    alt : mapping of str to mapping or tuple, optional
        Alternative dimensions: extra labels for the entries of a dimension the
        array already has, rather than an axis of their own. Each is given as
        ``{"ref_dim": name, "labels": values}``, or as the terser
        ``(name, values)`` pair, and has one label per entry of ``ref_dim``.
        Unlike a dimension's own labels these need not be unique -- an
        alternative dimension describes the entries, so several of them
        usually share a description -- and :meth:`alt_indexer` maps one of
        them to every position that carries it.

    Raises:
    ------
    DimensionError
        If a name is repeated or is not a string, if labels are not
        one-dimensional, null, or unique within a dimension, if a name given
        in ``natural`` is not a dimension, or if an alternative dimension
        collides with a dimension name, names an unknown ``ref_dim``, or has
        the wrong number of labels.

    Examples:
    --------
    >>> dims = Dimensions({"a": [1, 2], "b": ["x", "y", "z"]})
    >>> dims.shape
    (2, 3)
    >>> dims.get_indexer("b", ["z", "x"])
    array([2, 0])
    >>> tied = Dimensions({"a": [1, 2, 3]}, alt={"g": ("a", ["x", "x", "y"])})
    >>> tied.alt_indexer("g", "x")
    array([0, 1])
    """

    __slots__ = (
        "_labels",
        "_names",
        "_axes",
        "_shape",
        "_natural",
        "_alt",
        "_lookups",
        "_alt_lookups",
    )

    def __init__(
        self,
        dims: Mapping[str, Any] | Sequence[tuple[str, Any]] | Dimensions,
        natural: Iterable[str] | Mapping[str, str | None] = (),
        alt: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(dims, Dimensions):
            natural = {**dims._natural, **_as_natural(natural, dims.names)}
            alt = {**dims._alt, **(alt or {})}
            items = list(dims.items())
        elif isinstance(dims, Mapping):
            items = list(dims.items())
        else:
            items = [tuple(item) for item in dims]

        names = [name for name, _ in items]
        for name in names:
            if not isinstance(name, str):
                raise DimensionError(f"dimension names must be str, got {name!r}")
            if not name:
                raise DimensionError("dimension names must not be empty")
        repeated = duplicates(names)
        if repeated:
            raise DimensionError(
                f"dimension names must be unique, duplicated: {repeated}"
            )

        resolved = {name: _as_labels(name, labels) for name, labels in items}
        self._init(
            resolved,
            _as_natural(natural, names),
            _as_alt(alt, resolved),
        )

    def _init(
        self,
        labels: dict[str, np.ndarray],
        natural: dict[str, str | None],
        alt: dict[str, tuple[str, np.ndarray]],
    ) -> None:
        """Populate the instance from labels that are already validated."""
        self._labels = labels
        self._names = tuple(labels)
        self._axes = {name: axis for axis, name in enumerate(self._names)}
        self._shape = tuple(len(values) for values in labels.values())
        self._natural = natural
        self._alt = alt
        self._lookups: dict[str, dict[Hashable, int]] = {}
        self._alt_lookups: dict[str, dict[Hashable, np.ndarray]] = {}

    def _derive(
        self,
        labels: dict[str, np.ndarray],
        natural: dict[str, str | None] | None = None,
        alt: dict[str, tuple[str, np.ndarray]] | None = None,
    ) -> Dimensions:
        """Return a new instance from labels already known to be valid.

        Deriving one set of dimensions from another -- renaming, reordering,
        subsetting -- cannot introduce a null or a duplicate that was not
        already there, so re-validating every label array would be pure cost.
        """
        derived = object.__new__(type(self))
        derived._init(
            labels,
            self._natural if natural is None else natural,
            self._alt if alt is None else alt,
        )
        return derived

    # -- alternative constructors ------------------------------------------------

    @classmethod
    def from_shape(
        cls, shape: Sequence[int], names: Sequence[str] | None = None
    ) -> Dimensions:
        """Build dimensions with natural index labels for each entry of ``shape``.

        Parameters
        ----------
        shape : sequence of int
            Length of each dimension.
        names : sequence of str, optional
            Dimension names; defaults to ``dim_0, dim_1, ...``.
        """
        if names is None:
            names = [f"dim_{i}" for i in range(len(shape))]
        elif len(names) != len(shape):
            raise DimensionError(f"got {len(names)} names for {len(shape)} dimensions")
        return cls(
            {name: np.arange(n) for name, n in zip(names, shape, strict=False)},
            natural=names,
        )

    # -- mapping interface -------------------------------------------------------

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self._labels[name]
        except KeyError:
            raise DimensionError(
                f"unknown dimension {name!r}, expected one of {list(self._names)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` is a dimension."""
        return name in self._labels

    def __len__(self) -> int:
        return len(self._names)

    # -- properties --------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        """Dimension names, in axis order."""
        return self._names

    @property
    def shape(self) -> tuple[int, ...]:
        """Length of each dimension, in axis order."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._names)

    @property
    def size(self) -> int:
        """Number of cells spanned, i.e. the product of the dimension lengths."""
        return math.prod(self._shape)

    @property
    def natural(self) -> Mapping[str, str | None]:
        """Natural index dimensions, mapped to the coordinate each stands in for."""
        return dict(self._natural)

    def is_natural(self, name: str) -> bool:
        """Whether ``name`` is a natural index."""
        self._require([name])
        return name in self._natural

    def source(self, name: str) -> str | None:
        """Return the coordinate ``name`` stands in for, if it is a natural index."""
        self._require([name])
        return self._natural.get(name)

    @property
    def alt(self) -> Mapping[str, np.ndarray]:
        """Alternative dimension labels, keyed by name."""
        return {name: values for name, (_, values) in self._alt.items()}

    def is_alt(self, name: str) -> bool:
        """Whether ``name`` is an alternative dimension."""
        return name in self._alt

    def alt_ref(self, name: str) -> str:
        """Return the dimension alternative dimension ``name`` is tied to."""
        self._require_alt([name])
        return self._alt[name][0]

    def alt_on(self, name: str) -> dict[str, np.ndarray]:
        """Return the alternative dimensions tied to dimension ``name``."""
        self._require([name])
        return {alt: values for alt, (ref, values) in self._alt.items() if ref == name}

    def axis(self, name: str) -> int:
        """Return the axis number of dimension ``name``."""
        self._require([name])
        return self._axes[name]

    def _require(self, names: Iterable[str]) -> None:
        """Raise unless every name is a dimension."""
        unknown = [name for name in names if name not in self._labels]
        if unknown:
            raise DimensionError(
                f"unknown dimensions: {unknown}, expected some of {list(self._names)}"
            )

    def _require_alt(self, names: Iterable[str]) -> None:
        """Raise unless every name is an alternative dimension."""
        unknown = [name for name in names if name not in self._alt]
        if unknown:
            raise DimensionError(
                f"unknown alternative dimensions: {unknown}, "
                f"expected some of {list(self._alt)}"
            )

    # -- lookup ------------------------------------------------------------------

    def _lookup(self, name: str) -> dict[Hashable, int]:
        """Return (and cache) the ``label -> position`` map for ``name``."""
        if name not in self._lookups:
            self._lookups[name] = {
                label: position for position, label in enumerate(self[name].tolist())
            }
        return self._lookups[name]

    def get_indexer(self, name: str, labels: Any, missing: str = "raise") -> np.ndarray:
        """Return the positions of ``labels`` within dimension ``name``.

        Parameters
        ----------
        name : str
            Dimension to look up in.
        labels : scalar or array_like
            Labels to locate. Scalars are treated as a single-element sequence.
        missing : {'raise', 'ignore'}, default 'raise'
            Whether absent labels raise :class:`LabelNotFoundError` or are
            dropped from the result.

        Returns:
        -------
        numpy.ndarray
            Integer positions, in the order the labels were given.
        """
        if missing not in MISSING_MODES:
            raise ValueError(f"missing must be one of {MISSING_MODES}, got {missing!r}")

        query = self._as_query(name, labels)
        found = self.locate(name, query)
        if np.any(found < 0):
            if missing == "raise":
                raise LabelNotFoundError(name, query[found < 0].tolist())
            found = found[found >= 0]
        return found

    def _as_query(self, name: str, labels: Any) -> np.ndarray:
        """Return ``labels`` as a one-dimensional array comparable with ``name``'s."""
        return _comparable(self[name], name, labels)

    def _alt_lookup(self, name: str) -> dict[Hashable, np.ndarray]:
        """Return (and cache) the ``label -> positions`` map for alternative ``name``.

        Alternative labels need not be unique, so a label maps to every position
        carrying it rather than to a single one.
        """
        if name not in self._alt_lookups:
            grouped: dict[Hashable, list[int]] = {}
            for position, label in enumerate(self._alt[name][1].tolist()):
                grouped.setdefault(label, []).append(position)
            self._alt_lookups[name] = {
                label: np.asarray(positions, dtype=np.intp)
                for label, positions in grouped.items()
            }
        return self._alt_lookups[name]

    def alt_indexer(self, name: str, labels: Any, missing: str = "raise") -> np.ndarray:
        """Return positions along the dimension alternative ``name`` is tied to.

        Parameters
        ----------
        name : str
            Alternative dimension to look up in.
        labels : scalar or array_like
            Alternative labels to locate. Scalars are treated as a single-element
            sequence.
        missing : {'raise', 'ignore'}, default 'raise'
            Whether absent labels raise :class:`LabelNotFoundError` or are
            dropped from the result.

        Returns:
        -------
        numpy.ndarray
            Integer positions along ``alt_ref(name)``. Because the labels need
            not be unique, one label can yield several positions: they come back
            in the order the labels were requested, and ascending within each
            label. The blocks cannot overlap, since every position carries
            exactly one alternative label.

        Examples:
        --------
        >>> dims = Dimensions({"a": [1, 2, 3]}, alt={"g": ("a", ["x", "y", "x"])})
        >>> dims.alt_indexer("g", ["y", "x"])
        array([1, 0, 2])
        """
        if missing not in MISSING_MODES:
            raise ValueError(f"missing must be one of {MISSING_MODES}, got {missing!r}")
        self._require_alt([name])
        lookup = self._alt_lookup(name)
        query = _comparable(self._alt[name][1], name, labels)

        found, absent = [], []
        for label in query.tolist():
            positions = lookup.get(label)
            if positions is None:
                absent.append(label)
            else:
                found.append(positions)
        if absent and missing == "raise":
            raise LabelNotFoundError(name, absent)
        return np.concatenate(found) if found else np.empty(0, dtype=np.intp)

    def locate(self, name: str, labels: Any) -> np.ndarray:
        """Return the position of each label within dimension ``name``, -1 where absent.

        Unlike :meth:`get_indexer` this neither raises nor drops, so the caller
        can tell which labels were missing and where -- what
        :meth:`~labelled_array.LabelledArray.reindex` needs to know which cells
        to fill.

        Parameters
        ----------
        name : str
            Dimension to look up in.
        labels : scalar or array_like
            Labels to locate.

        Returns:
        -------
        numpy.ndarray
            One integer position per label, ``-1`` where the label is absent.

        Examples:
        --------
        >>> dims = Dimensions({"b": ["x", "y", "z"]})
        >>> dims.locate("b", ["z", "q"])
        array([ 2, -1])
        """
        lookup = self._lookup(name)
        query = self._as_query(name, labels)
        return np.asarray(
            [lookup.get(label, -1) for label in query.tolist()], dtype=np.intp
        )

    def get_positions(self, name: str, value: Any) -> np.ndarray:
        """Return integer positions along dimension ``name``.

        Parameters
        ----------
        name : str
            Dimension the positions belong to.
        value : int, slice, or array_like of int or bool
            Positions to keep. A boolean array is taken as a mask over the
            dimension; negative positions count from the end.

        Returns:
        -------
        numpy.ndarray
            Integer positions, in the order given.
        """
        length = len(self[name])
        if isinstance(value, slice):
            return np.arange(*value.indices(length), dtype=np.intp)

        positions = as_1d(value, name, "positions")
        if positions.dtype == bool:
            if positions.size != length:
                raise DimensionError(
                    f"boolean mask for dimension {name!r} has length {positions.size}, "
                    f"expected {length}"
                )
            return np.flatnonzero(positions).astype(np.intp)
        if positions.size and not np.issubdtype(positions.dtype, np.integer):
            raise DimensionError(
                f"positions for dimension {name!r} must be integers, got {positions.dtype}"
            )
        positions = np.asarray(positions, dtype=np.intp)
        if positions.size and (positions.min() < -length or positions.max() >= length):
            out_of_range = positions[(positions < -length) | (positions >= length)]
            raise IndexError(
                f"positions {out_of_range.tolist()} are out of range for dimension "
                f"{name!r} of length {length}"
            )
        return positions + length * (positions < 0) if positions.size else positions

    # -- derived instances -------------------------------------------------------

    def subset(self, indexers: Mapping[str, Any]) -> Dimensions:
        """Return dimensions with the given axes reduced to integer positions.

        Parameters
        ----------
        indexers : mapping of str to array_like of int
            Positions to keep, per dimension. Dimensions not named are unchanged.

        Raises:
        ------
        DimensionError
            If a dimension is unknown, or a position is asked for twice, which
            would make its label ambiguous.

        Notes:
        -----
        Alternative dimensions are subset alongside the dimension they are tied
        to, so they keep describing the same entries.
        """
        self._require(indexers)
        labels = dict(self._labels)
        for name, positions in indexers.items():
            positions = np.asarray(positions, dtype=np.intp)
            repeated = duplicates(positions)
            if repeated:
                raise DimensionError(
                    f"labels requested more than once for dimension {name!r}: "
                    f"{self._labels[name][repeated].tolist()}"
                )
            labels[name] = self._labels[name][positions]
        alt = {
            name: (ref, _gathered(values, indexers[ref]) if ref in indexers else values)
            for name, (ref, values) in self._alt.items()
        }
        return self._derive(labels, alt=alt)

    def rename(self, mapping: Mapping[str, str]) -> Dimensions:
        """Return dimensions with names remapped by ``mapping``.

        Names of the coordinates that natural index dimensions stand in for are
        remapped too, so the association survives a rename. An alternative
        dimension can be renamed by its own name, and follows its dimension when
        that is renamed.
        """
        sources = {source for source in self._natural.values() if source is not None}
        unknown = sorted(set(mapping) - set(self._names) - sources - set(self._alt))
        if unknown:
            raise DimensionError(f"unknown dimensions: {unknown}")
        labels = {
            mapping.get(name, name): values for name, values in self._labels.items()
        }
        if len(labels) != len(self._labels):
            raise DimensionError("dimension names must be unique after renaming")
        natural = {
            mapping.get(name, name): None
            if source is None
            else mapping.get(source, source)
            for name, source in self._natural.items()
        }
        alt = {
            mapping.get(name, name): (mapping.get(ref, ref), values)
            for name, (ref, values) in self._alt.items()
        }
        clashes = sorted(set(alt) & set(labels))
        if len(alt) != len(self._alt) or clashes:
            raise DimensionError("dimension names must be unique after renaming")
        return self._derive(labels, natural, alt)

    def transpose(self, order: Sequence[str]) -> Dimensions:
        """Return dimensions reordered to ``order``, which must be a permutation."""
        if sorted(order) != sorted(self._names):
            raise DimensionError(
                f"transpose order {list(order)} is not a permutation of {list(self._names)}"
            )
        return self._derive({name: self._labels[name] for name in order})

    def drop(self, names: str | Iterable[str]) -> Dimensions:
        """Return dimensions without ``names``."""
        names = set(as_name_list(names))
        self._require(names)
        labels = {
            name: values for name, values in self._labels.items() if name not in names
        }
        natural = {
            name: source for name, source in self._natural.items() if name not in names
        }
        alt = {
            name: (ref, values)
            for name, (ref, values) in self._alt.items()
            if ref not in names
        }
        return self._derive(labels, natural, alt)

    def drop_alt(self, names: str | Iterable[str]) -> Dimensions:
        """Return dimensions without the named alternative dimensions."""
        names = set(as_name_list(names))
        self._require_alt(names)
        return self._derive(
            dict(self._labels),
            alt={name: entry for name, entry in self._alt.items() if name not in names},
        )

    def replace(
        self,
        name: str,
        labels: Any,
        natural: bool | str | None = None,
        alt: Mapping[str, Any] | None = None,
    ) -> Dimensions:
        """Return dimensions with the labels of ``name`` replaced.

        Parameters
        ----------
        name : str
            Dimension to replace the labels of.
        labels : array_like
            New labels; need not have the same length as the current ones.
        natural : bool or str, optional
            Whether the dimension is a natural index, or the name of the
            coordinate it stands in for. Unchanged when omitted.
        alt : mapping of str to array_like, optional
            Alternative dimensions to carry on the replaced dimension, one label
            per new label. Unlike ``natural`` this does **not** default to
            leaving them alone: the label set has changed, so any alternative
            dimension tied to ``name`` is dropped unless it is given here.
            Alternative dimensions on the other dimensions are untouched.
        """
        self._require([name])
        replaced = dict(self._labels)
        replaced[name] = _as_labels(name, labels)
        flags = dict(self._natural)
        if natural is False:
            flags.pop(name, None)
        elif natural is not None:
            flags[name] = None if natural is True else natural
        kept = {
            alt_name: (ref, values)
            for alt_name, (ref, values) in self._alt.items()
            if ref != name
        }
        for alt_name, values in (alt or {}).items():
            kept[alt_name] = (
                name,
                _as_alt_labels(alt_name, values, len(replaced[name])),
            )
        return self._derive(replaced, flags, kept)

    # -- comparison and display --------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Compare names, order, labels, natural flags and alternative dimensions."""
        if not isinstance(other, Dimensions):
            return NotImplemented
        return (
            self._names == other._names
            and self._natural == other._natural
            and all(
                self._labels[name].dtype == other._labels[name].dtype
                and np.array_equal(self._labels[name], other._labels[name])
                for name in self._names
            )
            and set(self._alt) == set(other._alt)
            and all(
                self._alt[name][0] == other._alt[name][0]
                and self._alt[name][1].dtype == other._alt[name][1].dtype
                and np.array_equal(self._alt[name][1], other._alt[name][1])
                for name in self._alt
            )
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        """Return an ``eval``-able representation (needs ``array`` in scope)."""
        return self.to_repr()

    def to_repr(self, full: bool = True) -> str:
        """Return an ``eval``-able representation.

        Parameters
        ----------
        full : bool, default True
            Print every label. When False, long label arrays are summarised by
            numpy and the result is no longer ``eval``-able.
        """
        body = ", ".join(
            f"{name!r}: {array_repr(self._labels[name], full=full)}"
            for name in self._names
        )
        natural = f", natural={self._natural!r}" if self._natural else ""
        alt = ""
        if self._alt:
            pairs = ", ".join(
                f"{name!r}: ({ref!r}, {array_repr(values, full=full)})"
                for name, (ref, values) in self._alt.items()
            )
            alt = f", alt={{{pairs}}}"
        return f"{type(self).__name__}({{{body}}}{natural}{alt})"

    def __str__(self) -> str:
        """Return a one-line-per-dimension summary."""
        if not self._names:
            return f"{type(self).__name__}: <scalar>"
        width = max(len(name) for name in (*self._names, *self._alt))
        lines = [f"{type(self).__name__} ({self.ndim}d, size {self.size}):"]
        for name in self._names:
            labels = self._labels[name]
            source = self._natural.get(name)
            flag = (
                f" [natural for {source!r}]"
                if source
                else (" [natural]" if name in self._natural else "")
            )
            preview = np.array2string(labels, threshold=6, edgeitems=3, separator=", ")
            lines.append(
                f"  {name:<{width}} ({len(labels)}) {labels.dtype}{flag}: {preview}"
            )
        for name, (ref, labels) in self._alt.items():
            preview = np.array2string(labels, threshold=6, edgeitems=3, separator=", ")
            lines.append(
                f"  {name:<{width}} ({len(labels)}) {labels.dtype} "
                f"[alt for {ref!r}]: {preview}"
            )
        return "\n".join(lines)
