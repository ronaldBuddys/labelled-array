"""The three kinds of value the checkpoints take that regression-records cannot record
faithfully on its own: a labelled array, a slice and a class.

regression-records stores an object as its attributes, and writes every record with
sorted keys, so a dict comes back in alphabetical order. labelled-array reads meaning
into the order of its dicts: a ``Dimensions``' axes and labels, an array's variables.
An array over ``(b, a)`` would come back over ``(a, b)``, and replay would transpose
or fail. :class:`LabelledArraySerializer` stores every dict inside one as an ordered
list of pairs. Comparison is unaffected: regression-records compares dicts by key.

A ``slice`` (the indexers of ``isel`` and ``select``) is not a supported leaf type;
:class:`SliceSerializer` stores its three bounds. The implementation behind
``LabelledArray.from_dataframe`` takes the class as its first argument, which
:class:`ClassSerializer` stores by its import path.

:data:`REGISTRY` is a copy of regression-records' default registry with all three added,
so nothing outside the harness sees it.
"""

from __future__ import annotations

import importlib
from typing import Any, NamedTuple

import numpy as np
from regression_records import DeserializationError
from regression_records.serialization import (
    DeserializationContext,
    SerializationContext,
    SerializerRegistry,
    default_registry,
    from_json_tree,
    to_json_tree,
)

from labelled_array import Dimensions, LabelledArray

__all__ = [
    "REGISTRY",
    "LabelledArraySerializer",
    "OrderedItems",
    "SliceSerializer",
    "ClassSerializer",
]


def _resolve(module: str, qualname: str) -> Any:
    """Import ``module`` and walk ``qualname`` down from it, attribute by attribute."""
    target: Any = importlib.import_module(module)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


class OrderedItems(NamedTuple):
    """A dict's items in insertion order, as a record keeps them (a named tuple is
    stored by its fields, and its list in order)."""

    items: list


class ObjectArray(NamedTuple):
    """An object-dtype array as a flat list and a shape, which a record can hold.

    regression-records refuses ``dtype=object`` arrays, and polars hands back string
    labels as exactly that.
    """

    values: list
    shape: tuple


def _keep_order(value: Any) -> Any:
    """Replace every dict in ``value`` by its :class:`OrderedItems`, recursively.

    An object-dtype array becomes an :class:`ObjectArray`, for the same reason.
    """
    if isinstance(value, np.ndarray) and value.dtype == object:
        return ObjectArray(value.ravel().tolist(), value.shape)
    if type(value) is dict:
        return OrderedItems([(key, _keep_order(item)) for key, item in value.items()])
    if type(value) is list:
        return [_keep_order(item) for item in value]
    if type(value) is tuple:
        return tuple(_keep_order(item) for item in value)
    return value


def _restore_order(value: Any) -> Any:
    """Undo :func:`_keep_order`: each :class:`OrderedItems` back to a dict, in order."""
    if isinstance(value, ObjectArray):
        array = np.empty(len(value.values), dtype=object)
        array[:] = value.values
        return array.reshape(value.shape)
    if isinstance(value, OrderedItems):
        return {key: _restore_order(item) for key, item in value.items}
    if type(value) is list:
        return [_restore_order(item) for item in value]
    if type(value) is tuple:
        return tuple(_restore_order(item) for item in value)
    return value


def _attributes(obj: Any) -> dict[str, Any]:
    """Return ``obj``'s instance ``__dict__`` and every slot set in its class tree."""
    found = dict(vars(obj)) if hasattr(obj, "__dict__") else {}
    for klass in type(obj).__mro__:
        for name in getattr(klass, "__slots__", ()):
            if name != "__weakref__" and hasattr(obj, name):
                found[name] = getattr(obj, name)
    return found


class LabelledArraySerializer:
    """Store a ``LabelledArray`` (any subclass) or ``Dimensions`` with dict order kept.

    The payload is the class's import path and its attributes, as regression-records'
    own object path stores them, except that every dict among them is an
    :class:`OrderedItems`. Loading rebuilds the instance the same way, with
    ``object.__new__`` and no ``__init__``, and each dict in its recorded order.
    """

    type_tag = "labelled-array-records.labelled-array"

    def can_handle(self, obj: Any) -> bool:
        """Claim labelled arrays, of any subclass, and their dimensions."""
        return isinstance(obj, LabelledArray | Dimensions)

    def store(self, obj: Any, ctx: SerializationContext) -> dict:
        """Return the class's import path and its order-keeping attributes."""
        cls = type(obj)
        return {
            "module": cls.__module__,
            "qualname": cls.__qualname__,
            "attrs": to_json_tree(_keep_order(_attributes(obj)), ctx),
        }

    def load(self, blob: dict, ctx: DeserializationContext) -> Any:
        """Rebuild the instance, its dicts in their recorded order.

        Raises
        ------
        DeserializationError
            If the class no longer imports, or no longer takes its recorded attributes.
        """
        try:
            cls = _resolve(blob["module"], blob["qualname"])
        except (ImportError, AttributeError) as error:
            raise DeserializationError(
                f"recorded class {blob['module']}:{blob['qualname']} no longer "
                f"imports ({error})"
            ) from error
        obj = object.__new__(cls)
        for name, value in _restore_order(from_json_tree(blob["attrs"], ctx)).items():
            try:
                # object.__setattr__ writes a slot or the instance dict alike, past
                # any __setattr__ the class guards itself with.
                object.__setattr__(obj, name, value)
            except AttributeError as error:
                raise DeserializationError(
                    f"recorded attribute {name!r} no longer fits "
                    f"{blob['module']}:{blob['qualname']}; re-record"
                ) from error
        return obj


class ClassSerializer:
    """Store a class by its import path, as the argument of a classmethod's body."""

    type_tag = "labelled-array-records.class"

    def can_handle(self, obj: Any) -> bool:
        """Claim a class that imports back by its own name."""
        if not isinstance(obj, type) or "<" in obj.__qualname__:
            return False
        try:
            return _resolve(obj.__module__, obj.__qualname__) is obj
        except (ImportError, AttributeError):
            return False

    def store(self, obj: Any, ctx: SerializationContext) -> dict:
        """Return the class's import path."""
        return {"module": obj.__module__, "qualname": obj.__qualname__}

    def load(self, blob: dict, ctx: DeserializationContext) -> type:
        """Import the class the record names.

        Raises
        ------
        DeserializationError
            If that name no longer imports.
        """
        try:
            return _resolve(blob["module"], blob["qualname"])
        except (ImportError, AttributeError) as error:
            raise DeserializationError(
                f"recorded class {blob['module']}:{blob['qualname']} no longer "
                f"imports ({error})"
            ) from error


class SliceSerializer:
    """Store a ``slice`` as its start, stop and step."""

    type_tag = "labelled-array-records.slice"

    def can_handle(self, obj: Any) -> bool:
        """Claim a ``slice``."""
        return isinstance(obj, slice)

    def store(self, obj: Any, ctx: SerializationContext) -> dict:
        """Return the three bounds, each recorded as any other value is."""
        bounds = (obj.start, obj.stop, obj.step)
        return {"bounds": [to_json_tree(bound, ctx) for bound in bounds]}

    def load(self, blob: dict, ctx: DeserializationContext) -> slice:
        """Rebuild the slice from its bounds."""
        return slice(*(from_json_tree(bound, ctx) for bound in blob["bounds"]))


def _registry() -> SerializerRegistry:
    """Return regression-records' default registry, copied, plus the serializers."""
    registry = default_registry.copy()
    # Last, so every built-in serializer keeps what it claims.
    registry.register(LabelledArraySerializer())
    registry.register(SliceSerializer())
    registry.register(ClassSerializer())
    return registry


#: The serializers the harness's store records and replays with.
REGISTRY = _registry()
