"""The functions and methods recorded, and how the recorders are put in place.

regression-records is a dev dependency only, so nothing in ``src/`` is decorated.
Instead the recorders are installed for the duration of a block: on the defining
module for a function, on a class for a method. They are needed while recording, so
the scenarios' calls are written to the store, and while replaying, because replay
only calls a name that still holds a recorder (it will not call whatever a recorded
name happens to resolve to now). A recorder wraps the real function, so its identity -
module and qualname - is the function's own, and replay calls the real function
beneath it.

Each recorder sits behind a small gate, :func:`_gated`, which records only the
outermost checkpoint call: ``LabelledArray.ffill`` or ``+`` may reach ``reindex`` and
``align`` inside, and the records are the scenarios' calls, not the library's calls to
itself.

``LabelledArray``'s own methods are recorded through a forwarding method of the same
name (:data:`FORWARDED`), for two reasons. ``select`` and ``isel`` take ``**kwargs``,
which a recorder binds as one argument named ``kwargs`` and replay passes back as a
dimension of that name; the forwarder takes the ``indexers`` mapping only, so the
scenarios pass that. And the four arithmetic dunders are defined on ``ArithmeticMixin``,
but replay rebuilds a method's receiver as the class the method is defined on,
bypassing ``__init__``: a bare mixin cannot add. The forwarder is named as a
``LabelledArray`` method, so the receiver is rebuilt as one. On exit the original
attribute is restored, or the forwarder removed where the class had none of its own.
"""

from __future__ import annotations

import copyreg
import importlib
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

from regression_records import (
    RecordStore,
    ToleranceConfig,
    record_func,
    record_method,
)
from tests.records.serializers import REGISTRY

from labelled_array import GroupBy, LabelledArray, Rolling

#: The environment variable naming the store's directory. Records are not committed:
#: CI records the scenarios on a PR's base commit into a temporary store and replays
#: them on the head, and a store worth keeping lives on a development machine.
STORE_ENV = "LA_RECORDS_STORE"

#: The playbook gate's own variable, honoured when ``$LA_RECORDS_STORE`` is not set.
GATE_STORE_ENV = "RECORDS_STORE"

#: Where records are read and written: ``$LA_RECORDS_STORE``, else ``$RECORDS_STORE``
#: (set by the agent gate), else the git-ignored ``tests/records/store``. Its
#: serializer keeps the order of a labelled array's dicts.
STORE = RecordStore(
    os.environ.get(STORE_ENV)
    or os.environ.get(GATE_STORE_ENV)
    or Path(__file__).parent / "store",
    serializer_registry=REGISTRY,
)

#: How closely a replay must match. Every checkpoint is exact indexing or float64
#: arithmetic (sums, means, division) on one machine, so the closed-form defaults
#: hold and no path needs a looser override.
TOLERANCE = ToleranceConfig(rtol=1e-10, atol=1e-12)

#: ``(module, function, layer)`` per function checkpoint. These are the module-level
#: implementations the ``LabelledArray`` methods delegate to; they are reached through
#: the module (``_align.reindex(...)``), so a patched module attribute is hit.
#: ``layer`` is stored as a tag on every record.
CHECKPOINTS: tuple[tuple[str, str, str], ...] = (
    ("labelled_array._convert", "from_dataframe", "convert"),
    ("labelled_array._convert", "to_dataframe", "convert"),
    ("labelled_array._align", "reindex", "align"),
    ("labelled_array._align", "align", "align"),
    ("labelled_array._align", "concat", "align"),
    ("labelled_array._fill", "ffill", "fill"),
    ("labelled_array._ops", "groupby", "ops"),
    ("labelled_array._ops", "rolling", "ops"),
)

#: ``(class, method, layer)`` per method checkpoint, each defined on its own class.
#: Each record holds the receiver's state before and after the call.
METHOD_CHECKPOINTS: tuple[tuple[type, str, str], ...] = (
    (GroupBy, "reduce", "ops"),
    (Rolling, "reduce", "ops"),
)

#: ``(method, layer)`` per ``LabelledArray`` method recorded through a forwarder.
FORWARDED: tuple[tuple[str, str], ...] = (
    ("select", "index"),
    ("isel", "index"),
    ("__add__", "arithmetic"),
    ("__sub__", "arithmetic"),
    ("__mul__", "arithmetic"),
    ("__truediv__", "arithmetic"),
)

#: How many checkpoint calls are in progress, so only the outermost one records.
_depth = 0


def _gated(
    recorder: Callable[..., Any],
    original: Callable[..., Any],
    accepts: Callable[[Any], bool] | None = None,
) -> Callable[..., Any]:
    """Return ``original`` wrapped to go through ``recorder`` only when it should.

    Parameters
    ----------
    recorder : callable
        The regression-records recorder around ``original``.
    original : callable
        The real function or method.
    accepts : callable, optional
        For a method, a test of the receiver: only a receiver it accepts is recorded.

    Returns
    -------
    callable
        A plain function, so a class attribute still binds as a method. Its
        ``__wrapped__`` is the recorder, which is how replay finds the function
        beneath: regression-records follows ``__wrapped__`` to the recorder's marker.
    """

    def checkpoint(*args: Any, **kwargs: Any) -> Any:
        global _depth
        outermost = _depth == 0
        _depth += 1
        try:
            if outermost and (accepts is None or accepts(args[0])):
                return recorder(*args, **kwargs)
            return original(*args, **kwargs)
        finally:
            _depth -= 1

    # Not functools.wraps: it copies the recorder's attributes too, its marker
    # among them, and replay would then take the recorder for the real function.
    for name in ("__module__", "__name__", "__qualname__", "__doc__"):
        setattr(checkpoint, name, getattr(original, name))
    checkpoint.__wrapped__ = recorder
    return checkpoint


def _forwarders() -> dict[str, Callable[..., Any]]:
    """Return the forwarding methods of :data:`FORWARDED`, named as ``LabelledArray``'s.

    Each calls the method ``LabelledArray`` has today (inherited or its own), takes
    the same arguments but no ``**kwargs``, and carries the ``LabelledArray`` module
    and qualname, so a record of it replays on a ``LabelledArray`` receiver.
    """
    method = {name: getattr(LabelledArray, name) for name, _ in FORWARDED}

    def select(self, indexers=None, drop=False, missing="raise"):
        return method["select"](self, indexers, drop=drop, missing=missing)

    def isel(self, indexers=None, drop=False):
        return method["isel"](self, indexers, drop=drop)

    def binary(name: str) -> Callable[..., Any]:
        def forward(self, other):
            return method[name](self, other)

        return forward

    forwarders = {"select": select, "isel": isel}
    for name, _ in FORWARDED:
        forwarders.setdefault(name, binary(name))
        forward = forwarders[name]
        forward.__module__ = LabelledArray.__module__
        forward.__name__ = name
        forward.__qualname__ = f"{LabelledArray.__qualname__}.{name}"
        forward.__doc__ = method[name].__doc__
    return forwarders


def _copy_mapping_proxy(proxy: MappingProxyType) -> tuple[type, tuple[dict]]:
    """Reduce a read-only mapping to a copy of the dict it wraps, for ``copy``."""
    return MappingProxyType, (dict(proxy),)


@contextmanager
def _copyable_mapping_proxies() -> Iterator[None]:
    """Let ``copy.deepcopy`` copy a ``MappingProxyType`` for the duration of a block.

    ``record_method`` deep-copies the receiver's state before and after each call, and
    a ``LabelledArray`` keeps its coordinates in a read-only mapping, which
    ``deepcopy`` refuses. regression-records serializes such a mapping itself; only
    the copy needs teaching, and only while the recorders are in place.
    """
    previous = copyreg.dispatch_table.get(MappingProxyType)
    copyreg.pickle(MappingProxyType, _copy_mapping_proxy)
    try:
        yield
    finally:
        if previous is None:
            copyreg.dispatch_table.pop(MappingProxyType, None)
        else:
            copyreg.dispatch_table[MappingProxyType] = previous


@contextmanager
def installed(*, enabled: bool) -> Iterator[None]:
    """Put a recorder on every checkpoint, restoring the real ones on exit.

    A function's recorder replaces the attribute of its defining module, which
    ``core`` looks up at call time. A method's recorder replaces the attribute of its
    class. The arithmetic dunders are added to ``LabelledArray`` and deleted again.

    Parameters
    ----------
    enabled : bool
        Whether the recorders write (still only when recording is switched on). False
        installs pass-throughs, which is all replay needs.
    """
    originals: list[tuple[Any, str, Any]] = []
    added: list[tuple[type, str]] = []
    with _copyable_mapping_proxies():
        try:
            for module_name, name, layer in CHECKPOINTS:
                module = importlib.import_module(module_name)
                function = getattr(module, name)
                originals.append((module, name, function))
                recorder = record_func(STORE, enabled=enabled, tags={"layer": layer})
                setattr(module, name, _gated(recorder(function), function))
            for owner, name, layer in METHOD_CHECKPOINTS:
                # From the class's own namespace: the plain function.
                method = vars(owner)[name]
                originals.append((owner, name, method))
                recorder = record_method(STORE, enabled=enabled, tags={"layer": layer})
                accepts = lambda receiver, owner=owner: type(receiver) is owner  # noqa: E731
                setattr(owner, name, _gated(recorder(method), method, accepts))
            forwarders = _forwarders()
            layers = dict(FORWARDED)
            for name, forward in forwarders.items():
                if name in vars(LabelledArray):
                    originals.append((LabelledArray, name, vars(LabelledArray)[name]))
                else:
                    added.append((LabelledArray, name))
                recorder = record_method(
                    STORE, enabled=enabled, tags={"layer": layers[name]}
                )
                setattr(
                    LabelledArray,
                    name,
                    _gated(
                        recorder(forward),
                        forward,
                        lambda receiver: isinstance(receiver, LabelledArray),
                    ),
                )
            yield
        finally:
            for owner, name in added:
                delattr(owner, name)
            for target, name, original in reversed(originals):
                setattr(target, name, original)
