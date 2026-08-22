"""Registry of array backends.

Backends are registered by name and resolved either by name or by inspecting an
array. Adding support for a new array library needs only a new
:class:`~labelled_array.backends.base.ArrayBackend` subclass and a call to
:func:`register_backend`; nothing in the core needs to change.
"""

from __future__ import annotations

from typing import Any

from labelled_array._plugin import Registry
from labelled_array.backends.base import ArrayBackend
from labelled_array.backends.jax_backend import JaxBackend
from labelled_array.backends.numpy_backend import NumpyBackend
from labelled_array.backends.torch_backend import TorchBackend
from labelled_array.exceptions import BackendError

__all__ = [
    "ArrayBackend",
    "JaxBackend",
    "NumpyBackend",
    "TorchBackend",
    "DEFAULT_BACKEND",
    "available_backends",
    "get_backend",
    "register_backend",
    "registered_backends",
    "resolve_backend",
]

#: Name of the backend used when none is given.
DEFAULT_BACKEND = "numpy"

_REGISTRY: Registry[ArrayBackend] = Registry(
    base=ArrayBackend, noun="backend", error=BackendError, default=DEFAULT_BACKEND
)

#: Register a backend under its name, replacing any previous entry.
register_backend = _REGISTRY.register
#: Names of every registered backend, whether or not its library is installed.
registered_backends = _REGISTRY.names
#: Names of registered backends whose library can be imported.
available_backends = _REGISTRY.available
#: Return a backend by name, defaulting to numpy; instances pass through.
get_backend = _REGISTRY.get
#: Return the backend owning an array.
resolve_backend = _REGISTRY.resolve


def _resolve_or_none(obj: Any) -> ArrayBackend | None:
    """Return the backend owning ``obj``, or None if none recognises it."""
    for entry in _REGISTRY.entries.values():
        if entry.is_available() and entry.handles(obj):
            return entry
    return None


for _backend in (NumpyBackend, JaxBackend, TorchBackend):
    register_backend(_backend)
del _backend
