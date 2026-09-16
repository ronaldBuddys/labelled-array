"""Shared machinery for the array backend and DataFrame adapter registries.

Both extension points are the same shape: a named plugin wrapping an optional
third-party library, and a registry that resolves one by name or by inspecting
an object. Keeping that shape in one place is what stops the two from drifting.
"""

from __future__ import annotations

import importlib
import importlib.util
from abc import ABC, abstractmethod
from functools import cache
from typing import Any, Generic, TypeVar

__all__ = ["Plugin", "Registry"]


@cache
def _is_installed(library: str) -> bool:
    """Whether ``library`` can be imported. Cached: installs do not change."""
    return importlib.util.find_spec(library) is not None


class Plugin(ABC):
    """A named adapter for one optional third-party library.

    Subclasses declare the library they wrap and the type they recognise; the
    import is deferred until something is actually done with it, so a library
    that is not installed costs nothing but a failed availability check.

    Attributes
    ----------
    name : str
        Registry key, e.g. ``"numpy"``.
    library : str
        Import name of the library wrapped.
    """

    name: str
    library: str

    @classmethod
    def is_available(cls) -> bool:
        """Whether the wrapped library can be imported."""
        return _is_installed(cls.library)

    @classmethod
    @abstractmethod
    def _instance_type(cls) -> type | tuple[type, ...]:
        """Return the type(s) of object this plugin handles."""

    @classmethod
    def handles(cls, obj: Any) -> bool:
        """Whether ``obj`` is an object of this plugin's library."""
        return cls.is_available() and isinstance(obj, cls._instance_type())

    @property
    def _module(self) -> Any:
        """The wrapped library, imported on first use."""
        return importlib.import_module(self.library)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Plugin) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


P = TypeVar("P", bound=Plugin)


class Registry(Generic[P]):
    """A name-to-plugin registry, resolvable by name or by object.

    Parameters
    ----------
    base : type
        The plugin base class entries must subclass.
    noun : str
        What an entry is called in error messages, e.g. ``"backend"``.
    error : type
        Exception raised for an unknown, uninstalled or unrecognised entry.
    default : str
        Name returned when a lookup is given None.
    """

    def __init__(
        self, base: type[P], noun: str, error: type[Exception], default: str
    ) -> None:
        self.base = base
        self.noun = noun
        self.error = error
        self.default = default
        self.entries: dict[str, P] = {}

    def register(self, plugin: P | type[P]) -> P:
        """Register ``plugin`` under its ``name``, replacing any previous entry."""
        instance = plugin() if isinstance(plugin, type) else plugin
        if not isinstance(instance, self.base):
            article = "an" if self.base.__name__[0] in "AEIOU" else "a"
            raise TypeError(
                f"expected {article} {self.base.__name__}, got {type(instance).__name__}"
            )
        self.entries[instance.name] = instance
        return instance

    def unregister(self, name: str) -> None:
        """Remove the entry registered under ``name``."""
        del self.entries[name]

    def names(self) -> tuple[str, ...]:
        """Names of every entry, whether or not its library is installed."""
        return tuple(self.entries)

    def available(self) -> tuple[str, ...]:
        """Names of entries whose library can be imported."""
        return tuple(
            name for name, entry in self.entries.items() if entry.is_available()
        )

    def get(self, key: str | P | None = None) -> P:
        """Return an entry by name, passing instances through.

        Raises
        ------
        error
            If the name is unknown, or its library is not installed.
        """
        if isinstance(key, self.base):
            return key
        name = self.default if key is None else key
        if name not in self.entries:
            raise self.error(
                f"unknown {self.noun} {name!r}, registered {self.noun}s: {list(self.entries)}"
            )
        entry = self.entries[name]
        if not entry.is_available():
            raise self.error(
                f"{self.noun} {name!r} is registered but its library is not installed"
            )
        return entry

    def resolve(self, obj: Any) -> P:
        """Return the entry that handles ``obj``.

        Raises
        ------
        error
            If no registered entry recognises ``obj``.
        """
        for entry in self.entries.values():
            if entry.is_available() and entry.handles(obj):
                return entry
        raise self.error(
            f"no registered {self.noun} handles {type(obj).__name__}, "
            f"available {self.noun}s: {list(self.available())}"
        )
