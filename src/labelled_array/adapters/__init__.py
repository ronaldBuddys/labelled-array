"""Registry of DataFrame adapters.

Adding support for another DataFrame library needs only a new
:class:`~labelled_array.adapters.base.DataFrameAdapter` subclass and a call to
:func:`register_adapter`.
"""

from __future__ import annotations

from labelled_array._plugin import Registry
from labelled_array.adapters.base import DataFrameAdapter
from labelled_array.adapters.pandas_adapter import PandasAdapter
from labelled_array.adapters.polars_adapter import PolarsAdapter
from labelled_array.exceptions import AdapterError

__all__ = [
    "DataFrameAdapter",
    "PandasAdapter",
    "PolarsAdapter",
    "DEFAULT_ADAPTERS",
    "available_adapters",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
    "resolve_adapter",
]

#: Adapters used when none is given, in order of preference: the first whose
#: library is installed wins, so pandas is the fallback for a polars-less install.
DEFAULT_ADAPTERS = ("polars", "pandas")

_REGISTRY: Registry[DataFrameAdapter] = Registry(
    base=DataFrameAdapter,
    noun="DataFrame engine",
    error=AdapterError,
    default=DEFAULT_ADAPTERS,
)

#: Register an adapter under its name, replacing any previous entry.
register_adapter = _REGISTRY.register
#: Names of every registered adapter, whether or not its library is installed.
registered_adapters = _REGISTRY.names
#: Names of registered adapters whose library can be imported.
available_adapters = _REGISTRY.available
#: Return an adapter by name, defaulting to polars; instances pass through.
get_adapter = _REGISTRY.get
#: Return the adapter owning a DataFrame.
resolve_adapter = _REGISTRY.resolve

for _adapter in (PandasAdapter, PolarsAdapter):
    register_adapter(_adapter)
del _adapter
