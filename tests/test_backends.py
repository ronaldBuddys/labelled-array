"""Tests for the array backends and their registry."""

from __future__ import annotations

import numpy as np
import pytest

from labelled_array import BackendDtypeError, BackendError
from labelled_array.backends import (
    ArrayBackend,
    DEFAULT_BACKEND,
    NumpyBackend,
    available_backends,
    get_backend,
    register_backend,
    registered_backends,
    resolve_backend,
)


@pytest.fixture
def source():
    return np.arange(6, dtype=np.float64).reshape(2, 3)


# -- registry -------------------------------------------------------------------


def test_numpy_is_registered_and_default():
    assert "numpy" in registered_backends()
    assert "numpy" in available_backends()
    assert get_backend().name == DEFAULT_BACKEND == "numpy"


def test_every_registered_backend_is_installed_in_this_environment():
    """The dev environment installs jax and torch, so all three must resolve."""
    assert set(registered_backends()) == {"numpy", "jax", "torch"}
    assert set(available_backends()) == {"numpy", "jax", "torch"}


def test_get_backend_passes_through_instances():
    backend = get_backend("numpy")
    assert get_backend(backend) is backend


def test_get_unknown_backend_raises():
    with pytest.raises(BackendError, match="unknown backend 'nope'"):
        get_backend("nope")


def test_register_rejects_non_backend():
    with pytest.raises(TypeError, match="expected an ArrayBackend"):
        register_backend(object())


def test_register_accepts_class_and_instance():
    assert register_backend(NumpyBackend).name == "numpy"
    assert register_backend(NumpyBackend()).name == "numpy"


def test_registering_a_custom_backend_needs_no_core_change():
    class DummyBackend(NumpyBackend):
        name = "dummy"

    try:
        register_backend(DummyBackend)
        assert "dummy" in registered_backends()
        assert get_backend("dummy").name == "dummy"
    finally:
        from labelled_array.backends import _REGISTRY

        _REGISTRY.unregister("dummy")


def test_resolve_backend_identifies_arrays(backend, source):
    assert resolve_backend(backend.asarray(source)).name == backend.name


def test_resolve_unknown_object_raises():
    with pytest.raises(BackendError, match="no registered backend handles"):
        resolve_backend("not an array")


def test_backend_equality_and_hash():
    assert get_backend("numpy") == NumpyBackend()
    assert get_backend("numpy") != get_backend("jax")
    assert len({get_backend("numpy"), NumpyBackend()}) == 1
    assert repr(NumpyBackend()) == "NumpyBackend()"


def test_backend_is_abstract():
    with pytest.raises(TypeError):
        ArrayBackend()


# -- conversion and inspection --------------------------------------------------


def test_asarray_round_trips_through_numpy(backend, source):
    np.testing.assert_array_equal(backend.to_numpy(backend.asarray(source)), source)


def test_asarray_passes_through_own_arrays(backend, source):
    arr = backend.asarray(source)
    assert backend.asarray(arr) is arr or backend.array_equal(backend.asarray(arr), arr)


def test_shape_and_dtype(backend, source):
    """float32 is used because JAX narrows 64-bit dtypes (tested separately)."""
    arr = backend.asarray(source.astype(np.float32))
    assert backend.shape(arr) == (2, 3)
    assert backend.dtype(arr) == np.dtype("float32")


def test_astype(backend, source):
    arr = backend.astype(backend.asarray(source), np.dtype("float32"))
    assert backend.dtype(arr) == np.dtype("float32")


def test_copy_is_independent(backend, source):
    arr = backend.asarray(source)
    copied = backend.copy(arr)
    np.testing.assert_array_equal(backend.to_numpy(copied), source)
    source[0, 0] = 99
    assert backend.to_numpy(copied)[0, 0] == 0


# -- shape manipulation ---------------------------------------------------------


def test_take(backend, source):
    arr = backend.take(backend.asarray(source), np.array([2, 0]), axis=1)
    np.testing.assert_array_equal(backend.to_numpy(arr), source[:, [2, 0]])


def test_take_empty(backend, source):
    arr = backend.take(backend.asarray(source), np.array([], dtype=np.intp), axis=0)
    assert backend.shape(arr) == (0, 3)


def test_transpose(backend, source):
    arr = backend.transpose(backend.asarray(source), (1, 0))
    np.testing.assert_array_equal(backend.to_numpy(arr), source.T)


def test_reshape(backend, source):
    arr = backend.reshape(backend.asarray(source), (3, 2))
    np.testing.assert_array_equal(backend.to_numpy(arr), source.reshape(3, 2))


# -- dtype support --------------------------------------------------------------


def test_numpy_supports_every_dtype():
    numpy_backend = get_backend("numpy")
    for dtype in ["<U3", "O", "datetime64[ns]", "float16", "complex128", "uint64"]:
        assert numpy_backend.supports_dtype(dtype)


@pytest.mark.parametrize("name", ["jax", "torch"])
@pytest.mark.parametrize("dtype", ["<U3", "O", "datetime64[ns]", "timedelta64[D]"])
def test_array_backends_reject_non_numeric_dtypes(name, dtype):
    backend = get_backend(name)
    assert not backend.supports_dtype(dtype)
    with pytest.raises(BackendDtypeError, match="cannot hold dtype"):
        backend.asarray(np.zeros(2, dtype=dtype))


@pytest.mark.parametrize("name", ["jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "bool"])
def test_array_backends_accept_numeric_dtypes(name, dtype):
    assert get_backend(name).supports_dtype(dtype)


def test_check_dtype_names_the_variable():
    with pytest.raises(BackendDtypeError, match="for 'ticker'"):
        get_backend("torch").check_dtype("<U3", context="ticker")


def test_jax_arrays_are_immutable():
    """The reason scatter-style updates are never done in place by the core."""
    arr = get_backend("jax").asarray(np.zeros(3))
    with pytest.raises(TypeError):
        arr[0] = 1.0


def test_jax_narrows_64_bit_by_default():
    """Documented gotcha: JAX allocates 32-bit unless jax_enable_x64 is set."""
    import jax

    backend = get_backend("jax")
    arr = backend.asarray(np.arange(3, dtype=np.int64))
    expected = np.dtype("int64") if jax.config.read("jax_enable_x64") else np.dtype("int32")
    assert backend.dtype(arr) == expected


def test_torch_reports_numpy_dtypes():
    import torch

    backend = get_backend("torch")
    arr = backend.asarray(np.zeros(2, dtype=np.float32))
    assert isinstance(arr, torch.Tensor)
    assert backend.dtype(arr) == np.dtype("float32")


def test_torch_accepts_read_only_source():
    """Read-only numpy arrays (as dimension labels are) must still convert."""
    source = np.arange(3, dtype=np.float64)
    source.flags.writeable = False
    assert get_backend("torch").shape(get_backend("torch").asarray(source)) == (3,)
