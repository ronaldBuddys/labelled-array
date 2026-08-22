"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest

from labelled_array import LabelledArray
from labelled_array.adapters import available_adapters
from labelled_array.backends import available_backends, get_backend


@pytest.fixture(params=available_backends())
def backend(request):
    """Each array backend whose library is installed."""
    return get_backend(request.param)


@pytest.fixture(params=available_adapters())
def engine(request):
    """Each DataFrame engine whose library is installed."""
    return request.param


@pytest.fixture
def labels_2d():
    """A simple two-dimensional label mapping."""
    return {"a": np.array([1, 2]), "b": np.array(["x", "y", "z"])}


@pytest.fixture
def simple():
    """A two-variable, two-dimensional array with one unpopulated cell."""
    return LabelledArray(
        data={
            "w": np.arange(6).reshape(2, 3),
            "x": np.arange(6, dtype=np.float64).reshape(2, 3) / 2,
        },
        dims={"a": [1, 2], "b": ["x", "y", "z"]},
        mask=np.array([[True, True, False], [True, True, True]]),
        attrs={"source": "fixture"},
    )


@pytest.fixture
def sales_data():
    """A sparse frame: 'BBB' has no 'south' row, so cells are unpopulated."""
    return {
        "region": ["north", "north", "south", "north"],
        "product": ["AAA", "BBB", "AAA", "CCC"],
        "units": [10, 20, 30, 40],
        "price": [1.5, 2.5, 3.5, 4.5],
    }
