"""Replay every record in the regression-records store against the current code.

A failure means a checkpoint's behavior changed. The store is not committed, so with
no store the replay is skipped: CI replays a store recorded on the PR's base commit.
The serializer tests need no store. See ``tests/records/README.md``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from regression_records import replay_tests
from regression_records.serialization import (
    DeserializationContext,
    SerializationContext,
    SerializerRegistry,
    default_registry,
    from_json_tree,
    to_json_tree,
)
from tests.records.checkpoints import STORE, STORE_ENV, TOLERANCE, installed
from tests.records.serializers import REGISTRY

from labelled_array import Dimensions, LabelledArray


def through_json(value: Any, registry: SerializerRegistry, tmp_path) -> Any:
    """Store ``value`` as a record does (JSON with sorted keys), then load it back."""
    tree = to_json_tree(
        value, SerializationContext(data_dir=tmp_path, registry=registry)
    )
    text = json.dumps(tree, sort_keys=True)
    context = DeserializationContext(base_dir=tmp_path, registry=registry)
    return from_json_tree(json.loads(text), context)


def reversed_axes() -> LabelledArray:
    """Return an array whose axes ``(b, a)`` and variables ``(w, v)`` are unsorted."""
    data = {"w": np.arange(6.0).reshape(3, 2), "v": np.ones((3, 2))}
    return LabelledArray(data, dims={"b": [30, 10, 20], "a": ["y", "x"]})


def test_dimensions_keep_axis_order_through_a_record(tmp_path):
    dims = reversed_axes().dims
    back = through_json(dims, REGISTRY, tmp_path)
    assert isinstance(back, Dimensions)
    assert list(back.names) == ["b", "a"]
    assert [name for name, _ in back.items()] == ["b", "a"]
    assert back == dims


def test_default_registry_loses_the_variable_order(tmp_path):
    # The reason the serializer exists: without it dicts come back sorted.
    back = through_json(reversed_axes(), default_registry, tmp_path)
    assert back.variables == ("v", "w")


def test_labelled_array_survives_a_record(tmp_path):
    array = reversed_axes()
    back = through_json(array, REGISTRY, tmp_path)
    assert isinstance(back, LabelledArray)
    assert list(back.dims.names) == ["b", "a"]
    assert back.dims == array.dims
    assert back.variables == ("w", "v")
    np.testing.assert_array_equal(back["w"], array["w"])


def test_object_labels_survive_a_record(tmp_path):
    # polars returns string labels as dtype=object, which a record cannot hold as is.
    labels = np.array(["x", "y"], dtype=object)
    array = LabelledArray({"w": np.arange(2.0)}, dims={"k": labels})
    back = through_json(array, REGISTRY, tmp_path)
    assert back.dims["k"].dtype == object
    assert back.dims["k"].tolist() == ["x", "y"]


def test_slice_and_class_survive_a_record(tmp_path):
    for value in (
        slice(1, None, 2),
        slice(np.datetime64("2024-01-02"), None),
        LabelledArray,
    ):
        assert through_json(value, REGISTRY, tmp_path) == value


# A zero divisor is a recorded edge case; numpy's warning is not a failure of replay.
pytestmark = pytest.mark.filterwarnings(
    "ignore:divide by zero:RuntimeWarning",
    "ignore:invalid value:RuntimeWarning",
)

if STORE.iter_record_paths():

    @pytest.fixture(autouse=True, scope="module")
    def recorders():
        """Put pass-through recorders on the checkpoints: replay calls nothing else."""
        with installed(enabled=False):
            yield

    test_checkpoint = replay_tests(STORE, tolerance=TOLERANCE)

else:

    @pytest.mark.skip(
        reason=f"no regression records in {STORE.root}; record some with "
        f"`{STORE_ENV}=<dir> uv run python -m tests.records.cli record` "
        "(tests/records/README.md)"
    )
    def test_checkpoint():
        """Stand in for the replay tests while there is no store."""
