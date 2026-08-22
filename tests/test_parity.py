"""pandas and polars must produce the same labelled array from the same data."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import build_both as _build_both, build_frame

from labelled_array import LabelledArray

DIMS = ["region", "product"]

DATASETS = {
    "sparse": {
        "region": ["north", "north", "south", "north"],
        "product": ["AAA", "BBB", "AAA", "CCC"],
        "units": [10, 20, 30, 40],
        "price": [1.5, 2.5, 3.5, 4.5],
    },
    "dense": {
        "region": ["north", "north", "south", "south"],
        "product": ["AAA", "BBB", "AAA", "BBB"],
        "units": [1, 2, 3, 4],
        "price": [1.5, 2.5, 3.5, 4.5],
    },
    "single_row": {"region": ["north"], "product": ["AAA"], "units": [1], "price": [1.5]},
    "numeric_labels": {
        "region": [2, 1, 1],
        "product": [20, 10, 20],
        "units": [1, 2, 3],
        "price": [1.5, 2.5, 3.5],
    },
    "boolean_labels": {
        "region": [True, False, True],
        "product": ["AAA", "AAA", "BBB"],
        "units": [1, 2, 3],
        "price": [1.5, 2.5, 3.5],
    },
}


@pytest.fixture(params=sorted(DATASETS))
def dataset(request):
    return DATASETS[request.param]


def build_both(dataset, **kwargs):
    """Return the labelled array built from each engine's frame."""
    return _build_both(dataset, DIMS, **kwargs)


def test_engines_agree(dataset):
    from_pandas, from_polars = build_both(dataset)
    assert from_pandas.equals(from_polars)


def test_engines_agree_on_shape_and_labels(dataset):
    from_pandas, from_polars = build_both(dataset)
    assert from_pandas.shape == from_polars.shape
    assert from_pandas.dims.names == from_polars.dims.names
    for name in from_pandas.dims.names:
        assert from_pandas.dims[name].tolist() == from_polars.dims[name].tolist()


def test_engines_agree_on_the_mask(dataset):
    from_pandas, from_polars = build_both(dataset)
    if from_pandas.mask is None:
        assert from_polars.mask is None
    else:
        assert from_pandas.mask.tolist() == from_polars.mask.tolist()


def test_engines_agree_on_variable_dtypes(dataset):
    from_pandas, from_polars = build_both(dataset)
    assert from_pandas.dtypes == from_polars.dtypes


def test_engines_agree_with_options(dataset):
    from_pandas, from_polars = build_both(dataset, fill_value=-1, dtype="float64", mask=False)
    assert from_pandas.equals(from_polars)


def test_one_array_writes_the_same_content_to_either_engine(dataset):
    la = build_both(dataset)[0]
    from_pandas = la.to_dataframe("pandas", index=False)
    from_polars = la.to_dataframe("polars")
    assert from_polars.columns == list(from_pandas.columns)
    for name in from_polars.columns:
        assert from_polars[name].to_list() == from_pandas[name].tolist()


def test_string_labels_have_the_same_dtype_from_both_engines():
    from_pandas, from_polars = build_both(DATASETS["sparse"])
    assert from_pandas.dims["region"].dtype == from_polars.dims["region"].dtype
    assert from_pandas.dims["region"].dtype == np.dtype("O")


def test_cross_engine_round_trip():
    """A frame written by one engine, read back by the other, is unchanged."""
    la = LabelledArray.from_dataframe(
        build_frame("pandas", DATASETS["sparse"]), dims=DIMS
    )
    via_polars = LabelledArray.from_dataframe(la.to_dataframe("polars"), dims=DIMS)
    assert la.equals(via_polars)
