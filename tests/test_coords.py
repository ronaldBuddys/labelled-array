"""Coordinates given at a broadcastable shape, and stored once."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from labelled_array import DimensionError, LabelledArray

SHAPE = (4, 5)
TAU = np.array([0.25, 0.5, 1.0, 2.0])


@pytest.fixture
def chain():
    """A 4 x 5 array whose 'tau' coordinate varies along the first axis only."""
    return LabelledArray(
        {"price": np.arange(20.0).reshape(SHAPE)},
        {"expiry": list("pqrs"), "strike": [90, 95, 100, 105, 110]},
        coords={"tau": TAU[:, None]},
    )


# -- what broadcasts ------------------------------------------------------------


def test_a_column_coordinate_broadcasts_across_the_trailing_axis(chain):
    assert chain.coords["tau"].shape == SHAPE
    np.testing.assert_array_equal(
        chain.coords["tau"], np.broadcast_to(TAU[:, None], SHAPE)
    )


def test_a_row_coordinate_broadcasts_across_the_leading_axis():
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    array = LabelledArray(
        {"price": np.zeros(SHAPE)},
        {"expiry": list("pqrs"), "strike": [90, 95, 100, 105, 110]},
        coords={"k": strikes},
    )
    assert array.coords["k"].shape == SHAPE
    np.testing.assert_array_equal(array.coords["k"], np.broadcast_to(strikes, SHAPE))


def test_a_scalar_coordinate_broadcasts_to_every_cell():
    array = LabelledArray(
        {"price": np.zeros(SHAPE)},
        {"expiry": list("pqrs"), "strike": [90, 95, 100, 105, 110]},
        coords={"currency": "USD"},
    )
    assert array.coords["currency"].shape == SHAPE
    assert set(array.coords["currency"].ravel().tolist()) == {"USD"}


def test_a_full_shape_coordinate_is_still_accepted():
    values = np.arange(20.0).reshape(SHAPE)
    array = LabelledArray(
        {"price": np.zeros(SHAPE)},
        {"expiry": list("pqrs"), "strike": [90, 95, 100, 105, 110]},
        coords={"cell": values},
    )
    np.testing.assert_array_equal(array.coords["cell"], values)


def test_a_coordinate_that_does_not_broadcast_raises():
    with pytest.raises(DimensionError) as raised:
        LabelledArray(
            {"price": np.zeros(SHAPE)},
            {"expiry": list("pqrs"), "strike": [90, 95, 100, 105, 110]},
            coords={"tau": np.zeros(3)},
        )
    message = str(raised.value)
    assert "'tau'" in message
    assert "(3,)" in message
    assert str(SHAPE) in message


# -- stored once, and stored read-only ------------------------------------------


def test_a_broadcast_coordinate_is_held_once(chain):
    assert chain.coords["tau"].base.nbytes == TAU.nbytes
    assert chain.coords["tau"].nbytes > TAU.nbytes


def test_a_stored_coordinate_is_read_only(chain):
    assert chain.coords["tau"].flags.writeable is False
    with pytest.raises(ValueError):
        chain.coords["tau"][0, 0] = 99.0


def test_freezing_a_coordinate_does_not_freeze_the_callers_array():
    values = np.arange(20.0).reshape(SHAPE)
    LabelledArray(
        {"price": np.zeros(SHAPE)},
        {"expiry": list("pqrs"), "strike": [90, 95, 100, 105, 110]},
        coords={"cell": values},
    )
    assert values.flags.writeable is True, "the caller keeps its own array writeable"


# -- surviving every operation --------------------------------------------------


def _tau_at_every_cell(array):
    """Return the 'tau' coordinate as a plain writeable array, for comparison."""
    return np.asarray(array.coords["tau"])


def test_a_broadcast_coordinate_survives_select(chain):
    picked = chain.select({"strike": [95, 105]})
    np.testing.assert_array_equal(
        _tau_at_every_cell(picked), np.broadcast_to(TAU[:, None], (4, 2))
    )


def test_a_broadcast_coordinate_survives_isel(chain):
    picked = chain.isel({"expiry": [2, 0]})
    np.testing.assert_array_equal(
        _tau_at_every_cell(picked), np.broadcast_to(TAU[[2, 0]][:, None], (2, 5))
    )


def test_a_broadcast_coordinate_survives_transpose(chain):
    flipped = chain.transpose("strike", "expiry")
    np.testing.assert_array_equal(
        _tau_at_every_cell(flipped), np.broadcast_to(TAU, (5, 4))
    )


def test_a_broadcast_coordinate_survives_reindex(chain):
    moved = chain.reindex(expiry=["r", "p"])
    np.testing.assert_array_equal(
        _tau_at_every_cell(moved), np.broadcast_to(TAU[[2, 0]][:, None], (2, 5))
    )


def test_a_broadcast_coordinate_survives_a_deep_copy(chain):
    np.testing.assert_array_equal(
        _tau_at_every_cell(chain.copy(deep=True)), _tau_at_every_cell(chain)
    )


def test_a_broadcast_coordinate_survives_a_pickle_round_trip(chain):
    revived = pickle.loads(pickle.dumps(chain))
    np.testing.assert_array_equal(
        _tau_at_every_cell(revived), _tau_at_every_cell(chain)
    )
    assert revived.equals(chain)


def test_a_broadcast_coordinate_survives_to_dataframe(chain, engine):
    frame = chain.to_dataframe(engine=engine, index=False)
    assert sorted(np.asarray(frame["tau"]).tolist()) == sorted(
        _tau_at_every_cell(chain).ravel().tolist()
    )


def test_a_broadcast_coordinate_can_be_grouped_by(chain):
    grouped = chain.groupby("tau").sum()
    assert grouped.dims["expiry"].tolist() == TAU.tolist()
