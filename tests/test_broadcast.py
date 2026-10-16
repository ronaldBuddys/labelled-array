"""Variables and a mask given at a broadcastable shape with ``broadcast=True``."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from labelled_array import DimensionError, LabelledArray

DIMS = {"date": ["d0", "d1"], "expiry": list("pqr"), "strike": [90, 95, 100, 105]}
SHAPE = (2, 3, 4)
VOL = np.arange(24.0).reshape(SHAPE) / 100
#: Time to expiry varies by date and expiry, not by strike: ``(date, expiry, 1)``.
TAU = np.array([[0.1, 0.2, 0.5], [0.09, 0.19, 0.49]])[:, :, None]


def surface(backend="numpy", **kwargs):
    """A vol surface with time to expiry stored once per (date, expiry)."""
    return LabelledArray(
        {"vol": VOL, "tau": TAU},
        DIMS,
        backend=backend,
        broadcast=True,
        **kwargs,
    )


# -- what is accepted -----------------------------------------------------------


@pytest.mark.parametrize(
    "compact",
    [TAU, np.arange(8.0).reshape(2, 1, 4), np.ones((1, 1, 1))],
    ids=["(l,k,1)", "(l,1,m)", "(1,1,1)"],
)
def test_a_length_one_axis_is_broadcast_to_the_full_shape(backend, compact):
    array = LabelledArray(
        {"vol": VOL, "x": compact}, DIMS, backend=backend, broadcast=True
    )
    stored = array["x"]
    assert backend.shape(stored) == SHAPE
    # rtol: JAX holds float32 unless x64 is enabled.
    np.testing.assert_allclose(
        backend.to_numpy(stored), np.broadcast_to(compact, SHAPE), rtol=1e-6
    )


def test_a_numpy_broadcast_variable_is_held_once():
    stored = surface()["tau"]
    assert stored.strides[2] == 0
    assert stored.base.nbytes == TAU.nbytes


def lazy_surface():
    """A surface whose variable, coordinate and mask all vary along strike not at all."""
    present = np.array([[True, True, False], [True, False, True]])[:, :, None]
    return surface(mask=present, coords={"fwd": TAU * 2})


#: Each slicing call with the equivalent plain numpy index on the strike axis.
SLICINGS = {
    "isel list": (lambda a: a.isel({"strike": [1]}), np.array([1])),
    "select list": (lambda a: a.select({"strike": [95]}), np.array([1])),
    "isel slice": (lambda a: a.isel({"strike": slice(0, 3)}), np.array([0, 1, 2])),
    "isel reordered": (lambda a: a.isel({"strike": [3, 1, 2]}), np.array([3, 1, 2])),
}


@pytest.mark.parametrize("slicing", SLICINGS)
def test_isel_keeps_a_broadcast_axis_zero_stride(slicing):
    call, strike = SLICINGS[slicing]
    array = lazy_surface()
    sliced = call(array)
    cells = {"tau": sliced["tau"], "fwd": sliced.coords["fwd"], "mask": sliced.mask}
    for name, cell in cells.items():
        assert cell.strides[2] == 0, name
        assert cell.shape == (2, 3, len(strike)), name
        assert not cell.flags.writeable, name
    # Same values as the dense path: slice a materialised copy the old way.
    dense = {
        "tau": np.array(array["tau"]),
        "fwd": np.array(array.coords["fwd"]),
        "mask": np.array(array.mask),
    }
    for name, cell in cells.items():
        np.testing.assert_array_equal(cell, dense[name][:, :, strike], err_msg=name)


def test_slicing_a_non_broadcast_axis_keeps_the_broadcast_one():
    sliced = lazy_surface().isel({"expiry": [2, 0], "date": 1}, drop=True)
    assert sliced["tau"].strides[-1] == 0
    np.testing.assert_array_equal(
        sliced["tau"], np.broadcast_to(TAU[1, [2, 0]], (2, 4))
    )


def test_a_torch_broadcast_variable_is_held_once():
    torch = pytest.importorskip("torch")
    stored = surface(backend="torch")["tau"]
    assert isinstance(stored, torch.Tensor)
    assert stored.stride()[2] == 0


def test_a_numpy_broadcast_variable_refuses_a_write():
    # One write through a zero stride would change every strike at once.
    with pytest.raises(ValueError, match="read-only"):
        surface()["tau"][0, 0, 0] = 1.0


def test_a_full_shape_variable_is_stored_as_given():
    array = surface()
    assert array["vol"] is VOL or np.shares_memory(array["vol"], VOL)
    assert array["vol"].flags.writeable


def test_from_array_passes_the_flag_through():
    array = LabelledArray.from_array(TAU, DIMS, name="tau", broadcast=True)
    assert array.shape == SHAPE


# -- what is refused ------------------------------------------------------------


def test_without_the_flag_a_compact_variable_still_raises():
    with pytest.raises(DimensionError, match="'tau' has shape"):
        LabelledArray({"vol": VOL, "tau": TAU}, DIMS)


def test_a_shape_that_does_not_broadcast_raises():
    with pytest.raises(DimensionError, match=r"\(2, 2, 1\)"):
        LabelledArray({"x": np.zeros((2, 2, 1))}, DIMS, broadcast=True)


def test_fewer_axes_than_dimensions_raises():
    # numpy would line (4,) up with the strike axis; the flag makes the caller say so.
    with pytest.raises(DimensionError, match="needs 3 axes"):
        LabelledArray({"x": np.zeros(4)}, DIMS, broadcast=True)


def test_equal_sizes_cannot_bind_to_the_wrong_axis():
    square = {"date": [0, 1, 2], "strike": [90, 95, 100]}
    with pytest.raises(DimensionError, match="needs 2 axes"):
        LabelledArray({"x": np.zeros(3)}, square, broadcast=True)


def test_dims_are_required():
    with pytest.raises(DimensionError, match="dims must be given"):
        LabelledArray({"tau": TAU}, broadcast=True)


def test_assign_does_not_broadcast():
    with pytest.raises(DimensionError):
        surface().assign(other=TAU)


def test_assign_accepts_an_explicit_broadcast():
    array = surface().assign(other=np.broadcast_to(TAU, SHAPE))
    np.testing.assert_array_equal(array["other"], np.broadcast_to(TAU, SHAPE))


# -- the mask -------------------------------------------------------------------


def test_a_mask_is_broadcast():
    present = np.array([[True, True, False], [True, False, True]])[:, :, None]
    array = surface(mask=present)
    assert array.mask.shape == SHAPE
    np.testing.assert_array_equal(array.mask, np.broadcast_to(present, SHAPE))
    assert not array.mask.flags.writeable


def test_a_broadcast_mask_does_not_follow_the_callers_array():
    present = np.ones((2, 3, 1), dtype=bool)
    array = surface(mask=present)
    present[0, 0, 0] = False
    assert array.mask.all()


def test_without_the_flag_a_compact_mask_still_raises():
    with pytest.raises(DimensionError, match="mask has shape"):
        LabelledArray({"vol": VOL}, DIMS, mask=np.ones((2, 3, 1), dtype=bool))


# -- operations on a broadcast variable -----------------------------------------


def test_isel_with_a_slice_reads_the_broadcast_values():
    picked = surface().isel(expiry=slice(0, 2))
    np.testing.assert_array_equal(picked["tau"], np.broadcast_to(TAU[:, :2], (2, 2, 4)))


def test_fancy_isel_materialises_the_right_values():
    picked = surface().isel(expiry=[2, 0])
    np.testing.assert_array_equal(
        picked["tau"], np.broadcast_to(TAU[:, [2, 0]], (2, 2, 4))
    )


def test_select_reads_the_broadcast_values():
    picked = surface().select(strike=100, drop=True)
    np.testing.assert_array_equal(picked["tau"], TAU[:, :, 0])


def test_a_deep_copy_materialises_a_writable_array():
    copied = surface().copy(deep=True)
    np.testing.assert_array_equal(copied["tau"], np.broadcast_to(TAU, SHAPE))
    assert copied["tau"].flags.writeable


def test_a_pickle_round_trip_keeps_the_values():
    array = surface()
    assert pickle.loads(pickle.dumps(array)).equals(array)


def test_to_dataframe_round_trip(engine):
    array = surface()
    frame = array.to_dataframe(engine=engine, index=False)
    assert len(frame) == VOL.size
    back = LabelledArray.from_dataframe(frame, dims=list(DIMS))
    np.testing.assert_array_equal(back["tau"], np.broadcast_to(TAU, SHAPE))
    np.testing.assert_array_equal(back["vol"], VOL)


def test_transpose_keeps_the_view():
    flipped = surface().transpose("strike", "expiry", "date")
    assert flipped["tau"].base.nbytes == TAU.nbytes
    np.testing.assert_array_equal(
        flipped["tau"], np.broadcast_to(TAU, SHAPE).transpose(2, 1, 0)
    )
