# `price_to_vols_poc` — what `labelled-array` needs to provide

**Status:** requirements, not an implementation. Written to be picked up cold.

## Why this repo is involved

`mktdata-derived` is building a proof of concept that turns EOD SPX option prices into a
market volatility surface: read the chain out of the silver Delta table, attach a zero rate
at each expiry, back out the forward from put-call parity, and invert Black-76.

Two things that pipeline needs do not exist here yet. Neither is specific to options — both
are general gaps that this particular consumer happened to hit first.

This repo is at the bottom of the dependency graph and has no sibling dependencies, so it
lands first. Nothing here depends on any other change in the plan.

---

## 1. Coordinates that broadcast

### The problem

`_as_coord` (`src/labelled_array/core.py:1265`) rejects anything whose shape is not exactly
`dims.shape`:

```python
def _as_coord(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Return coordinate ``value`` as a numpy array of ``shape``."""
    array = np.asarray(value)
    if array.shape != shape:
        raise DimensionError(
            f"coordinate {name!r} has shape {array.shape}, expected {shape}"
        )
    return array
```

The POC builds an option chain over three dimensions:

```
dims   : (snap_datetime, expiry_point, smile_point)
coords : time_to_expiry  -- a property of (snap_datetime, expiry_point) only
         strike          -- genuinely varies per cell
```

`time_to_expiry` is constant along `smile_point`: every strike in one expiry shares one time
to expiry, by definition. Storing it full-shape repeats each value `n_smile_point` times —
for SPX that is a factor of several hundred — and the repetition carries no information.

The caller already knows how to say this. `tau[..., None]` has shape
`(n_snap, n_expiry, 1)`, which numpy broadcasts against `(n_snap, n_expiry, n_strike)`
without materialising anything. Today `_as_coord` refuses it.

### What to build

Accept any coordinate broadcastable to `dims.shape`. Store the broadcast result and freeze
it read-only:

```python
def _as_coord(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Return coordinate ``value`` as a read-only numpy array of ``shape``.

    A coordinate that does not vary along every dimension may be given at any
    broadcastable shape -- ``(n, 1)`` against ``(n, m)``, or ``(m,)`` against the
    trailing axis. It is stored broadcast, so every consumer still reads a
    full-shape array, but the values behind it are held once.
    """
    array = np.asarray(value)
    if array.shape != shape:
        try:
            array = np.broadcast_to(array, shape)
        except ValueError as exc:
            raise DimensionError(
                f"coordinate {name!r} has shape {array.shape}, "
                f"which is not broadcastable to {shape}"
            ) from exc
    array = array.view()
    array.flags.writeable = False
    return array
```

The stored coordinate is still full-shape, so `_map_cells`, `to_dataframe`,
`_reduce_coords`, `equals` and `to_repr` need no change at all. That is the whole reason to
prefer this over storing coordinates compactly with their own axis map.

Freezing is new but consistent: `_as_mask` already freezes, and a zero-stride view is exactly
the array you must not write through — one assignment would silently update every cell that
shares the underlying element.

### The limitation, which is documented rather than fixed

The first `select` or `isel` fancy-indexes the view, and numpy materialises the full array at
that point. So the saving is real at construction and through `transpose`, `copy`, `equals`
and `to_repr`, and gone after the first slice.

The alternative — store compactly and record which dimensions each coordinate spans, the way
`xarray` does — keeps the saving through slicing, but it breaks the invariant this package is
built on: that variables, coordinates and the mask are indexed by the same dimensions and
therefore move together through `_map_cells`. Every shape-changing operation would need a
per-coordinate axis map.

Record it in `TODO.md` as a follow-up, and let the POC's measured array sizes decide whether
it is worth the cost. Do not build it speculatively.

### Not extended to variables

`_validate` (`core.py:150`) keeps its exact shape check. Two reasons:

- Coordinates are always numpy. Variables live on a backend, so broadcasting them needs a new
  `ArrayBackend.broadcast_to` implemented across numpy, jax and torch.
- A zero-stride variable is a hazard. `__setitem__` and the arithmetic mixin both write to
  variables, and writing through a broadcast view corrupts every cell sharing the element.

Anyone who wants a broadcast variable can call `np.broadcast_to(...).copy()` themselves, and
the explicit copy is the point.

---

## 2. `LabelledArray.ffill(dim, limit=None)`

### The problem

There is no `ffill`, no `bfill` and no `skipna` anywhere in this package. Absence is the mask,
and a masked-in NaN is data — which is the right default, and is not what this case needs.

The POC aligns a zero rate curve to the dates option prices exist on. Those two calendars do
not agree: a curve is not published on every date the option table has a snapshot for. The
correct behaviour is to carry the most recent curve forward, and — critically — to know how
old the one you used is, because a three-day-stale discount curve is a different fact about
your forward than a same-day one.

`reindex` (`_align.py:86`) already does the first half: it puts the array on the option
snapshot dates, filling absent labels and marking them `False` in the mask. What is missing
is the carry.

### What to build

Forward fill along one dimension, in label order, mask-aware:

```python
def ffill(self, dim: str, limit: int | None = None) -> LabelledArray:
    """Carry the last present position forward along ``dim``.
```

Behaviour:

| case | result |
| --- | --- |
| absent cell after a present one | takes the last present value; becomes present in the mask |
| absent cell before any present one | stays absent, stays at its fill value |
| `limit=n` | carries at most `n` positions; beyond that, cells stay absent |
| array with no mask | a no-op, since every cell is present |
| unknown `dim` | `DimensionError` naming the dimension and listing the real ones |

Variables, coordinates and the mask must move together — route it through `_map_cells` like
every other shape-preserving operation, so the three cannot drift apart.

Filled cells become **present**. This matters: if they stayed absent, `to_dataframe` would
drop the very rows the fill was performed to create.

Note that the fill is over the *mask*, not over NaN. A masked-in NaN is a real observation of
nothing and is left alone, consistent with the rest of the package. A caller who wants NaN
treated as absence can say so by building the mask that way.

### The staleness counter needs no new concept

The original brief for the POC asked for "an alt-dimension to track the number of forward
fills a given curve has had". There is no need for one. Take the mask before the fill, derive
the age, and store it as a **coordinate** — which change 1 above makes free, because the age
is a property of `dim` alone:

```python
present = zc.mask.any(axis=-1)  # before the fill
age = _days_since_last_true(present, zc.dims["snap_date"])
zc = zc.ffill("snap_date")
zc = zc._replace(coords={**zc.coords, "fill_age_days": age[:, None]})
```

Document this recipe in the `ffill` docstring or the example. It is the reason the two changes
in this file ship together.

---

## Acceptance criteria

### `tests/test_coords.py` (new)

- A `(n, 1)` coordinate against `(n, m)` dimensions is accepted, and `array.coords[name]`
  reads back with shape `(n, m)` and the right values on every cell.
- A `(m,)` coordinate broadcasts along the trailing axis.
- A scalar coordinate broadcasts to the whole shape.
- A `(3,)` coordinate against `(4, 5)` dimensions raises `DimensionError`, and the message
  names the coordinate, its shape and the target shape.
- The stored array has `flags.writeable is False`.
- `array.coords[name].base.nbytes` is the compact size, not the full size — the assertion that
  proves the value is actually stored once.
- An array carrying a broadcast coordinate survives `select`, `isel`, `transpose`, `reindex`,
  `copy(deep=True)`, `pickle.loads(pickle.dumps(...))` and `to_dataframe` with the
  coordinate's values intact at every cell.
- Existing full-shape coordinates behave exactly as before — run the existing coordinate tests
  unchanged.

### `tests/test_ffill.py` (new)

- Fills a single interior gap along a dimension.
- Fills a run of consecutive gaps.
- `limit=1` fills one position and leaves the second absent.
- A gap before the first present position stays absent and keeps its fill value.
- A fully present array is returned unchanged (`equals` holds).
- Variables, coordinates and the mask all move together: assert on all three after a fill.
- A dimension that is not present raises `DimensionError` naming it.
- Parametrised over the `backend` and `engine` fixtures already in `conftest.py`, so it is
  proven on every backend rather than on numpy alone.
- The staleness recipe above produces the expected ages on a curve with a known gap pattern.

### Repo-wide

- `filterwarnings = ["error::RuntimeWarning"]` is on. No test may provoke a numpy warning, so
  aggregations over possibly-empty selections must be written to avoid one rather than to
  suppress one.
- Test names are full sentences, matching the existing convention
  (`test_missing_labels_raise_by_default`, `test_drop_only_removes_dimensions_that_were_selected`).
- `examples/`: either extend `04_natural_index_and_sparsity.py` with the broadcast
  coordinate — it is already the sparsity example, so this belongs there — or add
  `09_ffill_and_staleness.py` in the same Sphinx-Gallery `# %%` style.
  `tests/test_examples.py` runs every example, so a new one must actually work.
- `README.md`: the coordinate section states the broadcasting rule and the
  materialise-on-slice limitation; the operations table gains `ffill`.
- `TODO.md`: record compact coordinate storage as the deferred follow-up, with the reason it
  was deferred.

### The gate

```sh
uv sync --locked && uv run ruff check . && uv run ruff format --check . && uv run pytest
```

Note that CI runs `uv sync --locked` but this repo has historically not committed a
`uv.lock`. If that is still true, commit one in this change or the workflow cannot pass.
