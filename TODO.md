# TODO

Remaining work, in priority order. Sparsity is last: the dense layout is
workable today, and a natural index already covers the case that hurts most.

## Conversion

- **Explicit label sets.** `from_dataframe(..., labels={"date": ...})` to force
  a dimension's labels and their order, for aligning several frames onto one
  layout instead of taking each frame's own values.
- **More engines.** A `pyarrow` adapter, and `duckdb`/`ibis` relations.
- **Streaming construction.** Building from a frame too large to hold in memory,
  scattering chunk by chunk into a preallocated array.
- **Preserve extension dtypes.** Round-tripping pandas `category` and nullable
  dtypes would need the categories stored alongside the labels, rather than
  going through numpy. The lossy cases are tabulated in `README.md`.

## Operations

The alignment, `concat`, reduction, `groupby`, `reindex`/`fillna`, rolling and
multi-axis `take` items are done; `examples/08_combining_and_reducing.py` shows
them and `examples/07_timings_at_scale.py` times them. What they left behind:

- **Broadcasting.** Arithmetic aligns labels but still needs the same dimension
  *names* on both operands. Combining an array over `(asset, date)` with one
  over `(asset,)` would mean broadcasting the missing dimension, which is a
  bigger decision than aligning labels and is deliberately not made yet.
- **More reductions.** `std`, `var`, `prod`, `count`, `median`, `any`/`all`,
  and the `argmin`/`argmax` pair, which would return labels rather than
  values. `median` is the one consumers have had to hand-write so far
  (`mktdata-derived`'s parity forward is a masked median over strikes).
- **Vectorised grouped reductions.** `GroupBy.reduce` loops over the groups in
  python: one `isel` and one reduction each. The whole pass is still linear in
  the array, but 200 groups over a million cells measured about 65 ms against
  the 14 ms of 20 groups, and pandas does the same work in 6 ms.
  `np.add.reduceat` and friends would do it in one pass, at the cost of an
  optional backend hook -- torch and JAX have no `reduceat`.
- **Running totals for rolling sums.** Each window is reduced in full, so the
  cost is the array times the window: a 250-wide mean over a million cells is
  about 110 ms against pandas' 30 ms, which keeps a running total instead. A
  `cumsum`-based path would be four times quicker and window-independent, but a
  NaN anywhere poisons every window after it, so it needs the NaN bookkeeping
  pandas does rather than a bare `cumsum`.
- **Centred and expanding windows.** `rolling` is trailing only.
- **`bfill`.** `ffill` carries forward only. Backward fill is the same gather
  with a reversed running maximum, and was left out because nothing has needed
  it yet.
- **Compact coordinate storage.** A coordinate given at a broadcastable shape is
  stored broadcast, so it is held once -- until the first `select`, `isel` or
  `reindex` fancy-indexes the view and numpy materialises it. Keeping the saving
  through slicing means storing coordinates compactly and recording which
  dimensions each one spans, the way `xarray` does. That breaks the invariant
  this package is built on -- that variables, coordinates and the mask are
  indexed by the same dimensions and therefore move together through
  `_map_cells` -- because every shape-changing operation would then need a
  per-coordinate axis map. Deferred deliberately: let measured array sizes
  decide whether it is worth that, rather than building it speculatively.
- **Grouping by a multi-dimensional coordinate.** `groupby` needs a coordinate
  that varies along exactly one dimension. Grouping cells by a value that
  varies along several -- every cell of a given date, in a natural index panel
  where the groups do not share dates -- is a scatter, not a selection, and the
  result's dimensions are not obvious.
- **`concat` of arrays holding different variables**, taking the union and
  marking the cells the other array never had.
- **Carrying an alternative dimension through a widening `reindex`.** A label
  the array never had has no description; carrying one would mean a null
  alternative label, which could never be selected. `concat` along the described
  dimension joins them, so the gap is only the reindex case.
- **Alternative dimensions built from more than one column**, the way a
  composite dimension would be.

## Storage

- **On-disk formats.** hdf5, parquet and netCDF, with the dimensions, mask and
  coordinates stored alongside the variables.
- **`to_xarray` / `from_xarray`** for interoperating with the wider ecosystem.

## Project setup

None of this exists yet:

- **Linting and formatting.** No `ruff`/`black` configuration, and neither has
  been run over the package.
- **Type checking.** Everything is annotated, but `mypy`/`pyright` has never
  been run, so the annotations are unverified.
- **CI.** No workflow running the suite across Python versions, or with the
  optional libraries absent. Those "library not installed" branches — in
  `Registry.get` and the lazy backends — are the only lines the suite cannot
  reach in an environment where everything is installed.
- **Docs build.** The examples are written in sphinx-gallery style but there is
  no `docs/` tree or `conf.py` to render them.

## Deferred cleanups

Raised by the code-quality review and deliberately not done:

- **Merge the mask and coordinates into `data` as ordinary variables.** It would
  replace the three parallel loops in `_map_cells` with per-name special cases;
  the mask is genuinely different in kind — optional, boolean, and it drives
  `density` and row selection rather than becoming a column.
- **One `assert_round_trip` test helper.** `tests/test_roundtrip.py` and
  `tests/test_dtypes.py` each define a local `round_trip` returning different
  tuples, and the assertion is spelled out at about twenty call sites. Worth
  consolidating next time either file is touched.
- **`Dimensions.replace()`** now has callers -- `reindex`, `concat` and the
  grouped reductions all relabel one dimension through it -- so it stays.

## Smaller items

- `_repr_html_` for notebook display.
- Composite dimensions built from more than one column.
- A `chunks`/lazy backend (dask, zarr).

## Sparsity

The dense layout spans the product of the dimension label sets, so an array is
only as efficient as its `density` — the fraction of cells a source row
populates. When dimensions are close to independent, that fraction collapses:
in `examples/04_natural_index_and_sparsity.py`, twelve rows span 144 cells.

`natural=` is the current answer, and it applies only when one dimension's
labels are *unique within* the others (dates, event ids). That is a correctness
precondition, not a performance one — the index counts rows, so a repeated
label would become two rungs — and `from_dataframe` raises
`DuplicateIndexError` when it does not hold. A mode that ranks distinct labels
instead of rows would lift it (a long option chain wants exactly that) and is
not built. The unique case covers the common one, which is why the rest of this
is a nice to have rather than a blocker:

- **Sparse backends.** An `ArrayBackend` over `sparse`/COO or `scipy.sparse`,
  so the same object can hold a sparse buffer. The backend registry already
  allows this; the open question is which operations (`take` along one axis in
  particular) stay cheap.
- **Choose the layout automatically.** `from_dataframe` could measure the
  density it is about to produce and either warn, or promote a dimension to a
  natural index on its own, above a configurable threshold.
- **Dimension order heuristics.** Grouping a natural index by the dimensions
  with the fewest labels gives the smallest result; this is currently left to
  the caller.
- **Refuse to allocate.** A `max_cells` guard would turn a silent multi-gigabyte
  allocation into an error naming the dimension responsible.
