# labelled-array

N-dimensional arrays that carry their dimension names and labels, align and
reduce over them, convert losslessly to and from polars or pandas DataFrames,
and hold their data on numpy, JAX or torch.

```python
import polars as pl
from labelled_array import LabelledArray

df = pl.DataFrame(
    {
        "region": ["north", "north", "south", "north"],
        "product": ["AAA", "BBB", "AAA", "CCC"],
        "units": [10, 20, 30, 40],
        "price": [1.5, 2.5, 3.5, 4.5],
    }
)

sales = LabelledArray.from_dataframe(
    df, cols=["units", "price"], dims=["region", "product"]
)

sales.shape  # (2, 3)
sales.dims["product"].tolist()  # ['AAA', 'BBB', 'CCC']
sales.density  # 0.666...  -- ('south', 'BBB') has no row
sales.select({"region": "north"})  # length-1 'region' dimension
sales.select({"region": "north"}, drop=True)  # 'region' removed
sales.select({"region": "north", "product": ["CCC", "AAA"]})  # several at once
sales.to_dataframe()  # back to the original four rows, as a polars frame
sales.to_dataframe("pandas")  # or as pandas, dimensions in a MultiIndex
```

Either library's frame goes into `from_dataframe` and builds the same array;
`to_dataframe` writes polars unless asked for `"pandas"`, and falls back to
pandas only where polars is not installed. `examples/13_polars_and_pandas.py`
runs the two side by side.

## What it is

A `LabelledArray` holds:

| Part | What it is | Where it lives |
| --- | --- | --- |
| **variables** | one or more equally shaped arrays | the array backend |
| **dims** | ordered name → unique labels, one per axis | always numpy |
| **alt_dims** | extra, repeatable labels for one dimension's entries; no axis of their own | always numpy |
| **mask** | bool array, True where a source row existed | always numpy |
| **coords** | full-shape labels a natural index stands in for; given at any broadcastable shape | always numpy |
| **attrs** | free-form metadata | plain dict |

Dimension, alternative dimension, variable and coordinate names share one
namespace, because a DataFrame column can only be one of them. That is what lets
`select` take any of them without being told which kind it was given.

## Install

```bash
pip install -e ".[dev]"          # pandas, polars, pytest, coverage, hypothesis
pip install -e ".[jax]"          # optional array backends
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## The presence mask

A cell with no source row still needs a value. Rather than widening the dtype
or picking a sentinel, the array keeps the column's dtype, fills the gap with
something the dtype can hold, and records the gap in a boolean mask:

```python
sales["units"].dtype  # int64 -- not widened to float
sales["units"][1, 1]  # -9223372036854775808 -- a placeholder, not data
sales.mask[1, 1]  # False -- this is what says the cell is empty
```

The placeholder is never read by this package, but a bare `sales["units"]`
carries no mask with it, so it is chosen to be unmistakable: NaN, NaT or None
where the dtype has a null, the minimum of a signed integer and the maximum of
an unsigned one where it does not (`bool` has no value to spare and gets
`False`; strings get `""`). `default_fill_value(dtype)` says what any dtype
gets, and `fill_value=` on `from_dataframe`, `reindex`, `align` and `concat`
overrides it, for every variable or per variable -- worth doing under
`mask=False`, where the placeholder is all that marks an invented cell.

`to_dataframe` emits only the masked-in cells, which is why `int`, `bool`,
`str` and `datetime` columns all survive a sparse round trip unchanged.

See `examples/11_presence_mask.py` for how the mask moves through arithmetic,
reductions, alignment and filling, and `examples/12_ordering.py` for what each
operation does to the order of dimensions and of the labels within them.

## Coordinates that do not vary along every dimension

A coordinate labels every cell, but it need not be *given* that way. Anything
broadcastable to the array's shape is accepted and stored broadcast, so the
values behind it are held once:

```python
chain = LabelledArray(
    {"price": prices},  # (3, 100)
    {"expiry": expiries, "strike": strikes},
    coords={"time_to_expiry": tau[:, None]},  # (3, 1) -- one per expiry
)

chain.coords["time_to_expiry"].shape  # (3, 100) -- every consumer reads it full
chain.coords["time_to_expiry"].base.nbytes  # 24 -- three floats, held once
```

Every strike on one expiry shares its time to expiry by definition, so storing
it full-shape would repeat each value a hundred times and carry no information.

Two things follow, and are limitations rather than bugs:

- **A stored coordinate is read-only.** A broadcast one has a zero stride, and
  one assignment through it would silently update every cell sharing the
  underlying element.
- **The saving ends at the first slice, or the first deep copy.** `select`,
  `isel` and `reindex` fancy-index the view, and numpy materialises the full
  array there; so does `copy()` (the default is deep). The saving is real at
  construction and through `transpose`, `copy(deep=False)`, `equals` and `repr`.

Variables are not extended this way: they live on a backend, and arithmetic and
`__setitem__` write to them. Call `np.broadcast_to(...).copy()` yourself if you
want one, and the explicit copy is the point.

## Natural index dimensions

A dimension whose labels are nearly unique per group — dates, usually — makes
the dense array enormous. `natural=` numbers those values by position within
the other dimensions instead, and keeps the values themselves as a coordinate
so the round trip still returns the original column:

```python
LabelledArray.from_dataframe(panel, cols=["value"], dims=["group", "item", "date"])
# shape (3, 4, 12), 8% populated

LabelledArray.from_dataframe(
    panel, cols=["value"], dims=["group", "item", "date"], natural=["date"]
)
# shape (3, 4, 1), 100% populated, dims ('group', 'item', 'date_idx')
```

See `examples/04_natural_index_and_sparsity.py`.

## Alternative dimensions

A descriptive column often varies along one dimension and only that one: a
sector per asset, a currency per book. As a dimension of its own it spans every
combination and leaves the array mostly empty; as a coordinate it is stored at
the array's shape and cannot be selected on. An **alternative dimension** labels
the *entries* of a dimension the array already has:

```python
book = LabelledArray.from_dataframe(
    panel, cols=["px"], dims=["asset", "date"], alt={"sector": "asset"}
)
# or, given directly:
LabelledArray(
    {"px": prices},
    dims={"asset": assets, "date": dates},
    alt_dims={
        "sector": {"ref_dim": "asset", "labels": sectors}
    },  # or ("asset", sectors)
)

book.shape  # (4, 3) -- still two axes, and fully populated
book.dims.alt["sector"]  # 4 labels, one per asset, held once
book.select({"sector": "tech"})  # 3 assets x 3 dates
book.select({"sector": "fin", "date": "d2"})  # mixes with a real dimension
book.groupby("sector").mean()  # 'asset' comes back labelled by sector
```

Its labels **need not be unique** — that is the point, since several entries
usually share a description — so one of them keeps every entry carrying it, in
the order requested and ascending within each label. Names resolve as a
dimension first and an alternative dimension second, so `select` needs no extra
argument.

An alternative dimension describes the entries of its dimension, so it is
carried by whatever keeps those entries and dropped by whatever replaces them:

| Operation | What happens |
| --- | --- |
| `select`, `isel`, `transpose`, `rename`, `copy`, pickle, arithmetic | carried, subset alongside its dimension |
| `select(..., drop=True)` reducing its dimension away | dropped with it |
| `reindex` onto a subset or a permutation | carried |
| `reindex` onto labels the array did not have | dropped — a new label has no description, and a null one could never be selected |
| `groupby("sector")` | groups its own dimension; the result is labelled by group value, so it is dropped |
| a reduction or a `groupby` over its dimension | dropped — the entries are gone |
| `concat` along its dimension | joined, like that dimension's own labels |
| `concat` along another dimension | carried, and the operands must agree on it |
| `to_dataframe` | written out as an ordinary column |

Two things it deliberately is not:

- **It has no axis**, so `isel` cannot name it. It has no positions of its own,
  only the ones it points at.
- **Two names cannot select along one dimension in a single call** — an
  alternative dimension and its own dimension, say. That is an intersection
  whose order neither request defines, so it is asked for as two chained calls.

`from_dataframe(..., alt={"sector": "asset"})` folds a column in, and refuses one
that says two different things about a single asset, or one tied to a `natural=`
dimension whose labels are positions rather than values.

See `examples/10_alternative_dimensions.py`.

## Operations

Reductions, groups and windows are mask-aware in the same way the round trip
is: a cell with no source row does not contribute, and a result with nothing
behind it is filled and marked absent rather than being invented.

```python
book.sum("date")  # sum / mean / min / max over named dimensions
book.groupby(
    sectors, dim="asset"
).mean()  # or groupby("sector"), an alt dimension or a coordinate
book.rolling("date_idx", 20).mean()  # trailing windows; min_periods= for partial ones
```

Combining two arrays is a question about their labels first:

```python
book.reindex(date=week)  # onto another label set, filling what is new
book.ffill("date")  # carry the last present value forward; limit= caps how far
book.fillna(0.0)  # fill the absent cells and drop the mask
book.align(other, join="outer")  # one label set per shared dimension
book - other  # arithmetic aligns on its own (inner join)
LabelledArray.concat([jan, feb], dim="date")  # extend a dimension, or stack a new one
```

The operations are deliberately narrow, and say so rather than guessing:

- **Arithmetic aligns labels, but the dimension names must match.** Nothing
  broadcasts one dimension onto another.
- **`groupby` groups one dimension by one value per label.** A coordinate that
  varies along two dimensions is refused rather than flattened.
- **An alternative dimension is dropped, not invented, when its labels go.** A
  reduction, a `groupby`, or a `reindex` onto labels the array never had leaves
  nothing for it to describe, and it is removed rather than null-filled.
- **Windows are trailing**, never centred.
- **There is no `skipna`.** Absence is what the mask records; a NaN that is
  masked in is data, and propagates.
- **`ffill` carries over the mask, not over NaN**, for the same reason, and
  forward only. A cell it fills becomes present, so `to_dataframe` emits it.

See `examples/08_combining_and_reducing.py`,
`examples/09_ffill_and_staleness.py`, and `examples/07_timings_at_scale.py` for
what each of them costs on a million cells.

## Round trip gotchas

The round trip is exact for the great majority of dtypes. Where it is not, the
cause is that numpy has no equivalent of the source dtype. Every case below is
asserted in `tests/test_dtypes.py`, so none of it can change silently.

| Source | Becomes | Comes back as | Lost |
| --- | --- | --- | --- |
| int, uint, float, bool, complex | same | same | nothing |
| `datetime64`, `timedelta64` (any unit) | same | same | nothing |
| polars `Date`, `Datetime`, `Duration` | `datetime64` | same | nothing |
| `timedelta64[D]`, `[h]`, `[m]`, `[s]` and `datetime64[s]`, `[m]`, `[h]`, written to polars | -- | `Duration`/`Datetime` in `ms` | the unit; values are exact |
| `datetime64[W]`, `[M]`, `[Y]`, written to polars | -- | `Date` | the unit; values are exact |
| python/pandas/polars strings | `object` | the library's string dtype | nothing |
| pandas tz-aware datetime | `object` of `Timestamp` | same tz-aware dtype | vectorised datetime ops; cannot move to another backend |
| pandas `category` | `object` | `str` | the categories and their order |
| pandas nullable `Int64` | `int64` | `int64` | the extension dtype |
| pandas nullable `Int64` holding NA | `float64` | `float64` | the integer dtype (pandas widens on `to_numpy`) |
| pandas nullable `boolean` | `object` | `object` | the extension dtype |
| pandas `string` | `object` | `str` | the extension dtype |
| polars `Categorical`, `Enum` | `object` | `String` | the categories |
| polars `Null` | `float32` | `Float32` | the null dtype |

Beyond dtypes:

- **Row order is not preserved.** Rows come back in the array's stored layout,
  which straight after `from_dataframe` is sorted by the dimension labels, so
  compare against the source sorted by its dimension columns. A reordering
  `select`, `reindex` or `concat` changes that layout, and the rows follow it.
- **Labels are sorted.** Object columns whose values cannot be compared keep
  first-seen order instead.
- **Null labels are refused.** A NaN or None label could never be selected, as
  it does not compare equal to itself, so `from_dataframe` raises rather than
  building an array with a label you cannot reach.
- **Duplicate rows are refused.** Two rows for one cell are ambiguous;
  `DuplicateIndexError` names the offending labels.
- **A natural index needs labels unique within the other dimensions.** The
  index counts rows, not distinct labels, so `natural="expiry"` on a long
  option chain (one row per strike, expiries repeating) would number every
  strike as its own expiry. `from_dataframe` raises `DuplicateIndexError`
  rather than build that ladder.
- **`ffill` refuses a natural-index dimension.** The padding cells it would
  fill were never rows, and each would carry its source's label, so the
  result could not come back through `from_dataframe`. Fill along a real
  dimension instead.
- **Stored coordinates are read-only.** `array.coords["x"][...] = value`
  raises; build a new coordinate and `_replace` it, or write to your own array
  before passing it in.
- **JAX narrows 64-bit dtypes to 32-bit** unless `jax_enable_x64` is set, and
  neither JAX nor torch can hold strings, objects or datetimes.
- **An explicit fixed-width string dtype truncates** (`dtype="<U3"`); the
  default `object` labels do not.

## Extending it

Array backends and DataFrame engines are registries, and `LabelledArray` builds
every derived object through `type(self)`:

```python
class ClimateArray(LabelledArray):
    _idx_suffix = "_pos"

    def _validate(self):
        super()._validate()
        if "date" not in self.dims:
            raise DimensionError("a ClimateArray needs a 'date' dimension")


july.select({"station": "Oslo"})  # still a ClimateArray
```

See `examples/06_subclassing.py`, and `register_backend` / `register_adapter`
for supporting another array or DataFrame library from outside the package. An
`ArrayBackend` only has to provide the abstract methods: `take_multi`, `full`,
`where`, `concatenate`, `reduce` and `sliding_window` all have working
implementations in the base class, which a backend overrides only where its
library can do better -- `np.ix_` for a multi-axis gather, a view rather than a
copy for a sliding window.

## Examples and tests

```bash
python examples/01_basics.py                # 01 basics ... 13 polars and pandas
pytest -q                                   # the full suite
pytest --cov=labelled_array --cov-report=term-missing
```

`examples/07_timings_at_scale.py` measures where the time goes on a million-row
frame — what dominates `from_dataframe`, what a `select` costs, what the
operations above cost, and what sparsity costs in memory rather than in time.
Set `LABELLED_ARRAY_SCALE=2` to run it on four times the cells.

## License

MIT — see [LICENSE](LICENSE).

## Documentation

Sphinx docs -- getting started, an examples gallery built from `examples/`, and an API
reference:

```bash
uv sync --extra docs && uv run make -C docs html
```

See `docs/building.rst` for the `DOCS_RUN_EXAMPLES` switch that controls whether the
gallery executes the scripts.
