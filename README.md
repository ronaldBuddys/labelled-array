# labelled-array

N-dimensional arrays that carry their dimension names and labels, align and
reduce over them, convert losslessly to and from pandas or polars DataFrames,
and hold their data on numpy, JAX or torch.

```python
import pandas as pd
from labelled_array import LabelledArray

df = pd.DataFrame(
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
sales.to_dataframe(index=False)  # back to the original four rows
```

## What it is

A `LabelledArray` holds:

| Part | What it is | Where it lives |
| --- | --- | --- |
| **variables** | one or more equally shaped arrays | the array backend |
| **dims** | ordered name → unique labels, one per axis | always numpy |
| **mask** | bool array, True where a source row existed | always numpy |
| **coords** | full-shape labels a natural index stands in for | always numpy |
| **attrs** | free-form metadata | plain dict |

Dimension, variable and coordinate names share one namespace, because a
DataFrame column can only be one of them.

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
sales["units"][1, 1]  # 0     -- a placeholder, not data
sales.mask[1, 1]  # False -- this is what says the cell is empty
```

`to_dataframe` emits only the masked-in cells, which is why `int`, `bool`,
`str` and `datetime` columns all survive a sparse round trip unchanged.

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

## Operations

Reductions, groups and windows are mask-aware in the same way the round trip
is: a cell with no source row does not contribute, and a result with nothing
behind it is filled and marked absent rather than being invented.

```python
book.sum("date")  # sum / mean / min / max over named dimensions
book.groupby(
    sectors, dim="asset"
).mean()  # or groupby("sector"), a coordinate per label
book.rolling("date_idx", 20).mean()  # trailing windows; min_periods= for partial ones
```

Combining two arrays is a question about their labels first:

```python
book.reindex(date=week)  # onto another label set, filling what is new
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
- **Windows are trailing**, never centred.
- **There is no `skipna`.** Absence is what the mask records; a NaN that is
  masked in is data, and propagates.

See `examples/08_combining_and_reducing.py`, and
`examples/07_timings_at_scale.py` for what each of them costs on a million
cells.

## Round trip gotchas

The round trip is exact for the great majority of dtypes. Where it is not, the
cause is that numpy has no equivalent of the source dtype. Every case below is
asserted in `tests/test_dtypes.py`, so none of it can change silently.

| Source | Becomes | Comes back as | Lost |
| --- | --- | --- | --- |
| int, uint, float, bool, complex | same | same | nothing |
| `datetime64`, `timedelta64` (any unit) | same | same | nothing |
| polars `Date`, `Datetime`, `Duration` | `datetime64` | same | nothing |
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

- **Row order is not preserved.** Rows come back ordered by the dimension
  labels, so compare against the source sorted by its dimension columns.
- **Labels are sorted.** Object columns whose values cannot be compared keep
  first-seen order instead.
- **Null labels are refused.** A NaN or None label could never be selected, as
  it does not compare equal to itself, so `from_dataframe` raises rather than
  building an array with a label you cannot reach.
- **Duplicate rows are refused.** Two rows for one cell are ambiguous;
  `DuplicateIndexError` names the offending labels.
- **JAX narrows 64-bit dtypes to 32-bit** unless `jax_enable_x64` is set, and
  neither JAX nor torch can hold strings, objects or datetimes.
- **An explicit fixed-width string dtype truncates** (`dtype="<U3"`); the
  default `object` labels do not.

## Extending it

Array backends and DataFrame engines are registries, and `LabelledArray` builds
every derived object through `type(self)`:

```python
class FinancialArray(LabelledArray):
    _idx_suffix = "_pos"

    def _validate(self):
        super()._validate()
        if "date" not in self.dims:
            raise DimensionError("a FinancialArray needs a 'date' dimension")


book.select({"ticker": "AAA"})  # still a FinancialArray
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
python examples/01_basics.py                # 01 basics ... 08 combining and reducing
pytest -q                                   # the full suite
pytest --cov=labelled_array --cov-report=term-missing
```

`examples/07_timings_at_scale.py` measures where the time goes on a million-row
frame — what dominates `from_dataframe`, what a `select` costs, what the
operations above cost, and what sparsity costs in memory rather than in time.
Set `LABELLED_ARRAY_SCALE=2` to run it on four times the cells.

## License

MIT — see [LICENSE](LICENSE).
