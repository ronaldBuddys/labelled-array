# Coverage manifest

What the records harness covers, and, as explicitly, what it does not. 16 checkpoints,
98 records (seed bundle b49-labelled-array-seed). `r` counts the records that raise on
the current base. A record count that changes without a scenario change is itself a
finding. `record` writes one record per distinct call, so two scenario lines making the
same call count once.

## Covered

| Checkpoint | Records | Cases |
|---|---|---|
| `LabelledArray.select` | 13 (2 r) | label, list, scalar with `drop`, datetime labels, `missing="ignore"`, an int-labelled `(b, a)` array, a 3-D cube; a missing label (`LabelNotFoundError`); a label slice (refused as an unknown label today) |
| `LabelledArray.isel` | 10 | negative position, negative with `drop`, strided slice, integer arrays on two dimensions, `(b, a)`, 3-D with `drop` |
| `_align.reindex` | 4 | new dates and symbols (NaN-filled cells), `fill_value`, int labels on `(b, a)`, a shorter date set |
| `_fill.ffill` | 5 | along `date` on reindexed and on NaN-gapped arrays, with and without `limit`, along `sym` |
| `_align.align` | 5 (1 r) | inner, outer, left with `fill_value`, outer on differing symbols; `exact` on differing labels (`DimensionError`) |
| `_align.concat` | 4 | along an existing dimension (inner and outer join) and along a new one (string and int labels) |
| `_ops.groupby` | 4 (1 r) | by a list on `sym`, on `date` with gaps, on a cube dimension; a group list of the wrong length |
| `GroupBy.reduce` | 6 | `sum`, `mean`, `min`, `max` by symbol group (`.sum()` and `.mean()` reach the same record); `mean` over gappy dates; `sum` on a cube dimension |
| `_ops.rolling` | 4 | window 3 on clean and NaN data, `min_periods=1`, window 2 |
| `Rolling.reduce` | 6 | `mean`, `sum`, `max`, `min` with NaN in the window |
| `LabelledArray.__add__` | 8 (2 r) | scalar, ndarray broadcast, an aligned `LabelledArray` (shorter dates, reordered symbols), the same with axes swapped, a `(b, a)` array; a missing variable and a missing dimension (`ValueError`) |
| `LabelledArray.__sub__` | 5 | the same operands |
| `LabelledArray.__mul__` | 5 | the same operands |
| `LabelledArray.__truediv__` | 6 | the same operands, and division by an all-zero array (inf, NaN) |
| `_convert.to_dataframe` | 7 | polars: 2-D, gappy, 3-D, `(b, a)`, `dropna`, `index=False`, long form |
| `_convert.from_dataframe` | 6 | polars: the same frames back, one `cols`, `fill_value`, a reordered `dims` |

`(b, a)` is an array whose axes are in non-alphabetical order, so a record that came back
sorted would transpose or fail on replay. `tests/test_records.py` also checks the
serializer round trip directly.

Constructors: `LabelledArray.__init__` and `from_array` are not recorded directly; every
record's inputs and outputs are built through them, so a change there moves the records
that use them.

## Not covered

- **pandas engine** (`to_dataframe` / `from_dataframe` with `engine="pandas"`): the
  scenarios exist, but regression-records stores a pandas frame as parquet, which needs
  pyarrow (its `pandas` extra); it is not in `uv.lock`, so the block is skipped with a
  warning. Adding pyarrow to the dev group would enable it.
- **jax and torch backends**: heavy; their parity tests (`tests/test_parity.py`,
  `tests/test_backends.py`) cover them.
- **Reflected and other operators**: `__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`,
  `//`, `%`, `**`, unary operators. Only the four forward operators are checkpoints.
- **Reductions on the array itself** (`LabelledArray.sum` / `mean` / `min` / `max`,
  `_ops.reduce`) and `transpose`, `rename`, `copy`, `__getitem__` other than through
  `select`.
- **`**kwargs` shorthand** of `select` and `isel` (`la.select(sym="A")`): the recorder
  cannot replay a `**kwargs` parameter, so scenarios pass the `indexers` mapping.
- **Plotting** and **`_plugin`**: out of scope for numeric regression.
- **A record made by a call from inside the library** (`ffill` reaching `reindex`,
  `+` reaching `align`): only the outermost call records.
