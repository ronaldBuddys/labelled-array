# Alternative dimensions — outstanding issues

Tracking notes for the `alt_dims` feature. Not a backlog of the whole package —
`TODO.md` remains that. Everything here was found while building the feature and
deliberately left alone.

## Open

### 1. `LabelNotFoundError` calls an alternative dimension a "dimension"

`book.select(sector="energy")` reports:

```
labels not found in dimension 'sector': ['energy']
```

`'sector'` is an alternative dimension, not a dimension. The exception takes a
single `dim` argument (`src/labelled_array/exceptions.py:22`) and is raised from
both `Dimensions.get_indexer` and `Dimensions.alt_indexer`. Fixing it means
either a second argument naming what kind of thing was searched, or a sibling
exception. Left alone because the message is still unambiguous — the name is
right, only the noun is wrong — and widening a public exception's signature for
a noun is not obviously worth it.

### 2. A widening `reindex` drops the alternative dimension

`book.reindex(asset=["AAA", "ZZZ"])` drops `sector`, because `ZZZ` has no
description and a null alternative label could never be selected (the same rule
`_as_labels` applies to a dimension's own labels). The alternative would be a
sentinel — `""`, `NaN`, a dedicated `MISSING` object — which every consumer would
then have to know about.

`concat` along the described dimension *does* join the descriptions
(`_align._concat_alt`), so the gap is only the reindex case. Worth revisiting if
a caller actually hits it; the shape of the answer is a `fill_value=`-style
argument on `reindex` naming the label to give new entries.

### 3. `align` and arithmetic inherit that drop silently

`a - b` aligns on an inner join. Where that widens or narrows the described
dimension, the alternative dimension is carried (subset) or dropped (new labels)
by the `reindex` underneath, with no warning either way. Consistent with issue 2,
and with how coordinates behave, but it is silent.

### 4. `groupby` on an alternative dimension always drops it from the result

`book.groupby("sector").sum()` relabels `asset` by sector, so the alternative
dimension described labels that no longer exist and goes. That is right, but it
means the round trip `groupby("sector")` → `to_dataframe` emits a column named
`asset` holding sector values. Renaming the grouped dimension is the caller's
job today; `groupby(...).sum().rename(asset="sector")` collides with nothing
because the alternative dimension is already gone.

### 5. Two alternative dimensions on one dimension cannot be selected in one call

`select({"sector": "tech", "region": "emea"})` where both describe `asset` is
refused, because the intersection's ordering is not defined by either request.
Chained calls work. An explicit `select(..., how="intersect")` would be the way
to allow it if anyone wants it, with the order taken from the dimension rather
than from the request.

### 6. `from_dataframe(alt=...)` refuses a natural-index reference

A `natural=` dimension is labelled by position, so a label-keyed description has
nothing to attach to and the pairing is refused
(`_convert.from_dataframe`). The column belongs as a coordinate there, which is
what the error says. Recording it as a known limitation rather than a bug.

## Considered and rejected

- **Storing alternative dimensions as a separate `LabelledArray` field.** They
  live inside `Dimensions` instead, so `_fields()` carries them and every
  `_replace` — `copy`, pickle, `transpose`, `astype`, arithmetic, `ffill`,
  `fillna` — keeps them with no change. A parallel field would have needed a
  hand-written branch in each.
- **Making them sugar over `coords`.** A coordinate is stored at the array's
  shape and materialises on the first `select`; an alternative dimension is one
  label per entry and never broadcasts. For 4 assets over 250 dates that is 4
  strings against 1000.
- **Allowing duplicate *names* across the two namespaces.** Rejected for the
  same reason dims/variables/coords already share one: a DataFrame column can
  only be one thing.

## Verification

```bash
uv run pytest -q                                     # 1084 tests
uv run pytest -q tests/test_alt_dims.py              # the feature itself
uv run python examples/10_alternative_dimensions.py  # assertions are inline
uv run ruff check . && uv run ruff format --check .
uv run pytest --cov=labelled_array --cov-report=term-missing
```

Coverage is 100% on every module this touched (`dimensions`, `core`, `_align`,
`_ops`, `_convert`); the four uncovered lines in the package are the
pre-existing "library not installed" branches in `_plugin` and `_utils`.
