# Regression records

Recorded calls to labelled-array's core operations on small seeded arrays, recorded and
replayed per PR by the `Records` workflow. They show *which* numbers a change moved, and
where the move starts, not to stop numbers moving. `MANIFEST.md` lists every
checkpoint, the cases its scenarios exercise, and what is deliberately not recorded.

- `checkpoints.py` lists what is recorded. regression-records is a dev dependency only,
  so nothing in `src/` is decorated. Recorders are installed while recording or
  replaying, and removed again:
  - the **module-level implementations** the `LabelledArray` methods delegate to
    (`_convert.from_dataframe` / `to_dataframe`, `_align.reindex` / `align` / `concat`,
    `_fill.ffill`, `_ops.groupby` / `rolling`), on their modules. `core` looks them up
    through the module at call time, so the recorder is hit;
  - `GroupBy.reduce` and `Rolling.reduce`, on their classes;
  - `LabelledArray.select`, `isel`, `+`, `-`, `*` and `/`, through a forwarding method of
    the same name on `LabelledArray` (see the module docstring for why).
  Only the outermost checkpoint call records: `ffill` or `+` reaching `reindex` and
  `align` inside is not a record of its own.
- `serializers.py` teaches the store what regression-records cannot record alone: a
  `LabelledArray` or `Dimensions` with the order of its dicts kept (records are written
  with sorted keys, and axes and variables are ordered), object-dtype label arrays
  (polars returns string labels so), a `slice`, and a class.
- `scenarios.py` holds the calls recorded, from `default_rng(0)` and explicit dates.
  The polars block is skipped with a warning without polars, the pandas block without
  pandas **or pyarrow** (regression-records stores a pandas frame as parquet, through
  its `pandas` extra, which this repo does not install). Record and replay in the same
  environment.
- The store is **not committed** (`tests/records/.gitignore`). `$LA_RECORDS_STORE`
  names its directory, else `$RECORDS_STORE` (what the agent gate sets), else the
  git-ignored `tests/records/store`. With no store, `tests/test_records.py` skips the
  replay (its serializer tests always run).

## In CI

The `Records` workflow (`.github/workflows/records.yml`) runs on every PR: it records the
scenarios on the PR's base commit, replays them on the head, and writes the summary to
the job summary. A base from before the harness has no "before", so it records on the
head instead. A PR that moves numbers on purpose carries the `records-break` label and
pastes the summary into its description. The private `regression-records` dependency is
fetched with `SIBLING_CHECKOUT_TOKEN`, a repository secret.

## Locally

Run these from the repository root:

```bash
export LA_RECORDS_STORE=~/regression-records/labelled-array   # any directory
git switch main
uv run python -m tests.records.cli record            # the "before"
git switch my-branch
uv run python -m tests.records.cli replay            # summary of what differs
uv run python -m tests.records.cli replay --save r.json --markdown r.md
```

Commit before recording, so the manifest names a clean commit. `record` never replaces a
record. If a checkpoint's signature changes, its records replay as `raised`: update
`scenarios.py` in the same PR. If one moves or is renamed, update `checkpoints.py`.
