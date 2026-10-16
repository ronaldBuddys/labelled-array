"""Record the checkpoints on small seeded arrays, or replay the store.

Run from the repository root::

    uv run python -m tests.records.cli record   # write new records, and a manifest
    uv run python -m tests.records.cli replay   # print the replay summary

The store is ``$LA_RECORDS_STORE`` (else ``$RECORDS_STORE``, else the git-ignored ``tests/records/store``).
``record`` never replaces a record: unchanged behavior writes nothing, and changed
behavior adds a record beside the old one. See ``tests/records/README.md``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from regression_records import recording, replay, session
from regression_records.manifest import load_manifest
from tests.records import scenarios
from tests.records.checkpoints import STORE, STORE_ENV, TOLERANCE, installed

REPO = Path(__file__).resolve().parents[2]

#: Recorded in each manifest: what produced the numbers, beside labelled-array itself.
PACKAGES = ("labelled-array", "numpy", "polars", "pandas", "regression-records")


def record() -> None:
    """Run every scenario with recording on, inside a session."""
    with (
        installed(enabled=True),
        recording(),
        session(
            STORE,
            context={"data": "seeded synthetic arrays"},
            # Relative, so a manifest names no machine's home directory.
            repos={"labelled-array": os.path.relpath(REPO)},
            packages=PACKAGES,
        ) as run,
    ):
        scenarios.run_all()
    manifest = load_manifest(run.manifest_path)
    print(f"manifest {run.manifest_path}: {manifest['counts']}")
    dirty = manifest["environment"]["repos"]["labelled-array"]["dirty"]
    if dirty:
        print("note: labelled-array has uncommitted changes; commit before recording")


def replay_store(save: Path | None, markdown: Path | None) -> bool:
    """Replay the store, print the summary, and optionally save it.

    Parameters
    ----------
    save : Path or None
        Where to write the full report as JSON.
    markdown : Path or None
        Where to write the summary as a fenced block, e.g. for a CI job summary.

    Returns
    -------
    bool
        Whether every record replayed as recorded.

    Raises
    ------
    FileNotFoundError
        If the store has no records: replaying nothing would pass vacuously, which in
        CI would hide a failed recording on the base commit.
    """
    if not STORE.iter_record_paths():
        raise FileNotFoundError(
            f"no regression records in {STORE.root}; set {STORE_ENV} or run `record`"
        )
    with installed(enabled=False):
        report = replay(STORE, tolerance=TOLERANCE)
    summary = report.summary()
    print(summary)
    if save is not None:
        report.save(save)
        print(f"saved {save}")
    if markdown is not None:
        verdict = "all records match" if report.passed else "numbers moved"
        markdown.write_text(
            f"### Regression records: {verdict}\n\n```\n{summary}\n```\n"
        )
    return report.passed


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("record", help="record every scenario")
    replay_parser = commands.add_parser("replay", help="replay the store")
    replay_parser.add_argument(
        "--save", type=Path, help="also write the report as JSON here"
    )
    replay_parser.add_argument(
        "--markdown", type=Path, help="also write the summary as markdown here"
    )
    args = parser.parse_args(argv)
    if args.command == "record":
        record()
        return 0
    return 0 if replay_store(args.save, args.markdown) else 1


if __name__ == "__main__":
    raise SystemExit(main())
