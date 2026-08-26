"""The example scripts must keep running, so they cannot fall out of date."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.py"))


def test_the_examples_are_present():
    assert len(EXAMPLES) >= 6


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.stem)
def test_example_runs(path, capsys):
    """Run the script end to end; its inline assertions are the test."""
    runpy.run_path(str(path), run_name="__main__")
    capsys.readouterr()


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.stem)
def test_example_is_written_in_the_sphinx_gallery_style(path):
    lines = path.read_text().splitlines()
    assert lines[0] == '"""', "an example starts with a module docstring"
    assert set(lines[2]) == {"="}, "with an underlined rst title"
    assert any(line.startswith("# %%") for line in lines), "and is split into cells"
    for number, line in enumerate(lines):
        if line.startswith("# %%"):
            title, underline = lines[number + 1], lines[number + 2]
            assert title.startswith("# "), (
                f"{path.name}:{number + 2} cell needs a title"
            )
            assert set(underline.removeprefix("# ")) == {"-"}, (
                f"{path.name}:{number + 3} cell title needs an rst underline"
            )
