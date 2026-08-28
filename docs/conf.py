"""Sphinx configuration.

The docs are a handful of hand-written pages plus one Sphinx-Gallery build over
``examples/``. Sphinx-Gallery turns each script into an HTML page with downloadable
``.py`` and ``.ipynb``; whether it also *executes* the script is controlled by the
``DOCS_RUN_EXAMPLES`` environment variable (default ``1`` here -- see
``building.rst``). When examples run, ``abort_on_example_error`` makes a broken example a
build failure rather than a page with a traceback on it.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "labelled-array"
author = "Ronald MacEachern"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

exclude_patterns = ["_build"]
html_theme = "furo"
html_title = project

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Objects are re-exported from the package root, so a bare :class:`X` reference has two
# valid targets. That is not an error worth failing the build for.
suppress_warnings = ["ref.python"]

RUN_EXAMPLES = os.environ.get("DOCS_RUN_EXAMPLES", "1") == "1"

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    # The sphinx-gallery default is "/plot_", which would skip every script here.
    "filename_pattern": r".*\.py",
    "plot_gallery": RUN_EXAMPLES,
    "image_scrapers": (),
    "reset_modules": (),
    "download_all_examples": False,
    "abort_on_example_error": True,
    "remove_config_comments": True,
    "within_subsection_order": "FileNameSortKey",
}


_UNDERLINE = re.compile(r"^\s*-{3,}\s*$")


def _strip_section_colons(app, what, name, obj, options, lines):
    """Let napoleon see ``Notes:`` over a dashed underline as a numpy section.

    The docstrings follow the Google convention ruff enforces (a section name ends
    with a colon) but underline the name numpy-style. Neither napoleon parser accepts
    the mix, and docutils then reports the dashes as a too-short title. Dropping the
    colon where an underline follows makes it a plain numpy section.
    """
    for i in range(len(lines) - 1):
        if lines[i].rstrip().endswith(":") and _UNDERLINE.match(lines[i + 1]):
            lines[i] = lines[i].rstrip()[:-1]


def setup(app):
    """Register the docstring hook ahead of napoleon (priority 500)."""
    app.connect("autodoc-process-docstring", _strip_section_colons, priority=400)
