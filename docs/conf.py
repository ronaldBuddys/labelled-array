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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "labelled-array"
author = "Ronald MacEachern"
release = "0.2.0"

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
napoleon_google_docstring = False
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
