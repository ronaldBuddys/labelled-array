Building the docs locally
=========================

The docs are built with Sphinx from a clone of the repository. ``uv`` installs
everything, including the ``docs`` extra (Sphinx, Sphinx-Gallery, the theme,
``sphinxcontrib-bibtex`` for citations and matplotlib for figures):

.. code-block:: bash

   uv sync --extra docs
   uv run make -C docs html
   open docs/_build/html/index.html

``make html`` passes ``-W`` to ``sphinx-build``, so a warning -- a broken cross-reference,
malformed reStructuredText in a docstring, a gallery script without a title -- fails the
build instead of being buried in the log. CI runs the same command on every pull request.

Running the examples
--------------------

Examples execute at build time by default: nothing here needs data that is not in the
repository. ``DOCS_RUN_EXAMPLES=0`` renders them without running, which is faster and
is what a docs-only change wants.

.. code-block:: bash

   DOCS_RUN_EXAMPLES=1 uv run make -C docs html   # execute every example
   DOCS_RUN_EXAMPLES=0 uv run make -C docs html   # render only

Everything Sphinx-Gallery generates (``docs/auto_examples/``,
``docs/sg_execution_times.rst``) and the HTML under ``docs/_build/`` is ignored by git.
``make -C docs clean`` removes all of it.

Network access
--------------

Intersphinx links numpy and Python names to their own documentation, which means the
build downloads their inventories. Under ``-W`` a failed download fails the build, so
offline, build with ``DOCS_INTERSPHINX=0``: everything still builds, and those names are
simply not linked.

Adding an example
-----------------

Drop a script in one of the subsections of ``examples/`` -- ``basics/``, ``layout/``,
``operations/``, ``interop/`` or ``performance/`` -- with a two-digit prefix that places it
in the reading order. It must start with a module docstring holding an anchor and an rST
title with an underline. The title reads ``<what is used> : <prose>``, with each function,
class or module name as an inline literal so it renders as code, for example::

   ``select`` : selecting by label
   ===============================

The underline is at least as long as the title, backticks included. The page heading
and the table of contents show the name as code, and the gallery thumbnails and browser
tab show the plain text. ``# %%`` comment cells split the script into prose and code.
Files starting with an underscore are shared helpers and are skipped by the gallery.

Checks *assert* (``np.testing.assert_allclose`` with explicit tolerances, or an exact
comparison) rather than print, so a page that stops matching the code fails the build.
Timings belong on the performance page only. A page that states theory cites it with
``:footcite:`key``` from ``docs/references.bib`` and ends with ``.. footbibliography::``;
add a bibliography entry only after checking every field against the publisher or a DOI
resolver.
