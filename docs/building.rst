Building the docs locally
=========================

The docs are built with Sphinx from a clone of the repository. ``uv`` installs
everything, including the ``docs`` extra that carries Sphinx, Sphinx-Gallery and the
theme:

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

Adding an example
-----------------

Drop a script in ``examples/``. It must start with a module docstring whose first line
is an rST title with an underline; ``# %%`` comment cells split it into prose and code.
Files starting with an underscore are shared helpers and are skipped by the gallery.
