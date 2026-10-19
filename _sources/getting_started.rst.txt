Getting started
===============

Install
-------

.. code-block:: bash

   git clone https://github.com/ronaldBuddys/labelled-array
   cd labelled-array
   uv sync --extra docs              # numpy core, plus what these docs need

The DataFrame adapters and array backends are extras, so a consumer that only wants
numpy installs nothing else:

.. code-block:: bash

   uv sync --extra polars                  # DataFrame round trips
   uv sync --extra pandas                  # ... or through pandas, or both
   uv sync --extra jax --extra torch       # alternative array backends

Or add it to another ``uv`` project from GitHub:

.. code-block:: bash

   uv add "git+https://github.com/ronaldBuddys/labelled-array"

First use
---------

Most arrays are reached from a DataFrame: some columns identify a cell, the rest become
variables.

.. code-block:: python

   import polars as pl
   from labelled_array import LabelledArray

   df = pl.DataFrame(
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

   sales.shape                                  # (2, 3)
   sales.dims["product"].tolist()               # ['AAA', 'BBB', 'CCC']
   sales.density                                # 0.666... -- ('south', 'BBB') has no row
   sales.select({"region": "north"})            # length-1 'region' dimension
   sales.select({"region": "north"}, drop=True) # 'region' removed
   sales.to_dataframe()                         # back to the original four rows
   sales.to_dataframe("pandas")                 # the same, as a pandas frame

A cell with no source row keeps the column's dtype and is recorded in ``sales.mask``
rather than widened to float or given a sentinel; ``to_dataframe`` emits only the
masked-in cells.

A pandas frame goes into ``from_dataframe`` exactly as a polars one does, and builds
the same array. ``to_dataframe`` writes polars unless asked for ``"pandas"``, falling
back to pandas only where polars is not installed.

Where next
----------

* :doc:`auto_examples/index` -- thirteen scripts, from construction through selection,
  sparsity, backends, subclassing, combining, forward-filling and alternative
  dimensions, to the presence mask end to end, what each operation does to ordering,
  and polars and pandas side by side.
* :doc:`api` -- ``LabelledArray`` and everything around it.
