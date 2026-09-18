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

   uv sync --extra pandas --extra polars   # DataFrame round trips
   uv sync --extra jax --extra torch       # alternative array backends

Or add it to another ``uv`` project from GitHub:

.. code-block:: bash

   uv add "git+https://github.com/ronaldBuddys/labelled-array"

First use
---------

Most arrays are reached from a DataFrame: some columns identify a cell, the rest become
variables.

.. code-block:: python

   import pandas as pd
   from labelled_array import LabelledArray

   df = pd.DataFrame(
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
   sales.to_dataframe(index=False)              # back to the original four rows

A cell with no source row keeps the column's dtype and is recorded in ``sales.mask``
rather than widened to float or given a sentinel; ``to_dataframe`` emits only the
masked-in cells.

Where next
----------

* :doc:`auto_examples/index` -- ten scripts, from construction through selection,
  sparsity, backends, subclassing, combining, forward-filling and alternative
  dimensions.
* :doc:`api` -- ``LabelledArray`` and everything around it.
