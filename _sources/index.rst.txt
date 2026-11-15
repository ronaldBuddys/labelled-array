labelled-array
==============

N-dimensional arrays that carry their dimension names and labels, align and reduce over
them, convert losslessly to and from pandas or polars DataFrames, and hold their data on
numpy, JAX or torch.

A ``LabelledArray`` is one or more equally shaped **variables** plus an ordered mapping of
dimension name to labels, a presence **mask** that records which cells had a source row,
optional **alt_dims** and **coords**, and free-form ``attrs``. Selection is by label, the
shape of a result never depends on how many labels were asked for, and every operation
that returns a new array builds it through ``type(self)`` so subclasses keep their type.

The examples are executed at build time, so every number on those pages was produced by
the code, not typed in.

.. toctree::
   :maxdepth: 2

   getting_started
   auto_examples/index
   api
   building


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
