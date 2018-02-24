===============================================================================
PyGEST: Python Gene Expression Spatial Toolkit
===============================================================================

Brief Description
-----------------

This project serves first as a wrapper to the Allen Brain Institute's human
gene expression data. It tracks whether data have been downloaded to local
disk, read into memory, etc. and returns pandas dataframes of requested parts
of the dataset. If the data are in memory, a reference to the dataframe is
returned very quickly. If not, it may take some time to load from disk, or
download from brain-map.org first.

Usage
-----

This project is still in planning and initial development. Usage will
certainly change before the initial release.

Initialize with a path to BIDS structure.

.. code-block:: python
    
    import pygest as ge
    data = ge.data('/data')

Now use any of the functions.

.. code-block:: python
    
    ge.data.download('all')
    ge.data.extract('all')

    probes = data.probes()
    samples = data.samples()
    expr = data.expression()

Status
------

The current state of the project is very very immature. It is untested, in
development, and should not yet be used, other than for testing and
experimentation. Code that works today may not tomorrow.

