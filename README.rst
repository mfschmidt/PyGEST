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

It also performs computations comparing gene expression filtered from ABI
with other matrices like functional connectivity.

Usage (python3)
---------------

This project is still in planning and initial development. Usage will
certainly change before the initial release.

Initialize with a path to BIDS structure.

.. code-block:: python
    
    import pygest as ge
    from pygest.reporting import sample_overview

    data = ge.Data('/home/mike/ge_data')

Now use any of the functions.

.. code-block:: python
    
    probes = data.probes()
    samples = data.samples()
    expr = data.expression()

    args = {"donor": "H03511009", "hemisphere": "L", "ctx": "all"}
    pdf_path = sample_overview(data, args, save_as="/home/mike/report.pdf")

Usage (bash)
------------

Data must be structured appropriately, see nonexistent future documentation. ;)

.. code-block:: bash

    pygest push --probes richiardi --samples cor --comparator /var/fcfile.df -v

Status
------

The current state of the project is immature. It is untested, in
development, and should not yet be used, other than for testing and
experimentation. Code that works today may not tomorrow, and vice-versa.

