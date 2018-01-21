==============================================================================
A wrapper for human brain gene expression data from the Allen Brain Institute
==============================================================================

Brief Description
-----------------

This project serves first as a wrapper to the data. It tracks whether data
have been downloaded to local disk, read into memory, etc. and returns
pandas dataframes of requested parts of the dataset. If the data are in
memory, a reference to the dataframe is returned very quickly. If not, it
may take some time to load from disk, or download from brain-map.org first.

Usage
-----

This project is still in planning and initial development. Usage will
certainly change before the initial release.

Status
------

The current state of the project is very very immature. It is untested, in
development, and should not yet be used.
