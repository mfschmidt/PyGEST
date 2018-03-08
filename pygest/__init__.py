""" In __init__.py, we put commonly used items into the pygest namespace for easy usage.

Common usage is designed to look something like this:

   import pygest as ge

   data = ge.Data('/data')

"""

# Make convenience items accessible from the pygest namespace
from pygest.convenience import donors, donor_map
from pygest.convenience import richiardi_samples, richiardi_probes, richiardi_probe_names

# Make algorithms accessible from the pygest namespace
from pygest.algorithms import corr_expr_conn as corr

# Make plotting functions accessible from the pygest namespace
from pygest.plot import mantel_correlogram

# pygest also has a data manager class:
from pygest.data import ExpressionData as Data
# TODO: Rethink the capitalized Data. Lowercase data is a module that will not cause errors when imported, but when called.