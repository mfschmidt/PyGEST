""" In __init__.py, we put commonly used items into the pygest namespace for easy usage.

Common usage is designed to look something like this:

   import pygest as ge

   data = ge.Data('/home/user/ge_data')

"""

# Make convenience items accessible from the pygest namespace
from pygest.convenience import donor_name
from pygest.convenience import richiardi_samples, richiardi_probes, richiardi_probe_names
from pygest.convenience import fornito_samples, fornito_probes, fornito_probe_names

# Make algorithms accessible from the pygest namespace
from pygest.algorithms import correlate as corr

# Make plotting functions accessible from the pygest namespace
from pygest.plot import mantel_correlogram
from pygest.plot import expr_heat_map

# Make reporting functions accessible from the pygest namespace
from pygest import reporting

# pygest also has a data manager class:
from pygest.data import ExpressionData as Data
