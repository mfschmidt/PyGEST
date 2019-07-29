""" In __init__.py, we put commonly used items into the pygest namespace for easy usage.

Common usage is designed to look something like this:

   import pygest as ge

   data = ge.Data('/home/user/ge_data')

"""

# Lists and dictionaries are available in rawdata/*.py
from pygest import rawdata

# Make convenience items accessible from the pygest namespace
from pygest.convenience import donor_name

# Make algorithms accessible from the pygest namespace
from pygest.algorithms import correlate as corr

# Make plotting functions accessible from the pygest namespace
from pygest.plot import mantel_correlogram
from pygest.plot import heat_map

# Make reporting functions accessible from the pygest namespace
from pygest import reporting

# pygest also has a data manager class:
from pygest.data import ExpressionData as Data
