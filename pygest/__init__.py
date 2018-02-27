# In this file, we expose everything we want users to have easy access to.

# Common usage is designed to look something like this:

#    import pygest as ge

#    data = ge.Data('/data')
#    ge.Algo.whack-a-gene(data.expression('2002'))
#    ge.Plot.mantel_plot(data.expression('2002'), by=distance)

from pygest.config import donors, donor_map

# PyGEST consists of three modules:
#    data for accessing and manipulating the underlying ABI expression data
#    algo for processing, manipulating, and investigating the data
#    plot for visualizing the data
from pygest.data import ExpressionData as data
from pygest.algorithms import Algorithms as algo
from pygest.plot import Plot as plot
from pygest.convenience import Convenience as conv
