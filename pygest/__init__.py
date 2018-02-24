import pygest

# The preferred way to use this would be to import as follows:
#
#     from AllenHumanBrainGeneExpression import ExpressionData as abe
#
# then, to obtain an initialized wrapper:
#
#     data = abe.ExpressionData('/home/me/my_data_dir')

# In this file, expose everything we want users to have easy access to.
#    If someone imports this, they have access to everything in this file's namespace.
#
#    import AllenHumanBrainGeneExpression as ge
#
#    ge.probes()


# Try this:

#    import pygest as ge

#    data = ge.data('/data')
#    ge.ops.whack-a-gene(data.expression('2002'))
#    ge.vis.mantel(data.expression('2002'), by=distance)
