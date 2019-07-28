import os
import logging
import argparse
import json

import pygest as ge
from pygest.plot import push_plot_via_dict
from pygest.convenience import path_to


def plots_from_json(json_file):
    """ Determine individual plots from json_file

    :param json_file: json-formatted text file containing plot characteristics
    :return: list of plot characteristics for individual plots
    """

    error_list = []
    plot_list = []
    valid_types = ["riser", ]

    with open(json_file) as f:
        j = json.load(f)

    for ptype in j:
        if ptype in valid_types:
            intra_variables = {}
            controls = {}
            title = ptype
            outdir = '.'
            filename = ptype
            if "title" in j[ptype]:
                title = j[ptype]["title"]
            if "outdir" in j[ptype]:
                outdir = j[ptype]["outdir"]
            if "filename" in j[ptype]:
                filename = j[ptype]["filename"]
            for iv in j[ptype]["intra-variables"]:
                intra_variables[iv] = j[ptype]["intra-variables"][iv]
            for ic in j[ptype]["controls"]:
                controls[ic] = j[ptype]["controls"][ic]
            for io in j[ptype]["inter-variables"]:
                for ind in j[ptype]["inter-variables"][io]:
                    if '.' in filename:
                        filename = "-".join([filename.split('.')[0], io, ind + '.' + filename.split('.')[1]])
                    else:
                        filename = "-".join([filename, io, ind + '.png'])
                    spec_controls = controls.copy()
                    spec_controls[io] = ind
                    plot_list.append({
                        "title": "{}: {} = {}".format(title, io, ind),
                        "outdir": outdir,
                        "filename": filename,
                        "type": ptype,
                        'intra': intra_variables,
                        'controls': spec_controls
                    })
        else:
            error_list.append("Invalid plot type: {}".format(ptype))

    for e in error_list:
        print("ERROR: {}".format(e))

    for p in plot_list:
        print("PLOT: ")
        print(p)
    return plot_list


def make_overview(data, arguments, logger):
    """ Save a pdf with multiple plots describing the exp and cmp data.

    :param pygest.ExpressionData data: PyGEST ExpressionData instance, already initialized
    :param arguments: Command line arguments from shell
    :param logging.Logger logger: logger object, if desired
    """

    parser = argparse.ArgumentParser(description="PyGEST status command")
    parser.add_argument("donor", default='all',
                        help="A donor if status data should be filtered.")
    args = parser.parse_args(arguments)

    base_path = path_to('overview', args)
    the_sample = "_".join([ge.donor_name(args.donor), args.splitby, args.samples])
    report_path = base_path.replace('derivatives', 'reports')
    # image_path = os.path.join(report_path, 'images')

    report_file = os.path.join(report_path, the_sample + '_overview.pdf')

    logger.info("Building an overview report for {} in {}".format(the_sample, report_path))

    report_file = ge.reporting.sample_overview(data, args, report_file, logger=logger)

    logger.info("See {} for completed report.".format(report_file))


def make_plot(json_file, data, logger=None):
    """
    Read description of plot requested from jsonfile and do as it specifies.

    :param json_file: json-formatted file describing the desired plot characteristics
    :param data: PyGEST data object for access to all it holds
    :param logger: python logger for giving feedback to user
    :return: 0 for success, integer error code for failure
    """

    if os.path.isfile(json_file):
        plots = plots_from_json(json_file)
        for plot in plots:
            push_plot_via_dict(data, plot)
    else:
        msg = "{} does not exist. Giving up on making a plot.".format(json_file)
        if logger is None:
            print(msg)
        else:
            logger.warning(msg)
        return 1
