import os
import logging
import humanize

import numpy as np
import pandas as pd
import seaborn as sns

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from pygest import plot
from pygest.convenience import donor_name


def generate_pdf(save_as, args, images, strings, logger=None):
    """ Build a pdf about the process of analyzing these data

    :param str save_as: The full path to the requested pdf file
    :param args: Original command line arguments
    :param dict images: A dictionary of pre-built png paths to insert into the pdf
    :param dict strings: A dictionary of pre-built strings to place on the pdf
    :param logging.Logger logger: If provided, updates will be logged here
    :return: the path of the pdf, same as save_as
    """

    # First, ensure we have the files necessary to build this thing.
    # Where do the lists live?
    report_name = "{}'s {} hemisphere".format(args.donor, args.hemisphere)
    if not os.path.isdir(os.path.dirname(save_as)):
        logger.warning("{} is not a valid directory. I cannot build a pdf there.".format(
            os.path.dirname(save_as)
        ))
        return save_as
    else:
        logger.info("The pdf will be saved as {}.".format(
            save_as
        ))

    # Create a letter-size paper with half-inch margins and a reasonable font
    page_width, page_height = landscape(letter)
    margin = inch / 2.0
    c = canvas.Canvas(save_as, pagesize=landscape(letter))
    c.setFont("Helvetica", 14)
    c.setStrokeColorRGB(0.0, 0.0, 0.0)
    c.setFillColorRGB(0.0, 0.0, 0.0)

    # PLay with the size of some title and subtitle stuff.
    c.setFont("Helvetica", 24)
    c.drawCentredString(page_width / 2.0, page_height - margin - 24, report_name)
    c.setFont("Helvetica", 18)
    c.drawCentredString(page_width / 2.0, page_height - margin - 44,
                        "Eventually, they will be real scatters")
    c.setFont("Helvetica", 12)
    c.drawCentredString(page_width / 2.0, page_height - margin - 58,
                        "The correlogram will need to find a home here, too.")
    c.setFont("Courier", 10)

    c.setStrokeColorRGB(0.7, 0.7, 0.7)
    c.setFillColorRGB(0.7, 0.7, 0.7)
    c.rect(margin, page_height - margin - 72, (page_width - (margin * 2.0)) / 2.0, 12, fill=1)
    c.setStrokeColorRGB(0.0, 0.0, 0.0)
    c.setFillColorRGB(0.0, 0.0, 0.0)
    c.drawString(margin, page_height - margin - 70, "{} probes".format(strings['n']))

    # And stick some images on the page.
    w = (page_width - (margin * 2.0)) / 5
    h = w
    row_y = page_height - margin - 100 - h * 2
    if 'heat_expr' in images and os.path.isfile(images['heat_expr']):
        c.drawImage(images['heat_expr'], margin, row_y, w, h * 2)
    if 'dist_expr' in images and os.path.isfile(images['dist_expr']):
        c.drawImage(images['dist_expr'], margin + w, row_y, w, h)
    if 'dist_simi' in images and os.path.isfile(images['dist_simi']):
        c.drawImage(images['dist_simi'], margin + (2 * w), row_y, w, h)
    if 'dis2_expr' in images and os.path.isfile(images['dis2_expr']):
        c.drawImage(images['dis2_expr'], margin + (2 * w), row_y + h, w, h)

    bottom_row = ['dist_conn', 'dis2_conn', 'dist_dens', 'dist_dist', 'dis2_dist']
    for i, img in enumerate(bottom_row):
        if img in images and os.path.isfile(images[img]):
            c.drawImage(images[img], margin + (w * i), margin, w, h)

    c.save()

    return save_as


def sample_overview(data, args, save_as, logger=None):
    """ Build a pdf with distributions, heatmaps, and correlograms for a given sample.

    :param pygest.ExpressionData data: An initialized instance of ExpressionData
    :param args: A dictionary from argparse with expected donor, hemisphere, etc. values
    :param save_as str save_as: The file name for the resultant pdf
    :param logging.Logger logger: A logger to handle our non-file output and commentary
    :return: the path to the pdf file written
    """

    img_dir = os.path.join(os.path.dirname(save_as), 'images')
    images = {}

    # The very first step is to slice and dice the data into usable structures.
    expr = data.expression(
        probes='richiardi',
        samples=data.samples(donor=args.donor, hemisphere=args.hemisphere)
    )
    if args.samples[:3] == 'cor':
        expr = expr[[well_id for well_id in data.samples('richiardi').index if well_id in expr.columns]]

    conn = data.connectivity()

    overlapping_samples = [well_id for well_id in conn.index if well_id in expr.columns]

    dist_mat = data.distance_matrix(overlapping_samples)
    dist_vec = dist_mat[np.tril_indices(n=dist_mat.shape[0], k=-1)]

    conn_mat = conn.loc[overlapping_samples, overlapping_samples].as_matrix()
    conn_vec = conn_mat[np.tril_indices(n=conn_mat.shape[0], k=-1)]

    expr_raw = expr.loc[:, overlapping_samples].as_matrix().flatten()
    expr_mat = np.corrcoef(expr.loc[:, overlapping_samples], rowvar=False)
    expr_vec = expr_mat[np.tril_indices(n=expr_mat.shape[0], k=-1)]

    # Save out some strings describing our data
    strings = {
        "n": "{}".format(len(expr.index))
    }

    # First, generate distribution plots for each piece of data
    for vector in [('Connectivity', conn_vec, "b"),
                   ('Density of connectivity', data.connection_density(samples=overlapping_samples), "b"),
                   ('Distance', dist_vec, "g"),
                   ('Similarity (Expression)', expr_vec, "r"),
                   ('Expression', expr_raw, "r")]:
        logger.info("   -building {} distribution plot with {:,} values.".format(
            vector[0].lower(), len(vector[1])
        ))
        filename = "_".join([
            args.donor, args.hemisphere, args.samples[:3], vector[0][:4].lower(), 'distribution.png'
        ])
        sns.set(style="white", palette="muted", color_codes=True)
        fig, ax = plot.distribution_plot(vector[1], c=vector[2])
        ax.set_title(vector[0] + " distribution")
        fig.savefig(os.path.join(img_dir, filename))
        images['dist_' + vector[0][:4].lower()] = os.path.join(img_dir, filename)

    logger.info("  -generating {} x {} heat map".format(expr.shape[0], expr.shape[1]))
    fig, ax = plot.expr_heat_map(expr, fig_size=(4, 8), c_map="Reds", logger=logger)
    ax.set_title("Expression heat map")
    fig.savefig(os.path.join(img_dir, '_expr_heat.png'))
    images['heat_expr'] = os.path.join(img_dir, '_expr_heat.png')

    # Then, the expression distribution
    logger.info("  -generating distribution plot for expression")
    expr_mat = np.corrcoef(expr, rowvar=False)
    expr_vec = expr_mat[np.tril_indices(expr_mat.shape[0], k=-1)]
    fig, ax = plot.distribution_plot(expr_vec, c="red", logger=logger)
    ax.set_title("Expression similarity distribution 2")
    fig.savefig(os.path.join(img_dir, '_expr_distro.png'))
    images['dis2_expr'] = os.path.join(img_dir, '_expr_distro.png')

    # Then, the expression similarity matrix and distribution

    # Then, the comparator distribution
    logger.info("  -generating distribution plot for connectivity")
    conn_mat = conn.as_matrix()
    conn_vec = conn_mat[np.tril_indices(conn_mat.shape[0], k=-1)]
    fig, ax = plot.distribution_plot(conn_vec, c="blue", logger=logger)
    ax.set_title("Connectivity distribution 2")
    fig.savefig(os.path.join(img_dir, '_conn_distro.png'))
    images['dis2_conn'] = os.path.join(img_dir, '_conn_distro.png')

    # Then, distance distribution
    logger.info("  -generating distribution plot for distance")
    dist_mat = data.distance_matrix(expr.columns)
    dist_vec = dist_mat[np.tril_indices(n=dist_mat.shape[0], k=-1)]
    fig, ax = plot.distribution_plot(dist_vec, c="green", logger=logger)
    ax.set_title("Distance distribution 2")
    fig.savefig(os.path.join(img_dir, '_dist_distro.png'))
    images['dis2_dist'] = os.path.join(img_dir, '_dist_distro.png')

    # Then, scatterplots for expr vs cmp, with densities

    # For entire sample, and for discrete distance bins:
    #     generate scatterplot of expr vs cmp, with densities
    #     generate correlogram of correlation by bin

    # Create a pdf template and fill it with above graphics.

    return generate_pdf(save_as, args, images, strings, logger)


def log_status(data, root_dir, regarding='all', logger=None):
    """ Return a brief summary of the data available (or not)

    :param pygest.ExpressionData data: PyGEST ExpressionData instance, already initialized
    :param root_dir: the root directory containing pygest results and reports
    :param regarding: allows status to be tailored to the caller's interest - only 'all' coded thus far
    :param logging.Logger logger: A logger to handle our non-file output and commentary
    """

    if logger is None:
        logger = logging.getLogger('pygest')

    if not os.path.isdir(root_dir):
        logger.warning("{} is not a directory; no way to provide a status.".format(root_dir))
        return
    if os.path.isdir(os.path.join(root_dir, 'derivatives')):
        root_dir = os.path.join(root_dir, 'derivatives')

    if regarding == 'all':
        donors_of_interest = data.donors()
    else:
        donors_of_interest = [donor_name(regarding), ]

    logger.info("  scanning {} for results and reports from {}...".format(root_dir, regarding))

    # Summaries
    tsv_files = pd.DataFrame(
        columns=['donor', 'path', 'file', 'bytes', 'hem', 'ctx', 'alg', 'tgt', 'cmp', 'nul', ],
        data=[]
    )
    other_files = pd.DataFrame(
        columns=['donor', 'path', 'file', 'bytes', 'hem', 'ctx', 'alg', 'tgt', 'cmp', 'nul', ],
        data=[]
    )
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if name[-4:] == ".tsv":
                # Extract what we can from the filename
                remainder = name[name.find("sub-") + 4: name.rfind(".")]
                don = remainder[:remainder.find("_")]
                if remainder.find("NULL") > 0:
                    null_seed = int(remainder[remainder.find("NULL") + 4:])
                    remainder = remainder[remainder.find("_") + 1: remainder.find("NULL") - 1]
                else:
                    null_seed = 0
                    remainder = remainder[remainder.find("_") + 1: remainder.find(".") + 1]
                cmp = remainder[remainder.find("cmp-") + 4:]

                # Extract what we can from the path
                remainder = root[root.find("_hem-") + 5:]
                hem = remainder[:remainder.find("_")]
                ctx = remainder[remainder.find("ctx-") + 4: remainder.find(os.sep)]
                alg = remainder[remainder.rfind("_alg-") + 5:]
                tgt = remainder[remainder.rfind("tgt-") + 4: remainder.rfind("_alg-")]
                this_tsv = {'donor': don, 'path': root, 'file': name,
                            'bytes': os.stat(os.path.join(root, name)).st_size,
                            'hem': hem, 'ctx': ctx, 'alg': alg, 'tgt': tgt,
                            'cmp': cmp, 'nul': null_seed}
                tsv_files = tsv_files.append(this_tsv, ignore_index=True)

                json_name = name.replace("tsv", "json")
                if os.path.isfile(os.path.join(root, json_name)):
                    this_json = {'donor': don, 'path': root, 'file': json_name,
                                 'bytes': os.stat(os.path.join(root, json_name)).st_size,
                                 'hem': hem, 'ctx': ctx, 'alg': alg, 'tgt': tgt,
                                 'cmp': cmp, 'nul': null_seed}
                    other_files = other_files.append(this_json, ignore_index=True)

                log_name = name.replace("tsv", "log")
                if os.path.isfile(os.path.join(root, json_name)):
                    this_log = {'donor': don, 'path': root, 'file': log_name,
                                'bytes': os.stat(os.path.join(root, log_name)).st_size,
                                'hem': hem, 'ctx': ctx, 'alg': alg, 'tgt': tgt,
                                'cmp': cmp, 'nul': null_seed}
                    other_files = other_files.append(this_log, ignore_index=True)

    all_files = pd.concat([tsv_files, other_files], axis=0)
    all_files_of_interest = all_files.loc[all_files['donor'].isin(donors_of_interest)]
    tsv_files_of_interest = tsv_files.loc[tsv_files['donor'].isin(donors_of_interest)]
    logger.info("  {nf:,} file{p1} ({nr:,} analyses) from {d} donor{pd} consume{p3} {b}.".format(
        nf=all_files_of_interest['bytes'].count(),
        nr=len(tsv_files_of_interest),
        d=all_files['donor'].nunique(),
        p1="" if len(all_files_of_interest) == 1 else "s",
        pd="" if all_files['donor'].nunique() == 1 else "s",
        p3="s" if len(all_files_of_interest) == 1 else "",
        b=humanize.naturalsize(all_files_of_interest['bytes'].sum())
    ))

    # And, finally, build a grid of which portions are completed.
    def six_char_summary(df_all, donor, algo, hemi):
        df = df_all[(df_all['donor'] == donor) & (df_all['alg'] == algo)]
        s = " "
        for cort in ['cor', 'sub', 'all']:
            for mnmx in ['max', 'min']:
                relevant_filter = (df['hem'] == hemi) & (df['ctx'] == cort) & (df['tgt'] == mnmx)
                real_result_filter = relevant_filter & (df['nul'] == 0)
                null_distro_filter = relevant_filter & (df['nul'] > 0)
                nr = len(df[real_result_filter])
                nn = len(df[null_distro_filter])
                s += "{}{}{} ".format(
                    " " if nr == 0 else "{:01}".format(nr),
                    " " if nr == 0 or nn == 0 else "+",
                    "  " if nn == 0 else "{:02}".format(nn)
                )
        return s

    logger.info("    hemisphere      |{ss}Left{ss} |{ss}Right{ss}|{ss} All {ss}|".format(
        ss="             "
    ))
    logger.info("    cortical        |{cort_types}|{cort_types}|{cort_types}|".format(
        cort_types="    cor       sub       all    "
    ))
    logger.info("    minmax          |{plus_minus}|{plus_minus}|{plus_minus}|".format(
        plus_minus="  +    -    +    -    +    -   "
    ))
    logger.info("    ----------------+{dashes}+{dashes}+{dashes}|".format(
        dashes="-------------------------------"
    ))
    template_string = "    {d} ({a})|{l_vals}|{r_vals}|{a_vals}|"
    for d in data.donors():
        for a in tsv_files_of_interest['alg'].unique():
            logger.info(template_string.format(
                d=d, a=a,
                l_vals=six_char_summary(tsv_files_of_interest, d, a, 'L'),
                r_vals=six_char_summary(tsv_files_of_interest, d, a, 'R'),
                a_vals=six_char_summary(tsv_files_of_interest, d, a, 'A'),
            ))
