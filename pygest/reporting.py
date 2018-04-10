import os

import numpy as np
import seaborn as sns

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from pygest import plot


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
