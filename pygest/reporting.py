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
from pygest.convenience import donor_name, bids_val


def generate_pdf(save_as, args, images, strings, logger=None):
    """ Build a pdf about the process of analyzing these data

    :param str save_as: The full path to the requested pdf file
    :param args: Original command line arguments
    :param dict images: A dictionary of pre-built png paths to insert into the pdf
    :param dict strings: A dictionary of pre-built strings to place on the pdf
    :param logging.Logger logger: If provided, updates will be logged here
    :return: the path of the pdf, same as save_as
    """

    # safe image placement
    def place_image(name, x, y, w, h):
        if name in images and os.path.isfile(images[name]):
            c.drawImage(images[name], x, y, w, h)
        else:
            c.rect(x, y, w, h, fill=0)
            c.drawCentredString(x + (w / 2), y + (h / 2), "[missing {}]".format(name))

    # First, ensure we have the files necessary to build this thing.
    # Where do the lists live?
    report_name = "{}, {} hemisphere, {}".format(args.donor, args.hemisphere, args.samples)
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
    gap = margin / 2.0

    c = canvas.Canvas(save_as, pagesize=landscape(letter))
    c.setFont("Helvetica", 14)
    c.setStrokeColorRGB(0.0, 0.0, 0.0)
    c.setFillColorRGB(0.0, 0.0, 0.0)

    # PLay with the size of some title and subtitle stuff.
    c.setFont("Helvetica", 24)
    c.drawCentredString(page_width / 2.0, page_height - margin - 24, report_name)
    c.setTitle(report_name)
    c.setFont("Helvetica", 14)
    c.drawCentredString(page_width / 2.0, page_height - margin - 44,
                        "[ " + strings['total_probes'] + " x " + strings['total_samples'] + " ]")
    # c.setFont("Helvetica", 12)
    # c.drawCentredString(page_width / 2.0, page_height - margin - 58,
    #                     "The correlogram will need to find a home here, too.")
    # c.setFont("Courier", 10)

    # And stick some images on the page.
    width = 2.25 * inch
    height = 3.0 * inch
    row_y = page_height - margin - (24 + 14) - height - gap
    place_image('heat_exps', margin + (0 * (gap + width)), row_y, width, height)
    place_image('dist_cmbo', margin + (1 * (gap + width)), row_y, width, height)
    place_image('conn_simi_cmbo', margin + (2 * (gap + width)), row_y, width, height)
    place_image('conn_cmbo', margin + (3 * (gap + width)), row_y, width, height)
    # place_image('dis2_expr', margin + (2 * width), row_y + height, width, height)

    row_y = margin
    place_image('heat_cmbo', (1 * margin), row_y, width, height * 1.25)
    # c.rect(margin + (1 * (gap + width)), row_y, (width * 2) + gap, height * 1.25, fill=0)
    # c.drawCentredString(margin + (1 * (gap + width)) + (width * 2 + gap) / 2, row_y + height * 0.625,
    #                     "Future home of whack-a-probe curve")
    place_image('push_corr_' + args.samples, margin + (1 * (gap + width)), row_y, (width * 2) + gap, height * 1.25)
    place_image('conn_dens', margin + (3 * (gap + width)), row_y + (height * 1.25 - width), width, width)

    # A few arrows may help to guide the eye to the proper flow
    """
    arr_w = inch / 10.0
    arr_h = inch / 8.0
    p = c.beginPath()
    p.moveTo(margin + width / 2, margin + height)
    p.lineTo(margin + width / 2, margin + height + margin)
    p.lineTo(margin + width / 2 - arr_w, margin + height + margin - arr_h)
    p.lineTo(margin + width / 2, margin + height + margin)
    p.lineTo(margin + width / 2 + arr_w, margin + height + margin - arr_h)
    p.lineTo(margin + width / 2, margin + height + margin)
    c.drawPath(p)
    """

    """ NEW PAGE """
    c.showPage()

    w = (page_width / 2) - margin
    h = (page_height / 2) - margin
    place_image('mantel_expr_dist', margin, (page_height / 2), w, h)
    place_image("mantel_expr_conn", (page_width / 2), (page_height / 2), w, h)
    place_image("mantel_conn_dist", margin, margin, w, h)
    place_image("mantel_expr_cons", (page_width / 2), margin, w, h)

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

    if logger is None:
        logger = logging.getLogger('pygest')

    img_dir = os.path.join(os.path.dirname(save_as), 'images')
    os.makedirs(img_dir, exist_ok=True)
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

    # expr_raw = expr.loc[:, overlapping_samples].as_matrix().flatten()
    expr_mat = np.corrcoef(expr.loc[:, overlapping_samples], rowvar=False)
    expr_vec = expr_mat[np.tril_indices(n=expr_mat.shape[0], k=-1)]

    conn_dens = data.connection_density(samples=overlapping_samples)
    conn_simi_mat = data.connectivity_similarity(samples=overlapping_samples).as_matrix()
    conn_simi_vec = conn_simi_mat[np.tril_indices(n=conn_simi_mat.shape[0], k=-1)]

    # Save out some strings describing our data
    strings = {
        "total_probes": "{} probes".format(len(expr.index)),
        "total_samples": "{} samples".format(len(expr.columns)),
    }

    # First, generate distribution plots for each piece of data
    """
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
    """

    logger.info("  -generating {} x {} heat map".format(expr.shape[0], expr.shape[1]))
    fig, ax = plot.expr_heat_map(expr, fig_size=(4, 8), c_map="Reds", logger=logger)
    ax.set_title("Expression heat map")
    fig.savefig(os.path.join(img_dir, '_expr_heat.png'))
    images['heat_expr'] = os.path.join(img_dir, '_expr_heat.png')

    logger.info("  -generating {} x {} combo heat map".format(expr.shape[0], expr.shape[1]))
    fig = plot.heat_and_density_plot(expr, fig_size=(4, 9), density_position='top',
                                     ratio=4, c_map="Reds", title="Expression", logger=logger)
    fig.savefig(os.path.join(img_dir, '_expr_combo_heat.png'))
    images['heat_cmbo'] = os.path.join(img_dir, '_expr_combo_heat.png')

    # Then, the expression distribution
    """
    logger.info("  -generating distribution plot for expression")
    expr_mat = np.corrcoef(expr, rowvar=False)
    expr_vec = expr_mat[np.tril_indices(expr_mat.shape[0], k=-1)]
    fig, ax = plot.distribution_plot(expr_vec, c="red", logger=logger)
    ax.set_title("Expression similarity distribution 2")
    fig.savefig(os.path.join(img_dir, '_expr_distro.png'))
    images['dis2_expr'] = os.path.join(img_dir, '_expr_distro.png')
    """

    # Then, the expression similarity matrix and distribution
    logger.info("  -generating {} x {} combo expr simi heat map".format(expr_mat.shape[0], expr_mat.shape[1]))
    fig = plot.heat_and_density_plot(expr_mat, fig_size=(4, 6), density_position='top',
                                     ratio=2, c_map="Reds", title="Expression Similarity", logger=logger)
    fig.savefig(os.path.join(img_dir, '_exps_combo_heat.png'))
    images['heat_exps'] = os.path.join(img_dir, '_exps_combo_heat.png')

    # Then, the connectivity distribution
    """
    logger.info("  -generating distribution plot for connectivity")
    conn_mat = conn.as_matrix()
    conn_vec = conn_mat[np.tril_indices(conn_mat.shape[0], k=-1)]
    fig, ax = plot.distribution_plot(conn_vec, c="blue", logger=logger)
    ax.set_title("Connectivity distribution 2")
    fig.savefig(os.path.join(img_dir, '_conn_distro.png'))
    images['dis2_conn'] = os.path.join(img_dir, '_conn_distro.png')
    """

    logger.info("  -generating {} x {} combo connectivity heat map".format(
        conn_mat.shape[0], conn_mat.shape[1]
    ))
    fig = plot.heat_and_density_plot(conn_mat, fig_size=(4, 6), density_position='top',
                                     ratio=2, c_map="Blues", title="Connectivity", logger=logger)
    images['conn_cmbo'] = os.path.join(img_dir, 'conn_combo_heat.png')
    fig.savefig(images['conn_cmbo'])

    logger.info("  -generating {}-len connectivity density distribution".format(len(conn_dens)))
    fig, ax = plot.distribution_plot(conn_dens, title="Connectivity Density",
                                     fig_size=(4, 4), c="blue", logger=logger)
    images['conn_dens'] = os.path.join(img_dir, 'conn_dens_dist.png')
    fig.savefig(images['conn_dens'])

    logger.info("  -generating {} x {} combo connectivity similarity heat map".format(
        conn_simi_mat.shape[0], conn.shape[1]
    ))
    fig = plot.heat_and_density_plot(conn, fig_size=(4, 6), density_position='top',
                                     ratio=2, c_map="Blues", title="Connectivity Similarity", logger=logger)
    images['conn_simi_cmbo'] = os.path.join(img_dir, '_conn_simi_combo_heat.png')
    fig.savefig(images['conn_simi_cmbo'])

    # Then, distance distribution
    """
    logger.info("  -generating distribution plot for distance")
    dist_mat = data.distance_matrix(expr.columns)
    dist_vec = dist_mat[np.tril_indices(n=dist_mat.shape[0], k=-1)]
    fig, ax = plot.distribution_plot(dist_vec, c="green", logger=logger)
    ax.set_title("Distance distribution 2")
    fig.savefig(os.path.join(img_dir, '_dist_distro.png'))
    images['dis2_dist'] = os.path.join(img_dir, '_dist_distro.png')
    """

    logger.info("  -generating {} x {} combo distance heat map".format(dist_mat.shape[0], dist_mat.shape[1]))
    fig = plot.heat_and_density_plot(dist_mat, fig_size=(4, 6), density_position='top',
                                     ratio=2, c_map="Greens", title="Distance", logger=logger)
    fig.savefig(os.path.join(img_dir, '_dist_combo_heat.png'))
    images['dist_cmbo'] = os.path.join(img_dir, '_dist_combo_heat.png')

    # Try on a Mantel correlogram
    logger.info("   -generating an expr vs dist Mantel correlogram.")
    fig, ax = plot.mantel_correlogram(expr_vec, dist_vec, dist_vec, bins=7,
                                      title="Expression similarity vs Distance, over distance")
    images['mantel_expr_dist'] = os.path.join(img_dir, 'mantel_expr_dist.png')
    fig.savefig(images['mantel_expr_dist'])

    logger.info("   -generating an expr vs conn Mantel correlogram.")
    fig, ax = plot.mantel_correlogram(expr_vec, conn_vec, dist_vec, bins=7,
                                      title="Expression similarity vs Connectivity, over distance")
    images['mantel_expr_conn'] = os.path.join(img_dir, 'mantel_expr_conn.png')
    fig.savefig(images['mantel_expr_conn'])

    logger.info("   -generating an conn vs dist Mantel correlogram.")
    fig, ax = plot.mantel_correlogram(conn_vec, dist_vec, dist_vec, bins=7,
                                      title="Connectivity vs Distance, over distance")
    images['mantel_conn_dist'] = os.path.join(img_dir, 'mantel_conn_dist.png')
    fig.savefig(images['mantel_conn_dist'])

    logger.info("   -generating an conn_simi vs dist Mantel correlogram.")
    fig, ax = plot.mantel_correlogram(expr_vec, conn_simi_vec, dist_vec, bins=7,
                                      title="Expression similarity vs Connectivity similarity, over distance")
    images['mantel_expr_cons'] = os.path.join(img_dir, 'mantel_expr_cons.png')
    fig.savefig(images['mantel_expr_cons'])

    # Then, scatterplots for expr vs cmp, with densities

    # For entire sample, and for discrete distance bins:
    #     generate scatterplot of expr vs cmp, with densities
    #     generate correlogram of correlation by bin

    # Generate a whack-a-probe plot
    logger.info("   -generating a set of whack-a-probe curves.")
    conn_curves = []
    cons_curves = []
    null_curves = []
    for base_dir in sorted(os.listdir(data.path_to('derivatives', {}))):
        base_path = os.path.join(data.path_to('derivatives', {}), base_dir)
        # Match on subject, by bids dirname
        if os.path.isdir(base_path) and bids_val("sub", base_dir) == donor_name(args.donor):
            # Match on hemisphere, by bids dirname
            if bids_val("hem", base_dir) == args.hemisphere and bids_val("ctx", base_dir) == args.samples:
                for mid_dir in os.listdir(base_path):
                    mid_path = os.path.join(base_path, mid_dir)
                    if os.path.isdir(mid_path):
                        curve_name = "_".join([
                            args.hemisphere,
                            args.samples,
                            bids_val("tgt", mid_dir),
                            bids_val("alg", mid_dir),
                        ])
                        for file in os.listdir(mid_path):
                            file_path = os.path.join(mid_path, file)
                            if os.path.isfile(file_path) and file[-8:] == "conn.tsv":
                                df = pd.read_csv(file_path, sep='\t')
                                conn_curves.append((curve_name, df))
                            elif os.path.isfile(file_path) and file[-8:] == "cons.tsv":
                                df = pd.read_csv(file_path, sep='\t')
                                cons_curves.append((curve_name, df))
                            elif os.path.isfile(file_path) and file[-4:] == ".tsv":
                                df = pd.read_csv(file_path, sep='\t')
                                null_curves.append((curve_name, df))
    # This filtering ensures we ONLY deal with samples specified in args.
    conn_filter = [args.samples in x[0] for x in conn_curves]
    conn_rel_curves = [i for (i, v) in zip(conn_curves, conn_filter) if v]
    cons_filter = [args.samples in x[0] for x in cons_curves]
    cons_rel_curves = [i for (i, v) in zip(cons_curves, cons_filter) if v]
    null_filter = [args.samples in x[0] for x in null_curves]
    null_rel_curves = [i for (i, v) in zip(null_curves, null_filter) if v]
    fig = plot.whack_a_probe_plot(args.donor, args.hemisphere, args.samples,
                                  conn_rel_curves, cons_rel_curves, null_rel_curves,
                                  fig_size=(8, 5), logger=logger)
    name_string = 'push_corr_{}'.format(args.samples)
    images[name_string] = os.path.join(img_dir, '{}.png'.format(name_string))
    fig.savefig(images[name_string])
    logger.info("     saved whack-a-probes as {} with {} conn, {} cons, and {} null.".format(
        images[name_string], len(conn_rel_curves), len(cons_rel_curves), len(null_rel_curves)
    ))

    # Create a pdf template and fill it with above graphics.
    return generate_pdf(save_as, args, images, strings, logger)


def log_status(data, root_dir, regarding='all', logger=None):
    """ Log (to logger: file or stdout) a brief summary grid of the data available (or not)

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
    for d in data.donors('expr'):
        for a in tsv_files_of_interest['alg'].unique():
            logger.info(template_string.format(
                d=d, a=a,
                l_vals=six_char_summary(tsv_files_of_interest, d, a, 'L'),
                r_vals=six_char_summary(tsv_files_of_interest, d, a, 'R'),
                a_vals=six_char_summary(tsv_files_of_interest, d, a, 'A'),
            ))
