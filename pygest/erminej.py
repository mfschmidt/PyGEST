
def read_erminej(filename):
    """ Read output file from ErmineJ and return a dataframe.

        :param str filename: Full path to ermineJ output file
        :return: pandas dataframe containing ermineJ results
    """

    import re
    from io import StringIO
    import pandas as pd

    kept_lines = []
    dropped_lines = []
    head_count = 0
    data_count = 0
    drop_count = 0
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            matched = False

            # Keep the column header line that starts with '#!\t', but without those three characters.
            head_match = re.search('^#!\t', line)
            if head_match:
                head_count += 1
                matched = True
                kept_lines.append(line[3:].rstrip())

            # Keep the data lines that all start with '!\t', but we don't need the exclamation.
            data_match = re.search('^!\t', line)
            if data_match:
                data_count += 1
                matched = True
                kept_lines.append(line[2:].rstrip())

            # Track dropped lines, but we currently do nothing with them.
            if not matched:
                drop_count += 1
                dropped_lines.append(line.rstrip())

    # print("Kept {} head, {} data lines. Dropped {} lines.".format(head_count, data_count, drop_count))
    return pd.read_csv(StringIO("\n".join(kept_lines)), sep="\t").sort_values("Pval", ascending=True)


def run_erminej_on(rank_file, PYGEST_DATA="/data", overwrite=False):
    """ Use the given rank file as ranked entrez_ids, then run ermineJ with pre-set options.

        :param rank_file: The path to a file of ranked and ordered entrez_ids
        :param PYGEST_DATA: The base directory containing PyGEST data
        :param overwrite: Set to True to force overwriting existing gene ontology results.
        :return: The path to the ermineJ results file
    """

    import os
    import subprocess

    ontology = {
        'url': 'http://archive.geneontology.org/latest-termdb/go_daily-termdb.rdf-xml.gz',
        'file': '2019-07-09-erminej_go.rdf-xml',
    }
    annotation = {
        'url': 'https://gemma.msl.ubc.ca/annots/Generic_human_ncbiIds_noParents.an.txt.gz',
        'file': '2020-04-22-erminej_human_annotation_entrezid.txt',
    }

    # Read in 'rank_file', do ontology, write out 'result_file'.
    result_file = rank_file.replace(".entrez_rank", ".ejgo_roc_0002-2048")  # Based on min=2, max=2048 genes in GO group
    if overwrite or not os.path.isfile(result_file):
        p = subprocess.run(
            [
                os.path.join(PYGEST_DATA, 'genome', 'erminej', 'erminej-3.1.2', 'bin', 'ermineJ.sh'),
                '-d', os.path.join(PYGEST_DATA, 'genome', 'erminej', 'data'),
                '--annots', os.path.join(PYGEST_DATA, 'genome', 'erminej', 'data', annotation['file']),
                '--classFile', os.path.join(PYGEST_DATA, 'genome', 'erminej', 'data', ontology['file']),
                '--scoreFile', rank_file,
                '--test', 'ROC',  # Method for computing significance. GSR best for gene scores
                '--mtc', 'FDR',  # FDR indicates Benjamini-Hochberg corrections for false discovery rate
                '--reps', 'BEST',  # If a gene has multiple scores in input, use BEST
                '--genesOut',  # Include gene symbols in output
                '--minClassSize', '2',  # smallest gene set size to be considered
                '--maxClassSize', '2048',  # largest gene set size to be considered
                '-aspects', 'BCM',  # Test against all three GO components
                '--logTrans', 'false',  # If we fed p-values, we would set this to true
                '--output', result_file,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Write the log file
        with open(result_file + ".log", "w") as f:
            f.write("STDOUT:\n")
            f.write(p.stdout.decode())
            f.write("STDERR:\n")
            f.write(p.stderr.decode())

        return result_file, True

    return result_file, False
