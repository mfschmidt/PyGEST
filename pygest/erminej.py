
def ready_erminej(base_path):
    """ If ermineJ is not available, download and prepare it. """

    import os
    import urllib.request as request
    import gzip
    import subprocess

    # Ensure pre-requisites exist and are accessible
    ej_path = os.path.join(base_path, "genome", "erminej")
    os.makedirs(ej_path, exist_ok=True)
    ontology = {
        'url': 'http://archive.geneontology.org/latest-termdb/go_daily-termdb.rdf-xml.gz',
        'file': '2019-07-09-erminej_go.rdf-xml',
    }
    annotation = {
        'url': 'https://gemma.msl.ubc.ca/annots/Generic_human_ncbiIds_noParents.an.txt.gz',
        'file': '2020-04-22-erminej_human_annotation_entrezid.txt',
    }
    software = {
        'url': 'http://home.pavlab.msl.ubc.ca/ermineJ/distributions/ermineJ-3.1.2-generic-bundle.zip',
        'file': 'ermineJ-3.1.2-generic-bundle.zip',
    }
    for prereq in [ontology, annotation, ]:
        if not os.path.exists(os.path.join(ej_path, "data", prereq['file'])):
            print("Downloading fresh {}, it didn't exist.".format(prereq['file']))
            response = request.urlopen(prereq['url'])
            with open(os.path.join(ej_path, "data", prereq['file']), "wb") as f:
                f.write(gzip.decompress(response.read()))
    if not os.path.exists(os.path.join(ej_path, software['file'])):
        response = request.urlopen(software['url'])
        with open(os.path.join(ej_path, software['file']), "wb") as f:
            data = response.read()
            f.write(data)

    # Unzip ermineJ software, if necessary
    if not os.path.exists(os.path.join(ej_path, "ermineJ-3.1.2", "bin", "ermineJ.sh")):
        print("Unzipping ermineJ software.")
        p = subprocess.run(
            ['unzip', '-u', os.path.join(ej_path, software['file']), ],
            cwd=ej_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if p.returncode != 0:
            print(p.stdout.decode())
            print(p.stderr.decode())

        p = subprocess.run(
            ['chmod', "a+x", os.path.join(ej_path, "ermineJ-3.1.2", "bin", "ermineJ.sh"), ],
            cwd=ej_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if p.returncode != 0:
            print(p.stdout.decode())
            print(p.stderr.decode())

    return_dict = {
        "executable": os.path.join(ej_path, "ermineJ-3.1.2", "bin", "ermineJ.sh"),
        "ontology": os.path.join(ej_path, "data", ontology['file']),
        "annotation": os.path.join(ej_path, "data", annotation['file']),
        "data": os.path.join(ej_path, "data"),
        "ready": True,
    }
    for req_path in ["executable", "ontology", "annotation", "data", ]:
        if not os.path.exists(return_dict[req_path]):
            return_dict["ready"] = False
    return return_dict


def write_result_as_entrezid_ranking(tsv_file, force_replace=False):
    """ Read a tsv file, and write out its results as ranked entrez_ids for ermineJ to use.

    :param tsv_file: The path to a PyGEST results file
    :param force_replace: Set true to overwrite existing files
    """

    import os
    import pandas as pd
    from .convenience import map_pid_to_eid

    rank_file = tsv_file.replace(".tsv", ".entrez_rank")

    if force_replace or not os.path.isfile(rank_file):
        df = pd.read_csv(tsv_file, sep="\t", index_col=None, header=0)
        df['rank'] = df.index + 1
        df['entrez_id'] = df['probe_id'].apply(lambda x: map_pid_to_eid(x, "fornito"))
        df.sort_index(ascending=True).set_index('entrez_id')[['rank', ]].to_csv(rank_file, sep="\t")

    return rank_file


def run_gene_ontology(tsv_file, base_path="/data"):
    """ Run ermineJ gene ontology on one gene ordering.

    :param tsv_file: full path to a PyGEST result
    :param base_path: PyGEST data root
    """

    import os
    import subprocess

    ej = ready_erminej(base_path)

    go_path = tsv_file.replace(".tsv", ".ejgo_roc_0002-2048")

    rank_file = write_result_as_entrezid_ranking(tsv_file)

    # Only run gene ontology if it does not yet exist.
    print("run_gene_ontology: 'ready': {}, 'go_path': ...{}".format(ej["ready"], go_path[-40:]))
    if ej["ready"] and not os.path.isfile(go_path):
        print("initiating ermineJ run on '{}...{}'".format(tsv_file[:30], tsv_file[-30:]))
        print(ej)
        p = subprocess.run(
            [
                ej['executable'],
                '-d', ej["data"],
                '--annots', ej['annotation'],
                '--classFile', ej['ontology'],
                '--scoreFile', rank_file,
                '--test', 'ROC',  # Method for computing significance. ROC best for ordinal rankings
                '--mtc', 'FDR',  # FDR indicates Benjamini-Hochberg corrections for false discovery rate
                '--reps', 'BEST',  # If a gene has multiple probes/ids in input, use BEST
                '--genesOut',  # Include gene symbols in output
                '--minClassSize', '2',  # smallest gene set size to be considered
                '--maxClassSize', '2048',  # largest gene set size to be considered
                '-aspects', 'BCM',  # Test against all three GO components
                '--logTrans', 'false',  # If we fed p-values, we would set this to true
                '--output', go_path,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Write the log file
        with open(tsv_file.replace(".tsv", ".ejlog"), "a") as f:
            f.write("STDOUT:\n")
            f.write(p.stdout.decode())
            f.write("STDERR:\n")
            f.write(p.stderr.decode())


def read_erminej(filename):
    """ Read output file from ErmineJ and return a dataframe.

        :param str filename: Full path to ermineJ output file
        :return: pandas dataframe containing ermineJ results
    """

    import re
    from io import StringIO
    import pandas as pd

    buf = StringIO()  # Pretend to write to a file, then read it, but it's only in memory for speed

    # dropped_lines = []
    head_count = 0
    data_count = 0
    # drop_count = 0
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            # matched = False

            # Keep the column header line that starts with '#!\t', but without those three characters.
            head_match = re.search('^#!\t', line)
            if head_match:
                head_count += 1
                # matched = True
                buf.write(line[3:].rstrip() + "\n")

            # Keep the data lines that all start with '!\t', but we don't need the exclamation.
            data_match = re.search('^!\t', line)
            if data_match:
                data_count += 1
                # matched = True
                buf.write(line[2:].rstrip() + "\n")

            # Track dropped lines, but we currently do nothing with them.
            # if not matched:
            #     drop_count += 1
            #     dropped_lines.append(line.rstrip())

    buf.seek(0)  # Go back to the beginning of the buffer before asking pandas to read it.
    # print("Kept {} head, {} data lines. Dropped {} lines.".format(head_count, data_count, drop_count))
    return pd.read_csv(buf, sep="\t").sort_values("Pval", ascending=True)


def get_ranks_from_ejgo_file(f, rank_col="Pval", ascending=True):
    """ Read tsv-formatted results file and return the ranks, not values, of the sorted entrez_ids.

    :param f: filename for a tsv results file
    :param rank_col: new name for the ranking column (index is probe_ids)
    :param ascending: order for rank_col to be sorted
    :returns: dataframe with probe_id index and named ranking column
    """

    if ascending is None:
        ascending = True  # The expectation is a file with p-values from 0 (best) to 1 (worst)
    if rank_col is None:
        rank_col = "Pval"

    # If the file does not exist, an exception will be raised. The caller can do with it what they wish.
    df = read_erminej(f)

    # Keep only what we need, and sorted in reverse-whack-a-probe order
    new_df = df[[rank_col, 'ID']].set_index('ID').sort_values(rank_col, ascending=ascending)
    new_df.index.name = "id"

    # Avoid ambiguity; two columns named the same thing may cause problems.
    if rank_col == 'rank':
        rank_col = rank_col + "_old"
        new_df = new_df.rename(columns={"rank": rank_col})

    # Everything is in order, and indexed by ID->id, so slap on a new ranking and return it, sorted by id
    new_df['rank'] = range(1, len(new_df) + 1)
    return new_df[["rank", ]].sort_index()
