""" convenience.py
Define constant mappings and lookup tables and other simple shortcuts
"""

import re
import os
import pandas as pd

from pygest.rawdata import miscellaneous
import pickle


# A list of the data files available for each donor (ignores README)
donor_files = [
    'Ontology.csv',
    'Probes.csv',
    'SampleAnnot.csv',
    'MicroarrayExpression.csv',
    'PACall.csv'
]

# A list of the subdirectories expected in the data directory
#   Other directories may exist for anything else, but we only care about these.
data_sections = ['sourcedata', 'derivatives', 'logs', 'connectivity', 'code', 'cache']

# The human_genome_info dataframe and symbol_to_id map are not often used, so they will be populated on first use.
human_genome_info = pd.DataFrame(data={})
symbol_to_id_map = {'-': 0}


# Lists and dicts for mapping any donor description to his/her 'official' name
# This allows a user to refer to a donor by any of the keys and still reach the data
def donor_name(donor_string):
    donor_map = {
        'donor9861': 'H03512001',
        'donor10021': 'H03512002',
        'donor12876': 'H03511009',
        'donor14380': 'H03511012',
        'donor15496': 'H03511015',
        'donor15697': 'H03511016',
    }
    if donor_string.lower() == 'any':
        return 'H03511009'
    if donor_string.lower() == 'test':
        return 'test'
    if donor_string[:4] == 'sub-':
        donor_string = donor_string[4:]
    for char in ".-_ ":
        donor_string = donor_string.replace(char, "")
    if donor_string in donor_map:
        donor_string = donor_map[donor_string]
    if len(donor_string) == 4:
        donor_string = 'H0351' + donor_string
    return donor_string


def masks_sort(t):
    try:
        return 1000 + int(t)
    except ValueError:
        return -1


def bids_val(sub, whole):
    """ Return the string after the hyphen in a BIDS-inspired tag """

    if sub == "shuffle":
        if "derivatives" in whole:
            return "actual"
        if "edgeshuffles" in whole:
            return "edge"
        if "distshuffles" in whole:
            return "dist"
        if "shuffles" in whole:
            return "random"

    m = re.search(r"(?P<sub>{})-(?P<val>[a-zA-Z0-9+]+)".format(sub), whole)
    if m is None:
        # Don't make caller check for None, and an empty string shouldn't match anything of interest.
        return ''
    else:
        val = m.group('val')
        if '+' in val:
            parts = val.split('+')
            return '+'.join(sorted(parts, key=masks_sort))
        else:
            return m.group('val')


def shuffle_subdir_from_arg(arg):
    """ Map shuffle argument to the subdirectory it implies. """

    return {
        'none': 'derivatives',
        'raw': 'shuffles',
        'agno': 'shuffles',
        'dist': 'distshuffles',
        'edge': 'edgeshuffles',
        'edges': 'edgeshuffles',
        'bin': 'edgeshuffles',
    }.get(arg, None)


def shuffle_arg_from_subdir(arg):
    """ Map shuffle argument to the subdirectory it implies. """

    return {
        'derivatives': 'none',
        'shuffles': 'agno',
        'distshuffles': 'dist',
        'edgeshuffles': 'edge',
    }.get(arg, None)


def dict_from_bids(file_path):
    """ Return a dictionary based on the file_path BIDS keys and values. """

    local_dict = {}
    # remove file extension, but beware a leading ./
    if "." in file_path[1:]:
        local_dict['ext'] = file_path[file_path.rfind("."):]
        file_path = file_path[:file_path.rfind(".")]

    parsed = file_path.replace("/", "_").split("_")
    for key_value_pair_string in parsed:
        pair = key_value_pair_string.split("-")
        if len(pair) == 2:
            local_dict[pair[0]] = pair[1]
        elif len(pair) == 1:
            if shuffle_arg_from_subdir(pair[0]) is not None:
                local_dict['shuffle'] = shuffle_arg_from_subdir(pair[0])
                local_dict['top_subdir'] = pair[0]

    return local_dict


def swap_bids_item(file_path, swap_dict):
    """ Return a modified file path from changes in swap_dict """
    for k, v in swap_dict.items():
        start = file_path.find(k + "-")
        if start > 0:
            end = file_path[start:].find("_") + start
            old_sub = file_path[start:end]
            file_path = file_path.replace(old_sub, "{}-{}".format(k, v))
    return file_path


def short_cmp(comp):
    if comp.startswith('hcpnifti'):
        if 'old' in comp:
            return "oldmale"
        if 'young' in comp:
            return "yngmale"
    return comp


def p_string(val, use_asterisks=False):
    """ Return a string with properly formatted p-value.

    :param val: float p-value
    :param use_asterisks: True to report just asterisks rather than quantitative value
    :returns: formatted string for reporting
    """

    if val < 0.00001:
        if use_asterisks:
            return "*****"
        return "p<0.00001"
    elif val < 0.0001:
        if use_asterisks:
            return "****"
        return "p<0.0001"
    elif val < 0.001:
        if use_asterisks:
            return "***"
        return "p<0.001"
    else:
        if use_asterisks:
            if val < 0.01:
                return "**"
            elif val < 0.05:
                return "*"
            else:
                return ""
        return "p={:0.3f}".format(val)


def all_files_in(d, e):
    """ Return a DataFrame containing all files in directory d with extension e,
        with bids-formatted key-value pairs as columns.

    :param d: root directory to begin the recursive file search
    :param e: file extension to search for
    :returns: pandas dataframe containing all files, keyed by BIDS fields in the paths
    """

    pair_separator = "_"
    pair_joiner = "-"
    file_list = []
    if os.path.isdir(d):
        for root, dirs, files in os.walk(d, topdown=True):
            for f in files:
                if f[(-1 * len(e)):] == e:
                    bids_pairs = []
                    bids_dict = {'root': root, 'name': f}
                    fs_parts = os.path.join(root, f)[: (-1 * len(e)) - 1:].split(os.sep)
                    for fs_part in fs_parts:
                        if '-' in fs_part:
                            pairs = fs_part.split(pair_separator)
                            for pair in pairs:
                                if '-' in pair:
                                    p = pair.split(pair_joiner)
                                    bids_pairs.append((p[0], p[1]))
                                else:
                                    # There should never be an 'extra' but we catch it to debug problems.
                                    bids_pairs.append(('extra', pair))
                    for bp in bids_pairs:
                        bids_dict[bp[0]] = bp[1]
                    file_list.append(bids_dict)
    else:
        return None

    return pd.DataFrame(file_list)


def bids_clean_filename(filename):
    """
    External files are accepted, but their names must fit within BIDS formatting. This function BIDS-ifies.

    Any underscores, hyphens, periods, or spaces are simply removed.

    :param filename: actual name of the picked matrix file
    :return: BIDS-ified string derived from filename
    """

    if isinstance(filename, list):
        return map(bids_clean_filename, filename)

    if filename[-3:] == ".df":
        newname = os.path.basename(filename)[: -3]
    else:
        newname = os.path.basename(filename)

    for c in ["_", '-', '.', ' ', ]:
        newname = newname.replace(c, "")

    return newname


def split_file_name(d, ext):
    """ Return the filename for a split-half dataframe.
        :param dict d: A dictionary containing the parts of the split-half filename.
        :param str ext: 'csv' for list of wellids, 'df' for dataframes
    """
    if ext == 'df':
        return "parcelby-{parby}_splitby-{splby}.df".format_map(d)
    elif ext == 'csv':
        return "{parby}s_splitby-{splby}.csv".format_map(d)
    else:
        raise KeyError("Split file names only handle 'csv' or 'df' files.")


def split_log_name(d):
    """ Return the filename for a split-half log; both splits and phases will be done, so no split or batch exist yet.
        :param dict d: A dictionary containing the parts of the split-half filename.
    """
    return "parcelby-{parby}_seed-{seed:05}".format_map(d)  # .log will be added later for consistency w/results


required_bids_keys = [
    'sub', 'hem', 'samp', 'prob', 'parby', 'splby', 'batch', 'tgt', 'algo', 'comp', 'mask', 'norm', 'adj', 'top_subdir',
]


def result_description(file_path):
    """ From any file path, return a dictionary with an up-to-date description of its characteristics.

    :param str file_path: The path to the result file
    """

    d = dict_from_bids(file_path)

    if 'sub' in d:
        if d['sub'] in ['all', ]:
            pass
        elif d['sub'] in ['H03511009', 'H03511012', 'H03511015', 'H03511016', 'H03512001', 'H03512002', ]:
            d['samp'] = d.get('ctx', d.get('set', 'UNKNOWN'))
            d['prob'] = 'richiardi'
            d['parby'] = 'wellid'
            d['splby'] = 'none'
            d['batch'] = 'all'
        else:
            match = None
            if match is None:
                re_str = r"^(?P<pby>glasser|wellid)(?P<phase>test|train)(?P<seed>\d+)$"
                match = re.compile(re_str).match(d['sub'])
                if match:
                    d['splby'] = 'wellid'
            if match is None:
                re_str = r"^(?P<pby>glasser|wellid)(?P<phase>test|train)by(?P<sby>glasser|wellid)(?P<seed>\d+)$"
                match = re.compile(re_str).match(d['sub'])
                if match:
                    d['splby'] = match.group('sby')
            if match:
                d['sub'] = 'all'
                d['hem'] = 'A'
                d['samp'] = 'glasser'
                d['parby'] = match.group('pby')
                d['batch'] = "{}{:05}".format(match.group('phase'), int(match.group('seed')))

    if 'alg' in d:
        d['algo'] = d['alg']
    if 'prb' in d:
        d['prob'] = d['prb']
    if 'msk' in d:
        d['mask'] = d['msk']
    if 'cmp' in d:
        d['comp'] = d['cmp']
    if 'norm' not in d:
        d['norm'] = 'none'

    errors = []
    for k in required_bids_keys:
        if k not in d:
            errors.append("no {}".format(k))

    return d, errors


def path_to(cmd, args, path_type='result', include_file=True, dir_for_intermediates=False, log_file=False):
    """ Build the path requested and return it.

        Paths should be consistent, so this is the only place we want to be generating paths.
        But there are different paths even within the same command, so we need flexibility in
        determining which file/dir is required.

        :param str cmd: The command being run
        :param args: The command-line arguments passed to cmd
        :param str path_type: Generate a path to a split, result, log, etc
        :param bool include_file: Just the 'dir' or the whole 'file' path
        :param bool dir_for_intermediates: Shall we include an additional subdirectory for intermediate files?
        :param bool log_file: Set to true to get the path to a log file rather than data file
    """

    ext = ""  # normally, we won't need to append an extension.

    bids_dict = {
        'data': args.get('data', ""),
        'cmd': cmd,
        'sub': donor_name(args.get('donor', '')),
        'hem': args.get('hemisphere', ''),
        'splby': args.get('splitby', ''),
        'parby': args.get('parcelby', ''),
        'samp': args.get('samples', ''),
        'prob': args.get('probes', ''),
        'tgt': args.get('direction', ''),
        'algo': args.get('algorithm', ''),
        'norm': args.get('expr_norm', ''),
        'adj': args.get('adjust', ''),
        'start': args.beginning.strftime("%Y%m%d%H%M%S") if "beginning" in args else "",
        'seed': args.get('seed', 0),
        'batch': args.get('batch', 'whole'),
    }

    if 'comparator' in args:
        bids_dict['comp'] = bids_clean_filename(args.comparator)
    elif 'comp' in args:
        bids_dict['comp'] = args.comp
    elif 'cmp' in args:
        bids_dict['comp'] = args.cmp

    if "shuffle" in args:
        bids_dict['top_subdir'] = shuffle_subdir_from_arg(args.shuffle)

    if "masks" in args and len(args.masks) > 0:
        bids_dict['mask'] = '+'.join(bids_clean_filename(args.masks))
    else:
        bids_dict['mask'] = 'none'

    if "comparatorsimilarity" in args and args.comparatorsimilarity:
        bids_dict['comp'] = bids_dict['comp'] + "sim"

    # Make the BIDS name out of the dict values
    if log_file:
        bids_dict['make_log_file'] = True
        ext = ".log"
    if cmd == 'split' or ('command' in args and args.command == 'split') or path_type == 'split':
        bids_dict['top_subdir'] = 'splits'
        new_name = split_path_from_dict(bids_dict)
    else:
        new_name = result_path_from_dict(bids_dict)

    if dir_for_intermediates:
        intermediate_dir = 'intdata_' + '-'.join([bids_clean_filename(args.comparator), bids_dict['mask'], args.adjust])
        new_name = os.path.join(new_name[:new_name.rfind("/")], intermediate_dir)
        os.makedirs(os.path.abspath(new_name), exist_ok=True)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(new_name)), exist_ok=True)

    if include_file:
        return new_name + ext
    else:
        return new_name[:new_name.rfind("/")]


def sub_dir_source(d):
    """ build out the source portion of the directory structure.
    :param dict d: A dictionary holding BIDS terms for path-building
    """
    return "_".join([
        '-'.join(['sub', d['sub'], ]),
        '-'.join(['hem', d['hem'], ]),
        '-'.join(['samp', d['samp'], ]),
        '-'.join(['prob', d['prob'], ]),
    ])


def sub_dir_by_split(d):
    """ build out the split portion of the results directory structure.
    :param dict d: A dictionary holding BIDS terms for path-building
    """
    return "_".join([
        '-'.join(['parby', d['parby'], ]),
        '-'.join(['splby', d['splby'], ]),
        '-'.join(['batch', d['batch'], ]),
    ])


def sub_dir_algo(d):
    """ build out the algorithm portion of the directory structure.
    :param dict d: A dictionary holding BIDS terms for path-building
    """
    return "_".join([
        '-'.join(['tgt', d['tgt'], ]),
        '-'.join(['algo', d['algo'], ]),
    ])


def result_file_name(d):
    """ build out the split portion of the directory structure.

    :param dict d: A dictionary holding BIDS terms for path-building
    """

    return "_".join([
        '-'.join(['sub', d['sub'], ]),
        '-'.join(['comp', d['comp'], ]),
        '-'.join(['mask', d['mask'], ]),
        '-'.join(['norm', d['norm'], ]),
        '-'.join(['adj', d['adj'], ]),
    ])


def split_path_from_dict(d):
    """ Build the correct path for a split file from the d dict.

    :param dict d: A dictionary holding BIDS terms for path-building
    """

    # There are only three conditions where this function would be called.
    # 1. We are pushing and want to use a previously split file for expression data.
    #    In this case, we need to know which split and batch to use, and we need a file.
    if d['cmd'] == 'push':
        if 'parby' in d and 'parcelby' not in d:
            d['parcelby'] = d['parby']
        if 'splby' in d and 'splitby' not in d:
            d['splitby'] = d['splby']
        file_name = split_file_name(d, 'df')
        return os.path.join(d['data'], d['top_subdir'], sub_dir_source(d),
                            '-'.join(['batch', d['batch'], ]), file_name)

    # 2. We are splitting expression data and the logger wants to start a log file. (no parcelby or batch exist yet)
    elif d.get('make_log_file', False):
        file_name = split_log_name(d)
        return os.path.join(d['data'], 'splits', sub_dir_source(d), file_name)

    # 2. We are splitting expression data and need a base folder to start with. (no parcelby or batch exist yet)
    else:
        file_name = "IGNORED.FILE"
        return os.path.join(d['data'], 'splits', sub_dir_source(d), file_name)  # file_name will be stripped off later


def result_path_from_dict(d):
    """ Build the correct path for output from the d dict.
    
    :param dict d: A dictionary holding BIDS terms for path-building
    """
    
    file_name = result_file_name(d)
    if d['sub'] == 'test':
        file_name = '_'.join(['dt-' + d['start'], file_name])

    # The most common, default path construction:
    new_name = os.path.join(
        d['data'],
        d.get('top_subdir', ''),
        sub_dir_source(d),
        sub_dir_by_split(d),
        sub_dir_algo(d),
        file_name
    )

    # Building reports and generating data require different levels of name
    if d['cmd'] == 'order':
        new_name = '_'.join([new_name, 'order'])

    if d['top_subdir'] != 'derivatives':
        new_name = '_'.join([new_name, 'seed-{0:05d}'.format(int(d['seed']))])

    return new_name


def get_entrez_id_from_gene_name(gene_name, data_dir="/data"):
    """ Lookup the gene symbol gene_name and return its entrez_id

    :param gene_name:
    :param data_dir: The PYGEST_DATA base path
    :return:
    """

    global human_genome_info
    global symbol_to_id_map

    entrez_id = ""
    entrez_source = ""

    # Inevitably, we'll be called repeatedly on blank gene names.
    if gene_name == "":
        return 0, ""

    # The LOC00000 genes are named after their entrez_id. No point looking them up.
    if gene_name.startswith("LOC"):
        try:
            return int(gene_name[3:]), "loc"
        except ValueError:
            # The rest of the symbol is not a number; try it in the mapper later
            pass

    # Use the dictionary we built first. It's fastest. But if it doesn't work, we may need to spend some time.
    try:
        return symbol_to_id_map[gene_name], "map"
    except KeyError:
        # print("searching for {}, not mappable".format(gene_name))
        # Do this two ways to see if they get the same results.
        with open(os.path.join(data_dir, "genome", "Homo_sapiens.gene_info"), 'r') as f:
            for line in f:
                match = re.search(r"^(\d+)\s+(\d+)\s+.*{}.*$".format(gene_name), line)
                if match:
                    try:
                        entrez_id = int(match.group(2))
                        entrez_source = "extra fields"
                    except ValueError:
                        print("Found {} in file, but {} is not a number.".format(
                            gene_name, match.group(2)
                        ))
        mask = human_genome_info[['dbXrefs', 'description']].applymap(lambda x: gene_name in str(x)).any(axis=1)
        if mask.sum() == 1:
            if entrez_id == human_genome_info[mask].index[0]:
                print("+ Found matching entrez_id {} in file and df from {}".format(entrez_id, gene_name))
                return entrez_id, entrez_source
            else:
                print("- Found entrez_id {} in file, {} in df, from {}".format(
                    entrez_id, human_genome_info[mask].index[0], gene_name
                ))
        elif mask.sum() == 0:
            if entrez_id == "":
                print("- No ids for {}".format(gene_name))
            else:
                print("- Found entrez_id {} in file, but not in df, from {}".format(
                    entrez_id, gene_name
                ))
        else:
            print("- Gene {} has {} matches!".format(gene_name, mask.sum()))

    return "", 0


def create_symbol_to_id_map(gene_info_file='/data/genome/Homo_sapiens.gene_info', data_root="/data",
                            use_synonyms=True, print_dupes=False):
    """
    Load gene info file and convert it to a dictionary allowing rapid entrez_id lookup from symbols

    :param gene_info_file: the path to a gene_info file from the NCBI
    :param data_root: the path to the root of all pygest data
    :param use_synonyms: Set to False to only use gene symbols from the symbol column. By default, synonyms match too.
    :param print_dupes: Set to True to print out each time an entrez_id is overwritten during map creation.
    :return: dictionary mapping symbols to entrez ids
    """

    global symbol_to_id_map
    # First shot should just be returning it from memory
    if len(symbol_to_id_map.keys()) > 1:
        return symbol_to_id_map

    # Next shot should be loading it from disk
    symbol_to_id_map_path = os.path.join(data_root, "genome/symbol_to_id_map.dict")
    if os.path.isfile(symbol_to_id_map_path):
        with open(symbol_to_id_map_path, "rb") as f:
            symbol_to_id_map = pickle.load(f)
            return symbol_to_id_map

    # And only if those both fail, clear and rebuild it.
    # Clear the global dictionary
    syn_map = {}
    sid_map = {}

    # Import the data from a text file to the global dataframe
    global human_genome_info
    human_genome_info = pd.read_csv(gene_info_file, delimiter='\t')
    human_genome_info = human_genome_info.set_index('GeneID')

    symbols = set()
    # Map synonyms first. They can later be overwritten by primary symbols
    if use_synonyms:
        for i, row in human_genome_info.sort_index(ascending=False).iterrows():
            for symbol in row['Synonyms'].split("|"):
                symbols.add(symbol)
                # Store each synonymous symbol string as a key in the dictionary,
                if symbol in syn_map.keys() and row.name != syn_map[symbol]:
                    if print_dupes:
                        print("  appending synonym '{}' to {{{}:{}}}".format(row.name, symbol, syn_map[symbol]))
                    if isinstance(syn_map[symbol], list):
                        syn_map[symbol].append(row.name)
                    else:
                        syn_map[symbol] = [syn_map[symbol], row.name]
                else:
                    syn_map[symbol] = row.name

    # Reverse-map Entrez IDs and Symbols
    for i, row in human_genome_info.sort_index(ascending=False).iterrows():
        symbols.add(row['Symbol'])
        if row['Symbol'] in sid_map.keys() and row.name != sid_map[row['Symbol']]:
            if print_dupes:
                print("  appending id '{}' to {{{}:{}}}".format(row.name, row['Symbol'], sid_map[row['Symbol']]))
            if isinstance(sid_map[row['Symbol']], list):
                sid_map[row['Symbol']].append(row.name)
            else:
                sid_map[row['Symbol']] = [sid_map[row['Symbol']], row.name, ]
        # Store the canonical symbol string as a key in the dictionary
        else:
            sid_map[row['Symbol']] = row.name

    # Remove the empty symbol
    try:
        symbols.remove("-")
    except KeyError:
        pass

    # Determine the appropriate entrez_id to use for each gene symbol.
    symbol_list = []
    for gene in sorted(list(symbols)):
        this_gene = {'gene': gene, 'syn_hits': 0, 'id_hits': 0}
        # Try synonyms first, then they can be overwritten if necessary.
        try:
            if isinstance(syn_map[gene], list):
                this_gene['entrez_id'] = 0
                this_gene['syn_id'] = None
                this_gene['syn_hits'] = len(syn_map[gene])
            else:
                this_gene['entrez_id'] = int(syn_map[gene])
                this_gene['syn_id'] = int(syn_map[gene])
                this_gene['syn_hits'] = 1
        except KeyError:
            pass
        # IDs are priority. If we find one, overwrite a synonym.
        try:
            if isinstance(sid_map[gene], list):
                this_gene['entrez_id'] = 0
                this_gene['sid_id'] = None
                this_gene['id_hits'] = len(sid_map[gene])
            else:
                this_gene['entrez_id'] = int(sid_map[gene])
                this_gene['sid_id'] = int(sid_map[gene])
                this_gene['id_hits'] = 1
        except KeyError:
            # No id, there may already be a synonym. If so, leave it alone.
            pass
        symbol_list.append(this_gene)

    # Manually add a few that are in AHBA, but not in the NCBI file.
    symbol_list.append({'gene': 'FLJ23867', 'entrez_id': 200058})
    symbol_list.append({'gene': 'FLJ37035', 'entrez_id': 399821})
    symbol_list.append({'gene': 'FLJ21408', 'entrez_id': 400512})
    symbol_list.append({'gene': 'PP14571', 'entrez_id': 100130449})

    # Convert to a dataframe, report stats, and update the dictionary.
    df_symbols = pd.DataFrame(data=symbol_list)
    if use_synonyms:
        print("Individually, {} synonyms, {} ids".format(len(syn_map), len(sid_map)))
        print("Combined, {} synonyms, {} ids".format(
            len(df_symbols[df_symbols['syn_id'].notnull()]), len(df_symbols[df_symbols['sid_id'].notnull()])
        ))
    else:
        print("Individually, {} ids".format(len(sid_map)))
        print("Combined, {} ids".format(len(df_symbols[df_symbols['sid_id'].notnull()])))
    print("{} good keys. {} have ambiguous (multiple) mappings, {} nulls".format(
        len(df_symbols[df_symbols['entrez_id'] > 0]),
        len(df_symbols[df_symbols['entrez_id'] == 0]),
        len(df_symbols[df_symbols['entrez_id'].isnull()]),
    ))

    symbol_to_id_map = df_symbols.set_index('gene')['entrez_id'].to_dict()

    with open(symbol_to_id_map_path, "wb") as f:
        pickle.dump(symbol_to_id_map, f)

    return symbol_to_id_map


def map_pid_to_eid(probe_id, source="latest"):
    """ Return an entrez_id for any probe_id, 0 if the probe_id is not found. """
    try:
        if source == "original":
            return miscellaneous.map_pid_to_eid_original[probe_id]
        elif source == "richiardi":
            return miscellaneous.map_pid_to_eid_richiardi[probe_id]
        elif source in ["fornito", "aurina", "arnatkeviciute"]:
            return miscellaneous.map_pid_to_eid_fornito[probe_id]
        elif source in ["schmidt", "pantazatos"]:
            return miscellaneous.map_pid_to_eid_schmidt[probe_id]
        else:
            return miscellaneous.map_pid_to_eid_schmidt[probe_id]
    except KeyError:
        # If the probe_id is unmappable to a gene, it should not be 0, but a unique number that couldn't possibly
        # match another gene. NaN seems appropriate, but we still need to count uniques for overlap percentages.
        return -1 * int(probe_id)


def json_lookup(k, path):
    """ Return the value at key k in file path.

    :param str k: json key
    :param str path: filepath to json file
    """

    # Because we know we will be looking up values in non-conforming json files, we don't use json.
    # We just parse the text file.
    with open(path, 'r') as f:
        for line in f:
            match = re.compile("^\\s*\"" + k + "\": \"(?P<v>.*)\".*$").match(line)
            if match:
                return match.group('v')
    return None


# Null distributions
shuffle_dirs = {
    'none': 'derivatives',
    'raw': 'shuffles',
    'agno': 'shuffles',
    'dist': 'distshuffles',
    'dists': 'distshuffles',
    'bin': 'edgeshuffles',
    'edge': 'edgeshuffles',
    'edges': 'edgeshuffles',
}


# Canned lists of samples or probes to draw from
canned_map = {
    'richiardi': 'richiardi',
    'Richiardi': 'richiardi',
    'Rich': 'richiardi',
    'rich': 'richiardi',
    '16906': 'richiardi',
    '17k': 'richiardi',
    'schmidt': 'schmidt',
    'Schmidt': 'schmidt',
    'arnatkeviciute': 'fornito',
    'Arnatkeviciute': 'fornito',
    'fornito': 'fornito',
    'Fornito': 'fornito',
    'test': 'test',
    'testset': 'test',
    'test_set': 'test',
    'Test': 'test',
    'TestSet': 'test',
    'testSet': 'test',
    'test-set': 'test',
    'all': 'all',
    'All': 'all',
    'ALL': 'all',
    'every': 'all',
    'Every': 'all',
    'everything': 'all',
    'Everything': 'all',
    'conn': 'indi',
    'cons': 'indi',
    'INDI': 'indi',
    'Indi': 'indi',
    'indi': 'indi',
    'glasser': 'glasser',
    'Glasser': 'glasser',
}

# Text descriptions of items in the canned map
canned_description = {
    'richiardi': 'Cortical samples from Richiardi, et al.',
    'schmidt': 'Cortical samples from Schmidt, et al.',
    'fornito': 'Samples from Arnatkeviciute, et al.',
    'test': 'A pruned test set for quick runs',
    'all': 'Complete sets with no filters',
    'indi': 'Original INDI connectivity matrix from NKI',
}

# Each archive contains the same files, by name, although contents differ
file_map = {
    'ont': 'Ontology.csv',
    'ontology': 'Ontology.csv',
    'Ontology': 'Ontology.csv',
    'Ontology.csv': 'Ontology.csv',
    'pro': 'Probes.csv',
    'probes': 'Probes.csv',
    'Probes': 'Probes.csv',
    'Probes.csv': 'Probes.csv',
    'ann': 'SampleAnnot.csv',
    'annot': 'SampleAnnot.csv',
    'annotation': 'SampleAnnot.csv',
    'Annotation': 'SampleAnnot.csv',
    'Annotation.csv': 'SampleAnnot.csv',
    'SampleAnnot.csv': 'SampleAnnot.csv',
    'exp': 'MicroarrayExpression.csv',
    'expr': 'MicroarrayExpression.csv',
    'expression': 'MicroarrayExpression.csv',
    'Expression': 'MicroarrayExpression.csv',
    'Expression.csv': 'MicroarrayExpression.csv',
    'MicroarrayExpression.csv': 'MicroarrayExpression.csv',
    'pac': 'PACall.csv',
    'cal': 'PACall.csv',
    'call': 'PACall.csv',
    'pacall': 'PACall.csv',
    'PACall': 'PACall.csv',
    'PACall.csv': 'PACall.csv',
}

map_cmp_to_filename = {
    'hcpniftismoothgrandmean': "hcp_niftismooth_grandmean.df",
    'hcpniftismoothmaleyoungmean': "hcp_niftismooth_maleyoungmean.df",
    'hcpniftismoothmaleoldmean': "hcp_niftismooth_maleoldmean.df",
    'hcpniftismoothgrandmeansim': "hcp_niftismooth_grandmean_sim.df",
    'hcpniftismoothmaleyoungmeansim': "hcp_niftismooth_maleyoungmean_sim.df",
    'hcpniftismoothmaleoldmeansim': "hcp_niftismooth_maleoldmean_sim.df",
}

# Batch IDs are not supplied with ABA data. We can add them to our samples with the following map.
batch_id_map = {
    0: 0,
}
