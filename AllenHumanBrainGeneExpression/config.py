""" basics.py
Define constant mappings and lookup tables
"""

import pandas as pd

# Define some strings and subdirs to keep consistent later.
BIDS_subdir = 'expression'

# Lists and dicts for mapping any donor description to his/her 'official' name
donor_map = {
    'H03512001': 'H03512001',
    'H0351_2001': 'H03512001',
    'H0351.2001': 'H03512001',
    'sub-H03512001': 'H03512001',
    'donor9861': 'H03512001',
    '2001': 'H03512001',
    'H03512002': 'H03512002',
    'H0351_2002': 'H03512002',
    'H0351.2002': 'H03512002',
    'sub-H03512002': 'H03512002',
    'donor10021': 'H03512002',
    '2002': 'H03512002',
    'H03511009': 'H03511009',
    'H0351_1009': 'H03511009',
    'H0351.1009': 'H03511009',
    'sub-H03511009': 'H03511009',
    'donor12876': 'H03511009',
    '1009': 'H03511009',
    'H03511012': 'H03511012',
    'H0351_1012': 'H03511012',
    'H0351.1012': 'H03511012',
    'sub-H03511012': 'H03511012',
    'donor14380': 'H03511012',
    '1012': 'H03511012',
    'H03511015': 'H03511015',
    'H0351_1015': 'H03511015',
    'H0351.1015': 'H03511015',
    'sub-H03511015': 'H03511015',
    'donor15496': 'H03511015',
    '1015': 'H03511015',
    'H03511016': 'H03511016',
    'H0351_1016': 'H03511016',
    'H0351.1016': 'H03511016',
    'sub-H03511016': 'H03511016',
    'donor15697': 'H03511016',
    '1016': 'H03511016',
}

# A Lookup table to use the official name to get other metadata
aba_info = pd.DataFrame(
    columns=['subdir', 'zipfile', 'url'],
    index=['H03512001', 'H03512002', 'H03511009', 'H03511012', 'H03511015', 'H03511016'],
    data=[
        ['sub-H03512001', 'normalized_microarray_donor9861.zip',
         'http://human.brain-map.org/api/v2/well_known_file_download/178238387'],
        ['sub-H03512002', 'normalized_microarray_donor10021.zip',
         'http://human.brain-map.org/api/v2/well_known_file_download/178238373'],
        ['sub-H03511009', 'normalized_microarray_donor12876.zip',
         'http://human.brain-map.org/api/v2/well_known_file_download/178238359'],
        ['sub-H03511012', 'normalized_microarray_donor14380.zip',
         'http://human.brain-map.org/api/v2/well_known_file_download/178238316'],
        ['sub-H03511015', 'normalized_microarray_donor15496.zip',
         'http://human.brain-map.org/api/v2/well_known_file_download/178238266'],
        ['sub-H03511016', 'normalized_microarray_donor15697.zip',
         'http://human.brain-map.org/api/v2/well_known_file_download/178236545'],
    ]
)
aba_info.index.name = 'donor'

# Each archive contains the same files, by name, although contents differ
meta_files = {
    'ontology': 'Ontology.csv',
    'probes': 'Probes.csv',
    'annotation': 'SampleAnnot.csv',
}

data_files = {
    'expression': 'MicroarrayExpression.csv',
    'call': 'PACall.csv',
}

# Richiardi, et al .published supplemental data from their analyses.
richiardi_S1 = 'richiardi/Richiardi_Data_File_S1.xlsx'
richiardi_S2 = 'richiardi/Richiardi_Data_File_S2.xlsx'
