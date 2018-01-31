""" basics.py
Define constant mappings and lookup tables
"""

import pandas as pd
import os

# The default directory mapped in docker:
BIDS_dir = '/data'

# Define some strings and subdirs to keep consistent later.
BIDS_subdir = 'expression'

# A list of all possible donors from the Allen Brain Atlas
donors = ['H03512001',
          'H03512002',
          'H03511009',
          'H03511012',
          'H03511015',
          'H03511016']

# A list of the data files available for each donor (ignores README)
donor_files = ['Ontology.csv',
               'Probes.csv',
               'SampleAnnot.csv',
               'MicroarrayExpression.csv',
               'PACall.csv']

# Lists and dicts for mapping any donor description to his/her 'official' name
# This allows a user to refer to a donor by any of the keys and still reach the data
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

# A Lookup table to use the official name to get other metadata
aba_info = pd.DataFrame(
    columns=['subdir', 'zip_file', 'url', 'bytes', 'expanded_bytes'],
    index=donors,
    data=[
        ['sub-H03512001', 'normalized_microarray_donor9861.zip',
         'http://human.brain-map.org/api/v2/well_known_file_download/178238387',
         425988059, 1082992069],
        ['sub-H03512002', 'normalized_microarray_donor10021.zip',
         'http://human.brain-map.org/api/v2/well_known_file_download/178238373',
         400957002, 1020560085],
        ['sub-H03511009', 'normalized_microarray_donor12876.zip',
         'http://human.brain-map.org/api/v2/well_known_file_download/178238359',
         166233851, 418625823],
        ['sub-H03511012', 'normalized_microarray_donor14380.zip',
         'http://human.brain-map.org/api/v2/well_known_file_download/178238316',
         241359585, 607172090],
        ['sub-H03511015', 'normalized_microarray_donor15496.zip',
         'http://human.brain-map.org/api/v2/well_known_file_download/178238266',
         216077630, 541506317],
        ['sub-H03511016', 'normalized_microarray_donor15697.zip',
         'http://human.brain-map.org/api/v2/well_known_file_download/178236545',
         230640408, 578463821],
    ]
)
aba_info.index.name = 'donor'

# Richiardi, et al .published supplemental data from their analyses.
richiardi_S1 = 'richiardi/Richiardi_Data_File_S1.xlsx'
richiardi_S2 = 'richiardi/Richiardi_Data_File_S2.xlsx'


def rel_path_to(donor, file_key):
    """ provide a relative file path based on any donor and file shorthand we can map.
    """
    return os.path.join('sub-' + donor_map[donor], BIDS_subdir, file_map[file_key])
