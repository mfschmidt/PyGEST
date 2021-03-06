import os
import sys
import errno
import datetime
import socket

import humanize
import logging
import pickle

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

# Get strings & dictionaries & DataFrames from the local (not project) config
from pygest import donor_name, algorithms, rawdata
from pygest.rawdata import miscellaneous, glasser
from pygest.convenience import file_map, canned_map, bids_val, all_files_in
from pygest.convenience import bids_clean_filename


BIDS_subdir = 'expr'


class ExpressionData(object):
    """ Wrap Allen Brain Institute expression data

    The general strategy is to keep track of what data we have, and in which state.
    We can then provide data to callers, deciding intelligently whether to send them
    cached data, restructure from an upstream source, or download from ABA and start
    from scratch. We can then use the source data to present users with filtered
    versions many ways.
    """

    # Remember the base directory for all data. Default can be overridden in __init__
    _dir = '/data'

    # We need a place to store donor information; it will come from participants.tsv
    _donors = pd.DataFrame(
        columns=['donor', 'participant_id', 'sex', 'age', 'race',
                 'tissue_receipt_date', 'handed', 'left', 'right'],
        data=[]
    )

    # Store named subsets of probes, samples, and expression.
    # self._cache.loc['all-samples', 'dataframe']
    #     will hold a 3702 sample x 9-field DataFrame from all SampleAnnot.csv files.
    # self._cache.loc['all-expression', 'dataframe']
    #     will hold a 58,962 probe x 3702 sample DataFrame from all MicroarrayExpression files.
    _cache = pd.DataFrame(
        index=[],
        columns=['name,', 'type', 'dataframe', 'file', ],
        data=[]
    )

    _probe_name_to_id = {}

    _logger = logging.getLogger('pygest')

    def __init__(self, data_dir, external_logger=None):
        """ Initialize this object, raising an error if necessary.

        :param data_dir: the base directory housing all expression data and caches
        """

        # Make sure we actually have a place to work.
        if data_dir is not None and os.path.isdir(data_dir):
            self._dir = data_dir
        else:
            raise NotADirectoryError(errno.ENOENT, os.strerror(errno.ENOENT), data_dir)
        # Now, everything just uses self._dir internally.

        # Set up logging, either to our own handlers or to an external one
        self.configure_logging(external_logger)

        # Clear the cache, but not the disk, then
        # take a look at the data directory and log what we have access to.
        self.refresh()

        # We need about 5GB of disk just for raw data, plus however much assists with caching
        # req_space = aba_downloads['expanded_bytes'].sum()
        # if self.space_available() < req_space - self.space_used():
        #     raise ResourceWarning(
        #         "Expression data require at least {req}. {d} has {a}.".format(
        #             req=humanize.naturalsize(req_space - self.space_used()),
        #             d=self._dir,
        #             a=humanize.naturalsize(self.space_available()),
        #         )
        #     )

    def refresh(self, clean=False):
        """ Get the lay of the land and remember what we find.
        :param clean: if set to True, delete all cached values and start over.
        """

        # Wipe our cache pointers
        self._cache = pd.DataFrame(
            index=[],
            columns=self._cache.columns,
            data=[]
        )

        # Delete all files in the cache if asked to start from scratch
        if clean:
            path_to_cache = os.path.sep.join(self.cache_path('_').split(sep=os.path.sep)[:-1])
            for f in os.listdir(path_to_cache):
                f_path = os.path.join(path_to_cache, f)
                try:
                    if os.path.isfile(f_path):
                        print("  removing {}".format(f_path))
                        os.remove(f_path)
                except Exception as e:
                    print(e)

        # Find/make each section(subdirectory).
        try:
            self._donors = pd.read_csv(os.path.join(self._dir, 'sourcedata', 'participants.tsv'), sep='\t')
        except FileNotFoundError:
            self._donors = pd.DataFrame(
                columns=["participant_id", "sex", "age", "race", "tissue_receipt_date", "handed", "left", "right"]
            )
        self._donors['donor'] = [donor_name(x) for x in self._donors['participant_id']]
        self._logger.info("Found {} donors in {}".format(
            len(self._donors['donor']),
            os.path.join(self._dir, 'sourcedata', 'participants.tsv')
        ))

    def donors(self, criterion='expr'):
        if criterion == 'all':
            try:
                return list(self._donors['donor'])
            except KeyError:
                return []
        elif criterion == 'expr':
            try:
                expr_filter = (self._donors['left'] == 'yes') | (self._donors['right'] == 'yes')
                return list(self._donors.loc[expr_filter]['donor'])
            except KeyError:
                return []
        else:
            return []

    def expression(self, name=None, probes=None, samples=None, normalize=None, cache=False):
        """ The expression property
        Filterable by probe rows or sample columns

        :param name: a label used to store and retrieve a specific subset of expression data
        :param probes: a list of probes used to filter expression data
        :param samples: a list of samples (well_ids) used to filter expression data
        :param normalize: a normalization, already applied to expression data, to be loaded instead of raw.
        :param cache: set to True if a cache file should be written with this name.
        """

        # If a name is specified, with no other specs, we just return a cached DataFrame
        if name is not None:
            if probes is None and samples is None:
                if normalize is None:
                    return self.from_cache(name + '-expression')
                else:
                    return self.from_cache(name + '-expression-{}'.format(normalize))
            else:
                if normalize is None:
                    return self.from_cache('all-expression')
                else:
                    return self.from_cache('all-expression-{}'.format(normalize))

        # With filters, we will generate a filtered DataFrame.
        if normalize is None:
            filtered_expr = self.from_cache('all-expression')
        else:
            filtered_expr = self.from_cache('all-expression-{}'.format(normalize))

        if isinstance(probes, list) or isinstance(probes, pd.Series) or isinstance(probes, np.ndarray):
            filtered_expr = filtered_expr.loc[filtered_expr.index.isin(list(probes)), :]
        elif isinstance(probes, pd.DataFrame):
            filtered_expr = filtered_expr.loc[probes.index, :]
        elif isinstance(probes, str):
            filtered_expr = filtered_expr.loc[self.probes(name=probes).index, :]

        if isinstance(samples, list) or isinstance(samples, pd.Series):
            filtered_expr = filtered_expr.loc[:, list(samples)]
        elif isinstance(samples, pd.DataFrame):
            filtered_expr = filtered_expr.loc[:, samples.index]
        elif isinstance(samples, str):
            filtered_expr = filtered_expr.loc[:, self.samples(name=samples).index]

        # If we're given a name, cache the filtered DataFrame
        if cache and name is not None:
            if normalize is None:
                self.to_cache(name + '-expression', data=filtered_expr)
            else:
                self.to_cache(name + "-expression-{}".format(normalize), data=filtered_expr)

        return filtered_expr

    def samples(self, name=None, samples=None, donor=None, hemisphere=None, cache=False):
        """ The samples property
        Asking for ExpressionData.samples will return a dataframe with all samples.
        To get a sub-frame, call samples(data=list_of_wanted_well_ids).

        :param name: a label used to store and retrieve a specific subset of sample data
        :param samples: a list of samples (wellids) used to filter sample data
        :param donor: specifying donor will constrain the returned samples appropriately.
        :param hemisphere: specifying left or right will constrain the returned samples appropriately.
        :param cache: set to True if a cache file should be written with this name.
        :return: A DataFrame indexed by well_id, with sample information
        """

        # Do some interpretation before the heavy lifting.
        h = 'A' if hemisphere is None else hemisphere[0].upper()
        # self._logger.debug("[samples] got hemisphere of '{}', which becomes '{}'".format(hemisphere, h))

        # If a name is specified, with no other specs, we just return a cached DataFrame
        if name is not None:
            if samples is None and donor is None and hemisphere is None:
                if name in canned_map:
                    self._logger.debug("returning samples straight from {} cache.".format(canned_map[name]))
                    return self.from_cache(canned_map[name] + '-samples')
                elif name in self.donors():
                    self._logger.debug("returning samples straight from {} cache.".format(donor_name(name)))
                    return self.from_cache(donor_name(name) + '-samples')
                else:
                    self._logger.warning("[samples] could not find samples named {}.".format(name))
                    return self.from_cache(name + '-samples')

        # But if any filters are present, forget the name and build the DataFrame.
        filtered_samples = None
        if donor is not None:
            if donor_name(donor) in self.donors():
                filtered_samples = self.from_cache(donor_name(donor) + '-samples')
            elif donor.lower() != 'all':
                self._logger.warning("[samples] Donor {} (from {}); did not match any of {}".format(
                    donor_name(donor), donor, self.donors()
                ))
                self._logger.warning("[samples] Moving forward with the full sample set.")

        shape_str = 'None' if filtered_samples is None else filtered_samples.shape
        self._logger.debug("  1. filtered_samples (from donor) is shape {}".format(shape_str))

        # If we didn't get a donor, start with a full sample set.
        if filtered_samples is None:
            filtered_samples = self.from_cache('all-samples')

        shape_str = 'None' if filtered_samples is None else filtered_samples.shape
        self._logger.debug("  2. filtered_samples (from cache) is shape {}".format(shape_str))

        # With samples filters, we'll filter the dataframe, and maintain the order of samples.
        if isinstance(samples, list) or isinstance(samples, pd.Series) or isinstance(samples, pd.Index):
            filtered_samples = filtered_samples.reindex([s for s in samples if s in filtered_samples.index])
        elif isinstance(samples, pd.DataFrame):
            filtered_samples = filtered_samples.reindex([s for s in samples.index if s in filtered_samples.index])

        shape_str = 'None' if filtered_samples is None else filtered_samples.shape
        self._logger.debug("  3. filtered_samples (from samples) is shape {}".format(shape_str))

        # By hemisphere, we will restrict to left or right
        # MNI space defines right of mid-line as +x and left of midline as -x
        if h == 'L':
            left_indices = [s for s in filtered_samples.index if s in miscellaneous.left_samples]
            filtered_samples = filtered_samples.loc[left_indices, :]
        elif h == 'R':
            right_indices = [s for s in filtered_samples.index if s in miscellaneous.right_samples]
            filtered_samples = filtered_samples.loc[right_indices, :]
        elif h == '0':
            neither_indices = [s for s in filtered_samples.index if s in miscellaneous.nonlateral_samples]
            filtered_samples = filtered_samples.loc[neither_indices, :]
        elif h == 'A':
            pass
        else:
            self._logger.warning("{} is not interpretable as a hemisphere; ignoring it.".format(hemisphere))

        shape_str = 'None' if filtered_samples is None else filtered_samples.shape
        self._logger.debug("  4. filtered_samples (by hemi) is shape {}".format(shape_str))

        # If we're given a name, and didn't already pull it from cache, cache the filtered DataFrame
        # This will happen with mode= anything other than 'pull', which will return a cached copy if found first
        if cache and name is not None:
            self.to_cache(name + '-samples', data=filtered_samples)

        return filtered_samples

    def parcels(self, name=None, parcels=None, hemisphere=None, cache=False):
        """ The parcels property
        Asking for Data.parcels will return a dataframe with all parcels.
        To get a sub-frame, call parcels(data=list_of_wanted_parcel_ids).

        :param name: a label used to store and retrieve a specific subset of parcel data
        :param parcels: a list of parcels used to filter sample data
        :param hemisphere: specifying left or right will constrain the returned parcels appropriately.
        :param cache: set to True if a cache file should be written with this name.
        :return: A DataFrame indexed by parcel_id, with parcel information
        """

        # Do some interpretation before the heavy lifting.
        h = 'A' if hemisphere is None else hemisphere[0].upper()
        # self._logger.debug("[samples] got hemisphere of '{}', which becomes '{}'".format(hemisphere, h))

        # If a name is specified, with no other specs, we just return a cached DataFrame
        if name is not None:
            if parcels is None and hemisphere is None:
                if name in canned_map:
                    self._logger.debug("returning parcels straight from {} cache.".format(canned_map[name]))
                    return self.from_cache(canned_map[name] + '-parcels')
                else:
                    self._logger.warning("[parcels] could not find parcels named {}.".format(name))
                    return self.from_cache(name + '-parcels')

        # But if any filters are present, forget the name and build the DataFrame.
        filtered_parcels = self.from_cache('all-parcels')

        shape_str = 'None' if filtered_parcels is None else filtered_parcels.shape
        self._logger.debug("  1. filtered_parcels (from cache) is shape {}".format(shape_str))

        # With samples filters, we'll filter the dataframe
        if isinstance(parcels, list) or isinstance(parcels, pd.Series) or isinstance(parcels, pd.Index):
            filtered_parcels = filtered_parcels[filtered_parcels.index.isin(parcels)]
        elif isinstance(parcels, pd.DataFrame):
            filtered_parcels = filtered_parcels[filtered_parcels.index.isin(parcels.index)]

        shape_str = 'None' if filtered_parcels is None else filtered_parcels.shape
        self._logger.debug("  2. filtered_parcels (from parcels) is shape {}".format(shape_str))

        # By hemisphere, we will restrict to left or right
        # MNI space defines right of mid-line as +x and left of midline as -x
        if h == 'L':
            left_indices = [s for s in filtered_parcels.index if s.startswith('L_')]
            filtered_parcels = filtered_parcels.loc[left_indices, :]
        elif h == 'R':
            right_indices = [s for s in filtered_parcels.index if s.startswith('R_')]
            filtered_parcels = filtered_parcels.loc[right_indices, :]
        elif h == 'A':
            pass
        else:
            self._logger.warning("{} is not interpretable as a hemisphere; ignoring it.".format(hemisphere))

        shape_str = 'None' if filtered_parcels is None else filtered_parcels.shape
        self._logger.debug("  3. filtered_parcels (by hemi) is shape {}".format(shape_str))

        # If we're given a name, and didn't already pull it from cache, cache the filtered DataFrame
        # This will happen with mode= anything other than 'pull', which will return a cached copy if found first
        if cache and name is not None:
            self.to_cache(name + '-parcels', data=filtered_parcels)

        return filtered_parcels

    def probes(self, name=None, probes=None, cache=False):
        """ The probes property
        Asking for ExpressionData.probes will return a dataframe with all probes.
        To get a sub-frame, call samples(data=list_of_wanted_probe_ids).

        :param name: a label used to store and retrieve a specific subset of probe data
        :param probes: a list of probes used to filter probe data
        :param cache: set to True if a cache file should be written with this name.
        :return: a DataFrame full of probe and gene data
        """

        self._logger.debug("probes requested with {} and {} probes.".format(
            "name of '" + name + "'" if name is not None else 'no name',
            len(probes) if probes is not None else 'no list of'
        ))

        # If a name is specified, with no other specs, we just return a cached DataFrame
        if name is not None:
            if probes is None:
                if name in canned_map:
                    return self.from_cache(canned_map[name] + '-probes')
                else:
                    return self.from_cache(name + '-probes')

        # If filters are supplied, we will generate a filtered DataFrame.
        filtered_probes = self.from_cache('all-probes')
        if isinstance(probes, list) or isinstance(probes, pd.Series):
            filtered_probes = filtered_probes.loc[list(probes), :]
        elif isinstance(probes, pd.DataFrame):
            filtered_probes = filtered_probes.loc[probes.index, :]

        # If we're given a name, cache the filtered DataFrame
        if cache and name is not None:
            self.to_cache(name + '-probes', data=filtered_probes)

        return filtered_probes

    def connectivity(self, name=None, samples=None):
        """ The connectivity property

        Asking for connectivity will return a dataframe with all connectivity relations.
        To get a sub-frame, call connectivity(samples=list_of_wanted_well_ids).

        :param name: a label used to store and retrieve a specific subset of probe data
        :param samples: a list of samples used to filter probe data
        :return: a symmetrical DataFrame full of connectivity weights
        """

        self._logger.debug("    - connectivity requested with {} and {} samples.".format(
            "name of '" + name + "'" if name is not None else 'no name',
            len(samples) if samples is not None else 'no list of'
        ))

        if name is None:
            self._logger.debug("No specific connectivity requested, providing INDI by default.")
            name = 'indi'

        # Without filters, we can only return a cached DataFrame.
        if samples is None:
            if name in canned_map:
                return self.from_cache(canned_map[name] + '-conn')
            else:
                return self.from_cache(name + '-conn')

        # If filters are supplied, we will generate a filtered DataFrame.
        filtered_conn = self.from_cache('-'.join([name, 'conn']))
        self._logger.debug("    - filtering connectivity found with {} by {} samples.".format(
            "name of '" + name + "'" if name is not None else 'no name',
            len(samples) if samples is not None else 'no list of'
        ))
        if isinstance(samples, list) or isinstance(samples, pd.Series) or isinstance(samples, pd.Index):
            overlaps = [s for s in filtered_conn.index if s in samples]
            filtered_conn = filtered_conn.loc[overlaps, overlaps]
        elif isinstance(samples, pd.DataFrame):
            overlaps = [s for s in filtered_conn.index if s in samples.index]
            filtered_conn = filtered_conn.loc[overlaps, overlaps]
        self._logger.debug("    - connectivity (overlapping expression) down to [{} X {}].".format(
            filtered_conn.shape[0], filtered_conn.shape[1]
        ))

        # We may eventually cache other named connectivity sets, but not yet.
        # if cache and name is not None:
        #     self.to_cache(name + '-conn', data=filtered_conn)

        return filtered_conn

    def connection_density(self, name=None, samples=None):
        """ The connection_density property
        Asking for connection_density will return a Series with sums of all connectivity for each sample.
        To get a sub-frame, call connection_density(samples=list_of_wanted_well_ids).

        :param name: a label used to store and retrieve a specific subset of probe data
        :param samples: a list of samples used to filter probe data
        :return: a Series full of summed connectivity weights for each sample
        """

        self._logger.debug("    - connection_density requested with {} and {} samples.".format(
            "name of '" + name + "'" if name is not None else 'no name',
            len(samples) if samples is not None else 'no list of'
        ))

        if name is None:
            name = 'indi'

        return self.connectivity(name, samples).sum()

    def connectivity_similarity(self, name=None, samples=None):
        """ return a connectivity similarity matrix, without the influence of self-correlations.

        :param name: a label used to store and retrieve a specific subset of probe data
        :param samples: a list of samples used to filter probe data
        :return: a DataFrame full of connectivity similarity weights for each relation
        """

        self._logger.debug("    - connection_similarity requested with {} and {} samples.".format(
            "name of '" + name + "'" if name is not None else 'no name',
            len(samples) if samples is not None else 'no list of'
        ))

        if name is None:
            name = 'indi'

        conn_df = self.connectivity(name, samples)
        return algorithms.make_similarity(conn_df)

    def add_log_handler(self, handler):
        """ Allow apps using this library to handle its logging output

        :param handler: a logging.Handler object that can listen to PyGEST's output
        """

        # print("!!! manually adding log handler to pygest!!!")
        self._logger.addHandler(handler)

    def configure_logging(self, external_logger):
        """ Set up logging to direct appropriate information to stdout and a logfile.

        :param external_logger: if we're passed a logger or handler, we'll log to it rather than our own.
                                Otherwise, we'll set up a new one here.
        """

        # Set the logger to log EVERYTHING (1). Handlers can filter by level on their own.
        self._logger.setLevel(1)
        if external_logger is not None and isinstance(external_logger, logging.RootLogger):
            self._logger = external_logger
        elif external_logger is not None and isinstance(external_logger, logging.Logger):
            self._logger = external_logger
        elif external_logger is not None and isinstance(external_logger, logging.Handler):
            self._logger.addHandler(external_logger)
        else:
            # print("pygest is configuring its own logging; none was provided.")
            # By default, if no handler is provided, we will log everything to a log file.
            # This should be changed before public use to simply dump all logs to a NULL
            # handler. Output will then be caught only if callers would like to.
            log_formatter = logging.Formatter(
                fmt='%(asctime)s [%(levelname)s] | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

            file_handler_exists = False
            stream_handler_exists = False
            for h in self._logger.handlers:
                if isinstance(h, logging.FileHandler):
                    file_handler_exists = True
                if isinstance(h, logging.StreamHandler):
                    stream_handler_exists = True

            if not file_handler_exists:
                if not os.path.exists(os.path.join(self._dir, 'logs')):
                    os.makedirs(os.path.join(self._dir, 'logs'))
                file_name = 'pygest-' + datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S') + '.log'
                file_handler = logging.FileHandler(os.path.join(self._dir, 'logs', file_name))
                file_handler.setFormatter(log_formatter)
                file_handler.setLevel(logging.DEBUG)
                self._logger.addHandler(file_handler)

            if not stream_handler_exists:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(log_formatter)
                console_handler.setLevel(logging.INFO)
                self._logger.addHandler(console_handler)

        self._logger.info("PyGEST has initialized logging, and is running on host '{}'".format(
            socket.gethostname()
        ))

    def log_status(self, regarding='all'):
        """ Logs a brief summary of the data available (or not)

        :param regarding: allows status to be tailored to the caller's interest - only 'all' coded thus far
        """

        def abi_files_str(d):
            s = " "
            s += "Expr  " if os.path.isfile(self.path_to(d, {'name': 'exp'})) else "      "
            s += "Anno  " if os.path.isfile(self.path_to(d, {'name': 'ann'})) else "      "
            s += "Call  " if os.path.isfile(self.path_to(d, {'name': 'pac'})) else "      "
            s += "Prob  " if os.path.isfile(self.path_to(d, {'name': 'pro'})) else "      "
            s += "Onto  " if os.path.isfile(self.path_to(d, {'name': 'ont'})) else "      "
            anat_dir = os.path.dirname(self.path_to(d, {'name': 'exp'}).replace(BIDS_subdir, 'anat'))
            if os.path.isdir(anat_dir):
                s += "{} anat  ".format(len(os.listdir(anat_dir)))
            else:
                s += "        "
            dwi_dir = os.path.dirname(self.path_to(d, {'name': 'exp'}).replace(BIDS_subdir, 'dwi'))
            if os.path.isdir(dwi_dir):
                s += "{} dwi  ".format(len(os.listdir(dwi_dir)))
            else:
                s += "       "
            return s

        if regarding == 'all':
            donors_of_interest = self.donors()
        else:
            donors_of_interest = [donor_name(regarding), ]

        self._logger.info("    __Donor__ :  __Raw_ABI_Files_available__")
        for donor in donors_of_interest:
            # Determine file names and existence of them
            self._logger.info("    {} : {}".format(donor, abi_files_str(donor)))

        # Summaries
        csv_files = pd.DataFrame(
            columns=['donor', 'path', 'file', 'bytes', ],
            data=[]
        )
        for root, dirs, files in os.walk(os.path.join(self._dir, 'sourcedata')):
            for name in files:
                if name[-4:] == ".csv":
                    donor_string = root[root.find("sub-") + 4:]
                    donor_string = donor_string[:donor_string.find(os.sep)]
                    csv_files = csv_files.append({
                        'donor': donor_string,
                        'path': root,
                        'file': name,
                        'bytes': os.stat(os.path.join(root, name)).st_size,
                    }, ignore_index=True)
        csv_files_of_interest = csv_files.loc[csv_files['donor'].isin(donors_of_interest)]
        self._logger.info("  {n:,} raw files from {d} donor{p} consume {b}.".format(
            n=csv_files_of_interest['bytes'].count(),
            d=len(donors_of_interest),
            p="" if len(donors_of_interest) == 1 else "s",
            b=humanize.naturalsize(csv_files_of_interest['bytes'].sum())
        ))
        self._logger.info("  {n}amples have{s} been imported and cached.".format(
            n="S" if self._cache is None else "{:,} s".format(len(self.samples(name=regarding).index)),
            s=" not" if self._cache is None else ""
        ))
        self._logger.info("  {n}robes have{s} been imported and cached.".format(
            n="P" if self._cache is None else "{:,} p".format(len(self.probes(name='all').index)),
            s=" not" if self._cache is None else ""
        ))

    @staticmethod
    def structure_class(s, part='comment'):
        """ Parse structure_name field from ABI samples and return a sub-string from it

        :param str s: A complete description of a tissue class
        :param part: The part of the tissue class to extract
        :return: A substring from s, representing 'coarse', 'fine', 'side'
        """
        ses = [x.strip() for x in s.split(',')]
        # dict with default values
        sdict = {'full': s, 'side': '', 'coarse': '', 'fine': ''}
        i = -1
        for side in ['left', 'Left', 'right', 'Right']:
            try:
                i = ses.index(side)
                sdict['side'] = side.upper()[0]
                if i > 0:
                    sdict['coarse'] = ", ".join(ses[0: i])
                    ses.remove(ses[i])
                    sdict['fine'] = ", ".join(ses)
            except ValueError:
                pass
            except IndexError:
                pass
        if part == 'comment':
            if i == -1:
                return "no side   : {}".format(s)
            elif len(ses) == 0:
                return "only side : {}".format(s)
            else:
                return "{} parts, side @ {}".format(len(ses), i)
        elif part in sdict.keys():
            if i == -1:
                return s
            else:
                return sdict[part]
        else:
            return "na"

    def build_samples(self, name=None):
        """ Read all SampleAnnot.csv files and concatenate them into one 'samples' dataframe.
        """
        dfs = []
        for donor in self.donors():
            # Load annotation csv file for each donor
            filename = os.path.join(self.path_to(donor, {'name': 'annot'}))
            if os.path.isfile(filename):
                self._logger.debug("  building {d}'s samples from {f} and parsing coordinates".format(
                    d=donor, f=filename
                ))
                df = pd.read_csv(filename, index_col='well_id')

                # Remember which donor these data came from
                df['donor'] = pd.DataFrame(index=df.index, data={'donor': donor})['donor']

                # Convert text representations of coordinates into numeric tuples
                df['vox_xyz'] = df.apply(
                    lambda row: (row['mri_voxel_x'], row['mri_voxel_y'], row['mri_voxel_z']), axis=1
                )
                df = df.drop(labels=['mri_voxel_x', 'mri_voxel_y', 'mri_voxel_z'], axis=1)
                df['mni_xyz'] = df.apply(
                    lambda row: (row['mni_x'], row['mni_y'], row['mni_z']), axis=1
                )
                df = df.drop(labels=['mni_x', 'mni_y', 'mni_z'], axis=1)

                # Pull levels of tissue type out of structure_name field
                df['coarse_name'] = df['structure_name'].apply(
                    lambda x: self.structure_class(x, 'coarse')
                )
                df['fine_name'] = df['structure_name'].apply(
                    lambda x: self.structure_class(x, 'fine')
                )
                df['side'] = df['structure_name'].apply(
                    lambda x: self.structure_class(x, 'side')
                )

                # Cache this to disk as a named dataframe
                self._logger.debug("  disk-caching samples to {f}".format(f=self.cache_path(donor + '-samples')))
                self.to_cache(donor + '-samples', df)

                dfs.append(df)
            else:
                self._logger.debug("  skipping {}, no samples.".format(donor))

        df = pd.concat(dfs, axis=0)
        self._logger.debug("  caching samples to {f}".format(f=self.cache_path('all-samples')))
        self.to_cache('all-samples', df)

        if name is not None:
            try:
                key = canned_map[name.split(sep='-')[0]]
            except KeyError:
                key = name
            self._logger.debug("  [samples] seeking to cache {} as {}".format(name, key))
            if key == 'richiardi':
                self.to_cache('richiardi-samples', df[df.index.isin(rawdata.get_richiardi_samples())])
            elif key == 'fornito':
                self.to_cache('glasser-samples', df[df.index.isin(rawdata.get_glasser_samples())])
            elif key == 'schmidt':
                self.to_cache('schmidt-samples', df[df.index.isin(rawdata.get_schmidt_samples())])
            elif key == 'test':
                self.to_cache('test-samples', df[df.index.isin(rawdata.get_test_samples())])

    def build_probes(self, name=None):
        """ Read any one Probes.csv file and save it into a 'probes' dataframe.
        """
        donor = donor_name('any')
        filename = os.path.join(self.path_to(donor, {'name': 'probes'}))
        if os.path.isfile(filename):
            self._logger.debug("  building probes from {}".format(filename))
            df = pd.read_csv(filename, index_col='probe_id')

            self._logger.debug("  caching probes to {f}".format(f=self.cache_path('all-probes')))
            self.to_cache('all-probes', df)

            if name is not None:
                try:
                    key = canned_map[name.split(sep='-')[0]]
                except KeyError:
                    key = name
                self._logger.debug("  [probes] seeking to cache {} as {}".format(name, key))
                if key == 'richiardi':
                    self.to_cache('richiardi-probes', df[df.index.isin(rawdata.get_richiardi_probes())])
                elif key == 'fornito':
                    self.to_cache('fornito-probes', df[df.index.isin(rawdata.get_fornito_probes())])
                elif key == 'test':
                    self.to_cache('test-probes', df[df.index.isin(rawdata.get_test_probes())])
        else:
            self._logger.debug("  ignoring request to build {} probes, they don't exist.".format(donor))

    def build_connectivity(self, name=None):
        """ Read any one {name}-conn.df file and save it into a 'connectivity' dataframe.
            Currently, only one connectivity matrix exists. In future, this will need to
            expand into additional options, including passing in arbitrary csv files with headers.
        """

        if name is None:
            # default connectivity data, if left unspecified
            name = 'indi-connectivity'

        if os.path.isfile(name):
            filename = name
        elif os.path.isfile(os.path.join(self.path_to('conn', {'name': bids_clean_filename(name)}))):
            filename = self.path_to('conn', {'name': bids_clean_filename(name)})
        else:
            filename = self.path_to('conn', {'name': name})

        if os.path.isfile(filename):
            self._logger.debug("  building connectivity from {f}".format(f=filename))
            with open(filename, 'rb') as f:
                df = pickle.load(f)

            self._logger.debug("  caching connectivity to {f}".format(f=self.cache_path(name)))
            self.to_cache(name, df)
        else:
            self._logger.debug("  NOT building connectivity, no {} connectivity found.".format(name))

        if name is not None:
            try:
                key = canned_map[name.split(sep='-')[0]]
            except KeyError:
                key = name
            self._logger.debug("  [connectivity] seeking to cache {} as {}".format(name, key))
            if key == 'indi':
                # Already done above
                pass
            elif key == 'all':
                # This is the default, and already dealt with above
                pass

    def build_expression(self):
        """ Read all MicroarrayExpression.csv files and concatenate them into one 'expression' dataframe.

        This should only be called internally because the expression getter will dynamically detect whether
        it should use an in-memory cache, and on-disk cache, or have to call this function.
        """
        self._logger.debug("  building expression data")
        dfs = []
        for donor in self.donors():
            # Determine file names and existence of them
            filename_annot = self.path_to(donor, {'name': 'annot'})
            filename_expr = self.path_to(donor, {'name': 'expr'})

            if os.path.isfile(filename_annot) and os.path.isfile(filename_expr):
                # For each donor, load the well_ids (indices to expression data) and expression data
                self._logger.debug("    loading {d}'s well_ids from {f}".format(d=donor, f=filename_annot))
                df_ann = pd.read_csv(filename_annot)

                # Load the annotation, using prev
                self._logger.debug("    loading expression data from {f}".format(f=filename_expr))
                df_exp = pd.read_csv(filename_expr, header=None, index_col=0, names=df_ann['well_id'])

                # Cache this to disk as a named dataframe
                self._logger.debug("    disk-caching expression to {f}".format(
                    f=self.cache_path(donor + '-expression')
                ))
                self.to_cache(donor + '-expression', df_exp)

                # Then add this donor's data to a list for concatenation later.
                dfs.append(df_exp)
            else:
                self._logger.debug("  skipping {}, no expression data.".format(donor))

        self._logger.debug("  caching expression to {f}".format(f=self.cache_path('all-expression')))
        self.to_cache('all-expression', pd.concat(dfs, axis=1))

    def path_to(self, thing, file_dict=None):
        """ provide a full file path based on any donor and file shorthand we can map.
        """

        if file_dict is None:
            file_dict = {}

        if thing == 'base':
            return self._dir
        elif thing == 'conn':
            return os.path.join(self._dir, 'conn', file_dict['name'] + '.df')
        elif thing == 'derivatives':
            return os.path.join(self._dir, 'derivatives')
        elif thing == 'sourcedata':
            return os.path.join(self._dir, 'sourcedata')
        elif thing == 'genome':
            return os.path.join(self._dir, 'genome')
        elif thing == 'splits':
            return os.path.join(self._dir, 'splits')
        elif thing in self.donors():
            return os.path.join(self._dir, 'sourcedata', 'sub-' + donor_name(thing), BIDS_subdir,
                                file_map[file_dict['name']])
        else:
            logging.warning("path_to({}, dict) called. I cannot understand {}. Returning base {} dir.".format(
                thing, thing, self._dir
            ))
            return os.path.join(self._dir)

    def to_cache(self, name, data):
        """ Save data to disk and hold a reference in memory.
        """
        # Standardize the cache name before writing to file
        clean_name = name.split(sep='-')
        if len(clean_name) != 2:
            error_string = "Asked to cache a {} as '{}'".format(type(data), name)
            error_string += ". Cache names must be two-part, '[name]-[type]',"
            error_string += " like 'all-expression' or 'H03512002-samples'"
            raise ValueError(error_string)
        if clean_name[1][0].lower() == 's':
            clean_name[1] = 'samples'
        elif clean_name[1][0].lower() == 'p':
            clean_name[1] = 'probes'
        elif clean_name[1][0].lower() == 'e':
            clean_name[1] = 'expression'
        elif clean_name[1][0].lower() == 'c':
            clean_name[1] = 'connectivity'
        else:
            raise ValueError("The second part (post-hyphen) of a cache name should begin with 'e', 'p', or 's'")

        # If we have a copy in memory, write it to disk. Else complain.
        if isinstance(data, pd.DataFrame):
            data.to_pickle(self.cache_path(name))
            # In-memory cache
            self._cache.loc["-".join(clean_name)] = {
                'name': clean_name[0],
                'type': clean_name[1],
                'dataframe': data,
                'file': self.cache_path("-".join(clean_name)),
            }
            self._logger.debug("  added {} to cache, now {} records".format(
                name, len(self._cache.index)
            ))
        else:
            self._logger.warning("Cannot cache {}. Only Pandas DataFrame objects are cache-able for now.".format(
                type(data)
            ))

    def from_cache(self, name, refresh='auto'):
        """ Return cached data, or None if I can't find it.

        :param name: a two-part name that species the [name]-[type] parts of a dataset
        :param refresh: 'always' will rebuild caches from raw data,
                        'never' will only check existing cache;
                        'auto' will check cache first, then rebuild if necessary.
        """

        self._logger.debug("Asked to pull '{}' from cache (refresh set to '{}').".format(name, refresh))

        # Names can be abbreviated, causing cache misses if not fixed. We fix them with lookup here.
        name_parts = name.lower().split(sep="-")
        try:
            name_parts[0] = canned_map[name_parts[0]]
        except KeyError:
            pass
        clean_name = "-".join(name_parts)

        # If the call FORCES a rebuild, do it first.
        if refresh == 'always':
            self.build_probes(clean_name)
            self.build_samples(clean_name)
            self.build_expression()
            self.build_connectivity(clean_name)

        # If a full two-part name is provided, and found in memory, get it over with.
        if clean_name in self._cache.index:
            self._logger.debug("  found {} in memory".format(clean_name))
            return self._cache.loc[clean_name, 'dataframe']
        else:
            self._logger.debug("  {} not found in memory".format(clean_name))

        # One particular list is available in code, no need to search the disk.
        if clean_name == "glasser-samples":
            return pd.DataFrame.from_dict(glasser.glasser_parcel_map, orient='index')

        # If it's not in memory, check the disk.
        if not os.path.isdir(self.cache_path("")):
            os.makedirs(self.cache_path(""), exist_ok=True)
        if os.path.isfile(self.cache_path(clean_name)):
            self._logger.debug("  found {} cached on disk, loading...".format(clean_name))
            self._cache.loc[clean_name] = {
                'name': clean_name.split(sep='-')[0],
                'type': clean_name.split(sep='-')[1],
                'dataframe': pd.read_pickle(self.cache_path(clean_name)),
                'file': self.cache_path(clean_name)
            }
            return self._cache.loc[clean_name, 'dataframe']
        else:
            self._logger.debug("  {} not found on disk".format(clean_name))

        # No cached data were found. we need to generate them, unless we did to start.
        if refresh == 'auto':
            try:
                if clean_name.split(sep='-')[1][0].lower() == 'p':
                    self.build_probes(clean_name)
                elif clean_name.split(sep='-')[1][0].lower() == 's':
                    self.build_samples(clean_name)
                elif clean_name.split(sep='-')[1][0].lower() == 'e':
                    self.build_expression()
                elif clean_name.split(sep='-')[1][0].lower() == 'c':
                    self.build_connectivity(clean_name)
                else:
                    self.build_probes(clean_name)
                    self.build_samples(clean_name)
                    self.build_expression()
                    self.build_connectivity(clean_name)

            except IndexError:
                error_string = "Cache names should be two-part, [name]-[type], with type being either "
                error_string += "expr..., samp..., or prob..., like 'all-expression' or 'H03512002-samples'. "
                error_string += "Nothing I can do with '{}'. "
                raise ValueError(error_string.format(name))

        # And now that the cache has been filled, try the updated cache one more last time
        self._logger.debug("Going to re-check the cache. Missed once, but should have filled it now.")
        if clean_name in self._cache.index:
            return self._cache.loc[clean_name, 'dataframe']
        else:
            self._logger.debug("{} was not in the [{}] cache.".format(clean_name, list(self._cache.index)))

        # If still no match, not much we can do.
        return None

    def cache_path(self, name):
        """ prepend a path, and append an extension to the base filename provided

        :param name: base filename needing an appropriate path and extension
        :return: a fully formed absolute path containing the base filename provided
        """
        if name == "":
            return os.path.join(self._dir, 'cache')
        return os.path.join(self._dir, 'cache', name + '.df')

    def distance_matrix(self, samples, sample_type='wellid'):
        """ return a distance matrix between all samples in samples.
        :param samples: list or series of well_ids to be included in the distance matrix
        :param sample_type: Are we dealing with wellids, or parcels, or something else?

        """

        # Let distance_dataframe do the work (DRY)
        return self.distance_dataframe(samples, sample_type).to_numpy()

    def distance_dataframe(self, samples, sample_type='wellid'):
        """ return a distance matrix between all samples in samples.
        :param samples: list or series of well_ids to be included in the distance matrix
        :param sample_type: Are we dealing with wellids, or parcels, or something else?

        """
        if sample_type.lower() in ['wellid', 'well_id']:
            df = pd.DataFrame(self.samples(samples=samples)['mni_xyz'].apply(pd.Series))
        elif sample_type.lower() in ['glasser', 'parcel', 'parcelid', 'parcel_id']:
            df = pd.DataFrame(self.parcels(parcels=samples)['mni_xyz'].apply(pd.Series))
        else:
            df = pd.DataFrame()
        print("Building distance matrix from {} samples, which resulted in {} df".format(
            len(samples), df.shape
        ))
        df = pd.DataFrame(data=distance_matrix(df, df), index=df.index, columns=df.index)
        valid_samples = [s for s in df.index if s in samples]

        # Return the matrix as close to original requested sample-order as possible.
        return df.loc[valid_samples, valid_samples]

    def distance_vector(self, samples, sample_type='wellid'):
        """ return a distance vector (lower triangle of matrix) between all samples in samples.
        :param samples: list or series of well_ids to be included in the distance vector
        :param sample_type: Are we dealing with wellids, or parcels, or something else?

        """

        # Importantly, on matrix creation, columns and rows are ordered to match the order of samples provided.
        # So the lower triangular vector we return is from that ordering, too.
        m = self.distance_matrix(samples, sample_type=sample_type)
        return m[np.tril_indices_from(m, k=-1)]

    def map(self, from_term, to_term):
        """ get a dictionary mapping from_term keys to to_term values

        :param from_term: the term the caller has
        :param to_term: the term the caller wants
        :return: a dictionary mapping from_term to to_term
        """

        # Only so many columns in these datasets can serve as keys (unique to each observation)
        valid_probe_keys = ['probe_id', 'probe_name', ]
        valid_sample_keys = ['well_id', ]

        from_terms = None
        to_terms = None

        if from_term == 'probe_id':
            from_terms = self.probes('all').index
        elif from_term == 'well_id':
            from_terms = self.samples('all').index
        elif from_term in valid_probe_keys:
            from_terms = self.probes('all')[from_term]
        elif from_term in valid_sample_keys:
            from_terms = self.samples('all')[from_term]

        if to_term == 'probe_id':
            to_terms = self.probes('all').index
        elif to_term == 'well_id':
            to_terms = self.samples('all').index
        elif to_term in self.probes('all').columns:
            to_terms = self.probes('all')[to_term]
        elif to_term in self.samples('all').columns:
            to_terms = self.samples('all')[to_term]

        if from_terms is not None and to_terms is not None:
            if len(from_terms) == len(to_terms):
                return dict(pd.Series(to_terms, index=from_terms))
            else:
                print("Mapping requires equal sizes for 1:1 maps. {} has {} values; {} has {}.".format(
                    from_term, len(from_terms),
                    to_term, len(to_terms)
                ))

        # Hopefully, we've done our job. If not, something went wrong.
        if from_term not in valid_probe_keys + valid_sample_keys:
            raise KeyError("The term, \"{}\" is not a valid key for probes nor samples.".format(from_term))
        if to_term not in self.probes('all').columns and to_term not in self.samples('all').columns:
            raise KeyError("The term, \"{}\" was not found in probes nor samples.".format(to_term))

        return {}

    def all_files(self, ext="json"):
        return all_files_in(self._dir, ext)

    def derivatives_old(self, filters, exclusions=None):
        """ Scan through all results matching provided filters and return a list of files

        This is the older, more complicated, although faster version of "derivatives". A simpler,
        yet slower version has replaced it below.

        :param dict filters: dictionary with key-value pairs restricting the results
        :param list exclusions: a list of terms, which if substrings in the filepath, exclude it, regardless of filters
        :return: a list of paths surviving the filters
        """

        # This function is much faster, but the next function "derivatives" is simpler to read and understand.
        # This has been deprecated for the newer, more-readable slow code.

        if exclusions is None:
            exclusions = []
        if not isinstance(exclusions, list):
            exclusions = [exclusions, ]

        def val_ok(k, v, fs):
            """ Return True if this key-value pair passes through the filters dict """
            if k in fs:
                # A restriction on k is specified
                if fs[k] != v and fs[k] != 'all':
                    # The value specified violates that specification, but 'any' filters may still pass
                    if fs[k] == 'any' and v not in ['none', '']:
                        # a spec of 'any' indicates all values are OK except 'none'; 'none' still passes
                        return True
                    if fs[k] == 'none' and v in ['none', '']:
                        # a spec of 'none' indicates either 'none' or '' still pass
                        return True
                    if '+' in v:
                        # print("        {} has multiple values".format(fs[k]))
                        # multiple values are specified. Pass if any of them match
                        if fs[k] in v.split('+'):
                            # print("found {} in {}".format(fs[k], v))
                            return True
                    return False
            # We did not encounter any exclusions, so this f is OK
            return True

        def exclusions_ok(term, excls):
            """ Return False if any term in exclusions is a substring of term. """
            for exclusion in excls:
                if exclusion in term:
                    return False
            return True

        curves = []
        for base_dir in sorted(os.listdir(self.path_to('derivatives', {}))):
            base_path = os.path.join(self.path_to('derivatives', {}), base_dir)
            # Match on subject, by bids dirname
            if os.path.isdir(base_path) and val_ok("sub", donor_name(bids_val("sub", base_dir)), filters) \
                                        and val_ok("hem", bids_val("hem", base_dir), filters) \
                                        and val_ok("samp", bids_val("samp", base_dir), filters):
                for mid_dir in os.listdir(base_path):
                    mid_path = os.path.join(base_path, mid_dir)
                    if os.path.isdir(mid_path) and val_ok("tgt", bids_val("tgt", mid_dir), filters) \
                                               and val_ok("algo", bids_val("algo", mid_dir), filters) \
                                               and val_ok("shuf", bids_val("shuf", mid_dir), filters):
                        for file in os.listdir(mid_path):
                            file_path = os.path.join(mid_path, file)
                            if os.path.isfile(file_path) and file[-4:] == ".tsv" \
                                                         and val_ok("comp", bids_val("comp", file), filters) \
                                                         and val_ok("mask", bids_val("mask", file), filters) \
                                                         and val_ok("adj", bids_val("adj", file), filters):
                                if exclusions_ok(file_path, exclusions):
                                    curves.append(file_path)
        return curves

    def derivatives(self, filters, shuffle='none', as_df=False):
        """ Scan through all results matching provided filters and return a list of files

        :param dict filters: dictionary with key-value pairs restricting the results
            example {'donor': 'H03511009', 'ctx': 'cor'} will return all cortical results from donor H03511009
        :param str shuffle: 'none' or False for real runs, 'agno' 'dist' or 'edges' for null distributions
        :param bool as_df: True causes return of a DataFrame containing 'path' Series. The default is a list of paths.
        :return: a list of paths surviving the filters
        """

        # Gather all files, then repeatedly filter them by our filters criteria.
        curves = self.all_files(ext='tsv')
        for filter_key in filters:
            if filter_key == 'exclusions':
                for excl in filters[filter_key]:
                    curves = curves[~curves['name'].str.contains(excl) & ~curves['root'].str.contains(excl)]
            else:
                if isinstance(filters[filter_key], list):
                    curves = curves[curves[filter_key].isin(filters[filter_key])]
                else:
                    curves = curves[curves[filter_key] == filters[filter_key]]

        if shuffle != 'all':
            curves = curves[curves['root'].str.contains('shuf-{}'.format(shuffle))]

        # Make a full path for easy file reading and sort by it
        if len(curves) > 0:
            curves['path'] = curves.apply(lambda row: os.path.join(row['root'], row['name']), axis=1)
            curves['shuffle'] = shuffle
            curves = curves.sort_values(by=['path'], ascending=True)
            curves.index = range(len(curves.index))

            # Having the full dataframe eases changing this in the future, but we just need a list of paths for now.
            if as_df:
                return curves
            else:
                return list(curves['path'])
        else:
            if as_df:
                return pd.DataFrame()
            else:
                return []
