import os
import sys
import errno
import datetime

import requests
import humanize
import zipfile
import logging

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

# Get strings & dictionaries & DataFrames from the local (not project) config
from pygest import donors, donor_map
from pygest.convenience import file_map, BIDS_subdir, donor_files, data_sections, canned_map,\
    aba_downloads, richiardi_probes, richiardi_samples, type_map

# import utility  # local library containing a hash_file routine


class ExpressionData(object):
    """ Wrap Allen Brain Institute expression data

    The general strategy is to keep track of what data we have, and in which state.
    We can then provide data to callers, deciding intelligently whether to send them
    cached data, restructure from an upstream source, or download from ABA and start
    from scratch. We can then use the source data to present users with filtered
    versions many ways.

    TODO: A current difficulty is in reporting to a caller how long something may take.
          If we're going to be running for an hour to download data, or running for a
          day to do repeated correlations on many gene sets, it's best to kick off the
          job and inform them when it's done. But there are many ways to do this, none
          of them are necessarily very pythonic.
    TODO: Load up Richiardi's probe-set mapped with appropriate indices as a named and cached probes file.
    """

    # Remember the base directory for all data. Default can be overridden in __init__
    _dir = '/data'

    # Create an empty dataframe to hold information about existing files.
    _files = pd.DataFrame(
        columns=['section', 'donor', 'path', 'file', 'bytes', 'hash', 'full_path', ],
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

    def __init__(self, data_dir, handler=None):
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
        self.configure_logging(handler)

        # Clear the cache, but not the disk, then
        # take a look at the data directory and log what we have access to.
        self.refresh()

        # We need about 5GB of disk just for raw data, plus however much assists with caching
        req_space = aba_downloads['expanded_bytes'].sum()
        if self.space_available() < req_space - self.space_used():
            raise ResourceWarning(
                "Expression data require at least {req}. {d} has {a}.".format(
                    req=humanize.naturalsize(req_space - self.space_used()),
                    d=self._dir,
                    a=humanize.naturalsize(self.space_available()),
                )
            )

    def expression(self, name=None, probes=None, samples=None, mode='pull'):
        """ The expression property
        Filterable by probe rows or sample columns

        :param name: a label used to store and retrieve a specific subset of expression data
        :param probes: a list of probes used to filter expression data
        :param samples: a list of samples (well_ids) used to filter expression data
        :param mode: 'pull' to attempt pulling named expression from cache first. 'push' to build them, then store them
        """

        # Without filters, we can only return a cached DataFrame.
        if probes is None and samples is None:
            if name is not None:
                return self.from_cache(name + '-expression')
            else:
                return self.from_cache('all-expression')

        # With filters, we will generate a filtered DataFrame.
        filtered_expr = self.from_cache('all-expression')

        if isinstance(probes, list) or isinstance(probes, pd.Series):
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
        if name is not None:
            self.to_cache(name + '-expression', data=filtered_expr)

        return filtered_expr

    def samples(self, name=None, samples=None, donor=None, hemisphere=None, mode='pull'):
        """ The samples property
        Asking for ExpressionData.samples will return a dataframe with all samples.
        To get a sub-frame, call samples(data=list_of_wanted_well_ids).

        :param name: a label used to store and retrieve a specific subset of sample data
        :param samples: a list of samples (wellids) used to filter sample data
        :param donor: specifying donor will constrain the returned samples appropriately.
        :param hemisphere: specifying left or right will constrain the returned samples appropriately.
        :param mode: 'pull' to attempt pulling named samples from cache first. 'push' to build samples, then store them
        :return: A DataFrame indexed by well_id, with sample information
        """

        # Do some interpretation before the heavy lifting.
        h = 'A' if hemisphere is None else hemisphere[0].upper()
        self._logger.debug("[samples] got hemisphere of '{}', which becomes '{}'".format(hemisphere, h))

        # If a name is specified, all by itself, our job is easy.
        if name is not None and ((samples is None and donor is None and hemisphere is None) or mode == 'pull'):
            if name in canned_map:
                return self.from_cache(canned_map[name] + '-samples')
            else:
                self._logger.warning("[samples] could not find samples named {}.".format(name))
                return self.from_cache(name + '-samples')
                # TODO: What if the file doesn't exist? check from_cache code
                # return None

        # But if any filters are present, forget the name and build the DataFrame.
        filtered_samples = None
        if donor is not None:
            if donor in donor_map:
                filtered_samples = self.from_cache(donor_map[donor] + '-samples')
            else:
                self._logger.warning("[samples] donor {} was not recognized, using the full sample set.".format(donor))

        shape_str = 'None' if filtered_samples is None else filtered_samples.shape
        self._logger.debug("  1. filtered_samples (from donor) is shape {}".format(shape_str))

        # If we didn't get a donor, start with a full sample set.
        if filtered_samples is None:
            filtered_samples = self.from_cache('all-samples')

        shape_str = 'None' if filtered_samples is None else filtered_samples.shape
        self._logger.debug("  2. filtered_samples (from cache) is shape {}".format(shape_str))

        # With samples filters, we'll filter the dataframe
        if isinstance(samples, list) or isinstance(samples, pd.Series):
            filtered_samples = filtered_samples[filtered_samples.index.isin(samples)]
        elif isinstance(samples, pd.DataFrame):
            filtered_samples = filtered_samples[filtered_samples.index.isin(samples.index)]

        shape_str = 'None' if filtered_samples is None else filtered_samples.shape
        self._logger.debug("  3. filtered_samples (from samples) is shape {}".format(shape_str))

        # By hemisphere, we will restrict to left or right
        # MNI space defines right of mid-line as +x and left of midline as -x
        if h == 'L':
            l_filter = pd.DataFrame(filtered_samples['mni_xyz'].tolist(),
                                    index=filtered_samples.index,
                                    columns=['x', 'y', 'z']).x < 0
            filtered_samples = filtered_samples[l_filter]
        elif h == 'R':
            r_filter = pd.DataFrame(filtered_samples['mni_xyz'].tolist(),
                                    index=filtered_samples.index,
                                    columns=['x', 'y', 'z']).x > 0
            filtered_samples = filtered_samples[r_filter]
        elif h == '0':
            m_filter = pd.DataFrame(filtered_samples['mni_xyz'].tolist(),
                                    index=filtered_samples.index,
                                    columns=['x', 'y', 'z']).x == 0
            filtered_samples = filtered_samples[m_filter]
        elif h == 'A':
            pass
        else:
            self._logger.warning("{} is not interpretable as a hemisphere; ignoring it.".format(hemisphere))

        shape_str = 'None' if filtered_samples is None else filtered_samples.shape
        self._logger.debug("  4. filtered_samples (by hemi) is shape {}".format(shape_str))

        # If we're given a name, and didn't already pull it from cache, cache the filtered DataFrame
        # This will happen with mode= anything other than 'pull', which will return a cached copy if found first
        if name is not None:
            self.to_cache(name + '-samples', data=filtered_samples)

        return filtered_samples

    def probes(self, name=None, probes=None):
        """ The probes property
        Asking for ExpressionData.probes will return a dataframe with all probes.
        To get a sub-frame, call samples(data=list_of_wanted_probe_ids).

        :param name: a label used to store and retrieve a specific subset of probe data
        :param probes: a list of probes used to filter probe data
        :return: a DataFrame full of probe and gene data
        """

        self._logger.debug("probes requested with {} and {} probes.".format(
            "name of '" + name + "'" if name is not None else 'no name',
            len(probes) if probes is not None else 'no list of'
        ))

        # Without filters, we can only return a cached DataFrame.
        if probes is None:
            if name is not None:
                if name in canned_map:
                    return self.from_cache(canned_map[name] + '-probes')
                else:
                    return self.from_cache(name + '-probes')
            else:
                return self.from_cache('all-probes')

        # If filters are supplied, we will generate a filtered DataFrame.
        filtered_probes = self.from_cache('all-probes')
        if isinstance(probes, list) or isinstance(probes, pd.Series):
            filtered_probes = filtered_probes.loc[list(probes), :]
        elif isinstance(probes, pd.DataFrame):
            filtered_probes = filtered_probes.loc[probes.index, :]

        # If we're given a name, cache the filtered DataFrame
        if name is not None:
            self.to_cache(name + '-probes', data=filtered_probes)

        return filtered_probes

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

        # Wipe the cache if asked to start_from_scratch
        if clean:
            path_to_cache = os.path.sep.join(self.cache_path('_').split(sep=os.path.sep)[:-1])
            for f in os.listdir(path_to_cache):
                f_path = os.path.join(path_to_cache, f)
                try:
                    if os.path.isfile(f_path):
                        # TODO: Test this on several machines and paths, then remove the comment.
                        print("  would rm {}".format(f_path))
                        # os.remove(f_path)
                except Exception as e:
                    print(e)
            pass

        # Find/make each section(subdirectory).
        for section in data_sections:
            if os.path.isdir(os.path.join(self._dir, section)):
                self.find_files_in(section)
            else:
                os.mkdir(os.path.join(self._dir, section))
                if os.path.isdir(os.path.join(self._dir, section)):
                    self._logger.warning("{} could neither be found nor created at {}".format(
                        section, self._dir
                    ))

        return self.status()

    def add_log_handler(self, handler):
        """ Allow apps using this library to handle its logging output

        :param handler: a logging.Handler object that can listen to PyGEST's output
        """

        self._logger.addHandler(handler)

    def configure_logging(self, handler):
        """ Set up logging to direct appropriate information to stdout and a logfile.

        :param handler: if we're passed a handler, we'll log to it rather than our own. Otherwise, we'll set up here.
        """

        # Set the logger to log EVERYTHING (0). Handlers can filter by level on their own.
        self._logger.setLevel(0)
        if handler is not None and isinstance(handler, logging.Handler):
            self._logger.addHandler(handler)
        else:
            # By default, if no handler is provided, we will log everything to a log file.
            # This should be changed before public use to simply dump all logs to a NULL
            # handler. Output will then be caught only if callers would like to.
            log_formatter = logging.Formatter(
                fmt='%(asctime)s [%(levelname)s] | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

            file_name = 'pygest-' + datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S') + '.log'
            file_handler = logging.FileHandler(os.path.join(self._dir, 'logs', file_name))
            file_handler.setFormatter(log_formatter)
            file_handler.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(log_formatter)
            console_handler.setLevel(logging.INFO)

            self._logger.addHandler(file_handler)
            self._logger.addHandler(console_handler)

    def find_files_in(self, section):
        self._logger.debug("Refreshing {d}".format(d=os.path.join(self._dir, section)))
        if section == 'BIDS':
            for donor in donors:
                for donor_file in donor_files:
                    path = os.path.join(self._dir, section, 'sub-' + donor, BIDS_subdir, donor_file)
                    if os.path.isfile(path):
                        self._logger.debug("  found {f}, hashing it...".format(f=path))
                        self._files.append({
                            'section': section,
                            'donor': donor,
                            'path': os.path.join(self._dir, section, 'sub-' + donor),
                            'file': donor_file,
                            'bytes': os.stat(path).st_size,
                            'hash': '0',  # utility.hash_file(path),
                            'full_path': path,
                        }, ignore_index=True)
        elif section == 'cache':
            for f in os.listdir(os.path.join(self._dir, section)):
                path = os.path.join(os.path.join(self._dir, section), f)
                if os.path.isfile(path):
                    pass
                    # TODO: Persist this stuff, and check by altered date. Don't waste time hashing every init.
                    # TODO: Is this wasted time? For now, we'll never check it and don't care about the hash.
                    self._logger.debug("  found {f}, not hashing it...".format(f=path))
                    # self._files.append({
                    #     'section': section,
                    #     'donor': '',
                    #     'path': os.path.dirname(path),
                    #     'file': os.path.basename(path),
                    #     'bytes': os.stat(path).st_size,
                    #     'hash': '0',  # utility.hash_file(path),
                    #     'full_path': path,
                    # }, ignore_index=True)
                    # Don't bother remembering these. We will check for files dynamically, on request.
        elif section == 'downloads':
            self._logger.debug("  ignoring {} for now".format(section))
            """
            for ind, row in aba_downloads[:5].iterrows():
                path = os.path.join(self._dir, section, row['zip_file'])
                if os.path.isfile(path):
                    # TODO: Persist this stuff, and check by altered date. Don't waste time hashing every init.
                    self._logger.debug("  found {f}, hashing it...".format(f=path))
                    self._files.append({
                        'section': section,
                        'donor': donor_map[ind],
                        'path': os.path.dirname(path),
                        'file': os.path.basename(path),
                        'bytes': os.stat(path).st_size,
                        'hash': '0',  # utility.hash_file(path),
                        'full_path': path,
                    }, ignore_index=True)
            # TODO: If we ever download other data, build code to look for it here. Other files are just ignored.
            """
        else:
            self._logger.debug("  ignoring {} for now".format(section))

    def status(self, regarding='all'):
        """ Return a brief summary of the data available (or not)

        :param regarding: allows status to be tailored to the caller's interest - only 'all' coded thus far
        """
        files_string = "{n} files from {d} donors consume {b}.".format(
            n=self._files['bytes'].count(),
            d=len(self._files['donor'].unique()),
            b=humanize.naturalsize(self._files['bytes'].sum())
        )
        # TODO: count samples
        samples_string = "Samples have{s} been imported.".format(
            s=" not" if self._cache is None else ""
        )
        # TODO: count probes
        probes_string = "Probes have{s} been imported.".format(
            s=" not" if self._cache is None else ""
        )
        # TODO: enumerate cache files and list them with their sizes in caches_string
        if regarding == 'all':
            return "\n".join([files_string, samples_string, probes_string])
        elif regarding == 'files':
            return files_string
        elif regarding == 'probes':
            return probes_string
        elif regarding == 'samples':
            return samples_string
        else:
            return "I don't recognize the term {}, and cannot report a status for it.".format(regarding)

    def download(self, donor, base_url=None):
        """ Download zip file from Allen Brain Institute and return the path to the file.

        :param donor: Any donor string that maps to a BIDS-compatible name
        :param base_url: an alternative url containing the zip files needed
        :returns: A string representation of the full path of the downloaded zip file
        # TODO: Making this asynchronous would be nice.
        """
        if donor not in donor_map:
            self._logger.warning("Not aware of donor named {}. Nothing I can download.".format(donor))
            return None
        url = aba_downloads.loc[donor_map[donor], 'url']
        zip_file = aba_downloads.loc[donor_map[donor], 'zip_file']
        # Allow for override of url if we are just testing or want to keep it local
        if base_url is not None:
            url = base_url + '/' + zip_file
        self._logger.info("Downloading {} from {} to {} ...".format(
            zip_file,
            url,
            os.path.join(self._dir, 'downloads')
        ))
        r = requests.get(url=url, stream=True)
        with open(os.path.join(self._dir, 'downloads', zip_file), 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 ** 2):
                if chunk:
                    f.write(chunk)
        return zip_file

    def extract(self, donor, clean_up=True, add_to_tsv=True):
        """ Extract data from downloaded zip file into BIDS-compatible subdirectory.

        :param donor: Any donor string that maps to a BIDS-compatible name
        :param clean_up: default True, indicates the zip file should be deleted after extraction
        :param add_to_tsv: default True, adds the donor name to the participant.tsv file, maintaining BIDS compliance.
        :returns: A string representation of the path where the data files were extracted
        # TODO: Making this asynchrnous would be nice.
        """
        donor_name = donor_map[donor]
        extract_to = os.path.join(self._dir, aba_downloads.loc[donor_name, 'subdir'], BIDS_subdir)
        # We could try/except this, but we'd just have to raise another error to whoever calls us.
        os.makedirs(extract_to)
        zip_file = os.path.join(self._dir, 'downloads', aba_downloads.loc[donor_name, 'zipfile'])
        self._logger.info("Extracting {} to {} ...".format(
            zip_file,
            extract_to
        ))
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(extract_to)
        if clean_up:
            self._logger.info("  cleaning up by removing {}".format(zip_file))
            os.remove(zip_file)
        if add_to_tsv:
            self._logger.info("  adding {} to participants.tsv".format(donor_name))
            tsv_file = os.path.join(self._dir, 'participants.tsv')
            with open(tsv_file, 'a') as f:
                f.write(donor_name + "\n")
        self.refresh()
        return extract_to

    def space_available(self):
        return os.statvfs(self._dir).f_bsize * os.statvfs(self._dir).f_bavail

    def space_used(self):
        return self._files['bytes'].sum()

    def build_samples(self, name=None):
        """ Read all SampleAnnot.csv files and concatenate them into one 'samples' dataframe.
        # TODO: For a given name, only load what's necessary. We currently load everything.
        """
        dfs = []
        for donor in donors:
            # Load annotation csv file for each donor
            filename = os.path.join(self.path_to(donor, 'annot'))
            self._logger.debug("  building {d}'s samples from {f} and parsing coordinates".format(d=donor, f=filename))
            df = pd.read_csv(filename, index_col='well_id')

            # Remember which donor these data came from
            df['donor'] = pd.DataFrame(index=df.index, data={'donor': donor})['donor']

            # Convert text representations of coordinates into numeric tuples
            df['vox_xyz'] = df.apply(lambda row: (row['mri_voxel_x'], row['mri_voxel_y'], row['mri_voxel_z']), axis=1)
            df = df.drop(labels=['mri_voxel_x', 'mri_voxel_y', 'mri_voxel_z'], axis=1)
            df['mni_xyz'] = df.apply(lambda row: (row['mni_x'], row['mni_y'], row['mni_z']), axis=1)
            df = df.drop(labels=['mni_x', 'mni_y', 'mni_z'], axis=1)

            # Cache this to disk as a named dataframe
            self._logger.debug("  disk-caching samples to {f}".format(f=self.cache_path(donor + '-samples')))
            self.to_cache(donor + '-samples', df)

            dfs.append(df)

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
                self.to_cache('richiardi-samples', df[df.index.isin(richiardi_samples)])
            elif key == 'test':
                # TODO: Generate a test set of reduced data for profiling and testing algorithms.
                print("No test set is yet available, but it should be.")
            elif key == 'all':
                # This is the default and was already handled above.
                pass

    def build_probes(self, name=None):
        """ Read any one Probes.csv file and save it into a 'probes' dataframe.
        """
        donor = donor_map['any']
        filename = os.path.join(self.path_to(donor, 'probes'))
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
                self.to_cache('richiardi-probes', df[df.index.isin(richiardi_probes)])
            elif key == 'test':
                # TODO: Generate a test set of reduced data for profiling and testing algorithms.
                print("No test set is yet available, but it should be.")
            elif key == 'all':
                # This is the default, and already dealt with above
                pass

    def build_expression(self, name=None):
        """ Read all MicroarrayExpression.csv files and concatenate them into one 'expression' dataframe.

        This should only be called internally because the expression getter will dynamically detect whether
        it should use an in-memory cache, and on-disk cache, or have to call this function.

        # TODO: For a given name, only load what's necessary. We currently load everything.
        """
        self._logger.debug("  building expression data")
        dfs = []
        for donor in donors:
            # For each donor, load the well_ids (indices to expression data) and expression data
            filename = self.path_to(donor, 'annot')
            self._logger.debug("    loading {d}'s well_ids from {f}".format(d=donor, f=filename))
            df_ann = pd.read_csv(filename)

            # Load the annotation, using prev
            filename = self.path_to(donor, 'expr')
            self._logger.debug("    loading expression data from {f}".format(f=filename))
            df_exp = pd.read_csv(filename, header=None, index_col=0, names=df_ann['well_id'])

            # Cache this to disk as a named dataframe
            self._logger.debug("    disk-caching expression to {f}".format(f=self.cache_path(donor + '-expression')))
            self.to_cache(donor + '-expression', df_exp)

            # Then add this donor's data to a list for concatenation later.
            dfs.append(df_exp)

        # TODO: Build expression for canned items, if necessary

        self._logger.debug("  caching expression to {f}".format(f=self.cache_path('all-expression')))
        self.to_cache('all-expression', pd.concat(dfs, axis=1))

    def path_to(self, donor, file_key):
        """ provide a full file path based on any donor and file shorthand we can map.
        """
        # TODO: Think of a better linkage for directory structure and path mapping, reducing code dependencies
        return os.path.join(self._dir, 'BIDS', 'sub-' + donor_map[donor], BIDS_subdir, file_map[file_key])

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

        # Names can be abbreviated, causing cache misses if not fixed.
        try:
            a = canned_map[name.split(sep="-")[0]]
        except KeyError:
            a = name.split(sep="-")[0]
        try:
            b = type_map[name.split(sep="-")[1][0]]
        except KeyError:
            b = name.split(sep="-")[1]
        clean_name = "-".join([a, b])

        # If the call FORCES a rebuild, do it first.
        if refresh == 'always':
            self.build_probes(clean_name)
            self.build_samples(clean_name)
            self.build_expression(clean_name)

        # If a full two-part name is provided, and found, get it over with.
        if clean_name in self._cache.index:
            self._logger.debug("  found {} in memory".format(clean_name))
            return self._cache.loc[clean_name, 'dataframe']
        else:
            self._logger.debug("  {} not found in memory".format(clean_name))

        # If it's not in memory, check the disk.
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
                    self.build_expression(clean_name)
                else:
                    self.build_probes(clean_name)
                    self.build_samples(clean_name)
                    self.build_expression(clean_name)

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
        return os.path.join(self._dir, 'cache', name + '.df')

    def distance_matrix(self, samples):
        """ return a distance matrix between all samples in samples.
        :param samples: list or series of well_ids to be included in the distance matrix

        """
        df = pd.DataFrame(self.samples(samples=samples)['mni_xyz'].apply(pd.Series))
        return distance_matrix(df, df)

    def distance_vector(self, samples):
        """ return a distance vector (lower triangle of matrix) between all samples in samples.
        :param samples: list or series of well_ids to be included in the distance vector

        """
        m = self.distance_matrix(samples)
        return m[np.tril_indices_from(self.distance_matrix(samples), k=-1)]

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
