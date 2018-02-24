import os
import errno

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

import requests
import humanize
import zipfile

# Get strings & dictionaries & DataFrames from the local (not project) config
from .config import donors, donor_map, file_map, aba_info, default_dir, BIDS_subdir, donor_files, data_sections, canned_map


# import utility  # local library containing a hash_file routine


class data(object):
    """ Wrap Allen Brain Institute expression data

    The general strategy is to keep track of what data we have, and in which state.
    We can then provide data to callers, deciding intelligently whether to send them
    cached data, restructure from an upstream source, or download from ABA and start
    from scratch.

    TODO: A current difficulty is in reporting to a caller how long something may take.
          If we're going to be running for an hour to download data, or running for a
          day to do repeated correlations on many gene sets, it's best to kick off the
          job and inform them when it's done. But there are many ways to do this, none
          of them are necessarily very pythonic.
    TODO: Implement a verbosity flag/level with logging
    TODO: Implement a default logger that can be overridden by the user.
    TODO: Load up Richiardi's probe-set mapped with appropriate indices as a named and cached probes file.
    """

    # Remember the base directory for all data, which may differ from the default in config.py
    _dir = default_dir

    # Create an empty dataframe to hold information about existing files.
    _files = pd.DataFrame(
        columns=['section', 'donor', 'path', 'file', 'bytes', 'hash', 'full_path', ],
        data=[]
    )

    # Store named subsets of probes, samples, and expression.
    # _cache.loc['all_samples'] will hold a 3702 sample x 9-field DataFrame from all SampleAnnot.csv files.
    # _cache.loc['all_probes'] will hold a 58,962 probe x 3702 sample DataFrame from all MicroarrayExpression files.
    _cache = pd.DataFrame(
        index=[],
        columns=['dataframe', 'full_path', ],
        data=[]
    )

    _probe_name_to_id = {}

    def __init__(self, data_dir):
        """ Initialize this object, raising an error if necessary.

        :param data_dir: the base directory housing all expression data and caches
        """
        self.refresh(data_dir)

        # We need about 5GB of disk just for raw data, plus however much assists with caching
        req_space = aba_info['expanded_bytes'].sum()
        if self.space_available() < req_space - self.space_used():
            raise ResourceWarning(
                "Expression data require at least {req}. {d} has {a}.".format(
                    req=humanize.naturalsize(req_space - self.space_used()),
                    d=self._dir,
                    a=humanize.naturalsize(self.space_available()),
                ))

    def expression(self, name=None, probes=None, samples=None):
        """ The expression property
        Filterable by probe rows or sample columns

        :param name: a label used to store and retrieve a specific subset of expression data
        :param probes: a list of probes used to filter expression data
        :param samples: a list of samples (wellids) used to filter expression data
        """

        # Without filters, we can only return a cached DataFrame.
        if probes is None and samples is None:
            if name is not None:
                return self.from_cache(name + '_expr')
            else:
                return self.from_cache('all_expr')

        # With filters, we will generate a filtered DataFrame.
        filtered_expr = self.from_cache('all_expr')
        if isinstance(probes, list) or isinstance(probes, pd.Series):
            filtered_expr = filtered_expr.loc[filtered_expr.index.isin(list(probes)), :]
        elif isinstance(probes, pd.DataFrame):
            filtered_expr = filtered_expr.loc[probes.index, :]
        if isinstance(samples, list) or isinstance(samples, pd.Series):
            filtered_expr = filtered_expr.loc[:, list(samples)]
        elif isinstance(samples, pd.DataFrame):
            filtered_expr = filtered_expr.loc[:, samples.index]

        # If we're given a name, cache the filtered DataFrame
        if name is not None:
            self.to_cache(name + '_expr', data=filtered_expr)

        return filtered_expr

    def samples(self, name=None, samples=None, donor=None, hemisphere=None):
        """ The samples property
        Asking for ExpressionData.samples will return a dataframe with all samples.
        To get a sub-frame, call samples(data=list_of_wanted_well_ids).

        :param name: a label used to store and retrieve a specific subset of sample data
        :param samples: a list of samples (wellids) used to filter sample data
        :param donor: specifying donor will constrain the returned samples appropriately.
        :param hemisphere: specifying left or right will constrain the returned samples appropriately.
        :return: A DataFrame indexed by well_id, with sample information
        """

        # Do some interpretation before the heavy lifting.
        h = 'a' if hemisphere is None else hemisphere[0].lower()
        print("Got hemisphere of '{}', which becomes '{}'".format(hemisphere, h))

        # If a name is specified, all by itself, our job is easy.
        if name is not None and samples is None and donor is None and hemisphere is None:
            if name in canned_map:
                return self.from_cache(canned_map[name] + '_samples')
            else:
                print("Could not find samples named {}.".format(name))
                return self.from_cache(name + '_samples')
                # TODO: What if the file doesn't exist? check from_cache code
                # return None

        # But if any filters are present, forget the name and build the DataFrame.
        filtered_samples = None
        if donor is not None:
            if donor in donor_map:
                filtered_samples = self.from_cache(donor_map[donor] + '_samples')
            else:
                print("Donor {} was not recognized, using the full sample set.".format(donor))

        # If we didn't get a donor, start with a full sample set.
        if filtered_samples is None:
            filtered_samples = self.from_cache('all_samples')

        # With samples filters, we'll filter the dataframe
        if isinstance(samples, list) or isinstance(samples, pd.Series):
            filtered_samples = filtered_samples.loc[list(samples), :]
        elif isinstance(samples, pd.DataFrame):
            filtered_samples = filtered_samples.loc[samples.index, :]

        # By hemisphere, we will restrict to left or right
        # MNI space defines right of mid-line as +x and left of midline as -x
        if h == 'l':
            l_filter = pd.DataFrame(filtered_samples['mni_xyz'].tolist(),
                                    index=filtered_samples.index,
                                    columns=['x', 'y', 'z']).x < 0
            filtered_samples = filtered_samples[l_filter]
        elif h == 'r':
            r_filter = pd.DataFrame(filtered_samples['mni_xyz'].tolist(),
                                    index=filtered_samples.index,
                                    columns=['x', 'y', 'z']).x > 0
            filtered_samples = filtered_samples[r_filter]
        elif h == '0':
            m_filter = pd.DataFrame(filtered_samples['mni_xyz'].tolist(),
                                    index=filtered_samples.index,
                                    columns=['x', 'y', 'z']).x == 0
            filtered_samples = filtered_samples[m_filter]
        elif h == 'a':
            pass
        else:
            print("{} is not interpretable as a hemisphere; ignoring it.".format(hemisphere))

        # If we're given a name, and didn't already pull it from cache, cache the filtered DataFrame
        if name is not None:
            # if h in ['l', 'r']:
            #     self.to_cache(name + '_' + h + '_samples', data=filtered_samples)
            self.to_cache(name + '_samples', data=filtered_samples)

        return filtered_samples

    def probes(self, name=None, probes=None):
        """ The probes property
        Asking for ExpressionData.probes will return a dataframe with all probes.
        To get a sub-frame, call samples(data=list_of_wanted_probe_ids).

        :param name: a label used to store and retrieve a specific subset of probe data
        :param probes: a list of probes used to filter probe data
        """

        # Without filters, we can only return a cached DataFrame.
        if probes is None:
            if name is not None:
                if name in donor_map:
                    return self.from_cache(donor_map[name] + '_probes')
                elif name in canned_map:
                    return self.from_cache(canned_map[name] + '_probes')
                else:
                    return self.from_cache(name + '_probes')
            else:
                return self.from_cache('all_probes')

        # With filters, we will generate a filtered DataFrame.
        filtered_probes = self.from_cache('all_probes')
        if isinstance(probes, list) or isinstance(probes, pd.Series):
            filtered_probes = filtered_probes.loc[list(probes), :]
        elif isinstance(probes, pd.DataFrame):
            filtered_probes = filtered_probes.loc[probes.index, :]

        # If we're given a name, cache the filtered DataFrame
        if name is not None:
            self.to_cache(name + '_samples', data=filtered_probes)

        return filtered_probes

    def refresh(self, data_dir=None, clean=False):
        """ Get the lay of the land and remember what we find.
        :param data_dir: set the directory used for data storage and caching
        :param clean: if set to True, delete all cached values and start over.
        """

        # Make sure we actually have a place to work.
        if data_dir is not None and os.path.isdir(data_dir):
            self._dir = data_dir
        else:
            raise NotADirectoryError(errno.ENOENT, os.strerror(errno.ENOENT), data_dir)

        # Wipe the cache if asked to start_from_scratch
        if clean:
            # TODO: wipe the cache and clear the file list
            pass

        # Find/make each section(subdirectory).
        for section in data_sections:
            if os.path.isdir(os.path.join(self._dir, section)):
                self.find_files_in(section)
            else:
                os.mkdir(os.path.join(self._dir, section))
                if os.path.isdir(os.path.join(self._dir, section)):
                    print("{} could neither be found nor created at {}".format(section, self._dir))

        return self.status()

    def find_files_in(self, section):
        print("Refreshing {d}".format(d=os.path.join(self._dir, section)))
        if section == 'BIDS':
            for donor in donors:
                for donor_file in donor_files:
                    path = os.path.join(self._dir, section, 'sub-' + donor, BIDS_subdir, donor_file)
                    if os.path.isfile(path):
                        print("  found {f}, hashing it...".format(f=path))
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
                    # TODO: Persist this stuff, and check by altered date. Don't waste time hashing every init.
                    # TODO: Is this wasted time? For now, we'll never check it and don't care about the hash.
                    print("  found {f}, not hashing it...".format(f=path))
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
            for ind, row in aba_info.iterrows():
                path = os.path.join(self._dir, section, row['zip_file'])
                if os.path.isfile(path):
                    # TODO: Persist this stuff, and check by altered date. Don't waste time hashing every init.
                    print("  found {f}, hashing it...".format(f=path))
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
        else:
            print("  ignoring {} for now".format(section))

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
            print("Not aware of donor named {}. Nothing I can download.".format(donor))
            return None
        url = aba_info.loc[donor_map[donor], 'url']
        zip_file = aba_info.loc[donor_map[donor], 'zip_file']
        # Allow for override of url if we are just testing or want to keep it local
        if base_url is not None:
            url = base_url + '/' + zip_file
        print("Downloading {} from {} to {} ...".format(
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
        extract_to = os.path.join(self._dir, aba_info.loc[donor_name, 'subdir'], BIDS_subdir)
        # We could try/except this, but we'd just have to raise another error to whoever calls us.
        os.makedirs(extract_to)
        zip_file = os.path.join(self._dir, 'downloads', aba_info.loc[donor_name, 'zipfile'])
        print("Extracting {} to {} ...".format(
            zip_file,
            extract_to
        ))
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(extract_to)
        if clean_up:
            print("  cleaning up by removing {}".format(zip_file))
            os.remove(zip_file)
        if add_to_tsv:
            print("  adding {} to participants.tsv".format(donor_name))
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
            print("  loading {d}'s samples from {f} and parsing coordinates".format(d=donor, f=filename))
            df = pd.read_csv(filename)
            df = df.set_index('well_id')

            # Remember which donor these data came from
            df['donor'] = pd.DataFrame(index=df.index, data={'donor': donor})['donor']

            # Convert text representations of coordinates into numeric tuples
            df['vox_xyz'] = df.apply(lambda row: (row['mri_voxel_x'], row['mri_voxel_y'], row['mri_voxel_z']), axis=1)
            df = df.drop(labels=['mri_voxel_x', 'mri_voxel_y', 'mri_voxel_z'], axis=1)
            df['mni_xyz'] = df.apply(lambda row: (row['mni_x'], row['mni_y'], row['mni_z']), axis=1)
            df = df.drop(labels=['mni_x', 'mni_y', 'mni_z'], axis=1)

            # Cache this to disk as a named dataframe
            print("  disk-caching samples to {f}".format(f=self.cache_path(donor + '_samples')))
            self.to_cache(donor + '_samples', df)

            dfs.append(df)

        # TODO: Build samples for canned items, if necessary
        # TODO: Build alternate ultimate sample data source with batch_ids linked to samples.

        print("  caching samples to {f}".format(f=self.cache_path('all_samples')))
        self.to_cache('all_samples', pd.concat(dfs, axis=0))

    def build_probes(self, name=None):
        """ Read any one Probes.csv file and save it into a 'probes' dataframe.
        # TODO: For a given name, only load what's necessary. We currently load everything.
        """
        donor = donor_map['any']
        filename = os.path.join(self.path_to(donor, 'probes'))
        print("  loading probes from {}".format(filename))
        df = pd.read_csv(filename)
        df.set_index('probe_id')

        # TODO: Build probes for canned items, if necessary
        # TODO: Pre-build list of Richiardi probes in ABI probe_id format

        print("  caching probes to {f}".format(f=self.cache_path('all_probes')))
        self.to_cache('all_probes', df)

    def build_expression(self, name=None):
        """ Read all MicroarrayExpression.csv files and concatenate them into one 'expression' dataframe.

        This should only be called internally because the expression getter will dynamically detect whether
        it should use an in-memory cache, and on-disk cache, or have to call this function.

        # TODO: For a given name, only load what's necessary. We currently load everything.
        """
        dfs = []
        for donor in donors:
            # For each donor, load the well_ids (indices to expression data) and expression data
            filename = self.path_to(donor, 'ann')
            print("  loading {d}'s well_ids from {f}".format(d=donor, f=filename))
            df_ann = pd.read_csv(filename)

            # Load the annotation, using prev
            filename = self.path_to(donor, 'exp')
            print("    and expression data from {f}".format(f=filename))
            df_exp = pd.read_csv(filename, header=None, index_col=0, names=df_ann['well_id'])

            # Cache this to disk as a named dataframe
            print("  disk-caching expression to {f}".format(f=self.cache_path(donor + '_expr')))
            self.to_cache(donor + '_expr', df_exp)

            # Then add this donor's data to a list for concatenation later.
            dfs.append(df_exp)

        # TODO: Build expression for canned items, if necessary

        print("  caching expression to {f}".format(f=self.cache_path('all_expr')))
        self.to_cache('all_expr', pd.concat(dfs, axis=1))

    def path_to(self, donor, file_key):
        """ provide a full file path based on any donor and file shorthand we can map.
        """
        # TODO: Think of a better linkage for directory structure and path mapping, reducing code dependencies
        return os.path.join(self._dir, 'BIDS', 'sub-' + donor_map[donor], BIDS_subdir, file_map[file_key])

    def to_cache(self, name, data):
        """ Save data to disk and hold a reference in memory.
        """
        # On-disk cache
        if isinstance(data, pd.DataFrame):
            data.to_pickle(self.cache_path(name))
        else:
            print("Cannot cache {}. Only Pandas DataFrame objects are cachable for now.".format(
                type(data)
            ))
            # TODO: This only supports pd.DataFrame objects for now, build out other types?
            # with open(self.cache_path(name), 'wb') as f:
            #     pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # In-memory cache
        self._cache.loc[name] = {
            'dataframe': data,
            'full_path': self.cache_path(name),
        }
        print("  added {} to cache, now {} records".format(name, len(self._cache.index)))

    def from_cache(self, name):
        """ Return cached data, or None if I can't find it.
        """
        if name in self._cache.index:
            print("  found {} in memory".format(name))
            return self._cache.loc[name, 'dataframe']
        if os.path.isfile(self.cache_path(name)):
            print("  found {} cached on disk, loading...".format(name))
            self._cache.loc[name] = {
                'dataframe': pd.read_pickle(self.cache_path(name)),
                'full_path': self.cache_path(name)
            }
            return self._cache.loc[name, 'dataframe']

        # No cached data were found. If asked for keywords, we need to generate.
        print("  Building probes from raw data...")
        self.build_probes(name)
        print("  Building samples from raw data...")
        self.build_samples(name)
        print("  Building expression from raw data...")
        self.build_expression(name)

        # And now that the cache has been filled, try the cache again
        if name in self._cache.index:
            return self._cache.loc[name, 'dataframe']

        # If still no match, not much we can do.
        return None

    def cache_path(self, name):
        """ return a path to the file for caching under this name.
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
        if from_term in self.probes('all').columns and to_term in self.probes('all').columns:
            df_tmp = self.probes('all')
            df_tmp.index = df_tmp[from_term]
            return dict(df_tmp[to_term])
        elif from_term in self.samples('all').columns and to_term in self.probes('all').columns:
            df_tmp = self.samples('all')
            df_tmp.index = df_tmp[from_term]
            return dict(df_tmp[to_term])
        # Hopefully, we've done our job. If not, something went wrong.
        if from_term not in self.probes('all') and from_term not in self.samples('all'):
            raise KeyError("The term, \"{}\" was not found in probes or samples.".format(from_term))
        if to_term not in self.probes('all') and to_term not in self.samples('all'):
            raise KeyError("The term, \"{}\" was not found in probes or samples.".format(to_term))
        return {}
