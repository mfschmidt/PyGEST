import os
import errno

import pandas as pd
import requests
import humanize
import zipfile

# Get strings & dictionaries & dataframes from the local (not project) config
from .config import donors, donor_map, aba_info, BIDS_subdir, donor_files, rel_path_to
import utility


class ExpressionData(object):
    """ Wrap Allen Brain Institute expression data
    """

    # Track the files we have available on disk
    existing_files = None
    # samples will hold a 3702 observation x 9-column  dataframe from SampleAnnot.csv files.
    samples = None
    # probes will hold a 58,961 row by 3,702 column dataframe of gene expression values.
    # This is a big memory hog, 58,692 Int64s and 218,277,324 float64s = 1.7GB
    probes = None

    def __init__(self, data_dir):
        """ Initialize this object, raising an error if necessary.

        :param data_dir:
        """
        # Having a valid data_dir is critical. Make sure it will work.
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(errno.ENOENT, os.strerror(errno.ENOENT), data_dir)
        self.refresh(data_dir)
        req_space = aba_info['expanded_bytes'].sum()  # 4.2GB raw, plus whatever we store. 5GB is the low end.
        if self.space_available() < req_space - self.space_used():
            raise ResourceWarning(
                "Expression data require at least {req}. {d} has {a}.".format(
                    req=humanize.naturalsize(req_space - self.space_used()),
                    d=data_dir,
                    a=humanize.naturalsize(self.space_available()),
                ))
        self.data_dir = data_dir

    def refresh(self, data_dir=None):
        """ Get the lay of the land and remember what we find.
        """
        self.existing_files = pd.DataFrame(
            columns=['path', 'donor', 'file', 'bytes', 'hash'],
            data=[]
        )
        file_list = []
        if data_dir is not None:
            self.data_dir = data_dir
        print("Refreshing {d}".format(d=self.data_dir))
        for ind, row in aba_info.iterrows():
            for f in donor_files:
                path = os.path.join(self.data_dir, rel_path_to(ind, f))
                if os.path.isfile(path):
                    print("  found {f}, hashing it...".format(f=path))
                    file_list.append({
                        'path': os.path.dirname(path),
                        'donor': donor_map[ind],
                        'file': os.path.basename(path),
                        'bytes': os.stat(path).st_size,
                        'hash': utility.hash_file(path)
                    })
        self.existing_files = pd.DataFrame(file_list)
        return self.status()

    def status(self):
        files_string = "{n} files from {d} donors consume {b}.".format(
            n=self.existing_files['bytes'].count(),
            d=len(self.existing_files['donor'].unique()),
            b=humanize.naturalsize(self.existing_files['bytes'].sum())
        )
        samples_string = "Samples have{s} been imported.".format(
            s=" not" if self.samples is None else ""
        )
        probes_string = "Probes have{s} been imported.".format(
            s=" not" if self.probes is None else ""
        )
        return "\n".join([files_string, samples_string, probes_string])

    def download(self, donor):
        """ Download zip file from Allen Brain Institute and return the path to the file.

        :param donor: Any donor string that maps to a BIDS-compatible name
        :returns: A string representation of the full path of the downloaded zip file
        # TODO: Making this asynchrnous would be nice.
        """
        donor_name = donor_map[donor]
        url = aba_info.loc[donor_name, 'url']
        zip_file = aba_info.loc[donor_name, 'zip_file']
        r = requests.get(url=url, stream=True)
        with open(os.path.join(self.data_dir, zip_file), 'wb') as f:
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
        extract_to = os.path.join(self.data_dir, aba_info.loc[donor_name, 'subdir'], BIDS_subdir)
        # We could try/except this, but we'd just have to raise another error to whoever calls us.
        os.makedirs(extract_to)
        zip_file = os.path.join(self.data_dir, aba_info.loc[donor_name, 'zipfile'])
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(extract_to)
        if clean_up:
            os.remove(zip_file)
        if add_to_tsv:
            tsv_file = os.path.join(self.data_dir, 'participants.tsv')
            with open(tsv_file, 'a') as f:
                f.write(donor_name + "\n")
        self.refresh()
        return extract_to

    def space_available(self):
        return os.statvfs(self.data_dir).f_bsize * os.statvfs(self.data_dir).f_bavail

    def space_used(self):
        return self.existing_files['bytes'].sum()

    def import_samples(self):
        """ Read all SampleAnnot.csv files and concatenate them into one 'samples' dataframe.
        """
        dfs = []
        for donor in donors:
            filename = os.path.join(self.data_dir, rel_path_to(donor, 'annot'))
            print("  loading {f}".format(f=filename))
            df = pd.read_csv(filename)
            df = df.set_index('well_id')
            df['donor'] = donor_map[donor]
            df['vox_xyz'] = df.apply(lambda row: (row['mri_voxel_x'], row['mri_voxel_y'], row['mri_voxel_z']), axis=1)
            df = df.drop(columns=['mri_voxel_x', 'mri_voxel_y', 'mri_voxel_z'])
            df['mni_xyz'] = df.apply(lambda row: (row['mni_x'], row['mni_y'], row['mni_z']), axis=1)
            df = df.drop(columns=['mni_x', 'mni_y', 'mni_z'])
            dfs.append(df)
        # Concatenate all SampleAnnot data, maintaining columns and adding new rows
        self.samples = pd.concat(dfs, axis=0)
        print("  pickling samples to {f}".format(f='samples.pkl'))
        self.samples.to_pickle(os.path.join(self.data_dir, 'samples.pkl'))
        return self.samples

    def import_probes(self):
        """ Read all MicroarrayExpression.csv files and concatenate them into one 'probes' dataframe.
        """
        dfs = []
        for donor in donors:
            filename = os.path.join(self.data_dir, rel_path_to(donor, 'annot'))
            print("  loading {f}".format(f=filename))
            df_ann = pd.read_csv(filename)
            filename = os.path.join(self.data_dir, rel_path_to(donor, 'exp'))
            print("  loading {f}".format(f=filename))
            df_exp = pd.read_csv(filename, header=None, index_col=0, names=df_ann['well_id'])
            dfs.append(df_exp)
        # Concatenate all expression data, maintaining row indices and adding new columns.
        self.probes = pd.concat(dfs, axis=1)
        print("  pickling probes to {f}".format(f='expression.pkl'))
        self.probes.to_pickle(os.path.join(self.data_dir, 'expression.pkl'))
        return self.probes
