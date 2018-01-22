import os
import errno

import pandas
import requests
import humanize
import zipfile

# Get strings & dictionaries & dataframes from the local (not project) config
from .config import donor_map, aba_info, BIDS_subdir

class ExpressionData(object):
    """ Wrap Allen Brain Institute expression data
    """

    samples = {}
    probes = {}

    def __init__(self, data_dir):
        # Having a valid data_dir is critical. Make sure it will work.
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(errno.ENOENT, os.strerror(errno.ENOENT), data_dir)
        free_space = os.statvfs(data_dir).f_bsize * os.statvfs(data_dir).f_bavail
        req_space = 5 * (1024 ** 3)  # 4.2GB raw, plus whatever we store. 5GB is the low end.
        if free_space < req_space:
            raise ResourceWarning(
                "Expression data require at least {req}. {d} has {a}.".format(
                    req=humanize.naturalsize(req_space),
                    d=data_dir,
                    a=humanize.naturalsize(free_space),
                ))
        self.data_dir = data_dir

    def is_available(self, string, loc='file'):
        return False
    
    def download(self, donor):
        """ Download zip file from Allen Brain Institute and return the path to the file.

        :param donor: Any donor string that maps to a BIDS-compatible name
        :returns: A string representation of the full path of the downloaded zip file
        # TODO: Making this asynchrnous would be nice.
        """
        donor_name = donor_map[donor]
        url = aba_info.loc[donor_name, 'url']
        zipfile = aba_info.loc[donor_name, 'zipfile']
        r = requests.get(url=url, stream=True)
        with open(os.path.join(self.data_dir, zipfile), 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024**2):
                if chunk:
                    f.write(chunk)
        return zipfile

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
        return extract_to

    def space_available(self):
        return os.statvfs(self.data_dir).f_bsize * os.statvfs(self.data_dir).f_bavail

