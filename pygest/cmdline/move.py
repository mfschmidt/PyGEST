import os
from os.path import isfile

import shutil
from pygest.convenience import bids_val, path_to, dict_from_bids

from pygest.cmdline.command import Command


class Move(Command):
    """ A command to move Mantel correlation results from an old location/format to a new location """

    def __init__(self, args, logger=None):
        super().__init__(args, command="move", description="PyGEST push command", logger=logger)

    def _add_arguments(self):
        """ Add command-specific arguments, then Command's generic arguments. """
        self._parser.add_argument("source", default='NONE',
                                  help="Which file would you like to move?")
        self._parser.add_argument("--data", dest="data", nargs='?', type=str, default='NONE',
                                  help="Where are the BIDS and cache directories?")
        self._parser.add_argument("--dryrun", dest="dryrun", action='store_true', default=False,
                                  help="Report what would happen without actually doing it.")
        super()._add_arguments()

    def run(self):
        """ Move some source of results into another collection. """

        errors = []

        if not os.path.isfile(self._args.source):
            errors.append("{} does not exist.".format(self._args.source))

        # 1. Determine old and new paths
        old_base = self._args.source[:self._args.source.rfind(".")]
        new_base = self.new_path(self._args.source, self._args.data, ext="NONE")
        self._logger.debug("  old path: {}".format(self._args.source))
        self._logger.debug("  new path: {}".format(new_base))

        # 2. Do all old-style files exist?
        if self.new_set_status(old_base) in ["partial", ]:
            errors.append("{} exists, but does not have a complete set of files.".format(self._args.source))

        # 2. Does it already exist in PYGEST_DATA?
        if self.new_set_status(new_base) in ["complete", "partial", ]:
            errors.append("DUPE: Some duplicate files already exist at {}".format(new_base))

        # 4. A complete old set exists, and would not overwrite anything if moved.
        if len(errors) > 0:
            for e in errors:
                self._logger.error(e)
            return 1

        return self.move_set(old_base, new_base)

    def move_set(self, old_base, new_base):
        """ Move three files from old_base to new_base. """

        if not self._args.dryrun:
            os.makedirs(new_base[:new_base.rfind("/")], exist_ok=True)
        for ext in ['.json', '.tsv', '.log']:
            header = "WOULD MOVE" if self._args.dryrun else "MOVING"
            self._logger.info("{:10}: {}".format(header, old_base + ext))
            self._logger.info("{:10}: {}".format("TO", new_base + ext))
            if not self._args.dryrun:
                shutil.move(old_base + ext, new_base + ext)
        if not self._args.dryrun:
            self.fix_json(new_base + ".json")
        return 0

    @staticmethod
    def new_set_status(base):
        if isfile(base + ".json") and isfile(base + ".tsv") and isfile(base + ".log"):
            return "complete"
        elif isfile(base + ".json") or isfile(base + ".tsv") or isfile(base + ".log"):
            return "partial"
        else:
            return "empty"

    @staticmethod
    def fix_sub(old_sub):
        """ Split a convoluted old-style split sub string to new format. """

        if "by" in old_sub['sub']:
            seed = old_sub['sub'][-5:]
            parts = old_sub['sub'][:-5].split("by")
            if parts[0][-5:] == "train":
                old_sub['sub'] = "split" + parts[0][:-5]
                old_sub['set'] = "train" + seed
            elif parts[0][-4:] == "test":
                old_sub['sub'] = "split" + parts[0][:-4]
                old_sub['set'] = "test" + seed
            else:
                old_sub['sub'] = "splitUNKNOWN"
                old_sub['set'] = "UNKNOWN" + seed
            old_sub['sby'] = parts[1]
        return old_sub

    def fixed_shuffle(self, old_shuffle):
        """ If I just appended a date to a shuffle subdir, clean it up. """

        if old_shuffle.lower().startswith("deriv"):
            return "derivatives"
        if old_shuffle.lower().startswith("shuffle"):
            return "shuffles"
        if old_shuffle.lower().startswith("dist"):
            return "distshuffles"
        if old_shuffle.lower().startswith("edge"):
            return "edgeshuffles"
        self._logger.error("What kind of shuffle is {}? I can't handle that.".format(old_shuffle))

    def fix_json(self, json_file):
        """ Remove errant commas from the end of json and re-save the file. """

        mtime = os.path.getmtime(json_file)
        with open(json_file, "r") as f:
            contents = f.read()
            final_quote = contents.rfind("\"")
            final_brace = contents.rfind("}")
            contents = "".join([
                contents[: final_quote + 1],
                contents[final_quote + 1:final_brace].replace(",", ""),
                contents[final_brace:]
            ])
        with open(json_file, "w") as f:
            f.write(contents)
        # Put the previous modified timestamp back on the file
        os.utime(json_file, (mtime, mtime))
        self._logger.debug("Re-wrote json file.")

    def new_path(self, old_path, new_base, ext="KEEP"):
        """ Convert an old-version PyGEST output file to a current path/file.

        :param old_path: The path to results that need moving
        :param new_base: The path to the base directory of the destination
        :param ext: ext='KEEP' to use same extension as source file, 'NONE' for extensionless, or literal ext str

        :return: the new path
        """

        parts = old_path.split("/")
        n = len(parts)
        if n < 5:
            self._logger.error("The path provided does not have enough directories for PyGEST.")
            return ""

        if True:
            old_bids_dict = dict_from_bids(old_path)
        else:
            old_bids_dict = {}
            for bids_key in ['sub', 'hem', 'splby', 'ctx', 'samp', 'parby', 'prb', 'prob',
                             'tgt', 'alg', 'algo', 'cmp', 'comp', 'nrm', 'norm', 'msk', 'mask', 'adj', 'seed', 'batch']:
                old_bids_dict[bids_key] = bids_val(bids_key, old_path)
                if not (bids_key in old_bids_dict.keys()):
                    old_bids_dict[bids_key] = 'null'
                if old_bids_dict[bids_key] == '':
                    old_bids_dict[bids_key] = 'null'

        old_bids_dict['top_subdir'] = parts[n - 4]
        if ext == "NONE":
            old_bids_dict['ext'] = ""
        elif ext == "KEEP":
            old_bids_dict['ext'] = parts[n - 1][parts[n - 1].rfind("."):]
        else:
            old_bids_dict['ext'] = "." + ext

        # Copy over some old-style abbreviations to new.
        def replace_key_if_empty(old_key, new_key):
            if new_key not in old_bids_dict or old_bids_dict[new_key] == 'null':
                old_bids_dict[new_key] = old_bids_dict[old_key]

        replace_key_if_empty('nrm', 'norm')
        replace_key_if_empty('ctx', 'samp')
        replace_key_if_empty('prb', 'prob')
        replace_key_if_empty('alg', 'algo')
        replace_key_if_empty('msk', 'mask')
        replace_key_if_empty('cmp', 'comp')

        if 'batch' not in old_bids_dict or old_bids_dict['batch'] == 'null':
            old_bids_dict['batch'] = 'whole'  # default, and most common for old runs
            if old_bids_dict['seed'] == 'null':
                old_bids_dict['batch'] = "whole"
            else:
                old_bids_dict['batch'] = "neither{:05}".format(int(old_bids_dict.get('seed', "00000")))

        old_bids_dict = self.fix_sub(old_bids_dict)

        old_bids_dict['data'] = new_base
        old_bids_dict['cmd'] = self._command

        # Use the PyGEST path-maker so any changes to PyGEST are reflected here.
        return path_to(self._command, old_bids_dict)
