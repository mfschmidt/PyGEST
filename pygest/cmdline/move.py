import os
from os.path import isfile

import shutil
from pygest.convenience import result_path_from_dict, result_description, json_lookup
from pygest.algorithms import file_is_equivalent
from pygest.cmdline.command import Command


extensions = ['json', 'tsv', 'log']


class Move(Command):
    """ A command to move Mantel correlation results from an old location/format to a new location """

    def __init__(self, args, logger=None):
        if '--log' not in args:
            args += ['--log', 'null']
        super().__init__(args, command="move", description="PyGEST push command", logger=logger)

    def _add_arguments(self):
        """ Add command-specific arguments, then Command's generic arguments. """
        self._parser.add_argument("source", default='NONE', type=str,
                                  help="Which file would you like to move?")
        self._parser.add_argument("--data", dest="data", nargs='?', type=str, default='NONE',
                                  help="Where are the BIDS and cache directories?")
        self._parser.add_argument("--dryrun", dest="dryrun", action='store_true', default=False,
                                  help="Report what would happen without actually doing it.")
        super()._add_arguments()

    def run(self):
        """ Move some source of results into another collection. """

        errors = []

        self._logger.debug("  old path: {}".format(self._args.source))
        if not os.path.isfile(self._args.source):
            errors.append("{} does not exist.".format(self._args.source))

        # 1. Do all old-style files exist?
        old_base = self._args.source[:self._args.source.rfind(".")]
        if self.new_set_status(old_base) in ["partial", ]:
            errors.append("{} exists, but does not have a complete set of files.".format(self._args.source))

        # 2. Determine new path
        new_base, valid = self.new_path()
        if valid:
            self._logger.debug("  new path: {}".format(new_base))
            if self.new_set_status(new_base) in ["complete", "partial", ]:
                v_new = json_lookup('pygest version', new_base + ".json")
                v_old = json_lookup('pygest version', old_base + ".json")
                if file_is_equivalent(old_base + ".tsv", new_base + ".tsv", verbose=True):
                    # We have two sets of results that give the same answer. Delete one.
                    if v_old > v_new:
                        # The candidates are newer than the destination (and match); overwrite them.
                        self._logger.info("DUPE tsv files agree; overwriting version {} with version {}".format(
                            v_new, v_old
                        ))
                        return self.move_set(old_base, new_base)
                    else:
                        # The newest version is already at the destination; delete the move candidates
                        for ext in extensions:
                            if self._args.dryrun:
                                self._logger.info("WOULD REMOVE DUPE: {}".format(old_base + "." + ext))
                            else:
                                try:
                                    os.remove(old_base + "." + ext)
                                except FileNotFoundError:
                                    self._logger.info("   NOT FOUND: {}".format(old_base + "." + ext))
                                self._logger.info("REMOVED DUPE: {}".format(old_base + "." + ext))
                        if not self._args.dryrun:
                            self.clean_up(old_base)
                        return 0
                else:
                    self._logger.info("CONFLICT: '{}' vs '{}'; new version {}, replacement candidate {}".format(
                        new_base, old_base, v_new, v_old
                    ))
                    return 0
        else:
            errors.append("INVALID: " + new_base)

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
        for ext in extensions:
            header = "WOULD MOVE" if self._args.dryrun else "MOVING"
            self._logger.info("{:10}: {}".format(header, old_base + "." + ext))
            self._logger.info("{:10}: {}".format("TO", new_base + "." + ext))
            if not self._args.dryrun:
                if os.path.exists(new_base + "." + ext):
                    os.remove(new_base + "." + ext)
                shutil.move(old_base + "." + ext, new_base + "." + ext)
        if not self._args.dryrun:
            self.fix_json(new_base + ".json")
            self.clean_up(old_base)
        return 0

    @staticmethod
    def new_set_status(base):
        if isfile(base + ".json") and isfile(base + ".tsv") and isfile(base + ".log"):
            return "complete"
        elif isfile(base + ".json") or isfile(base + ".tsv") or isfile(base + ".log"):
            return "partial"
        else:
            return "empty"

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

    def clean_up(self, path_string):
        """ Remove empty directories if left behind at old path.

        :param str path_string: path to a file or directory that can be removed if it's empty, and its parent, etc.
        """
        still_going = True
        top_removed = ""
        dirs_removed = []
        reported_dirs_removed = []
        while "/" in path_string and still_going:
            if os.path.isdir(path_string) and not os.listdir(path_string):
                os.rmdir(path_string)
                top_removed = path_string
                dirs_removed.append(path_string)
            else:
                still_going = False
            path_string = path_string[:path_string.rfind("/")]

        for d in dirs_removed:
            if d != top_removed:
                reported_dirs_removed.append("..." + d[len(top_removed):])

        if top_removed != "":
            self._logger.info("Removed {}, {}".format(top_removed, ", ".join(reported_dirs_removed)))

    def new_path(self):
        """ Convert an old-version PyGEST output file to a current path/file.

        :return: the new path
        """

        d, e = result_description(self._args.source)

        d['data'] = self._args.data
        d['cmd'] = self._command

        # Use the PyGEST path-maker so any changes to PyGEST are reflected here.
        if len(e) > 0:
            return "{} from {}".format(", ".format(e), self._args.source), False
        return result_path_from_dict(d), True
