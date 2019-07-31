import os
from os.path import isfile
import filecmp

import shutil
from pygest.convenience import result_path_from_dict, result_description, json_lookup

from pygest.cmdline.command import Command


class Move(Command):
    """ A command to move Mantel correlation results from an old location/format to a new location """

    def __init__(self, args, logger=None):
        if '--log' not in args:
            args += ['--log', 'null']
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
                dupes = 0
                for ext in [".json", ".tsv", ".log"]:
                    if filecmp.cmp(old_base + ext, new_base + ext, shallow=False):
                        dupes += 1
                if dupes > 0:
                    errors.append("DUPE: {} duplicate files already exist at {}".format(dupes, new_base))
                else:
                    v_new = json_lookup('pygest version', new_base + ".json")
                    v_old = json_lookup('pygest version', old_base + ".json")
                    errors.append("DUPE: Different files exist at {}; existing version {}, replacement {}".format(
                        new_base, v_new, v_old
                    ))
        else:
            errors.append(new_base)

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
