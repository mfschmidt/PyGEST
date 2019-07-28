import os
import sys
import argparse
import logging
import pkg_resources
import pygest as ge

from datetime import datetime
from pygest.convenience import path_to


class Command(object):
    """ Each command has overlapping functionality. Each command can inherit the shared functionality from here.
    """

    def __init__(self, arguments, command="", description="A PyGEST command", logger=None):
        """ Initialize the command basics, but do not parse args or run. """
        self._arguments = arguments
        self._command = command
        self._description = description
        self._parser = argparse.ArgumentParser(description=self._description)
        self._add_arguments()
        self._args = self._parser.parse_args(self._arguments)
        self._args.beginning = datetime.now()
        self._post_process_arguments()
        self._setup_data()  # _setup_data must come before logging, as it ascertains the base path for everything.
        self._setup_logging(logger)  # _setup_logging must come before later ge.Data() so it can pass the logger.
        if "data" in self._args and self._args.data is not None:
            self.data = ge.Data(self._args.data, self._logger)
        if self._args.verbose:
            self._report_context()

    def _add_arguments(self):
        """ Add on some standard arguments used by all commands. """
        self._parser.add_argument("-v", "--verbose", action="store_true",
                                  help="increase output verbosity")
        self._parser.add_argument("--log", dest="log", default='',
                                  help="Provide a path to a log file to save output.")

    def _setup_data(self):
        """ If a data path is provided, use it to initiate pygest.data. """
        if "data" not in self._args or self._args.data is None or self._args.data == "NONE":
            if "PYGEST_DATA" in os.environ:
                self._args.data = os.environ['PYGEST_DATA']
            else:
                print("I don't know where to find data. Try one of the following, with your own path:")
                print("")
                print("    $ pygest {} --data /home/mike/ge_data".format(" ".join(sys.argv[1:])))
                print("")
                print("or, better, set it in your environment (use ~/.bashrc as a permanent solution)")
                print("")
                print("    $ export PYGEST_DATA=/home/mike/ge_data")
                print("    $ pygest {}".format(" ".join(sys.argv[1:])))
                print("")
                sys.exit(1)
        if not os.path.isdir(self._args.data):
            print("Data directory '{}' does not exist.".format(self._args.data))
            sys.exit(1)

    def _setup_logging(self, logger=None):
        """ Create a logger and handlers. """

        if logger is None:
            # Set up logging, formatted with datetimes.
            log_formatter = logging.Formatter(fmt='%(asctime)s | %(message)s', datefmt='%Y-%m-%d_%H:%M:%S')
            self._logger = logging.getLogger('pygest')
            self._logger.setLevel(1)

            # Set up the console (stdout) log handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(log_formatter)
            console_handler.setLevel(logging.DEBUG if self._args.verbose else logging.INFO)
            self._logger.addHandler(console_handler)

            # Set up the file handler, if requested
            if "log" in self._args and self._args.log not in ['', 'null']:
                file_handler = logging.FileHandler(self._args.log, mode='a+')
                file_handler.setFormatter(log_formatter)
                # file_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
                # As a design decision, heavy logging to a file is almost always desirable, without clogging stdout
                file_handler.setLevel(logging.DEBUG)
                self._logger.addHandler(file_handler)
                self._logger.debug("logger added filehandler at {}, from cmdline argument.".format(self._args.log))
            elif "log" in self._args and self._args.log == '':
                # By default, we log everything. This can only be turned off by setting --log to null
                file_handler = logging.FileHandler(path_to(self._command, self._args, log_file=True), mode='a+')
                file_handler.setFormatter(log_formatter)
                file_handler.setLevel(logging.DEBUG)
                self._logger.addHandler(file_handler)
                self._logger.debug("logger added default filehandler at '{}'.".format(file_handler.baseFilename))
            else:
                pass  # no file handlers get set up
        else:
            self._logger = logger

    def _report_context(self, indent=""):
        """ Report our interpretation of command-line arguments. """

        def log_indented(s):
            """ output to the logger with the provided indentation """
            self._logger.info("{}{}".format(indent, s))

        log_indented("--------------------------------------------------------------------------------")
        log_indented(" Command: {}".format(" ".join(sys.argv[:])))
        log_indented(" OPENBLAS_NUM_THREADS = {}".format(os.environ['OPENBLAS_NUM_THREADS']))
        try:
            log_indented(" PyGEST is running version {}".format(pkg_resources.require("pygest")[0].version))
        except pkg_resources.DistributionNotFound:
            log_indented(" PyGEST is running uninstalled; no version information is available.")
        log_indented("   interpretation of arguments:")
        for k in self._args.__dict__:
            if self._args.__dict__[k] is not None:
                log_indented("     - {} = {}".format(k, self._args.__dict__[k]))
        path_type = 'split' if self._command == 'split' else 'result'
        log_indented("   {} path:".format(path_type))
        log_indented("   '{}'".format(path_to(self._command, self._args, path_type=path_type, log_file=True)))
        log_indented("--------------------------------------------------------------------------------")

    def _post_process_arguments(self):
        """ Provide the opportunity to interpret and modify command-line arguments before reporting them. """
        pass

    def run(self):
        """ Perform the task this command is created for. """
        raise NotImplementedError()
