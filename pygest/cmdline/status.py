import pygest as ge

from pygest.cmdline.command import Command


class Status(Command):
    """ A command to report the status of the data collected """

    def __init__(self, args, logger=None):
        super().__init__(args, command="push", description="PyGEST status command", logger=logger)

    def _add_arguments(self):
        """ Add command-specific arguments, then Command's generic arguments. """
        self._parser.add_argument("donor", default='all',
                                  help="A donor if status data should be filtered.")
        super()._add_arguments()

    def run(self):
        """ Read data from csvfile, header from hdrfile, and pickle new dataframe to outfile. """

        # Keep track of any problems, then report them all together at the end.
        self._logger.info("SOURCES:")
        self.data.log_status(regarding=self._args.donor)
        self._logger.info("RESULTS:")
        ge.reporting.log_status(self.data, self._args.data, regarding=self._args.donor, logger=self._logger)

        return 0
