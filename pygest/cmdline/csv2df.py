import os
import pandas as pd

from pygest import algorithms
from pygest.cmdline.command import Command


class Csv2df(Command):
    """ A command to convert csv or tsv files to dataframes """

    def __init__(self, args, logger=None):
        super().__init__(args, command="csv2df", description="PyGEST csv2df command", logger=logger)

    def _add_arguments(self):
        """ Add command-specific arguments, then Command's generic arguments. """
        self._parser.add_argument("--csv", "--csvfile", "--csv_file", dest="csvfile",
                                  help="A csv or tsv file to convert to a pandas dataframe.")
        self._parser.add_argument("--hdr", "--header", "--hdrfile", "--hdr_file", "--headerfile", "--header_file",
                                  dest="hdrfile",
                                  help="A text file containing header rows that match the csvfile.")
        self._parser.add_argument("--df", "--dffile", "--df_file", "--out", "--outfile", "--out_file", dest="outfile")
        self._parser.add_argument("--data", dest="data", nargs='?', type=str, default='NONE',
                                  help="Where are the BIDS and cache directories?")
        super()._add_arguments()

    def run(self):
        """ Read data from csvfile, header from hdrfile, and pickle new dataframe to outfile. """

        # Keep track of any problems, then report them all together at the end.
        errors = []
        headers = None
        values = None
        outfile = None

        if "hdrfile" in self._args and self._args.hdrfile is not None:
            if os.path.isfile(self._args.hdrfile):
                headers = pd.read_csv(self._args.hdrfile, index_col='well_id', sep='\t')
            else:
                errors.append("File {} does not exist.".format(self._args.hdrfile))
        else:
            # These data are required, but may be embedded in the data file, so no worries, yet.
            pass

        if "csvfile" in self._args and self._args.csvfile is not None:
            if os.path.isfile(self._args.csvfile):
                if headers is None:
                    values = pd.read_csv(self._args.csvfile, header=0, index_col=0)
                    self._logger.info("Read {} x {} matrix. ".format(values.shape[0], values.shape[1]))
                    self._logger.info("We ASSUME the columns and indices match and have appropriate well_id values.")
                else:
                    """ Headers exist; and the matrix file exists.
                        Actual conversion happens right here """
                    values = pd.read_csv(self._args.csvfile, header=None, index_col=None)
                    values.columns = list(headers.index)
                    values.index = headers.index
                    self._logger.info("Read {} x {} matrix, and {} labels from the header file. ".format(
                        values.shape[0], values.shape[1], len(headers.index)
                    ))
            else:
                errors.append("File {} does not exist.".format(self._args.hdrfile))
        else:
            errors.append("A csv (or tsv) file must be supplied with --csvfile filename")

        if "outfile" in self._args and self._args.outfile is not None:
            outfile = self._args.outfile
        else:
            if len(errors) == 0:
                outfile = self._args.csvfile[: -4] + ".df"
        if outfile is not None and os.path.isfile(outfile):
            errors.append("Output file {} already exists. I won't overwrite it, ".format(self._args.outfile) +
                          "so please delete it manually, rename it, or choose another name (--outfile).")

        if len(errors) > 0:
            for e in errors:
                self._logger.error(e)
        else:
            if values is not None:
                values.to_pickle(outfile)
                self._logger.info("Matrix written to {}; Creating similarity file. ".format(outfile))
                values_conn = algorithms.make_similarity(values)
                values_conn.to_pickle(outfile[: -3] + "_sim.df")
                # any trouble would have thrown an exception in to_pickle
                self._logger.info("Matrix with {} shape saved to {} and {}.".format(
                    values.shape, outfile, outfile[: -3] + "_conn.df"))

        return len(errors)
