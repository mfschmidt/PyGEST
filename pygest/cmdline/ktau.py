import os
import pandas as pd
import argparse

from scipy.stats import kendalltau


class Ktau:
    """ A command to convert csv or tsv files to dataframes """
    # Normally, all commands would inherit from Command, but this is a simple command that needs none of the
    # overhead supplied by Command, so no need for it.

    def __init__(self, args):
        self._arguments = args
        self._parser = argparse.ArgumentParser(description="ktau calculator")
        self._add_arguments()
        self._args = self._parser.parse_args(self._arguments)

    def _add_arguments(self):
        """ Add command-specific arguments, then Command's generic arguments. """
        self._parser.add_argument("a",
                                  help="A csv or tsv file to compare with another.")
        self._parser.add_argument("b",
                                  help="A csv or tsv file to compare with another.")

    def _post_process_arguments(self):
        """ After arguments are processed, tweak what we need to. """
        if self._args.log == '':
            # We really don't need to log ktau results to a log file by default.
            self._args.log = 'null'

    def run(self):
        """ Read data from tsv files and calculate the order similarity of their ordered probes. """

        def vet_file(f):
            if os.path.isfile(f):
                return f
            else:
                print("File {} does not exist.".format(f))
                return None

        a = vet_file(self._args.a)
        b = vet_file(self._args.b)

        if a is None or b is None:
            return 1

        # Read each file provided
        df_a = pd.read_csv(a, sep='\t' if a[-4:] == '.tsv' else ',')
        df_b = pd.read_csv(b, sep='\t' if b[-4:] == '.tsv' else ',')

        tau, p = kendalltau(df_a.probe_id, df_b.probe_id)
        print("{:0.3f}".format(tau))

        return 0
