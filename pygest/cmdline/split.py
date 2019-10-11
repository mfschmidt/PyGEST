import os
import csv
import pandas as pd
import numpy as np

from pygest.convenience import path_to, split_file_name
from pygest.rawdata.glasser import glasser_parcel_map

from pygest.cmdline.command import Command


class Split(Command):

    def __init__(self, args, logger=None):
        super().__init__(args, command="split", description="PyGEST split command", logger=logger)

    def _add_arguments(self):
        """ Add command-specific arguments, then Command's generic arguments. """
        self._parser.add_argument("--donor", dest="donor", default="all",
                                  help="Which ABI donor, or split-half sample, would you like to include?")
        self._parser.add_argument("--hemisphere", dest="hemisphere", default="A",
                                  help="Which hemisphere would you like to include?")
        self._parser.add_argument("--samples", dest="samples", default="all",
                                  help="A subset of samples/parcels (well_ids), 'glasser' for those in glasser parcels")
        self._parser.add_argument("--probes", dest="probes", default="richiardi",
                                  help="Whose probes should we use? 'richiardi' or 'fornito' for now")
        self._parser.add_argument("--parcelby", dest="parcelby", default="wellid", choices=['wellid', 'glasser'],
                                  help="Specify a parcel by which to average wellids.")
        self._parser.add_argument("--dryrun", dest="dryrun", action='store_true', default=False,
                                  help="Report what would happen without actually doing it.")
        self._parser.add_argument("--seed", dest="seed", type=int, default=0,
                                  help="Provide a seed for randomizing the expression data shuffle.")
        self._parser.add_argument("--data", dest="data", nargs='?', type=str, default='NONE',
                                  help="Where are the BIDS and cache directories?")
        super()._add_arguments()

    def _post_process_arguments(self):
        """ This command logs to file, by default - others commands may not. """
        if self._args.log == '':
            self._args.log = path_to(self._command, self._args, path_type="result", log_file=True)

    def run(self):
        """ Split samples into a train and test set.

            An expression DataFrame is generated, then split in half into a train set and a test set.
            Each set is saved out to {PYGEST_DATA}/splits/{set string}/* as six files:
                2 csv lists of wellids included,
                4 df dataframes of expression similarity data: 2 split on wellid, 2 split on parcel
        """

        def average_expr_per_parcel(wellid_expression, parcel_map):
            """ Average expression values over all wellids in each parcel_map-defined parcel. """

            parcels = pd.DataFrame(
                data={'parcel': [parcel_map[x] for x in wellid_expression.columns]},
                index=wellid_expression.columns
            )
            parcel_means = {}
            for parcel in sorted(list(set(parcel_map.values()))):
                parcel_idx = parcels[parcels['parcel'] == parcel].index
                if len(parcel_idx) > 0:
                    parcel_means[parcel] = wellid_expression.loc[:, parcel_idx].mean(axis=1)
                else:
                    if self._args.verbose:
                        self._logger.debug("  parcel {} had no samples".format(parcel))
            return pd.DataFrame(data=parcel_means)

        def write_subset(prefix, splitby, df_wellid, df_parcel):
            """ Write out three files for samples, expression, and each parcellated expression. """

            d = {'splby': splitby, 'phase': prefix, 'seed': self._args.seed}
            base_path = os.path.join(
                path_to(self._command, self._args, path_type='split', include_file=False),
                "batch-{}{:05}".format(d['phase'], d['seed'])
            )
            os.makedirs(os.path.abspath(base_path), exist_ok=True)

            if self._args.dryrun:
                self._logger.info("NOT actually writing any files!")
            else:
                self._args.phase = prefix

                def write_a_split(df, parcelby):
                    d['parby'] = parcelby
                    with open(os.path.join(base_path, split_file_name(d, 'csv')), 'w') as f:
                        writer = csv.writer(f)
                        writer.writerow(df.columns)
                    df.to_pickle(os.path.join(base_path, split_file_name(d, 'df')))
                    self._logger.info("  final {}-split {} set is {} probes x {} {}s, {} {} wellids.".format(
                        splitby, prefix, df.shape[0], df.shape[1], parcelby,
                        'from' if splitby == 'wellid' else 'comprising', len(df.columns)
                    ))

                write_a_split(df_wellid, 'wellid')
                write_a_split(df_parcel, 'glasser')

        expr = self.data.expression(probes=self._args.probes, samples=self._args.samples)

        # Control the randomization and repeatability with a seed. Default is 0 if not specified at commandline.
        np.random.seed(self._args.seed)

        """ Split expression into train/test halves by first splitting on wellids, then averaging by parcel. """
        train_wellids = sorted(list(
            np.random.choice(expr.columns, int(len(expr.columns) / 2), replace=False)
        ))
        train_wellids_expr = expr.loc[:, train_wellids]
        train_wellids_expr_parcellated = average_expr_per_parcel(train_wellids_expr, glasser_parcel_map)
        write_subset("train", "wellid", train_wellids_expr, train_wellids_expr_parcellated)

        test_wellids = sorted([x for x in expr.columns if x not in train_wellids])
        test_wellids_expr = expr.loc[:, test_wellids]
        test_wellids_expr_parcellated = average_expr_per_parcel(test_wellids_expr, glasser_parcel_map)
        write_subset("test", "wellid", test_wellids_expr, test_wellids_expr_parcellated)
        # With the same seed, everything up to here ends up identical. After this, things differ. ??? Why ???

        """ Split expression into train/test halves by first averaging by parcel, then splitting on parcel. """
        expr_parcellated = average_expr_per_parcel(expr, glasser_parcel_map)

        train_parcels = sorted(list(
            np.random.choice(expr_parcellated.columns, int(len(expr_parcellated.columns) / 2), replace=False)
        ))
        train_parcels_expr = expr_parcellated.loc[:, train_parcels]
        train_wellids = [x for x in sorted(glasser_parcel_map.keys()) if glasser_parcel_map[x] in train_parcels]
        write_subset("train", "glasser", expr.loc[:, train_wellids], train_parcels_expr)

        test_parcels = sorted([x for x in expr_parcellated.columns if x not in train_parcels])
        test_parcels_expr = expr_parcellated.loc[:, test_parcels]
        test_wellids = [x for x in sorted(glasser_parcel_map.keys()) if glasser_parcel_map[x] in test_parcels]
        write_subset("test", "glasser", expr.loc[:, test_wellids], test_parcels_expr)

        """ Share some advice on how to use what we just built. """
        self._logger.info("To maximize probes in split-half data, run something like")
        p_and_s = "--probes fornito --samples glasser"
        s_and_p = "--splitby wellid --parcelby glasser"
        comparator = os.path.join(self.data.path_to('base'), 'conn', 'some_file.df')
        self._logger.info("    pygest push {} {} --minmax max --comparator {} --batch train{:05d} ...".format(
            p_and_s, s_and_p, comparator, self._args.seed
        ))
