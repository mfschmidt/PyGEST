#!/usr/bin/env python3

import os
import sys
import socket
import pkg_resources
import humanize
import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm
from datetime import datetime

from pygest import algorithms
from pygest.convenience import canned_map, canned_description, bids_clean_filename, path_to
from pygest.rawdata import richiardi, schmidt
from pygest.cmdline.command import Command


class Push(Command):
    """ A command to push Mantel correlations through a greedy algorithm """

    def __init__(self, args, logger=None):
        super().__init__(args, command="push", description="PyGEST push command", logger=logger)

    def _add_arguments(self):
        """ Add command-specific arguments, then Command's generic arguments. """
        self._parser.add_argument("--donor", default="all", nargs='?',
                                  help="Which ABI donor, or complete dataframe, would you like to use for expression?")
        self._parser.add_argument("--hemisphere", default="A", nargs='?',
                                  help="Which hemisphere would you like to include?")
        self._parser.add_argument("--samples", dest="samples", default="all",
                                  help="A subset of samples (well_ids) to include, ['cor', 'sub', 'glasser']")
        self._parser.add_argument("--probes", dest="probes", default="richiardi",
                                  help="Which probe-set should we use? 'richiardi' or 'fornito' for now")
        self._parser.add_argument("--splitby", dest="splitby", default="none",
                                  help="Specify a split-half portion of the data. ['wellid', 'parcel', 'none']")
        self._parser.add_argument("--parcelby", dest="parcelby", default="wellid",
                                  help="Specify parcel for parcellated data.")
        self._parser.add_argument("--batch", dest="batch", default="none",
                                  help="Looking for a specific batch, a split-half with a particular seed?")
        self._parser.add_argument("--comparator", dest="comparator", default="conn",
                                  help="Parcel-matched data to correlate 'conn' or 'cons' or 'dist' or a df file")
        self._parser.add_argument("--minmax", dest="direction", default="max",
                                  help="What direction to push the correlations? 'max' or 'min'?")
        self._parser.add_argument("-n", "--n_cpu", "--cores", dest="cores", default="0", type=int,
                                  help="How many cores should we use? 0 means to use {n_cpus} - 1")
        self._parser.add_argument("--algo", dest="algorithm", default="smrt",
                                  help="How aggressive should we be in finding max/min? 'once', 'smrt', 'evry'")
        self._parser.add_argument("--masks", dest="masks", default=[], nargs='+',
                                  help="Mask out some edges to nullify their effect. 'none', 'fine', 'coarse', '#'")
        self._parser.add_argument("--adj", dest="adjust", default="none",
                                  help="How should we correct for proximity? 'none', 'slope', 'linear', 'log'")
        # As a hack, I'm adding "slope" to indicate a regression, but without adjusting for distance.
        # Eventually, we should be able to specify a target of maxr, maxm, minr, minm, etc. with an adjustment for each.
        self._parser.add_argument("--shuffle", dest="shuffle", default='none',
                                  help="Shuffle columns. ['agno', 'dist', 'edge', 'be08'] for null distributions.")
        self._parser.add_argument("--comparator-similarity", dest="comparatorsimilarity", action="store_true",
                                  default=False,
                                  help="Correlate comparator before running, generating comparator similarity matrix.")
        self._parser.add_argument("--output-intermediate-vertices", dest="output_intvertices", action="store_true",
                                  default=False,
                                  help="Write edge vertices to a subdirectory after each probe removal.")
        self._parser.add_argument("--expr-norm", dest="expr_norm", default="none",
                                  help="Use a pre-computed ajustment for expression data, like 'srs'.")
        self._parser.add_argument("--seed", dest="seed", type=int, default=0,
                                  help="Provide a seed for randomizing the expression data shuffle.")
        self._parser.add_argument("--upload", dest="upload", nargs="+", default=[],
                                  help="Request to have results uploaded to an external location.")
        self._parser.add_argument("--dryrun", dest="dryrun", action='store_true', default=False,
                                  help="Report what would happen without actually doing it.")
        self._parser.add_argument("--data", dest="data", nargs='?', type=str, default='NONE',
                                  help="Where are the BIDS and cache directories?")
        super()._add_arguments()

    def _post_process_arguments(self):
        """ Interpret arguments before reporting them or moving on to run anything. """
        if self._args.direction.lower() == 'min':
            self._args.going_up = False
            self._args.going_down = True
        else:
            self._args.going_up = True
            self._args.going_down = False

        self._args.hemisphere = self._args.hemisphere.upper()[0]  # 'L', 'R', or 'A'

        # Standardize to a single value to avoid mis-spellings f'ing us later
        if self._args.algorithm in algorithms.algorithms:
            self._args.algorithm = algorithms.algorithms[self._args.algorithm]
        else:
            self._args.algorithm = algorithms.algorithms['smrt']

        if self._args.dryrun:
            self._args.verbose = True

        if self._args.shuffle == 'none' and self._args.seed != 0:
            # Apply the default shuffle if a seed is specified without a shuffle type.
            self._args.shuffle = 'dist'

        # This command logs to file, by default - others commands may not
        if self._args.log == '':
            self._args.log = path_to(self._command, self._args, path_type="result", log_file=True)
            print("<in Push Command _post_process_arguments> No logfile supplied; logging to {}".format(self._args.log))

    def run(self):
        """ Figure out the most influential genes by dropping each least influential, cumulatively.

            The pandas DataFrame object is written as a tsv-file to /{data}/derivatives/.../{name}.tsv
        """

        # Figure out our temporary and final file names
        base_path = path_to(self._command, self._args)

        # Pull data
        exp = self.get_expression()
        cmp = self.get_comparator(self._args.comparator, exp.columns)
        dst = self.get_comparator('dist', exp.columns)
        valid_samples = sorted(list(set(exp.columns).intersection(set(cmp.columns)).intersection(set(dst.columns))))

        # Should we null the distribution first?
        shuffle_edge_seed = None
        shuffle_bin_size = None
        orig_cols = exp.columns.copy(deep=True)

        # Report on changes to mean distance by shuffling.
        def log_distance_changes(df, df_dist, forward_map):
            """ Report the mean distance between old and new shuffled wellids.
            :param df: shuffled dataframe
            :param df_dist: distance dataframe
            :param forward_map: dict mapping original columns as keys to shuffled replacements as values
            :return: None
            """
            reverse_map = {v: k for k, v in forward_map.items()}
            distances = [df_dist.loc[col, forward_map[col]] for col in df.columns.map(reverse_map)]
            # for i, real_id in enumerate(df.columns.map(reverse_map)):
            #     distances.append(df_dist.loc[real_id, df.columns[i]])
            self._logger.debug("      Mean distance between old and new loci is {:0.2f}".format(np.mean(distances)))
            self._logger.debug("    : {}, ..., {}".format(", ".join("{:0.2f}".format(x) for x in distances[:5]),
                                                          ", ".join("{:0.2f}".format(x) for x in distances[-5:]), ))

        self._logger.debug("Orig: {}, ..., {}".format(", ".join(str(x) for x in exp.columns[:5]),
                                                      ", ".join(str(x) for x in exp.columns[-5:])))
        if self._args.shuffle in ['agno', 'raw', 'dist', ]:
            exp, shuf_map = algorithms.cols_shuffled(exp, dist_df=dst, algo=self._args.shuffle, seed=self._args.seed)
            self._logger.debug("{}-shuffled: {}, ..., {}".format(
                self._args.shuffle,
                ", ".join(str(x) for x in exp.columns[:5].map(shuf_map)),
                ", ".join(str(x) for x in exp.columns[-5:].map(shuf_map))
            ))
            log_distance_changes(exp, dst, shuf_map)
            shuffle_map = pd.DataFrame(
                {'orig': orig_cols, 'shuf': exp.columns.map(shuf_map), 'kept': orig_cols.isin(valid_samples)}
            )
            pickle.dump(shuffle_map, open(os.path.join(base_path + ".shuffle_map.df"), "wb"))
        elif self._args.shuffle in ['edge', 'edges', 'bin', ]:
            # The shuffle_edge_seed variable indicates the need for bin-edge-shuffling at each iteration.
            shuffle_edge_seed = self._args.seed
            shuffle_bin_size = 0
        elif self._args.shuffle[:2] == "be":
            shuffle_edge_seed = self._args.seed
            shuffle_bin_size = int(self._args.shuffle[3:])

        # This alignment must happen BEFORE distance-masking, then never again. Future unlabeled vectors MUST match.
        exp = exp.loc[:, valid_samples]
        cmp = cmp.loc[valid_samples, valid_samples]
        dst = dst.loc[valid_samples, valid_samples]

        intermediate_path = ""
        if self._args.output_intvertices:
            intermediate_path = path_to(self._command, self._args, dir_for_intermediates=True)

        # DEBUGGING: Write distances to csv to check distributions later.
        # pd.DataFrame(distances, columns=["d",]).to_csv(
        #     "/home/mike/" + args.donor + "_" + bids_clean_filename(args.comparator) + "_" + args.shuffle + ".csv",
        #     index=False
        # )

        # If we've already done this, don't waste the time.
        if os.path.exists(base_path + '.tsv'):
            self._logger.info("{} already exists in {}. no need to {}imize the same correlations again.".format(
                os.path.basename(base_path + '.tsv'), os.path.abspath(base_path), self._args.direction
            ))
        else:
            self._logger.info("Removing probes to {}imize correlation.".format(self._args.direction))
            v_mask = self.cum_mask(exp, self._args.masks, self._args.parcelby)
            gene_df = algorithms.push_score(
                exp, cmp, dst, algo=self._args.algorithm, ascending=self._args.going_up, mask=v_mask,
                adjust=self._args.adjust,
                dump_intermediates=intermediate_path,
                edge_tuple=(shuffle_edge_seed, shuffle_bin_size),
                progress_file=base_path + '.partial.df', cores=self._args.cores, logger=self._logger
            )
            self._logger.info("Saving r-{}imization over gene removal to {}.".format(
                self._args.direction, base_path + '.tsv'
            ))
            gene_df.sort_index(ascending=False).to_csv(
                base_path + '.tsv', sep='\t', na_rep="n/a", float_format="%0.20f"
            )
            # The complete list is now written to tsv; get rid of the temporary cached list.
            os.remove(base_path + '.partial.df',)

            self.write_sidecar(base_path)

    def do_order(self):
        """ Figure out the most influential genes by whacking each only once.

            The pandas DataFrame object is pickled to /{data}/results/{name}.df
        """

        exp = self.get_expression()
        cmp = self.get_comparator(self._args.comparator, exp.columns)
        dst = self.get_comparator('dist', exp.columns)

        base_path = path_to(self._command, self._args)

        # Order probes once
        sample_type = 'glasser' if self._args.donor.lower().startswith('glasser') else 'wellid'
        v_mask = self.cum_mask(exp, self._args.masks, sample_type)
        probe_order = algorithms.reorder_probes(
            exp, cmp, dst,
            ascending=self._args.going_up, procs=self._args.cores, mask=v_mask, adjust=self._args.adjust,
            include_full=True, logger=self._logger
        )
        self._logger.info("Saving probe order to {}.".format(base_path + '.tsv'))
        probe_order.sort_values(by=['delta'], ascending=self._args.going_up).to_csv(
            base_path + '.tsv', sep='\t', na_rep="n/a", float_format="%0.20f"
        )

        # And leave some notes about the process
        self.write_sidecar(base_path)

    def write_sidecar(self, base_name):
        """ Write a json file to accompany other files using base_name

        :param base_name: The full path, sans extension, of other related files
        """

        end_time = datetime.now()

        with open(base_name + '.json', 'a+') as f:
            f.write("{\n")
            f.write("    \"host\": \"{}\",\n".format(socket.gethostname()))
            f.write("    \"command\": \"{}\",\n".format(" ".join(sys.argv[:])))
            f.write("    \"blas\": \"{} thread{}\",\n".format(
                os.environ['OPENBLAS_NUM_THREADS'],
                '' if int(os.environ['OPENBLAS_NUM_THREADS']) == 1 else 's'
            ))
            try:
                f.write("    \"pygest version\": \"{}\",\n".format(pkg_resources.require("pygest")[0].version))
            except pkg_resources.DistributionNotFound:
                f.write("    \"pygest version\": \"running uninstalled without version\",\n")
            f.write("    \"log\": \"{}\",\n".format(base_name + '.log'))
            f.write("    \"data\": \"{}\",\n".format(base_name + '.tsv'))
            f.write("    \"began\": \"{}\",\n".format(self._args.beginning.strftime("%Y-%m-%d %H:%M:%S")))
            f.write("    \"completed\": \"{}\",\n".format(end_time.strftime("%Y-%m-%d %H:%M:%S")))
            f.write("    \"elapsed\": \"{}\",\n".format(end_time - self._args.beginning))
            f.write("    \"duration\": \"{}\",\n".format(humanize.naturaldelta(end_time - self._args.beginning)))
            f.write("}\n")

    def one_mask(self, df, mask_type, sample_type):
        """ return a vector of booleans from the lower triangle of a matching-matrix based on 'mask_type'

        :param df: pandas.DataFrame with samples as columns
        :param str mask_type: A list of strings to specify matching masks, or a minimum distance to mask out
        :param str sample_type: Samples can be 'wellid' or 'parcelid'
        :return: Boolean 1-D vector to remove items (False values in mask) from any sample x sample triangle vector
        """

        # If mask is a number, use it as a distance filter
        try:
            # Too-short values to mask out are False, keepers are True.
            min_dist = float(mask_type)
            distance_vector = self.data.distance_vector(df.columns, sample_type=sample_type)
            if len(distance_vector) != (len(df.columns) * (len(df.columns) - 1)) / 2:
                self._logger.warn("        MISMATCH in expr and dist!!! Some sample IDs probably not found.")
            mask_vector = np.array(distance_vector > min_dist, dtype=bool)
            self._logger.info("        masking out {:,} of {:,} edges closer than {}mm apart.".format(
                np.count_nonzero(np.invert(mask_vector)), len(mask_vector), min_dist
            ))
            self._logger.info("        mean dist of masked edges: {:0.2f}; unmasked: {:0.2f}.".format(
                np.mean(distance_vector[~mask_vector]), np.mean(distance_vector[mask_vector]),
            ))
            return mask_vector
        except TypeError:
            pass
        except ValueError:
            pass

        # Mask is not a number, see if it's a pickled dataframe
        if os.path.isfile(mask_type):
            with open(mask_type, 'rb') as f:
                mask_df = pickle.load(f)
            if isinstance(mask_df, pd.DataFrame):
                # Note what we started with so we can report after we tweak the dataframe.
                # Too-variant values to mask out are False, keepers are True.
                orig_vector = mask_df.values[np.tril_indices(n=mask_df.shape[0], k=-1)]
                orig_falses = np.count_nonzero(~orig_vector)
                orig_length = len(orig_vector)
                self._logger.info("Found {} containing {:,} x {:,} mask".format(
                    mask_type, mask_df.shape[0], mask_df.shape[1]
                ))
                self._logger.info("    generating {:,}-len vector with {:,} False values to mask.".format(
                    orig_length, orig_falses
                ))

                # We can only use well_ids found in BOTH df and our new mask, make shapes match.
                unmasked_ids = [well_id for well_id in df.columns if well_id not in mask_df.columns]
                usable_ids = [well_id for well_id in df.columns if well_id in mask_df.columns]
                usable_df = mask_df.reindex(index=usable_ids, columns=usable_ids)
                usable_vector = usable_df.values[np.tril_indices(n=len(usable_ids), k=-1)]
                usable_falses = np.count_nonzero(~usable_vector)
                usable_length = len(usable_vector)
                self._logger.info("    {:,} well_ids not found in the mask; padding with Falses.".format(
                    len(unmasked_ids)
                ))
                pad_rows = pd.DataFrame(np.zeros((len(unmasked_ids), len(mask_df.columns)), dtype=bool),
                                        columns=mask_df.columns, index=unmasked_ids)
                mask_df = pd.concat([mask_df, pad_rows], axis=0)
                pad_cols = pd.DataFrame(np.zeros((len(mask_df.index), len(unmasked_ids)), dtype=bool),
                                        columns=unmasked_ids, index=mask_df.index)
                mask_df = pd.concat([mask_df, pad_cols], axis=1)
                mask_vector = mask_df.values[np.tril_indices(n=mask_df.shape[0], k=-1)]
                mask_falses = np.count_nonzero(~mask_vector)
                mask_trues = np.count_nonzero(mask_vector)
                self._logger.info("    padded mask matrix out to {:,} x {:,}".format(
                    mask_df.shape[0], mask_df.shape[1]
                ))
                self._logger.info("      with {:,} True, {:,} False, {:,} NaNs in triangle.".format(
                    mask_trues, mask_falses, np.count_nonzero(np.isnan(mask_vector))
                ))

                shaped_mask_df = mask_df.reindex(index=df.columns, columns=df.columns)
                shaped_vector = shaped_mask_df.values[np.tril_indices(n=len(df.columns), k=-1)]
                self._logger.info("    masking out {:,} (orig {:,}, {:,} usable) hi-var".format(
                    np.count_nonzero(~shaped_vector), orig_falses, usable_falses,
                ))
                self._logger.info("      of {:,} (orig {:,}, {:,} usable) edges.".format(
                    len(shaped_vector), orig_length, usable_length
                ))
                return shaped_vector
            else:
                self._logger.warning("{} is a file, but not a pickled dataframe. Skipping this mask.".format(mask_type))
                do_nothing_mask = np.ones((len(df.columns), len(df.columns)), dtype=bool)
                return do_nothing_mask[np.tril_indices(n=len(df.columns), k=-1)]

        # Mask is not a number, so treat it as a matching filter
        if mask_type[:4] == 'none':
            items = list(df.columns)
        elif mask_type[:4] == 'fine':
            items = self.data.samples(samples=df.columns)['fine_name']
        elif mask_type[:6] == 'coarse':
            items = self.data.samples(samples=df.columns)['coarse_name']
        else:
            items = self.data.samples(samples=df.columns)['structure_name']
        mask_array = np.ndarray((len(items), len(items)), dtype=bool)

        # There is, potentially, a nice vectorized way to mark matching values as True, but I can't find it.
        # So, looping works and is easy to read, although it might cost us a few extra ms.
        for i, y in enumerate(items):
            for j, x in enumerate(items):
                # Generate one edge of the match matrix
                mask_array[i][j] = True if mask_type == 'none' else (x != y)
        mask_vector = mask_array[np.tril_indices(n=mask_array.shape[0], k=-1)]

        print("        masking out {:,} of {:,} '{}' edges.".format(
            sum(np.invert(mask_vector)), len(mask_vector), mask_type
        ))

        # if len(mask_vector) == 0:
        #     mask_vector = np.ones(int(len(df.columns) * (len(df.columns) - 1) / 2), dtype=bool)

        return mask_vector

    def cum_mask(self, df, mask_types, sample_type):
        """ return a cumulate vector of booleans from the lower triangle of each mask specified in mask_types

        :param df: pandas.DataFrame with samples as columns
        :param str mask_types: A list of strings to specify matching masks, or a minimum distance to mask out
        :param str sample_type: Samples can be 'wellid' or 'parcelid'
        :return: A boolean vector to remove unwanted items from any sample x sample triangle vector
        """

        # Generate a mask of all True values. We can then use it as-is or 'logical and' it with others.
        full_mask = self.one_mask(df, 'none', sample_type)

        if mask_types == [] or mask_types == ['none']:
            return full_mask

        for mask_type in mask_types:
            full_mask = full_mask & self.one_mask(df, mask_type, sample_type)

        # The resulting mask should be a logical and mask of all masks in mask_types
        self._logger.info("Final mask ({}) is {:,} True, {:,} False, {:,}-length".format(
            "+".join(bids_clean_filename(mask_types)),
            np.count_nonzero(full_mask),
            np.count_nonzero(~full_mask),
            len(full_mask)
        ))
        return full_mask

    def get_expression(self):
        """ Gather expression data. """

        self._logger.info("Gathering expression data for {}.".format(self._args.donor))

        expr = None

        # Figure out where to get the expression dataframe.
        if self._args.donor.lower().endswith(".df") and os.path.isfile(self._args.donor):
            # If we're passed a file, just load it.
            with open(self._args.donor, "rb") as f:
                expr = pickle.load(f)
        elif self._args.donor.lower() == 'test':
            # If we're testing, load the canned test set.
            expr = self.data.expression(probes='test', samples='test')
        elif (self._args.splitby != 'none') and (self._args.batch != 'none'):
            # Looking for split-half data
            possible_expression_file = path_to(self._command, self._args, path_type='split')
            if self._args.expr_norm == 'srs' and ".srs." not in possible_expression_file:
                possible_expression_file = possible_expression_file.replace(".df", ".srs.df")
            elif self._args.expr_norm in ['none', 'raw', ] and 'raw' not in possible_expression_file:
                possible_expression_file = possible_expression_file.replace(".df", ".raw.df")
            if ".srs." in possible_expression_file:
                self._args.expr_norm = 'srs'  # Ensure output files are appropriately named.
            if ".raw." in possible_expression_file:
                self._args.expr_norm = 'none'  # Ensure output files are appropriately named.
            self._logger.info("    Attempting to load {} for split expression.".format(possible_expression_file))
            if os.path.isfile(possible_expression_file):
                with open(possible_expression_file, "rb") as f:
                    expr = pickle.load(f)
                self._logger.info("    Loading [{} x {}] expr directly from {}".format(
                    expr.shape[0], expr.shape[1], possible_expression_file
                ))
        else:
            # This should only execute if 'srs' is set AND none of the three clauses above are true.
            if self._args.expr_norm == 'srs':
                expr = self.data.expression(
                    probes=self._args.probes,
                    samples=self.data.samples(donor=self._args.donor, hemisphere=self._args.hemisphere),
                    normalize='exprsrs'
                )
            else:
                """ The default, 'normal' expression for most Mantel maximizations """
                expr = self.data.expression(
                    probes=self._args.probes,
                    samples=self.data.samples(
                        donor=self._args.donor,
                        hemisphere=self._args.hemisphere,
                        samples=self._args.samples
                    )
                )

        # Now that we have a dataframe, filter it by the sample-space specified.
        if expr is not None:
            self._logger.info("    retrieved [{}-probe X {}-sample] DataFrame.".format(
                len(expr.index), len(expr.columns)
            ))
            # Cache a list of richiardi samples; we need to filter both IN and OUT of this list
            len_before = len(expr.columns)

            if self._args.samples == 'cor':
                expr = expr[[well_id for well_id in expr.columns if well_id in richiardi.richiardi_samples]]
            elif self._args.samples == 'sub':
                expr = expr[[well_id for well_id in expr.columns if well_id not in richiardi.richiardi_samples]]
            elif self._args.samples == 'scx':
                expr = expr[[well_id for well_id in expr.columns if well_id in schmidt.schmidt_samples]]

            # For samples like 'glasser', nothing is done here. They loaded from their own dataframe, as-is

            self._logger.info("    {}-only data requested, keeping {} of the original {} samples.".format(
                self._args.samples, len(expr.columns), len_before
            ))
        else:
            self._logger.info("    Failed to find expression dataframe from {}_{}_{}.".format(
                self._args.donor, self._args.splitby, self._args.samples
            ))
            return pd.DataFrame()

        return expr

    def get_comparator(self, name, sample_filter):
        """ Gather comparison data

        Comparators can be named:
            "conn": The original connectivity matrix from INDI,
            "conregr1": A connectivity matrix with improved regressors from 9/2/2018,
            "conhalf1": A connectivity matrix generated from half the INDI population,
            "conhalf2": A connectivity matrix generated from half the INDI population,
            "cons": A connectivity similarity matrix generated from the original conn,
            "consregr1": A connectivity similarity matrix generated from conregr1,
            "conshalf1": A connectivity similarity matrix generated from conhalf1,
            "conshalf2": A connectivity similarity matrix generated from conhalf1,
            "dist": A distance matrix,
            "resid"
        Or they can come from a pickled dataframe file:
            "~/some_new_connectivity_matrix.df"
        """

        # Older versions tested for (name[:4].lower() == 'conn') so to avoid wrong data without any warnings
        # in older versions, none of the con* strings should start with 'conn'. They should also avoid _ or -
        # in their names because that could be problematic with BIDS naming.

        if os.path.isfile(name):
            # We have a named dataframe file we need to use.
            self._logger.info("Explicitly asked for comparator data from {}, calling it {}.".format(
                name, bids_clean_filename(name)
            ))
            # Avoid all the caching complexity and just read the comparator in.
            if self._args.comparatorsimilarity:
                sim_name = name[: name.rfind('.')] + '_sim.df'
                if os.path.isfile(sim_name):
                    self._logger.info("  found pre-computed similarity matrix at {}...".format(name))
                    with open(sim_name, 'rb') as f:
                        comp = pickle.load(f)
                else:
                    self._logger.debug("  loading comparator matrix from {f}".format(f=name))
                    with open(name, 'rb') as f:
                        comp = pickle.load(f)
                    self._logger.info("  and creating similarity matrix from it...")
                    comp = algorithms.make_similarity(comp)
                    # Saving this out (combined with logic above) prevents us from having to rebuild these.
                    comp.to_pickle(sim_name)
            else:
                self._logger.debug("  loading comparator matrix from {f}".format(f=name))
                with open(name, 'rb') as f:
                    comp = pickle.load(f)

            # Match up samples with expression samples
            common_samples = [x for x in sample_filter if x in comp.columns]

            self._logger.info("    loaded [{} x {}], using [{} x {}] comparator {}matrix.".format(
                len(comp.index), len(comp.columns), len(common_samples), len(common_samples),
                'similarity ' if self._args.comparatorsimilarity else ''
            ))
            return comp.loc[common_samples, common_samples]
        elif name.lower() == 'conn':
            self._logger.info("Gathering {} connectivity data (for {}).".format(
                canned_description[canned_map[name]], self._args.donor
            ))
            # We should have a square connectivity matrix from INDI, 2551 x 2551
            comp = self.data.connectivity(canned_map[name], samples=sample_filter)
            self._logger.info("    using [{} X {}] connectivity matrix.".format(
                len(comp.index), len(comp.columns)
            ))
        elif name.lower() == 'cons':
            self._logger.info("Gathering {} connectivity similarity data (for {}).".format(
                canned_description[canned_map[name]], self._args.donor
            ))
            # We should have a square connectivity matrix from INDI, 2551 x 2551
            comp = self.data.connectivity_similarity(canned_map[name], samples=sample_filter)
            self._logger.info("    using [{} X {}] connectivity similarity matrix.".format(
                len(comp.index), len(comp.columns)
            ))
        elif name[:4].lower() == 'dist':
            self._logger.info("Gathering distance data for {}.".format(self._args.donor))
            # We need to generate a square distance matrix from selected samples
            comp = self.data.distance_dataframe(sample_filter, sample_type=self._args.parcelby)
            self._logger.info("    using [{} X {}] distance matrix.".format(comp.shape[0], comp.shape[1]))
        elif name[:5].lower() == 'resid':
            # We need to adjust our comparator for a covariate before using it.
            comp_name = name[5:9]
            comp = self.get_comparator(comp_name, sample_filter)
            endog = pd.DataFrame(data={comp_name: comp.values[np.tril_indices(n=comp.shape[0], k=-1)]})
            nuisance_name = name[12:16] if name[9:12] == 'log' else name[9:13]
            nuisance = self.get_comparator(nuisance_name, sample_filter)
            # comparator and nuisance aren't necessarily the same size; make them so
            if len(nuisance) < len(comp):
                comp = comp.loc[nuisance.index, nuisance.index]
            elif len(comp) < len(nuisance):
                nuisance = nuisance.loc[comp.index, comp.index]
            # Log-transform nuisance (assumed distance) if requested
            if name[9:12] == 'log':
                exog_vals = np.log(nuisance.values[np.tril_indices(n=nuisance.shape[0], k=-1)])
            else:
                exog_vals = nuisance.values[np.tril_indices(n=nuisance.shape[0], k=-1)]
            # Add a constant, to allow a y-intercept, and run the linear model
            exog = sm.add_constant(pd.DataFrame(data={nuisance_name: exog_vals}))
            model = sm.GLM(endog, exog, family=sm.families.Gaussian())
            comp.values[np.tril_indices(n=comp.shape[0], k=-1)] = model.fit().resid_pearson
            # The precomp DataFrame will be lower-triangle-adjusted and upper-triangle-raw. We only extract the lower
            # triangle later, so this is adequate. Mapping the order of triu_indices from tril_indices can be done,
            # but would be a PITA, especially when it will simply be tossed away later.
            return comp
        else:
            self._logger.warning("Expression can be assessed vs connectivity or distance.")
            self._logger.warning("I don't understand '{}', and cannot continue.".format(name))
            sys.exit()

        return comp
