import os


def load_ints_from_one_column_csv(filename):
    """ Load a csv file and convert the single-column to a python list of ints """
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename), "r") as f:
        return [int(x) for x in f.read().splitlines()]


def get_test_samples():
    """ Return a list of 128 well_ids to allow for consistent testing of filtered data. """
    return load_ints_from_one_column_csv("samples_test_128.csv")


def get_test_probes():
    """ Return a list of 4096 probe_ids to allow for consistent testing of filtered data. """
    return load_ints_from_one_column_csv("probes_test_4096.csv")


def get_schmidt_samples():
    """ Return a list of 1,762 well_ids from left cortex, as selected by Mike Schmidt. """
    return load_ints_from_one_column_csv("samples_schmidt_1762.csv")


def get_richiardi_samples():
    """ Return a list of 1,777 wellids used in Richardi, 2015. """
    return load_ints_from_one_column_csv("samples_richiardi_1777.csv")


def get_richiardi_probes():
    """ Return a list of 16,906 probes used in Richardi, 2015. """
    return load_ints_from_one_column_csv("probes_richiardi_16906.csv")


def get_glasser_samples():
    """ Return a list of 1280 well_ids that can be mapped into cortical Glasser parcels. """
    return load_ints_from_one_column_csv("samples_glasser_1280.csv")


def get_fornito_probes():
    """ Return a list of 15745 probes from the Arnatkeviciute, 2018 paper. """
    return load_ints_from_one_column_csv("probes_fornito_15745.csv")


def get_left_samples():
    """ Return a list of the 2,656 AHBA well_ids with x-axis locations less than zero. """
    return load_ints_from_one_column_csv("samples_left_2656.csv")


def get_right_samples():
    """ Return a list of the 2,656 AHBA well_ids with x-axis locations less than zero. """
    return load_ints_from_one_column_csv("samples_right_908.csv")


def get_nonlateral_samples():
    """ Return a list of the 138 AHBA well_ids not clearly assignable to left or right. """
    return load_ints_from_one_column_csv("samples_nonlateral_138.csv")
