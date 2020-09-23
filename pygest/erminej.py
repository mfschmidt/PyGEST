
def read_erminej(filename):
    """ Read output file from ErmineJ and return a dataframe.

        :param str filename: Full path to ermineJ output file
        :return: pandas dataframe containing ermineJ results
    """

    import re
    from io import StringIO
    import pandas as pd

    kept_lines = []
    dropped_lines = []
    head_count = 0
    data_count = 0
    drop_count = 0
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            matched = False

            # Keep the column header line that starts with '#!\t', but without those three characters.
            head_match = re.search('^#!\t', line)
            if head_match:
                head_count += 1
                matched = True
                kept_lines.append(line[3:].rstrip())

            # Keep the data lines that all start with '!\t', but we don't need the exclamation.
            data_match = re.search('^!\t', line)
            if data_match:
                data_count += 1
                matched = True
                kept_lines.append(line[2:].rstrip())

            # Track dropped lines, but we currently do nothing with them.
            if not matched:
                drop_count += 1
                dropped_lines.append(line.rstrip())

    # print("Kept {} head, {} data lines. Dropped {} lines.".format(head_count, data_count, drop_count))
    return pd.read_csv(StringIO("\n".join(kept_lines)), sep="\t").sort_values("Pval", ascending=True)
