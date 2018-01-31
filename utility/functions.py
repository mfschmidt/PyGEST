#!/usr/bin/env python3

"""
The following functions are a miscellaneous collection of utility functions
that can be useful in making code shorter and easier to read.
"""


def shortened_hash(s, n):
    """ Return a shortened string with the first and last bits of a hash
    :param s: the full string to shorten
    :param n: the desired length of the string returned
    :return: An n-character string with the first and last bits of s
    """
    side_len = int((n - 3) / 2)
    if len(s) <= n:
        return s
    else:
        return s[0:side_len] + "..." + s[(-1 * side_len):]


def hash_file(filename, block_size=32768):
    """ Return a 256-bit (64 character) sha256 hash of filename
    :param filename: the full path of the file to hash
    :param block_size: for large files, how big of a chunk should we read at a time
    :return: A 64-character string representing the 256-bit sha256 hash
    """
    import hashlib
    try:
        with open(filename, 'rb') as f:
            hasher = hashlib.sha256()
            while True:
                buf = f.read(block_size)
                if not buf:
                    break
                hasher.update(buf)
    except IOError:
        return None
    return hasher.hexdigest()


def tree(base_dir, padding='  ', print_files=True, is_last=False, is_first=False):
    """ Return a list of strings that can be combined to form ASCII-art-style
    directory listing
    :param base_dir: the path to explore
    :param padding: a string to prepend to each line
    :param print_files: True to print directories and files, False for just directories
    :param is_last: only used recursively
    :param is_first: only used recursively
    """
    import os
    out_lines = []
    if is_first:
        out_lines.append(base_dir)
    else:
        if is_last:
            out_lines.append(padding[:-2] + '└─ ' + os.path.basename(os.path.abspath(base_dir)))
        else:
            out_lines.append(padding[:-2] + '├─ ' + os.path.basename(os.path.abspath(base_dir)))
    if print_files:
        files = os.listdir(base_dir)
    else:
        files = [x for x in os.listdir(base_dir) if os.path.isdir(base_dir + os.sep + x)]
    if not is_first:
        padding = padding + '  '
    files = sorted(files, key=lambda s: s.lower())
    count = 0
    last = len(files) - 1
    for i, file in enumerate(files):
        count += 1
        path = base_dir + os.sep + file
        is_last = i == last
        if os.path.isdir(path):
            if count == len(files):
                if is_first:
                    out_lines = out_lines + tree(path, padding + '  ', print_files, is_last, False)
                else:
                    out_lines = out_lines + tree(path, padding + '  ', print_files, is_last, False)
            else:
                out_lines = out_lines + tree(path, padding + '│ ', print_files, is_last, False)
        else:
            if is_last:
                out_lines.append(padding + '└─ ' + file)
            else:
                out_lines.append(padding + '├─ ' + file)

    return out_lines
