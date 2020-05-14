"""This holds utility functions."""

from statistics import mean
import os
import re
import sys
import argparse
from datetime import timedelta
from collections.abc import Iterable
import shutil

import h5py
import numpy as np
import logging
import configparser
from pprint import pprint
import argparse
import seaborn as sns
import matplotlib.pyplot as plt


def check_fn(item):
    if isinstance(item, list) or isinstance(item, np.ndarray):
        if len(item) == 0:
            return np.nan
    return item


def boolean_indexing(v, fillval=np.nan):
    """Index a numpy array using a boolean mask."""
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def daterange(start_date, end_date):
    """Yield a generator of dates between start and end date."""
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def make_dir_if_not_exists(location):
    """Make directory structure for given location."""
    os.makedirs(location, exist_ok=True)


def mycolors(subject, colors_dict=None):
    """Colour options for subject based on number."""
    try:
        i = int(subject)
        if i > 10:
            i = i % 4
        mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange',
                    'tab:brown', 'deeppink', 'tab:olive', 'tab:pink',
                    'steelblue', 'firebrick', 'mediumseagreen']
        color = mycolors[i]
    except ValueError:
        if colors_dict == None:
            print('Color input for required')
            exit(-1)
        color = colors_dict[subject]
    return color


def split_list(list, chunk_limit):
    """Split a list into small chunks based on chunk_limit."""
    new_list = [list[i:i + chunk_limit]
                for i in range(0, len(list), chunk_limit)]
    return new_list


def walk_dict(d, depth=0):
    """Walk a Dictionary."""
    for k, v in sorted(d.items(), key=lambda x: x[0]):
        spaces = ("  ") * depth
        if hasattr(v, "items"):
            print(spaces + ("%s" % k))
            walk_dict(v, depth + 1)
        else:
            print(spaces + "%s %s" % (k, v))


def get_attrs(d, depth=0):
    """Walk through attributes."""
    for k, v in sorted(d.items(), key=lambda x: x[0]):
        spaces = ("  ") * depth
        if len(v.attrs) != 0:
            print("{}{} has attrs {}".format(
                spaces, k, list(v.attrs.items())
            ))
        if hasattr(v, "items"):
            get_attrs(v, depth=depth + 1)


def print_h5(file_location):
    """Print a summary of a h5 file."""
    import h5py
    with h5py.File(file_location, 'r', libver='latest') as f:
        for key, val in f.attrs.items():
            print(key, val)
        print()
        walk_dict(f)
        print()


def has_ext(filename, ext):
    """
    Check if the filename ends in the extension.

    Parameters
    ----------
    filename : str
        The name of the file
    ext : str
        The extension, may have leading dot (e.g txt == .txt)

    Returns
    -------
    bool indicating if the filename has the extension

    """
    if ext is None:
        return True
    if ext[0] != ".":
        ext = "." + ext
    return filename[-len(ext):].lower() == ext.lower()


def get_all_files_in_dir(
        in_dir, ext=None, return_absolute=True,
        recursive=False, verbose=False, re_filter=None):
    """
    Get all files in the directory with the given extension.

    Parameters
    ----------
    in_dir : str
        The absolute path to the directory
    ext : str, optional. Defaults to None.
        The extension of files to get.
    return_absolute : bool, optional. Defaults to True.
        Whether to return the absolute filename or not.
    recursive: bool, optional. Defaults to False.
        Whether to recurse through directories.
    verbose: bool, optional. Defaults to False.
        Whether to print the files found.

    Returns
    -------
    List : A list of filenames

    """
    if not os.path.isdir(in_dir):
        print("Non existant directory " + str(in_dir))
        return []

    def match_filter(f):
        if re_filter is None:
            return True
        if not isinstance(re_filter, list):
            search_res = re.search(re_filter, f)
            return search_res is not None
        else:
            for re_filt in re_filter:
                search_res = re.search(re_filt, f)
                if search_res is None:
                    return False
                return True

    def ok_file(root_dir, f):
        return (
            has_ext(f, ext) and match_filter(f) and
            os.path.isfile(os.path.join(root_dir, f)))

    def convert_to_path(root_dir, f):
        return os.path.join(root_dir, f) if return_absolute else f

    if verbose:
        print("Adding following files from {}".format(in_dir))

    if recursive:
        onlyfiles = []
        for root, _, filenames in os.walk(in_dir):
            start_root = root[:len(in_dir)]

            if len(root) == len(start_root):
                end_root = ""
            else:
                end_root = root[len(in_dir + os.sep):]
            for filename in filenames:
                filename = os.path.join(end_root, filename)
                if ok_file(start_root, filename):
                    to_add = convert_to_path(start_root, filename)
                    if verbose:
                        print(to_add)
                    onlyfiles.append(to_add)

    else:
        onlyfiles = [
            convert_to_path(in_dir, f) for f in sorted(os.listdir(in_dir))
            if ok_file(in_dir, f)
        ]
        if verbose:
            for f in onlyfiles:
                print(f)

    if verbose:
        print()
    return onlyfiles


def log_exception(ex, more_info=""):
    """
    Log an expection and additional info.

    Parameters
    ----------
    ex : Exception
        The python exception that occured
    more_info :
        Additional string to log

    Returns
    -------
    None

    """
    template = "{0} because exception of type {1} occurred. Arguments:\n{2!r}"
    message = template.format(more_info, type(ex).__name__, ex.args)
    print(message)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def save_dict_to_csv(filename, d):
    """Save d to a file."""
    with open(filename, "w") as f:
        for k, v in d.items():
            out_str = k.replace(" ", "_")
            if isinstance(v, Iterable):
                if isinstance(v, np.ndarray):
                    v = v.flatten()
                else:
                    v = np.array(v).flatten()
                str_arr = [str(x) for x in v]
                out_str = out_str + "," + ",".join(str_arr)
            else:
                out_str += "," + str(v)
            f.write(out_str + "\n")


def make_path_if_not_exists(fname):
    """Makes directory structure for given fname"""
    os.makedirs(os.path.dirname(fname), exist_ok=True)


def setup_logging(in_dir):
    fname = os.path.join(in_dir, 'nc_output.log')
    if os.path.isfile(fname):
        open(fname, 'w').close()
    logging.basicConfig(
        filename=fname, level=logging.DEBUG)
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(level=logging.WARNING)


def print_config(config, msg=""):
    """Prints the contents of a config file"""
    if msg != "":
        print(msg)
    config_dict = [{x: tuple(config.items(x))} for x in config.sections()]
    pprint(config_dict, width=120)


def read_cfg(location, verbose=True):
    config = configparser.ConfigParser()
    config.read(location)

    if verbose:
        print_config(config, "Program started with configuration")
    return config


def parse_args(parser, verbose=True):
    args, unparsed = parser.parse_known_args()

    if len(unparsed) != 0:
        print("Unrecognised command line argument passed")
        print(unparsed)
        exit(-1)

    if verbose:
        if len(sys.argv) > 1:
            print("Command line arguments", args)
    return args


def get_dirs_matching_regex(start_dir, re_filters=None, return_absolute=True):
    """
    Recursively get all directories from start_dir that match regex.
    Parameters
    ----------
    start_dir : str
        The path to the directory to start at.
    re_filter : str, optional. Defaults to None.
        The regular expression to match.
        Returns all directories is passed as None.
    Returns
    -------
    list
        A list of directories matching the regex.
    """
    if not os.path.isdir(start_dir):
        raise ValueError("Non existant directory " + str(start_dir))

    def match_filter(f):
        if re_filters is None:
            return True
        for re_filter in re_filters:
            search_res = re.search(re_filter, f)
            if search_res is None:
                return False
        return True

    dirs = []
    for root, _, _ in os.walk(start_dir):
        start_root = root[:len(start_dir)]

        if len(root) == len(start_root):
            end_root = ""
        else:
            end_root = root[len(start_dir + os.sep):]

        if match_filter(end_root):
            to_add = root if return_absolute else end_root
            dirs.append(to_add)
    return dirs


def interactive_refilt(start_dir, ext=None, write=False, write_loc=None):
    re_filt = ""

    # Do the interactive setup
    files = get_all_files_in_dir(
        start_dir, re_filter=None, return_absolute=False,
        ext=ext, recursive=True)
    print("Without any regex, the result from {} is:".format(start_dir))
    for f in files:
        print(f)
    while True:
        this_re_filt = input(
            "Please enter the regexes seperated by SIM_SEP to test or quit / qt to move on:\n")
        done = (
            (this_re_filt.lower() == "quit") or
            (this_re_filt.lower() == "qt"))
        if done:
            break
        if this_re_filt == "":
            re_filt = None
        else:
            re_filt = this_re_filt.split(" SIM_SEP ")
        files = get_all_files_in_dir(
            start_dir, re_filter=re_filt, return_absolute=False,
            ext=ext, recursive=True)
        print("With {} the result is:".format(re_filt))
        for f in files:
            print(f)
    if re_filt == "":
        re_filt = None
    print("The final regex was: {}".format(re_filt))

    # Save the result
    if write:
        if write_loc is None:
            raise ValueError("Pass a location to write to when writing.")
        regex_filts = re_filt
        neuro_file = open(write_loc, "r")
        temp_file = open("temp.txt", "w")
        for line in neuro_file:

            if line.startswith("    regex_filter ="):
                line = "    regex_filter = " + str(regex_filts) + "\n"
                print("Set the regex filter to: " + line)
                temp_file.write(line)
            elif line.startswith("interactive ="):
                line = "interactive = False\n"
                temp_file.write(line)
            else:
                temp_file.write(line)

        neuro_file.close()
        temp_file.close()

        os.remove(write_loc)
        shutil.move("temp.txt", write_loc)

    return re_filt


def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    import more_itertools as mit
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0], group[0]
        else:
            yield group[0], group[-1]


def find_in(a, b):
    """Returns a boolean array of len(b) with True if any elements in a is found in b"""
    def t_val(x): return x in a
    truth = [t_val(x) for x in b]
    return truth


def ordered_set(arr):
    """Returns set in order it was first seen"""
    set_a = []
    for x in arr:
        if x not in set_a:
            set_a.append(x)
    return set_a


def get_dist(x, plot=False):
    """
    x: list
        Prints Min, Max, Mean and Values in x
    plot: boolean, False
        Shows plot of distribution

    """
    if type(x) is np.ndarray:
        print("Min: ", np.min(x), "\nMax: ", np.max(x),
              "\nMean: ", np.mean(x), "\nValues: ", x)

    elif type(x) is list:
        print("Min: ", min(x), "\nMax: ", max(x),
              "\nMean: ", mean(x), "\nValues: ", x)

    if plot:
        sns.distplot(x)
        plt.show()
    exit(-1)


def test_all_hier_clustering(data, verbose=False):
    """ For testing all perumatations for hierarchical clustering linkage """
    import pandas as pd
    import scipy.cluster.hierarchy as shc
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist

    link_methods = ['single', 'complete',
                    'average', 'weighted', 'centroid', 'median', 'ward']
    link_metric = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
                   'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

    coph = np.zeros((len(link_methods), len(link_metric)))
    for i, link in enumerate(link_methods):
        for j, metric in enumerate(link_metric):
            try:
                Z = shc.linkage(data, method=link, metric=metric)
                c, coph_dists = cophenet(Z, pdist(data))
                coph[i, j] = c
            except:
                coph[i, j] = np.nan

    max_coph = np.nanmax(coph)
    a, b = np.where(coph == max_coph)
    coph_df = pd.DataFrame(coph, index=link_methods,
                           columns=link_metric)
    if verbose:
        print(coph_df)
    print('\nMax Cophentic Correlation Coefficient: ',
          coph_df.index.values[a], coph_df.columns.values[b], max_coph)
    return coph_df


if __name__ == "__main__":
    """Main entry point."""
    PARSER = argparse.ArgumentParser(
        description='Process modifiable parameters from command line')
    PARSER.add_argument("--loc", type=str, help="h5 location")
    ARGS, UNPARSED = PARSER.parse_known_args()
    if ARGS.loc is None:
        print("Please enter a location through cmd, see -h for help")
        exit(-1)
    print_h5(ARGS.loc)
