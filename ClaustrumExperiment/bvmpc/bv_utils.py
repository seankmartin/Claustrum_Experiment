"""This holds utility functions."""

import os
import re
import argparse
from datetime import timedelta
from collections.abc import Iterable

import h5py
import numpy as np


def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def make_dir_if_not_exists(location):
    """Make directory structure for given location."""
    os.makedirs(location, exist_ok=True)


def mycolors(subject):
    """Colour options for subject based on number"""
    i = int(subject)
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange',
                'tab:brown', 'deeppink', 'tab:olive', 'tab:pink',
                'steelblue', 'firebrick', 'mediumseagreen']
    return mycolors[i]


def split_list(list, chunk_limit):
    """Splits a list into small chunks based on chunk_limit"""
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
        if len(v.attrs) is not 0:
            print("{}{} has attrs {}".format(
                spaces, k, list(v.attrs.items())
            ))
        if hasattr(v, "items"):
            get_attrs(v, depth=depth + 1)


def print_h5(file_location):
    """Print a summary of a h5 file."""
    with h5py.File(file_location, 'r', libver='latest') as f:
        for key, val in f.attrs.items():
            print(key, val)
        print()
        walk_dict(f)
        print()


def has_ext(filename, ext):
    """
    Check if the filename ends in the extension

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
    if ext == None:
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
        search_res = re.search(re_filter, f)
        return search_res is not None

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
    Log an expection and additional info

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
    """Saves d to a file"""
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
