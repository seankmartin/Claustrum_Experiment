"""This holds utility functions."""

import h5py
import os
import argparse


def make_dir_if_not_exists(location):
    """Make directory structure for given location."""
    os.makedirs(location, exist_ok=True)


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
