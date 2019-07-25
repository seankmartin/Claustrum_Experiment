"""
This holds utility functions.
"""

import os


def make_dir_if_not_exists(location):
    """Makes directory structure for given location"""
    os.makedirs(location, exist_ok=True)
