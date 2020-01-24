import math
import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from neurochat.nc_lfp import NLfp
from neurochat.nc_utils import butter_filter


class LfpODict:
    """This class holds LFP files over multiple channels in a recording."""

    def __init__(
            self, filename, channels="all", filt_params=(False, None, None)):
        """
        Load the channels from filename.

        Args:
            filename (str): The basename of the file.
                filename+".eegX" are the final loaded files
            channels (str or List, optional): Defaults to [1, 2, ..., 32].
                The list of channels to load.
            filt_params (tuple(bool, float, float), optional):
                Defaults to (False, None, None)
                (Should filter, lower_bound, upper_bound)

        Returns:
            None

        """
        self._init_lfp(filename, channels)
        self.lfp_filt_odict = OrderedDict()
        if filt_params[0]:
            self.lfp_filt_odict = self.filter(*filt_params[1:])
        else:
            self.lfp_filt_odict = self.lfp_odict

    def get_signal(self, key=None):
        """Return signal at key, or full dict if key is None."""
        if type(key) is int:
            return list(self.lfp_odict.values())[key]
        if key is not None:
            return self.lfp_odict.get(key, None)
        return self.lfp_odict

    def get_filt_signal(self, key=None):
        """Return filtered signal at key, or full dict if key is None."""
        if type(key) is int:
            return list(self.lfp_filt_odict.values())[key]
        if key is not None:
            return self.lfp_filt_odict.get(key, None)
        return self.lfp_filt_odict

    def get_signals(self, keys):
        """Return a list of NLFP objects at the given keys."""
        out_list = []
        for key in keys:
            to_add = self.lfp_filt_odict.get(key, None)
            if to_add is None:
                print("Warning {} is not in LFP Dict".format(key))
            out_list.append(to_add)
        return out_list

    def filter(self, lower, upper):
        """
        Filter all the signals in the stored lfp dict.

        Args:
            lower, upper (float, float):
                lower and upper bands of the lfp signal in Hz

        Returns:
            OrderedDict of filtered singals.

        """
        if upper < lower:
            print("Must provide lower less than upper when filtering")
            exit(-1)
        lfp_filt_odict = OrderedDict()
        for key, lfp in self.lfp_odict.items():
            filt_lfp = deepcopy(lfp)
            fs = filt_lfp.get_sampling_rate()
            filtered_lfp_samples = butter_filter(
                filt_lfp.get_samples(), fs, 10,
                lower, upper, 'bandpass')
            filt_lfp._set_samples(filtered_lfp_samples)
            lfp_filt_odict[key] = filt_lfp
        return lfp_filt_odict

    def _init_lfp(self, filename, channels="all"):
        """
        Setup an orderedDict of lfp objects, one for each channel.

        Args:
            filename (str): The basename of the file.
                filename+".eegX" are the final loaded files
            channels (str or List, optional): Defaults to [1, 2, ..., 32].
                The list of channels to load.

        Returns:
            None

        """
        lfp_odict = OrderedDict()
        if channels == "all":
            channels = [i + 1 for i in range(32)]

        for i in channels:
            end = ".eeg"
            if i != 1 and i != "1":
                end = end + str(i)
            load_loc = filename + end
            lfp = NLfp(system="Axona")
            lfp.load(load_loc)
            if lfp.get_samples() is None:
                raise Exception("Failed to load lfp {}".format(load_loc))
            lfp_odict[str(i)] = lfp
        self.lfp_odict = lfp_odict

    def __len__(self):
        return len(self.get_signal())
