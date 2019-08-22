"""
This module handles reading Session data from a MedPC file.

Written by Sean Martin and Gao Xiang Ham
"""

import os
import numpy as np
import h5py
from datetime import datetime
from bv_session_config import SessionInfo


class SessionExtractor:
    """
    Session Extractor pulls info from MEDPC files.

    This info is stored in a list of Session objects.
    """

    def __init__(self, file_location, verbose=False):
        """
        Initialise with an extraction location and then extract.

        Parameters
        ----------
        file_location : str
            Where the MEDPC file is to extract from.
        verbose : bool
            If this is true, print information during loading.

        """
        self.file_location = file_location
        self.sessions = []  # sessions extracted are stored in this list
        self.verbose = verbose

        self.extract_sessions()

    def get_sessions(self):
        """Return the list of Session objects that were extracted."""
        return self.sessions

    def extract_sessions(self):
        """
        Extract MPC sessions.

        Returns
        -------
        A List of sessions, one element for each session.

        """
        with open(self.file_location, 'r') as f:
            lines = f.read().splitlines()  # reads lines into list
            lines = np.array(
                list(filter(None, lines)))  # removes empty space

            s_starts = np.flatnonzero(
                np.core.defchararray.find(lines, "Start Date:") != -1)
            s_ends = np.zeros_like(s_starts)
            s_ends[:-1] = s_starts[1:]
            s_ends[-1] = lines.size
            print((s_ends))
            print((s_starts))

            for start, end in zip(s_starts, s_ends):
                s_data = np.array(lines[start:end])
                self.sessions.append(
                    Session(lines=s_data, verbose=self.verbose))
            return self.sessions

    def __repr__(self):
        """Session names that will be extracted."""
        out_str = self._get_session_names()
        return out_str

    def print_session_names(self):
        """Print the session names."""
        print(self._get_session_names())

    def _get_session_names(self):
        """Session names that will be extracted."""
        str_list = []
        str_list.append("Sessions in file:\n")
        for i, s in enumerate(self.sessions):
            str_list.append("{} -> {}\n".format(i, s.get_subject_type()))
        return "".join(str_list)

    def __len__(self):
        """Return Number of sessions in the extractor."""
        return len(self.sessions)

    def __getitem__(self, i):
        """Get the ith indexed session."""
        return self.sessions[i]


class Session:
    """The base class to hold MEDPC behaviour information."""

    def __init__(self, h5_file=None, lines=None, verbose=False):
        """
        Initialise the Session with lines from a MEDPC file.

        Then extract this info into native attributes.

        Parameters
        ----------
        lines : List of str
            The lines in the MedPC file for this session.
        verbose: bool - Default False
            Whether to print information while loading.

        """
        self.session_info = SessionInfo()
        self.metadata = {}
        self.info_arrays = {}
        self.verbose = verbose
        self.out_dir = None

        a = lines is None
        b = h5_file is None
        if (a and b or (not a and not b)):
            print("Error: Do not pass lines and h5_file to Session")
            return

        if lines is not None:
            self.lines = lines
            self._extract_metadata()
            self._extract_session_arrays()

        elif h5_file is not None:
            self.h5_file = h5_file
            self._extract_h5_info()

        else:
            print("Error: Unknown situation in Session init")
            exit(-1)

    def save_to_h5(self, out_dir, name=None):
        """Save information to a h5 file"""
        location = name
        if name == None:
            location = "{}_{}_{}_{}.h5".format(
                self.get_metadata("subject"),
                self.get_metadata("start_date").replace("/", "-"),
                self.get_metadata("start_time")[:-3].replace(":", "-"),
                self.get_metadata("name"))
        self.h5_file = os.path.join(out_dir, location)
        self._save_h5_info()

    def get_metadata(self, key=None):
        """
        Get the metadata for the Session.

        Parameters
        ----------
        key : str - Default None
            Possible Keys: "start_date", "end_date", "subject",
            "experiment", "group", "box", "start_time", "end_time", "name"

        Returns
        -------
        str : If key is a valid key.
        Dict : If key is None, all the metadata.

        """
        if key:
            return self.metadata[key]
        return self.metadata

    def get_subject_type(self):
        """Return the subject and session type as a string."""
        subject = self.get_metadata("subject")
        name = self.get_metadata("name")
        return 'Subject: {}, Trial Type {}'.format(subject, name)

    def get_arrays(self, key=None):
        """
        Return the info arrays in the session.

        Parameters
        ----------
        key : str - Default None
            The name of the info array to get.

        Returns
        -------
        np.ndarray: If key is a valid key.
        [] : If key is not a valid key
        Dict : If key is None, all the info arrays.

        """
        if key:
            return self.info_arrays.get(key, [])
        return self.info_arrays

    def get_lever_ts(self, include_un=True):
        """
        Get the timestamps of the lever presses.

        Parameters
        ----------
        include_un : bool - Default True
            Include lever presses that were unnecessary for the reward.

        Returns
        -------
        np.ndarray : A numpy array of sorted timestamps.

        """
        levers = [
            self.get_arrays("R"),
            self.get_arrays("L")]
        if include_un:
            levers.append(self.get_arrays("Un_R"))
            levers.append(self.get_arrays("Un_L"))
        return np.sort(np.concatenate(levers, axis=None))

    def get_err_lever_ts(self, include_un=True):
        """
        Get the timestamps of the error/opposite lever presses.

        Parameters
        ----------
        include_un : bool - Default True
            Include lever presses that were unnecessary for the reward.

        Returns
        -------
        np.ndarray : A numpy array of sorted timestamps.

        """
        levers = [
            self.get_arrays("FR_Err"),
            self.get_arrays("FI_Err")]
        if include_un:
            levers.append(self.get_arrays("Un_FR_Err"))
            levers.append(self.get_arrays("Un_FI_Err"))
        return np.sort(np.concatenate(levers, axis=None))

    def time_taken(self):
        """Calculate how long the Session took in mins."""
        start_time = self.get_metadata("start_time")[-8:].replace(' ', '0')
        end_time = self.get_metadata("end_time")[-8:].replace(' ', '0')
        fmt = '%H:%M:%S'
        tdelta = (
            datetime.strptime(end_time, fmt) -
            datetime.strptime(start_time, fmt))
        tdelta_mins = int(tdelta.total_seconds() / 60)
        return tdelta_mins

    def _save_h5_info(self):
        """Private function to save info to h5 file"""
        with h5py.File(self.h5_file, "w", libver="latest") as f:
            for key, val in self.get_metadata().items():
                print("{} {}".format(key, val))
                f.attrs[key] = val
            for key, val in f.attrs.items():
                print(key, val)
            for key, val in self.get_arrays().items():
                f.create_dataset(key, data=val, dtype=np.float32)

    def _extract_h5_info(self):
        """Private function to pull info from h5 file"""
        with h5py.File(self.h5_file, "r", libver="latest") as f:
            for key, val in f.attrs.items():
                self.metadata[key] = val
            for key in f.keys():
                self.info_arrays[key] = f[key][()]

    def _extract_metadata(self):
        """Private function to pull metadata out of lines."""
        for i, name in enumerate(self.session_info.get_metadata()):
            start = self.session_info.get_metadata_start(i)
            self.metadata[name] = self.lines[i][start:]

    def _extract_session_arrays(self):
        """Private function to pull session arrays out of lines."""
        print("Extracting arrays for {}".format(self))
        data_info = self.session_info.get_session_type_info(
            self.get_metadata("name"))

        if data_info is None:
            print("Unable to parse information")
            return

        if self.verbose:
            print("Parameters extracted:")
        for i, (start_char, end_char, parameter) in enumerate(data_info):
            c_data = self._extract_array(self.lines, start_char, end_char)
            self.info_arrays[parameter] = c_data
            if self.verbose:
                print(i, '-> {}: {}'.format(parameter, len(c_data)))

        return self.info_arrays

    @staticmethod
    def _extract_array(lines, start_char, end_char):
        """Private function to pull a single session from lines."""
        def parse_line(line, dtype=np.float32):
            return np.array(line.lstrip().split()[1:]).astype(dtype)

        start_index = np.flatnonzero(lines == start_char)
        stop_index = np.flatnonzero(lines == end_char)
        if end_char == 'END':
            # Last timepoint does not have a end_char
            stop_index = [lines.size]
        if start_index[0] + 1 == stop_index[0]:
            return np.array([])
        data_lines = lines[start_index[0] + 1:stop_index[0]]

        last_line = parse_line(data_lines[-1])
        arr = np.empty(
            5 * (len(data_lines) - 1) + len(last_line),
            dtype=np.float32)
        for i, line in enumerate(data_lines):
            numbers = parse_line(line)
            st = 5 * i
            arr[st:st + len(numbers)] = numbers
        return arr

    def __repr__(self):
        """
        Return string representation of the Session.

        Currently includes the date, subject and trial type.
        """
        return (
            self.get_metadata("start_date") + " " + self.get_subject_type())
