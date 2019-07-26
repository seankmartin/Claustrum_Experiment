# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:19:11 2019

Written by Sean Martin and Gao Xiang Ham
"""

import numpy as np
# import h5py
from datetime import datetime
from session_config import SessionInfo


class SessionExtractor:
    def __init__(self, file_location, verbose=False):
        self.file_location = file_location
        self.sessions = []
        self.verbose = verbose

        self.extract_sessions()

    def get_sessions(self):
        return self.sessions

    def extract_sessions(self):
        """
        Extract MPC sessions in file_location.

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

            for start, end in zip(s_starts, s_ends):
                s_data = np.array(lines[start:end])
                self.sessions.append(Session(s_data, self.verbose))
            return self.sessions

    def __repr__(self):
        out_str = self._get_session_names()
        return out_str

    def print_session_names(self):
        print(self._get_session_names())

    def _get_session_names(self):
        str_list = []
        str_list.append("Sessions in file:\n")
        for i, s in enumerate(self.sessions):
            str_list.append("{} -> {}\n".format(i, s.get_name()))
        return "".join(str_list)

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, i):
        return self.sessions[i]


class Session:
    def __init__(self, lines, verbose=False):
        self.lines = lines
        self.session_info = SessionInfo()
        self.metadata = {}
        self.timestamps = {}
        self.verbose = verbose

        self._extract_metadata()
        self._extract_session_arrays()

    def get_metadata(self, name=None):
        if name:
            return self.metadata[name]
        return self.metadata

    def get_name(self):
        subject = self.lines[
            self.session_info.get_metadata("subject")]
        name = self.lines[
            self.session_info.get_metadata("name")]
        return '{}, {}'.format(subject, name)

    def get_timestamps(self, name=None):
        if name:
            return self.timestamps.get(name, [])
        return self.timestamps

    def get_lever_ts(self, include_un=True):
        levers = [
            self.get_timestamps("R"),
            self.get_timestamps("L")]
        if include_un:
            levers.append(self.get_timestamps("Un_R"))
            levers.append(self.get_timestamps("Un_L"))
        return np.sort(np.concatenate(levers, axis=None))

    def get_lines(self):
        return self.lines

    def _extract_metadata(self):
        for i, name in enumerate(self.session_info.get_metadata()):
            self.metadata[name] = self.lines[i]

    def _extract_session_arrays(self):
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
            self.timestamps[parameter] = c_data
            if self.verbose:
                print(i, '-> {}: {}'.format(parameter, len(c_data)))

        return self.timestamps

    @staticmethod
    def _extract_array(lines, start_char, end_char):

        def parse_line(line, dtype=np.float32):
            return np.array(line.lstrip().split()[1:]).astype(dtype)

        start_index = np.flatnonzero(lines == start_char)
        stop_index = np.flatnonzero(lines == end_char)
        if end_char == 'END':
            # Last timepoint does not have a end_char
            stop_index = [lines.size]

        data_lines = lines[start_index[0] + 1:stop_index[0]]
        if not data_lines.size:
            return np.array([])

        last_line = parse_line(data_lines[-1])
        arr = np.empty(
            5 * (len(data_lines) - 1) + len(last_line),
            dtype=np.float32)
        for i, line in enumerate(data_lines):
            numbers = parse_line(line)
            st = 5 * i
            arr[st:st + len(numbers)] = numbers
        return arr

    def time_taken(self):
        start_time = self.get_metadata("start_time")[-8:].replace(' ', '0')
        end_time = self.get_metadata("end_time")[-8:].replace(' ', '0')
        fmt = '%H:%M:%S'
        tdelta = (
            datetime.strptime(end_time, fmt) -
            datetime.strptime(start_time, fmt))
        tdelta_mins = int(tdelta.total_seconds() / 60)
        return tdelta_mins

    def __repr__(self):
        return (
            self.get_metadata("start_date") + " " + self.get_name())
