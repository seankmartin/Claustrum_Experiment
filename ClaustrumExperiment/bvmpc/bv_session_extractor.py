"""
This module handles reading Session data from a MedPC file.

Written by Sean Martin and Gao Xiang Ham
"""
import numpy as np
from bvmpc.bv_session import Session


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

            for start, end in zip(s_starts, s_ends):
                s_data = np.array(lines[start:end])
                self.sessions.append(
                    Session(
                        lines=s_data, verbose=self.verbose, file_origin=self.file_location))
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
