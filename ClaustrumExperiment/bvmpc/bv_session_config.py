"""Session information is held here."""

import numpy as np


class SessionInfo:
    """Holds mappings of information in MEDPC sessions."""

    def __init__(self):
        """Init and setup the session information map."""
        self.metadata_info = [
            "start_date", "end_date", "subject",
            "experiment", "group",
            "box", "start_time", "end_time", "name"]

        self.metadata_start_idx = [
            len("Start Date: "),
            len("End Date: "),
            len("Subject: "),
            len("Experiment: "),
            len("Group: "),
            len("Box: "),
            len("Start Time: "),
            len("End Time: "),
            len("MSN: ")
        ]

        self.session_info_dict = {}

        # Empty keys that should probably be updated at some stage
        self.session_info_dict['2_MagazineHabituation_p'] = (
            np.array([
                ['A:', 'B:', 'Experiment Variables'],
                ['D:', 'E:', 'Reward'],
                ['E:', 'END', 'Nosepoke']
            ]))
        self.session_info_dict['3_LeverHabituation_p'] = (
            np.array([
                ['A:', 'B:', 'Experiment Variables'],
                ['D:', 'E:', 'Reward'],
                ['E:', 'L:', 'Nosepoke'],
                ['L:', 'M:', 'L'],
                ['M:', 'N:', 'Un_L'],
                ['N:', 'O:', 'Un_R'],
                ['O:', 'R:', 'Un_Nosepoke'],
                ['R:', 'END', 'R']
            ]))
        # self.session_info_dict['DNMTS'] = None

        self.session_info_dict['4_LeverTraining_p'] = (
            np.array([
                ['A:', 'B:', 'Experiment Variables'],
                ['D:', 'E:', 'Reward'],
                ['E:', 'L:', 'Nosepoke'],
                ['L:', 'M:', 'L'],
                ['M:', 'N:', 'Un_L'],
                ['N:', 'O:', 'Un_R'],
                ['O:', 'R:', 'Un_Nosepoke'],
                ['R:', 'END', 'R']
            ]))

        self.session_info_dict['5a_FixedRatio_p'] = (
            np.array([
                ['A:', 'B:', 'Experiment Variables'],
                ['D:', 'E:', 'Reward'],
                ['E:', 'M:', 'Nosepoke'],
                ['M:', 'N:', 'FR Changes'],
                ['N:', 'O:', 'Un_R'],
                ['O:', 'R:', 'Un_Nosepoke'],
                ['R:', 'END', 'R']
            ]))

        self.session_info_dict['5b_FixedInterval_p'] = (
            np.array([
                ['A:', 'B:', 'Experiment Variables'],
                ['D:', 'E:', 'Reward'],
                ['E:', 'N:', 'Nosepoke'],
                ['N:', 'O:', 'Un_L'],
                ['O:', 'R:', 'Un_Nosepoke'],
                ['R:', 'END', 'L']
            ]))

        self.session_info_dict['6_RandomisedBlocks_p'] = (
            np.array([
                ['A:', 'B:', 'Experiment Variables'],
                ['D:', 'E:', 'Reward'],
                ['E:', 'L:', 'Nosepoke'],
                ['L:', 'M:', 'L'],
                ['M:', 'N:', 'Un_L'],
                ['N:', 'O:', 'Un_R'],
                ['O:', 'R:', 'Un_Nosepoke'],
                ['R:', 'U:', 'R'],
                ['U:', 'V:', 'Trial Type'],  # 1 is FR, 0 is FI
                # ['V:', 'END', 'Per Trial Pellets']
            ]))

        self.session_info_dict['7_RandomisedBlocksExtended_p'] = (
            np.array([
                ['A:', 'B:', 'Experiment Variables'],
                ['D:', 'E:', 'Reward'],
                ['E:', 'F:', 'Nosepoke'],
                # L during FR Trial; L when R available
                ['F:', 'G:', 'FR_Err'],
                # R during FI Trial; R when L available
                ['G:', 'H:', 'FI_Err'],
                # L during FR Trial_Un; L before reward collection
                ['H:', 'I:', 'Un_FR_Err'],
                # R during FI Trial_Un; R during waiting time
                ['I:', 'L:', 'Un_FI_Err'],
                ['L:', 'M:', 'L'],
                ['M:', 'N:', 'Un_L'],
                ['N:', 'O:', 'Un_R'],
                ['O:', 'R:', 'Un_Nosepoke'],
                ['R:', 'U:', 'R'],
                ['U:', 'V:', 'Trial Type'],
                # ['V:', 'END', 'Per Trial Pellets']
            ]))

    def get_session_type_info(self, key=None):
        """
        Return the mapping for a given session type.

        If key is passed as None, return a Dict of all possible maps.

        Parameters
        ----------
        key : str
            Which session type info to get, default None.

        Returns
        -------
        np.ndarray : an array of mappings.
        Dict: if key is None, all the mappings.

        """
        if key:
            return self.session_info_dict.get(key, None)
        return self.session_info_dict

    def get_metadata(self, key=None):
        """
        Return the metadata index of the session.

        If key is passed as None, return a list of metadata keys.

        Parameters
        ----------
        key : str - Default None
            Which metadata item to get.

        Returns
        -------
        int : the index for key.
        List : if key is None, all the keys.

        """
        if key:
            return self.metadata_info.index(key)
        return self.metadata_info

    def get_metadata_start(self, idx=None):
        """
        Return the start index of the metadata information.

        Parameters
        ----------
        idx : int
            The index for which to return the start point.

        Returns
        -------
        int : The start index for the information
        List : If idx is None, all the start indices.
        """
        if idx is not None:
            return self.metadata_start_idx[idx]
        return self.metadata_start_idx
