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

        self.session_info_dict = {}

        # Empty keys that should probably be updated at some stage
        self.session_info_dict['MSN: 2_MagazineHabituation_p'] = None
        self.session_info_dict['MSN: 3_LeverHabituation_p'] = None
        self.session_info_dict['MSN: DNMTS'] = None

        self.session_info_dict['MSN: 4_LeverTraining_p'] = (
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

        self.session_info_dict['MSN: 5a_FixedRatio_p'] = (
            np.array([
                ['A:', 'B:', 'Experiment Variables'],
                ['D:', 'E:', 'Reward'],
                ['E:', 'M:', 'Nosepoke'],
                ['M:', 'N:', 'FR Changes'],
                ['N:', 'O:', 'Un_R'],
                ['O:', 'R:', 'Un_Nosepoke'],
                ['R:', 'END', 'R']
            ]))

        self.session_info_dict['MSN: 5b_FixedInterval_p'] = (
            np.array([
                ['A:', 'B:', 'Experiment Variables'],
                ['D:', 'E:', 'Reward'],
                ['E:', 'N:', 'Nosepoke'],
                ['N:', 'O:', 'Un_L'],
                ['O:', 'R:', 'Un_Nosepoke'],
                ['R:', 'END', 'L']
            ]))

        self.session_info_dict['MSN: 6_RandomisedBlocks_p'] = (
            np.array([
                ['A:', 'B:', 'Experiment Variables'],
                ['D:', 'E:', 'Reward'],
                ['E:', 'L:', 'Nosepoke'],
                ['L:', 'M:', 'L'],
                ['M:', 'N:', 'Un_L'],
                ['N:', 'O:', 'Un_R'],
                ['O:', 'R:', 'Un_Nosepoke'],
                ['R:', 'Q:', 'R'],
                ['Q:', 'U:', 'Possible Trials'],
                ['U:', 'V:', 'Selected Trials'],
                ['V:', 'X:', 'Per Trial Pellets']
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
