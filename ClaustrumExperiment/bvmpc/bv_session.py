"""
This module is a representation of a single Operant box Session.

Written by Sean Martin and Gao Xiang Ham
"""
import os
from datetime import datetime

import numpy as np
import h5py
import pandas as pd

from bvmpc.bv_session_info import SessionInfo
import bvmpc.bv_analyse as bv_an


class Session:
    """The base class to hold MEDPC behaviour information."""

    def __init__(
            self, h5_file=None, lines=None, neo_file=None,
            neo_backend="nix", verbose=False, file_origin=None):
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
        self.file_origin = file_origin
        self.out_dir = None
        self.trial_df = None
        self.trial_df_norm = None

        a = 0 if lines is None else 1
        b = 0 if h5_file is None else 1
        c = 0 if neo_file is None else 1
        if (a + b + c != 1):
            print("Error: Session takes one of h5_file, lines and neo_file")
            return

        if lines is not None:
            self.lines = lines
            self._extract_metadata()
            self._extract_session_arrays()

        elif h5_file is not None:
            self.file_origin = h5_file
            self.h5_file = h5_file
            self._extract_h5_info()

        elif neo_file is not None:
            self.file_origin = neo_file
            self.neo_file = neo_file
            self.neo_backend = neo_backend
            self._extract_neo_info()

        else:
            print("Error: Unknown situation in Session init")
            exit(-1)

    # convert to trial based dataframe
    def init_trial_dataframe(self):
        self._init_trial_df()

    def save_to_h5(self, out_dir, name=None):
        """Save information to a h5 file"""
        location = name
        if name == None:
            location = self._get_hdf5_name()
        self.h5_file = os.path.join(out_dir, location)
        self._save_h5_info()

    def save_to_neo(
            self, out_dir, name=None, neo_backend="nix", remove_existing=False):
        location = name
        self.neo_backend = neo_backend
        if name == None:
            ext = self._get_neo_io(get_ext=True)
            location = self._get_hdf5_name(ext=ext)
        self.neo_file = os.path.join(out_dir, location)
        self._save_neo_info(remove_existing)

    def get_trial_df(self):
        """Get dataframe split into trials without normalisation (time)."""
        if self.trial_df is None:
            self.init_trial_dataframe()
        return self.trial_df

    def get_trial_df_norm(self):
        """Get dataframe split into trials with normalisation (time)."""
        if self.trial_df_norm is None:
            self.init_trial_dataframe()
        return self.trial_df_norm

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
            return self.metadata.get(key, None)
        return self.metadata

    def get_subject_type(self):
        """Return the subject and session type as a string."""
        subject = self.get_metadata("subject")
        name = self.get_metadata("name")
        return 'Subject: {}, Trial Type {}'.format(subject, name)

    def get_ratio(self):
        """Return the fixed ratio as an int."""
        ratio = self.get_metadata("fixed_ratio")
        if ratio is not None:
            return int(ratio)
        return None

    def get_interval(self):
        """Return the fixed ratio as an int."""
        interval = self.get_metadata("fixed_interval (secs)")
        if interval is not None:
            return int(interval)
        return None

    def get_stage(self):
        """Obtain the stage number (without _)"""
        session_type = self.get_metadata('name')
        stage = session_type[:2].replace('_', '')
        return stage

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

    def get_rw_ts(self):
        """
        Get the timestamps of rewards. 

        Corrected for session switching/ending without reward collection.

        Returns
        -------
        np.ndarray : A numpy array of sorted timestamps.

        """
        session_type = self.get_metadata('name')
        stage = session_type[:2].replace('_', '')
        pell_ts = self.get_arrays("Reward")
        reward_times = self.get_arrays("Nosepoke")

        # Check for double pellets
        dpell_bool = np.diff(pell_ts) < 0.5
        # Provides index of double pell in pell_ts
        dpell_idx = np.nonzero(dpell_bool)[0] + 1
        pell_ts_exdouble = np.delete(pell_ts, dpell_idx)

        reward_times = self.get_arrays("Nosepoke")
        trial_len = self.get_metadata("trial_length (mins)") * 60
        if stage == '7' or stage == '6':
            trial_len += 5
            repeated_trial_len = (trial_len) * 6

        # if last reward time < last pellet dispensed, assume animal picked reward at end of session.
        if reward_times[-1] < pell_ts[-1]:
            if stage == '7' or stage == '6':
                reward_times = np.append(reward_times, repeated_trial_len)
            else:
                reward_times = np.append(reward_times, trial_len)

        if stage == '7' or stage == '6':
            # Check if trial switched before reward collection -> Adds collection as switch time
            blocks = np.arange(trial_len, repeated_trial_len, trial_len)
            split_pell_ts = np.split(
                pell_ts_exdouble, np.searchsorted(pell_ts_exdouble, blocks))
            split_reward_ts = np.split(
                reward_times, np.searchsorted(reward_times, blocks))

            for i, (pell, reward) in enumerate(zip(split_pell_ts, split_reward_ts[:-1])):
                if len(pell) > len(reward):
                    reward_times = np.insert(reward_times, np.searchsorted(
                        reward_times, blocks[i]), blocks[i])

        return np.sort(reward_times, axis=None)

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

    def _save_neo_info(self, remove_existing):
        """Private function to save info to neo file"""
        if os.path.isfile(self.neo_file):
            if remove_existing:
                os.remove(self.neo_file)
            else:
                print("Skipping {} as it already exists".format(
                    self.neo_file) +
                    " - set remove_existing to True to remove")
                return

        from neo.core import Block, Segment, Event
        from quantities import s

        anots = self.get_metadata()
        anots["nix_name"] = "Block_Main"
        anots["protocol"] = anots.pop("name")

        blk = Block(name="Block_Main", **anots)

        seg = Segment(name="Segment_Main", nix_name="Segment_Main")
        blk.segments.append(seg)
        # Could consider splitting by trials using index in seg
        for key, val in self.get_arrays().items():
            e = Event(
                times=val * s, labels=None,
                name=key, nix_name="Event_" + key)
            seg.events.append(e)
        nio = self._get_neo_io()
        nio.write_block(blk)
        nio.close()

    def _extract_neo_info(self):
        """Private function to extract info from neo file"""
        nio = self._get_neo_io()
        block = nio.read()[0]
        nio.close()
        annotations = block.annotations
        for key, val in annotations.items():
            if key == "protocol":
                key = "name"
            self.metadata[key] = val
        for event in block.segments[0].events:
            key = event.name
            self.info_arrays[key] = (
                event.times.rescale('s').magnitude)

    def _get_neo_io(self, get_ext=False):
        backend = self.neo_backend
        if backend == "nix":
            if get_ext:
                return "nix"
            from neo.io import NixIO
            nio = NixIO(filename=self.neo_file)
        elif backend == "nsdf":
            if get_ext:
                return "h5"
            from neo.io import NSDFIO
            nio = NSDFIO(filename=self.neo_file)
        elif backend == "matlab":
            if get_ext:
                return "mat"
            from neo.io import NeoMatlabIO
            nio = NeoMatlabIO(filename=self.neo_file)
        else:
            print(
                "Backend {} not recognised, defaulting to nix".format(
                    backend))
            from neo.io import NixIO
            if get_ext:
                return "nix"
            nio = NixIO(filename=self.neo_file)
        return nio

    def _save_h5_info(self):
        """Private function to save info to h5 file"""
        with h5py.File(self.h5_file, "w", libver="latest") as f:
            for key, val in self.get_metadata().items():
                f.attrs[key] = val
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
        data_info = self.session_info.get_session_type_info(
            self.get_metadata("name"))

        if data_info is None:
            print("Not parsing {}".format(self))
            return

        print("Parsing {}".format(self))
        if self.verbose:
            print("Parameters extracted:")
        for i, (start_char, end_char, parameter) in enumerate(data_info):
            c_data = self._extract_array(self.lines, start_char, end_char)
            if parameter == "Experiment Variables":
                mapping = self.session_info.get_session_variable_list(
                    self.get_metadata("name"))
                for m, v in zip(mapping, c_data):
                    if m.endswith("(ticks)"):
                        m = m[:-7] + "(secs)"
                        v = v / 100
                    self.metadata[m] = v
            self.info_arrays[parameter] = c_data
            if self.verbose:
                print(i, '-> {}: {}'.format(parameter, len(c_data)))

        return self.info_arrays

    def _init_trial_df(self):
        session_type = self.get_metadata('name')
        stage = session_type[:2].replace('_', '')  # Obtain stage number w/o _
        timestamps = self.get_arrays()
        pell_ts = timestamps["Reward"]

        dpell_bool = np.diff(pell_ts) < 0.5
        # Provides index of double pell in pell_ts
        dpell_idx = np.nonzero(dpell_bool)[0] + 1
        dpell = pell_ts[dpell_idx]

        # pell drop ts excluding double ts
        pell_ts_exdouble = np.delete(pell_ts, dpell_idx)
        reward_times = self.get_rw_ts()

        # Assign schedule type to trials
        schedule_type = []
        if stage == '7' or stage == '6':
            norm_r_ts, _, _, _, _ = bv_an.split_sess(
                self, norm=False, plot_all=True)
            sch_type = self.get_arrays('Trial Type')

            for i, block in enumerate(norm_r_ts):
                if sch_type[i] == 1:
                    b_type = 'FR'
                elif sch_type[i] == 0:
                    b_type = 'FI'
                for l, _ in enumerate(block):
                    schedule_type.append(b_type)
        else:
            if stage == '4':
                b_type = 'CR'
            elif stage == '5a':
                b_type = 'FR'
            elif stage == '5b':
                b_type = 'FI'
            else:
                b_type = 'NA'
            for i in reward_times:
                schedule_type.append(b_type)

        # Rearrange timestamps based on trial per row
        lever_ts = self.get_lever_ts(True)
        err_ts = self.get_err_lever_ts(True)
        trial_lever_ts = np.split(
            lever_ts, (np.searchsorted(lever_ts, reward_times)[:-1]))
        trial_err_ts = np.split(
            err_ts, (np.searchsorted(err_ts, reward_times)[:-1]))
        trial_dr_ts = np.split(
            dpell, (np.searchsorted(dpell, reward_times)[:-1]))

        # Initialize array for lever timestamps
        # Max lever press per trial
        trials_max_l = len(max(trial_lever_ts, key=len))
        lever_arr = np.empty((len(reward_times), trials_max_l,))
        lever_arr.fill(np.nan)
        trials_max_err = len(max(trial_err_ts, key=len)
                             )  # Max err press per trial
        err_arr = np.empty((len(reward_times), trials_max_err,))
        err_arr.fill(np.nan)

        # Arrays used for normalization of timestamps to trials
        from copy import deepcopy
        trial_norm = np.insert(reward_times, 0, 0)
        norm_lever = deepcopy(trial_lever_ts)
        norm_err = deepcopy(trial_err_ts)
        norm_dr = deepcopy(trial_dr_ts)
        norm_rw = deepcopy(reward_times)
        norm_pell = deepcopy(pell_ts_exdouble)

        # Normalize timestamps based on start of trial
        for i, _ in enumerate(norm_rw):
            norm_lever[i] -= trial_norm[i]
            norm_err[i] -= trial_norm[i]
            norm_dr[i] -= trial_norm[i]
            norm_pell[i] -= trial_norm[i]
            norm_rw[i] -= trial_norm[i]

        # # 2D array of lever timestamps (Incomplete)
        # for i, (l, err) in enumerate(zip(trial_lever_ts, trial_err_ts)):
        #     l_end = len(l)
        #     lever_arr[i,:l_end] = l[:]
        #     err_end = len(err)
        #     err_arr[i,:err_end] = err[:]

        # Timestamps kept as original starting from session start
        session_dict = {
            'Reward_ts': reward_times,
            'Pellet_ts': pell_ts_exdouble,
            'D_Pellet_ts': trial_dr_ts,
            'Schedule': schedule_type,
            'Levers_ts': trial_lever_ts,
            'Err_ts': trial_err_ts
        }

        # Timestamps normalised to each trial start
        trial_dict = {
            'Reward_ts': norm_rw,
            'Pellet_ts': norm_pell,
            'D_Pellet_ts': norm_dr,
            'Schedule': schedule_type,
            'Levers_ts': norm_lever,
            'Err_ts': norm_err
        }

        for key, val in trial_dict.items():
            print(key, ':', len(val))

        self.trial_df = pd.DataFrame(session_dict)
        self.trial_df_norm = pd.DataFrame(trial_dict)

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

    def _get_hdf5_name(self, ext="h5"):
        return "{:03d}_{}_{}_{}.{}".format(
            int(self.get_metadata("subject")),
            self.get_metadata("start_date").replace("/", "-"),
            self.get_metadata("start_time")[:-3].replace(":", "-"),
            self.get_metadata("name"), ext)

    def __repr__(self):
        """
        Return string representation of the Session.

        Currently includes the date, subject and trial type.
        """
        return (
            self.get_metadata("start_date") + " " + self.get_subject_type())
