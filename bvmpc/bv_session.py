"""
This module is a representation of a single Operant box Session.

Written by Sean Martin and Gao Xiang Ham
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bvmpc.bv_session_info import SessionInfo
from bvmpc.bv_axona import AxonaInput, AxonaSet
from bvmpc.bv_array_methods import split_into_blocks
from bvmpc.bv_array_methods import split_array
from bvmpc.bv_array_methods import split_array_with_another
from bvmpc.bv_array_methods import split_array_in_between_two


class Session:
    """
    The base class to hold MEDPC behaviour information.

    An Operant Chamber Session holds data and methods related to this.
    Provide ONE of h5_file, lines, neo_file, and axona_file to initialise.

    Parameters
    ----------
    h5_file: str - Default None
        h5 file location to load the session from
    lines: List of str - Default None
        The lines in the MedPC file for this session.
    neo_file: str - Default None
        The location of a neo file to load from
    axona_file: str - Default None
        The location of a .inp file to load from
    s_type: str - Default None
        The type of session when using axona_file
        For now, leaving as 6 will work for everything.
    neo_backend: str - Default "nix
        If using neo_file, what backend to use
    verbose: bool - Default False
        Whether to print information while loading.
    file_origin: str - Default None
        Where the file originated from,
        mostly only useful if directly passing lines

    """

    def __init__(
            self, h5_file=None, lines=None, neo_file=None,
            axona_file=None, s_type="6",
            neo_backend="nix", verbose=False, file_origin=None):
        """See help(Session) for more info."""
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
        d = 0 if axona_file is None else 1
        if (a + b + c + d != 1):
            print(
                "Error: Session takes one of" +
                "h5_file, lines, neo_file, and axona_file")
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

        elif axona_file is not None:
            if s_type is None:
                print("Error: Please provide a session type (6 or 7)")
                exit(-1)
            self.s_type = s_type
            self.file_origin = axona_file
            self.axona_file = axona_file
            self._extract_axona_info()

        else:
            print("Error: Unknown situation in Session init")
            exit(-1)

    def init_trial_dataframe(self):
        """Initialise the trial based Pandas dataframe for this session."""
        self._init_trial_df()

    def split_pell_ts(self):
        """
        Return a tuple of arrays splitting up the rewards.

        (Pellets without doubles, time of doubles)

        """
        pell_ts = self.get_arrays("Reward")

        dpell_bool = np.diff(pell_ts) < 0.8
        # Provides index of double pell in pell_ts
        dpell_idx = np.nonzero(dpell_bool)[0] + 1
        dpell = pell_ts[dpell_idx]

        # pell drop ts excluding double ts
        pell_ts_exdouble = np.delete(pell_ts, dpell_idx)

        return pell_ts_exdouble, dpell


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

    def split_sess(
            self, norm=True, blocks=None, plot_error=False, plot_all=False):
        """
        Split a session up into multiple blocks.

        blocks: defines timepoints to split.

        returns 5 outputs:
            1) timestamps split into rows depending on blocks input
                    -> norm_reward_ts, norm_lever_ts, norm_err_ts, norm_double_r_ts
            2) print to include in title and file name. Mainly for stage 7.
        """
        session_type = self.get_metadata('name')
        stage = session_type[:2].replace('_', '')
        reward_times = self.get_rw_ts()
        timestamps = self.get_arrays()
        lever_ts = self.get_lever_ts()
        pell_ts = timestamps["Reward"]
        pell_double = np.nonzero(np.diff(pell_ts) < 0.5)
        # returns reward ts after d_pell
        reward_double = reward_times[
            np.searchsorted(
                reward_times, pell_ts[pell_double], side='right')]
        err_lever_ts = []

        if blocks is not None:
            pass
        else:
            blocks = np.arange(5, 1830, 305)  # Default split into schedules

        incl = ""  # Initialize print for type of extracted lever_ts
        if stage == '7' and plot_error:  # plots errors only
            incl = '_Errors_Only'
            lever_ts = self.get_err_lever_ts()
        elif stage == '7' and plot_all:  # plots all responses incl. errors
            incl = '_All'
            err_lever_ts = self.get_err_lever_ts()
            lever_ts = np.sort(np.concatenate((
                lever_ts, err_lever_ts), axis=None))
        elif stage == '7':  # plots all responses exclu. errors
            incl = '_Correct Only'

        split_lever_ts = np.split(lever_ts,
                                np.searchsorted(lever_ts, blocks))
        split_reward_ts = np.split(reward_times,
                                np.searchsorted(reward_times, blocks))
        split_double_r_ts = np.split(reward_double,
                                    np.searchsorted(reward_double, blocks))
        split_err_ts = np.split(err_lever_ts,
                                np.searchsorted(err_lever_ts, blocks))
        norm_reward_ts = []
        norm_lever_ts = []
        norm_err_ts = []
        norm_double_r_ts = []
        if norm:
            for i, l in enumerate(split_lever_ts[1:]):
                norm_lever_ts.append(np.append([0], l - blocks[i], axis=0))
                norm_reward_ts.append(split_reward_ts[i + 1] - blocks[i])
                norm_double_r_ts.append(split_double_r_ts[i + 1] - blocks[i])
                if stage == '7' and plot_all:  # plots all responses incl. errors
                    norm_err_ts.append(split_err_ts[i + 1] - blocks[i])
        else:
            norm_lever_ts = split_lever_ts[1:]
            norm_reward_ts = split_reward_ts[1:]
            norm_err_ts = split_err_ts[1:]
            norm_double_r_ts = split_double_r_ts[1:]
        return (
            norm_reward_ts, norm_lever_ts, norm_err_ts, norm_double_r_ts, incl)

    def extract_features(self):
        if self.trial_df_norm is None:
            self.init_trial_dataframe()
        features = np.zeros(shape=(len(self.trial_df_norm.index), 12))
        for row in self.trial_df_norm.itertuples():
            index = row.Index
            features[index, 0] = row.Reward_ts
            features[index, 1] = row.Pellet_ts
            double_pellet_time = row.D_Pellet_ts
            if len(double_pellet_time) == 0:
                d_pell_feature = 0
            else:
                d_pell_feature = double_pellet_time
            features[index, 2] = d_pell_feature
            lever_hist = np.histogram(
                row.Levers_ts, bins=8, density=True)[0]
            features[index, 3:11] = lever_hist
            # err_hist = np.histogram(
                # row.Err_ts, bins=5, density=False)[0]
            features[index, 11] = row.Err_ts.size
        return features

    def perform_pca(self, n_components=3, should_scale=True):
        """
        Perform PCA on per trial features 
        
        Parameters
        ------
        n_components : int or float 
            the number of PCA components to compute
            if float, uses enough components to reach that much variance
        should_scale - whether to scale the data to unit variance

        Returns
        -------
        tuple : (ndarray, ndarray, PCA)
            (features 2d array, PCA of features, PCA object)
        """
        data = self.extract_features()
        scaler = StandardScaler()
        pca = PCA(n_components=n_components)

        # Standardise the data to improve PCA performance
        if should_scale:
            std_data = scaler.fit_transform(data)
            after_pca = pca.fit_transform(std_data)
        else:
            after_pca = pca.fit_transform(data)

        print(
            "PCA fraction of explained variance", pca.explained_variance_ratio_)
        return data, after_pca, pca

    def save_to_h5(self, out_dir, name=None):
        """Save information to a h5 file."""
        location = name
        if name is None:
            location = self._get_hdf5_name()
        self.h5_file = os.path.join(out_dir, location)
        self._save_h5_info()

    def save_to_neo(
            self, out_dir, name=None,
            neo_backend="nix", remove_existing=False):
        """Save information to a neo file."""
        location = name
        self.neo_backend = neo_backend
        if name is None:
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
        """Obtain the stage number (without _)."""
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

        pell_ts_exdouble = self.split_pell_ts()[0]

        reward_times = self.get_arrays("Nosepoke")
        trial_len = self.get_metadata("trial_length (mins)") * 60
        if stage == '7' or stage == '6':
            trial_len += 5
            repeated_trial_len = (trial_len) * 6

        if stage == '7' or stage == '6':
            # Check if trial switched before reward collection
            # -> Adds collection as switch time
            split_pell_ts = split_into_blocks(
                pell_ts_exdouble, trial_len, 6)
            split_reward_ts = split_into_blocks(
                reward_times, trial_len, 6)

            for i, (pell, reward) in enumerate(
                    zip(split_pell_ts, split_reward_ts)):
                if len(pell) > len(reward):
                    block_len = trial_len * (i+1)
                    reward_times = np.insert(
                        reward_times, np.searchsorted(
                            reward_times, block_len), block_len)

        # if last reward time < last pellet dispensed,
        # assume animal picked reward at end of session.
        # Checks if last block contains responses first
        if reward_times[-1] < pell_ts[-1]:
            if stage == '7' or stage == '6':
                reward_times = np.append(reward_times, repeated_trial_len)
            else:
                reward_times = np.append(reward_times, trial_len)

        return np.sort(reward_times, axis=None)

    def _save_neo_info(self, remove_existing):
        """Private function to save info to neo file."""
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
        """Private function to extract info from neo file."""
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
        """Private function to save info to h5 file."""
        import h5py
        with h5py.File(self.h5_file, "w", libver="latest") as f:
            for key, val in self.get_metadata().items():
                f.attrs[key] = val
            for key, val in self.get_arrays().items():
                f.create_dataset(key, data=val, dtype=np.float32)

    def _extract_h5_info(self):
        """Private function to pull info from h5 file."""
        import h5py
        with h5py.File(self.h5_file, "r", libver="latest") as f:
            for key, val in f.attrs.items():
                self.metadata[key] = val
            for key in f.keys():
                self.info_arrays[key] = f[key][()]

    def _extract_axona_info(self):
        """Extract from .inp .set and .log files for session conversion."""
        # Extract metadata from set file
        self.axona_setfile = self.axona_file[:-3] + "set"
        self.axona_set = AxonaSet(self.axona_setfile)
        for k, v in self.session_info.get_axona_metadata_map().items():
            self.metadata[k] = self.axona_set.get_val(v)

        # Numerical metadata extract from log file **No log file in some cases. Static method for now**
        self.metadata["fixed_ratio"] = 6
        self.metadata["double_reward_window (secs)"] = 10
        self.metadata["fixed_interval (secs)"] = 30
        self.metadata["num_trials"] = 6
        self.metadata["trial_length (mins)"] = 5
        self.metadata['subject'] = os.path.basename(
            self.axona_file).split("_")[0]

        # AND "trial_length (mins)", "fixed_ratio"
        # "num_trials" should be added too

        # Extract the timestamp arrays from the axona data .inp file
        self.axona_info = AxonaInput(self.axona_file)
        for k, v in self.session_info.get_input_channel(self.s_type).items():
            self.info_arrays[k] = self.axona_info.get_times(
                "I", v[0], v[1])
        for k, v in self.session_info.get_output_channel(self.s_type).items():
            self.info_arrays[k] = self.axona_info.get_times(
                "O", v[0], v[1])
        self._convert_axona_info()

    def _convert_axona_info(self):
        """Perform postprocessing on the extracted axona timestamp arrays."""
        left_presses = self.info_arrays.get("left_lever", [])
        right_presses = self.info_arrays.get("right_lever", [])
        nosepokes = self.info_arrays.get("all_nosepokes", [])

        # Extract nosepokes as necessary and unecessary
        pell_ts_exdouble, _ = self.split_pell_ts()
        good_nosepokes, un_nosepokes = split_array_with_another(
            nosepokes, pell_ts_exdouble)

        split_nosepokes = split_into_blocks(
            good_nosepokes, 305, 6)
        split_pellets = split_into_blocks(
            pell_ts_exdouble, 305, 6)
        for i, (b1, b2) in enumerate(zip(split_nosepokes, split_pellets)):
            if len(b2) > len(b1):
                print(good_nosepokes)
                last_trial_idx = np.nonzero(
                    np.abs(good_nosepokes - b1[-1]) <= 0.00001)[0]
                good_nosepokes[last_trial_idx+1] = 305 * (i+1)
                if i < 5:
                    split_nosepokes[i+1] = split_nosepokes[i+1][1:]
                print(good_nosepokes)

        split_nosepokes = split_into_blocks(
            good_nosepokes, 305, 6)
        split_pellets = split_into_blocks(
            pell_ts_exdouble, 305, 6)
        for i, (b1, b2) in enumerate(zip(split_nosepokes, split_pellets)):
            if b2.shape != b1.shape:
                print("Error, nosepokes in blocks don't match pellets")
                print("{} nosepokes, {} pellets, in block {}".format(
                    len(b1), len(b2), i + 1))
                exit(-1)

        self.info_arrays["Nosepoke"] = good_nosepokes
        self.info_arrays["Un_Nosepoke"] = un_nosepokes

        fi_starts = self.info_arrays["left_light"]
        fr_starts = self.info_arrays["right_light"]
        trial_types = np.zeros(6)

        # set trial types - 1 is FR, 0 is FI
        for i in range(3):
            j = int(fi_starts[i] // 305)
            trial_types[j] = 0
            j = int(fr_starts[i] // 305)
            trial_types[j] = 1
        self.info_arrays["Trial Type"] = trial_types

        # TODO parse other file to remove fixed value
        self.metadata["fixed_interval (secs)"] = 30
        fi = self.get_metadata("fixed_interval (secs)")

        # Set left presses and unnecessary left presses
        split_left_presses = split_into_blocks(
            left_presses, 305, 6)
        left_presses_fi = np.concatenate(
            split_left_presses[np.nonzero(trial_types == 0)])
        left_presses_fr = np.concatenate(
            split_left_presses[np.nonzero(trial_types == 1)])
        good_nosepokes_fi = np.concatenate(
            split_into_blocks(good_nosepokes, 305, 6)[
                np.nonzero(trial_types == 0)])

        fi_allow_times = np.add(good_nosepokes_fi, fi)
        good_left_presses, un_left_presses = split_array_with_another(
            left_presses_fi, fi_allow_times)
        self.info_arrays["L"] = good_left_presses
        self.info_arrays["Un_L"] = un_left_presses

        un_fr_err, fr_err = split_array_in_between_two(
            left_presses_fr, pell_ts_exdouble, good_nosepokes)
        self.info_arrays["Un_FR_Err"] = un_fr_err
        self.info_arrays["FR_Err"] = fr_err

        # set right presses and unnecessary right presses
        split_right_presses = split_into_blocks(
            right_presses, 305, 6)
        right_presses_fr = np.concatenate(
            split_right_presses[np.nonzero(trial_types == 1)])
        right_presses_fi = np.concatenate(
            split_right_presses[np.nonzero(trial_types == 0)])

        un_right_presses, good_right_presses = split_array_in_between_two(
            right_presses_fr, pell_ts_exdouble, good_nosepokes)
        self.info_arrays["R"] = good_right_presses
        self.info_arrays["Un_R"] = un_right_presses

        un_fi_err, fi_err = split_array_in_between_two(
            right_presses_fi, pell_ts_exdouble, good_nosepokes)
        self.info_arrays["Un_FI_Err"] = un_fi_err
        self.info_arrays["FI_Err"] = fi_err

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

        pell_ts_exdouble, dpell = self.split_pell_ts()
        reward_times = self.get_rw_ts()

        # Assign schedule type to trials
        schedule_type = []
        if stage == '7' or stage == '6':
            norm_r_ts, _, _, _, _ = self.split_sess(
                norm=False, plot_all=True)
            sch_type = self.get_arrays('Trial Type')

            for i, block in enumerate(norm_r_ts):
                if sch_type[i] == 1:
                    b_type = 'FR'
                elif sch_type[i] == 0:
                    b_type = 'FI'
                for _, _ in enumerate(block):
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
        return "{}_{}_{}_{}.{}".format(
            self.get_metadata("subject"),
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
