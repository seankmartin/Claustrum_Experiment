"""
This module is a representation of a single Operant box Session.

Written by Sean Martin and Gao Xiang Ham
"""
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cophenet
import os
from copy import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bvmpc.bv_session_info import SessionInfo
from bvmpc.bv_axona import AxonaInput, AxonaSet
from bvmpc.bv_array_methods import check_error_during_tone
from bvmpc.bv_array_methods import split_into_blocks
from bvmpc.bv_array_methods import split_array
from bvmpc.bv_array_methods import split_array_with_another
from bvmpc.bv_array_methods import split_array_in_between_two
import bvmpc.bv_plot as bv_plot
import scipy.cluster.hierarchy as shc


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
            axona_file=None, recording=None, s_type="6",
            neo_backend="nix", verbose=False, file_origin=None):
        """See help(Session) for more info."""
        self.session_info = SessionInfo()
        self.fname = None
        self.metadata = {}
        self.info_arrays = {}
        self.verbose = verbose
        self.file_origin = file_origin
        self.out_dir = None
        self.trial_df = None
        self.trial_df_norm = None
        self.cluster_results = None
        self.clust_reord_trial_df = None
        self.insert = []  # Stores inserted reward ts due to post-hoc correction
        self.title = None

        a = 0 if lines is None else 1
        b = 0 if h5_file is None else 1
        c = 0 if neo_file is None else 1
        d = 0 if axona_file is None else 1
        e = 0 if recording is None else 1
        if (a + b + c + d + e != 1):
            print(
                "Error: Session takes one of" +
                "h5_file, lines, neo_file, recording, and axona_file")
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
            self.fname = axona_file[:-3]
            self._extract_axona_info()
        
        elif recording is not None:
            data = recording.data
            self.metadata = copy(recording.attrs)
            print(list(recording.attrs.keys()))
            self.metadata["start_date"] = self.metadata["date"]
            self.metadata["end_date"] = self.metadata["date"]
            self.metadata["subject"] = self.metadata["rat_id"]
            self.metadata["start_time"] = self.metadata["time"]
            self.metadata["end_time"] = str(
                datetime.strptime(self.metadata["time"], "%H:%M:%S") +
                timedelta(seconds=self.metadata["duration"])).split(" ")[1]

            session_type = recording.attrs["maze_type"]
            if session_type == "RandomisedBlocks":
                session_number = "6_RandomisedBlocks_p"
            elif session_type == "RandomisedBlocksFlipped":
                session_number = "6_RandomisedBlocks_p"
            elif session_type == "RandomisedBlocksExtended":
                session_number = "7_RandomisedBlocksExtended_p"
            self.metadata["name"] = session_number

            df = data.processing["operant_behaviour"]["times"].to_dataframe()
            for (column_name, column_data) in df.items():
                self.info_arrays[column_name] = column_data.values[~np.isnan(column_data.values)]

        else:
            print("Error: Unknown situation in Session init")
            exit(-1)

    def init_trial_dataframe(self):
        """Initialise the trial based Pandas dataframe for this session."""
        self._init_trial_df()

    def get_title(self):
        if self.title is None:
            date = self.get_metadata('start_date').replace('/', '_')
            sub = self.get_metadata('subject')
            stage = self.get_stage()
            self.title = '{} {} S{}'.format(sub, date, stage)
        return self.title

    def get_date(self):
        """Return date in datetime format"""
        import datetime
        date = self.get_metadata('start_date')
        dd = datetime.datetime.strptime(date, '%m/%d/%y').date()
        return dd

    def get_insert(self):
        """
        Return a list of inserted reward timestamps due to post-hoc correction.

        """
        return self.insert

    def add_insert(self, to_insert):
        """
        Add to list of inserted reward timestamps due to post-hoc correction.

        """
        insert_list = self.get_insert()
        insert_list.append(to_insert)
        self.insert = insert_list

    def split_pell_ts(self):
        """
        Return a tuple of arrays splitting up the rewards.

        (Pellets without doubles, time of doubles)

        """
        pell_ts = self.get_arrays("Reward")

        dpell_bool = np.diff(pell_ts) < 0.8
        # Provides index of double pell in pell_ts
        dpell_idx = np.flatnonzero(dpell_bool) + 1
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
            self, norm=True, blocks=None, error_only=False, all_levers=False):
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
        pell_double = np.flatnonzero(np.diff(pell_ts) < 0.8)
        # returns reward ts after d_pell
        reward_double = reward_times[
            np.searchsorted(
                reward_times, pell_ts[pell_double], side='right')]
        err_lever_ts = []

        if blocks is not None:
            pass
        else:
            # blocks = np.arange(5, 1830, 305)  # Default split into schedules
            blocks = self.get_tone_starts() + 5  # Default split into schedules

        incl = ""  # Initialize print for type of extracted lever_ts
        if stage == '7' and error_only:  # plots errors only
            incl = '_Errors_Only'
            lever_ts = self.get_err_lever_ts()
        elif stage == '7' and all_levers:  # plots all responses incl. errors
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
                if stage == '7' and all_levers:  # plots all responses incl. errors
                    norm_err_ts.append(split_err_ts[i + 1] - blocks[i])
        else:
            norm_lever_ts = split_lever_ts[1:]
            norm_reward_ts = split_reward_ts[1:]
            norm_err_ts = split_err_ts[1:]
            norm_double_r_ts = split_double_r_ts[1:]
        return (
            norm_reward_ts, norm_lever_ts, norm_err_ts, norm_double_r_ts, incl)

    def old_extract_features(self, should_scale=True):
        """
        Extract per trial features
        This is the previous version of this function.

        Parameters
        ------
        should_scale : bool, True
            Optional. Whether to scale the data to unit variance

        Returns
        -------
        feat_df : pd.df
            pandas dataframe with
        """
        df = self.get_trial_df_norm()
        feat_names = ["Trial_Len", "Pell",
                      "Rw_lat", "Err", "1st_Resp", "Resp_n",
                      "Avg_Press_Rate", "Avg_Press_Gradient"]

        # Number of features before lever histogram
        n_feat_bh = len(feat_names)

        # Set hist bin size for lever responses here
        h_bin_final = 4

        for i in np.arange(h_bin_final):
            feat_names.append("L_Hist-{}".format(i))

        features = np.zeros(
            shape=(len(df.index), len(feat_names)))

        for row in df.itertuples():
            index = row.Index

            # Trial duration
            features[index, 0] = row.Reward_ts

            # Completion Latency
            features[index, 1] = row.Pellet_ts

            # Reward Latency
            features[index, 2] = row.Reward_ts - row.Pellet_ts

            # Number of err responses
            features[index, 3] = row.Err_ts.size

            # Time to first response
            features[index, 4] = row.First_response

            # Number of lever responses
            features[index, 5] = np.count_nonzero(
                ~np.isnan(row.Levers_ts))

            x = row.Levers_ts
            press_times = x[~np.isnan(x)]

            # Average length between lever presses
            if len(press_times) > 1:
                features[index, 6] = np.mean(np.diff(press_times))
            else:
                features[index, 6] = row.Reward_ts

            # Average rate of change of the rate of change.
            if len(press_times) > 2:
                features[index, 7] = np.mean(
                    np.gradient(np.gradient(press_times, 1), 1))
            else:
                features[index, 7] = 0

            # Histogram values of lever responses
            lever_hist = np.histogram(
                x[~np.isnan(x)], bins=h_bin_final, density=True)[0]
            features[index, n_feat_bh:(n_feat_bh + h_bin_final)] = lever_hist

            # double_pellet_time = row.D_Pellet_ts
            # if len(double_pellet_time) == 0:
            #     d_pell_feature = 0
            # else:
            #     d_pell_feature = double_pellet_time
            # features[index, 2] = d_pell_feature
            # err_hist = np.histogram(
            # row.Err_ts, bins=5, density=False)[0]

        feat_df = pd.DataFrame(features, columns=feat_names)

        # Drop features/columns with all zeros
        feat_df = feat_df.loc[:, (feat_df != 0).any(axis=0)]

        if should_scale:  # Normalize features
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(feat_df)
            feat_df = pd.DataFrame(data_scaled, columns=feat_df.columns)

        return feat_df

    def extract_features(self, should_scale=True):
        """
        Extract per trial features

        Parameters
        ------
        should_scale : bool, True
            Optional. Whether to scale the data to unit variance

        Returns
        -------
        feat_df : pd.df
            pandas dataframe with
        """
        df = self.get_trial_df_norm()
        feat_names = ["Trial_Len", "Resp_n",
                      "Avg_Press_Rate (s)", "Avg_Press_Gradient",
                      "Max_in_10s"]

        # Number of features before lever histogram
        n_feat_bh = len(feat_names)

        # Set hist bin size for lever responses here
        # h_bin_final = 3
        # bins = np.arange(0, 60 + 0.00001, 60 / h_bin_final)

        # for i in np.arange(h_bin_final):
        #     feat_names.append("L_Hist-{}".format(i))

        features = np.zeros(
            shape=(len(df.index), len(feat_names)))

        for row in df.itertuples():
            index = row.Index

            # Trial duration
            features[index, 0] = row.Reward_ts

            x = row.Levers_ts
            press_times = x[~np.isnan(x)]

            # Number of lever responses
            features[index, 1] = np.count_nonzero(
                ~np.isnan(row.Levers_ts))

            # Average length between lever presses
            if len(press_times) > 1:
                features[index, 2] = np.mean(np.diff(press_times))
            else:
                features[index, 2] = row.Reward_ts

            # Average rate of change of the rate of change.
            if len(press_times) > 2:
                features[index, 3] = np.mean(
                    np.gradient(np.gradient(press_times, 1), 1))
            else:
                features[index, 3] = 0

            # Find the max number of response 10 seconds apart
            time_len = 10
            dt = time_len / 2
            max_val = 0
            for value in press_times:
                in_range = np.logical_and(
                    press_times >= value - dt,
                    press_times <= value + dt)
                num_in_range = np.count_nonzero(in_range)
                if num_in_range > max_val:
                    max_val = num_in_range

            features[index, 4] = max_val

            # Histogram values of lever responses
            # lever_hist = np.histogram(
            #     x[~np.isnan(x)], bins=bins, density=True)[0]
            # features[index, n_feat_bh:(n_feat_bh + h_bin_final)] = lever_hist

        feat_df = pd.DataFrame(features, columns=feat_names)

        # Drop features/columns with all zeros
        feat_df = feat_df.loc[:, (feat_df != 0).any(axis=0)]

        if should_scale:  # Normalize features
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(feat_df)
            feat_df = pd.DataFrame(data_scaled, columns=feat_df.columns)

        return feat_df

    def perform_pca(self, n_components=5, should_scale=True):
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
        tuple : (pd.df, pd.df, PCA)
            (features 2d array, PCA of features, PCA object)

        """
        data = self.extract_features(should_scale=should_scale)
        pca = PCA(n_components=n_components)

        after_pca = pca.fit_transform(data)
        after_pca = pd.DataFrame(
            after_pca, columns=['PC{}'.format(x) for x in np.arange(
                after_pca.shape[1]) + 1])

        print(
            "\nPCA fraction of explained variance",
            pca.explained_variance_ratio_)
        print(
            "Total explained variance: ", sum(pca.explained_variance_ratio_))
        return data, after_pca, pca

    def perform_HDBSCAN(self, should_scale=True):
        """ Testing purposes only. Perform HDBSCAN on per trial features. """

        fig, ax = plt.subplots(1, 2, figsize=(15, 8))
        import hdbscan
        _, data, _ = self.perform_pca(
            n_components=None, should_scale=should_scale)
        ax[0].set_xlabel('PCA 1')
        ax[0].set_ylabel('PCA 2')

        df = self.get_trial_df_norm()
        markers = df['Schedule']
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2)
        clusterer.fit(data)
        print(clusterer.labels_.max())
        # HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
        #         gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
        #         metric='euclidean', min_cluster_size=5, min_samples=None, p=None)

        sns.scatterplot(data[:, 0], data[:, 1], s=50, linewidth=0,
                        style=markers, hue=clusterer.labels_, palette='deep', ax=ax[0])

        clusterer.condensed_tree_.plot(select_clusters=True,
                                       selection_palette=sns.color_palette('deep')[1:], axis=ax[1])  # Plots condensed dendogram
        plt.suptitle('HDBSCAN', fontsize=24)
        fig.text(0.5, 0.91, self.get_title(),
                 transform=fig.transFigure, ha='center')

        plt.show()
        exit(-1)

    def perform_UMAP(self, should_scale=True):
        """ Testing purposes only. Perform UMAP on per trial features. """
        import umap
        data = self.extract_features(should_scale=should_scale)
        df = self.get_trial_df_norm()
        color = [sns.color_palette()[x]
                 for x in df.Schedule.astype('category').cat.codes]

        for n in (2, 5, 10, 20, 50, 100, 200):
            self.draw_umap(data, color, n_neighbors=n,
                           title='n_neighbors = {}'.format(n))
        plt.show()
        exit(-1)

        # reducer = umap.UMAP()
        # embedding = reducer.fit_transform(data)
        # plt.scatter(embedding[:, 0], embedding[:, 1], c=[
        #             sns.color_palette()[x] for x in df.Schedule.astype('category').cat.codes])
        # plt.gca().set_aspect('equal', 'datalim')
        # plt.title('UMAP projection of the Behav', fontsize=24)
        # plt.show()

        return data

    def draw_umap(
            self, data, c, n_neighbors=15, min_dist=0.1, n_components=2,
            metric='euclidean', title=''):
        """ For testing out UMAP parameters """
        import umap
        fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric
        )
        u = fit.fit_transform(data)
        fig = plt.figure()
        if n_components == 1:
            ax = fig.add_subplot(111)
            ax.scatter(u[:, 0], range(len(u)), c=c)
        if n_components == 2:
            ax = fig.add_subplot(111)
            ax.scatter(u[:, 0], u[:, 1], c=c)
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=c, s=100)
        plt.title(title, fontsize=18)

    def hier_cluster(
            self, should_scale=True, should_pca=True, cutoff=None,
            test_methods=False):
        """
        Perform hierarchical clustering on per trial features

        Parameters
        ------
        should_scale : bool, True
            whether to scale the data to unit variance
        cutoff: int, None
            Optional. Level at which to cutoff dendrogram

        Returns
        -------
        dict: {'Z' : np.array, 'reindex' : list[int], 'clusters' : np.array}
            {linkage matrix, index for sorting trials, cluster indices}

        """
        if should_pca:
            _, data, _ = self.perform_pca(
                n_components=5)  # Use PCA as features
        else:
            data = self.extract_features(
                should_scale=should_scale)  # Use raw features

        from bvmpc.bv_utils import test_all_hier_clustering
        if test_methods:
            test_all_hier_clustering(data, verbose=True)

        Z = shc.linkage(data, method='centroid')  # linkage matrix
        c, coph_dists = cophenet(Z, pdist(data))

        print('Cophentic Correlation Coefficient: ', c)

        dend = shc.dendrogram(Z, no_plot=True, color_threshold=cutoff)

        reindex = list(map(int, dend['ivl']))  # index for sorting trials

        if cutoff is None:
            cutoff = 0.7 * max(Z[:, 2])

        clusters = shc.fcluster(Z, cutoff, criterion='distance')
        self.cluster_results = {
            'Z': Z,
            'reindex': reindex,
            'clusters': clusters,
        }

        return self.cluster_results

    def get_cluster_results(self, should_pca=True, cutoff=None):
        """
        Returns cluster results from hier_cluster

        Returns
        -------
        dict: {'Z' : np.array, 'reindex' : list[int], 'clusters' : np.array}
            {linkage matrix, index for sorting trials, cluster indices}

        """
        if self.cluster_results is None:
            self.hier_cluster(should_pca=should_pca, cutoff=cutoff)
        return self.cluster_results

    def get_cluster_ordering(self):
        """ Returns cluster ordering from hier_cluster """
        if self.cluster_results is None:
            self.hier_cluster()
        return self.cluster_results['reindex']

    def get_cluster_ordered_df(self):
        """ Returns trial_df_norm in cluster order """
        if self.clust_reord_trial_df is None:
            reindex = self.cluster_results['reindex']
            self.clust_reord_trial_df = self.get_trial_df_norm().reindex(reindex)
        return self.clust_reord_trial_df

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
        """Get dataframe split into trials without normalisation (time).

        Returns
        -------
        mod: used to identify trials where reward timestamps modified post-hoc

            trial_df = {
                'Reward_ts': reward_times,
                'Pellet_ts': pell_ts_exdouble,
                'D_Pellet_ts': trial_dr_ts,
                'Schedule': schedule_type,
                'Levers_ts': trial_lever_ts,
                'Err_ts': trial_err_ts,
                'Tone_s': trial_tone_start,
                'Trial_s': trial_norm[:-1],
                'Mod': mod
            }
        """
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

    def get_one_lever_ts(self, lever_type, include_un=True):
        """
        Get the timestamps of an individual lever press side.
        Lever type is expected to be "L" or "R".

        """
        if (lever_type != "L") and (lever_type != "R"):
            raise ValueError(
                "lever_type is expected to be L or R, entered {}".format(
                    lever_type))
        levers = [self.get_arrays(lever_type)]
        if include_un:
            levers.append(self.get_arrays("Un_" + lever_type))
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
            # Check if trial switched before reward collection
            # -> Adds collection as switch time
            block_ends = self.get_block_ends()
            split_pell_ts = split_into_blocks(
                pell_ts_exdouble, blocks=block_ends)
            split_reward_ts = split_into_blocks(
                reward_times, blocks=block_ends)

            for i, (pell, reward) in enumerate(
                    zip(split_pell_ts, split_reward_ts)):
                if len(pell) > len(reward):
                    block_len = block_ends[i]
                    reward_times = np.insert(
                        reward_times, np.searchsorted(
                            reward_times, block_len), block_len)

        # if last reward time < last pellet dispensed,
        # assume animal picked reward at end of session.
        # Checks if last block contains responses first
        if reward_times[-1] < pell_ts[-1]:
            if stage == '7' or stage == '6':
                reward_times = np.append(reward_times, block_ends[-1])
            else:
                reward_times = np.append(reward_times, trial_len)

        return np.sort(reward_times, axis=None)

    def get_block_ends(self):
        """Return block end times - Tone end times + end of session"""
        sound = self.info_arrays.get("sound", [])
        if len(sound) != 0:
            # Axona
            block_ends = np.concatenate([sound[1:], [sound[-1] + 305]])
        else:
            # MED-PC
            trial_len = self.get_metadata("trial_length (mins)") * 60
            trial_len += 5
            repeated_trial_len = (trial_len) * 6
            block_ends = np.arange(trial_len, repeated_trial_len, trial_len)
        return block_ends

    def get_tone_starts(self):
        """Return tone start times"""
        sound = self.info_arrays.get("sound", [])
        if len(sound) != 0:
            # Axona
            block_starts = np.array(sound - 5)
        else:
            # MED-PC
            trial_len = self.get_metadata("trial_length (mins)") * 60
            repeated_trial_len = (trial_len) * 6
            block_starts = np.arange(0, repeated_trial_len, trial_len)
        return block_starts

    def get_valid_tdf(self, excl_dr=False, norm=False):
        """
        Returns trial_df of excluding:
            First trial in FI blocks
            Trials with post-hoc modification of reward timestamp
            Trials more than 60s

        Parameters
        ----------
        excl_dr : bool, False
            Optional. Excludes double reward trials
        norm : bool, False
            Optional. Return trial_df_norm with masks applied

        """
        if norm:
            df = self.get_trial_df_norm()
        else:
            df = self.get_trial_df()

        # Masks used for filtering trials
        mod_mask = df.Mod.notnull()  # trials with post-hoc rw correction
        FI_s_mask = (df.Tone_s.str.len() != 0) & (
            df.Schedule == 'FI')  # 1st trial of each FI blocks
        tlen_mask = (df.Reward_ts - df.Trial_s) > 60  # trials longer then 60s

        df = df[(~mod_mask) & (~FI_s_mask) & (~tlen_mask)]

        if excl_dr:
            dr_mask = df.D_Pellet_ts.str.len() != 0
            df = df[(~dr_mask)]

        return df

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

        # Numerical metadata extract from log file
        # **No log file in some cases. Static method for now**
        self.metadata["fixed_ratio"] = 6
        self.metadata["double_reward_window (secs)"] = 10
        self.metadata["fixed_interval (secs)"] = 30
        self.metadata["num_trials"] = 6
        self.metadata["trial_length (mins)"] = 5
        self.metadata['subject'] = os.path.basename(
            self.axona_file).split("_")[0]

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
        block_ends = self.get_block_ends()
        block_s = self.get_tone_starts() + 5  # End of tone

        # Check for errors in presses during Tone presentation
        left_presses = check_error_during_tone(
            left_presses, block_s, l_type=" left ")
        right_presses = check_error_during_tone(
            right_presses, block_s, l_type=" right ")

        # Extract nosepokes as necessary and unecessary
        pell_ts_exdouble, _ = self.split_pell_ts()
        good_nosepokes, un_nosepokes = split_array_with_another(
            nosepokes, pell_ts_exdouble)

        split_nosepokes = split_into_blocks(
            good_nosepokes, blocks=block_ends)
        split_pellets = split_into_blocks(
            pell_ts_exdouble, blocks=block_ends)
        split_all_nosepokes = split_into_blocks(
            nosepokes, blocks=block_ends)
        for i in range(6):
            b1, b2 = split_nosepokes[i], split_pellets[i]
            if len(b2) > len(b1):
                print("block: {}, End-time: {}".format(i, block_ends[i]))
                # print("good nosepokes: {}".format(good_nosepokes))
                # print("nosepokes: {}".format(b1))
                # print("pellets: {}".format(b2))

                last_nosepoke_idx = -1
                for j in range(i, -1, -1):
                    bl = split_all_nosepokes[j]
                    if len(bl) == 0:
                        continue
                    # Finds last nosepoke in this block, or previous block
                    last_nosepoke_arr = np.nonzero(
                        np.abs(nosepokes - bl[-1]) <= 0.00001)
                    last_nosepoke_idx = last_nosepoke_arr[0][0]
                    break

                # Replaces overflowed nosepoke w block end in main array
                to_insert = block_ends[i] - 0.001
                print(' Insert: ', to_insert)
                self.add_insert(to_insert)
                # if i == 5:
                #     to_insert = pell_ts_exdouble[-1] + 0.01
                nosepokes = np.insert(
                    nosepokes, last_nosepoke_idx + 1, to_insert)
                good_nosepokes, un_nosepokes = split_array_with_another(
                    nosepokes, pell_ts_exdouble)
                # if i < 5:
                # ignores first nosepoke in next block in split arrays
                split_nosepokes = split_into_blocks(
                    good_nosepokes, blocks=block_ends)
                split_all_nosepokes = split_into_blocks(
                    nosepokes, blocks=block_ends)
                # print("Corrected:", good_nosepokes)

        split_nosepokes = split_into_blocks(
            good_nosepokes, blocks=block_ends)
        split_pellets = split_into_blocks(
            pell_ts_exdouble, blocks=block_ends)
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
            left_presses, blocks=block_ends)
        left_presses_fi = np.concatenate(
            split_left_presses[np.flatnonzero(trial_types == 0)])
        left_presses_fr = np.concatenate(
            split_left_presses[np.flatnonzero(trial_types == 1)])
        good_nosepokes_fi = np.concatenate(
            split_into_blocks(good_nosepokes, blocks=block_ends)[
                np.flatnonzero(trial_types == 0)])

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
            right_presses, blocks=block_ends)
        right_presses_fr = np.concatenate(
            split_right_presses[np.flatnonzero(trial_types == 1)])
        right_presses_fi = np.concatenate(
            split_right_presses[np.flatnonzero(trial_types == 0)])

        un_right_presses, good_right_presses = split_array_in_between_two(
            right_presses_fr, pell_ts_exdouble, good_nosepokes)
        self.info_arrays["R"] = good_right_presses
        self.info_arrays["Un_R"] = un_right_presses

        un_fi_err, fi_err = split_array_in_between_two(
            right_presses_fi, pell_ts_exdouble, good_nosepokes)
        self.info_arrays["Un_FI_Err"] = un_fi_err
        self.info_arrays["FI_Err"] = fi_err

        # extract trial start times
        reward_times = self.get_rw_ts()

        t_start_blocks = np.split(
            reward_times, np.searchsorted(reward_times, block_ends))

        t_start = []
        for i, (x, tone_end) in enumerate(zip(t_start_blocks, block_ends)):
            # replace trial start w next block start if empty
            if len(x) == 0:
                x = np.array([tone_end])
            # replace last start in block w next block start
            elif x[-1] < tone_end:
                x[-1] = tone_end
            t_start.append(np.array(x))
        t_start = np.concatenate(t_start).ravel()
        t_start = np.insert(t_start, 0, block_s[0])
        # Trials start after tone ends
        self.info_arrays["Trial_Start"] = t_start[:-1]

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
        """ Generates pandas dataframe based on trials per row. """

        session_type = self.get_metadata('name')
        stage = session_type[:2].replace('_', '')  # Obtain stage number w/o _

        pell_ts_exdouble, dpell = self.split_pell_ts()
        reward_times = self.get_rw_ts()
        trial_starts = self.get_arrays("Trial_Start")

        # Assign schedule type to trials
        schedule_type, sch_block = [], []
        mod = []
        # Reward times artificially inserted
        mod_rw = self.get_insert()

        if stage == '7' or stage == '6':
            # End of tone
            block_s = self.get_tone_starts() + 5

            # from matplotlib import pyplot as plt
            # plt.eventplot(blocks, colors='r', label='s')
            # plt.eventplot(mod_rw, colors='g', label='e')
            # plt.legend()
            # plt.show()
            tstarts_in_blocks = np.split(
                trial_starts, np.searchsorted(trial_starts, block_s[1:]))
            sch_type = self.get_arrays('Trial Type')
            for i, block in enumerate(tstarts_in_blocks):
                if sch_type[i] == 1:
                    b_type = 'FR'
                elif sch_type[i] == 0:
                    b_type = 'FI'
                for rw in block:
                    schedule_type.append(b_type)
                    sch_block.append(b_type + '-{}'.format(i))
        else:
            if stage == '4':
                b_type = 'CR'
            elif stage == '5a':
                b_type = 'FR'
            elif stage == '5b':
                b_type = 'FI'
            else:
                b_type = 'NA'
            for _ in trial_starts:
                schedule_type.append(b_type)
                sch_block.append(None)

        # Rearrange timestamps based on trial per row
        lever_ts = self.get_lever_ts(True)
        err_ts = self.get_err_lever_ts(True)

        trial_rw_ts = np.split(
            reward_times,
            (np.searchsorted(reward_times, trial_starts, side='right')[1:]))
        trial_pell_ts = np.split(
            pell_ts_exdouble,
            (np.searchsorted(pell_ts_exdouble, trial_starts)[1:]))
        trial_lever_ts = np.split(
            lever_ts, (np.searchsorted(lever_ts, trial_starts)[1:]))
        trial_err_ts = np.split(
            err_ts, (np.searchsorted(err_ts, trial_starts)[1:]))
        trial_dr_ts = np.split(
            dpell, (np.searchsorted(dpell, trial_starts)[1:]))
        # Tone end ts
        trial_tone_end = np.split(
            block_s, (np.searchsorted(block_s, trial_starts)[1:]))

        # # Testing - print values
        # x = trial_rw_ts
        # for i, (a, b) in enumerate(zip(x, trial_starts)):
        #     print("Trial ", i)
        #     print("T_start: ", b)
        #     print(a)
        # exit(-1)

        # Assign mod value to trial type:
        for rw in trial_rw_ts:
            if rw:
                if np.isclose(rw, mod_rw):
                    mod.append(1)
                else:
                    mod.append(None)
            else:
                mod.append(None)

        # Reconstruct Tone start times in correct trial location
        from copy import deepcopy
        trial_tone_start = deepcopy(trial_tone_end)
        for i, t in enumerate(trial_tone_end):
            if not len(t) == 0:
                trial_tone_start[i] = trial_tone_end[i] - 5

        # Initialize array for lever timestamps
        # Max lever press per trial
        trials_max_l = len(max(trial_lever_ts, key=len))
        lever_arr = np.empty((len(trial_starts), trials_max_l,))
        lever_arr.fill(np.nan)
        first_response_arr = []
        trials_max_err = len(max(trial_err_ts, key=len)
                             )  # Max err press per trial
        err_arr = np.empty((len(trial_starts), trials_max_err,))
        err_arr.fill(np.nan)

        # 2D array of lever timestamps
        for i, (l, err) in enumerate(zip(trial_lever_ts, trial_err_ts)):
            l_end = len(l)
            lever_arr[i, :l_end] = l[:]
            err_end = len(err)
            err_arr[i, :err_end] = err[:]
            first_response_arr.append(np.array([l[0]]))

        # Splits lever ts in each trial into np.arrs for handling in pandas
        # lever_arr = np.vsplit(lever_arr, i+1)
        # err_arr = np.vsplit(err_arr, i+1)
        lever_arr = list(lever_arr)
        err_arr = list(err_arr)

        # Arrays used for normalization of timestamps to trials
        trial_norm = trial_starts

        norm_rw = deepcopy(trial_rw_ts)
        norm_pell = deepcopy(trial_pell_ts)
        norm_lever = deepcopy(lever_arr)
        norm_dr = deepcopy(trial_dr_ts)
        norm_err = deepcopy(err_arr)
        norm_tone = deepcopy(trial_tone_start)
        norm_trial_s = deepcopy(trial_starts)
        norm_first_response = deepcopy(first_response_arr)

        # Normalize timestamps based on start of trial
        for i, _ in enumerate(norm_rw):
            norm_lever[i] -= trial_norm[i]
            norm_err[i] -= trial_norm[i]
            norm_dr[i] -= trial_norm[i]
            norm_pell[i] -= trial_norm[i]
            norm_rw[i] -= trial_norm[i]
            norm_tone[i] -= trial_norm[i]
            norm_trial_s[i] -= trial_norm[i]
            norm_first_response[i] -= trial_norm[i]

        # Convert trial_starts into list of np.arr of list
        trial_starts = [np.array([x]) for x in trial_starts]

        # Timestamps kept as original starting from session start
        session_dict = {
            'Reward_ts': trial_rw_ts,
            'Pellet_ts': trial_pell_ts,
            'D_Pellet_ts': trial_dr_ts,
            'Schedule': schedule_type,
            'Levers_ts': trial_lever_ts,
            'Err_ts': trial_err_ts,
            'Tone_s': trial_tone_start,
            'Trial_s': trial_starts,
            'First_response': first_response_arr,
            'Sch_block': sch_block,
            'Mod': mod
        }
        # for key, x in session_dict.items():
        #     print(key, len(x))

        # Timestamps normalised to each trial start
        trial_dict = {
            'Reward_ts': norm_rw,
            'Pellet_ts': norm_pell,
            'D_Pellet_ts': norm_dr,
            'Schedule': schedule_type,
            'Levers_ts': norm_lever,
            'Err_ts': norm_err,
            'Tone_s': norm_tone,
            'Trial_s': norm_trial_s,
            'First_response': norm_first_response,
            'Sch_block': sch_block,
            'Mod': mod
        }

        for key, val in trial_dict.items():
            print(
                "Initialised trial df with {} trials and keys {}".format(
                    len(val), list(trial_dict.keys())))
            break
        self.trial_df = pd.DataFrame(session_dict)
        self.trial_df_norm = pd.DataFrame(trial_dict)

        # from bvmpc.bv_utils import check_fn
        # self.trial_df = pd.DataFrame(session_dict).applymap(check_fn)
        # self.trial_df_norm = pd.DataFrame(trial_dict).applymap(check_fn)

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
