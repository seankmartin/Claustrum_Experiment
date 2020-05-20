import os
import json
import argparse

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import neurochat.nc_plot as nc_plot
import bvmpc.bv_plot as bv_plot
import bvmpc.bv_analyse as bv_an
from bvmpc.bv_utils import make_dir_if_not_exists
from bvmpc.bv_utils import get_all_files_in_dir
from bvmpc.bv_utils import chunks
from bvmpc.bv_utils import make_path_if_not_exists, read_cfg, parse_args
from bvmpc.bv_utils import interactive_refilt
from bvmpc.lfp_odict import LfpODict
from bvmpc.compare_lfp import compare_lfp
from bvmpc.lfp_plot import plot_lfp, lfp_csv
from bvmpc.bv_file import load_bv_from_set


def main(fname, out_main_dir, config):
    '''
    Parameters
    ----------
    fname : str
        filenames to be analysed

    Saves plots in a !LFP folder inside out_main_dir

    '''
    # Setup output directory
    o_dir = os.path.join(out_main_dir, "!LFP")
    make_dir_if_not_exists(o_dir)

    # Parse the config file
    analysis_flags = json.loads(config.get("Setup", "analysis_flags"))
    alignment = json.loads(config.get("Setup", "alignment"))
    chan_amount = int(config.get("Setup", "chans"))
    chans = [i for i in range(1, chan_amount + 1)]
    region_dict = config._sections["Regions"]
    regions = []
    for _, val in region_dict.items():
        to_add = val.split(" * ")
        adding = [to_add[0]] * int(to_add[1])
        regions += adding

    shuttle_dict = config._sections["Shuttles"]
    shuttles = []
    for _, val in shuttle_dict.items():
        to_add = val.split(" * ")
        adding = [to_add[0]] * int(to_add[1])
        shuttles += adding

    filt = bool(int(config.get("Setup", "filt")))
    filt_btm = float(config.get("Setup", "filt_btm"))
    filt_top = float(config.get("Setup", "filt_top"))
    filt_params = (filt, filt_btm, filt_top)

    artf = bool(int(config.get("Artefact Params", "artf")))
    sd_thres = float(config.get("Artefact Params", "sd_thres"))
    min_artf_freq = float(config.get("Artefact Params", "min_artf_freq"))
    rep_freq = config.get("Artefact Params", "rep_freq")
    if rep_freq == "":
        rep_freq = None
    else:
        rep_freq = float(config.get("Artefact Params", "rep_freq"))
    artf_params = (artf, sd_thres, min_artf_freq, rep_freq, filt)

    # Split the LFP signals into chunked sets of size 16
    lfp_list = []
    for chan_set in chunks(chans, 16):
        lfp_odict = LfpODict(
            fname, channels=chan_set,
            filt_params=(filt, filt_btm, filt_top),
            artf_params=(artf, sd_thres, min_artf_freq, rep_freq, filt))
        lfp_list.append(lfp_odict)

    # Add events (lever press times etc.) to the mne object
    if "Pre" not in fname:
        session = load_bv_from_set(fname)

    # At this point all of the config parsing and file setup is done

    # TODO move to correct location - Compute ICA
    do_ICA = False
    if do_ICA:
        from sklearn.decomposition import FastICA
        for lfp_odict in lfp_list:
            ori_keys, ori_lfp_list = [], []
            for key, lfp in lfp_odict.get_filt_signal().items():
                ori_lfp_list.append(lfp.get_samples())
                ori_keys.append(key)
            lfp_ts = lfp_odict.get_filt_signal(key=1).get_timestamp()
            X = np.column_stack(ori_lfp_list)

            # Compute ICA
            ica = FastICA(random_state=1)
            S_ = ica.fit_transform(X)  # Reconstruct signals
            A_ = ica.mixing_  # Get estimated mixing matrix
            n_chans = S_.shape[1]

            # Reconstruct excluding speficied ICs
            from copy import deepcopy
            remove_IC_list = [1, 2, 5, 6, 7, 8, 10, 11]
            rS_ = deepcopy(S_)
            if len(remove_IC_list) > 0:
                rS_[:, [x - 1 for x in remove_IC_list]] = 0
            N_ = ica.inverse_transform(rS_, copy=True)

            # Plotting parameters
            win_s, win_e = 0, int(30 * 250)
            lw = 0.5
            rows, cols = n_chans, 1
            OR_gf = bv_plot.GridFig(rows, cols)  # Origi/Recon Figure
            IC_gf = bv_plot.GridFig(rows, cols)  # ICA Figure
            for i, (key, ori, decom, recon) in enumerate(
                    zip(ori_keys, X.T, S_.T, N_.T), 1):
                OR_ax = OR_gf.get_next()
                OR_ax.plot(lfp_ts[win_s:win_e], ori[win_s:win_e], lw=lw,
                           label="T{}".format(key), c='b')
                OR_ax.plot(lfp_ts[win_s:win_e], recon[win_s:win_e], lw=lw,
                           label='rT{}'.format(i), c='hotpink')
                if i == 1:
                    OR_gf.fig.suptitle('Original/Reconstructed Trace')
                OR_ax.legend(fontsize=8, loc='upper left')
                OR_ax.set_xlabel('Time (s)')

                # highlight removed IC in red
                if i in remove_IC_list:
                    c = 'r'
                else:
                    c = 'k'
                IC_ax = IC_gf.get_next()
                IC_ax.plot(lfp_ts[win_s:win_e], decom[win_s:win_e], lw=lw,
                           label='IC{}'.format(i), c=c)
                IC_ax.legend(fontsize=8, loc='upper left')
                if i == 1:
                    IC_gf.fig.suptitle('Independent Components')

                # plt.subplot(n_chans, 3, i*3)
                # plt.plot(recon[win_s:win_e], lw=lw, label='rT{}'.format(i))
                # plt.legend(fontsize=8, loc='upper left')
                # if i == 1:
                #     plt.title('Reconstructed Trace')
                IC_ax.set_xlabel('Time (s)')
            plt.show()

    # TODO temporary location to test MNE
    do_mne = True
    if do_mne:
        import mne
        import bvmpc.bv_mne

        # Setup the mne object with our data
        lfp_odict = LfpODict(
            fname, channels=chans,
            filt_params=filt_params, artf_params=artf_params)
        ch_names = None
        mne_array = bvmpc.bv_mne.create_mne_array(
            lfp_odict, fname=fname, ch_names=ch_names,
            regions=regions, o_dir=o_dir, plot_mon=False)
        base_name = os.path.basename(fname)

        # Add annotations to the created mne object
        annote_loc = os.path.join(
            o_dir, os.path.basename(fname) + '_mne-annotations.txt')
        bvmpc.bv_mne.set_annotations(mne_array, annote_loc)
        # mne_array.plot(n_channels=20, block=True, duration=50,
        #                show=True, clipping="clamp",
        #                title="Raw LFP Data from {}".format(base_name),
        #                remove_dc=False, scalings="auto")
        # bvmpc.bv_mne.save_annotations(mne_array, annote_loc)

        events_dict, mne_events, annot_from_events = bvmpc.bv_mne.generate_events(
            mne_array, session)

        # Add events to annotations
        mne_array.set_annotations(
            mne_array.annotations + annot_from_events)

        # # Plot raw LFP w events
        # mne_array.plot(n_channels=20, block=True, duration=50,
        #                show=True, clipping="clamp",
        #                title="Raw LFP Data w Events from {}".format(base_name),
        #                remove_dc=False, scalings="auto")

        # Do ICA artefact removal on the mne object
        recon_raw = bvmpc.bv_mne.ICA_pipeline(mne_array, regions, chans_to_plot=len(lfp_odict),
                                              base_name=base_name, exclude=[4, 6, 12])
        # recon_raw = mne_array

        # Epoch events
        reject_criteria = dict(eeg=600e-6)  # 600 uV

        # Pick chans based on regions/tetrode number
        # sel = ['ACC'] # Use None to select all channels
        sel = None
        picks = bvmpc.bv_mne.pick_chans(recon_raw, sel=sel)

        epochs = mne.Epochs(recon_raw, mne_events, picks=picks, event_id=events_dict, tmin=-0.5, tmax=0.5,
                            reject=reject_criteria, preload=True)
        comp_conds = ['Right', 'Left']
        epochs.equalize_event_counts(comp_conds)
        epoch_list = []
        for conds in comp_conds:
            epoch_list.append(epochs[conds])
        del recon_raw, epochs  # Free up memory

        for epoch in epoch_list:
            picks = ['RSC-14', 'ACC-9', 'CLA-7', 'AI-5']
            epoch.plot_image(picks=picks)

        # # Test plot_image in groups
        # # Get channels for each region in dict format. For use with plot_image
        # grps = bvmpc.bv_mne.get_reg_chans(mne_array, regions)
        # for epoch in epoch_list:
        #     epoch.plot_image(picks=None, group_by=grps)

        # # Test plot PSD for epoch
        # for i, epoch in enumerate(epoch_list):
        #     fig, ax = plt.subplots()
        #     fig.suptitle('{} - {}'.format(comp_conds[i], sel))
        #     epoch.plot_psd(fmin=1., fmax=40., ax=ax, show=True)
        #     # epoch.plot_psd_topomap(ch_type='eeg', normalize=True)
        # exit(-1)

        # # Test plot power spectrum
        # frequencies = np.arange(2, 30, 3)
        # power = mne.time_frequency.tfr_morlet(epoch, n_cycles=2, return_itc=False,
        #                                       freqs=frequencies, decim=3)
        # for chan in chan_picks:
        #     power.plot([chan])

    if "Pre" in fname:
        behav = False
        Pre = True
        s = None

    else:
        Pre = False
        behav = bool(int(config.get("Behav Params", "behav")))
        behav_plot = json.loads(config.get("Behav Params", "behav_plot"))

        # Load behaviour-related data
        s = session
        sch_type = s.get_arrays('Trial Type')  # FR = 1, FI = 0
        valid = True

        # Plot trial length histogram
        bv_hist = bool(int(config.get("Behav Plot", "hist")))
        if bv_hist:
            hist_name = os.path.join(
                o_dir, os.path.basename(fname) + "_bv_h-tlen.png")
            fig = bv_an.trial_length_hist(s, valid=valid)
            bv_plot.savefig(fig, hist_name)

        # Plot lever response histogram
        bv_hist_lev = bool(int(config.get("Behav Plot", "hist_lev")))
        if bv_hist_lev:
            # excl_dr True to exclude double reward trials from lever hist
            excl_dr = False
            if excl_dr:
                txt = '_exdr'
            else:
                txt = ''
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            hist_name = os.path.join(
                o_dir, os.path.basename(fname) + "_bv_h-lev{}.png".format(txt))
            bv_an.lever_hist(s, ax=ax[0], valid=valid,
                             excl_dr=excl_dr, split_t=False)
            bv_an.lever_hist(s, ax=ax[1], valid=valid,
                             excl_dr=excl_dr, split_t=True)
            bv_plot.savefig(fig, hist_name)

        # Plot behaviour raster
        bv_raster = bool(int(config.get("Behav Plot", "raster")))
        if bv_raster:
            if alignment[0]:
                align_txt = "_rw"
            elif alignment[1]:
                align_txt = "_pell"
            elif alignment[2]:
                align_txt = "_int"
            elif alignment[3]:
                align_txt = "_FRes"
            else:
                align_txt = "_start"
            raster_name = os.path.join(
                o_dir, os.path.basename(fname) +
                "_bv_raster{}.png".format(align_txt))
            fig = bv_an.plot_raster_trials(s, align=alignment[:4])
            bv_plot.savefig(fig, raster_name)

        # Plot cumulative lever response
        bv_cum = bool(int(config.get("Behav Plot", "cumulative")))
        if bv_cum:
            cum_name = os.path.join(
                o_dir, os.path.basename(fname) + "_bv_cum.png")
            fig = bv_an.cumplot_axona(s)

            bv_plot.savefig(fig, cum_name)

        bv_clust = bool(int(config.get("Behav Plot", "clust")))
        plot_feat_box, plot_feat_pp = [0, 1]  # True to plot feat details

        # Do trial based clustering based on animal responses.
        if bv_clust:
            # s.perform_UMAP()  # Testing out UMAP
            # s.perform_HDBSCAN()  # Testing out HDBSCAN

            clust_name = os.path.join(
                o_dir, os.path.basename(fname) + "_bv_clust-KMeans.png")
            fig, feat_df, bef_PCA = bv_an.trial_clustering(
                s, should_pca=True, num_clusts=4, p_2D=False)
            bv_plot.savefig(fig, clust_name)

            fig = bv_an.trial_clust_hier(s, should_pca=True, cutoff=8.5)
            clust_name = os.path.join(
                o_dir, os.path.basename(fname) + "_bv_clust-hier.png")
            bv_plot.savefig(fig, clust_name)

            # Boxplot the features in the dataframe.
            if plot_feat_box:
                fig = bv_an.plot_feats(feat_df)
                fig.text(0.5, 0.895, s.get_title(),
                         transform=fig.transFigure, ha='center')
                feat_plot_name = os.path.join(
                    o_dir, os.path.basename(fname) + "_bv_c_feats.png")
                bv_plot.savefig(fig, feat_plot_name)

                # Boxplot for bef_PCA
                fig = bv_an.plot_feats(bef_PCA)
                fig.text(0.5, 0.895, s.get_title(),
                         transform=fig.transFigure, ha='center')
                feat_plot_name = os.path.join(
                    o_dir, os.path.basename(fname) + "_bv_c_feats_bef.png")
                bv_plot.savefig(fig, feat_plot_name)

            # Do a pairplot of the features in the dataframe
            if plot_feat_pp:
                hue_grouping = 'Schedule'

                # Markers based on schedule
                if hue_grouping == 'Schedule':
                    df = s.get_trial_df_norm()
                    feat_df['markers'] = df["Schedule"].astype('category')
                    bef_PCA['markers'] = df["Schedule"].astype('category')
                    hue = 'markers'

                # Markers based on hier_clustering
                elif hue_grouping == 'clusters':
                    clusters = s.get_cluster_results()['clusters']
                    feat_df['markers'] = pd.Categorical(clusters)
                    bef_PCA['markers'] = pd.Categorical(clusters)
                    hue = 'markers'

                else:
                    hue = None

                # Pairplot for feat_df
                pp = sns.pairplot(feat_df, hue=hue)
                sns_plot_name = os.path.join(
                    o_dir, os.path.basename(fname) + "_bv_c_feats_pp.png")
                pp.savefig(sns_plot_name)

                # Pairplot for bef_PCA
                pp = sns.pairplot(bef_PCA, hue=hue)
                sns_plot_name = os.path.join(
                    o_dir, os.path.basename(fname) + "_bv_c_feats_befpp.png")
                pp.savefig(sns_plot_name)

        # Tone start times excluding first + end time
        blocks = np.append(s.get_tone_starts()[1:], s.get_block_ends()[-1])
        # print(blocks)

    # Plots raw LFP for all tetrodes or output csv with artf_removal results
    r_SI = bool(int(config.get("Setup", "r_SI")))
    r_plot = bool(int(config.get("Setup", "r_plot")))
    # Plot raw LFP in terms of trials
    rt_plot = bool(int(config.get("Setup", "rt_plot")))
    r_csv = bool(int(config.get("Setup", "r_csv")))
    badchans = [x for x in config.get("Setup", "bad_chans").split(", ")]

    # Differential Recording mode (lfp1 - lfp2 in same shuttle)
    dr_mode = bool(int(config.get("Setup", "dr_mode")))

    # Assign specific colors to regions
    if dr_mode:
        regions = regions[::2]
        print(regions)
    gm = bv_plot.GroupManager(regions)

    for p, lfp_odict in enumerate(lfp_list):
        if r_plot:
            ro_dir = os.path.join(o_dir, "Raw")
            make_dir_if_not_exists(ro_dir)
            # Plot raw LFP for all tetrodes in segments
            if s is not None:
                splits = np.concatenate([[0], s.get_block_ends()])
                # splits = np.concatenate([[0], [s.get_block_ends()[-1]]])
            else:
                splits = None
            plot_lfp(ro_dir, lfp_odict, splits=splits,
                     sd=sd_thres, filt=filt, artf=artf,
                     dr_mode=dr_mode, session=s)
        if rt_plot:
            rto_dir = os.path.join(o_dir, "Raw", 'Trials')
            make_dir_if_not_exists(rto_dir)
            if s is not None:
                splits = np.concatenate(
                    s.get_trial_df()['Trial_s'].tolist())
                plot_lfp(rto_dir, lfp_odict, splits=splits,
                         sd=sd_thres, filt=filt, artf=artf,
                         dr_mode=dr_mode, session=s)
            else:
                pass

        if r_csv:
            shut_s, shut_end = p * 16, 16 + p * 16
            lfp_csv(fname, o_dir, lfp_odict, sd_thres,
                    min_artf_freq, shuttles[shut_s:shut_end], filt)

    # Plots Similarity Index for LFP tetrodes
    if r_SI:
        compare_lfp(fname, o_dir, ch=chan_amount)

    # Plot periodograms and ptr for each tetrode seperately
    if analysis_flags[0]:
        """
        Plot periodograms and ptr for each tetrode seperately
        ptr - includes vlines indicating tone and reward points

        """
        for p, lfp_odict in enumerate(lfp_list):
            # Set indiv to true for individual periodograms on a 4x4 grid
            indiv = False
            # Set spec to true for individual spectrograms per .png
            spec = False

            # Old code to plot each periodogram in a seperate .png
            # for i, (key, lfp) in enumerate(lfp_odict.get_filt_signal().items()):
            #     graph_data = lfp.spectrum(
            #         ptype='psd', prefilt=False,
            #         db=False, tr=False)
            #     fig = nc_plot.lfp_spectrum(graph_data)
            #     plt.ylim(0, 0.01)
            #     # plt.xlim(0, 40)
            #     out_name = os.path.join(o_dir, "p", key + "p.png")
            #     make_path_if_not_exists(out_name)
            #     fig.suptitle("T" + key + " " + regions[i] + " Periodogram")
            #     bv_plot.savefig(fig, out_name)
            #     plt.close()

            if indiv:
                # Setup 4x4 summary grid
                rows, cols = [4, 4]
                gf = bv_plot.GridFig(rows, cols, wspace=0.3,
                                     hspace=0.3, tight_layout=False)

                # Plot individual periodograms on a 4x4 grid
                for i, (key, lfp) in enumerate(
                        lfp_odict.get_filt_signal().items()):
                    graph_data = lfp.spectrum(
                        ptype='psd', prefilt=False,
                        db=False, tr=False)
                    ax = gf.get_next(along_rows=False)
                    color = gm.get_next_color()
                    nc_plot.lfp_spectrum(
                        graph_data, ax, color, style="Thin-Dashed")
                    plt.ylim(0, 0.015)
                    # plt.xlim(0, 40)
                    ax.text(0.49, 1.08, regions[i + p * 16], fontsize=20,
                            ha='center', va='center', transform=ax.transAxes)
                extra_text_1 = " " + str(p) if p else ""
                extra_text_2 = "_" + str(p) if p else ""
                bname = os.path.basename(fname)
                gf.fig.suptitle(
                    bname[4:] + " Periodogram" + extra_text_1, fontsize=30)
                out_name = os.path.join(
                    o_dir, bname + "_p_sum" + extra_text_2 + ".png")
                make_path_if_not_exists(out_name)
                bv_plot.savefig(gf.get_fig(), out_name)
                plt.close()

            # Plot spectrogram for each eeg as a seperate .png
            if spec:
                for i, (key, lfp) in enumerate(lfp_odict.get_filt_signal().items()):
                    graph_data = lfp.spectrum(
                        ptype='psd', prefilt=False,
                        db=True, tr=True)
                    if graph_data['t'][-1] > 305:
                        rows, cols = [6, 1]
                        gf = bv_plot.GridFig(rows, cols, wspace=0.3,
                                             hspace=0.3, size_multiplier_x=40, tight_layout=False)
                        for k, j in enumerate(blocks):
                            tone_ts = s.get_tone_starts() + 5
                            ax = gf.get_next(along_rows=True)

                            if k == 0:
                                new_lfp = lfp.subsample(
                                    sample_range=(0, j))
                            else:
                                new_lfp = lfp.subsample(
                                    sample_range=(blocks[k - 1], j))
                            graph_data = new_lfp.spectrum(
                                ptype='psd', prefilt=False,
                                db=True, tr=True)
                            nc_plot.lfp_spectrum_tr(graph_data, ax)
                            plt.tick_params(labelsize=20)
                            ax.xaxis.label.set_size(25)
                            ax.yaxis.label.set_size(25)
                            if j == 0:
                                plt.title("T" + key + " " +
                                          regions[i + p * 16] + " Spectrogram", fontsize=40, y=1.05)
                            plt.ylim(0, filt_top)
                            if behav:
                                ax, b_legend = bv_plot.behav_vlines(
                                    ax, s, behav_plot)
                                ax.axvline(tone_ts, linestyle='-',
                                           color='r', linewidth='1.5')  # vline demarcating end of tone
                        fig = gf.get_fig()
                    else:
                        fig, ax = plt.subplots(figsize=(20, 5))
                        nc_plot.lfp_spectrum_tr(graph_data, ax)
                        plt.ylim(0, filt_top)
                        fig.suptitle("T" + key + " " +
                                     regions[i + p * 16] + " Spectrogram")
                    out_name = os.path.join(o_dir, "ptr", key + "ptr.png")
                    make_path_if_not_exists(out_name)
                    bv_plot.savefig(fig, out_name)
                    plt.close()

    # Compile graphs per session in a single .png
    if analysis_flags[1]:
        spec = False
        # Plot all periodograms on 1 plot
        fig, ax = plt.subplots(figsize=(20, 20))
        legend = []
        max_p = 0
        for p, lfp_odict in enumerate(lfp_list):
            if dr_mode:
                signal_used = lfp_odict.get_dr_signals()
                if artf:
                    signal_used = lfp_odict.get_clean_signal()
            else:
                if artf:
                    signal_used = lfp_odict.get_clean_signal()
                else:
                    signal_used = lfp_odict.get_filt_signal()

            for i, (key, lfp) in enumerate(signal_used.items()):
                print(i, key)
                color = gm.get_next_color()
                if key in badchans:
                    print("T{} skipped.".format(key))
                    continue
                graph_data = lfp.spectrum(
                    ptype='psd', prefilt=False,
                    db=False, tr=False)
                nc_plot.lfp_spectrum(
                    graph_data, ax, color, style="Dashed")
                label = regions[i + p * 16] + " T" + key
                legend.append(label)

                # Plot Peaks in lfp
                peaks = True
                if peaks:
                    verbose = False
                    from scipy.signal import find_peaks, peak_prominences
                    peaks, _ = find_peaks(
                        graph_data['Pxx'], distance=4, prominence=1e-6)
                    if verbose:
                        print("{} : {}".format(label, [round(x, 2)
                                                       for x in graph_data['f'][peaks]]))
                        print("Max : ", graph_data['f'][np.nonzero(
                            graph_data['Pxx'] == max(graph_data['Pxx'][peaks]))])
                    ax.scatter(graph_data['f'][peaks], graph_data['Pxx']
                               [peaks], s=250, marker='X', c=np.repeat([color], len(peaks), axis=0), edgecolor='k')

                    cur_max_p = max(graph_data['Pxx'])
                    if cur_max_p > max_p:
                        max_p = cur_max_p
                    else:
                        continue
        plt.tick_params(labelsize=25)
        ax.xaxis.label.set_size(35)
        ax.yaxis.label.set_size(35)
        plt.ylim(0, max_p + max_p * 0.1)
        plt.xlim(0, 40)
        # ax.set_yscale('log')
        # plt.xlim(0, 120)
        # plt.ylim(0, 0.0001)
        # plt.xlim(30, 120)
        # plt.xlim(0, filt_top)
        plt.legend(legend, fontsize=25)

        ro_dir = os.path.join(o_dir, "Raw")
        if dr_mode:  # Hard fix for naming if Differential recording is used
            if artf:
                plt.title(fname.split("\\")[-1][4:] +
                          " Compiled Periodogram - Clean_dr", fontsize=40, y=1.02)
                out_name = os.path.join(
                    ro_dir, fname.split("\\")[-1] + "_p_Clean_dr.png")
            else:
                plt.title(fname.split("\\")[-1][4:] +
                          " Compiled Periodogram_dr", fontsize=40, y=1.02)
                out_name = os.path.join(
                    ro_dir, fname.split("\\")[-1] + "_p_dr.png")
        else:
            if artf:
                plt.title(fname.split("\\")[-1][4:] +
                          " Compiled Periodogram - Clean", fontsize=40, y=1.02)
                out_name = os.path.join(
                    ro_dir, fname.split("\\")[-1] + "_p_Clean.png")
            else:
                plt.title(fname.split("\\")[-1][4:] +
                          " Compiled Periodogram", fontsize=40, y=1.02)
                out_name = os.path.join(
                    ro_dir, fname.split("\\")[-1] + "_p.png")
        make_path_if_not_exists(out_name)
        bv_plot.savefig(fig, out_name)
        plt.close()

        if spec:
            for p, lfp_odict in enumerate(lfp_list):
                # Plot spectrograms in set of 16s
                rows, cols = [4, 4]
                gf = bv_plot.GridFig(rows, cols, wspace=0.5, hspace=0.5)

                if artf:
                    signal_used = lfp_odict.get_clean_signal()
                else:
                    signal_used = lfp_odict.get_filt_signal()
                for i, (key, lfp) in enumerate(signal_used.items()):
                    graph_data = lfp.spectrum(
                        ptype='psd', prefilt=True,
                        db=True, tr=True)   # Function from nc_lfp
                    ax = gf.get_next(along_rows=False)
                    nc_plot.lfp_spectrum_tr(graph_data, ax)
                    plt.ylim(0, 40)
                    # plt.xlim(0, 40)
                    color = gm.get_next_color()
                    ax.text(0.49, 1.08, regions[i + p * 16], fontsize=20, color=color,
                            ha='center', va='center', transform=ax.transAxes)

                if p:
                    gf.fig.suptitle(
                        (fname.split("\\")[-1][4:] + " Spectrogram " + str(p)), fontsize=30)
                    out_name = os.path.join(
                        o_dir, "Sum_ptr", fname.split("\\")[-1] + "_ptr_sum_" + str(p) + ".png")
                else:
                    gf.fig.suptitle(
                        (fname.split("\\")[-1][4:] + " Spectrogram"), fontsize=30)
                    out_name = os.path.join(
                        o_dir, "Sum_ptr", fname.split("\\")[-1] + "_ptr_sum.png")
                make_path_if_not_exists(out_name)
                bv_plot.savefig(gf.get_fig(), out_name)
                plt.close()

    # Compare periodograms for FR and FI for specific eegs
    if analysis_flags[2]:
        gm_sch = bv_plot.GroupManager(list(sch_type))

        for p, lfp_odict in enumerate(lfp_list):
            rows, cols = [4, 4]
            gf = bv_plot.GridFig(rows, cols, wspace=0.3, hspace=0.3)
            for i, (key, lfp) in enumerate(lfp_odict.get_clean_signal().items()):
                ax = gf.get_next(along_rows=False)
                legend = []
                sch_name = []
                for k, j in enumerate(blocks):
                    if k == 0:
                        new_lfp = lfp.subsample(sample_range=(0, j))
                    else:
                        new_lfp = lfp.subsample(
                            sample_range=(blocks[k - 1], j))
                    graph_data = new_lfp.spectrum(
                        ptype='psd', prefilt=False,
                        db=False, tr=False)
                    color = gm_sch.get_next_color()
                    # ax = gf.get_next(along_rows=False)
                    if sch_type[k] == 1:
                        sch_name.append("FR")
                        legend.append("{}-FR".format(k))
                    elif sch_type[k] == 0:
                        sch_name.append("FI")
                        legend.append("{}-FI".format(k))
                    nc_plot.lfp_spectrum(
                        graph_data, ax, color, style='Thin-Dashed')
                    plt.ylim(0, 0.0045)
                    plt.xlim(0, 40)
                    # plt.xlim(0, filt_top)
                # plt.tick_params(labelsize=15)
                plt.legend(legend)
                reg_color = gm.get_next_color()
                plt.title(regions[i + p * 16] + " T" + key,
                          fontsize=15, color=reg_color)
            if p:
                plt.suptitle(fname.split(
                    "\\")[-1] + " Periodogram - Blocks " + str(p), y=0.92, fontsize=30)
                if artf:
                    out_name = os.path.join(o_dir, fname.split(
                        "\\")[-1] + "_p_com_clean_" + str(p) + ".png")
                else:
                    out_name = os.path.join(o_dir, fname.split(
                        "\\")[-1] + "_p_com_" + str(p) + ".png")
            else:
                plt.suptitle(fname.split("\\")
                             [-1] + " Periodogram - Blocks", y=0.92, fontsize=30)
                if artf:
                    out_name = os.path.join(o_dir, fname.split(
                        "\\")[-1] + "_p_com.png")
                else:
                    out_name = os.path.join(o_dir, fname.split(
                        "\\")[-1] + "_p_com.png")
            make_path_if_not_exists(out_name)
            bv_plot.savefig(gf.get_fig(), out_name)
            plt.close()

    # Compare coherence in terms of freq between pairs of areas
    if analysis_flags[3]:
        # lfp_list = select_lfp(fname, ROI)
        matlab = False
        cross = False
        cohere = True

        wchans = [int(x) for x in config.get("Wavelet", "wchans").split(", ")]
        import itertools
        wave_combi = list(itertools.combinations(wchans, 2))
        for chan1, chan2 in wave_combi:
            wlet_chans = [chan1, chan2]
            reg_sel = []
            for chan in wlet_chans:  # extracts name of regions selected
                reg_sel.append(regions[chan - 1] + "-" + str(chan))

            lfp_odict = LfpODict(fname, channels=wlet_chans, filt_params=(
                filt, filt_btm, filt_top), artf_params=(artf, sd_thres, min_artf_freq, rep_freq, filt))
            legend = []
            lfp_list1, lfp_list2 = [], []
            if not Pre:
                blocks_re = []
                sch_name = []
                gm_sch = bv_plot.GroupManager(list(sch_type))
                for k, j in enumerate(blocks):
                    if k == 0:
                        new_lfp1 = lfp_odict.get_clean_signal(0).subsample(
                            sample_range=(0, j))
                        new_lfp2 = lfp_odict.get_clean_signal(1).subsample(
                            sample_range=(0, j))
                    else:
                        new_lfp1 = lfp_odict.get_clean_signal(0).subsample(
                            sample_range=(blocks[k - 1], j))
                        new_lfp2 = lfp_odict.get_clean_signal(1).subsample(
                            sample_range=(blocks[k - 1], j))
                    if sch_type[k] == 1:
                        sch_name.append("FR")
                        legend.append("{}-FR".format(k))
                    elif sch_type[k] == 0:
                        sch_name.append("FI")
                        legend.append("{}-FI".format(k))
                    lfp_list1.append(new_lfp1)
                    lfp_list2.append(new_lfp2)
            else:
                lfp_list1 = lfp_odict.get_clean_signal(0)
                lfp_list2 = lfp_odict.get_clean_signal(1)
                sch_name = ["Pre"]

            if cohere:
                an_name = 'wcohere'
                wo_dir = os.path.join(
                    # o_dir, "wcohere_T{}vsT{}".format(chan1, chan2))
                    o_dir, "{}_{}vs{}".format(an_name, reg_sel[0], reg_sel[1]))
                make_dir_if_not_exists(wo_dir)

                # Plots wavelet coherence for each block in a seperate .png
                for b, (lfp1, lfp2, sch) in enumerate(zip(lfp_list1, lfp_list2, sch_name)):
                    if artf:
                        out_name = os.path.join(
                            wo_dir, os.path.basename(fname) + "_{}_T{}-T{}_Clean_".format(an_name, chan1, chan2) + str(b + 1) + ".png")
                    else:
                        out_name = os.path.join(
                            wo_dir, os.path.basename(fname) + "_{}_T{}-T{}_".format(an_name, chan1, chan2) + str(b + 1) + ".png")
                    sch_n = str(b + 1) + "-" + sch

                    if matlab:
                        rw_ts = s.get_rw_ts()
                        bv_an.test_matlab_wcoherence(
                            lfp1, lfp2, rw_ts, sch_n, reg_sel, out_name)
                    from bvmpc.lfp_coherence import plot_wave_coherence
                    fig, ax = plt.subplots(figsize=(24, 10))
                    title = ("{} vs {} Wavelet Coherence {}".format(
                        reg_sel[0], reg_sel[1], sch_n))
                    _, result = plot_wave_coherence(
                        lfp1.get_samples(), lfp2.get_samples(), lfp1.get_timestamp(),
                        plot_arrows=True, plot_coi=False, resolution=12, title=title,
                        plot_period=False, all_arrows=False, ax=ax, quiv_x=5)

                    if behav:
                        # Plot behav timepoints
                        ax, b_legend = bv_plot.behav_vlines(
                            ax, s, behav_plot, lw=2)
                        plt.legend(handles=b_legend, fontsize=15,
                                   loc='upper right')

                    # Plot customization params
                    plt.tick_params(labelsize=20)
                    ax.xaxis.label.set_size(25)
                    ax.yaxis.label.set_size(25)

                    ax.set_title(title, fontsize=30, y=1.01)
                    bv_plot.savefig(fig, out_name[:-4] + '_pycwt.png')

            if cross:
                an_name = 'crosswave'
                wo_dir = os.path.join(
                    # o_dir, "wcohere_T{}vsT{}".format(chan1, chan2))
                    o_dir, "{}_{}vs{}".format(an_name, reg_sel[0], reg_sel[1]))
                make_dir_if_not_exists(wo_dir)

                # Plots wavelet coherence for each block in a seperate .png
                for b, (lfp1, lfp2, sch) in enumerate(zip(lfp_list1, lfp_list2, sch_name)):
                    if artf:
                        out_name = os.path.join(
                            wo_dir, os.path.basename(fname) + "_{}_T{}-T{}_Clean_".format(an_name, chan1, chan2) + str(b + 1) + ".png")
                    else:
                        out_name = os.path.join(
                            wo_dir, os.path.basename(fname) + "_{}_T{}-T{}_".format(an_name, chan1, chan2) + str(b + 1) + ".png")
                    sch_n = str(b + 1) + "-" + sch

                    fig, ax = plt.subplots(figsize=(24, 10))
                    title = ("{} vs {} Cross-Wavelet Correlation {}".format(
                        reg_sel[0], reg_sel[1], sch_n))
                    from bvmpc.lfp_coherence import plot_cross_wavelet
                    _, result = plot_cross_wavelet(
                        lfp1.get_samples(), lfp2.get_samples(), lfp1.get_timestamp(),
                        plot_coi=False, resolution=12, title=title,
                        plot_period=False, all_arrows=False, ax=ax, quiv_x=5)

                    if behav:
                        # Plot behav timepoints
                        ax, b_legend = bv_plot.behav_vlines(
                            ax, s, behav_plot, lw=2)
                        plt.legend(handles=b_legend, fontsize=15,
                                   loc='upper right')

                    # Plot customization params
                    plt.tick_params(labelsize=20)
                    ax.xaxis.label.set_size(25)
                    ax.yaxis.label.set_size(25)

                    ax.set_title(title, fontsize=30, y=1.01)
                    bv_plot.savefig(fig, out_name[:-4] + '_pycwt.png')

            # # Plots coherence by comparing FI vs FR
            # sch_f, sch_Cxy = [], []
            # for lfp1, lfp2 in zip(lfp_list1, lfp_list2):
            #     from bvmpc.lfp_coherence import calc_coherence
            #     f, Cxy = calc_coherence(lfp1, lfp2)
            #     sch_f.append(f)
            #     sch_Cxy.append(Cxy)
            # fig, ax = plt.subplots(figsize=(20, 10))
            # for f, Cxy in zip(sch_f, sch_Cxy):
            #     color = gm_sch.get_next_color()
            #     ax = plot_coherence(f, Cxy, ax=ax, color=color, legend=legend)
            # plt.xlim(0, 60)
            # plt.suptitle(fname.split("\\")
            #              [-1], y=0.95, fontsize=25)
            # plt.text(x=0.5, y=0.89, s="{}_{} Coherence - Blocks".format(reg_sel[0], reg_sel[1]),
            #          fontsize=10, ha="center", transform=fig.transFigure)
            # if artf:
            #     out_name = os.path.join(wo_dir, fname.split(
            #         "\\")[-1] + "_cohere_Clean.png")
            # else:
            #     out_name = os.path.join(wo_dir, fname.split(
            #         "\\")[-1] + "_cohere.png")
            # make_path_if_not_exists(out_name)
            # bv_plot.savefig(fig, out_name)
            # plt.close()

    # Calculate coherence and plot based on trials
    if analysis_flags[4]:
        wchans = [int(x) for x in config.get("Wavelet", "wchans").split(", ")]
        import itertools
        wave_combi = list(itertools.combinations(wchans, 2))
        for chan1, chan2 in wave_combi:
            wlet_chans = [chan1, chan2]
            reg_sel = []
            for chan in wlet_chans:  # extracts name of regions selected
                reg_sel.append(regions[chan - 1] + "-" + str(chan))
            print("Analysing coherence for {} vs {}...".format(
                reg_sel[0], reg_sel[1]))
            lfp_odict = LfpODict(fname, channels=wlet_chans, filt_params=(
                filt, filt_btm, filt_top), artf_params=(artf, sd_thres, min_artf_freq, rep_freq, filt))
            legend = []
            lfp_list1, lfp_list2 = [], []
            if Pre:
                lfp_list1 = lfp_odict.get_clean_signal(0)
                lfp_list2 = lfp_odict.get_clean_signal(1)
                sch_name = ["Pre"]
            else:
                blocks_re = []
                sch_name = []
                gm_sch = bv_plot.GroupManager(list(sch_type))
                for k, j in enumerate(blocks):
                    if k == 0:
                        blocks_re.append([0, j])
                    else:
                        blocks_re.append([blocks[k - 1], j])
                    if sch_type[k] == 1:
                        sch_name.append("FR")
                        legend.append("{}-FR".format(k))
                    elif sch_type[k] == 0:
                        sch_name.append("FI")
                        legend.append("{}-FI".format(k))
                    # lfp_list1.append(new_lfp1)
                    # lfp_list2.append(new_lfp2)

                # Description of alignment
                # 0 - Align to reward
                # 1 - Align to pellet drop
                # 2 - Align to FI
                # 3 - Align to First Response
                # 4 - Align to Double Reward
                # 5 - Align to Tone
                # if all 0, plots from start of trial

                # alignment = [0, 0, 0, 0, 0, 0]

                trial_df = s.get_valid_tdf()
                # trial_df = s.get_trial_df()

                if alignment[0]:
                    align_df = trial_df['Reward_ts']
                    align_txt = "Reward"
                    t_win = [-5, 5]  # Set time window for plotting from reward
                elif alignment[1]:
                    align_df = trial_df['Pellet_ts']
                    align_txt = "Pell"
                    t_win = [-10, 5]  # Set time window for plotting from pell
                elif alignment[2]:
                    align_df = trial_df['Reward_ts']
                    # Exclude first and last trial
                    align_df = align_df[1:-1].add(30)
                    # Set time window for plotting from interval
                    t_win = [-30, 5]
                    align_txt = "Interval"
                elif alignment[3]:
                    align_df = trial_df['First_response']
                    align_txt = "FResp"
                    t_win = [-5, 5]  # Set time window for plotting from FResp
                elif alignment[4]:
                    align_df = trial_df['D_Pellet_ts']
                    align_txt = "DPell"
                    t_win = [-30, 5]  # Set time window for plotting from dpell
                elif alignment[5]:
                    align_df = s.get_tone_starts() + 5
                    align_txt = "Tone"
                    t_win = [-10, 25]  # Set time window for plotting from tone
                else:  # Start aligned
                    align_df = trial_df['Trial_s']
                    align_txt = "Start"
                    t_win = [-5, 5]
                quiv_x = 0.5
                t_sch = trial_df['Schedule']
                trials = []

                # Generate n by 2 list of timestamps corresponding to window selected
                for t, ts in enumerate(align_df):
                    if not ts:  # To skip empty ts (eg. double pellet only)
                        continue
                    elif (ts + t_win[0]) < 0:
                        trials.append([ts[0], t_win[1]])
                        print('t_win less than trial {} start'.format(t))
                    else:
                        trials.append([ts[0] + t_win[0], ts[0] + t_win[1]])

            # Test full wcohere using axis lims to plot
            an_name = 'wcohere'
            wo_dir = os.path.join(
                # o_dir, "wcohere_T{}vsT{}".format(chan1, chan2))
                o_dir, "{}_{}vs{}".format(an_name, reg_sel[0], reg_sel[1]))
            make_dir_if_not_exists(wo_dir)

            lfp1 = lfp_odict.get_clean_signal(0)
            lfp2 = lfp_odict.get_clean_signal(1)

            from bvmpc.lfp_coherence import calc_wave_coherence, plot_wcohere, plot_arrows, zero_lag_wcohere
            import pickle

            # Pickle wcohere_results for faster performance
            overwrite_pickles = bool(
                int(config.get("Wavelet", "overwrite_pickles")))
            pickle_name = os.path.join(wo_dir, "wcohere_results.p")
            if overwrite_pickles:
                print('Delete pickle wcohere_results from', pickle_name)
                os.remove(pickle_name)
            try:
                wcohere_results = pickle.load(open(pickle_name, "rb"))
                print('Loading pickle wcohere_results from', pickle_name)

            except:
                wcohere_results = calc_wave_coherence(lfp1.get_samples(
                ), lfp2.get_samples(), lfp1.get_timestamp())
                pickle.dump(wcohere_results, open(pickle_name, "wb"))
                print('Saving pickle wcohere_results to', pickle_name)

            # Apply mask to WCT based on phase lag threshold in ms
            zlag = True
            if zlag:
                _, t, freq, _, _, aWCT = wcohere_results
                zlag_mask = zero_lag_wcohere(aWCT, freq, thres=2)
            else:
                zlag_mask = None

            # Initialize full Wavelet Coherence figure
            fig = plt.figure(figsize=(24, 15))
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(2, 2, width_ratios=[100, 1], wspace=0.1)
            r_ax = fig.add_subplot(gs[0, :-1])
            ax = fig.add_subplot(gs[1, :-1])
            cax = fig.add_subplot(gs[1, -1])

            # from bvmpc.lfp_coherence import plot_wave_coherence
            # _, result = plot_wave_coherence(
            #     lfp1.get_samples(
            #     ), lfp2.get_samples(), lfp1.get_timestamp(),
            #     plot_arrows=True, plot_coi=False, resolution=12,
            #     plot_period=False, all_arrows=False, ax=ax, quiv_x=quiv_x)

            _, wcohere_pvals = plot_wcohere(
                *wcohere_results[:3], ax=ax, mask=zlag_mask, cax=cax)

            # Plot raw signal
            sns.lineplot(lfp1.get_timestamp(),
                         lfp1.get_samples(), ci=None, ax=r_ax, label=reg_sel[0], linewidth=1)
            sns.lineplot(lfp2.get_timestamp(),
                         lfp2.get_samples(), ci=None, ax=r_ax, label=reg_sel[1], linewidth=1)

            # Plot customization params
            plt.tick_params(labelsize=15)
            ax.xaxis.label.set_size(15)
            ax.yaxis.label.set_size(15)

            if behav:
                # Plot behav timepoints
                ax, b_legend = bv_plot.behav_vlines(
                    ax, s, behav_plot, lw=2)
                plt.legend(handles=b_legend, fontsize=15,
                           loc='upper right')

            if artf:
                out_name = os.path.join(
                    wo_dir, os.path.basename(fname) + "_{}_T{}-T{}_Clean_".format(an_name, chan1, chan2))
            else:
                out_name = os.path.join(
                    wo_dir, os.path.basename(fname) + "_{}_T{}-T{}_".format(an_name, chan1, chan2))
            title = ("{} vs {} Wavelet Coherence ".format(
                reg_sel[0], reg_sel[1]))

            # Save fig for whole wcohere
            if Pre:
                plot_arrows(ax, wcohere_pvals, wcohere_results[-1], quiv_x=2.5)
                whol_name = out_name + ".png"
                bv_plot.savefig(fig, whol_name)

            p_blocks = bool(int(config.get("Wavelet", "p_blocks")))
            # Save wcohere plot in blocks
            if p_blocks:
                plot_arrows(ax, wcohere_pvals, wcohere_results[-1], quiv_x=5)
                b_out_name = os.path.join(
                    wo_dir, "Blocks", os.path.basename(out_name))

                for b, ((b_start, b_end), sch) in enumerate(zip(blocks_re, sch_name)):
                    o_name = b_out_name + str(b + 1) + ".png"
                    make_path_if_not_exists(o_name)
                    sch_n = str(b + 1) + "-" + sch
                    fig1, a1 = fig, ax
                    a1.set_xlim(b_start, b_end)
                    a1.set_title(title + sch_n, fontsize=30, y=1.01)
                    print("Saving result to {}".format(o_name))
                    fig1.savefig(o_name, dpi=150)
                    # bv_plot.savefig(fig1, o_name)
                plt.close(fig1)

            # plot mean of coherence & phase across freqs comparing schedules
            p_blocks_dist = False
            if p_blocks_dist:
                WCT, t, freq, _, _, aWCT = wcohere_results
                FR_WCT, FR_aWCT, FI_WCT, FI_aWCT = [], [], [], []

                plt.close(fig)
                for b, ((b_start, b_end), sch) in enumerate(zip(blocks_re, sch_name)):
                    start_idx = np.searchsorted(t, b_start)
                    end_idx = np.searchsorted(t, b_end)
                    curr_WCT = WCT[:, start_idx:end_idx]
                    curr_aWCT = aWCT[:, start_idx:end_idx]
                    if sch == 'FR':
                        FR_WCT.append(curr_WCT)
                        FR_aWCT.append(curr_aWCT)
                    elif sch == 'FI':
                        FI_WCT.append(curr_WCT)
                        FI_aWCT.append(curr_aWCT)
                dist_dict = {'FR_WCT': FR_WCT, 'FR_aWCT': FR_aWCT,
                             'FI_WCT': FI_WCT, 'FI_aWCT': FI_aWCT}
                for (key, val) in dist_dict.items():
                    dist_dict[key] = np.hstack((val))

                dist_fig, dist_ax = plt.subplots(
                    2, 1, figsize=(8, 8), sharex=True)
                WCT_ls, aWCT_ls = [], []
                for key, val in dist_dict.items():
                    print(key[3:])
                    # mean = np.mean(val, axis=1)
                    data = pd.DataFrame(val)
                    data['Sch'] = key[:2]
                    data['Frequency'] = freq
                    if key[3:] == 'WCT':
                        WCT_ls.append(data)
                    elif key[3:] == 'aWCT':
                        aWCT_ls.append(data)

                WCT_df = pd.melt(pd.concat(
                    WCT_ls), id_vars=['Sch', 'Frequency'])
                sns.lineplot(
                    x='Frequency', y='value', data=WCT_df, hue='Sch', ax=dist_ax[0], ci='sd')
                aWCT_df = pd.melt(pd.concat(
                    aWCT_ls), id_vars=['Sch', 'Frequency'])
                sns.lineplot(
                    x='Frequency', y='value', data=aWCT_df, hue='Sch', ax=dist_ax[1], ci='sd')
                dist_ax[0].set_title(
                    '{}vs{} Mean Coherence'.format(reg_sel[0], reg_sel[1]))
                dist_ax[1].set_title(
                    '{}vs{} Mean Phase'.format(reg_sel[0], reg_sel[1]))

                from matplotlib.ticker import ScalarFormatter
                for ax_curr in dist_ax:
                    ax_curr.set_xscale('log', basex=2)
                    ax_curr.xaxis.set_major_formatter(ScalarFormatter())
                    ax_curr.set_xticks([64, 32, 16, 8, 4, 2, 1])

                dist_ax[0].set_ylim([0, 1])
                dist_ax[0].set_ylabel('Coherence')
                dist_ax[1].set_ylim([-np.pi, np.pi])
                dist_ax[1].set_ylabel('Phase Angle (rads)')
                o_name = out_name + "_dist.png"
                make_path_if_not_exists(o_name)

                print("Saving result to {}".format(o_name))
                dist_fig.savefig(o_name, dpi=150)

            p_trials = bool(int(config.get("Wavelet", "p_trials")))
            # Save wcohere plot in trials
            if p_trials:
                plot_arrows(ax, wcohere_pvals,
                            wcohere_results[-1], quiv_x=0.3)
                tr_out_name = os.path.join(
                    wo_dir, "Trials", os.path.basename(out_name))
                plot_df = s.get_trial_df()
                for t, (t_start, t_end, sch) in enumerate(zip(plot_df['Trial_s'], plot_df['Pellet_ts'], plot_df['Schedule'])):
                    fig1, a1, a2 = fig, ax, r_ax
                    # Standardize window of trial displayed with reference to pell_ts
                    win = [-20, 10]
                    a1.set_xlim([t_end + win[0], t_end + win[1]])
                    a2.set_xlim([t_end + win[0], t_end + win[1]])
                    name = '{}_Tr{}.png'.format(
                        tr_out_name[:-4], t + 1)
                    make_path_if_not_exists(name)
                    a1.set_title("{}Tr{} {}".format(title, str(t), sch),
                                 fontsize=20, y=1.01)
                    a2.set_title("Raw Trace Tr{} {}".format(str(t), sch),
                                 fontsize=20, y=1.01)
                    print("Saving result to {}".format(name))
                    fig1.savefig(name, dpi=150)
                    # bv_plot.savefig(fig1, name)
                plt.close(fig1)

            # Trial based wcohere zoomed to +- 5s from pellet
            p_trials_pell = False
            if p_trials_pell:
                plot_arrows(ax, wcohere_pvals,
                            wcohere_results[-1], quiv_x=0.1)
                tr_out_name = os.path.join(
                    wo_dir, "Trials_Pell", os.path.basename(out_name))
                plot_df = s.get_trial_df()
                for t, (t_start, t_end, sch) in enumerate(zip(plot_df['Trial_s'], plot_df['Pellet_ts'], plot_df['Schedule'])):
                    fig1, a1, a2 = fig, ax, r_ax
                    # Standardize window of trial displayed with reference to pell_ts
                    win = [-5, 5]
                    # win = [-20, 10]
                    a1.set_xlim([t_end + win[0], t_end + win[1]])
                    a2.set_xlim([t_end + win[0], t_end + win[1]])
                    name = '{}_Tr{}.png'.format(
                        tr_out_name[:-4], t + 1)
                    make_path_if_not_exists(name)
                    a1.set_title("{}Tr{} {}".format(title, str(t), sch),
                                 fontsize=30, y=1.01)
                    a2.set_title("Raw Trace",
                                 fontsize=30, y=1.01)
                    print("Saving result to {}".format(name))
                    fig1.savefig(name, dpi=150)
                    # bv_plot.savefig(fig1, name)
                plt.close(fig1)

            p_wcohere_mean = bool(int(config.get("Wavelet", "p_wcohere_mean")))
            split_sch = bool(int(config.get("Wavelet", "split_sch")))
            # Plot average coherence across t_blocks
            if p_wcohere_mean:
                t_block_list, t_block_sch, fr_blocks, fi_blocks = [], [], [], []
                if split_sch:
                    for i, (sch, block) in enumerate(zip(t_sch, trials)):
                        if sch == 'FR':
                            fr_blocks.append(block)
                        elif sch == 'FI':
                            fi_blocks.append(block)
                    if not len(fr_blocks) == 0:
                        t_block_list.append(fr_blocks)
                        t_block_sch.append("FR")
                    if not len(fi_blocks) == 0:
                        t_block_list.append(fi_blocks)
                        t_block_sch.append("FI")
                else:
                    t_block_list.append(trials)
                    t_block_sch.append("")

                for i, (trials, b_sch) in enumerate(zip(t_block_list, t_block_sch)):
                    if split_sch:
                        sch_print = "_{}_{}".format(align_txt, b_sch)
                    else:
                        sch_print = "_{}".format(align_txt)
                    o_name = out_name + "mean{}.png".format(sch_print)
                    fig, ax = plt.subplots(figsize=(24, 10))
                    from bvmpc.lfp_coherence import wcohere_mean
                    mean_WCT, norm_u, norm_v, magnitute = wcohere_mean(
                        wcohere_results[0], wcohere_results[-1], t_blocks=trials)

                    # for i, x in enumerate(magnitute):
                    #     fig3 = sns.distplot(x, hist=False, rug=True, label=i)

                    _, wcohere_pvals = plot_wcohere(mean_WCT, np.arange(
                        t_win[0], t_win[1], 1 / 250.0), wcohere_results[2], ax=ax)
                    plot_arrows(ax, wcohere_pvals, u=norm_u,
                                v=norm_v, magnitute=magnitute, quiv_x=quiv_x)
                    ax.axvline(0, linestyle='-',
                               color='w', linewidth=1)
                    plt.text(-0.1, 0, align_txt, rotation=90,
                             color='w', va='bottom', ha='right')
                    plt.text(0.1, 0, "n = " + str(len(trials)), rotation=90,
                             color='w', va='bottom', ha='left')
                    ax.set_title("{}Mean{}".format(title, sch_print),
                                 fontsize=30, y=1.01)

                    bv_plot.savefig(fig, o_name)

            target_freq = int(config.get("Wavelet", "target_freq"))
            if target_freq != 0:
                plot = True
                dist = True
                # Single frequency extraction of wcohere
                from bvmpc.lfp_coherence import plot_single_freq_wcohere
                t_WCT_df, tf_fig, tf_fig2 = plot_single_freq_wcohere(
                    target_freq, *wcohere_results[:3], wcohere_results[-1], trials, t_win, trial_df, align_txt, s, reg_sel, plot=plot, sort=False, dist=dist)
                tfreq_out_name = os.path.join(
                    wo_dir, "{}Hz".format(target_freq), os.path.basename(out_name))

                if tf_fig is not None:
                    o_name = tfreq_out_name + \
                        "{}Hz_{}.png".format(target_freq, align_txt)
                    make_path_if_not_exists(o_name)
                    bv_plot.savefig(tf_fig, o_name)
                if tf_fig2 is not None:
                    o_name = tfreq_out_name + \
                        "{}Hz_{}_dist.png".format(target_freq, align_txt)
                    make_path_if_not_exists(o_name)
                    bv_plot.savefig(tf_fig2, o_name)


def main_entry(config_name, base_dir=None, dummy=False, interactive=False):
    """Batch files to use based on the config file passed."""
    here = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(config_name):
        config_path = os.path.join(here, "Configs", "LFP", config_name)
    else:
        config_path = config_name
    config = read_cfg(config_path)

    in_dir = config.get("Setup", "in_dir")
    if in_dir[0] == "\"":
        in_dir = in_dir[1:-1]
    out_main_dir = config.get("Setup", "out_dir")
    if out_main_dir == "":
        out_main_dir = in_dir
    if base_dir is not None:
        in_dir = base_dir + in_dir[len(base_dir):]
        out_main_dir = base_dir + out_main_dir[len(out_main_dir):]

    if interactive:
        print("Starting interactive regex console:")
        regex_filter = interactive_refilt(
            in_dir, ext=".eeg", write=True, write_loc=config_path)
    else:
        regex_filter = config.get("Setup", "regex_filter")
        regex_filter = None if regex_filter == "None" else regex_filter

    print("Automatically obtaining {} files from {} matching {}".format(
        ".eeg", in_dir, regex_filter))
    filenames = get_all_files_in_dir(
        in_dir, ext=".eeg", recursive=True,
        verbose=True, re_filter=regex_filter)

    filenames = [fname[:-4] for fname in filenames]
    if len(filenames) == 0:
        print("No set files found for analysis!")
        exit(-1)

    for fname in filenames:
        if dummy:
            print("Would run on {}".format(fname))
        else:
            print("Running on {}".format(fname))
            out_main_dir = os.path.dirname(fname)
            main(fname, out_main_dir, config)


if __name__ == "__main__":
    """Setup which config file(s) to use and run - see main_entry."""
    # Use this to do batching ATM
    # for i in np.arange(3, 7):
    #     config_name = "CAR-SA{}.cfg".format(i)
    #     main_entry(config_name)

    # Use this for single runs without providing cmd line arg
    # config_name = "Eoin.cfg"
    # config_name = "CAR-SA2.cfg"
    # config_name = "Batch_3.cfg"
    i = 5
    config_name = "CAR-SA{}.cfg".format(i)

    # Can parse command line args
    parser = argparse.ArgumentParser(
        description="Arguments passable via command line:")
    parser.add_argument(
        "-c", "--config", type=str, default=config_name, required=False,
        help="The name of the config file in configs/LFP OR path to cfg.")
    parser.add_argument(
        "-d", "--dummy", action="store_true", required=False,
        help="If this flag is present, only prints the files that would be run."
    )
    parser.add_argument(
        "-b", "--basedir", type=str, default=None, required=False,
        help="Replace the base directory in config file with this.")
    parser.add_argument(
        "-i", "--interactive", action="store_true", required=False,
        help="Run an interactive console to select a regex filter set.")
    parsed = parse_args(parser, verbose=True)

    # Actually run here
    main_entry(
        parsed.config, base_dir=parsed.basedir, dummy=parsed.dummy,
        interactive=parsed.interactive)
