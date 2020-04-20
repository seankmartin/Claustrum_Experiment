import os
import math
import json
from datetime import date, timedelta, datetime

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import interpolate, signal

import neurochat.nc_plot as nc_plot
import bvmpc.bv_plot as bv_plot
import bvmpc.bv_analyse as bv_an
from bvmpc.bv_utils import make_dir_if_not_exists, print_h5, mycolors
from bvmpc.bv_utils import daterange, split_list, get_all_files_in_dir
from bvmpc.bv_utils import log_exception, chunks, save_dict_to_csv
from bvmpc.bv_utils import make_path_if_not_exists, read_cfg, parse_args
from bvmpc.bv_utils import get_dist
from bvmpc.bv_session_extractor import SessionExtractor
from bvmpc.bv_session import Session
from bvmpc.lfp_odict import LfpODict
from bvmpc.compare_lfp import compare_lfp
from bvmpc.lfp_plot import plot_lfp, plot_coherence, lfp_csv
from bvmpc.bv_file import load_bv_from_set, select_lfp


def main(fname, out_main_dir, config):
    '''
    Parameters
    ----------
    fname : str
        filenames to be analysed

    Saves plots in a !LFP folder inside out_main_dir

    '''
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

    artf = bool(int(config.get("Artefact Params", "artf")))
    sd_thres = float(config.get("Artefact Params", "sd_thres"))
    min_artf_freq = float(config.get("Artefact Params", "min_artf_freq"))
    rep_freq = config.get("Artefact Params", "rep_freq")
    if rep_freq == "":
        rep_freq = None
    else:
        rep_freq = float(config.get("Artefact Params", "rep_freq"))

    gm = bv_plot.GroupManager(regions)

    # Split the LFP signals into chunked sets of size 16
    lfp_list = []
    for chans in chunks(chans, 16):
        lfp_odict = LfpODict(
            fname, channels=chans,
            filt_params=(filt, filt_btm, filt_top),
            artf_params=(artf, sd_thres, min_artf_freq, rep_freq, filt))
        lfp_list.append(lfp_odict)

    if "Pre" in fname:
        behav = False
        Pre = True

    else:
        Pre = False
        behav = bool(int(config.get("Behav Params", "behav")))
        behav_plot = json.loads(config.get("Behav Params", "behav_plot"))

        # Load behaviour-related data
        s = load_bv_from_set(fname)
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
                o_dir, os.path.basename(fname) + "_bv_raster{}.png".format(align_txt))
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
    r_csv = bool(int(config.get("Setup", "r_csv")))

    # Differential Recording mode (lfp1 - lfp2 in same shuttle)
    DR = bool(int(config.get("Setup", "DR")))

    for p, lfp_odict in enumerate(lfp_list):
        if r_plot:
            ro_dir = os.path.join(o_dir, "Raw")
            make_dir_if_not_exists(ro_dir)
            # Plot raw LFP for all tetrodes in segments
            if s:
                splits = np.concatenate([[0], s.get_block_ends()])
            else:
                splits = None
            plot_lfp(ro_dir, lfp_odict, splits=splits,
                     sd=sd_thres, filt=filt, artf=artf, session=s)
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
            if artf:
                signal_used = lfp_odict.get_clean_signal()
            else:
                signal_used = lfp_odict.get_filt_signal()

            for i, (key, lfp) in enumerate(signal_used.items()):
                color = gm.get_next_color()
                if DR:  # Hardfix for DR mode
                    if i % 2 == 1:
                        lfp._set_samples(
                            lfp.get_samples() - signal_used[str(i)].get_samples())
                        graph_data = lfp.spectrum(
                            ptype='psd', prefilt=False,
                            db=False, tr=False)
                        nc_plot.lfp_spectrum(
                            graph_data, ax, color, style="Dashed")
                        legend.append(regions[i + p * 16] + " T" + key)
                        cur_max_p = max(graph_data['Pxx'])
                        if cur_max_p > max_p:
                            max_p = cur_max_p
                        else:
                            continue
                else:
                    graph_data = lfp.spectrum(
                        ptype='psd', prefilt=False,
                        db=False, tr=False)
                    nc_plot.lfp_spectrum(
                        graph_data, ax, color, style="Dashed")
                    legend.append(regions[i + p * 16] + " T" + key)
                    cur_max_p = max(graph_data['Pxx'])
                    if cur_max_p > max_p:
                        max_p = cur_max_p
                    else:
                        continue
        plt.tick_params(labelsize=20)
        ax.xaxis.label.set_size(25)
        ax.yaxis.label.set_size(25)
        plt.ylim(0, max_p + max_p * 0.1)
        plt.xlim(0, filt_top)
        plt.legend(legend, fontsize=15)

        if DR:  # Hard fix for naming if Differential recording is used
            if artf:
                plt.title(fname.split("\\")[-1][4:] +
                          " Compiled Periodogram - Clean_dr", fontsize=40, y=1.02)
                out_name = os.path.join(
                    o_dir, fname.split("\\")[-1] + "_p_Clear_dr.png")
            else:
                plt.title(fname.split("\\")[-1][4:] +
                          " Compiled Periodogram_dr", fontsize=40, y=1.02)
                out_name = os.path.join(
                    o_dir, fname.split("\\")[-1] + "_p_dr.png")
        else:
            if artf:
                plt.title(fname.split("\\")[-1][4:] +
                          " Compiled Periodogram - Clean", fontsize=40, y=1.02)
                out_name = os.path.join(
                    o_dir, fname.split("\\")[-1] + "_p_Clear.png")
            else:
                plt.title(fname.split("\\")[-1][4:] +
                          " Compiled Periodogram", fontsize=40, y=1.02)
                out_name = os.path.join(
                    o_dir, fname.split("\\")[-1] + "_p.png")
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
            if not Pre:
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
            else:
                lfp_list1 = lfp_odict.get_clean_signal(0)
                lfp_list2 = lfp_odict.get_clean_signal(1)
                sch_name = ["Pre"]

            # Test full wcohere using axis lims to plot
            an_name = 'wcohere'
            wo_dir = os.path.join(
                # o_dir, "wcohere_T{}vsT{}".format(chan1, chan2))
                o_dir, "{}_{}vs{}".format(an_name, reg_sel[0], reg_sel[1]))
            make_dir_if_not_exists(wo_dir)

            lfp1 = lfp_odict.get_clean_signal(0)
            lfp2 = lfp_odict.get_clean_signal(1)

            from bvmpc.lfp_coherence import calc_wave_coherence, plot_wcohere, plot_arrows
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
                t_win = [-30, 5]  # Set time window for plotting from interval
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

            for t, ts in enumerate(align_df):
                if not ts:  # To skip empty ts (eg. double pellet only)
                    continue
                elif (ts+t_win[0]) < 0:
                    trials.append([ts[0], t_win[1]])
                    print('t_win less than trial {} start'.format(t))
                else:
                    trials.append([ts[0] + t_win[0], ts[0] + t_win[1]])

            # trials = [[0, 60], [60, 120]]

            # Initialize full Wavelet Coherence figure
            fig, ax = plt.subplots(figsize=(24, 10))

            # from bvmpc.lfp_coherence import plot_wave_coherence
            # _, result = plot_wave_coherence(
            #     lfp1.get_samples(
            #     ), lfp2.get_samples(), lfp1.get_timestamp(),
            #     plot_arrows=True, plot_coi=False, resolution=12,
            #     plot_period=False, all_arrows=False, ax=ax, quiv_x=quiv_x)

            _, wcohere_pvals = plot_wcohere(*wcohere_results[:3], ax=ax)

            # Plot customization params
            plt.tick_params(labelsize=20)
            ax.xaxis.label.set_size(25)
            ax.yaxis.label.set_size(25)

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

            p_blocks = bool(int(config.get("Wavelet", "p_blocks")))
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

            p_trials = bool(int(config.get("Wavelet", "p_trials")))
            if p_trials:
                plot_arrows(ax, wcohere_pvals,
                            wcohere_results[-1], quiv_x=0.5)
                tr_out_name = os.path.join(
                    wo_dir, "Trials", os.path.basename(out_name))
                plot_df = s.get_trial_df()
                print(plot_df)
                for t, (t_start, t_end, sch) in enumerate(zip(plot_df['Trial_s'], plot_df['Pellet_ts'], plot_df['Schedule'])):
                    fig1, a1 = fig, ax
                    # Standardize window of trial displayed with reference to pell_ts
                    win = [-20, 10]
                    a1.set_xlim([t_end+win[0], t_end+win[1]])
                    name = '{}_Tr{}.png'.format(
                        tr_out_name[:-4], t + 1)
                    make_path_if_not_exists(name)
                    a1.set_title("{}Tr{} {}".format(title, str(t), sch),
                                 fontsize=30, y=1.01)
                    print("Saving result to {}".format(name))
                    fig1.savefig(name, dpi=150)
                    # bv_plot.savefig(fig1, name)
                plt.close(fig1)

            p_wcohere_mean = bool(int(config.get("Wavelet", "p_wcohere_mean")))
            split_sch = bool(int(config.get("Wavelet", "split_sch")))
            if p_wcohere_mean:  # Plot average coherence across t_blocks
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
                # Single frequency extraction of wcohere
                from bvmpc.lfp_coherence import plot_single_freq_wcohere
                t_WCT_df, tf_fig = plot_single_freq_wcohere(
                    target_freq, *wcohere_results[:3], wcohere_results[-1], trials, t_win, trial_df, align_txt, s, reg_sel, plot=plot)
                if tf_fig is not None:
                    o_name = out_name + \
                        "{}Hz_{}.png".format(target_freq, align_txt)
                    bv_plot.savefig(tf_fig, o_name)


def main_entry(config_name):
    here = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(here, "Configs", "LFP", config_name)
    config = read_cfg(config_path)

    in_dir = config.get("Setup", "in_dir")
    if in_dir[0] == "\"":
        in_dir = in_dir[1:-1]
    out_main_dir = config.get("Setup", "out_dir")
    if out_main_dir == "":
        out_main_dir = in_dir
    regex_filter = config.get("Setup", "regex_filter")
    regex_filter = None if regex_filter == "None" else regex_filter
    filenames = get_all_files_in_dir(
        in_dir, ext=".eeg", recursive=True,
        verbose=True, re_filter=regex_filter)

    filenames = [fname[:-4] for fname in filenames]
    if len(filenames) == 0:
        print("No set files found for analysis!")
        exit(-1)

    for fname in filenames:
        main(fname, out_main_dir, config)


if __name__ == "__main__":
    # config_name = "Eoin.cfg"
    # config_name = "CAR-SA2.cfg"
    config_name = "Batch_3.cfg"
    main_entry(config_name)
