import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from bvmpc.bv_session_extractor import SessionExtractor
from bvmpc.bv_session import Session
import bvmpc.bv_analyse as bv_an
from bvmpc.bv_utils import make_dir_if_not_exists, print_h5, mycolors, daterange, split_list, get_all_files_in_dir, log_exception, chunks, save_dict_to_csv, make_path_if_not_exists, read_cfg, parse_args
import bvmpc.bv_plot as bv_plot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import interpolate, signal
from datetime import date, timedelta, datetime

from bvmpc.lfp_odict import LfpODict
import neurochat.nc_plot as nc_plot

from bvmpc.lfp_plot import plot_lfp, plot_coherence


def main(fname, analysis_flags, o_main_dir=None, alignment=None):
    '''
    Parameters
    ----------
    fname : str
        filenames to be analysed

    analysis_flags : bool, optional. Defaults to True.
        Sets analysis to be used.
        0 - plot periodograms and ptrs in seperate plots for each tetrode
        1 - plot graphs from all tetrodes in 1 .png

    alignment : bool, optional. Defaults to None #TODO write alignment function
        Sets alignment points to be used.
        0 - Align to reward
        1 - Align to pellet drop
        2 - Align to FI
        3 - Align to Tone

    o_main_dir: dir, optional. Defaults to None.
        None - Saves plots in a LFP folder where .eeg was found
        Else - Saves plots in a LFP folder of given drive
    '''

    # Setup Region info for eeg
    # # Axona single screw Drive settings
    # chans = [i for i in range(1, 17*2-1)]
    # regions = ["CLA"] * 28 + ["ACC"] * 2 + ["RSC"] * 2

    # Single Hemi Multisite Drive settings
    chans = [i for i in range(1, 17)]
    regions = ["CLA"] * 8 + ["ACC"] * 4 + ["RSC"] * 4

    gm = bv_plot.GroupManager(regions)

    # Change filt values here. Default order 10.
    filt_btm = 1.0
    filt_top = 50

    lfp_list = []
    for chans in chunks(chans, 16):
        lfp_odict = LfpODict(
            fname, channels=chans, filt_params=(True, filt_btm, filt_top))
        lfp_list.append(lfp_odict)

    # Load behaviour-related data
    s = load_bv_from_set(fname)
    rw_ts = s.get_rw_ts()
    sch_type = s.get_arrays('Trial Type')  # FR = 1, FI = 0

    if o_main_dir is None:
        o_dir = os.path.join(
            os.path.dirname(fname), "!LFP")
    else:
        o_dir = os.path.join(o_main_dir, "!LFP")
    make_dir_if_not_exists(o_dir)

    if analysis_flags[0]:   # Plot periodograms and ptr for each tetrode seperately
        """
        Plot periodograms and ptr for each tetrode seperately
            ptr - includes vlines indicating tone and reward points

        """
        for p, lfp_odict in enumerate(lfp_list):
            # # Plot periodogram for each eeg as a seperate .png
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
            #     fig.savefig(out_name)
            #     plt.close()

            # Setup 4x4 summary grid
            rows, cols = [4, 4]
            gf = bv_plot.GridFig(rows, cols, wspace=0.3,
                                 hspace=0.3, tight_layout=False)

            # Plot individual periodograms on a 4x4 grid
            for i, (key, lfp) in enumerate(lfp_odict.get_filt_signal().items()):
                graph_data = lfp.spectrum(
                    ptype='psd', prefilt=False,
                    db=False, tr=False)
                ax = gf.get_next(along_rows=False)
                color = gm.get_next_color()
                nc_plot.lfp_spectrum(graph_data, ax, color)
                plt.ylim(0, 0.015)
                # plt.xlim(0, 40)
                ax.text(0.49, 1.08, regions[i+p*16], fontsize=20,
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            if p:
                gf.fig.suptitle(
                    (fname.split("\\")[-1][4:] + " Periodogram " + str(p)), fontsize=30)
                out_name = os.path.join(
                    o_dir, fname.split("\\")[-1] + "_p_sum_" + str(p) + ".png")
            else:
                gf.fig.suptitle(
                    (fname.split("\\")[-1][4:] + " Periodogram"), fontsize=30)
                out_name = os.path.join(
                    o_dir, fname.split("\\")[-1] + "_p_sum.png")
            make_path_if_not_exists(out_name)
            print("Saving result to {}".format(out_name))
            gf.fig.savefig(out_name)
            plt.close()

           # Plot spectrogram for each eeg as a seperate .png
            for i, (key, lfp) in enumerate(lfp_odict.get_filt_signal().items()):
                graph_data = lfp.spectrum(
                    ptype='psd', prefilt=False,
                    db=True, tr=True)
                if graph_data['t'][-1] > 305:
                    block_size = 305
                    rows, cols = [6, 1]
                    gf = bv_plot.GridFig(rows, cols, wspace=0.3,
                                         hspace=0.3, size_multiplier_x=40, tight_layout=False)
                    for j in range(0, block_size*6, block_size):
                        tone_ts = range(j+5, j+305, 300)
                        ax = gf.get_next(along_rows=True)
                        new_lfp = lfp.subsample(sample_range=(j, j+block_size))
                        graph_data = new_lfp.spectrum(
                            ptype='psd', prefilt=False,
                            db=True, tr=True)
                        nc_plot.lfp_spectrum_tr(graph_data, ax)
                        if j == 0:
                            plt.title("T" + key + " " +
                                      regions[i+p*16] + " Spectrogram", fontsize=40)
                        plt.ylim(0, filt_top)
                        for rw in rw_ts:
                            ax.axvline(rw, linestyle='-',
                                       color='orange', linewidth='1')    # vline demarcating reward point/end of trial
                        ax.axvline(tone_ts, linestyle='-',
                                   color='r', linewidth='1')    # vline demarcating end of tone
                    fig = gf.get_fig()
                else:
                    fig, ax = plt.subplots(figsize=(20, 5))
                    nc_plot.lfp_spectrum_tr(graph_data, ax)
                    plt.ylim(0, filt_top)
                    fig.suptitle("T" + key + " " +
                                 regions[i+p*16] + " Spectrogram")
                out_name = os.path.join(o_dir, "ptr", key + "ptr.png")
                make_path_if_not_exists(out_name)
                print("Saving result to {}".format(out_name))
                fig.savefig(out_name)
                plt.close()

            # plot_lfp(o_dir, lfp_odict.get_filt_signal(), segment_length=60)   # Plot raw LFP for all tetrodes in segments

    if analysis_flags[1]:   # Complie graphs per session in a single .png
        # Plot all periodograms on 1 plot
        fig, ax = plt.subplots(figsize=(20, 20))
        legend = []
        for p, lfp_odict in enumerate(lfp_list):
            for i, (key, lfp) in enumerate(lfp_odict.get_filt_signal().items()):
                graph_data = lfp.spectrum(
                    ptype='psd', prefilt=False,
                    db=False, tr=False)
                color = gm.get_next_color()
                nc_plot.lfp_spectrum(graph_data, ax, color)
                legend.append(regions[i+p*16] + " T" + key)
        plt.ylim(0, 0.015)
        plt.xlim(0, filt_top)
        plt.legend(legend)
        plt.title(fname.split("\\")[-1][4:] +
                  " Compiled Periodogram", fontsize=25)
        out_name = os.path.join(o_dir, fname.split("\\")[-1] + "_p.png")
        make_path_if_not_exists(out_name)
        fig.savefig(out_name)
        plt.close()

        for p, lfp_odict in enumerate(lfp_list):
            # Plot spectrograms in set of 16s
            rows, cols = [4, 4]
            gf = bv_plot.GridFig(rows, cols, wspace=0.5, hspace=0.5)
            for i, (key, lfp) in enumerate(lfp_odict.get_filt_signal().items()):
                graph_data = lfp.spectrum(
                    ptype='psd', prefilt=True,
                    db=True, tr=True)
                ax = gf.get_next(along_rows=False)
                nc_plot.lfp_spectrum_tr(graph_data, ax)
                plt.ylim(0, 40)
                # plt.xlim(0, 40)
                color = gm.get_next_color()
                ax.text(0.49, 1.08, regions[i+p*16], fontsize=20, color=color,
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

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
            print("Saving result to {}".format(out_name))
            gf.fig.savefig(out_name)
            plt.close()

    if analysis_flags[2]:    # Compare periodograms for FR and FI for specific eegs
        gm_sch = bv_plot.GroupManager(list(sch_type))

        for p, lfp_odict in enumerate(lfp_list):
            rows, cols = [4, 4]
            gf = bv_plot.GridFig(rows, cols, wspace=0.5, hspace=0.5)
            for i, (key, lfp) in enumerate(lfp_odict.get_filt_signal().items()):
                ax = gf.get_next(along_rows=False)
                legend = []
                block_size = 305
                sch_name = []
                for k, j in enumerate(range(0, block_size*6, block_size)):
                    new_lfp = lfp.subsample(sample_range=(j, j+block_size))
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
                    nc_plot.lfp_spectrum(graph_data, ax, color)
                    plt.ylim(0, 0.010)
                    plt.xlim(0, filt_top)
                plt.legend(legend)
                plt.title(regions[i+p*16] + " T" + key, fontsize=12)
            if p:
                plt.suptitle(fname.split(
                    "\\")[-1] + " Periodogram - Blocks " + str(p), y=0.95, fontsize=25)
                out_name = os.path.join(o_dir, fname.split(
                    "\\")[-1] + "_p_com_" + str(p) + ".png")
            else:
                plt.suptitle(fname.split("\\")
                             [-1] + " Periodogram - Blocks", y=0.95, fontsize=25)
                out_name = os.path.join(o_dir, fname.split(
                    "\\")[-1] + "_p_com.png")
            make_path_if_not_exists(out_name)
            print("Saving result to {}".format(out_name))
            gf.fig.savefig(out_name)
            plt.close()

    if analysis_flags[3]:   # Compare coherence in terms of freq between ACC & RSC
        # lfp_list = select_lfp(fname, ROI)
        chans = [30, 31]
        gm_sch = bv_plot.GroupManager(list(sch_type))

        lfp_odict = LfpODict(fname, channels=chans)
        legend = []
        sch_name = []
        block_size = 305
        lfp_list1, lfp_list2 = [], []
        for k, j in enumerate(range(0, block_size*6, block_size)):
            new_lfp1 = lfp_odict.get_signal(0).subsample(
                sample_range=(j, j+block_size))
            new_lfp2 = lfp_odict.get_signal(1).subsample(
                sample_range=(j, j+block_size))
            if sch_type[k] == 1:
                sch_name.append("FR")
                legend.append("{}-FR".format(k))
            elif sch_type[k] == 0:
                sch_name.append("FI")
                legend.append("{}-FI".format(k))
            lfp_list1.append(new_lfp1)
            lfp_list2.append(new_lfp2)

        sch_f, sch_Cxy = [], []
        for lfp1, lfp2 in zip(lfp_list1, lfp_list2):
            from bvmpc.lfp_coherence import calc_coherence
            f, Cxy = calc_coherence(lfp1, lfp2)
            sch_f.append(f)
            sch_Cxy.append(Cxy)
        fig, ax = plt.subplots(figsize=(1, 1))
        for f, Cxy in zip(sch_f, sch_Cxy):
            color = gm_sch.get_next_color()
            ax = plot_coherence(f, Cxy, ax=ax, color=color, legend=legend)
        plt.show()


def select_lfp(fname, ROI):  # Select lfp based on region
    # Single Hemi Multisite Drive settings
    lfp_list = []
    chans = [i for i in range(1, 17)]
    regions = ["CLA"] * 8 + ["ACC"] * 4 + ["RSC"] * 4

    # Actual function
    for r in ROI:
        idx = [i for i, x in enumerate(regions) if x == ROI[r]]
        lfp_odict = LfpODict(
            fname, channels=chans[idx], filt_params=(True, filt_btm, filt_top))
        lfp_list.append(lfp_odict)
    return lfp_list


def load_bv_from_set(fname):
    """ Loads session based from .inp """
    return Session(axona_file=fname + ".inp")


if __name__ == "__main__":
    in_dir = r"F:\Ham Data\A1_CAR-R1\CAR-R1_20191126"
    # in_dir = r"F:\Ham Data\A9_CAR-SA1\CAR-SA1_20191215"
    # in_dir = r"F:\Ham Data\A10_CAR-SA2\CAR-SA2_20200110"

    filenames = get_all_files_in_dir(
        in_dir, ext=".eeg", recursive=True,
        verbose=True, re_filter="CAR-R1")

    filenames = [fname[:-4] for fname in filenames]
    if len(filenames) == 0:
        print("No set files found for analysis!")
        exit(-1)

    # Description of analysis_flags
    # 0 - plot periodograms and ptrs in seperate plots for each tetrode
    # 1 - plot graphs from all tetrodes in a single .png
    # 2 - Compare periodograms for FR and FI for specific eegs
    # 3 - Compare coherence in terms of freq between ACC & RSC

    analysis_flags = [0, 0, 0, 1]

    # Description of alignment
    # 0 - Align to reward
    # 1 - Align to pellet drop
    # 2 - Align to FI
    # 3 - Align to Tone

    aligment = [0, 0, 0, 0]

    for fname in filenames:
        main(fname, analysis_flags, in_dir)
