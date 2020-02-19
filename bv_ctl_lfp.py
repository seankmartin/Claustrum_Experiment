import os
import math
import json
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

from bvmpc.lfp_plot import plot_lfp, plot_coherence, lfp_csv


def main(fname, out_main_dir, config):
    '''
    Parameters
    ----------
    fname : str
        filenames to be analysed

    Saves plots in a !LFP folder inside out_main_dir
    '''

    analysis_flags = json.loads(config.get("Setup", "analysis_flags"))
    alignment = json.loads(config.get("Setup", "alignment"))
    chan_amount = int(config.get("Setup", "chans"))
    chans = [i for i in range(1, chan_amount+1)]
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

    lfp_list = []
    for chans in chunks(chans, 16):
        lfp_odict = LfpODict(fname, channels=chans, filt_params=(
            filt, filt_btm, filt_top), artf_params=(artf, sd_thres, min_artf_freq, rep_freq, filt))
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

    o_dir = os.path.join(out_main_dir, "!LFP")
    make_dir_if_not_exists(o_dir)

    # Plots raw LFP for all tetrodes or output csv with artf_removal results
    r_plot = bool(int(config.get("Setup", "r_plot")))
    r_csv = bool(int(config.get("Setup", "r_csv")))

    for p, lfp_odict in enumerate(lfp_list):
        if r_plot:
            ro_dir = os.path.join(o_dir, "Raw")
            make_dir_if_not_exists(ro_dir)
            # Plot raw LFP for all tetrodes in segments
            plot_lfp(ro_dir, lfp_odict, segment_length=305,
                     sd=sd_thres, filt=filt, artf=artf, session=s)
        if r_csv:
            shut_s, shut_end = p*16, 16+p*16
            lfp_csv(fname, o_dir, lfp_odict, sd_thres,
                    min_artf_freq, shuttles[shut_s:shut_end], filt)

    if analysis_flags[0]:   # Plot periodograms and ptr for each tetrode seperately
        """
        Plot periodograms and ptr for each tetrode seperately
            ptr - includes vlines indicating tone and reward points

        """
        for p, lfp_odict in enumerate(lfp_list):
            indiv = False   # Set to true for individual periodograms on a 4x4 grid
            spec = False    # Set to true for individual spectrograms per .png

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
            #     fig.savefig(out_name)
            #     plt.close()

            if indiv:
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

            if spec:
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
                            new_lfp = lfp.subsample(
                                sample_range=(j, j+block_size))
                            graph_data = new_lfp.spectrum(
                                ptype='psd', prefilt=False,
                                db=True, tr=True)
                            nc_plot.lfp_spectrum_tr(graph_data, ax)
                            plt.tick_params(labelsize=20)
                            ax.xaxis.label.set_size(25)
                            ax.yaxis.label.set_size(25)
                            if j == 0:
                                plt.title("T" + key + " " +
                                          regions[i+p*16] + " Spectrogram", fontsize=40, y=1.05)
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
                                     regions[i+p*16] + " Spectrogram")
                    out_name = os.path.join(o_dir, "ptr", key + "ptr.png")
                    make_path_if_not_exists(out_name)
                    print("Saving result to {}".format(out_name))
                    fig.savefig(out_name)
                    plt.close()

    if analysis_flags[1]:   # Complie graphs per session in a single .png
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
                graph_data = lfp.spectrum(
                    ptype='psd', prefilt=False,
                    db=False, tr=False)
                color = gm.get_next_color()
                nc_plot.lfp_spectrum(graph_data, ax, color)
                legend.append(regions[i+p*16] + " T" + key)
                cur_max_p = max(graph_data['Pxx'])
                if cur_max_p > max_p:
                    max_p = cur_max_p
                else:
                    continue
        plt.tick_params(labelsize=20)
        ax.xaxis.label.set_size(25)
        ax.yaxis.label.set_size(25)
        plt.ylim(0, max_p+max_p*0.1)
        plt.xlim(0, filt_top)
        plt.legend(legend, fontsize=15)
        if artf:
            plt.title(fname.split("\\")[-1][4:] +
                      " Compiled Periodogram - Clean", fontsize=40, y=1.02)
            out_name = os.path.join(
                o_dir, fname.split("\\")[-1] + "_p_Clean.png")
        else:
            plt.title(fname.split("\\")[-1][4:] +
                      " Compiled Periodogram", fontsize=40, y=1.02)
            out_name = os.path.join(o_dir, fname.split("\\")[-1] + "_p.png")
        make_path_if_not_exists(out_name)
        print("Saving result to {}".format(out_name))
        fig.savefig(out_name)
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
            gf = bv_plot.GridFig(rows, cols, wspace=0.3, hspace=0.3)
            for i, (key, lfp) in enumerate(lfp_odict.get_clean_signal().items()):
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
                    plt.ylim(0, 0.0045)
                    plt.xlim(0, filt_top)
                # plt.tick_params(labelsize=15)
                plt.legend(legend)
                reg_color = gm.get_next_color()
                plt.title(regions[i+p*16] + " T" + key,
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
            print("Saving result to {}".format(out_name))
            gf.fig.savefig(out_name)
            plt.close()

    if analysis_flags[3]:   # Compare coherence in terms of freq between ACC & RSC
        # lfp_list = select_lfp(fname, ROI)
        matlab = False

        wchans = [int(x) for x in config.get("Wavelet", "wchans").split(", ")]
        import itertools
        wave_combi = list(itertools.combinations(wchans, 2))
        for chan1, chan2 in wave_combi:
            wlet_chans = [chan1, chan2]
            reg_sel = []
            for chan in wlet_chans:  # extracts name of regions selected
                reg_sel.append(regions[chan-1] + "-" + str(chan))

            lfp_odict = LfpODict(fname, channels=wlet_chans, filt_params=(
                filt, filt_btm, filt_top), artf_params=(artf, sd_thres, min_artf_freq, rep_freq, filt))
            legend = []
            lfp_list1, lfp_list2 = [], []
            if not Pre:
                sch_name = []
                block_size = 305
                gm_sch = bv_plot.GroupManager(list(sch_type))
                for k, j in enumerate(range(0, block_size*6, block_size)):
                    new_lfp1 = lfp_odict.get_clean_signal(0).subsample(
                        sample_range=(j, j+block_size))
                    new_lfp2 = lfp_odict.get_clean_signal(1).subsample(
                        sample_range=(j, j+block_size))
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

            wo_dir = os.path.join(
                # o_dir, "wcohere_T{}vsT{}".format(chan1, chan2))
                o_dir, "wcohere_{}vs{}".format(reg_sel[0], reg_sel[1]))
            make_dir_if_not_exists(wo_dir)

            # Plots wavelet coherence for each block in a seperate .png
            for b, (lfp1, lfp2, sch) in enumerate(zip(lfp_list1, lfp_list2, sch_name)):
                if artf:
                    out_name = os.path.join(
                        wo_dir, os.path.basename(fname) + "_wcohere_T{}-T{}_Clean_".format(chan1, chan2) + str(b+1) + ".png")
                else:
                    out_name = os.path.join(
                        wo_dir, os.path.basename(fname) + "_wcohere_T{}-T{}_".format(chan1, chan2) + str(b+1) + ".png")
                sch_n = str(b+1) + "-" + sch
                if matlab:
                    rw_ts = s.get_rw_ts()
                    test_matlab_wcoherence(
                        lfp1, lfp2, rw_ts, sch_n, reg_sel, out_name)
                from bvmpc.lfp_coherence import calc_wave_coherence
                fig, ax = plt.subplots(figsize=(24, 10))
                # title = ("{} vs {} Wavelet Coherence {}".format(
                #     reg_sel[0], reg_sel[1], sch_n))
                # _, result = calc_wave_coherence(
                #     lfp1.get_samples(), lfp2.get_samples(), lfp1.get_timestamp(),
                #     plot_arrows=True, plot_coi=False, resolution=12, title=title,
                #     plot_period=False, all_arrows=False, ax=ax, quiv_x=5)

                title = ("{} vs {} Wavelet Correlation {}".format(
                    reg_sel[0], reg_sel[1], sch_n))
                from bvmpc.lfp_coherence import calc_wave_correlation
                _, result = calc_wave_correlation(
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
                plt.show()
                exit(-1)
                print("Saving result to {}".format(out_name[:-4]+'_pycwt.png'))
                fig.savefig(out_name[:-4]+'_pycwt.png')

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
            # print("Saving result to {}".format(out_name))
            # fig.savefig(out_name)
            # plt.close()


# matlab version of wcoherence
def test_matlab_wcoherence(lfp1, lfp2, rw_ts, sch_n, reg_sel=None, name='default.png'):
    import matlab.engine
    import numpy as np
    from scipy import signal

    eng = matlab.engine.start_matlab()
    fs = lfp1.get_sampling_rate()
    t = lfp1.get_timestamp()
    x = lfp1.get_samples()
    y = lfp2.get_samples()
    x_m = matlab.double(list(x))
    y_m = matlab.double(list(y))

    rw_ts = matlab.double(list(rw_ts))
    o = 7.0
    eng.wcoherence(x_m, y_m, fs, 'NumOctaves', o, nargout=0)
    aspect_ratio = matlab.double([2, 1, 1])
    eng.pbaspect(aspect_ratio, nargout=0)

    title = ("{} vs {} Wavelet Coherence {}".format(
        reg_sel[0], reg_sel[1], sch_n))
    eng.title(title)
    # eng.hold("on", nargout=0)

    # for rw in rw_ts:
    #     # vline demarcating reward point/end of trial
    #     eng.xline(rw, "-r")
    # eng.hold("off", nargout=0)
    fig = eng.gcf()
    print("Saving result to {}".format(name))
    eng.saveas(fig, name, nargout=0)


def test_wct(lfp1, lfp2, sig=True):  # python CWT
    import pycwt as wavelet
    dt = 1/lfp1.get_sampling_rate()
    WCT, aWCT, coi, freq, sig = wavelet.wct(
        lfp1.get_samples(), lfp2.get_samples(), dt, sig=sig)
    _, ax = plt.subplots()
    t = lfp1.get_timestamp()
    ax.contourf(t, freq, WCT, 6, extend='both', cmap="viridis")
    extent = [t.min(), t.max(), 0, max(freq)]
    N = lfp1.get_total_samples()
    sig95 = np.ones([1, N]) * sig[:, None]
    sig95 = WCT / sig95
    ax.contour(t, freq, sig95, [-99, 1], colors='k', linewidths=2,
               extent=extent)
    ax.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                            t[:1] - dt, t[:1] - dt]),
            np.concatenate([coi, [1e-9], freq[-1:],
                            freq[-1:], [1e-9]]),
            'k', alpha=0.3, hatch='x')

    ax.set_title('Wavelet Power Spectrum')
    ax.set_ylabel('Freq (Hz)')
    ax.set_xlabel('Time (s)')

    plt.show()
    exit(-1)


def select_lfp(fname, ROI):  # Select lfp based on region
    # Single Hemi Multisite Drive settings
    lfp_list = []
    chans = [i for i in range(1, 17)]
    regions = ["CLA"] * 8 + ["ACC"] * 4 + ["RSC"] * 4
    # Change filt values here. Default order 10.
    filt_btm = 1.0
    filt_top = 50

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
    config_name = "CAR-SA2.cfg"
    main_entry(config_name)
