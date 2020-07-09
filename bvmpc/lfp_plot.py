import math
import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from bvmpc.bv_utils import make_dir_if_not_exists, find_in, ordered_set
from neurochat.nc_lfp import NLfp
from neurochat.nc_utils import butter_filter
import bvmpc.bv_plot as bv_plot
import pycwt as wavelet


def plot_long_lfp(
    lfp,
    out_name,
    nsamples=None,
    offset=0,
    nsplits=3,
    figsize=(32, 4),
    ylim=(-0.4, 0.4),
):
    """
    Create a figure to display a long LFP signal in nsplits rows.

    Args:
        lfp (NLfp): The lfp signal to plot.
        out_name (str): The name of the output, including directory.
        nsamples (int, optional): Defaults to all samples.
            The number of samples to plot.
        offset (int, optional): Defaults to 0.
            The number of samples into the lfp to start plotting at.
        nsplits (int, optional): The number of rows in the resulting figure.
        figsize (tuple of int, optional): Defaults to (32, 4)
        ylim (tuple of float, optional): Defaults to (-0.4, 0.4)

    Returns:
        None

    """
    if nsamples is None:
        nsamples = lfp.get_total_samples()

    fig, axes = plt.subplots(nsplits, 1, figsize=figsize)
    for i in range(nsplits):
        start = int(offset + i * (nsamples // nsplits))
        end = int(offset + (i + 1) * (nsamples // nsplits))
        if nsplits == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.plot(
            lfp.get_timestamp()[start:end], lfp.get_samples()[start:end], color="k",
        )
        ax.set_ylim(ylim)
    plt.tight_layout()
    bv_plot.savefig(fig, out_name, dpi=400)
    plt.close(fig)


def plot_lfp(
    out_dir,
    lfp_odict,
    segment_length=150,
    in_range=None,
    dpi=50,
    sd=4,
    filt=False,
    artf=False,
    dr_mode=False,
    session=None,
    splits=None,
    x_pad=60,
    return_figs=False,
    verbose=False
):
    """
    Create a number of figures to display lfp signal on multiple channels.

    There will be one figure for each split of thetotal length
    into segment_length, and a row for each value in lfp_odict.

    It is assumed that the input lfps are prefiltered if filtering is required.

    Args:
        out_dir (str): The name of the file to plot to, including dir.
        lfp_odict (OrderedDict): Keys as channels and NLfp objects as vals.
        segment_length (float): Time in seconds of LFP to plot in each figure.
        in_range (tuple(int, int), optional): Time(s) of LFP to plot overall.
            Defaults to None.
        dpi (int, optional): Resulting plot dpi.
        filt (bool): Uses filtered lfp if True
        artf (bool): Should plot artefacts
        session (None) : Optional behavioural session to plot data from.
        splits (np.ndarray) : Optional timepoints to split the lfp at.
            This should contain a number of elements = num_plots + 1
            e.g. to split a 300 sec lfp into 0 - 100 and 100 - 300
            pass splits = [0, 100, 300]
        x_pad (float) : Optional. Pad x_axis for easy comparison. Sets xlim to be larger of x_pad or actual split len

    Returns:
        None

    """
    if session:
        rw_ts = session.get_rw_ts()
        FRes = session.get_trial_df()["First_response"].tolist()
        lev_ts = session.get_lever_ts()
    if dr_mode:
        lfp_dict_s = lfp_odict.get_dr_signals()
        get_info = lfp_odict.get_dr_info
    else:
        if filt:
            lfp_dict_s = lfp_odict.get_filt_signal()
        else:
            lfp_dict_s = lfp_odict.get_signal()
        get_info = lfp_odict.get_info

    if in_range is None:
        in_range = (0, max([lfp.get_duration() for lfp in lfp_dict_s.values()]))
    y_axis_max = max([max(lfp.get_samples()) for lfp in lfp_dict_s.values()])
    y_axis_min = min([min(lfp.get_samples()) for lfp in lfp_dict_s.values()])
    if splits is None:
        seg_splits = np.arange(in_range[0], in_range[1], segment_length)
        if np.abs(in_range[1] - seg_splits[-1]) > 0.0001:
            seg_splits = np.concatenate([seg_splits, [in_range[1]]])
    else:
        seg_splits = splits

    # print(seg_splits[:5])
    # exit(-1)
    max_split_len = max(np.diff(seg_splits))
    if verbose:
        print("Longest segment is {:.2f}s".format(max_split_len))
    if max_split_len < x_pad:
        x_pad = max_split_len

    if return_figs:
        figures = []

    for j, split in enumerate(seg_splits[:-1]):
        fig, axes = plt.subplots(
            nrows=len(lfp_dict_s), figsize=(40, len(lfp_dict_s) * 2)
        )
        a = np.round(split, 2)
        b = np.round(min(seg_splits[j + 1], in_range[1]), 2)
        if list(lfp_dict_s.keys())[0] == "17":
            out_name = os.path.join(
                out_dir, "1_{}_{:.2f}s_to_{:.2f}s.png".format(j, a, b)
            )
        elif dr_mode:
            out_name = os.path.join(
                out_dir, "dr_{}_{:.2f}s_to_{:.2f}s.png".format(j, a, b)
            )
        else:
            out_name = os.path.join(
                out_dir, "0_{}_{:.2f}s_to_{:.2f}s.png".format(j, a, b)
            )
        for i, (key, lfp) in enumerate(lfp_dict_s.items()):
            convert = lfp.get_sampling_rate()
            c_start, c_end = math.floor(a * convert), math.floor(b * convert)
            lfp_sample = lfp.get_samples()[c_start:c_end]
            x_pos = lfp.get_timestamp()[c_start:c_end]
            axes[i].plot(x_pos, lfp_sample, color="k")

            if artf:
                from bvmpc.bv_utils import find_ranges

                shading = list(find_ranges(get_info(key, "thr_locs")))

                for x, y in shading:  # Shading of artf portions
                    times = lfp.get_timestamp()
                    axes[i].axvspan(times[x], times[y], color="red", alpha=0.5)
                mean = get_info(key, "mean")
                std = get_info(key, "std")
                # Label thresholds
                axes[i].axhline(
                    mean - sd * std, linestyle="-", color="red", linewidth="1.5"
                )
                axes[i].axhline(
                    mean + sd * std, linestyle="-", color="red", linewidth="1.5"
                )

            if session:
                for rw in rw_ts:
                    axes[i].axvline(
                        rw, linestyle="-", color="green", linewidth="1.5"
                    )  # vline demarcating reward point/end of trial
                for lev in lev_ts:
                    axes[i].axvline(
                        lev, linestyle="-", color="blue", linewidth="1.5"
                    )  # vline demarcating First Resonse
                for res in FRes:
                    axes[i].axvline(
                        res, linestyle="-", color="orange", linewidth="1.5"
                    )  # vline demarcating First Resonse

            axes[i].text(
                0.03,
                1.02,
                "Channel " + key,
                transform=axes[i].transAxes,
                color="k",
                fontsize=15,
            )
            axes[i].set_ylim(y_axis_min, y_axis_max)
            axes[i].tick_params(labelsize=12)
            # axes[i].set_xlim(a, b)
            seg_len = np.diff([a, b])
            if seg_len > x_pad:
                x_pad = seg_len
            axes[i].set_xlim(a, a + x_pad)

        if not return_figs:
            bv_plot.savefig(fig, out_name)
        else:
            figures.append((fig, out_name))
        plt.close("all")
    
    if return_figs:
        return figures


def lfp_csv(fname, out_dir, lfp_odict, sd, min_artf_freq, shuttles, filt=False):
    """
    Outputs csv for Tetrodes to be used in analysis based on data crossing sd.
    """
    if filt:
        lfp_dict_s = lfp_odict.get_filt_signal()
    else:
        lfp_dict_s = lfp_odict.get_signal()

    tetrodes, threshold, ex_thres, choose, mean_list, std_list, removed = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, (key, lfp) in enumerate(lfp_dict_s.items()):
        mean, std, thr_locs, thr_vals, thr_time, per_removed = lfp.find_artf(
            sd, min_artf_freq
        )
        tetrodes.append("T" + str(key))
        threshold.append(sd * std)
        ex_thres.append(len(thr_locs))
        mean_list.append(mean)
        std_list.append(std)
        removed.append(per_removed)

    import pandas as pd

    min_shut = []
    for a in set(shuttles):
        compare = []
        for j, shut in enumerate(shuttles):
            if shut == a:
                compare.append(ex_thres[j])
        min_shut.append(min(compare))

    truth = find_in(min_shut, ex_thres)
    for c in truth:
        if c:
            choose.append("C")
        else:
            choose.append("")

    csv = {
        "Tetrode": tetrodes,
        "Shuttles": shuttles,
        "Choose": choose,
        "Threshold": threshold,
        "ex_Thres": ex_thres,
        "Mean": mean_list,
        "STD": std_list,
        "% Removed": removed,
    }
    df = pd.DataFrame(csv)

    csv_filename = os.path.join(out_dir, "Tetrode_Summary.csv")
    if tetrodes[0] == "T17":
        check_name = fname.split("\\")[-1] + "_p2"
    else:
        check_name = fname.split("\\")[-1]
    if os.path.exists(csv_filename):
        import csv

        exist = False
        with open(csv_filename, "rt", newline="") as f:
            reader = csv.reader(f, delimiter=",")
            print("Processing {}...".format(fname))
            for row in reader:
                if [check_name] == row:
                    exist = True
                    print("{} info exists.".format(check_name))
                    break
                else:
                    continue
        if not exist:
            with open(csv_filename, "a", newline="") as f:
                f.write("\n{}\n".format(check_name))
                df.to_csv(f, index=False)
            print("Saved {} to {}".format(check_name, csv_filename))
    else:
        with open(csv_filename, "w", newline="") as f:
            f.write("{}\n".format(check_name))
            df.to_csv(f, encoding="utf-8", index=False)
        print("Saved {} to {}".format(check_name, csv_filename))


def plot_sample_of_signal(
    load_loc,
    out_dir=None,
    name=None,
    offseta=0,
    length=50,
    filt_params=(False, None, None),
):
    """
    Plot a small filtered sample of the LFP signal in the given band.

    offseta and length are times
    """
    in_dir = os.path.dirname(load_loc)
    lfp = NLfp()
    lfp.load(load_loc)

    if out_dir is None:
        out_loc = "nc_signal"
        out_dir = os.path.join(in_dir, out_loc)

    if name is None:
        name = "full_signal_filt.png"

    make_dir_if_not_exists(out_dir)
    out_name = os.path.join(out_dir, name)
    fs = lfp.get_sampling_rate()
    filt, lower, upper = filt_params
    lfp_to_plot = lfp
    if filt:
        lfp_to_plot = deepcopy(lfp)
        lfp_samples = lfp.get_samples()
        lfp_samples = butter_filter(lfp_samples, fs, 10, lower, upper, "bandpass")
        lfp_to_plot._set_samples(lfp_samples)
    plot_long_lfp(
        lfp_to_plot,
        out_name,
        nsplits=1,
        ylim=(-0.325, 0.325),
        figsize=(20, 2),
        offset=lfp.get_sampling_rate() * offseta,
        nsamples=lfp.get_sampling_rate() * length,
    )


def plot_coherence(f, Cxy, ax=None, color="k", legend=None, dpi=100, tick_freq=10):
    ax, fig1 = _make_ax_if_none(ax)
    # ax.semilogy(f, Cxy)
    ax.plot(f, Cxy, c=color, linestyle="dotted")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("Coherence")
    ax.set_xticks(np.arange(0, f.max(), tick_freq))
    ax.set_ylim(0, 1)
    plt.legend(legend)
    return ax
    # if name is None:
    #     plt.show()
    # else:
    #     fig.savefig(name, dpi=dpi)


def plot_polar_coupling(polar_vectors, mvl, name=None, dpi=100):
    # Kind of the right idea here, but need avg line in bins
    # instead of scatter...
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    res_vec = np.sum(polar_vectors)
    norm = np.abs(res_vec) / mvl

    from neurochat.nc_circular import CircStat

    cs = CircStat()
    r = np.abs(polar_vectors)
    theta = np.rad2deg(np.angle(polar_vectors))
    print(r, theta)
    ax.scatter(theta, r)
    cs.set_rho(r)
    cs.set_theta(theta)
    count, ind, bins = cs.circ_histogram()
    from scipy.stats import binned_statistic

    binned_amp = (r,)
    bins = np.append(bins, bins[0])
    rate = np.append(count, count[0])
    print(bins, rate)
    # ax.plot(np.deg2rad(bins), rate, color="k")
    res_line = res_vec / norm
    print(res_vec)
    ax.plot([np.angle(res_vec), np.angle(res_vec)], [0, norm * mvl], c="r")
    ax.text(np.pi / 8, 0.00001, "MVL {:.5f}".format(mvl))
    ax.set_ylim(0, r.max())
    if name is None:
        plt.show()
    else:
        fig.savefig(name, dpi=dpi)


def _make_ax_if_none(ax, **kwargs):
    """
    Makes a figure and gets the axis from this if no ax exists

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Input axis

    Returns
    -------
    ax, fig
        The created figure and axis if ax is None, else
        the input ax and None
    """

    fig = None
    if ax is None:
        fig = plt.figure()
        ax = plt.gca(**kwargs)
    return ax, fig


def calc_wPowerSpectrum(
    dat,
    sample_times,
    min_freq=1,
    max_freq=256,
    detrend=True,
    c_sig=False,
    resolution=12,
):
    """
    Calculate wavelet power spectrum using pycwt.

    TODO is this function needed?

    Parameters
    ----------
    dat : np.ndarray
        The values of the waveform.
    sample_times : : np.ndarray
        The times at which waveform samples occur.
    min_freq : float
        Supposed to be minimum frequency, but not quite working.
    max_freq : float
        Supposed to be max frequency, but not quite working.
    c_sig : bool, default False
        Optional. Calculate significance.
    detrend : bool, default False
        Optional. Detrend and normalize the input data by its standard deviation.
    resolution : int
        How many wavelets should be at each level. Number of sub-octaves per octaves

    """
    t = np.asarray(sample_times)
    dt = np.mean(np.diff(t))
    dj = resolution

    s0 = min_freq * dt
    if s0 < 2 * dt:
        s0 = 2 * dt
    max_J = max_freq * dt
    J = dj * np.int(np.round(np.log2(max_J / np.abs(s0))))

    # alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red artf

    if detrend:
        p = np.polyfit(t - t[0], dat, 1)
        dat_notrend = dat - np.polyval(p, t - t[0])
        std = dat_notrend.std()  # Standard deviation
        var = std ** 2  # Variance
        dat_norm = dat_notrend / std  # Normalized dataset
    else:
        std = dat.std()  # Standard deviation
        dat_nomean = dat - np.mean(dat)
        dat_norm = dat_nomean / std  # Normalized dataset

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J)

    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs

    if c_sig:
        # calculate the normalized wavelet and Fourier power spectra, and the Fourier equivalent periods for each wavelet scale.
        signif, fft_theor = wavelet.significance(
            1.0, dt, scales, 0, alpha, significance_level=0.95, wavelet=mother
        )
        sig95 = np.ones([1, N]) * signif[:, None]
        sig95 = power / sig95

        # Calculate the global wavelet spectrum and determine its significance level.
        glbl_power = power.mean(axis=1)
        dof = N - scales  # Correction for padding at edges
        glbl_signif, tmp = wavelet.significance(
            var, dt, scales, 1, alpha, significance_level=0.95, dof=dof, wavelet=mother,
        )
