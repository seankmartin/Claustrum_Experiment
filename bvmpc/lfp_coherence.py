from copy import deepcopy

from scipy.signal import hilbert
from scipy.signal import coherence
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import pycwt as wavelet

from neurochat.nc_utils import butter_filter
from neurochat.nc_circular import CircStat
from neurochat.nc_lfp import NLfp
import time


def mean_vector_length(
        low_freq_lfp, high_freq_lfp, amp_norm=True, return_all=False):
    """
    Compute the mean vector length from Hulsemann et al. 2019

    If amp_norm is true, use the sum of the amplitudes to normalise,
    as opposed to the number of observations.
    if return_all is true, return the complex values
    """
    amplitude = split_into_amp_phase(high_freq_lfp)[0]
    phase = split_into_amp_phase(low_freq_lfp, deg=False)[1]
    if amplitude.size != phase.size:
        raise ValueError(
            "Amp and phase: {} {} elements, equal size needed in MVL".format(
                amplitude.size, phase.size))
    # This could also be computed using CircStat
    norm = np.sum(amplitude) if amp_norm else amplitude.size
    polar_vectors = np.multiply(amplitude, np.exp(1j * phase))
    res_vector = np.sum(polar_vectors)
    mvl = np.abs(res_vector) / norm
    if return_all:
        return polar_vectors, mvl
    else:
        return mvl


def mvl_shuffle(low_freq_lfp, high_freq_lfp, amp_norm=True, nshuffles=200):
    """Compute a shuffled distribution from Hulsemann et al. 2019"""
    samples = high_freq_lfp.get_samples()
    new_lfp = deepcopy(high_freq_lfp)
    observed_mvl = mean_vector_length(
        low_freq_lfp, high_freq_lfp, amp_norm=amp_norm)
    shuffled_mvl = np.zeros(shape=(nshuffles))
    for i in range(len(shuffled_mvl)):
        sample_idx = int(np.floor(
            (low_freq_lfp.get_total_samples() + 1) * np.random.random_sample()))
        reversed_arr1 = samples[0:sample_idx][::-1]
        reversed_arr2 = samples[sample_idx:samples.size][::-1]
        permuted_amp_time = np.concatenate(
            [reversed_arr1, reversed_arr2], axis=None)
        new_lfp._set_samples(permuted_amp_time)
        mvl = mean_vector_length(
            low_freq_lfp, new_lfp, amp_norm=amp_norm)
        shuffled_mvl[i] = mvl
    z_val = (observed_mvl - np.mean(shuffled_mvl)) / np.std(shuffled_mvl)
    mvl95 = np.percentile(shuffled_mvl, 95)
    p_val = norm.sf(z_val)

    return observed_mvl, mvl95, z_val, p_val


def calc_coherence(lfp1, lfp2, squared=True):
    f, Cxy = coherence(
        lfp1.get_samples(), lfp2.get_samples(),
        fs=lfp1.get_sampling_rate(), nperseg=1024)
    if squared:
        return f, Cxy
    result = f, np.sqrt(Cxy)
    return result


def split_into_amp_phase(lfp, deg=False):
    """It is assumed that the lfp signal passed in is already filtered."""
    if type(lfp) is NLfp:
        lfp_samples = lfp.get_samples()
    else:
        lfp_samples = lfp
    complex_lfp = hilbert(lfp_samples)
    phase = np.angle(complex_lfp, deg=deg)
    amplitude = np.abs(complex_lfp)
    return amplitude, phase


def plot_wave_coherence(
        wave1, wave2, sample_times,
        min_freq=1, max_freq=256,
        sig=False, ax=None, title="Wavelet Coherence",
        plot_arrows=True, plot_coi=True, plot_period=False,
        resolution=12, all_arrows=True, quiv_x=5, quiv_y=24, block=None):
    """
    Calculate wavelet coherence between wave1 and wave2 using pycwt.

    TODO fix min_freq, max_freq and add parameters to control arrows.
    TODO also test out sig on a large dataset

    Parameters
    ----------
    wave1 : np.ndarray
        The values of the first waveform.
    wave2 : np.ndarray
        The values of the second waveform.
    sample_times : np.ndarray
        The times at which waveform samples occur.
    min_freq : float
        Supposed to be minimum frequency, but not quite working.
    max_freq : float
        Supposed to be max frequency, but not quite working.
    sig : bool, default False
        Optional Should significance of waveform coherence be calculated.
    ax : plt.axe, default None
        Optional ax object to plot into.
    title : str, default "Wavelet Coherence"
        Optional title for the graph
    plot_arrows : bool, default True
        Should phase arrows be plotted.
    plot_coi : bool, default True
        Should the cone of influence be plotted
    plot_period : bool
        Should the y-axis be in period or in frequency (Hz)
    resolution : int
        How many wavelets should be at each level of the graph
    all_arrows : bool
        Should phase arrows be plotted uniformly or only at high coherence
    quiv_x : float
        sets quiver window in time domain in seconds
    quiv_y : float
        sets number of quivers evenly distributed across freq limits
    block : [int, int]
        Plots only points between ints.

    Returns
    -------
    tuple : (fig, result)
        Where fig is a matplotlib Figure
        and result is a tuple consisting of WCT, aWCT, coi, freq, sig
        WCT - 2D numpy array with coherence values
        aWCT - 2D numpy array with same shape as aWCT indicating phase angles
        coi - 1D numpy array with a frequency value for each time 
        freq - 1D numpy array with the frequencies wavelets were calculated at
        sig - 2D numpy array indicating where data is significant by monte carlo

    """
    t = np.asarray(sample_times)
    dt = np.mean(np.diff(t))
    # Set up the scales to match min max input frequencies
    dj = resolution
    s0 = min_freq * dt
    if s0 < 2 * dt:
        s0 = 2 * dt
    max_J = max_freq * dt
    J = dj * np.int(np.round(np.log2(max_J / np.abs(s0))))
    # freqs = np.geomspace(max_freq, min_freq, num=50)
    freqs = None

    # Do the actual calculation
    print("Calculating coherence...")
    start_time = time.time()
    WCT, aWCT, coi, freq, sig = wavelet.wct(
        wave1, wave2, dt,  # Fixed params
        dj=(1.0 / dj), s0=s0, J=J, sig=sig, normalize=True, freqs=freqs,
    )
    print("Time Taken: %s s" % (time.time() - start_time))
    if np.max(WCT) > 1 or np.min(WCT) < 0:
        print('WCT was out of range: min {},max {}'.format(
            np.min(WCT), np.max(WCT)))
        WCT = np.clip(WCT, 0, 1)

    # Convert frequency to period if necessary
    if plot_period:
        y_vals = np.log2(1 / freq)
    if not plot_period:
        y_vals = np.log2(freq)

    # Calculates the phase between both time series. The phase arrows in the
    # cross wavelet power spectrum rotate clockwise with 'north' origin.
    # The relative phase relationship convention is the same as adopted
    # by Torrence and Webster (1999), where in phase signals point
    # upwards (N), anti-phase signals point downwards (S). If X leads Y,
    # arrows point to the right (E) and if X lags Y, arrow points to the
    # left (W).
    angle = 0.5 * np.pi - aWCT
    u, v = np.cos(angle), np.sin(angle)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    # Set the x and y axes of the plot
    extent_corr = [t.min(), t.max(), 0, max(y_vals)]

    # Fill the plot with the magnitude squared coherence values
    # That is, MSC = abs(Pxy) ^ 2 / (Pxx * Pyy)
    # TODO I think this might be the wrong way to plot this
    # It assumes that the samples are linearly spaced
    im = NonUniformImage(ax, interpolation='bilinear', extent=extent_corr)
    if plot_period:
        im.set_data(t, y_vals, WCT)
    else:
        im.set_data(t, y_vals[::-1], WCT[::-1, :])
    ax.images.append(im)
    # pcm = ax.pcolormesh(WCT)

    # Plot the cone of influence - Periods greater than
    # those are subject to edge effects.
    if plot_coi:
        # Performed by plotting a polygon
        x_positions = np.zeros(shape=(len(t),))
        x_positions = t

        y_positions = np.zeros(shape=(len(t),))
        if plot_period:
            y_positions = np.log2(coi)
        else:
            y_positions = np.log2(1 / coi)

        ax.plot(x_positions, y_positions,
                'w--', linewidth=2, c="w")

    # Plot the significance level contour plot
    if sig:
        ax.contour(t, y_vals, sig,
                   [-99, 1], colors='k', linewidths=2, extent=extent_corr)

    # Add limits, titles, etc.
    ax.set_ylim(min(y_vals), max(y_vals))
    if block:
        ax.set_xlim(t[block[0]], t[int(block[1]*1/dt)])
    else:
        ax.set_xlim(t.min(), t.max())

    # TODO split graph into smaller time chunks
    # Test for smaller timescale
    # quiv_x = 1

    # Plot the arrows on the plot
    if plot_arrows:
        # TODO currently this is a uniform grid, could be changed to WCT > 0.5

        x_res = int(1/dt * quiv_x)
        y_res = int(np.floor(len(y_vals)/quiv_y))
        if all_arrows:
            ax.quiver(t[::x_res], y_vals[::y_res],
                      u[::y_res, ::x_res], v[::y_res, ::x_res], units='height',
                      angles='uv', pivot='mid', linewidth=1, edgecolor='k', scale=30,
                      headwidth=10, headlength=10, headaxislength=5, minshaft=2,
                      )
        else:
            # t[::x_res], y_vals[::y_res],
            # u[::y_res, ::x_res], v[::y_res, ::x_res]
            high_points = np.nonzero(WCT[::y_res, ::x_res] > 0.5)
            sub_t = t[::x_res][high_points[1]]
            sub_y = y_vals[::y_res][high_points[0]]
            sub_u = u[::y_res, ::x_res][np.array(
                high_points[0]), np.array(high_points[1])]
            sub_v = v[::y_res, ::x_res][high_points[0], high_points[1]]
            res = 1
            ax.quiver(sub_t[::res], sub_y[::res],
                      sub_u[::res], sub_v[::res], units='height',
                      angles='uv', pivot='mid', linewidth=1, edgecolor='k', scale=30,
                      headwidth=10, headlength=10, headaxislength=5, minshaft=2,
                      )
    # splits = [0, 60, 120 ...]
    # Add the colorbar to the figure
    if fig is not None:
        fig.colorbar(im)
    else:
        plt.colorbar(im, ax=ax, use_gridspec=True)

    if plot_period:
        y_ticks = np.linspace(min(y_vals), max(y_vals), 8)
        # TODO improve ticks
        y_ticks = [np.log2(x) for x in [0.004, 0.008, 0.016,
                                        0.032, 0.064, 0.125, 0.25, 0.5, 1]]
        y_labels = [str(x) for x in (np.round(np.exp2(y_ticks), 3))]
        ax.set_ylabel("Period")
    else:
        y_ticks = np.linspace(min(y_vals), max(y_vals), 8)
        # TODO improve ticks
        # y_ticks = [np.log2(x) for x in [256, 128, 64, 32, 16, 8, 4, 2, 1]]
        y_ticks = [np.log2(x) for x in [64, 32, 16, 8, 4, 2, 1]]
        y_labels = [str(x) for x in (np.round(np.exp2(y_ticks), 3))]
        ax.set_ylabel("Frequency (Hz)")
    plt.yticks(y_ticks, y_labels)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")

    return (fig, [WCT, aWCT, coi, freq, sig])


def calc_wave_coherence(wave1, wave2, sample_times, min_freq=1, max_freq=128, sig=False, resolution=12):
    """
    Calculate wavelet coherence between wave1 and wave2 using pycwt.

    Parameters
    ----------
    wave1 : np.ndarray
        The values of the first waveform.
    wave2 : np.ndarray
        The values of the second waveform.
    sample_times : np.ndarray
        The times at which waveform samples occur.
    min_freq : float
        Supposed to be minimum frequency, but not quite working.
    max_freq : float
        Supposed to be max frequency, but not quite working.
    sig : bool, default False
        Optional Should significance of waveform coherence be calculated.
    resolution : int
        How many wavelets should be at each level

    Returns
    -------
    WCT, t, freq, coi, sig, aWCT
        WCT - 2D numpy array with coherence values
        t - 2D numpy array with sample_times
        freq - 1D numpy array with the frequencies wavelets were calculated at
        coi - 1D numpy array with a frequency value for each time 
        sig - 2D numpy array indicating where data is significant by monte carlo
        aWCT - 2D numpy array with same shape as aWCT indicating phase angles
    """

    t = np.asarray(sample_times)
    dt = np.mean(np.diff(t))  # dt = 0.004

    dj = resolution
    s0 = 1/max_freq
    if s0 < (2 * dt):
        s0 = 2 * dt
    max_J = 1/min_freq
    J = dj * np.int(np.round(np.log2(max_J / np.abs(s0))))

    # # Original by Sean
    # s0 = min_freq * dt
    # if s0 < (2 * dt):
    #     s0 = 2 * dt
    # max_J = max_freq * dt
    # J = dj * np.int(np.round(np.log2(max_J / np.abs(s0))))

    # Do the actual calculation
    print("Calculating coherence...")
    start_time = time.time()
    WCT, aWCT, coi, freq, sig = wavelet.wct(
        wave1, wave2, dt,  # Fixed params
        dj=(1.0 / dj), s0=s0, J=J, sig=sig, normalize=True)
    print("Time Taken: %s s" % (time.time() - start_time))
    if np.max(WCT) > 1 or np.min(WCT) < 0:
        print('WCT was out of range: min {},max {}'.format(
            np.min(WCT), np.max(WCT)))
        WCT = np.clip(WCT, 0, 1)

    return WCT, t, freq, coi, sig, aWCT


def plot_wcohere(WCT, t, freq, coi=None, sig=None, plot_period=False, ax=None, title="Wavelet Coherence", block=None):
    """
    Plot wavelet coherence using results from calc_wave_coherence.

    Parameters
    ----------
    *First 5 parameters can be obtained from from calc_wave_coherence
        WCT: 2D numpy array with coherence values
        t : 2D numpy array with sample_times
        freq : 1D numpy array with the frequencies wavelets were calculated at
        sig : 2D numpy array, default None
            Optional. Plots significance of waveform coherence contours.
        coi : 2D numpy array, default None
            Optional. Pass coi to plot cone of influence
    plot_period : bool
        Should the y-axis be in period or in frequency (Hz)
    ax : plt.axe, default None
        Optional ax object to plot into.
    title : str, default "Wavelet Coherence"
        Optional title for the graph
    block : [int, int]
        Plots only points between ints.

    Returns
    -------
    tuple : (fig, wcohere_pvals)
        Where fig is a matplotlib Figure
        and result is a tuple consisting of [WCT, t, y_vals]
    """
    dt = np.mean(np.diff(t))

    if plot_period:
        y_vals = np.log2(1 / freq)
    if not plot_period:
        y_vals = np.log2(freq)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    # Set the x and y axes of the plot
    extent_corr = [t.min(), t.max(), 0, max(y_vals)]
    # Fill the plot with the magnitude squared coherence values
    im = NonUniformImage(ax, interpolation='bilinear', extent=extent_corr)
    if plot_period:
        im.set_data(t, y_vals, WCT)
    else:
        im.set_data(t, y_vals[::-1], WCT[::-1, :])
    im.set_clim(0, 1)
    ax.images.append(im)

    # Plot the cone of influence - Periods greater thanthose are subject to edge effects.
    if coi is not None:
        # Performed by plotting a polygon
        x_positions = np.zeros(shape=(len(t),))
        x_positions = t

        y_positions = np.zeros(shape=(len(t),))
        if plot_period:
            y_positions = np.log2(coi)
        else:
            y_positions = np.log2(1 / coi)

        ax.plot(x_positions, y_positions,
                'w--', linewidth=2, c="w")

    # Plot the significance level contour plot
    if sig is not None:
        ax.contour(t, y_vals, sig,
                   [-99, 1], colors='k', linewidths=2, extent=extent_corr)

    # Add limits, titles, etc.
    ax.set_ylim(min(y_vals), max(y_vals))
    if block:
        ax.set_xlim(t[block[0]], t[int(block[1]*1/dt)])
    else:
        ax.set_xlim(t.min(), t.max())

    if fig is not None:
        fig.colorbar(im)
    else:
        plt.colorbar(im, ax=ax, use_gridspec=True)

    if plot_period:
        y_ticks = np.linspace(min(y_vals), max(y_vals), 8)
        # TODO improve ticks
        y_ticks = [np.log2(x) for x in [0.004, 0.008, 0.016,
                                        0.032, 0.064, 0.125, 0.25, 0.5, 1]]
        y_labels = [str(x) for x in (np.round(np.exp2(y_ticks), 3))]
        ax.set_ylabel("Period")
    else:
        y_ticks = np.linspace(min(y_vals), max(y_vals), 8)
        # TODO improve ticks
        # y_ticks = [np.log2(x) for x in [256, 128, 64, 32, 16, 8, 4, 2, 1]]
        y_ticks = [np.log2(x) for x in [64, 32, 16, 8, 4, 2, 1]]
        y_labels = [str(x) for x in (np.round(np.exp2(y_ticks), 3))]
        ax.set_ylabel("Frequency (Hz)")
    plt.yticks(y_ticks, y_labels)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")

    return fig, [WCT, t, y_vals]


def plot_arrows(ax, wcohere_pvals, aWCT=None, u=None, v=None, magnitute=None, quiv_x=5, quiv_y=24, all_arrows=False):
    """
    Plots phase arrows for wavelet coherence plot using results from plot_wcohere

    Parameters
    ----------
    wcohere_pvals:
        input structure, includes [WCT, t, y_vals]
        the first three parameters are out from plot_wcohere
    aWCT : 2D numpy
        array with same shape as aWCT indicating phase angles
        *Can be obtained from last value in calc_wave_coherence
    u : 2D numpy array of unit vector's cos angle
    v : 2D numpy array of unit vector's sin angle
    magnitute : 2D numpy array of vector magnitute at each freq and timepoint
    quiv_x : float
        sets quiver window in time domain in seconds
    quiv_y : float
        sets number of quivers evenly distributed across freq limits
    all_arrows : bool
        Should phase arrows be plotted uniformly or only at high coherence

    """
    WCT, t, y_vals = wcohere_pvals

    if aWCT is not None:
        angle = 0.5 * np.pi - aWCT
        u, v = np.cos(angle), np.sin(angle)
    elif u is None or v is None:
        raise ValueError("Must pass aWCT or [u, v]")

    dt = np.mean(np.diff(t))

    x_res = int(1/dt * quiv_x)
    y_res = int(np.floor(len(y_vals)/quiv_y))
    if all_arrows:
        ax.quiver(t[::x_res], y_vals[::y_res],
                  u[::y_res, ::x_res], v[::y_res, ::x_res], units='height',
                  angles='uv', pivot='mid', linewidth=1, edgecolor='k', scale=30,
                  headwidth=10, headlength=10, headaxislength=5, minshaft=2,
                  )
    else:
        # t[::x_res], y_vals[::y_res],
        # u[::y_res, ::x_res], v[::y_res, ::x_res]
        if magnitute is not None:
            f_mean = np.empty_like(magnitute)
            for i, f in enumerate(magnitute):
                # Plot arrows if magnitute > mean of particular frequency
                f_mean[i, :] = np.mean(f) + np.std(f)
            high_points = np.nonzero(
                magnitute[::y_res, ::x_res] > f_mean[::y_res, ::x_res])

        else:
            high_points = np.nonzero(WCT[::y_res, ::x_res] > 0.5)
        sub_t = t[::x_res][high_points[1]]
        sub_y = y_vals[::y_res][high_points[0]]
        sub_u = u[::y_res, ::x_res][np.array(
            high_points[0]), np.array(high_points[1])]
        sub_v = v[::y_res, ::x_res][high_points[0], high_points[1]]
        res = 1
        ax.quiver(sub_t[::res], sub_y[::res],
                  sub_u[::res], sub_v[::res], units='height',
                  angles='uv', pivot='mid', linewidth=1, edgecolor='k', scale=30,
                  headwidth=10, headlength=10, headaxislength=5, minshaft=2,
                  )

    return ax


def wcohere_mean(WCT, aWCT, t_blocks=None):
    """
    Calculates mean of WCT and corresponding vector magnitute and phase across t_blocks

    Parameters
    ----------
    WCT: 2D numpy array with coherence values
    aWCT : 2D numpy
        array with same shape as aWCT indicating phase angles
        *Can be obtained from last value in calc_wave_coherence
    t_blocks : list, [start end]
        list of 2 elements with start and end time of each alignment block
    Returns
    -------
    mean_WCT : Mean WCT across blocks
    norm_u : unit u across blocks
    norm_v : unit v across blocks
    magnitute : magnitute of vector summed across blocks

    """

    angle = 0.5 * np.pi - aWCT
    u, v = np.cos(angle), np.sin(angle)

    t_win = int((t_blocks[0][1] - t_blocks[0][0])*250)
    print("t_win:", t_win)

    WCT_trial = np.empty((
        len(t_blocks), WCT.shape[0], t_win), dtype=np.float32)
    all_u = np.empty((
        len(t_blocks), u.shape[0], t_win), dtype=np.float32)
    all_v = np.empty((
        len(t_blocks), v.shape[0], t_win), dtype=np.float32)
    WCT_trial.fill(np.nan)
    all_u.fill(np.nan)
    all_v.fill(np.nan)

    for i, (a, b) in enumerate(t_blocks):
        start_idx = int(a*250)
        end_idx = start_idx + t_win

        WCT_this_trial = np.empty(
            (WCT.shape[0], t_win), dtype=np.float32)
        u_this_trial = np.empty(
            (u.shape[0], t_win), dtype=np.float32)
        v_this_trial = np.empty(
            (v.shape[0], t_win), dtype=np.float32)
        WCT_this_trial.fill(np.nan)
        u_this_trial.fill(np.nan)
        v_this_trial.fill(np.nan)

        if a >= 0:
            # For when window exceeds session length (eg. Last trial)
            _, y = WCT[:, start_idx:end_idx].shape
            WCT_this_trial[:, :y] = WCT[:, start_idx:end_idx]
            u_this_trial[:, :y] = u[:, start_idx:end_idx]
            v_this_trial[:, :y] = v[:, start_idx:end_idx]
        else:
            # for blocks shorter then t_win with missing values before alignment point (eg. first trial)
            WCT_this_trial[:, -start_idx:] = WCT[:, 0:end_idx]
            u_this_trial[:, -start_idx:] = u[:, 0:end_idx]
            v_this_trial[:, -start_idx:] = v[:, 0:end_idx]
            # print(a, b)
            # print(WCT_trial)

        WCT_trial[i] = WCT_this_trial
        all_u[i] = u_this_trial
        all_v[i] = v_this_trial
        # else:
        # wcohere_trial = np.dstack(wcohere_trial, WCT[:, int(a):int(b)])
        # print(wcohere_trial[:, :, i])
        # print("Shape:", wcohere_trial.shape)
        # exit(-1)
    mean_WCT = np.nanmean(WCT_trial, axis=0)

    sum_u = np.nansum(all_u, axis=0)
    sum_v = np.nansum(all_v, axis=0)

    # magnitute = np.linalg.norm(np.array(sum_u, sum_v))
    magnitute = np.sqrt(np.square(sum_u) + np.square(sum_v))
    # Normalise vectors to obtain unit vector
    norm_u = sum_u / magnitute
    norm_v = sum_v / magnitute

    print(len(norm_u))

    # TODO validate length threshold and which arrows to plot....
    return mean_WCT, norm_u, norm_v, magnitute


def plot_cross_wavelet(
        wave1, wave2, sample_times,
        min_freq=1, max_freq=256,
        sig=False, ax=None, title="Cross-Wavelet",
        plot_coi=True, plot_period=False,
        resolution=12, all_arrows=True, quiv_x=5, quiv_y=24, block=None):
    """
    Plot cross wavelet correlation between wave1 and wave2 using pycwt.

    TODO Fix this function
    TODO also test out sig on a large dataset

    Parameters
    ----------
    wave1 : np.ndarray
        The values of the first waveform.
    wave2 : np.ndarray
        The values of the second waveform.
    sample_times : np.ndarray
        The times at which waveform samples occur.
    min_freq : float
        Supposed to be minimum frequency, but not quite working.
    max_freq : float
        Supposed to be max frequency, but not quite working.
    sig : bool, default False
        Optional Should significance of waveform coherence be calculated.
    ax : plt.axe, default None
        Optional ax object to plot into.
    title : str, default "Wavelet Coherence"
        Optional title for the graph
    plot_coi : bool, default True
        Should the cone of influence be plotted
    plot_period : bool
        Should the y-axis be in period or in frequency (Hz)
    resolution : int
        How many wavelets should be at each level of the graph
    all_arrows : bool
        Should phase arrows be plotted uniformly or only at high coherence
    quiv_x : float
        sets quiver window in time domain in seconds
    quiv_y : float
        sets number of quivers evenly distributed across freq limits
    block : [int, int]
        Plots only points between ints.

    Returns
    -------
    tuple : (fig, result)
        Where fig is a matplotlib Figure
        and result is a tuple consisting of WCT, aWCT, coi, freq, sig
        WCT - 2D numpy array with coherence values
        aWCT - 2D numpy array with same shape as aWCT indicating phase angles
        coi - 1D numpy array with a frequency value for each time 
        freq - 1D numpy array with the frequencies wavelets were calculated at
        sig - 2D numpy array indicating where data is significant by monte carlo

    """
    t = np.asarray(sample_times)
    dt = np.mean(np.diff(t))
    # Set up the scales to match min max input frequencies
    dj = resolution
    s0 = min_freq * dt
    if s0 < 2 * dt:
        s0 = 2 * dt
    max_J = max_freq * dt
    J = dj * np.int(np.round(np.log2(max_J / np.abs(s0))))

    # Do the actual calculation
    W12, coi, freq, signif = wavelet.xwt(
        wave1, wave2, dt,  # Fixed params
        dj=(1.0 / dj), s0=s0, J=J, significance_level=0.8646, normalize=True,
    )

    cross_power = np.abs(W12)**2

    if np.max(W12) > 6**2 or np.min(W12) < 0:
        print('W12 was out of range: min{},max{}'.format(
            np.min(W12), np.max(W12)))
        cross_power = np.clip(cross_power, 0, 6**2)
    # print('cross max:', np.max(cross_power))
    # print('cross min:', np.min(cross_power))

    cross_sig = np.ones([1, len(t)]) * signif[:, None]
    cross_sig = cross_power / cross_sig  # Power is significant where ratio > 1

    # Convert frequency to period if necessary
    if plot_period:
        y_vals = np.log2(1 / freq)
    if not plot_period:
        y_vals = np.log2(freq)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    # Set the x and y axes of the plot
    extent_corr = [t.min(), t.max(), 0, max(y_vals)]

    # Fill the plot with the magnitude squared correlation values
    # That is, MSC = abs(Pxy) ^ 2 / (Pxx * Pyy)
    # TODO I think this might be the wrong way to plot this
    # It assumes that the samples are linearly spaced
    im = NonUniformImage(ax, interpolation='bilinear', extent=extent_corr)

    if plot_period:
        im.set_data(t, y_vals, cross_power)
    else:
        im.set_data(t, y_vals[::-1], cross_power[::-1, :])
    ax.images.append(im)
    # pcm = ax.pcolormesh(WCT)

    # Plot the cone of influence - Periods greater than
    # those are subject to edge effects.
    if plot_coi:
        # Performed by plotting a polygon
        x_positions = np.zeros(shape=(len(t),))
        x_positions = t

        y_positions = np.zeros(shape=(len(t),))
        if plot_period:
            y_positions = np.log2(coi)
        else:
            y_positions = np.log2(1 / coi)

        ax.plot(x_positions, y_positions,
                'w--', linewidth=2, c="w")

    # Plot the significance level contour plot
    if sig:
        ax.contour(t, y_vals, cross_sig,
                   [-99, 1], colors='k', linewidths=2, extent=extent_corr)

    # Add limits, titles, etc.
    ax.set_ylim(min(y_vals), max(y_vals))

    if block:
        ax.set_xlim(t[block[0]], t[int(block[1]*1/dt)])
    else:
        ax.set_xlim(t.min(), t.max())

    # TODO split graph into smaller time chunks
    # Test for smaller timescale
    # quiv_x = 1

    # Add the colorbar to the figure
    if fig is not None:
        fig.colorbar(im)
    else:
        plt.colorbar(im, ax=ax, use_gridspec=True)

    if plot_period:
        y_ticks = np.linspace(min(y_vals), max(y_vals), 8)
        # TODO improve ticks
        y_ticks = [np.log2(x) for x in [0.004, 0.008, 0.016,
                                        0.032, 0.064, 0.125, 0.25, 0.5, 1]]
        y_labels = [str(x) for x in (np.round(np.exp2(y_ticks), 3))]
    else:
        y_ticks = np.linspace(min(y_vals), max(y_vals), 8)
        # TODO improve ticks
        # y_ticks = [np.log2(x) for x in [256, 128, 64, 32, 16, 8, 4, 2, 1]]
        y_ticks = [np.log2(x) for x in [64, 32, 16, 8, 4, 2, 1]]
        y_labels = [str(x) for x in (np.round(np.exp2(y_ticks), 3))]
    plt.yticks(y_ticks, y_labels)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")

    if plot_period:
        ax.set_ylabel("Period")
    else:
        ax.set_ylabel("Frequency (Hz)")

    return (fig, [W12, coi, freq, sig])


def test_wave_coherence():
    # Compare this to Matlab
    fs = 1000.0
    t = np.arange(0, 2.0011, 0.001)
    x = 0.25 * np.random.randn(t.size)
    temp = np.zeros(t.size)
    np.cos(2 * np.pi * 10 * t, where=((t >= 0.5) & (t < 1.1)), out=temp)
    x += temp
    temp.fill(0)
    np.cos(2 * np.pi * 50 * t, where=((t >= 0.2) & (t < 1.4)), out=temp)
    x += temp
    temp.fill(0)

    y = 0.35 * np.random.randn(t.size)
    np.sin(2 * np.pi * 10 * t, where=((t >= 0.6) & (t < 1.2)), out=temp)
    y += temp
    temp.fill(0)
    np.sin(2 * np.pi * 50 * t, where=((t >= 0.4) & (t < 1.6)), out=temp)
    y += temp
    fig, data = plot_wave_coherence(
        x, y, t, min_freq=2, max_freq=1000,
        plot_arrows=True, plot_coi=True, plot_period=True,
        resolution=12, all_arrows=True)
    fig.savefig("test_coherence_o.png", dpi=400)
    do_matlab = False

    if do_matlab:
        import matlab.engine

        eng = matlab.engine.start_matlab()
        x_m = matlab.double(list(x))
        y_m = matlab.double(list(y))

        eng.wcoherence(x_m, y_m, fs, nargout=0)
        fig = eng.gcf()
        eng.saveas(fig, "test_matlab_freq.png", nargout=0)


def test_coherence():
    from scipy import signal
    import matplotlib.pyplot as plt
    from lfp_plot import plot_coherence
    fs = 10e3
    N = 1e5
    amp = 20
    freq = 1234.0
    noise_power = 0.001 * fs / 2
    time = np.arange(N) / fs
    b, a = signal.butter(2, 0.25, 'low')
    x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    y = signal.lfilter(b, a, x)
    x += amp * np.sin(2 * np.pi * freq * time)
    y += np.random.normal(scale=0.1 * np.sqrt(noise_power), size=time.shape)
    f, Cxy = coherence(x, y, fs, nperseg=1024)
    plot_coherence(f, Cxy, tick_freq=1000)


def test_mvl():
    from scipy import signal
    import matplotlib.pyplot as plt
    t = np.linspace(0, 1, 500)
    # 10 Hz sawtooth wave sampled at 500Hz
    sig1 = signal.sawtooth(2 * np.pi * 10 * t)
    # 50Hz sine wave sampled at 500Hz
    sig2 = 0.02 * np.sin(2 * np.pi * 50 * t)
    sig3 = 0.2 * np.sin(np.pi + 2 * np.pi * 10 * t)
    sig6 = 0.1 * np.sin(2 * np.pi * 20 * t)
    sig4 = 0.02 * np.sin(2 * np.pi * 55 * t)
    sig5 = 0.02 * np.sin(np.pi + 2 * np.pi * 65 * t)
    coupled_sig = sig2 + sig3 + sig6
    fig, ax = plt.subplots()
    ax.plot(sig1, c="b")
    ax.plot(coupled_sig, c="r")
    ax.plot(sig2 + sig4 + sig5, c="g")
    plt.show()
    plt.close("all")
    all1, res1 = mean_vector_length(
        sig1, coupled_sig, amp_norm=False, return_all=True)
    all2, res2 = mean_vector_length(
        sig1, sig2 + sig4 + sig5, amp_norm=False, return_all=True)
    print(res1, res2)
    from lfp_plot import plot_polar_coupling
    plot_polar_coupling(all1, res1)
    plot_polar_coupling(all2, res2)


if __name__ == "__main__":
    """Test out these functions."""
    test_record = False
    test_sim = True

    if test_sim:
        test_coherence()
        test_mvl()

    if test_record:
        recording = r"C:\Users\smartin5\Recordings\ER\LFP-cla-V2L\LFP-cla-V2L-ctrl\05112019-white\05112019-white-D"
        channels = [1, 5]
        from lfp_odict import LfpODict
        lfp_odict = LfpODict(recording, channels=channels)
        f, Cxy = calc_coherence(
            lfp_odict.get_signal(0),
            lfp_odict.get_signal(1))
        from lfp_plot import plot_coherence
        plot_coherence(f, Cxy)
