# -*- coding: utf-8 -*-
"""
Plots and analysis for MEDPC behaviour.

@author: HAMG
"""

import os.path
import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns

from bvmpc.bv_utils import mycolors
import bvmpc.bv_plot as bv_plot


def split_lever_ts(session):
    """
    Split lever timestamps into schedule-based arrays of trials.

    Note
    ----
    Still in progress!

    Parameters
    ----------
    session : bvmpc.bv_session.Session
        The session to split lever timestamps of

    Returns
    -------
    (List, List)
        (Split up list of ratio trials, Split up list of interval trials)

    """
    lever_ts = session.get_lever_ts()
    switch_ts = np.arange(5, 1830, 305)
    reward_times = session.get_rw_ts()
    trial_type = session.get_arrays('Trial Type')

    ratio_lever_ts = []
    interval_lever_ts = []
    block_lever_ts = np.split(lever_ts, np.searchsorted(lever_ts, switch_ts))
    block_reward_ts = np.split(reward_times,
                               np.searchsorted(reward_times, switch_ts))
    for i, (l, r) in enumerate(zip(block_lever_ts, block_reward_ts)):
        if trial_type[i] == 1:  # FR Block
            split_lever_ts = np.split(l, np.searchsorted(l, r))
            ratio_lever_ts.append(split_lever_ts)
        elif trial_type[i] == 0:  # FR Block
            split_lever_ts = np.split(l, np.searchsorted(l, r))
            interval_lever_ts.append(split_lever_ts)
        else:
            print('Not Ready for analysis!')
    print('len of ratio_l'+len(ratio_lever_ts))
    print('len of interval_l'+len(interval_lever_ts))
    return ratio_lever_ts, interval_lever_ts


def cumplot_axona(s, ax=None, p_errors=False, p_all=False):
    """
    Plot a 1x2 cumulative plot for a Session.

    Parameters
    ----------
    s : session object
    ax : plt.axe, default None
        Optional. ax object to plot into.
    p_errors : bool, False
        Optional. Plots error lever presses only
    p_all : bool, False
        Optional. Plot all lever presses including errors. Only applies to stage 7.

    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    else:
        fig = None

    date = s.get_metadata('start_date').replace('/', '_')
    sub = s.get_metadata('subject')
    stage = s.get_stage()
    blocks = s.get_block_starts()
    lever_ts = s.get_lever_ts()
    reward_times = s.get_rw_ts()
    pell_ts = s.get_arrays("Reward")
    pell_double = np.nonzero(np.diff(pell_ts) < 0.8)

    # for printing of error rates and rewards on graph
    err_FI = 0
    err_FR = 0
    rw_FR = 0
    rw_FI = 0
    reward_double = reward_times[np.searchsorted(
        reward_times, pell_ts[pell_double], side='right')]

    norm_r_ts, norm_l_ts, norm_err_ts, norm_dr_ts, incl = s.split_sess(blocks=blocks,
                                                                       error_only=p_errors, all_levers=p_all)
    sch_type = s.get_arrays('Trial Type')
    ratio_c = plt.cm.get_cmap('Wistia')
    interval_c = plt.cm.get_cmap('winter')

    if stage == '7':
        err_lever_ts = s.get_err_lever_ts()
        lever_ts = np.sort(np.concatenate((
            lever_ts, err_lever_ts), axis=None))
    lever_times = np.insert(lever_ts, 0, 0, axis=0)
    ax[0].step(lever_times, np.arange(
        lever_times.size), where="post", label=sub, zorder=1)
    if stage == '7':    # plots error press in red
        ax[0].scatter(err_lever_ts, np.isin(
            lever_times, err_lever_ts).nonzero()[0],
            c='r', label='Errors', s=1, zorder=2)
    bins = lever_times
    reward_y = np.digitize(reward_times, bins) - 1
    double_y = np.digitize(reward_double, bins) - 1
    ax[0].scatter(reward_times, reward_y, marker="x", c="grey",
                  label='Reward Collected', s=25)
    if len(reward_double) > 0:
        dr_print = "\nTotal # of Double Rewards: " + \
            str(len(reward_double))
        ax[0].scatter(reward_double, double_y, marker="x", c="magenta",
                      label='Double Reward', s=25)
    else:
        dr_print = ""
    ax[0].legend(loc='lower right')

    # for printing of error rates on graph
    norm_r_ts, _, norm_err_ts, _, _ = s.split_sess(
        blocks=blocks, all_levers=True)
    sch_type = s.get_arrays('Trial Type')
    for i, l in enumerate(norm_err_ts):
        if sch_type[i] == 1:
            err_FR = err_FR + len(norm_err_ts[i])
        elif sch_type[i] == 0:
            err_FI = err_FI + len(norm_err_ts[i])
    for i, l in enumerate(norm_r_ts):
        if sch_type[i] == 1:
            rw_FR = rw_FR + len(norm_r_ts[i])
        elif sch_type[i] == 0:
            rw_FI = rw_FI + len(norm_r_ts[i])
    rw_print = "\nCorrect FR \\ FI: " + \
        str(rw_FR) + r" \ " + str(rw_FI)

    if err_FR > 0 or err_FI > 0:
        err_print = "\nErrors FR \\ FI: " + \
            str(err_FR) + r" \ " + str(err_FI)
    else:
        err_print = ""
    text = ('Total # of Lever Press: ' + '{}\nTotal # of Rewards: {}{}{}{}'.format(
            len(lever_ts), len(reward_times) + len(reward_double),
            dr_print, rw_print, err_print))
    ax[0].text(0.03, 0.97, text, ha="left",
               va="top", transform=ax[0].transAxes)
    ax[0].set_title('Cumulative Lever Presses', fontsize=10)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Cumulative Lever Presses')

    # Plot cummulative lever press split into blocks
    for i, l in enumerate(norm_l_ts):
        if sch_type[i] == 1:
            ax[1].step(l, np.arange(l.size), c=ratio_c(i*45), where="post",
                       label='B'+str(i+1)+' - FR', zorder=1)
        elif sch_type[i] == 0:
            ax[1].step(l, np.arange(l.size), c=interval_c(i*45), where="post",
                       label='B'+str(i+1)+' - FI', zorder=1)
        bins = l
        reward_y = np.digitize(norm_r_ts[i], bins) - 1
        double_y = np.digitize(norm_dr_ts[i], bins) - 1
        if stage == '7' and p_all:  # plots all responses incl. errors
            ax[1].scatter(norm_err_ts[i], np.isin(
                l, norm_err_ts[i]).nonzero()[0],
                c='r', s=1, zorder=2)
            incl = '_All'

        ax[1].scatter(norm_r_ts[i], reward_y,
                      marker="x", c="grey", s=25)
        ax[1].scatter(norm_dr_ts[i], double_y,
                      marker="x", c="magenta", s=25)
    ax[1].set_title('{}, Block-Split {}'.format(
        sub, incl), fontsize=10)
    ax[1].legend(loc='upper left')
    fig.suptitle('{} {} S{}'.format(sub, date, stage),
                 y=0.99, fontsize=20)
    return fig


def cumplot(session, out_dir, ax=None, int_only=False, zoom=False,
            zoom_sch=False, plot_error=False, plot_all=False):
    """Perform a cumulative plot for a Session."""
    date = session.get_metadata('start_date').replace('/', '_')
    timestamps = session.get_arrays()
    lever_ts = session.get_lever_ts()
    session_type = session.get_metadata('name')
    stage = session_type[:2].replace('_', '')
    subject = session.get_metadata('subject')
    reward_times = session.get_rw_ts()
    pell_ts = timestamps["Reward"]
    pell_double = np.nonzero(np.diff(pell_ts) < 0.5)

    # for printing of error rates and rewards on graph
    err_FI = 0
    err_FR = 0
    rw_FR = 0
    rw_FI = 0
    reward_double = reward_times[np.searchsorted(
        reward_times, pell_ts[pell_double], side='right')]
    single_plot = False
    ratio = session.get_ratio()
    interval = session.get_interval()

    if ax is None:
        single_plot = True
        fig, ax = plt.subplots()
        ax.set_title('Cumulative Lever Presses\n', fontsize=15)
        if session_type == '5a_FixedRatio_p':
            plt.suptitle('\n{}, {} {}, {}'.format(
                subject, session_type[:-2], ratio, date),
                color=mycolors(subject), fontsize=10, y=.98, x=.51)
        elif session_type == '5b_FixedInterval_p':
            plt.suptitle('\n{}, {} {}s, {}'.format(
                subject, session_type[:-2], interval, date),
                color=mycolors(subject), fontsize=10, y=.98, x=.51)
        elif session_type == '6_RandomisedBlocks_p:':
            plt.suptitle('\n{}, {} FR{}/FI{}s, {}'.format(
                subject, session_type[:-2], ratio, interval, date),
                color=mycolors(subject), fontsize=10, y=.98, x=.51)
        else:
            plt.suptitle('\n{}, S{}, {}'.format(
                subject, stage, date), color=mycolors(subject),
                fontsize=10, y=.98, x=.51)
    else:
        if session_type == '5a_FixedRatio_p':
            ax.set_title('\n{}, S{}, FR{}, {}'.format(
                subject, stage, ratio, date),
                color=mycolors(subject), fontsize=10)
        elif session_type == '5b_FixedInterval_p':
            ax.set_title('\n{}, S{}, FI{}s, {}'.format(
                subject, stage, interval, date),
                color=mycolors(subject), fontsize=10)
        elif session_type == '6_RandomisedBlocks_p' or stage == '7':
            switch_ts = np.arange(5, 1830, 305)
            for x in switch_ts:
                plt.axvline(x, color='g', linestyle='-.', linewidth='.4')
            ax.set_title('\n{}, S{}, FR{}/FI{}s, {}'.format(
                subject, stage, ratio, interval, date),
                color=mycolors(subject), fontsize=10)
        else:
            ax.set_title('\n{}, S{}, {}'.format(
                subject, stage, date), color=mycolors(subject),
                fontsize=10, y=1, x=.51)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Lever Presses')

    # Broken and plots are ugly - Plots single trials
    if zoom:
        trial_lever_ts = np.split(lever_ts,
                                  np.searchsorted(lever_ts, reward_times))
        norm_reward_ts = []
        norm_lever_ts = []
        reward_times_0 = np.append([0], reward_times, axis=0)
        for i, l in enumerate(trial_lever_ts[:-1]):
            norm_lever_ts.append(np.append([0], l-reward_times_0[i], axis=0))
            norm_reward_ts.append(reward_times[i]-reward_times_0[i])
        ax.set_xlim(0, np.max(norm_reward_ts))
        color = plt.cm.get_cmap('autumn')
        for i, l in enumerate(norm_lever_ts):
            ax.step(l, np.arange(l.size), c=color(i*20), where="post")
            bins = l
            reward_y = np.digitize(norm_reward_ts[i], bins) - 1
            plt.scatter(norm_reward_ts[i], reward_y,
                        marker="x", c="grey", s=25)
        ax.set_title('\n{}, Trial-Based'.format(
            subject), color=mycolors(subject), fontsize=10)
        ax.legend(loc='lower right')
        return fig

    elif zoom_sch and (session_type == '6_RandomisedBlocks_p' or stage == '7'):
        # plots cum graph based on schedule type (i.e. FI/FR)
        norm_r_ts, norm_l_ts, norm_err_ts, norm_dr_ts, incl = session.split_sess(
            plot_error=plot_error, all_levers=plot_all)

        sch_type = session.get_arrays('Trial Type')
        ratio_c = plt.cm.get_cmap('Wistia')
        interval_c = plt.cm.get_cmap('winter')
        ax.set_xlim(0, 305)
        for i, l in enumerate(norm_l_ts):
            if sch_type[i] == 1 and not int_only:
                ax.step(l, np.arange(l.size), c=ratio_c(i*45), where="post",
                        label='B'+str(i+1)+' - FR', zorder=1)
            elif sch_type[i] == 0:
                ax.step(l, np.arange(l.size), c=interval_c(i*45), where="post",
                        label='B'+str(i+1)+' - FI', zorder=1)
            bins = l
            reward_y = np.digitize(norm_r_ts[i], bins) - 1
            double_y = np.digitize(norm_dr_ts[i], bins) - 1
            if stage == '7' and plot_all:  # plots all responses incl. errors
                ax.scatter(norm_err_ts[i], np.isin(
                    l, norm_err_ts[i]).nonzero()[0],
                    c='r', s=1, zorder=2)
                incl = '_All'
            if int_only:
                if sch_type[i] == 0:
                    plt.scatter(norm_r_ts[i], reward_y,
                                marker="x", c="grey", s=25)
                    plt.scatter(norm_dr_ts[i], double_y,
                                marker="x", c="magenta", s=25)
            else:
                plt.scatter(norm_r_ts[i], reward_y,
                            marker="x", c="grey", s=25)
                plt.scatter(norm_dr_ts[i], double_y,
                            marker="x", c="magenta", s=25)
        ax.set_title('\n{}, Block-Split {}'.format(
            subject, incl), color=mycolors(subject), fontsize=10)
        ax.legend(loc='upper left')
        return fig

    elif zoom_sch:  # plots cum graph split into 5 min blocks
        if session_type == '5a_FixedRatio_p':
            sch_type = 'FR'
            ax.set_title('\n{}, FR{} Split'.format(
                subject, ratio), color=mycolors(subject),
                fontsize=10)
        elif session_type == '5b_FixedInterval_p':
            sch_type = 'FI'
            ax.set_title('\n{}, FI{}s Split'.format(
                subject, interval), color=mycolors(subject),
                fontsize=10)
        elif stage == 6 or stage == 7:
            pass
        else:
            return print("Unable to split session")
        # Change values to set division blocks
        blocks = np.arange(0, 60*30, 300)
        norm_r_ts, norm_l_ts, norm_err_ts, norm_dr_ts, _ = session.split_sess(
            blocks)
        ax.set_xlim(0, 305)
        for i, l in enumerate(norm_l_ts):
            ax.step(l, np.arange(l.size), c=mycolors(i), where="post",
                    label='B'+str(i+1)+' - {}'.format(sch_type))
            bins = l
            reward_y = np.digitize(norm_r_ts[i], bins) - 1
            double_y = np.digitize(norm_dr_ts[i], bins) - 1
            plt.scatter(norm_r_ts[i], reward_y,
                        marker="x", c="grey", s=25)
            plt.scatter(norm_dr_ts[i], double_y,
                        marker="x", c="magenta", s=25)
        ax.legend(loc='upper left')
        return fig

    else:
        if stage == '7':
            err_lever_ts = session.get_err_lever_ts()
            lever_ts = np.sort(np.concatenate((
                lever_ts, err_lever_ts), axis=None))
        lever_times = np.insert(lever_ts, 0, 0, axis=0)
        ax.step(lever_times, np.arange(
            lever_times.size), c=mycolors(subject),
            where="post", label='Animal'+subject, zorder=1)
        if stage == '7':  # plots error press in red
            ax.scatter(err_lever_ts, np.isin(
                lever_times, err_lever_ts).nonzero()[0],
                c='r', label='Errors', s=1, zorder=2)
        if reward_times[-1] > lever_times[-1]:
            ax.plot(
                [lever_times[-1], reward_times[-1] + 2],
                [lever_times.size - 1, lever_times.size - 1],
                c=mycolors(subject))
        bins = lever_times
        reward_y = np.digitize(reward_times, bins) - 1
        double_y = np.digitize(reward_double, bins) - 1
        # for printing of error rates on graph
        norm_r_ts, _, norm_err_ts, _, _ = session.split_sess(
            all_levers=True)
        sch_type = session.get_arrays('Trial Type')
        for i, l in enumerate(norm_err_ts):
            if sch_type[i] == 1:
                err_FR = err_FR + len(norm_err_ts[i])
            elif sch_type[i] == 0:
                err_FI = err_FI + len(norm_err_ts[i])
        if stage == '6' or stage == '7':
            for i, l in enumerate(norm_r_ts):
                if sch_type[i] == 1:
                    rw_FR = rw_FR + len(norm_r_ts[i])
                elif sch_type[i] == 0:
                    rw_FI = rw_FI + len(norm_r_ts[i])
            rw_print = "\nCorrect FR \\ FI: " + \
                str(rw_FR) + r" \ " + str(rw_FI)

    ax.scatter(reward_times, reward_y, marker="x", c="grey",
               label='Reward Collected', s=25)
    if len(reward_double) > 0:
        dr_print = "\nTotal # of Double Rewards: " + str(len(reward_double))
        ax.scatter(reward_double, double_y, marker="x", c="magenta",
                   label='Double Reward', s=25)
    else:
        dr_print = ""
    ax.legend(loc='lower right')
#    ax.set_xlim(0, 30 * 60 + 30)

    if err_FR > 0 or err_FI > 0:
        err_print = "\nErrors FR \\ FI: " + str(err_FR) + r" \ " + str(err_FI)
    else:
        err_print = ""

    if single_plot:
        out_name = (subject.zfill(3) + "_CumulativeHist_" + date +
                    "_" + session_type[:-2] + ".png")
        out_name = os.path.join(out_dir, out_name)
        print("Saved figure to {}".format(out_name))
        # Text Display on Graph
        if stage == '6' or stage == '7':
            text = (
                'Total # of Lever Press: ' +
                '{}\nTotal # of Rewards: {}{}{}{}'.format(
                    len(lever_ts), len(reward_times) + len(reward_double),
                    dr_print, rw_print, err_print))
            ax.text(0.55, 0.15, text, transform=ax.transAxes)
        bv_plot.savefig(fig, out_name)
        plt.close()
    else:
        # Text Display on Graph
        if stage == '6' or stage == '7':
            text = (
                'Total # of Lever Press: ' +
                '{}\nTotal # of Rewards: {}{}{}{}'.format(
                    len(lever_ts), len(reward_times) + len(reward_double),
                    dr_print, rw_print, err_print))
            ax.text(0.05, 0.75, text, transform=ax.transAxes)
        return fig


def IRT(session, out_dir, ax=None, showIRT=False):
    """
    Perform an inter-response time plot for a Session.

    IRT calculated from prev reward to next lever press resulting in reward
    """
    single_plot = False

    # General session info
    date = session.get_metadata('start_date').replace('/', '_')
    session_type = session.get_metadata('name')
    stage = session_type[:2].replace('_', '')
    subject = session.get_metadata('subject')
    ratio = session.get_ratio()
    interval = session.get_interval()

    # Timestameps data extraction
    time_taken = session.time_taken()
    # lever ts without unnecessary presses
    good_lever_ts = session.get_lever_ts(False)

    # Only consider rewards for lever pressing
    rewards_i = session.get_rw_ts()
    reward_idxs = np.nonzero(rewards_i >= good_lever_ts[0])
    rewards = rewards_i[reward_idxs]

    # b assigns ascending numbers to rewards within lever presses
    b = np.digitize(rewards, bins=good_lever_ts)
    _, a = np.unique(b, return_index=True)  # returns index for good rewards

    good_rewards = rewards[a]  # nosepoke ts for pressing levers
    if session_type == '5a_FixedRatio_p':
        last_lever_ts = []
        for i in b:
            last_lever_ts.append(good_lever_ts[i-1])
    else:
        last_lever_ts = good_lever_ts
    if len(last_lever_ts[1:]) > len(good_rewards[:-1]):
        IRT = last_lever_ts[1:] - good_rewards[:]  # Ended sess w lever press

    else:
        # Ended session w nosepoke
        IRT = last_lever_ts[1:] - good_rewards[:-1]

    hist_count, hist_bins, _ = ax.hist(
        IRT, bins=math.ceil(np.amax(IRT)),
        range=(0, math.ceil(np.amax(IRT))), color=mycolors(subject))

    # Plotting of IRT Graphs
    if ax is None:
        single_plot = True
        fig, ax = plt.subplots()
        ax.set_title('Inter-Response Time\n', fontsize=15)
        if session_type == '5a_FixedRatio_p':
            plt.suptitle('\n({}, {} {}, {})'.format(
                subject, session_type[:-2], ratio, date),
                fontsize=10, y=.98, x=.51)
        elif session_type == '5b_FixedInterval_p':
            plt.suptitle('\n({}, {} {}s, {})'.format(
                subject, session_type[:-2], interval, date),
                fontsize=10, y=.98, x=.51)
        else:
            plt.suptitle('\n({}, {}, {})'.format(
                subject, session_type[:-2], date),
                fontsize=10, y=.98, x=.51)
    else:
        ax.set_title('\n{}, S{}, IRT'.format(
            subject, stage), color=mycolors(subject),
            fontsize=10, y=1, x=.51)

    ax.set_xlabel('IRT (s)')
    ax.set_ylabel('Counts')
    maxidx = np.argmax(np.array(hist_count))
    maxval = (hist_bins[maxidx + 1] - hist_bins[maxidx]) / \
        2 + hist_bins[maxidx]
    ax.text(0.45, 0.85, 'Session Duration: {} mins\nMost Freq. IRT Bin: {} s'
            .format(time_taken, maxval), transform=ax.transAxes)

    if showIRT:
        show_IRT_details(IRT, maxidx, hist_bins)
    if single_plot:
        # Text Display on Graph
        text = (
            'Session Duration: ' +
            '{} mins\nMost Freq. IRT Bin: {} s'.format(
                time_taken, maxval))
        ax.text(0.55, 0.8, text, transform=ax.transAxes)
        out_name = (subject.zfill(3) + "_IRT_Hist_" +
                    session_type[:-2] + "_" + date + ".png")
        print("Saved figure to {}".format(
            os.path.join(out_dir, out_name)))
        bv_plot.savefig(fig, os.path.join(out_dir, out_name))
        plt.close()
    else:
        return ax


def show_IRT_details(IRT, maxidx, hist_bins):
    """Display further information for an IRT."""
    plt.show()
    print('Most Freq. IRT Bin: {} s'.format(
        ((hist_bins[maxidx + 1] - hist_bins[maxidx]) / 2) + hist_bins[maxidx]))
    print(
        'Median Inter-Response Time (IRT): {0:.2f} s'.format(np.median(IRT)))
    print('Min IRT: {0:.2f} s'.format(np.amin(IRT)))
    print('Max IRT: {0:.2f} s'.format(np.amax(IRT)))
    print('IRTs: ', np.round(IRT, decimals=2))


def trial_length_hist(s, ax=None, loop=None, sub_colors_dict=None):
    ''' 
    Plot histrogram of trial durations 

    Parameters
    ----------
    s : session object
    ax : plt.axe, default None
        Optional ax object to plot into.
    loop: str, None
        indicates number of iterations through plot without changing axis
        *Mainly used to identify morning/afternoon sessions for each animal
    sub_colors_dict: dict, None
        dict with subject: colors, used to assign color to title

    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = None

    t_df = s.get_trial_df_norm()

    # Trial duration in ms for FR and FI
    t_len = {
        'FR': t_df[t_df['Schedule'] == 'FR']['Reward_ts'],
        'FI': t_df[t_df['Schedule'] == 'FI']['Reward_ts']}

    if loop:
        txt = "_p" + str(loop)
    else:
        txt = ""

    log = False
    if log:
        sns.distplot(np.log(t_len['FR']), ax=ax,
                     label='FR{}'.format(txt))
        sns.distplot(np.log(t_len['FI']), ax=ax,
                     label='FI{}'.format(txt))
    else:
        sns.distplot((t_len['FR']), ax=ax,
                     label='FR{}'.format(txt))
        sns.distplot((t_len['FI']), ax=ax,
                     label='FI{}'.format(txt))

    # Plot customization
    date = s.get_metadata('start_date').replace('/', '_')
    sub = s.get_metadata('subject')
    stage = s.get_stage()
    plot_name = 'Hist'

    ax.tick_params(axis='both', labelsize=15)
    ax.set_xlabel('Time (s)', fontsize=20)
    # ax.set_ylabel('Trials', fontsize=20)
    if sub_colors_dict == None:
        color = "k"
    else:
        color = mycolors(sub, sub_colors_dict)
    ax.set_title('{} {} S{} {}'.format(sub, date, stage, plot_name),
                 y=1.02, fontsize=25, color=color)
    ax.legend(fontsize=20)
    return fig


def plot_raster_trials(s, ax=None, sub_colors_dict=None, align=[1, 0, 0]):
    ''' 
    Plot raster of behaviour related ts aligned to different points.

    Parameters
    ----------
    s : session object
    ax : plt.axe, default None
        Optional ax object to plot into.
    sub_colors_dict: dict, None
        dict with subject: colors, used to assign color to title
    align : list of bool, [1, 0, 0]
        input structure - [align_rw, align_pell, align_FI]


    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
    else:
        fig = None

    # alignment decision
    align_rw, align_pell, align_FI = [0, 1, 0]

    # Retrive session related variables
    date = s.get_metadata('start_date').replace('/', '_')
    sub = s.get_metadata('subject')
    stage = s.get_stage()
    trial_df = s.get_trial_df_norm()

    norm_lever = []
    norm_err = []
    norm_dr = []
    norm_pell = []
    norm_rw = []
    schedule_type = []

    # Extract data from pandas_df
    norm_lever[:] = trial_df['Levers_ts']
    norm_err[:] = trial_df['Err_ts']
    norm_dr[:] = trial_df['D_Pellet_ts']
    norm_pell[:] = trial_df['Pellet_ts']
    norm_rw[:] = trial_df['Reward_ts']
    schedule_type[:] = trial_df['Schedule']

    color = []

    # Alignment Specific Parameters
    if align_rw:
        plot_type = 'Reward-Aligned'
        norm_arr = np.copy(norm_rw)

    elif align_pell:
        plot_type = 'Pell-Aligned'
        norm_arr = np.copy(norm_pell)
        xmax = 5
        xmin = -30
    elif align_FI:
        plot_type = 'Interval-Aligned'
        norm_arr = np.empty_like(norm_rw)
        norm_arr.fill(30)
    else:
        plot_type = 'Start-Aligned'
        norm_arr = np.zeros_like(norm_rw)
        xmax = 60
        xmin = 0
    plot_name = 'Raster ({})'.format(plot_type)

    ax.axvline(0, linestyle='-.', color='g',
               linewidth=1)
    ax.text(0.1, -1, plot_type.split('-')[0], fontsize=12,
            color='g', ha='left', va='top')

    for i, _ in enumerate(norm_rw):
        # color assigment for trial type
        if schedule_type[i] == 'FR':
            color.append('black')
        elif schedule_type[i] == 'FI':
            color.append('b')
        else:
            color.append('g')
        norm_lever[i] -= norm_arr[i]
        norm_err[i] -= norm_arr[i]
        norm_dr[i] -= norm_arr[i]
        norm_pell[i] -= norm_arr[i]
        norm_rw[i] -= norm_arr[i]

    # Plotting of raster
    ax.eventplot(norm_lever[:], color=color)
    ax.eventplot(norm_err[:], color='red')
    rw_plot = ax.scatter(norm_rw, np.arange(len(norm_rw)), s=5,
                         color='orange', label='Reward Collection')
    ax.eventplot(
        norm_dr[:], color='magenta')

    # Figure labels
    ax.set_xlim(xmin, xmax)  # Uncomment to set x limit
    ax.set_ylim(-3, len(norm_rw)+3)  # Uncomment to set y limit
    ax.tick_params(axis='both', labelsize=15)
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel('Trials', fontsize=20)

    if sub_colors_dict == None:
        color = "k"
    else:
        color = mycolors(sub, sub_colors_dict)

    ax.set_title('{} {} S{} {}'.format(sub, date, stage, plot_name),
                 y=1.02, fontsize=25, color=color)

    # Legend construction
    from matplotlib import lines
    FR_label = lines.Line2D([], [], color='black', marker='|', linestyle='None',
                            markersize=10, markeredgewidth=1.5, label='FR press')
    FI_label = lines.Line2D([], [], color='b', marker='|', linestyle='None',
                            markersize=10, markeredgewidth=1.5, label='FI press')
    drw_label = lines.Line2D([], [], color='magenta', marker='|', linestyle='None',
                             markersize=10, markeredgewidth=1.5, label='Double Reward')

    ax.legend(handles=[FR_label, FI_label, drw_label, rw_plot], fontsize=12)

    # Highlight specific trials
    hline, h_ref = [], []
    h_dr, h_err, h_fr = [1, 0, 0]

    if h_dr:
        c = 'pink'
        h_ref = norm_dr
    elif h_err:
        c = 'magenta'
        h_ref = norm_err
    elif h_fr:
        c = 'k'
        for s in schedule_type:
            if s == 'FR':
                h_ref.append([1])
            else:
                h_ref.append([])
    else:
        pass

    # highlight if array is not empty
    for i, ts in enumerate(h_ref):
        if len(ts) > 0:
            hline.append(i)
    for l in hline:
        plt.axhline(l, linestyle='-', color=c, linewidth='5', alpha=0.1)

    return fig


def trial_clustering(session, should_pca=False, num_clusts=2):
    if should_pca:
        data = session.perform_pca(should_scale=False)[1]
    else:
        data = session.extract_features()
    cluster = KMeans(num_clusts)
    cluster.fit_predict(data)
    markers = session.trial_df_norm["Schedule"]
    plot_dim1 = 0
    plot_dim2 = 1
    fig, ax = plt.subplots()
    sns.scatterplot(
        data[:, plot_dim1], data[:, plot_dim2], ax=ax,
        style=markers, hue=cluster.labels_)
    plot_loc = os.path.join("PCAclust.png")
    fig.savefig(plot_loc, dpi=400)
