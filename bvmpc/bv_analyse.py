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
            plt.suptitle('\nSubject {}, {} {}, {}'.format(
                subject, session_type[:-2], ratio, date),
                color=mycolors(subject), fontsize=10, y=.98, x=.51)
        elif session_type == '5b_FixedInterval_p':
            plt.suptitle('\nSubject {}, {} {}s, {}'.format(
                subject, session_type[:-2], interval, date),
                color=mycolors(subject), fontsize=10, y=.98, x=.51)
        elif session_type == '6_RandomisedBlocks_p:':
            plt.suptitle('\nSubject {}, {} FR{}/FI{}s, {}'.format(
                subject, session_type[:-2], ratio, interval, date),
                color=mycolors(subject), fontsize=10, y=.98, x=.51)
        else:
            plt.suptitle('\nSubject {}, S{}, {}'.format(
                subject, stage, date), color=mycolors(subject),
                fontsize=10, y=.98, x=.51)
    else:
        if session_type == '5a_FixedRatio_p':
            ax.set_title('\nSubject {}, S{}, FR{}, {}'.format(
                subject, stage, ratio, date),
                color=mycolors(subject), fontsize=10)
        elif session_type == '5b_FixedInterval_p':
            ax.set_title('\nSubject {}, S{}, FI{}s, {}'.format(
                subject, stage, interval, date),
                color=mycolors(subject), fontsize=10)
        elif session_type == '6_RandomisedBlocks_p' or stage == '7':
            switch_ts = np.arange(5, 1830, 305)
            for x in switch_ts:
                plt.axvline(x, color='g', linestyle='-.', linewidth='.4')
            ax.set_title('\nSubject {}, S{}, FR{}/FI{}s, {}'.format(
                subject, stage, ratio, interval, date),
                color=mycolors(subject), fontsize=10)
        else:
            ax.set_title('\nSubject {}, S{}, {}'.format(
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
        ax.set_title('\nSubject {}, Trial-Based'.format(
            subject), color=mycolors(subject), fontsize=10)
        ax.legend(loc='lower right')
        return

    elif zoom_sch and (session_type == '6_RandomisedBlocks_p' or stage == '7'):
        # plots cum graph based on schedule type (i.e. FI/FR)
        norm_r_ts, norm_l_ts, norm_err_ts, norm_dr_ts, incl = session.split_sess(
            plot_error=plot_error, plot_all=plot_all)

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
        ax.set_title('\nSubject {}, Block-Split {}'.format(
            subject, incl), color=mycolors(subject), fontsize=10)
        ax.legend(loc='upper left')
        return

    elif zoom_sch:  # plots cum graph based on schedule type (i.e. FI/FR)
        if session_type == '5a_FixedRatio_p':
            sch_type = 'FR'
            ax.set_title('\nSubject {}, FR{} Split'.format(
                subject, ratio), color=mycolors(subject),
                fontsize=10)
        elif session_type == '5b_FixedInterval_p':
            sch_type = 'FI'
            ax.set_title('\nSubject {}, FI{}s Split'.format(
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
        return

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
            plot_all=True)
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
        fig.savefig(out_name, dpi=400)
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
        return


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
            plt.suptitle('\n(Subject {}, {} {}, {})'.format(
                subject, session_type[:-2], ratio, date),
                fontsize=10, y=.98, x=.51)
        elif session_type == '5b_FixedInterval_p':
            plt.suptitle('\n(Subject {}, {} {}s, {})'.format(
                subject, session_type[:-2], interval, date),
                fontsize=10, y=.98, x=.51)
        else:
            plt.suptitle('\n(Subject {}, {}, {})'.format(
                subject, session_type[:-2], date),
                fontsize=10, y=.98, x=.51)
    else:
        ax.set_title('\nSubject {}, S{}, IRT'.format(
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
        fig.savefig(os.path.join(out_dir, out_name), dpi=400)
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
