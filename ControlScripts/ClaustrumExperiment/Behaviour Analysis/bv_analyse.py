# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:19:11 2019

@author: HAMG
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import os.path


def cumplot(session, out_dir, smooth=False, ax=None):
    date = session.get_metadata('start_date')[-8:].replace('/', '_')
    timestamps = session.get_timestamps()
    lever_ts = session.get_lever_ts()
    session_type = session.get_metadata('name')
    subject = session.get_metadata('subject')


    # You have the array sorted, no need to histogram
    reward_times = timestamps["Nosepoke"]
    fig, ax = plt.subplots()
    ax.set_title('Cumulative Lever Presses\n', fontsize=15)
    if session_type == 'MSN: 5a_FixedRatio_p':
        ratio = int(timestamps["Experiment Variables"][3])
        plt.suptitle('\n(Subject {}, {} {}, {})'.format(
            subject[9:], session_type[5:-2], ratio, date),
            fontsize=9, y=.98, x=.51)
    elif session_type == 'MSN: 5b_FixedInterval_p':
        interval = int(timestamps["Experiment Variables"][3] / 100)
        plt.suptitle('\n(Subject {}, {} {}s, {})'.format(
            subject[9:], session_type[5:-2], interval, date),
            fontsize=9, y=.98, x=.51)
    else:
        plt.suptitle('\n(Subject {}, {}, {})'.format(
           subject[9:], session_type[5:-2], date),
            fontsize=9, y=.98, x=.51)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Lever Presses')

    if smooth:
        values, base = np.histogram(lever_ts, bins=len(lever_ts) * 4)
        cumulative = np.cumsum(values)
        plot_arr_x = np.append(base[:-1], base[-1] + 50)
        plot_arr_y = np.append(cumulative, cumulative[-1])
        ax.plot(plot_arr_x, plot_arr_y, c='blue')
        bins = base[:-1]

    else:
        lever_times = np.insert(lever_ts, 0, 0, axis=0)
        ax.step(lever_times, np.arange(
            lever_times.size), c='blue', where="post")
        ax.plot(
            [lever_times[-1], lever_times[-1] + 40],
            [lever_times.size - 1, lever_times.size - 1],
            c='blue', label='Lever Response')
        bins = lever_times

    reward_y = np.digitize(reward_times, bins) - 1

    if smooth:
        reward_y = cumulative[reward_y]

    out_name = (subject[9:].zfill(3) + "_CumulativeHist_" +
                session_type[5:-2] + "_" + date + ".png")   
    out_name = os.path.join(out_dir, out_name)
    print("Saved figure to {}".format(out_name))
    plt.scatter(reward_times, reward_y, marker="x", c="r",
                label='Reward Collected')
    # Text Display on Graph
    ax.text(0.55, 0.15, 'Total # of Lever Press: {}\nTotal # of Rewards: {}'
        .format(len(lever_ts),len(reward_times)), transform=ax.transAxes)

    ax.legend()
    ax.set_xlim(0, 30 * 60)
    fig.savefig(out_name, dpi=400)
    plt.close()


def IRT(session, out_dir, showIRT=False, ax=None):
    date = session.get_metadata('start_date')[-8:].replace('/', '_')
    time_taken = session.time_taken()
    timestamps = session.get_timestamps()
    good_lever_ts = session.get_lever_ts(False)
    session_type = session.get_metadata('name')
    subject = session.get_metadata('subject')
    
    rewards_i = timestamps["Reward"]
    nosepokes_i = timestamps["Nosepoke"]
    # Session ended w/o reward collection
    if len(rewards_i) > len(nosepokes_i):
        # Assumes reward collected at end of session
        nosepokes_i = np.append(
            nosepokes_i, [timestamps["Experiment Variables"][0] * 60])
    # Only consider after the first lever press
    reward_idxs = np.nonzero(rewards_i >= good_lever_ts[0])
    rewards = rewards_i[reward_idxs]
    nosepokes = nosepokes_i[reward_idxs]
    # b assigns ascending numbers to rewards within lever presses
    b = np.digitize(rewards, bins=good_lever_ts)
    _, a = np.unique(b, return_index=True)  # returns index for good rewards
    good_nosepokes = nosepokes[a]  # nosepoke ts for pressing levers
    if session_type == 'MSN: 5a_FixedRatio_p':
        ratio = int(timestamps["Experiment Variables"][3])
        good_lever_ts = good_lever_ts[::ratio]
    if len(good_lever_ts[1:]) > len(good_nosepokes[:-1]):
        IRT = good_lever_ts[1:] - good_nosepokes[:]  # Ended sess w lever press
    else:
        IRT = good_lever_ts[1:] - good_nosepokes[:-1]  # Ended session w nosepoke
    fig, ax = plt.subplots()
    hist_count, hist_bins, _ = ax.hist(
        IRT, bins=math.ceil(np.amax(IRT)),
        range=(0, math.ceil(np.amax(IRT))))

    # Plotting of IRT Graphs
    ax.set_title('Inter-Response Time\n', fontsize=15)
    if session_type == 'MSN: 5a_FixedRatio_p':
        fig.suptitle('\n(Subject {}, {} {}, {})'.format(
            subject[9:], session_type[5:-2], ratio, date),
            fontsize=9, y=.98, x=.51)
    elif session_type == 'MSN: 5b_FixedInterval_p':
        interval = int(timestamps["Experiment Variables"][3] / 100)
        fig.suptitle('\n(Subject {}, {} {}s, {})'.format(
            subject[9:], session_type[5:-2], interval, date),
            fontsize=9, y=.98, x=.51)
    else:
        fig.suptitle('\n(Subject {}, {}, {})'.format(
            subject[9:], session_type[5:-2], date),
            fontsize=9, y=.98, x=.51)
    ax.set_xlabel('IRT (s)')
    ax.set_ylabel('Counts')
    maxidx = np.argmax(np.array(hist_count))
    maxval = (hist_bins[maxidx + 1] - hist_bins[maxidx]) / \
        2 + hist_bins[maxidx]

    # Text Display on Graph
    ax.text(0.55, 0.8, 'Session Duration: {} mins\nMost Freq. IRT Bin: {} s'
            .format(time_taken, maxval), transform=ax.transAxes)

    out_name = (subject[9:].zfill(3) + "_IRT_Hist_" +
                session_type[5:-2] + "_" + date + ".png")
    print("Saved figure to {}".format(
        os.path.join(out_dir, out_name)))
    fig.savefig(os.path.join(out_dir, out_name), dpi=400)
    if showIRT:
        show_IRT_details(IRT, maxidx, hist_bins)
#    plt.show()
    plt.close()


def show_IRT_details(IRT, maxidx, hist_bins):
    plt.show()
    print('Most Freq. IRT Bin: {} s'.format((hist_bins[maxidx + 1] -
                                             hist_bins[maxidx]) / 2 + hist_bins[maxidx]))
    print('Median Inter-Response Time (IRT): {0:.2f} s'.format(np.median(IRT)))
    print('Min IRT: {0:.2f} s'.format(np.amin(IRT)))
    print('Max IRT: {0:.2f} s'.format(np.amax(IRT)))
    print('IRTs: ', np.round(IRT, decimals=2))
