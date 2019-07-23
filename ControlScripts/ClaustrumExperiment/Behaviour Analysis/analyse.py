# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:19:11 2019

@author: HAMG
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import extract_session as ex


def cumplot(c_session, data, includeUN=False, smooth=False):
    date = c_session[0][-8:].replace('/', '_')
    lever_ts = ex.lever_ts(c_session, data, includeUN)

    # You have the array sorted, no need to histogram
    reward_times = data[2]
    plt.title('Cumulative Lever Presses\n', fontsize=15)
    if c_session[8] == 'MSN: 5a_FixedRatio_p':
        ratio = int(data[0][3])
        plt.suptitle('\n(Subject {}, {} {}, {})'.format(
                c_session[2][9:], c_session[8][5:-2], ratio, date),
                fontsize=9, y=.98, x=.51)
    elif c_session[8] == 'MSN: 5b_FixedInterval_p':
        interval = int(data[0][3]/100)
        plt.suptitle('\n(Subject {}, {} {}s, {})'.format(
                c_session[2][9:], c_session[8][5:-2], interval, date),
                fontsize=9, y=.98, x=.51)
    else:
        plt.suptitle('\n(Subject {}, {}, {})'.format(
                c_session[2][9:], c_session[8][5:-2], date),
                fontsize=9, y=.98, x=.51)
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Lever Presses')

    if smooth:
        values, base = np.histogram(lever_ts, bins=len(lever_ts) * 4)
        cumulative = np.cumsum(values)
        plot_arr_x = np.append(base[:-1], base[-1] + 50)
        plot_arr_y = np.append(cumulative, cumulative[-1])
        plt.plot(plot_arr_x, plot_arr_y, c='blue')
        bins = base[:-1]

    else:
        lever_times = np.insert(lever_ts, 0, 0, axis=0)
        plt.step(lever_times, np.arange(
            lever_times.size), c='blue', where="post")
        plt.plot(
            [lever_times[-1], lever_times[-1] + 40],
            [lever_times.size - 1, lever_times.size - 1],
            c='blue', label='Lever Response')
        bins = lever_times

    reward_y = np.digitize(reward_times, bins) - 1

    if smooth:
        reward_y = cumulative[reward_y]

    plt.scatter(reward_times, reward_y, marker="x", c="r",
                label='Reward Collected')
#    plt.xlim(900, 1200)
#    plt.ylim(90, 140)
    plt.legend()
#    plt.ylim(0, 140)
    plt.xlim(0, 30*60)
    plt.savefig(c_session[2][9:].zfill(3) + "_CumulativeHist_" +
                c_session[8][5:-2] + "_" + date + ".png", dpi=400)
    plt.close()


def IRT(c_session, data, showIRT=False):
    date = c_session[0][-8:].replace('/', '_')
    lever_ts = ex.lever_ts(c_session, data, False)
    rewards_i = data[1]
    nosepokes_i = data[2]
    # Session ended w/o reward collection
    if len(rewards_i) > len(nosepokes_i):
        # Assumes reward collected at end of session
        nosepokes_i = np.append(nosepokes_i, [data[0][0]*60])
    # Only consider after the first lever press
    reward_idxs = np.nonzero(rewards_i >= lever_ts[0])
    rewards = rewards_i[reward_idxs]
    nosepokes = nosepokes_i[reward_idxs]
    # b assigns ascending numbers to rewards within lever presses
    b = np.digitize(rewards, bins=lever_ts)
    _, a = np.unique(b, return_index=True)  # returns index for good rewards
    good_nosepokes = nosepokes[a]  # nosepoke ts for pressing levers
    if c_session[8] == 'MSN: 5a_FixedRatio_p':
        ratio = int(data[0][3])
        lever_ts = lever_ts[::ratio]
    if len(lever_ts[1:]) > len(good_nosepokes[:-1]):
        IRT = lever_ts[1:] - good_nosepokes[:]  # Ended sess w lever press
    else:
        IRT = lever_ts[1:] - good_nosepokes[:-1]  # Ended session w nosepoke
    fig, ax = plt.subplots()
    hist_count, hist_bins, _ = ax.hist(
            IRT, bins=math.ceil(np.amax(IRT)),
            range=(0, math.ceil(np.amax(IRT))))

    # Plotting of IRT Graphs
    ax.set_title('Inter-Response Time\n', fontsize=15)
    if c_session[8] == 'MSN: 5a_FixedRatio_p':
        fig.suptitle('\n(Subject {}, {} {}, {})'.format(
                c_session[2][9:], c_session[8][5:-2], ratio, date),
                fontsize=9, y=.98, x=.51)
    elif c_session[8] == 'MSN: 5b_FixedInterval_p':
        interval = int(data[0][3]/100)
        plt.suptitle('\n(Subject {}, {} {}s, {})'.format(
                c_session[2][9:], c_session[8][5:-2], interval, date),
                fontsize=9, y=.98, x=.51)
    else:
        fig.suptitle('\n(Subject {}, {}, {})'.format(
                c_session[2][9:], c_session[8][5:-2], date),
                fontsize=9, y=.98, x=.51)
    ax.set_xlabel('IRT (s)')
    ax.set_ylabel('Counts')
    maxidx = np.argmax(np.array(hist_count))
    tdelta_mins = ex.time_taken(c_session, data)
    maxval = (hist_bins[maxidx+1] - hist_bins[maxidx])/2 + hist_bins[maxidx]
    ax.text(0.55, 0.8, 'Session Duration: {} mins\nMost Freq. IRT Bin: {} s'
            .format(tdelta_mins, maxval), transform=ax.transAxes)
    out_name = (c_session[2][9:].zfill(3) + "_IRT_Hist_" +
                c_session[8][5:-2] + "_" + date + ".png")
    print("Saved figure to {}".format(out_name))
    fig.savefig(out_name, dpi=400)
    if showIRT:
        show_IRT_details(IRT, maxidx, hist_bins)
#    plt.show()
    plt.close()


def show_IRT_details(IRT, maxidx, hist_bins):
    plt.show()
    print('Most Freq. IRT Bin: {} s'.format((hist_bins[maxidx+1] -
          hist_bins[maxidx])/2 + hist_bins[maxidx]))
    print('Median Inter-Response Time (IRT): {0:.2f} s'.format(np.median(IRT)))
    print('Min IRT: {0:.2f} s'.format(np.amin(IRT)))
    print('Max IRT: {0:.2f} s'.format(np.amax(IRT)))
    print('IRTs: ', np.round(IRT, decimals=2))
