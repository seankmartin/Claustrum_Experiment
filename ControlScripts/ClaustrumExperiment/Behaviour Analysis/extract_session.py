# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:19:11 2019

@author: HAMG
"""

import matplotlib.pyplot as plt
import numpy as np
import math


def main(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()  # reads lines into list
        lines = np.array(
            list(filter(None, lines)))  # removes empty space
        sessions = parse_sessions(lines)
        print_session_info(lines)  # uncomment to print sessions info

#        s_index = 3  # change number based on desired session

    for s_index in np.arange(len(sessions)):  # Batch run for file
        c_session = sessions[s_index]
        # set to True to display parameter index
        data = extract_session_data(c_session, False)
        if not data:
            print('Not ready for analysis!')
        else:
            IRT(c_session, data)
            cumplot(c_session, data, True)


def extract_lever_ts(c_session, data, includeUN=False):
    if c_session[8] == 'MSN: 4_LeverTraining_p':
        if includeUN:
            lever_ts = np.sort(np.concatenate(
                    (data[2], data[6], data[3], data[4]), axis=None))
        else:
            lever_ts = np.sort(np.concatenate((data[2], data[6]), axis=None))
    elif c_session[8] == 'MSN: 5a_FixedRatio_p':
        if includeUN:
            lever_ts = np.sort(np.concatenate((data[5], data[3]), axis=None))
        else:
            lever_ts = data[5]
#    print("Lever Responses at:", lever_ts)
    return lever_ts


def cumplot(c_session, data, includeUN=False, smooth=False):
    date = c_session[0][-8:].replace('/', '_')
    lever_ts = extract_lever_ts(c_session, data, includeUN)

    # You have the array sorted, no need to histogram
    reward_times = data[1]
    plt.title('Cumulative Lever Presses\n', fontsize=15)
    plt.suptitle('\n(Subject {}, {}, {})'.format(
        c_session[2][9:], c_session[8][5:], date), fontsize=9, y=.98, x=.51)
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
    plt.legend()
    plt.savefig(c_session[2][9:] + "_CumulativeHist_" +
                c_session[8][5:] + "_" + date + ".png", dpi=400)
    plt.close()


def IRT(c_session, data):
    date = c_session[0][-8:].replace('/', '_')
    lever_ts = extract_lever_ts(c_session, data, False)
    # b assigns ascending numbers to rewards within lever presses
    b = np.digitize(data[0], bins=lever_ts)
    _, a = np.unique(b, return_index=True)  # returns index for good rewards
    good_nosepokes = data[1][a]  # nosepoke ts for pressing levers
    if c_session[8] == 'MSN: 5a_FixedRatio_p':
        ratio = 3  # !!Need to find correct reference
        lever_ts = lever_ts[::ratio]
    print((good_nosepokes))
    print((lever_ts))
    if len(lever_ts[1:]) > len(good_nosepokes[:-1]):
        IRT = lever_ts[1:] - good_nosepokes[:]  # Ended sess w lever press
    else:
        IRT = lever_ts[1:] - good_nosepokes[:-1]  # Ended session w nosepoke
    hist_count, hist_bins, _ = plt.hist(IRT, bins=math.ceil(np.amax(IRT)),
                                        range=(0, math.ceil(np.amax(IRT))))
    plt.title('Inter-Response Time\n', fontsize=15)
    plt.suptitle('\n(Subject {}, {}, {})'.format(
        c_session[2][9:], c_session[8][5:], date), fontsize=9, y=.98, x=.51)
    plt.xlabel('IRT (s)')
    plt.ylabel('Counts')
    plt.savefig(c_session[2][9:] + "_IRT_Hist_" +
                c_session[8][5:] + "_" + date + ".png", dpi=400)
    plt.close()

    maxidx = np.argmax(np.array(hist_count))
    print('Most Freq. IRT Bin: {} s'.format((hist_bins[maxidx+1] -
          hist_bins[maxidx])/2 + hist_bins[maxidx]))
    print('Median Inter-Response Time (IRT): {0:.2f} s'.format(np.median(IRT)))
    print('Min IRT: {0:.2f} s'.format(np.amin(IRT)))
    print('Max IRT: {0:.2f} s'.format(np.amax(IRT)))
    print('IRTs: ', np.round(IRT, decimals=2))


def extract_session_data(c_session, dispPara=False):
    print("")
    print(c_session[0][-14:])  # Date
    print(c_session[2])  # Subject number
    print(c_session[8])  # Session Type
    print("")

    if c_session[8] == 'MSN: 2_MagazineHabituation_p':
        return print('To be updated...')
    elif c_session[8] == 'MSN: 3_LeverHabituation_p':
        return print('To be updated...')
    elif c_session[8] == 'MSN: 4_LeverTraining_p':
        data_info = np.array([['D:', 'E:', 'Reward'],
                              ['E:', 'L:', 'Nosepoke'],
                              ['L:', 'M:', 'L'],
                              ['M:', 'N:', 'Un_L'],
                              ['N:', 'O:', 'Un_R'],
                              ['O:', 'R:', 'Un_Nosepoke'],
                              ['R:', 'END', 'R']])
    elif c_session[8] == 'MSN: 5a_FixedRatio_p':
        data_info = np.array([['D:', 'E:', 'Reward'],
                              ['E:', 'M:', 'Nosepoke'],
                              ['M:', 'N:', 'FR Changes'],
                              ['N:', 'O:', 'Un_R'],
                              ['O:', 'R:', 'Un_Nosepoke'],
                              ['R:', 'END', 'R']])
    elif c_session[8] == 'MSN: 5b_FixedInterval_p':
        return print('To be updated...')
    elif c_session[8] == 'MSN: DNMTS':
        return print('To be updated...')
    else:
        return print('Error! Invalid session type!')
    data = []
    if dispPara:
        i = 0
        print("Parameters extracted:")
        for start_char, end_char, parameter in data_info:
            c_data = extract_data(c_session, start_char, end_char)
            data.append(c_data)
            print(i, '->', parameter)
            i += 1
        print('')
    else:
        for start_char, end_char, parameter in data_info:
            c_data = extract_data(c_session, start_char, end_char)
            data.append(c_data)
    return data


def print_session_info(lines):
    s_list = np.flatnonzero(
        np.core.defchararray.find(lines, "MSN:") != -1)
    # returns index in np.array for cells containing "MSN:"
    a_list = np.flatnonzero(
        np.core.defchararray.find(lines, "Subject:") != -1)
    p_list = np.stack((a_list, s_list), axis=-1)
    i = 0
    print('Sessions in file:')
    for a, s in p_list:
        print(i, '->', lines[a], ',', lines[s])
        i += 1
    return print('')


def parse_sessions(lines):
    s_starts = np.flatnonzero(
        np.core.defchararray.find(lines, "Start Date:") != -1)
    s_ends = np.zeros_like(s_starts)
    s_ends[:-1] = s_starts[1:]
    s_ends[-1] = lines.size
    sessions = []
    for start, end in zip(s_starts, s_ends):
        s_data = np.array(lines[start:end])
        sessions.append(s_data)
    return sessions

# =============================================================================
# #  Previous method to parse sessions based on specific session types
# def parse_session_type(lines, s_type):
#     if s_type == '2':
#         s_type = 'MSN: 2_MagazineHabituation_p'
#     elif s_type == '3':
#         s_type = 'MSN: 3_LeverHabituation_p'
#     elif s_type == '4':
#         s_type = 'MSN: 4_LeverTraining_p'
#     elif s_type == '5a':
#         s_type = 'MSN: 5a_FixedRatio_p'
#     elif s_type == '5b':
#         s_type = 'MSN: 5b_FixedInterval_p'
#     elif s_type == 'DNMTS':
#         s_type = 'MSN: DNMTS'
#     else:
#         print('Error! Invalid session type!')
#         return None
#
#     id_increment = 8
#     s_starts = np.flatnonzero(
#         np.core.defchararray.find(lines, "Start Date:") != -1)
#     s_ends = np.zeros_like(s_starts)
#     s_ends[:-1] = s_starts[1:]
#     s_ends[-1] = lines.size
#     s_identifiers = lines[s_starts + id_increment]
#     s_id_indices = np.nonzero(s_identifiers == s_type)
#     s_id_starts = s_starts[s_id_indices]
#     s_id_ends = s_ends[s_id_indices]
#
#     sessions = []
#     for start, end in zip(s_id_starts, s_id_ends):
#         s_data = np.array(lines[start:end])
#         sessions.append(s_data)
#     return sessions
# =============================================================================


def extract_data(lines, start_char, end_char):
    start_index = np.flatnonzero(lines == start_char)
    stop_index = np.flatnonzero(lines == end_char)
    if end_char == 'END':
        stop_index = [lines.size]  # Last timepoint does not have a end_char
    data_list = []
    for start, end in zip(start_index, stop_index):
        data_lines = lines[start + 1:end]
        if not data_lines.size:
            arr = np.array([])
        else:
            last_line = parse_line(data_lines[-1])
            arr = np.empty(
                5 * (len(data_lines) - 1) + len(last_line),
                dtype=np.float32)
            for i, line in enumerate(data_lines):
                numbers = parse_line(line)
                st = 5 * i
                arr[st:st + len(numbers)] = numbers
        data_list.append(arr)
#        print(len(data_list[0]))
    return data_list[0]


def parse_line(line, dtype=np.float32):
    return np.array(line.lstrip().split()[1:]).astype(dtype)


if __name__ == "__main__":
    filename = r"E:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\!2019-07-22"
#    filename = r"G:\test"
    main(filename)
