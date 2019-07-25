# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:19:11 2019

Written by Sean Martin and Gao Xiang Ham
"""

import numpy as np
# import h5py
from datetime import datetime
from session_config import SessionInfo


# def create_hdf5_storage(hdf5_file, config):
#     for


def lever_ts(c_session, data, includeUN=True):
    if c_session[8] == 'MSN: 4_LeverTraining_p':
        if includeUN:
            lever_ts = np.sort(np.concatenate(
                (data[3], data[7], data[4], data[5]), axis=None))
        else:
            lever_ts = np.sort(np.concatenate((data[3], data[7]), axis=None))
    elif c_session[8] == 'MSN: 5a_FixedRatio_p':
        if includeUN:
            lever_ts = np.sort(np.concatenate((data[6], data[4]), axis=None))
        else:
            lever_ts = data[6]
    elif c_session[8] == 'MSN: 5b_FixedInterval_p':
        if includeUN:
            lever_ts = np.sort(np.concatenate((data[3], data[5]), axis=None))
        else:
            lever_ts = data[5]
#    print("Lever Responses at:", lever_ts)
    return lever_ts


def session_data(c_session, dispPara=False):
    print("")
    print(c_session[0][-14:])  # Date
    print(c_session[2])  # Subject number
    print(c_session[8])  # Session Type
    print("")

    s_info = SessionInfo()
    data_info = s_info.get_session_type_info(c_session[8])
    if data_info is None:
        print("Unable to parse information for {}".format(c_session[8]))
        return

    data = []
    if dispPara:
        i = 0
        print("Parameters extracted:")
        for start_char, end_char, parameter in data_info:
            c_data = raw_data(c_session, start_char, end_char)
            print(i, '-> {}: {}'.format(parameter, len(c_data)))
            data.append(c_data)
            i += 1
        print('')
    else:
        for start_char, end_char, parameter in data_info:
            c_data = raw_data(c_session, start_char, end_char)
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


def raw_data(lines, start_char, end_char):
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


def time_taken(c_session, data):
    start_t = c_session[6][-8:]
    end_t = c_session[7][-8:]
    fmt = '%H:%M:%S'
    tdelta = datetime.strptime(end_t, fmt) - datetime.strptime(start_t, fmt)
    tdelta_mins = int(tdelta.total_seconds() / 60)
    return tdelta_mins


def parse_line(line, dtype=np.float32):
    return np.array(line.lstrip().split()[1:]).astype(dtype)
