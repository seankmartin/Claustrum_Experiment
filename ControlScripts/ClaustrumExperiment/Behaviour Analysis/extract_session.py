# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:19:11 2019

@author: HAMG
"""
import numpy as np


def main(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()  # reads lines into list
        lines = np.array(
            list(filter(None, lines)))  # removes empty space
        sessions = parse_sessions(lines)
        print_session_info(lines)

#    s_type = '4'  # change number based on desired session type
#    session = parse_session_type(lines, s_type)
    
    s_index = 1  # change number based on desired session
    data = extract_data(sessions[s_index], "L:", "M:")
    print(data)



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
        print(len(s_data))
        sessions.append(s_data)
    return sessions


#def parse_session_type(lines, s_type):
#    #  Extracts specific data from specific session types
#    if s_type == '2':
#        s_type = 'MSN: 2_MagazineHabituation_p'
#    elif s_type == '3':
#        s_type = 'MSN: 3_LeverHabituation_p'
#    elif s_type == '4':
#        s_type = 'MSN: 4_LeverTraining_p'
#    elif s_type == '5a':
#        s_type = 'MSN: 5a_FixedRatio_p'
#    elif s_type == '5b':
#        s_type = 'MSN: 5b_FixedInterval_p'
#    elif s_type == 'DNMTS':
#        s_type = 'MSN: DNMTS'
#    else:
#        print('Error! Invalid session type!')
#        return None
#
#    id_increment = 8
#    s_starts = np.flatnonzero(
#        np.core.defchararray.find(lines, "Start Date:") != -1)
#    s_ends = np.zeros_like(s_starts)
#    s_ends[:-1] = s_starts[1:]
#    s_ends[-1] = lines.size
#    s_identifiers = lines[s_starts + id_increment]
#    s_id_indices = np.nonzero(s_identifiers == s_type)
#    s_id_starts = s_starts[s_id_indices]
#    s_id_ends = s_ends[s_id_indices]
#
#    sessions = []
#    for start, end in zip(s_id_starts, s_id_ends):
#        s_data = np.array(lines[start:end])
#        sessions.append(s_data)
#    return sessions


def extract_data(lines, start_char, end_char):
    start_index = np.where(lines == start_char)
    stop_index = np.where(lines == end_char)
    data_list = []
    for start, end in zip(start_index[0], stop_index[0]):
        data_lines = lines[start + 1:end]
#        print(data_lines)
        last_line = parse_line(data_lines[-1])
        arr = np.empty(
            5 * (len(data_lines) - 1) + len(last_line),
            dtype=np.float32)
        for i, line in enumerate(data_lines):
            numbers = parse_line(line)
            st = 5 * i
            arr[st:st + len(numbers)] = numbers
        data_list.append(arr)
#        print(type(data_list))
    return data_list[0]


def parse_line(line, dtype=np.float32):
    return np.array(line.lstrip().split()[1:]).astype(dtype)


if __name__ == "__main__":
    filename = r"E:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\!2019-07-17"
#    filename = r"G:\test"
    main(filename)
