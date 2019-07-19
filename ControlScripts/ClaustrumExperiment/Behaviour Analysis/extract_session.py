# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:19:11 2019

@author: HAMG
"""
import numpy as np


def main(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()  # reads lines into list
        lines = np.array(list(filter(None, lines))
                         )  # removes empty spaces
    info = extract_information(lines, "M:", "N:")
    print(info)


def extract_information(lines, start_char, end_char):
    arr_index = np.where(lines == start_char)
    stop_index = np.where(lines == end_char)
    data_list = []
    for start, end in zip(arr_index[0], stop_index[0]):
        data_lines = lines[start + 1:end]
        last_line = parse_line(data_lines[-1])
        arr = np.empty(
            5 * (len(data_lines) - 1) + len(last_line),
            dtype=np.float32)
        for i, line in enumerate(data_lines):
            numbers = parse_line(line)
            st = 5 * i
            arr[st:st + len(numbers)] = numbers
        data_list.append(arr)
    return data_list


def parse_line(line, dtype=np.float32):
    return np.array(line.lstrip().split()[1:]).astype(dtype)


if __name__ == "__main__":
    # filename = r"E:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\!2019-07-17"
    filename = r"G:\t"
    main(filename)
