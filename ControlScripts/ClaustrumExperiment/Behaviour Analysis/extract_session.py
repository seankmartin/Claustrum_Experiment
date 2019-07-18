# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:19:11 2019

@author: HAMG
"""
import numpy as np

fileName = "E:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\!2019-07-17"

with open(fileName, 'r') as f:
    lineList = f.read().splitlines()  # reads lines into list
    lineList = np.array(list(filter(None, lineList)))  # removes empty spaces
#    print(lineList)
    arr_index = np.where(lineList == 'E:')
    stop_index = np.where(lineList == "L:")
#    print(arr_index)
    for start, end in zip(arr_index[0], stop_index[0]):
        data_lines = lineList[start+1:end]
        for line in data_lines:
            line = line.lstrip()
#            for i, el in enumerate(line):
#                numbers = line.split("   ")
#            print(numbers)
#            for num in numbers[2:]:
#                print(num)
     