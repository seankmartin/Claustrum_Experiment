# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:19:11 2019

@author: HAMG
"""
import os
import numpy as np
from parse_sessions import SessionExtractor
import bv_analyse as bv_an
from bv_utils import make_dir_if_not_exists


def main(filename):
    out_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\Plots"
#    out_dir = r"G:\out_plots"
    make_dir_if_not_exists(out_dir)

    s_extractor = SessionExtractor(filename, verbose=True)
    print(s_extractor)

    for s in s_extractor:  # Batch run for file
        time_taken = s.time_taken()
        c_session = s.get_lines()
        timestamps = s.get_timestamps()
        lever_ts = s.get_lever_ts()
        good_lever_ts = s.get_lever_ts(False)

        print("Session duration {} mins".format(time_taken))
        if len(timestamps.keys()) == 0:
            print('Not ready for analysis!')
            continue

#       Will need to refactor these
        bv_an.IRT(c_session, timestamps, good_lever_ts,
               time_taken, out_dir, False)
        bv_an.cumplot(c_session, timestamps, lever_ts, out_dir, False)


if __name__ == "__main__":
    # Batch processing of sessions in folder
    in_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\\"
    in_files = os.listdir(in_dir)
    for file in in_files:
        filename = in_dir + file
        if os.path.isfile(filename):
            main(filename)
#    # Running single session files
#    filename = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\!2019-07-25"
#    filename = r"G:\test"
#    main(filename)
