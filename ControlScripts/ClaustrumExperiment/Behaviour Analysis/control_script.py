# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:19:11 2019

@author: HAMG
"""

import numpy as np
from parse_sessions import SessionExtractor
import analyse as an
from bv_utils import make_dir_if_not_exists


def main(filename):
    # out_dir = r"E:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\Plots"
    out_dir = r"G:\out_plots"
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

        # Will need to refactor these
        an.IRT(c_session, timestamps, good_lever_ts,
               time_taken, out_dir, False)
        an.cumplot(c_session, timestamps, lever_ts, out_dir, False)


if __name__ == "__main__":
    # filename = r"E:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\!2019-07-23"
    filename = r"G:\test"
    main(filename)
