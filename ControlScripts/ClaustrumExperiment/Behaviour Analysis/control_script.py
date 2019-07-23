# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:19:11 2019

@author: HAMG
"""

import numpy as np
import extract_session as ex
import analyse as an


def main(filename):
    out_dir = r"E:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\Plots"

    with open(filename, 'r') as f:
        lines = f.read().splitlines()  # reads lines into list
        lines = np.array(
            list(filter(None, lines)))  # removes empty space
        sessions = ex.parse_sessions(lines)
        ex.print_session_info(lines)  # uncomment to print sessions info

#        s_index = 0  # change number based on desired session

    for s_index in np.arange(len(sessions)):  # Batch run for file
        c_session = sessions[s_index]
        # set to True to display parameter index
        data = ex.session_data(c_session, True)
        ex.time_taken(c_session, data)  # extracts time taken for session
        if not data:
            print('Not ready for analysis!')
        else:
            # True prints IRT details on console
            an.IRT(c_session, data, out_dir, False)
            an.cumplot(c_session, data, out_dir, True)


if __name__ == "__main__":
    filename = r"E:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\!2019-07-23"
#    filename = r"G:\test"
    main(filename)
