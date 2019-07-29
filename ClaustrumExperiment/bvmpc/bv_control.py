"""Control script for MEDPC behaviour analysis."""
import os
import numpy as np
from bvmpc.bv_parse_sessions import SessionExtractor
import bvmpc.bv_analyse as bv_an
from bvmpc.bv_utils import make_dir_if_not_exists


def main(filename, out_dir):
    """Take in a filename and out_dir then run the main control logic."""
    make_dir_if_not_exists(out_dir)

    s_extractor = SessionExtractor(filename, verbose=True)
    print(s_extractor)

    for s in s_extractor:  # Batch run for file
        time_taken = s.time_taken()
        timestamps = s.get_arrays()

        print("Session duration {} mins".format(time_taken))
        if len(timestamps.keys()) == 0:
            print('Not ready for analysis!')
            continue

        # Will need to refactor these
        bv_an.IRT(s, out_dir, False)
        bv_an.cumplot(s, out_dir, False)
        # bv_an.cumplot(s, out_dir, False, ax)  #plot multiple sessions


if __name__ == "__main__":
    """Main control."""
    #    # Batch processing of sessions in folder
    #    in_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\\"
    #    in_files = os.listdir(in_dir)
    #    for file in in_files:
    #        filename = in_dir + file
    #        if os.path.isfile(filename):
    #            main(filename)
    #    # Running single session files
    filename = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\!2019-07-28"
    # filename = r"G:\test"

    out_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\Plots"
    # out_dir = r"G:\out_plots"
    main(filename, out_dir)
