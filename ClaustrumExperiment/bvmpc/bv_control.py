"""Control script for MEDPC behaviour analysis."""
import os
import numpy as np
from bv_parse_sessions import SessionExtractor, Session
import bv_analyse as bv_an
from bv_utils import make_dir_if_not_exists, print_h5


def convert_to_hdf5(filename, out_dir):
    """Convert all sessions in filename to hdf5 and store in out_dir."""
    make_dir_if_not_exists(out_dir)

    s_extractor = SessionExtractor(filename, verbose=True)

    for s in s_extractor:  # Batch run for file
        s.save_to_h5(out_dir)


def load_hdf5(filename, out_dir):
    print_h5(filename)
    session = Session(h5_file=filename)
    print(session)

    bv_an.IRT(session, out_dir, False)
    bv_an.cumplot(session, out_dir, False)
    return session


def run_mpc_file(filename, out_dir):
    """Take in a filename and out_dir then run the main control logic."""
    make_dir_if_not_exists(out_dir)

    s_extractor = SessionExtractor(filename, verbose=True)

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
    # filename = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\!2019-07-28"
    # filename = r"G:\test"
    # filename = r"/home/sean/Documents/Data/!2019-07-22"

    # out_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\Plots"
    # out_dir = r"G:\out_plots"
    out_dir = r"/home/sean/Documents/Data/results"
    # run_mpc_file(filename, out_dir)

    filename = r"/home/sean/Documents/Data/h5_files/1_07-22-19_11-30_5a_FixedRatio_p.h5"

    load_hdf5(filename, out_dir)
