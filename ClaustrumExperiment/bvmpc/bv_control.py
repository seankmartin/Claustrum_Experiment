"""Control script for MEDPC behaviour analysis."""
import os
import numpy as np
from bv_parse_sessions import SessionExtractor, Session
import bv_analyse as bv_an
from bv_utils import make_dir_if_not_exists, print_h5
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#def convert_to_hdf5(filename, out_dir):
#    """Convert all sessions in filename to hdf5 and store in out_dir."""
#    make_dir_if_not_exists(out_dir)
#
#    s_extractor = SessionExtractor(filename, verbose=True)
#
#    for s in s_extractor:  # Batch run for file
#        s.save_to_h5(out_dir)
#
#

def plot_sessions():
    # Parameters for specifying session
    sub_list = ['1', '2', '3', '4', '5', '6']
    s_list = ['5a', '5b']
    d_list = ['07-31']
    in_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\hdf5"
    out_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\Plots"
    
    for sub_list in sub_list:
            
        #extracts hdf5 session based on specification
        s_grp, out_name = extract_hdf5s(in_dir, out_dir, sub_list, s_list, d_list)
    
        # Plots summary of day
        cols = 2
        rows = len(s_grp)
        size_multiplier = 5
        fig = plt.figure(
                figsize=(cols * size_multiplier, rows * size_multiplier), 
                tight_layout=False)
        gs = gridspec.GridSpec(rows, cols, wspace=0.2, hspace=0.3)
            
        for i, s in enumerate(s_grp):
            ax1 = fig.add_subplot(gs[i, 0])
            bv_an.cumplot(s, out_dir, False, ax1)
            ax2 = fig.add_subplot(gs[i, 1])        
            bv_an.IRT(s, out_dir, False, ax2)
            plt.tight_layout()
    
        print("Saved figure to {}".format(
            os.path.join(out_dir, out_name)))
        fig.savefig(os.path.join(out_dir, out_name), dpi=400)
        plt.close()
    


def extract_hdf5s(in_dir, out_dir, sub_list=None, s_list=None, d_list=None):
    '''Extracts specified sessions from hdf5 files '''
    
    def should_use(val, vlist):
        if vlist is None:
            return True
        if val in vlist:
            return True
        return False

    in_files = os.listdir(in_dir)
    s_grp = []
    name_dict = {}
    if sub_list is not None:
        name_dict["sub_list"] = sub_list
    if s_list is not None:
        name_dict["s_list"] = s_list
    if d_list is not None:
        name_dict["d_list"] = d_list
    out_name = "Sum_plot"
    names = ["sub_list", "s_list", "d_list"]
    for name in names:
        out_name = out_name + "_" + str(name_dict.get(name, ""))
    out_name.replace("__", "_")
    out_name.replace("__", "_")
    for file in in_files:
        splits = file.split('_')
        subject = splits[0]
        date = splits[1][:5]
        s_type = splits[3]
        subject_ok = should_use(subject, sub_list)
        type_ok = should_use(s_type, s_list)
        date_ok = should_use(date, d_list)
        if subject_ok and type_ok and date_ok:
            filename = os.path.join(in_dir, file)
            if os.path.isfile(filename):
                session = load_hdf5(filename, out_dir)
                s_grp.append(session)
    print('Total Files extracted: {}'.format(len(s_grp)))
    return s_grp, out_name

                

def load_hdf5(filename, out_dir):
    print_h5(filename)
    session = Session(h5_file=filename)
    print(session)

#    bv_an.IRT(session, out_dir, False)
#    bv_an.cumplot(session, out_dir, False)
    return session


#def run_mpc_file(filename, out_dir):
#    """Take in a filename and out_dir then run the main control logic."""
#    make_dir_if_not_exists(out_dir)
#
#    s_extractor = SessionExtractor(filename, verbose=True)
#
#    for s in s_extractor:  # Batch run for file
#
#        time_taken = s.time_taken()
#        timestamps = s.get_arrays()
#
#        print("Session duration {} mins".format(time_taken))
#        if len(timestamps.keys()) == 0:
#            print('Not ready for analysis!')
#            continue
#
#        bv_an.IRT(s, out_dir, False)
#        bv_an.cumplot(s, out_dir, False)


if __name__ == "__main__":
    """Main control."""
#    # Batch processing of sessions in folder
#    in_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\\"
#    out_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\hdf5"
#    in_files = os.listdir(in_dir)
#    for file in in_files:
#        filename = os.path.join(in_dir, file)
#        if os.path.isfile(filename):
#            main(filename)  # Uncomment to run from mpc file
#            convert_to_hdf5(filename, out_dir)  # Uncomment to convert to hdf5
    
    # Processing specific sessions from hdf5
    plot_sessions()

#        # Running single session files
#        filename = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\!2019-07-31"
#     filename = r"G:\test"
#     filename = r"/home/sean/Documents/Data/!2019-07-22"

#    out_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\Plots"
#    out_dir = r"G:\out_plots"
#    out_dir = r"/home/sean/Documents/Data/results"
#    run_mpc_file(filename, out_dir)

#    filename = r"/home/sean/Documents/Data/h5_files/1_07-22-19_11-30_5a_FixedRatio_p.h5"

#    load_hdf5(filename, out_dir)
