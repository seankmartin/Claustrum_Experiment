"""This module implements functions related to reading and writing files."""
import os

from bvmpc.bv_session import Session
from bvmpc.bv_session_extractor import SessionExtractor
from bvmpc.lfp_odict import LfpODict

import bvmpc.bv_utils


def extract_sessions(
        in_dir, sub_list=None, s_list=None, d_list=None,
        load_backend="neo", neo_backend="nix"):
    '''Extracts specified sessions from files '''

    def should_use(val, vlist):
        if vlist is None:
            return True
        if val in vlist:
            return True
        return False

    in_files = sorted(os.listdir(in_dir))
    s_grp = []
    for file in in_files:
        splits = file.split('_')
        subject = splits[0]
        subject = str(subject)
        # NOTE date not have year
        date = splits[1][:-3]
        s_type = splits[3]
        subject_ok = should_use(subject, sub_list)
        type_ok = should_use(s_type, s_list)
        date_ok = should_use(date, d_list)
        if subject_ok and type_ok and date_ok:
            filename = os.path.join(in_dir, file)
            if os.path.isfile(filename):
                if load_backend == "neo":
                    session = load_neo(filename)
                elif load_backend == "hdf5":
                    session = load_hdf5(filename)
                else:
                    print("Backend {} invalid, using neo".format(
                        load_backend))
                    session = load_neo(filename)

                s_grp.append(session)
    print('Total Files extracted: {}'.format(len(s_grp)))
    return s_grp


def convert_to_hdf5(filename, out_dir):
    """Convert all sessions in filename to hdf5 and store in out_dir."""
    bvmpc.bv_utils.make_dir_if_not_exists(out_dir)

    s_extractor = SessionExtractor(filename, verbose=True)

    for s in s_extractor:  # Batch run for file
        stage = s.get_metadata('name')
        if stage not in s.session_info.session_info_dict.keys():
            continue
        else:
            s.save_to_h5(out_dir)


def convert_to_neo(filename, out_dir, neo_backend="nix", remove_existing=False):
    """Convert all sessions in filename to hdf5 and store in out_dir."""
    bvmpc.bv_utils.make_dir_if_not_exists(out_dir)
    print("Converting files in {} to neo".format(
        os.path.basename(filename)))
    s_extractor = SessionExtractor(
        filename, verbose=False)

    for s in s_extractor:  # Batch run for file
        stage = s.get_metadata('name')
        if stage not in s.session_info.session_info_dict.keys():
            continue
        else:
            s.save_to_neo(
                out_dir, neo_backend=neo_backend,
                remove_existing=remove_existing)


def convert_axona_to_neo(
        filename, out_dir, neo_backend="nix", remove_existing=False):
    """Convert .inp files to Sessions and store in out_dir."""
    bvmpc.bv_utils.make_dir_if_not_exists(out_dir)
    print("Converting {} to neo".format(os.path.basename(filename)))
    s = Session(axona_file=filename)
    s.save_to_neo(
        out_dir, neo_backend=neo_backend,
        remove_existing=remove_existing)


def load_hdf5(filename, verbose=False):
    if verbose:
        bvmpc.bv_utils.print_h5(filename)
    session = Session(h5_file=filename)

    return session


def load_neo(filename, neo_backend="nix"):
    session = Session(neo_file=filename, neo_backend="nix")
    return session


def session_from_mpc_file(filename, out_dir):
    """Use this to work on MEDPC files without converting to HDF5."""
    bvmpc.bv_utils.make_dir_if_not_exists(out_dir)

    s_extractor = SessionExtractor(filename, verbose=False)

    return s_extractor


def load_bv_from_set(fname):
    """Loads session based from .inp"""
    if os.path.isfile(fname + ".inp"):
        return Session(axona_file=fname + ".inp")
    elif os.path.isfile(fname[:-3] + "inp"):
        return Session(axona_file=fname[:-3] + "inp")
    else:
        print(".inp does not exist.")
        return None


def select_lfp(fname, ROI):  # Select lfp based on region
    # Single Hemi Multisite Drive settings
    lfp_list = []
    chans = [i for i in range(1, 17)]
    regions = ["CLA"] * 8 + ["ACC"] * 4 + ["RSC"] * 4
    # Change filt values here. Default order 10.
    filt_btm = 1.0
    filt_top = 50

    # Actual function
    for r in ROI:
        idx = [i for i, x in enumerate(regions) if x == ROI[r]]
        lfp_odict = LfpODict(
            fname, channels=chans[idx], filt_params=(True, filt_btm, filt_top))
        lfp_list.append(lfp_odict)
    return lfp_list
