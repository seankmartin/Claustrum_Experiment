import numpy as np
from bvmpc.bv_session_extractor import SessionExtractor
from bvmpc.bv_utils import make_dir_if_not_exists
import os


def correct_name(name):
    return (
        name == 'Match_to_sample_0_delay' or
        name == 'DNMTS_0_delay')


def run_mpc_file(filename, outname, n_splits=2, keep_nan=True):
    """Use this to work on MEDPC files without converting to HDF5."""
    make_dir_if_not_exists(os.path.dirname(outname))

    s_extractor = SessionExtractor(filename, verbose=False)

    # Check all session divide equally
    for s in s_extractor:
        if correct_name(s.get_metadata("name")):
            num = len(s.get_arrays("Results"))
            if num == 1000:
                continue
            elif num % n_splits != 0:
                print(
                    "ERROR: {} must divide {}, result is {}".format(
                        n_splits, num, num / n_splits))
            s1 = s
            break

    with open(outname, "w") as f:
        headers = s1.get_metadata().keys()
        header_str = ",".join(headers)
        splits = [
            int((num / n_splits) * (i))
            for i in range(n_splits + 1)]
        split_strs = [
            "Percent in trials {}-{}".format(
                splits[i], splits[i + 1]) for i in range(n_splits)]
        header_str += "," + ",".join(split_strs)
        header_str += ",Total percent correct\n"
        f.write(header_str)

        for s in s_extractor:  # Batch run for file
            if correct_name(s.get_metadata("name")):
                out_str = ""
                for header in headers:
                    val = s.get_metadata(header)
                    out_str += val + ","
                results = np.array(
                    s.get_arrays("Results"), dtype=int)
                if results.size != 0:
                    if (results.size % n_splits != 0):
                        continue
                    total = 100 * np.sum(results) / results.size
                    sub_arrs = np.split(results, n_splits)
                    for arr in sub_arrs:
                        pc_correct = 100 * np.sum(arr) / arr.size
                        out_str += "{:.2f},".format(pc_correct)
                    out_str += "{:.2f}\n".format(total)
                    f.write(out_str)
                elif keep_nan:
                    for i in range(n_splits + 1):
                        out_str += "Nan,"
                    out_str += "\n"
                    f.write(out_str)


def exact_split_divide(filename, n_splits=2):
    s_extractor = SessionExtractor(filename, verbose=False)

    for s in s_extractor:
        if correct_name(s.get_metadata("name")):
            num = len(s.get_arrays("Results"))
            if num % n_splits != 0:
                print(
                    "ERROR: {} must be divide {}, result is {}".format(
                        n_splits, num, num / n_splits))
            s1 = s
            break

    num = len(s1.get_arrays("Results"))
    return (num % n_splits == 0), num


if __name__ == "__main__":
    in_dir = r"C:\Users\smartin5\Google Drive\NeuroScience\Code\MPC_F"
    filename = os.path.join(in_dir, "01_09_19")
    outname = filename + ".csv"
    n_splits = 5
    run_mpc_file(filename, outname, n_splits)
