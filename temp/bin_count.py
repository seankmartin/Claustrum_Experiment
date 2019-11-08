import numpy as np
from bvmpc.bv_session_extractor import SessionExtractor
from bvmpc.bv_utils import make_dir_if_not_exists
import os


def run_mpc_file(filename, outname, n_splits=2):
    """Use this to work on MEDPC files without converting to HDF5."""
    make_dir_if_not_exists(os.path.dirname(outname))

    s_extractor = SessionExtractor(filename, verbose=False)

    s1 = s_extractor.get_sessions()[0]
    num = len(s1.get_arrays("Results"))
    if num % n_splits != 0:
        print(
            "ERROR: {} must be divide {}, result is {}".format(
                n_splits, num, num / n_splits))

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
            out_str = ""
            for header in headers:
                val = s.get_metadata(header)
                out_str += val + ","
            results = np.array(
                s.get_arrays("Results"), dtype=int)
            total = 100 * np.sum(results) / results.size
            sub_arrs = np.split(results, n_splits)
            for arr in sub_arrs:
                pc_correct = 100 * np.sum(arr) / arr.size
                out_str += "{:.2f},".format(pc_correct)
            out_str += "{:.2f}\n".format(total)
            f.write(out_str)


if __name__ == "__main__":
    in_dir = r"C:\Users\smartin5\Google Drive\NeuroScience\Code\MPC_F"
    filename = os.path.join(in_dir, "01_09_19")
    outname = filename + ".csv"
    n_splits = 5
    run_mpc_file(filename, outname, n_splits)
