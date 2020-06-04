import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from neurochat.nc_data import NData
from bvmpc.bv_utils import make_dir_if_not_exists


def load_lfp(load_loc, i, data):
    end = str(i + 1)
    if end == "1":
        load_loc = load_loc
    else:
        load_loc = load_loc + end
    data.lfp.load(load_loc)


def get_normalised_diff(s1, s2):
    # MSE of one divided by MSE of main - Normalized squared differnce
    # Symmetric
    return np.sum(np.square(s1 - s2)) / (np.sum(np.square(s1) + np.square(s2)) / 2)
    # return np.sum(np.square(s1 - s2)) / np.sum(np.square(s1))  # Non-symmetric


def compare_lfp(fname, out_base_dir=None, ch=16):
    """
    Parameters
    ----------
    fname : str
        full path name without extension
    out_base_dir : str, None
        Path for desired output location. Default - Saves output to folder named !LFP in base directory.
    ch: int
        Number of LFP channels in session
    """
    if out_base_dir == None:
        out_base_dir = os.path.join(os.path.dirname(fname), r"!LFP")
        make_dir_if_not_exists(out_base_dir)
    load_loc = fname + ".eeg"
    out_name = os.path.basename(fname) + "_SI.csv"
    out_loc = os.path.join(out_base_dir, out_name)

    ndata1 = NData()
    ndata2 = NData()
    grid = np.meshgrid(np.arange(ch), np.arange(ch), indexing="ij")
    stacked = np.stack(grid, 2)
    pairs = stacked.reshape(-1, 2)
    result_a = np.zeros(shape=pairs.shape[0], dtype=np.float32)

    for i, pair in enumerate(pairs):
        load_lfp(load_loc, pair[0], ndata1)
        load_lfp(load_loc, pair[1], ndata2)
        res = get_normalised_diff(ndata1.lfp.get_samples(), ndata2.lfp.get_samples())
        result_a[i] = res

    with open(out_loc, "w") as f:
        headers = [str(i) for i in range(1, ch + 1)]
        out_str = ",".join(headers)
        f.write(out_str)
        out_str = ""
        for i, (pair, val) in enumerate(zip(pairs, result_a)):
            if i % ch == 0:
                f.write(out_str + "\n")
                out_str = ""

            out_str += "{:.2f},".format(val)
            # f.write("({}, {}): {:.2f}\n".format(pair[0], pair[1], val))
        f.write(out_str + "\n")

    reshaped = np.reshape(result_a, newshape=[ch, ch])
    sns.heatmap(reshaped)
    plt.xticks(np.arange(0.5, ch + 0.5), labels=np.arange(1, ch + 1), fontsize=8)
    plt.xlabel("LFP Channels")
    plt.yticks(np.arange(0.5, ch + 0.5), labels=np.arange(1, ch + 1), fontsize=8)
    plt.ylabel("LFP Channels")
    plt.title("Raw LFP Similarity Index")
    fig_path = os.path.join(out_base_dir, os.path.basename(fname) + "_LFP_SI.png")
    print("Saving figure to {}".format(fig_path))
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    return result_a


if __name__ == "__main__":
    # lfp_base_dir = r"F:\Ham Data\Batch 3\A14_CAR-SA6\CAR-SA6_20200228"
    # lfp_base_name = "CAR-SA6_2020-02-28.eeg"
    # # lfp_base_dir = r"F:\Eoin's rat\R2 6-OHDA\15_11_19"
    # # lfp_base_name = "R26OHDA151119.eeg"
    # fname = os.path.join(lfp_base_dir, lfp_base_name)
    # out_base_dir = os.path.join(lfp_base_dir, r"!LFP")
    # make_dir_if_not_exists(out_base_dir)
    # results = compare_lfp(fname, out_base_dir)

    fname = r"F:\Ham Data\Batch 3\A14_CAR-SA6\CAR-SA6_20200228\CAR-SA6_2020-02-28"
    results = compare_lfp(fname)
