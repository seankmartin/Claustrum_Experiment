import h5py
import os
import matplotlib.pyplot as plt


def main(file_location):
    with h5py.File(file_location, 'r') as f:
        s_rate = 1 / 48000
        x = [s_rate * i for i in range(len(f["channels"]["3"]))]
        plt.plot(x, f["channels"]["11"])
        plt.show()


if __name__ == "__main__":
    file_location = r"C:\Users\smartin5\Recordings\Raw_1min-20190619T104708Z-001\Raw_1min\hdf_vals.h5"
    main(file_location)
