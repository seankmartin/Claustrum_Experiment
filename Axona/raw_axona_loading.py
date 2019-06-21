import os
from time import time

import numpy as np
import h5py


def create_hdf_storage(hdf5_file, info, size, channels):
    channel_data = hdf5_file.create_group("channels")
    channel_data.attrs["total_samples"] = 3 * size // info["chunksize"]
    channel_data.attrs["channel_list"] = channels
    shape = (3 * size // info["chunksize"], )
    # shape = (len(channels), 3 * size // info["chunksize"])
    # channel_data.create_dataset(
    #     "voltage", shape, np.int16,
    #     chunks=(1, 3 * size // info["chunksize"]),
    #     compression="lzf")
    for channel in channels:
        channel_data.create_dataset(
            str(channel), shape, np.int16,
            compression="lzf")
    # TODO add other things here such as timings positions etc if needed


def get_file_size(location):
    """ Returns file size in bytes"""
    return os.path.getsize(location)


def init_info():
    remap_channels = [
        32, 33, 34, 35, 36, 37, 38, 39,
        0, 1, 2, 3, 4, 5, 6, 7,
        40, 41, 42, 43, 44, 45, 46, 47,
        8, 9, 10, 11, 12, 13, 14, 15,
        48, 49, 50, 51, 52, 53, 54, 55,
        16, 17, 18, 19, 20, 21, 22, 23,
        56, 57, 58, 59, 60, 61, 62, 63,
        24, 25, 26, 27, 28, 29, 30, 31]
    info = {
        "remap": remap_channels,
        "sample_bytes": 2,
        "channel_bytes": 128,
        "header_bytes": 32,
        "chunksize": 432}
    return info


def extract_channel(chunk, channel, info):
    """ Gets all three samples from a 432 byte chunk"""
    place = info["remap"][channel - 1]
    offset = info["header_bytes"] + (place * info["sample_bytes"])
    samples = np.empty(3, dtype=np.int16)
    for i in range(3):
        start_idx = i * info["channel_bytes"] + offset
        # Later use np.int16 to use two's complement
        samples[i] = chunk[start_idx + 1] * 256 + chunk[start_idx]
    return samples


def read_axona_raw(in_location, out_location, channels="all"):
    """Extract certain channel information from the axona bin file"""
    info = init_info()
    size = get_file_size(in_location)
    counter = 0
    if channels == "all":
        channels = [i for i in range(1, 65)]

    write_rate = 100000
    start_time = time()
    total_samples = 3 * size // info["chunksize"]
    with h5py.File(out_location, mode="w", libver="latest") as hdf5_file:
        hdf5_file.swmr_mode = True
        create_hdf_storage(hdf5_file, info, size, channels)
        write_set = hdf5_file["channels"]
        temp = np.empty((len(channels), write_rate * 3), np.int16)
        with open(in_location, 'rb') as file:
            try:
                start_write = time()
                chunk = file.read(info["chunksize"])
                while chunk:
                    for i, channel in enumerate(channels):
                        sample = extract_channel(chunk, channel, info)
                        t_counter = counter % write_rate
                        temp[i, 3 * t_counter: 3 * (t_counter + 1)] = sample
                    chunk = file.read(info["chunksize"])
                    counter += 1
                    if counter % write_rate == 0:
                        for i, channel in enumerate(channels):
                            wr = write_set[str(channel)]
                            wr[3 * counter - 3 * write_rate:3 * counter] = (
                                temp[i, :])
                        print(
                            "Currently on {} out of {} took {:2f}".format(
                                counter,
                                size // info["chunksize"],
                                time() - start_write) +
                            " seconds to write last chunk")
                        start_write = time()

                for i, channel in enumerate(channels):
                    wr = write_set[str(channel)]
                    left = total_samples % (3 * write_rate)
                    start = total_samples - left
                    wr[start:] = temp[i, :left]

            except Exception as e:
                log_exception(e, "on run {}".format(counter))
    print(
        "Finished writing to HDF5 at {} in {:2f} seconds".format(
            out_location, time() - start_time))


def log_exception(ex, more_info=""):
    """
    Log an expection and additional info

    Parameters
    ----------
    ex : Exception
        The python exception that occured
    more_info :
        Additional string to log

    Returns
    -------
    None

    """

    template = "{0} because exception of type {1} occurred. Arguments:\n{2!r}"
    message = template.format(more_info, type(ex).__name__, ex.args)
    print(message)


if __name__ == "__main__":
    in_location = r"C:\Users\smartin5\Recordings\Raw_1min-20190619T104708Z-001\Raw_1min\190619_LCA4_1m_raw.bin"
    out_location = r"C:\Users\smartin5\Recordings\Raw_1min-20190619T104708Z-001\Raw_1min\hdf_vals.h5"
    read_axona_raw(in_location, out_location)
