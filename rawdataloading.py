from time import time

remap_channels = [
    32, 33, 34, 35, 36, 37, 38, 39,
    0, 1, 2, 3, 4, 5, 6, 7,
    40, 41, 42, 43, 44, 45, 46, 47,
    8, 9, 10, 11, 12, 13, 14, 15,
    48, 49, 50, 51, 52, 53, 54, 55,
    16, 17, 18, 19, 20, 21, 22, 23,
    56, 57, 58, 59, 60, 61, 62, 63,
    24, 25, 26, 27, 28, 29, 30, 31]

header_bytes = 32

CHUNKSIZE = 432
# Could read and store chunks

# Can read specific channels like this


def extract_channel(chunk, channel):
    place = remap_channels[channel - 1]
    num_bytes_per_sample = 2
    num_bytes_per_channel = 128
    offset = header_bytes + (place * num_bytes_per_sample)
    samples = [None, None, None]
    for i in range(3):
        start_idx = i * num_bytes_per_channel + offset
        samples[i] = chunk[start_idx + 1] * 256 + chunk[start_idx]
    return samples


def read_axona_raw(location, channels="all"):
    counter = 0
    if channels == "all":
        channels = [i for i in range(1, 65)]
    channel_data = [[] for i in range(len(channels))]

    start = time()
    with open(location, 'rb') as file:
        try:
            chunk = file.read(CHUNKSIZE)
            while chunk:
                for i, channel in enumerate(channels):
                    sample = extract_channel(chunk, channel)
                    for val in sample:
                        channel_data[i].append(val)
                chunk = file.read(CHUNKSIZE)
                counter += 1

        except Exception as e:
            log_exception(e, "on run {}".format(counter))
    print(time() - start)


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
    location = r"C:\Users\smartin5\Recordings\Raw_1min-20190619T104708Z-001\Raw_1min\190619_LCA4_1m_raw.bin"
    read_axona_raw(location)
