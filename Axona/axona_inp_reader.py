import os
import argparse
import numpy as np


def parse_header(file):
    """Parse the header of the input file from Axona."""
    out_dict = {}
    finding_start = True
    while finding_start:
        line = file.readline().decode("latin-1")
        if line.startswith("bytes_per_sample"):
            out_dict["sample_bytes"] = int(line.split(" ")[1])
        elif line.startswith("timebase"):
            out_dict["timebase"] = int(line.split(" ")[1])
        elif line.startswith("num_inp_samples"):
            out_dict["samples"] = int(line.split(" ")[1])
            finding_start = False
    file.seek(10, 1)
    return out_dict


def info_from_chunk(chunk, timebase):
    """Extract the info from a 7 byte chunk."""
    time_val = (
        16777216 * chunk[0] +
        65536 * chunk[1] +
        256 * chunk[2] +
        chunk[3]) / timebase

    input_type = chr(chunk[4])

    channels = np.zeros(2, np.uint8)
    channels[0] = chunk[5]
    channels[1] = chunk[6]
    return time_val, input_type, channels


def info_to_file(time, char, inp, out_loc, inp_types=["I", "O"]):
    """Output the input file information to csv."""
    with open(out_loc, "w") as file:
        header = "Time,Type"
        for i in range(16):
            header = "{},Ch{}".format(header, i + 1)
        file.write(header + "\n")
        for (i, (t, c)) in enumerate(zip(time, char)):
            if c in inp_types:
                out_str = "{:2f},{}".format(t, c)
                b_a = np.unpackbits(inp[2 * i:2 * i + 2])[::-1]
                for bit in b_a:
                    out_str = "{},{}".format(out_str, bit)
                file.write(out_str + "\n")


def read_axona_inp(in_location):
    """Extract certain channel information from the axona bin file"""
    counter = 0
    with open(in_location, 'rb') as file:
        header = parse_header(file)

        time_arr = np.zeros(header["samples"], np.float32)
        char_arr = np.zeros(header["samples"], str)
        inp_arr = np.zeros(header["samples"] * 2, np.uint8)

        chunk = file.read(header["sample_bytes"])
        while counter != header["samples"]:
            info = info_from_chunk(chunk, header["timebase"])
            time_arr[counter] = info[0]
            char_arr[counter] = info[1]
            inp_arr[2 * counter:2 * counter + 2] = info[2]
            counter = counter + 1

            chunk = file.read(header["sample_bytes"])
            # For debugging
            # value = 256 * info[2][0] + info[2][1]
            # value_b = np.unpackbits(info[2])[::-1]

    return time_arr, char_arr, inp_arr


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


def main(in_location, out_location=None):
    if out_location is None:
        out_location = in_location[:-3] + "csv"
    ta, ca, ia = read_axona_inp(in_location)
    info_to_file(ta, ca, ia, out_location)


def main_cmd():
    parser = argparse.ArgumentParser(description="Parse a program location")
    parser.add_argument("--loc", "-l", type=str, help="inp file location")
    parsed = parser.parse_args()
    if parsed.loc == None:
        print("Please enter a location through cmd with -l LOCATION")
        exit(-1)
    main(parsed.loc)


def main_py():
    location = os.path.join(
        r"C:\Users\smartin5\Downloads",
        "CLA1_2019-09-28.inp")
    main(location)


if __name__ == "__main__":
    # main_cmd()
    main_py()
