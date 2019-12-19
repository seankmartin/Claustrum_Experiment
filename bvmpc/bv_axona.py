"""Module to read Axona .inp files."""

import os
import argparse
import calendar
import datetime
from collections import OrderedDict

import numpy as np


class AxonaInput:
    """Class which decodes Axona .inp files."""

    def __init__(self, location):
        """
        Initialise an axona input holder.

        Can be initialised with a location.

        Parameters
        ----------
        location : str, default None
            location of .inp file

        Returns
        -------
        None

        """
        if location is not None:
            self.location = location
            AxonaInpReader.load(location, self)

    def save_to_file(self, location=None, inp_types=["I", "O"]):
        """
        Save the AxonaInput object information to a csv file.

        Parameters
        ----------
        location: str, default None:
            Where to save the file to
            if None, determines the name from input file
        inp_types: List, default ["I", "O"]
            What channel information types to save

        Returns
        -------
        None

        """
        if location is None:
            if self.location is not None:
                location = os.path.splitext(self.location)[0] + ".csv"
            else:
                print("Please provide a location to save the file to")
                exit(-1)

        with open(location, "w") as file:
            header = "Time,Type"
            for i in range(16):
                header = "{},Ch{}".format(header, i + 1)
            file.write(header + "\n")
            for (i, (t, c, p)) in enumerate(
                    zip(self.timestamps, self.channels, self.pins)):
                if c in inp_types:
                    out_str = "{:2f},{}".format(t, c)
                    for bit in p:
                        out_str = "{},{}".format(out_str, bit)
                    file.write(out_str + "\n")

    def link_info(self, t_arr, c_arr, p_arr):
        """
        Link array information into this object.

        Parameters
        ----------
        t_arr: np.ndarray
            array of timestamp values
        c_arr: np.ndarray
            array of channel values("I", "O", or "V")
        p_arr: np.ndarray
            array of active pins for the related channel at that time
            represented as a little endian 16 bit unsigned int

        Returns
        -------
        None

        """
        self.timestamps = t_arr
        self.channels = c_arr
        self.pins = np.array([
            np.unpackbits(p_arr[2 * i:2 * i + 2])[::-1]
            for i in range(len(p_arr) // 2)])

    def get_times(self, channel, pin, inverted=False):
        """
        Get timestamps of pin changes on a channel.

        Parameters
        ----------
        channel: str
            Which channel to get
        pin: int
            Which pin to retrieve, 1 indexed
        inverted: bool - default False
            if True, 1 is off and 0 is on

        Returns
        -------
        np.ndarray:
            The times that the value of pin changes

        """
        if inverted:
            on = 0
        else:
            on = 1
        good_idx = np.nonzero(self.channels == channel)
        times = self.timestamps[good_idx]
        pins = self.pins[good_idx]
        pins = pins[:, pin - 1].flatten()
        differences = np.diff(pins)
        change_idxs = np.add(np.nonzero(differences), 1).flatten()
        pin_values_at_change = pins[change_idxs]
        times_at_change = times[change_idxs]
        on_change_idxs = np.nonzero(pin_values_at_change == on)
        return times_at_change[on_change_idxs]


class AxonaInpReader:
    """Axona .inp reader - this class is completely static."""

    @classmethod
    def load(cls, in_location, axona_input):
        """
        Extract information from an Axona inp file.

        Parameters
        ----------
        in_location : str
            Location of the .inp file to load
        axona_input : AxonaInput
            Object to save the information into

        Returns
        -------
        None

        """
        counter = 0
        with open(in_location, 'rb') as file:
            header = cls._parse_header(file)

            time_arr = np.zeros(header["samples"], np.float32)
            char_arr = np.zeros(header["samples"], str)
            inp_arr = np.zeros(header["samples"] * 2, np.uint8)

            chunk = file.read(header["sample_bytes"])
            while counter != header["samples"]:
                info = cls._info_from_chunk(chunk, header["timebase"])
                time_arr[counter] = info[0]
                char_arr[counter] = info[1]
                inp_arr[2 * counter:2 * counter + 2] = info[2]
                counter = counter + 1
                chunk = file.read(header["sample_bytes"])

        axona_input.link_info(time_arr, char_arr, inp_arr)

    @staticmethod
    def _info_from_chunk(chunk, timebase):
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

    @staticmethod
    def _parse_header(file):
        """
        Parse the header of the input file from Axona.

        And leave the file just after the data_start string,
        ready for reading the binary data in chunks
        """
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


class AxonaSet:
    def __init__(self, location=None):
        self.data = OrderedDict()
        self.change_dict = {}
        self.change_dict["trial_date"] = self.change_date
        if location is not None:
            self.location = location
            self.load(location)

    def set_end_time(self):
        """Calculate how long the Session took in mins."""
        dur = int(self.get_val("duration"))
        start_time = self.get_val("trial_time")
        fmt = '%H:%M:%S'
        start_fmt = datetime.datetime.strptime(start_time, fmt)
        final_time = start_fmt + datetime.timedelta(seconds=dur)
        out_str = "{}:{}:{}".format(
            final_time.hour, final_time.minute, final_time.second)
        self.data["end_time"] = out_str

    def load(self, location):
        with open(location, 'r', encoding='latin-1') as f_set:
            lines = f_set.readlines()
            for line in lines:
                split = line.index(" ")
                key, val = line[:split], line[split + 1:].strip()
                # if key in list(self.data.keys()):
                #     print(("Duplicate Key found {}".format(key)))
                if key in list(self.change_dict.keys()):
                    val = self.change_dict[key](val)
                self.data[key] = val
        self.set_end_time()

    def get_val(self, key=None):
        if key is None:
            return self.data
        return self.data.get(key, None)

    @staticmethod
    def change_date(date):
        change = {v: k for k, v in enumerate(calendar.month_abbr)}
        rest = date[date.index(" ") + 1:]
        day, month, year = rest.split(" ")
        month = change[month]
        month = "0" + str(month) if month < 10 else month
        output = "{}/{}/{}".format(month, day, year[2:])
        return output


if __name__ == "__main__":
    in_location = r"C:\Users\smartin5\Recordings\CLA1_2019-09-28.inp"
    ai = AxonaInput(in_location)
    ai.save_to_file()
    in_location = r"C:\Users\smartin5\Recordings\ER\29082019-nt2\29082019-nt2-LFP-2nd-Saline.set"
    set_info = AxonaSet(in_location)
    print(set_info.get_val("trial_date"))
    print(set_info.get_val("end_time"))
