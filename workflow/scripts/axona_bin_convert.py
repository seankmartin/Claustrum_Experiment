import os

import simuran
from BinConverter.core.ConversionFunctions import convert_basename


class FakeGUI(object):
    def __init__(self):
        self.LogAppend = FakeLog()

class FakeLog(object):
    def __init__(self):
        self.myGUI_signal = FakeEmit()

class FakeEmit(object):
    def __init__(self):
        self.emit = print

def convert_from_axona(input_file_location):
    fake = FakeGUI()
    convert_basename(fake, input_file_location, 3)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        use_snakemake = False
    else:
        use_snakemake = True
    if use_snakemake:
        simuran.set_only_log_to_file(snakemake.log[0])
        main(snakemake.input[0], snakemake.output[0])
    else:
        f_location = r"G:\Downloads\20191122\CAR-R1_2019-11-22.set"
        convert_from_axona(f_location)
