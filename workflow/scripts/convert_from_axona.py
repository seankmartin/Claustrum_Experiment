import os

from skm_pyutils.path import get_all_files_in_dir
from skm_pyutils.table import df_from_file, df_to_file
import simuran
from BinConverter.core.ConversionFunctions import convert_basename


def main(input_file_location, output_file_location):
    df = df_from_file(input_file_location)
    converted_list = []
    for i, row in df.iterrows():
        files = get_all_files_in_dir(row["directory"])
        converted = False
        for f in files:
            if f.endswith(".eeg2"):
                converted = True
                break
        converted_list.append(converted)
    df.loc[:, "converted"] = converted_list
    df_to_file(df, output_file_location)

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

        exit(-1)
        main(
            os.path.join("results", "metadata_parsed.csv"),
            os.path.join("results", "converted_data.csv"),
        )
