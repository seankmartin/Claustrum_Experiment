import os
from pathlib import Path
from typing import TYPE_CHECKING
import pandas as pd

import logging
from configparser import ConfigParser

import numpy as np
import simuran
from skm_pyutils.path import get_all_files_in_dir
from skm_pyutils.table import df_from_file, df_to_file
from bvmpc.bv_session import Session

if TYPE_CHECKING:
    from pandas import DataFrame, Series


def find_only_data(df: "DataFrame") -> "DataFrame":
    """
    Find only data rows in the metadata file.

    Parameters
    ----------
        df (DataFrame): The metadata file.

    Returns
    -------
        DataFrame: The data rows.
    """
    behaviour_rows = df["filename"].str.contains("Behav") | df["comments"].str.contains(
        "Behaviour only"
    )
    no_behaviour_df = df.loc[~behaviour_rows]

    has_data = []
    for index, row in no_behaviour_df.iterrows():
        files_in_folder = get_all_files_in_dir(row["directory"])
        found = False
        for f in files_in_folder:
            if f.endswith(".bin"):
                found = True
        has_data.append(found)

    df_with_data = no_behaviour_df.loc[has_data]
    return df_with_data


def parse_metadata(df: "DataFrame") -> "DataFrame":
    """
    Parse the metadata file.

    Parameters
    ----------
        df (DataFrame): The metadata file.

    Returns
    -------
        DataFrame: The parsed metadata file.
    """
    df_to_use = df.copy()
    df_to_use.loc[:, "rat_id"] = df_to_use.apply(find_rat_id, axis=1)
    df_to_use = df_to_use.dropna(subset=["rat_id"])
    df_to_use.loc[:, "maze_type"] = df_to_use.apply(find_maze_type, axis=1)
    df_new = df_to_use.apply(
        find_brain_regions, axis="columns", result_type="expand"
    )
    df_to_use = pd.concat([df_to_use, df_new], axis="columns")
    df_to_use.loc[:, "converted"] = check_converted(df_to_use)
    df_to_use.loc[:, "has_behaviour"] = check_has_behaviour(df_to_use)
    df_new = df_to_use.apply(
        parse_behaviour_metadata, axis="columns", result_type="expand"
    )
    df_to_use = pd.concat([df_to_use, df_new], axis="columns")
    return df_to_use

def parse_behaviour_metadata(row: "Series"):
    output_dict = {
        "fixed_ratio": np.nan,
        "double_reward_window (secs)": np.nan,
        "fixed_interval (secs)": np.nan,
        "num_trials": np.nan,
        "trial_length (mins)": np.nan,
    }
    if row["has_behaviour"]:
        inp_file = os.path.join(row["directory"], row["filename"])[:-4] + ".inp"
        session_type = row["maze_type"]
        if session_type == "RandomisedBlocks":
            session_number = "6"
        elif session_type == "RandomisedBlocksFlipped":
            session_number = "6"
        elif session_type == "RandomisedBlocksExtended":
            session_number = "7"
        else:
            return output_dict
        session = Session(axona_file=inp_file, s_type=session_number)
        metadata = session.get_metadata()
        output_dict["fixed_ratio"] = metadata["fixed_ratio"]
        output_dict["double_reward_window (secs)"] = metadata[
            "double_reward_window (secs)"
        ]
        output_dict["fixed_interval (secs)"] = metadata["fixed_interval (secs)"]
        output_dict["num_trials"] = metadata["num_trials"]
        output_dict["trial_length (mins)"] = metadata["trial_length (mins)"]
    return output_dict



def check_has_behaviour(df: "DataFrame") -> "DataFrame":
    """
    Check if the data has behaviour.

    Parameters
    ----------
        df (DataFrame): The metadata file.

    Returns
    -------
        DataFrame: The metadata file with a new column for converted.
    """
    has_behaviour = []
    for i, row in df.iterrows():
        source_file = os.path.join(row["directory"], row["filename"])[:-4] + ".inp"
        found = os.path.exists(source_file)
        has_behaviour.append(found)
    return has_behaviour

def check_converted(df: "DataFrame") -> "DataFrame":
    """
    Check if the data has been converted.

    Parameters
    ----------
        df (DataFrame): The metadata file.

    Returns
    -------
        DataFrame: The metadata file with a new column for converted.
    """
    converted_list = []
    for i, row in df.iterrows():
        files = get_all_files_in_dir(row["directory"])
        found = [False, False, False, False]
        for f in files:
            if f.endswith(".eeg2"):
                found[0] = True
            if f.endswith(".1"):
                found[1] = True
            if f.endswith(".pos"):
                found[2] = True
            if f.endswith(".5"):
                found[3] = True
        converted_list.append(all(found))
    return converted_list

def find_rat_id(row: "Series") -> str:
    """
    Find the rat id from the filename.

    Parameters
    ----------
        row (Series): The row to find the rat id for.

    Returns
    -------
        str: The rat id.
    """
    start = row["filename"].find("C")
    end = row["filename"].find("_", start)
    if end == -1:
        end = row["filename"].find(".", start)
    name = row["filename"][start:end]
    if start == -1:
        p = Path(row["directory"])
        start = p.name.find("C")
        end = p.name.find("_", start)
        name = p.name[start:end]
        if start == -1:
            start = p.parent.name.find("C")
            end = p.parent.name.find("_", start)
            name = p.parent.name[start:end]
    if start == -1:
        return np.nan
    if name == "CLA":
        name = "CLA1"
    return name


def find_maze_type(row: "Series") -> str:
    """
    Find the maze type from the filename.

    Parameters
    ----------
        row (Series): The row to find the maze type for.

    Returns
    -------
        str: The maze type.

    """
    if "Pre" in row["filename"]:
        return "Pre box"
    if "Open" in row["filename"] or ("Open" in row["directory"]):
        return "Open field"
    if "OF" in row["filename"] or ("OF" in row["directory"]):
        return "Open field"
    else:
        set_file_location = os.path.join(row["directory"], row["filename"])
        with open(set_file_location, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("script"):
                    script = line.split(" ", 2)[1].strip()
                    break
            script = Path(script).name[2:-4]
        return script


def find_brain_regions(row):
    rat_id = row["rat_id"]
    config_location = os.path.abspath(os.path.join("configs", "LFP", f"{rat_id}.cfg"))
    if os.path.exists(config_location):
        config = ConfigParser()
        config.read(config_location)
        chan_amount = int(config.get("Setup", "chans"))
        region_dict = config._sections["Regions"]
        regions = []
        for _, val in region_dict.items():
            to_add = val.split(" * ")
            adding = [to_add[0]] * int(to_add[1])
            regions += adding

        shuttle_dict = config._sections["Shuttles"]
        shuttles = []
        for _, val in shuttle_dict.items():
            to_add = val.split(" * ")
            adding = [to_add[0]] * int(to_add[1])
            shuttles += adding

        return {
            "number_of_channels": chan_amount,
            "brain_regions": regions,
            "shuttles": shuttles,
        }
    else:
        logging.warning(f"{rat_id} not found in configs/LFP/{rat_id}.cfg")
        return None


def main(df_location, df_out_location):
    df = df_from_file(df_location)
    df = find_only_data(df)
    df = parse_metadata(df)
    df_to_file(df, df_out_location)


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
        main(
            os.path.join("results", "axona_file_index.csv"),
            os.path.join("results", "metadata_parsed.csv"),
        )
