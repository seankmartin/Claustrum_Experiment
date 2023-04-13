import os
from pathlib import Path
from typing import TYPE_CHECKING

import logging
from configparser import ConfigParser

import numpy as np
import simuran
from skm_pyutils.path import get_all_files_in_dir
from skm_pyutils.table import df_from_file, df_to_file

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
            if f.endswith(".bin") or f.endswith(".eeg2"):
                found = True
                break
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
    df_to_use.loc[:, "brain_regions"] = df_to_use.apply(find_brain_regions, axis=1)
    return df_to_use


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
    config_location = os.path.abspath(
        os.path.join("..", "..", "configs", "LFP", f"{rat_id}.cfg")
    )
    if os.path.exists(config_location):
        config = ConfigParser()
        config.read(config_location)
        brain_regions = config["Regions"]
        return brain_regions
    else:
        logging.warning(f"{rat_id} not found in ../../configs/LFP/{rat_id}.cfg")
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
