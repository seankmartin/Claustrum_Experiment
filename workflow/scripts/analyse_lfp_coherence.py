import os
import numpy as np
from typing import TYPE_CHECKING
from pathlib import Path

from skm_pyutils.table import df_from_file, list_to_df, df_to_file
import simuran as smr
from scipy import signal
import pandas as pd

from bvmpc.bv_session import Session

if TYPE_CHECKING:
    from simuran import Recording

def main(input_table_path, out_file, config, overwrite=False):
    """Run the main analysis."""
    df = df_from_file(input_table_path)
    df = df[df["has_behaviour"]]
    df = df[~df["directory"].str.contains("Batch_1")]
    df = df[df["brain_regions"].str.contains(r"^(?=.*CLA)(?=.*RSC)(?=.*ACC)")]
    rc = smr.RecordingContainer.from_table(df, loader=smr.loader("NWB"))

    analyse_lfp_coherence(rc, out_file, config, overwrite=overwrite)


def analyse_lfp_coherence(recording_container, out_file: str, config, overwrite=False):
    """Analyse the LFP coherence of the recording."""
    overall_df = None
    if os.path.exists(out_file):
        overall_df = df_from_file(out_file)
    try:
        for recording in recording_container:
            if (overall_df is not None) and (not overwrite) and (recording.source_file in list(coherence_df["source_file"])):
                continue
            coherence_df = perform_coherence_in_block(recording)
            coherence_df = add_bands_to_df(coherence_df, config)
            overall_df = pd.concat([overall_df, coherence_df])
    except Exception as e:
        print(f"Failed due to {e}, saving what we have")
        df_to_file(overall_df, out_file)


def convert_time_to_idx(times, sr):
    return np.round(times * sr).astype(int)


def add_bands_to_df(coherence_df, config):
    bands = [
        (config["delta_min"], config["delta_max"]),
        (config["theta_min"], config["theta_max"]),
        (config["beta_min"], config["beta_max"]),
        (config["low_gamma_min"], config["low_gamma_max"]),
        (config["high_gamma_min"], config["high_gamma_max"]),
    ]
    band_names = ["Delta", "Theta", "Beta", "Low gamma", "High gamma"]

    for row in coherence_df.iterrows():
        band_values = find_coherence_in_bands(row["Cxy"], row["f"], bands)
        for i, band in enumerate(bands):
            coherence_df.loc[row[0], band_names[i]] = band_values[i]
    band_values = find_coherence_in_bands()

    return coherence_df


def find_coherence_in_bands(Cxy, f, bands):
    band_values = []
    for band in bands:
        band_values.append(np.mean(Cxy[(f >= band[0]) & (f <= band[1])]))
    return band_values


def perform_coherence_in_block(recording):
    nwbfile = recording.data
    session = Session(recording=recording)
    trial_times, trial_types = extract_trial_info(session)
    brain_regions = set(list(nwbfile.electrodes.to_dataframe()["location"]))
    info = []
    region_pairs = [("CLA", "RSC"), ("CLA", "ACC"), ("RSC", "ACC")]
    if "CLA" not in brain_regions:
        region_pairs = [("CLA/DI", "RSC"), ("CLA/DI", "ACC"), ("RSC", "ACC")]
    for pair in region_pairs:
        region1, region2 = pair
        pair_name = f"{region1}-{region2}"
        lfp1 = nwbfile.processing["average_lfp"][f"{region1}_avg"].data[:]
        lfp1_sr = nwbfile.processing["average_lfp"][f"{region1}_avg"].rate
        lfp2 = nwbfile.processing["average_lfp"][f"{region2}_avg"].data[:]
        lfp2_sr = nwbfile.processing["average_lfp"][f"{region2}_avg"].rate

        for i, t in enumerate(trial_times):
            lfp_indices1 = convert_time_to_idx(t, lfp1_sr)
            lfp_indices2 = convert_time_to_idx(t, lfp2_sr)
            lfp_subset1 = lfp1[lfp_indices1[0] : lfp_indices1[1]]
            lfp_subset2 = lfp2[lfp_indices2[0] : lfp_indices2[1]]
            f, Cxy = signal.coherence(
                lfp_subset1, lfp_subset2, lfp1_sr, nperseg=2 * lfp1_sr
            )
            info.append(
                [pair_name, Cxy, f, trial_types[i], t[0], t[1], recording.source_file]
            )

    headers = [
        "region_pair",
        "coherence",
        "freq",
        "trial_type",
        "start",
        "end",
        "source_file",
    ]

    return list_to_df(info, headers)


def extract_trial_info(session: "Session"):
    trial_ends = session.get_rw_ts()
    trial_types = session.info_arrays["Trial Type"]
    block_types = ["FR" if x == 1 else "FI" for x in trial_types]
    block_times = session.get_block_ends()
    trial_types = []
    for t in trial_ends:
        for i, b in enumerate(block_times):
            if t < b:
                trial_types.append(block_types[i])
                break
    trial_start_ends = []
    first_trials_after_block = np.searchsorted(trial_ends, block_times)
    block_trial_types = []
    for i, t in enumerate(trial_ends):
        if i in first_trials_after_block:
            continue
        trial_start_ends.append((trial_ends[i - 1], t))
        block_trial_types.append(trial_types[i - 1])

    return np.array(trial_start_ends), np.array(block_trial_types)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        snakemake.config["simuran_config"],
        snakemake.output[0],
        snakemake.threads,
    )
