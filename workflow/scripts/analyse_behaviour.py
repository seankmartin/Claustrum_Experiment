import simuran as smr
import numpy as np
import pandas as pd

from behaviour_detect import full_trial_info
from bvmpc.bv_session import Session


def main(input_table_path, out_file1, out_file2):
    """Run the main analysis."""
    df = smr.load_table(input_table_path)
    df = df[df["has_behaviour"]]
    df = df[df["brain_regions"].str.contains(r"^(?=.*CLA)(?=.*RSC)(?=.*ACC)")]
    rc = smr.RecordingContainer.from_table(df, loader=smr.loader("NWB"))
    collate_behaviour(rc, out_file1, out_file2)


def collate_behaviour(recording_container, out_file1, out_file2):
    """Analyse the LFP coherence of the recording."""
    full_info = []
    lever_presses = []
    for recording in recording_container.load_iter():
        try:
            session = Session(recording=recording)
            trial_info = full_trial_info(session)
        except Exception as e:
            print(e)
            continue
        for i, row in trial_info.iterrows():
            full_info.append(
                [
                    row["Trial type"],
                    row["Estimated trial type"],
                    max(0, row["Trial end"] - row["Trial start"]),
                    len(row["Lever presses"]),
                ]
            )
            presses = row["Lever presses"] - row["Trial start"]
            for p in presses:
                lever_presses.append(
                    [
                        row["Trial type"],
                        row["Estimated trial type"],
                        p,
                    ]
                )

    full_info = pd.DataFrame(
        full_info,
        columns=[
            "Trial type",
            "Estimated trial type",
            "Trial length (s)",
            "Number of lever presses",
        ],
    )
    lever_presses = pd.DataFrame(
        lever_presses,
        columns=["Trial type", "Estimated trial type", "Lever press time (s)"],
    )

    smr.save_table(full_info, out_file1)
    smr.save_table(lever_presses, out_file2)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        snakemake.output[0],
        snakemake.output[1],
    )
