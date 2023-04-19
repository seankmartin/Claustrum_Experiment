import numpy as np
from bvmpc.bv_session import Session
from skm_pyutils.table import list_to_df


def extract_trial_info(session: "Session"):
    trial_ends = session.get_rw_ts()
    trial_types = session.info_arrays["Trial Type"]
    block_types = ["FR" if x == 1 else "FI" for x in trial_types]
    block_times = session.get_block_ends()
    if len(block_times) != len(block_types):
        raise ValueError(
            "Block times and types don't match, {} != {}".format(
                block_times, block_types
            )
        )
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


def full_trial_info(session: "Session"):
    trial_times, trial_types = extract_trial_info(session)
    lever_presses = session.get_lever_ts()
    estimated_trial_types = []
    lever_splits = []
    for trial_time in trial_times:
        lever_times = lever_presses[
            np.logical_and(lever_presses > trial_time[0], lever_presses < trial_time[1])
        ]
        lever_splits.append(lever_times)
        trial_length = trial_time[1] - trial_time[0]
        if len(lever_times) == 0:
            estimated_trial_types.append("Fail")
            continue
        if len(lever_times) < 6:
            estimated_trial_types.append("FI")
            continue
        first_press = lever_times[0]
        if first_press > trial_time[0] + 25:
            estimated_trial_types.append("FI")
            continue
        average_press_rate = len(lever_times) / trial_length
        if average_press_rate > 0.2:
            estimated_trial_types.append("FR")
            continue
        if len(lever_times) > 7:
            estimated_trial_types.append("FR")
            continue
        estimated_trial_types.append("FI")

    headers = [
        "Trial start",
        "Trial end",
        "Trial type",
        "Estimated trial type",
        "Lever presses",
    ]
    df = list_to_df(
        [
            trial_times[:, 0],
            trial_times[:, 1],
            trial_types,
            estimated_trial_types,
            lever_splits,
        ],
        transpose=True,
        headers=headers,
    )
    return df
