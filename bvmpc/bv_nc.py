import os

import numpy as np
import matplotlib.pyplot as plt

from neurochat.nc_spike import NSpike
from neurochat.nc_event import NEvent
# import spikeinterface.spiketoolkit as st


def events_from_session(session):
    right_presses = session.get_one_lever_ts("R", True)
    left_presses = session.get_one_lever_ts("L", True)
    reward_times = session.get_rw_ts()
    collection_times = session.get_arrays("Nosepoke")
    names = ["Right", "Left", "Reward", "Collection"]
    all_info = [right_presses, left_presses, reward_times, collection_times]

    nc_events = NEvent()
    event_train = []
    event_names = []
    event_tags = []

    # This could be sped up by directly using np arrays
    # But it is still fast since the arrays are small.
    for tag, (info, name) in enumerate(zip(all_info, names)):
        for val in info:
            event_names.append(name)
            event_tags.append(tag)
            event_train.append(val)
    event_train = np.array(event_train)
    event_names = np.array(event_names)
    event_tags = np.array(event_tags)

    # Order the events based on time
    ordering = event_train.argsort()
    ordered_train = event_train[ordering]
    ordered_names = event_names[ordering]
    ordered_tags = event_tags[ordering]

    # Plug these values into neurochat
    nc_events._event_train = ordered_tags
    nc_events._event_names = ordered_names
    nc_events._timestamp = ordered_train

    print(nc_events)
    return nc_events


def load_phy(folder_name):
    import spikeinterface.extractors as se
    to_exclude = ["mua", "noise"]
    return se.PhySortingExtractor(
        folder_name, exclude_cluster_groups=to_exclude, load_waveforms=True,
        verbose=True)


def plot_all_forms(sorting, out_loc, channels_per_group=4):
    unit_ids = sorting.get_unit_ids()
    wf_by_group = [
        sorting.get_unit_spike_features(u, "waveforms") for u in unit_ids]
    for i, wf in enumerate(wf_by_group):
        try:
            tetrode = sorting.get_unit_property(unit_ids[i], "group")
        except Exception:
            try:
                tetrode = sorting.get_unit_property(
                    unit_ids[i], "ch_group")
            except Exception:
                print("Unable to find cluster group or group in units")
                print(sorting.get_shared_unit_property_names())
                return

        fig, axes = plt.subplots(channels_per_group)
        for j in range(channels_per_group):
            try:
                wave = wf[:, j, :]
            except Exception:
                wave = wf[j, :]
            axes[j].plot(wave.T, color="k", lw=0.3)
        o_loc = os.path.join(
            out_loc, "tet{}_unit{}_forms.png".format(
                tetrode, unit_ids[i]))
        print("Saving waveform {} on tetrode {} to {}".format(
            i, tetrode, o_loc))
        fig.savefig(o_loc, dpi=200)
        plt.close("all")


def extract_sorting_info(sorting):
    sample_rate = sorting.params['sample_rate']
    all_unit_trains = sorting.get_units_spike_train()
    timestamps = np.concatenate(all_unit_trains) / float(sample_rate)
    unit_tags = np.zeros(len(timestamps))
    start = 0
    for u_i, u in enumerate(sorting.get_unit_ids()):
        end = start + all_unit_trains[u_i].size
        unit_tags[start:end] = u
        start = end

    # out_loc = os.path.join(
    #     os.path.dirname(sorting.params['dat_path']), "nc_results")
    # os.makedirs(out_loc, exist_ok=True)
    # plot_all_forms(sorting, out_loc)

    waveforms = {}
    unit_ids = sorting.get_unit_ids()
    for u in unit_ids:
        waveforms[str(u)] = sorting.get_unit_spike_features(u, "waveforms")

    return timestamps, unit_tags, waveforms


def load_spike_phy(self, folder_name):
    print("loading Phy sorting information from {}".format(folder_name))
    sorting = load_phy(folder_name)
    timestamps, unit_tags, waveforms = extract_sorting_info(sorting)

    self._set_duration(timestamps.max())
    self._set_timestamp(timestamps)
    self.set_unit_tags(unit_tags)

    # TODO note that waveforms do not follow NC convention
    # It is just a way to store them for the moment.
    self._set_waveform(waveforms)


NSpike.load_spike_phy = load_spike_phy

if __name__ == "__main__":
    folder = r"D:\Ham_Data\Batch_3\A13_CAR-SA5\CAR-SA5_20200212\phy_klusta"
    nspike = NSpike()
    nspike.load_spike_phy(folder)
    print(nspike.get_unit_list())
    print(nspike.get_timestamp(13))
