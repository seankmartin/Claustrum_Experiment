"""This module holds analysis using mne and links to mne."""
import os


import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_eloc(ch_names, o_dir, base_name, dummy=False):
    """
    Read or generate csv with 3D tetrode coordinates.
    Generate via user input.

    Parameters
    ----------
    ch_names : List of str
        mne eeg channels names. Must match keys in mne raw object.

    o_dir : dir
        location to save eloc output
    base_name : str
        basename of file. Used in naming of eloc output: helps to track session, date, subject.
    dummy : bool, default False
        Create dummy tetrode locations for better visualisation.

    """
    def get_next_i(rows, cols, idx):
        """Local helper function."""
        row_idx = (idx // cols)
        col_idx = (idx % cols)
        return row_idx, col_idx

    eloc_path = os.path.join(o_dir, base_name + "_eloc.csv")
    if dummy:
        eloc = {}
        n_acc, n_rsc, n_cla = 0, 0, 0
        scale = 1
        for i, ch in enumerate(ch_names):
            if "ACC" in ch:
                x, y = get_next_i(2, 2, n_acc)
                coord_ls = [x/2, 2 - y/2, 0+y]
                n_acc += 1
            elif "RSC" in ch:
                x, y = get_next_i(2, 2, n_rsc)
                # coord_ls = [x, y - 10, 0]
                coord_ls = [x/2, y/2 - 2, 0 + y]
                n_rsc += 1
            else:
                x, y = get_next_i(4, 4, n_cla)
                # coord_ls = [x + 5, 2 - y, -5]
                coord_ls = [x + 3, 2 - y, 1-x/2]
                n_cla += 1
            eloc[ch] = np.array([x/scale for x in coord_ls])

    else:
        try:
            df = pd.read_csv(eloc_path, index_col=0)
            d = df.to_dict("split")
            eloc = dict(zip(d["index"], d["data"]))
        except Exception:
            eloc = {}
            for s, ch in enumerate(ch_names, 1):
                # Duplicate pos for every second tetrode
                if (s + 2) % 2 == 0:
                    eloc[ch] = eloc[ch_names[s - 2]]
                    eloc[ch][0] += 0.01
                else:
                    eloc[ch] = np.empty(3)
                    for i, axis in enumerate(["x", "y", "z"]):
                        eloc[ch][i] = float(input(
                            "Enter {} coordinate for S{}-{}: ".format(
                                axis, s // 2, ch)))
            df = pd.DataFrame.from_dict(eloc, orient="index")
            df.to_csv(eloc_path)
        print(eloc)
    return eloc


def lfp_odict_to_np(lfp_odict):
    """Convert an lfp_odict into an mne compatible numpy array."""
    # Extract LFPs from the odict
    ori_lfp_list = []
    # TODO based on MNE pipeline should this be the filtered or non filtered
    for key, lfp in lfp_odict.get_filt_signal().items():
        ori_lfp_list.append(lfp.get_samples())
    ori_lfp_list.append([0] * len(ori_lfp_list[0]))

    data = np.array(ori_lfp_list, float)
    # MNE expects LFP data to be in Volts. But Neurochat stores LFP in mV.
    data = data / 1000

    return data


def create_mne_array(
        lfp_odict, fname, ch_names=None, regions=None, o_dir="", plot_mon=True):
    """
    Populate a full mne raw array object with information.

    lfp_odict : bvmpc.lfp_odict.LfpODict
        The lfp_odict object to convert to numpy data.
    fname : str
        The full path to the associated session file.
    ch_names : List of str, Default None
        Optional. What to name the mne eeg channels, default: region+chan_idx.
    regions : List of str, Default None
        Optional. A list of region strings the same length as lfp_odict.
        This is only used when ch_names is None.
        If ch_names and regions are both None, names are chan_idx.
    o_dir : str, Default ""
        Optional. Path to directory to store the output files in.
    plot_mon : bool, Default True
        Plot montage of electrode positions used.
    """
    raw_data = lfp_odict_to_np(lfp_odict)

    if ch_names is None:
        if regions is None:
            ch_names = list(lfp_odict.lfp_odict.keys())
        else:
            ch_names = ["{}-{}".format(x, y) for x,
                        y in zip(regions, lfp_odict.lfp_odict.keys())]

    # Read or create tetrode locations
    base_name = os.path.basename(fname)
    eloc = get_eloc(ch_names, o_dir, base_name, dummy=True)
    montage = mne.channels.make_dig_montage(ch_pos=eloc)

    # Convert LFP data into mne format
    example_lfp = lfp_odict.get_filt_signal(key=1)
    sfreq = example_lfp.get_sampling_rate()
    ch_types = (["eeg"] * len(lfp_odict)) + ["stim", ]
    ch_names = ch_names + ["Events", ]
    info = mne.create_info(
        ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage(montage)

    raw = mne.io.RawArray(raw_data, info)

    # cont = input("Show raw mne info? (y|n) \n")
    #     if cont.strip().lower() == "y":
    #         print(raw.info)

    # Plot montage of electrode positions
    if plot_mon:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # sphere = [0, 0, 0, 10]
        raw.plot_sensors(kind='3d', ch_type='eeg', show_names=False,
                         axes=ax, block=True, to_sphere=True)

    return raw


def set_annotations(mne_array, annotation_fname):
    """Read annots from annotation_fname and store them on mne_array."""
    try:
        print('Loading mne_annotations from', annotation_fname)
        annot_from_file = mne.read_annotations(annotation_fname)
        mne_array.set_annotations(annot_from_file)
    except FileNotFoundError:
        print("WARNING: No annotations found at {}".format(annotation_fname))
    # else:
    #     raise ValueError(
    #         "An error occured while reading {}".format(annotation_fname))


def save_annotations(mne_array, annotation_fname):
    """Save the annotations on mne_array to file annotation_fname."""
    if len(mne_array.annotations) > 0:
        print(mne_array.annotations)
        cont = input("Save mne annotations? (y|n) \n")
        if cont.strip().lower() == "y":
            mne_array.annotations.save(annotation_fname)


def add_nc_event_to_mne(mne_array, nc_events, sample_rate=250):
    print("Adding {} events to mne".format(len(nc_events._timestamp)))
    event_data = np.zeros(
        shape=(len(nc_events._timestamp), 3))
    for i, (a, b) in enumerate(
            zip(nc_events._timestamp, nc_events._event_train)):
        sample_number = int(a * sample_rate)
        event_data[i] = np.array([sample_number, 0, b + 1])
    mne_array.add_events(event_data, stim_channel="Events")

    event_name_dict = {}
    for i, (b, a) in enumerate(
            zip(nc_events._event_names, nc_events._event_train)):
        event_name_dict[b] = a + 1
    return event_name_dict


def get_layout(o_dir, layout_name):
    """
    **Layout functions in mne depreciated.**
    Read or generate layout. Layout generated by clicking on input image.

    Parameters
    ----------
    o_dir : dir
        dir in which layout is to be read or saved into
    layout_name : str, default None
        basename of image used as layout template. if None, "topo.lout" is used.

    Returns
    -------
    lt : mne layout object

    """
    # Layout function depreciated. Load tetrode layout
    layout_path = o_dir
    if layout_name is None:
        layout_name = "SA5_topo.lout"
    try:
        lt = mne.channels.read_layout(
            layout_name, path=layout_path, scale=False)
    except:
        # Generate layout of tetrodes
        # This code opens the image so you can click on it to designate tetrode positions
        from mne.viz import ClickableImage  # noqa
        from mne.viz import (
            plot_alignment, snapshot_brain_montage, set_3d_view)
        # The click coordinates are stored as a list of tuples
        template_loc = 'F:\!Imaging\LKC\SA5\SA5_Histology-07.tif'  # template location
        im = plt.imread(template_loc)
        click = ClickableImage(im)
        click.plot_clicks()

        # Generate a layout from our clicks and normalize by the image
        print('Generating and saving layout...')
        lt = click.to_layout()
        # To save if we want
        lt.save(os.path.join(layout_path, layout_name))
        print('Saved layout to', os.path.join(
            layout_path, layout_name))

    # Load and display layout
    lt = mne.channels.read_layout(
        layout_name, path=layout_path, scale=False)
    x = lt.pos[:, 0] * float(im.shape[1])
    y = (1 - lt.pos[:, 1]) * float(im.shape[0])  # Flip the y-position
    fig, ax = plt.subplots()
    ax.imshow(im)
    ax.scatter(x, y, s=120, color='r')
    plt.autoscale(tight=True)
    ax.set_axis_off()
    plt.show()

    return lt


def generate_events(mne_array, session, plot=False):
    """Generate events based on session object.
    Parameters
    ----------
    mne_array : mne.raw object
    session : bvmpc.session object
    plot : bool, default False
        Plot events figure. Mainly for checking.

    Returns
    -------
    events_dict : dict of (event: index). Eg. {'Right': 1, 'Left': 2}
    mne_events : mne.events object
    annot_from events : mne.events converted to mne.annotations

    """
    from bvmpc.bv_nc import events_from_session
    # Generate events
    nc_events = events_from_session(session)
    events_dict = add_nc_event_to_mne(mne_array, nc_events)

    mne_events = mne.find_events(
        mne_array, stim_channel='Events',
        shortest_event=1, min_duration=(0.1 / mne_array.info['sfreq']),
        consecutive=True, initial_event=True)

    if plot:
        fig = mne.viz.plot_events(
            mne_events, sfreq=mne_array.info['sfreq'],
            first_samp=mne_array.first_samp, event_id=events_dict, show=True)

    # Set annotations from events
    # Swap key and values
    events_map = {value: key[:3] for key, value in events_dict.items()}
    onsets = mne_events[:, 0] / mne_array.info['sfreq']
    durations = np.zeros_like(onsets)  # assumes instantaneous events
    descriptions = [events_map[event_id]
                    for event_id in mne_events[:, 2]]
    annot_from_events = mne.Annotations(onset=onsets, duration=durations,
                                        description=descriptions,
                                        orig_time=None)
    return events_dict, mne_events, annot_from_events


def pick_chans(raw, sel=None):
    """
    Pick channels based on sel input. Identifies channels with str or int in sel.
    Example:
        Input
            raw.chan_names = ['ACC-1','ACC-2','RSC-3']
            sel = ['ACC']
        Returns
            picks = ['ACC-1','ACC-2']

    Parameters
    ----------
    raw : mne.raw object
    sel : list of str OR list of int, Default None
        str/int to include in picks. Sel must be in chan_name(eg. ACC-13)
        if list of int, channel number will be used to select channels.

    Returns
    -------
    picks : list of str
        list of channel names selected.

    """
    if sel == None:
        picks = None
    elif type(sel[0]) == int:
        picks = []
        for s in sel:
            for ch in raw.ch_names[:-1]:
                try:
                    if ch.split('-')[1] == str(s):
                        picks.append(ch)
                except:
                    pass
        print('Picked chans:', picks)
    else:
        picks = []
        for s in sel:
            for ch in raw.ch_names:
                if s in ch:
                    picks.append(ch)
        print('Picked chans:', picks)
    return picks


def get_reg_chans(raw, regions):
    """ Get dict with {regions: list of channels}

    Parameters
    ----------
    raw : mne.raw object
    region_dict : list of str,
        list of regions to be grouped. set(regions) used to determine dict keys.

    Returns
    -------
    grps : dict(region: list of channels)

    """
    grps = {}
    for reg in set(regions):
        reg_ls = []
        for i, ch in enumerate(raw.ch_names):
            if reg in ch:
                reg_ls.append(i)
        grps[reg] = reg_ls
    print(grps)
    return grps


def ICA_pipeline(mne_array, regions, chans_to_plot=20, base_name="", exclude=None):
    """
    This is example code using mne.

    Parameters
    ----------


    """
    raw = mne_array

    # # Plot raw signal
    # raw.plot(
    #     n_channels=chans_to_plot, block=True, duration=50,
    #     show=True, clipping="transparentdict(eeg=650e-6) #     title="Raw LFP Data from {}".format(base_name),
    #     remove_dc=False, scalings="auto")

    # Perform ICA using mne
    from mne.preprocessing import ICA
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=1., h_freq=None)
    ica = ICA(method='fastica', random_state=97)
    # ica = ICA(method='picard', random_state=97)
    ica.fit(filt_raw)

    # ica.exclude = [4, 6, 12]
    raw.load_data()
    if exclude is None:
        # Plot raw ICAs
        print('Select channels to exclude using this plot...')
        ica.plot_sources(
            raw, block=True, stop=50, title='ICA from {}'.format(base_name))

        print('Click topo to get more ICA properties')
        ica.plot_components(inst=raw)

        # Overlay ICA cleaned signal over raw. Seperate plot for each region.
        # TODO Add scroll bar or include window selection option.
        cont = input("Plot region overlay? (y|n) \n")
        if cont.strip().lower() == "y":
            reg_grps = []
            for reg in set(regions):
                temp_grp = []
                for ch in raw.info.ch_names:
                    if reg in ch:
                        temp_grp.append(ch)
                reg_grps.append(temp_grp)
            for grps in reg_grps:
                ica.plot_overlay(raw, stop=int(30 * 250), title='{}'.format(
                    grps[0][:3]), picks=grps)
    else:
        # ICAs to exclude
        ica.exclude = exclude
    # Apply ICA exclusion
    reconst_raw = raw.copy()
    ica.apply(reconst_raw)

    # Plot reconstructed signals w/o excluded ICAs
    reconst_raw.plot(block=True, show=True, clipping="transparent", duration=50,
                     title="Reconstructed LFP Data from {}".format(
                         base_name),
                     remove_dc=False, scalings=dict(eeg=350e-6))

    return reconst_raw
