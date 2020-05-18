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
        for i, ch in enumerate(ch_names):
            if "ACC" in ch:
                x, y = get_next_i(2, 2, n_acc)
                eloc[ch] = np.array([x, 2 - y, 0])
                n_acc += 1
            elif "RSC" in ch:
                x, y = get_next_i(2, 2, n_rsc)
                eloc[ch] = np.array([x, y - 10, 0])
                n_rsc += 1
            else:
                x, y = get_next_i(4, 4, n_cla)
                eloc[ch] = np.array([x + 5, 2 - y, -5])
                n_cla += 1
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
    ori_keys, ori_lfp_list = [], []
    # TODO based on MNE pipeline should this be the filtered or non filtered
    for key, lfp in lfp_odict.get_filt_signal().items():
        ori_lfp_list.append(lfp.get_samples())
        ori_keys.append(key)
    data = np.array(ori_lfp_list, float)
    # MNE expects LFP data to be in Volts. But Neurochat stores LFP in mV.
    data = data / 1000

    return data


def create_mne_array(
        lfp_odict, fname, ch_names=None, regions=None, o_dir=""):
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
    info = mne.create_info(
        ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage)

    raw = mne.io.RawArray(raw_data, info)

    return raw


def set_annotations(mne_array, annotation_fname):
    """Read annots from annotation_fname and store them on mne_array."""
    try:
        print('Loading mne_annotations from', annotation_fname)
        annot_from_file = mne.read_annotations(annotation_fname)
        mne_array.set_annotations(annot_from_file)
    except FileNotFoundError:
        print("WARNING: No annotations found at {}".format(annotation_fname))
    else:
        raise ValueError(
            "An error occured while reading {}".format(annotation_fname))


def save_annotations(mne_array, annotation_fname):
    """Save the annotations on mne_array to file annotation_fname."""
    if len(mne_array.annotations) > 0:
        print(mne_array.annotations)
        cont = input("Save mne annotations? (y|n) \n")
        if cont.strip().lower() == "y":
            mne_array.annotations.save(annotation_fname)


def nc_to_mne_events(nc_events):
    mne_events = nc_events
    return mne_events


def mne_example(mne_array, regions, chans_to_plot=20, base_name=""):
    """
    This is example code using mne.

    Parameters
    ----------
    lfp_odict :
    fname : full path of session file
    o_dir : dir to save mne variables or figures

    """
    raw = mne_array

    # Test montage
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # sphere = [0, 0, 0, 10]
    raw.plot_sensors(kind='3d', ch_type='eeg', show_names=False,
                     axes=ax, block=True, to_sphere=True)
    # exit(-1)

    # cont = input("Show raw mne info? (y|n) \n")
    # if cont.strip().lower() == "y":
    #     print(raw.info)

    # # Layout function depreciated. Load tetrode layout
    # layout_path = o_dir
    # layout_name = "SA5_topo.lout"
    # try:
    #     lt = mne.channels.read_layout(
    #         layout_name, path=layout_path, scale=False)
    # except:
    #     # Generate layout of tetrodes
    #     # This code opens the image so you can click on it to designate tetrode positions
    #     from mne.viz import ClickableImage  # noqa
    #     from mne.viz import (
    #         plot_alignment, snapshot_brain_montage, set_3d_view)
    #     # The click coordinates are stored as a list of tuples
    #     template_loc = 'F:\!Imaging\LKC\SA5\SA5_Histology-07.tif'  # template location
    #     im = plt.imread(template_loc)
    #     click = ClickableImage(im)
    #     click.plot_clicks()

    #     # Generate a layout from our clicks and normalize by the image
    #     print('Generating and saving layout...')
    #     lt = click.to_layout()
    #     # To save if we want
    #     lt.save(os.path.join(layout_path, layout_name))
    #     print('Saved layout to', os.path.join(
    #         layout_path, layout_name))

    # # Load and display layout
    # lt = mne.channels.read_layout(
    #     layout_name, path=layout_path, scale=False)
    # x = lt.pos[:, 0] * float(im.shape[1])
    # y = (1 - lt.pos[:, 1]) * float(im.shape[0])  # Flip the y-position
    # fig, ax = plt.subplots()
    # ax.imshow(im)
    # ax.scatter(x, y, s=120, color='r')
    # plt.autoscale(tight=True)
    # ax.set_axis_off()
    # plt.show()

    # Plot raw signal
    raw.plot(
        n_channels=chans_to_plot, block=True, duration=50,
        show=True, clipping="clamp",
        title="Raw LFP Data from {}".format(base_name),
        remove_dc=False, scalings="auto")

    # Perform ICA using mne
    from mne.preprocessing import ICA
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=1., h_freq=None)
    ica = ICA(random_state=97)
    ica.fit(filt_raw)

    raw.load_data()
    # Plot raw ICAs
    print('Select channels to exclude using this plot...')
    ica.plot_sources(
        raw, block=True, stop=50, title='ICA from {}'.format(base_name))

    print('Click topo to get more ICA properties')
    ica.plot_components(inst=raw)
    # ICAs to exclude
    # ica.exclude = [4, 6, 12]

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

    # Apply ICA exclusion
    reconst_raw = raw.copy()
    ica.apply(reconst_raw)

    # Plot reconstructed signals w/o excluded ICAs
    reconst_raw.plot(block=True, show=True, clipping="clamp", duration=50,
                     title="Reconstructed LFP Data from {}".format(
                         base_name),
                     remove_dc=False, scalings="auto")
