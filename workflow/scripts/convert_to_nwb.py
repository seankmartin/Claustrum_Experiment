"""Process openfield LFP into power spectra etc. saved to NWB"""

import logging
import os
from pathlib import Path
from ast import literal_eval

import numpy as np
import pandas as pd
import simuran as smr
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable
from neurochat.nc_lfp import NLfp
from neurochat.nc_spike import NSpike
from neurochat.nc_utils import RecPos
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import CompassDirection, Position, SpatialSeries
from pynwb.ecephys import LFP, ElectricalSeries
from pynwb.file import Subject
from scipy.signal import decimate
from skm_pyutils.table import df_from_file, df_to_file, list_to_df
from bvmpc.bv_session import Session


def describe_columns():
    return [
        {
            "name": "tetrode_chan_id",
            "type": str,
            "doc": "label of tetrode_label of channel",
        },
        {
            "name": "num_spikes",
            "type": int,
            "doc": "the number of spikes identified",
        },
        {
            "name": "timestamps",
            "type": np.ndarray,
            "doc": "identified spike times in seconds",
        },
    ]


def describe_columns_waves():
    return [
        {
            "name": "tetrode_chan_id",
            "type": str,
            "doc": "label of tetrode_label of channel",
        },
        {
            "name": "num_spikes",
            "type": int,
            "doc": "the number of spikes identified",
        },
        {
            "name": "waveforms",
            "type": np.ndarray,
            "doc": "Flattened sample values around the spike time on each of the four channels, unflattened has shape (X, 50)",
        },
    ]


def describe_columns_behaviour():
    return [
        {
            "name": "left_lever",
            "type": np.ndarray,
            "doc": "left lever press timestamps",
        },
        {
            "name": "right_lever",
            "type": np.ndarray,
            "doc": "right lever press timestamps",
        },
        {
            "name": "all_nosepokes",
            "type": np.ndarray,
            "doc": "all nosepoke timestamps",
        },
        {
            "name": "Reward",
            "type": np.ndarray,
            "doc": "reward timestamps",
        },
        {
            "name": "left_out",
            "type": np.ndarray,
            "doc": "when the left lever was pushed out and light (indicating trial change to FI)",
        },
        {
            "name": "right_out",
            "type": np.ndarray,
            "doc": "when the right lever was pushed out and light (indicating trial change to FR)",
        },
        {
            "name": "sound",
            "type": np.ndarray,
            "doc": "sound timestamps",
        },
        {
            "name": "Nosepoke",
            "type": np.ndarray,
            "doc": "nosepoke timestamps for rewards",
        },
        {
            "name": "Un_Nosepoke",
            "type": np.ndarray,
            "doc": "nosepoke timestamps for no rewards",
        },
        {
            "name": "Trial Type",
            "type": np.ndarray,
            "doc": "trial type for each trial",
        },
        {
            "name": "L",
            "type": np.ndarray,
            "doc": "left lever press timestamps that were necessary for reward",
        },
        {
            "name": "R",
            "type": np.ndarray,
            "doc": "right lever press timestamps that were necessary for reward",
        },
        {
            "name": "Un_L",
            "type": np.ndarray,
            "doc": "left lever press timestamps that were not necessary for reward",
        },
        {
            "name": "Un_R",
            "type": np.ndarray,
            "doc": "right lever press timestamps that were not necessary for reward",
        },
        {
            "name": "Un_FR_Err",
            "type": np.ndarray,
            "doc": "left lever press timestamps that were not necessary for reward during FR",
        },
        {
            "name": "Un_FI_Err",
            "type": np.ndarray,
            "doc": "right lever press timestamps that were not necessary for reward during FI",
        },
        {
            "name": "FR_Err",
            "type": np.ndarray,
            "doc": "left lever press timestamps that were necessary for reward during FR",
        },
        {
            "name": "FI_Err",
            "type": np.ndarray,
            "doc": "right lever press timestamps that were necessary for reward during FI",
        },
        {
            "name": "Trial_Start",
            "type": np.ndarray,
            "doc": "trial start timestamps",
        },
    ]


pd.options.mode.chained_assignment = None  # default='warn'

here = Path(__file__).resolve().parent
module_logger = logging.getLogger("simuran.custom.convert_to_nwb")


def main(
    table,
    config,
    output_directory,
    overwrite=False,
    except_errors=False,
):
    table = table[table["converted"]]
    table = table[table["number_of_channels"].notnull()]

    loader = smr.loader(config["loader"], **config["loader_kwargs"])
    rc = smr.RecordingContainer.from_table(table, loader)
    used = []
    filenames = []

    for i in range(len(rc)):
        fname, e = convert_to_nwb_and_save(
            rc, i, output_directory, config["cfg_base_dir"], overwrite
        )
        if fname is not None:
            filenames.append(fname)
            used.append(i)
        elif not except_errors:
            print(f"Error with recording {rc[i].source_file}")
            raise e
        else:
            print(f"Error with recording {rc[i].source_file}, check logs")

    if len(used) != len(table):
        missed = len(table) - len(used)
        print(f"WARNING: unable to convert all files, missed {missed}")
    table = table.iloc[used, :]
    table["nwb_file"] = filenames
    df_to_file(table, output_directory / "converted_data.csv")
    return filenames


def convert_to_nwb_and_save(rc, i, output_directory, rel_dir=None, overwrite=False):
    save_name = rc[i].get_name_for_save(rel_dir)
    filename = output_directory / "nwbfiles" / f"{save_name}.nwb"

    if not overwrite and filename.is_file():
        module_logger.debug(f"Already converted {rc[i].source_file}")
        return filename, None

    module_logger.info(f"Converting {rc[i].source_file} to NWB")
    try:
        r = rc.load(i)
    except Exception as e:
        module_logger.error(f"Could not load {rc[i].source_file} due to {e}")
        return None, e
    try:
        nwbfile = convert_recording_to_nwb(r, rel_dir)
    except Exception as e:
        module_logger.error(f"Could not convert {rc[i].source_file} due to {e}")
        return None, e
    return write_nwbfile(filename, r, nwbfile)


def write_nwbfile(filename, r, nwbfile, manager=None):
    filename.parent.mkdir(parents=True, exist_ok=True)
    try:
        with NWBHDF5IO(filename, "w", manager=manager) as io:
            io.write(nwbfile)
        return filename, None
    except Exception as e:
        module_logger.error(
            f"Could not write nwbfile from {r.source_file} out to {filename}"
        )
        if filename.is_file():
            filename.unlink()
        return None, e

def export_nwbfile(filename, r, nwbfile, src_io, debug=False):
    filename.parent.mkdir(parents=True, exist_ok=True)
    try:
        with NWBHDF5IO(filename, "w") as io:
            io.export(src_io=src_io, nwbfile=nwbfile)
        return filename, None
    except Exception as e:
        module_logger.error(
            f"Could not write nwbfile from {r.source_file} out to {filename}"
        )
        if debug:
            breakpoint()
        if filename.is_file():
            filename.unlink()
        return None, e

def convert_recording_to_nwb(recording, rel_dir=None):
    name = recording.get_name_for_save(rel_dir=rel_dir)
    nwbfile = create_nwbfile_with_metadata(recording, name)
    piw_device = add_devices_to_nwb(nwbfile)
    num_electrodes = add_electrodes_to_nwb(recording, nwbfile, piw_device)

    add_behavior(recording, nwbfile)
    add_lfp_data_to_nwb(recording, nwbfile, num_electrodes)
    add_waveforms_and_times_to_nwb(recording, nwbfile)
    add_position_data_to_nwb(recording, nwbfile)

    return nwbfile


def add_behavior(recording, nwbfile):
    set_file = str(recording.source_file)[:-3] + "inp"
    session_type = recording.attrs["maze_type"]
    if session_type == "RandomisedBlocks":
        session_number = "6"
    elif session_type == "RandomisedBlocksFlipped":
        session_number = "6"
    elif session_type == "RandomisedBlocksExtended":
        session_number = "7"
    else:
        return

    if recording.attrs["has_behaviour"]:
        session = Session(axona_file=set_file, s_type=session_number)
        array_info = session.get_arrays()
        df = pd.DataFrame.from_dict(array_info, orient="index")
        df = df.T
        columns = describe_columns_behaviour()
        hdmf_table = DynamicTable.from_dataframe(df=df, name="times", columns=columns)
        mod = nwbfile.create_processing_module(
            "operant_behaviour", "Store operant times (NaN padded)"
        )
        mod.add(hdmf_table)


def add_position_data_to_nwb(recording, nwbfile):
    filename = os.path.join(recording.attrs["directory"], recording.attrs["filename"])
    rec_pos = RecPos(filename, load=True)
    position_data = np.transpose(
        np.array(
            [
                recording.data["spatial"].position[0],
                recording.data["spatial"].position[1],
            ]
        )
    )
    position_timestamps = recording.data["spatial"].timestamps
    spatial_series = SpatialSeries(
        name="SpatialSeries",
        description="(x,y) position in camera",
        data=position_data,
        timestamps=position_timestamps,
        reference_frame="(0,0) is top left corner",
        unit="centimeters",
    )
    position_obj = Position(spatial_series=spatial_series)
    recording.data["spatial"].direction
    hd_series = SpatialSeries(
        name="SpatialSeries",
        description="head direction",
        data=recording.data["spatial"].direction,
        timestamps=position_timestamps,
        reference_frame="0 degrees is west, rotation is anti-clockwise",
        unit="degrees",
    )
    compass_obj = CompassDirection(spatial_series=hd_series)

    speed_ts = TimeSeries(
        name="running_speed",
        description="Smoothed running speed calculated from position",
        data=recording.data["spatial"].speed,
        timestamps=position_timestamps,
        unit="cm/s",
    )

    raw_pos = rec_pos.get_raw_pos()
    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="processed behavior data"
    )
    pos = np.transpose(np.array(raw_pos, dtype=np.uint16))
    big_led_ts = TimeSeries(
        name="led_pixel_positions",
        description="LED positions, note 1023 indicates untracked data. Order is Big LED x, Big LED y, Small LED x, Small LED y",
        data=pos,
        rate=50.0,
        unit="centimeters",
        conversion=(1 / rec_pos.pixels_per_cm),
    )
    behavior_module.add(big_led_ts)

    behavior_module.add(position_obj)
    behavior_module.add(speed_ts)

    if filename.endswith(".pos"):
        behavior_module.add(compass_obj)


def add_waveforms_and_times_to_nwb(recording, nwbfile):
    try:
        spike_files = recording.attrs["source_files"]["Spike"]
    except KeyError:
        module_logger.warning(f"No spike files for {recording.source_file}")
        return
    nc_spike = NSpike()
    df_list = []
    df_list_waves = []
    for sf in spike_files:
        if not os.path.exists(sf):
            continue
        times, waves = nc_spike.load_spike_Axona(sf, return_raw=True)
        ext = os.path.splitext(sf)[-1][1:]
        for chan, val in waves.items():
            name = f"{ext}_{chan}"
            num_spikes = len(times)
            df_list.append([name, num_spikes, np.array(times)])
            df_list_waves.append([name, num_spikes, val.flatten().astype(np.float32)])
    max_spikes = max(d[1] for d in df_list)
    for df_ in df_list:
        df_[2] = np.pad(df_[2], (0, max_spikes - df_[1]), mode="empty")
    for df_wave in df_list_waves:
        df_wave[2] = np.pad(df_wave[2], (0, (max_spikes * 50) - (df_wave[1] * 50)))

    final_df = list_to_df(df_list, ["tetrode_chan_id", "num_spikes", "timestamps"])
    hdmf_table = DynamicTable.from_dataframe(
        df=final_df, name="times", columns=describe_columns()
    )
    mod = nwbfile.create_processing_module("spikes", "Store unsorted spike times")
    mod.add(hdmf_table)

    final_df = list_to_df(df_list_waves, ["tetrode_chan_id", "num_spikes", "waveforms"])
    hdmf_table = DynamicTable.from_dataframe(
        df=final_df, name="waveforms", columns=describe_columns_waves()
    )
    mod.add(hdmf_table)


def add_lfp_array_to_nwb(
    nwbfile,
    num_electrodes,
    lfp_data,
    module=None,
    conversion=0.001,
    rate=250.0,
):
    all_table_region = nwbfile.create_electrode_table_region(
        region=list(range(num_electrodes)), description="all electrodes"
    )

    compressed_data = H5DataIO(
        data=lfp_data, compression="gzip", compression_opts=9
    )
    lfp_electrical_series = ElectricalSeries(
        name="ElectricalSeries",
        data=compressed_data,
        electrodes=all_table_region,
        starting_time=0.0,
        rate=rate,
        conversion=conversion,
        filtering="Notch filter at 50Hz",
    )
    lfp = LFP(electrical_series=lfp_electrical_series)

    if module is None:
        module = nwbfile.create_processing_module(
            name="ecephys",
            description="Processed extracellular electrophysiology data",
        )
    module.add(lfp)


def add_lfp_data_to_nwb(recording, nwbfile, num_electrodes):

    def convert_eeg_path_to_egf(p):
        p = Path(p)
        p = p.with_suffix(f".egf{p.suffix[4:]}")
        if p.is_file():
            return p
        else:
            None

    egf_files = [
        convert_eeg_path_to_egf(f) for f in recording.attrs["source_files"]["Signal"]
    ]
    if egf_files := [f for f in egf_files if f is not None]:
        data = []
        for f in egf_files[:num_electrodes]:
            lfp = NLfp()
            lfp.load(f, system="Axona")
            data.append(lfp.get_samples())
        lfp_data = decimate(np.array(data).T, 3, axis=0).astype(np.float16)
        rate = float(lfp.get_sampling_rate()) / 3
        module = nwbfile.create_processing_module(
            name="high_rate_ecephys",
            description="High sampling rate extracellular electrophysiology data",
        )
        add_lfp_array_to_nwb(
            nwbfile, num_electrodes, lfp_data, rate=rate, module=module
        )
    else:
        module_logger.warning(f"No egf files found for {recording.source_file}")
    lfp_data = np.transpose(
        np.array([s.samples for s in recording.data["signals"] if len(s.samples) != 0])
    )
    add_lfp_array_to_nwb(
        nwbfile, num_electrodes, lfp_data.astype(np.float32), rate=250.0
    )


def add_electrodes_to_nwb(recording, nwbfile, piw_device):
    nwbfile.add_electrode_column(name="label", description="electrode label")
    num_electrodes = add_tetrodes(recording, nwbfile, piw_device)

    return num_electrodes


def add_tetrodes(recording, nwbfile, piw_device):
    num_channels = recording.attrs["number_of_channels"]

    def add_nwb_electrode(nwbfile, brain_region, electrode_group, label):
        nwbfile.add_electrode(
            x=np.nan,
            y=np.nan,
            z=np.nan,
            imp=np.nan,
            location=brain_region,
            filtering="Notch filter at 50Hz",
            group=electrode_group,
            label=label,
        )

    for i in range(16):
        brain_region = get_brain_region_for_tetrode(recording, i)
        electrode_group = nwbfile.create_electrode_group(
            name=f"TT{i}",
            device=piw_device,
            location=brain_region,
            description=f"Tetrode {i} electrodes placed in {brain_region}",
        )
        for j in range(int(num_channels // 16)):
            add_nwb_electrode(nwbfile, brain_region, electrode_group, f"TT{i}_E{j}")
    return int(num_channels)


def get_brain_region_for_tetrode(recording, i):
    num_signals = recording.attrs["number_of_channels"]
    brain_region = literal_eval(recording.attrs["brain_regions"])[
        i * int((num_signals // 16))
    ]
    return brain_region


def add_devices_to_nwb(nwbfile):
    piw_device = nwbfile.create_device(
        name="Platinum-iridium wires 25um thick",
        description="Bundles of 4 connected to 32-channel Axona microdrive",
        manufacturer="California Fine Wire",
    )

    return piw_device


def create_nwbfile_with_metadata(recording, name):
    nwbfile = NWBFile(
        session_description=f"Recording {name}",
        identifier=f"CLA--{name}",
        session_start_time=recording.datetime,
        experiment_description="CLA, RSC, and related areas during Operant box task",
        experimenter="Gao Xiang Ham",
        lab="O'Mara lab",
        institution="TCD",
    )
    nwbfile.subject = Subject(
        species="Lister Hooded rat",
        sex="M",
        subject_id=recording.attrs["rat_id"],
        weight=0.330,
    )

    return nwbfile


def convert_listed_data_to_nwb(
    overall_datatable,
    config_path,
    output_directory,
    overwrite=False,
    except_errors=False,
):
    """These are processed in order of individual_tables"""
    config = smr.ParamHandler(source_file=config_path, name="params")
    table = df_from_file(overall_datatable)
    main(
        table,
        config,
        output_directory,
        overwrite=overwrite,
        except_errors=except_errors,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        use_snakemake = False
    else:
        use_snakemake = True
    if use_snakemake:
        smr.set_only_log_to_file(snakemake.log[0])
        module_logger.setLevel(logging.DEBUG)
        convert_listed_data_to_nwb(
            snakemake.input[0],
            snakemake.config["simuran_config"],
            Path(snakemake.output[0]).parent,
            snakemake.config["overwrite_nwb"],
            except_errors=snakemake.config["except_nwb_errors"],
        )
    else:
        module_logger.setLevel(logging.DEBUG)
        convert_listed_data_to_nwb(
            here.parent.parent / "results" / "metadata_parsed.csv",
            here.parent.parent / "config" / "params.yaml",
            here / "results" / "nwbfiles",
            overwrite=True,
            except_errors=False,
        )
