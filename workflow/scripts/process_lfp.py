"""Process openfield LFP into power spectra etc. saved to NWB"""
import itertools
import logging
from math import ceil, floor
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import scipy
import simuran as smr
from hdmf.common import DynamicTable
from pynwb import TimeSeries
from simuran.loaders.nwb_loader import NWBLoader
from skm_pyutils.table import df_from_file, df_to_file, list_to_df

from convert_to_nwb import add_lfp_array_to_nwb, export_nwbfile
from frequency_analysis import calculate_psd
from lfp_clean import LFPAverageCombiner, NWBSignalSeries

here = Path(__file__).resolve().parent
module_logger = logging.getLogger("simuran.custom.process_lfp")


def describe_columns():
    return [
        {"name": "label", "type": str, "doc": "label of electrode"},
        {"name": "region", "type": str, "doc": "brain region of electrode"},
        {"name": "frequency", "type": np.ndarray, "doc": "frequency values in Hz"},
        {"name": "power", "type": np.ndarray, "doc": "power values in dB"},
        {"name": "max_psd", "type": float, "doc": "maximum power value (uV)"},
    ]


def describe_coherence_columns():
    return [
        {"name": "label", "type": str, "doc": "label of coherence pair"},
        {"name": "frequency", "type": np.ndarray, "doc": "frequency values in Hz"},
        {"name": "coherence", "type": np.ndarray, "doc": "coherence values unitless"},
    ]


def get_sample_times(n_samples, samples_per_second, sr):
    skip_rate = int(sr / samples_per_second)
    slicer = slice(skip_rate, -skip_rate, skip_rate)
    return [i / sr for i in range(n_samples)][slicer]


def power_of_signal(lfp_signal, lfp_sr, low_f, high_f):
    slep_win = scipy.signal.hann(lfp_signal.size, False)
    f, psd = scipy.signal.welch(
        lfp_signal,
        fs=lfp_sr,
        window=slep_win,
        nperseg=len(lfp_signal),
        nfft=256,
        noverlap=0,
    )
    idx_band = np.logical_and(f >= low_f, f <= high_f)
    abs_power = scipy.integrate.simps(psd[idx_band], x=f[idx_band])
    total_power = scipy.integrate.simps(psd, x=f)

    return abs_power, total_power


def process_lfp(ss, config, type_):
    combiner = LFPAverageCombiner(
        z_threshold=config["z_score_threshold"],
        remove_outliers=True,
    )
    results_dict = combiner.combine(ss)

    clean_kwargs = config[type_]
    sub_ss = ss.select_electrodes(
        clean_kwargs["pick_property"], clean_kwargs["options"]
    )
    selected_res = combiner.combine(sub_ss)
    return results_dict, selected_res


def add_lfp_info(recording, config):
    ss = NWBSignalSeries(recording)
    ss.filter(config["fmin"], config["fmax"], **config["filter_kwargs"])
    type_ = "clean_kwargs"
    results_all, results_picked = process_lfp(ss, config, type_)

    nwbfile = recording.data
    # nwb_proc = nwbfile.copy()
    nwb_proc = nwbfile
    did_anything = [store_normalised_lfp(ss, results_all, nwb_proc)]
    did_anything.append(store_average_lfp(results_picked, results_all, nwb_proc))
    did_anything.append(calculate_and_store_lfp_power(config, nwb_proc))
    did_anything.append(
        store_coherence(nwb_proc, flims=(config["fmin"], config["fmax"]))
    )
    for d in did_anything:
        if d is not False:
            return nwb_proc, True

    return nwb_proc, False


def calculate_and_store_lfp_power(config, nwb_proc):
    if "lfp_power" in nwb_proc.processing:
        return False
    signals = nwb_proc.processing["normalised_lfp"]["LFP"]["ElectricalSeries"].data[:].T
    brain_regions = sorted(list(set(nwb_proc.electrodes.to_dataframe()["location"])))
    br_avg = [f"{br}_avg".replace("/", "_") for br in brain_regions]
    average_signals = np.array(
        [nwb_proc.processing["average_lfp"][br].data[:] for br in br_avg]
    )
    all_sigs = np.concatenate((signals, average_signals), axis=0)
    regions = list(nwb_proc.electrodes.to_dataframe()["location"])
    regions.extend(brain_regions)
    labels = list(nwb_proc.electrodes.to_dataframe()["label"])
    labels.extend(br_avg)
    results_list = []
    for sig, region, label in zip(all_sigs, regions, labels):
        warn = bool(label.endswith("_avg"))
        f, Pxx, max_psd = calculate_psd(
            sig,
            scale="decibels",
            fmin=config["fmin"],
            fmax=config["fmax"],
            warn=warn,
        )
        results_list.append([label, region, f, Pxx, max_psd])
    results_df = list_to_df(
        results_list, headers=["label", "region", "frequency", "power", "max_psd"]
    )
    results_df.index.name = "Index"
    hdmf_table = DynamicTable.from_dataframe(
        df=results_df, name="power_spectra", columns=describe_columns()
    )
    mod = nwb_proc.create_processing_module(
        "lfp_power", "Store power spectra and spectograms"
    )
    mod.add(hdmf_table)


def store_average_lfp(results_picked, results_all, nwb_proc):
    if "average_lfp" in nwb_proc.processing:
        return False
    mod = nwb_proc.create_processing_module(
        "average_lfp", "A single averaged LFP signal per brain region"
    )

    for brain_region, result in results_picked.items():
        if np.sum(np.abs(result["average_signal"])) < 0.1:
            module_logger.warning(
                f"Average signal from first channels is none for brain region {brain_region}"
            )
            signal = results_all[brain_region]["average_signal"]
        else:
            signal = result["average_signal"]
        ts = TimeSeries(
            name=f"{brain_region}_avg".replace("/", "_"),
            data=0.001 * signal,
            unit="V",
            conversion=1.0,
            rate=250.0,
            starting_time=0.0,
            description="A single averaged normalised LFP signal per brain region",
        )
        mod.add(ts)


def store_normalised_lfp(ss, results_all, nwb_proc):
    if "normalised_lfp" in nwb_proc.processing:
        return False
    mod = nwb_proc.create_processing_module(
        "normalised_lfp",
        "Store filtered and z-score normalised LFP, with outlier information",
    )
    lfp_array = np.zeros_like(ss.data)
    electrode_type = np.zeros(shape=(lfp_array.shape[0]), dtype=object)
    region_to_idx_dict = ss.group_by_brain_region(index=True)

    for brain_region, result in results_all.items():
        indices = region_to_idx_dict[brain_region]
        signals = result["signals"]
        good_idx = result["good_idx"]
        outliers = result["outliers"]
        outliers_idx = result["outliers_idx"]
        for sig, idx in zip(signals, good_idx):
            lfp_array[indices[idx]] = sig
            electrode_type[indices[idx]] = "Normal"
        for sig, idx in zip(outliers, outliers_idx):
            lfp_array[indices[idx]] = sig
            electrode_type[indices[idx]] = "Outlier"

    nwb_proc.add_electrode_column(
        name="clean",
        description="The LFP signal matches others from this brain region or is an outlier",
        data=list(electrode_type),
    )
    add_lfp_array_to_nwb(
        nwb_proc, len(ss.data), 0.001 * lfp_array.T, mod, conversion=1.0
    )


def store_coherence(nwb_proc, flims=None):
    if "lfp_coherence" in nwb_proc.processing:
        return False
    average_signals = nwb_proc.processing["average_lfp"]
    fields = average_signals.data_interfaces.keys()
    if len(fields) < 2:
        return False
    coherence_list = []
    for fd in sorted(itertools.combinations(fields, 2)):
        x = average_signals[fd[0]].data[:]
        y = average_signals[fd[1]].data[:]
        fs = average_signals[fd[0]].rate
        f, Cxy = scipy.signal.coherence(x, y, fs, nperseg=2 * fs)

        if flims is not None:
            fmin, fmax = flims
            f = f[np.nonzero((f >= fmin) & (f <= fmax))]
            Cxy = Cxy[np.nonzero((f >= fmin) & (f <= fmax))]

        key = f"{fd[0][:-4]}_{fd[1][:-4]}"
        coherence_list.append([key, f, Cxy])

    headers = ["label", "frequency", "coherence"]
    results_df = list_to_df(coherence_list, headers=headers)
    hdmf_table = DynamicTable.from_dataframe(
        df=results_df, name="coherence_table", columns=describe_coherence_columns()
    )
    mod = nwb_proc.create_processing_module("lfp_coherence", "Store coherence")
    mod.add(hdmf_table)


def main(table_path, config_path, output_path, num_cpus, overwrite=False):
    config = smr.ParamHandler(source_file=config_path)
    config["num_cpus"] = num_cpus

    if isinstance(table_path, pd.DataFrame):
        datatable = table_path
    else:
        datatable = df_from_file(table_path)
    loader = NWBLoader(mode="a") if overwrite else NWBLoader(mode="r")
    rc = smr.RecordingContainer.from_table(datatable, loader)
    out_df = datatable.copy()

    could_not_process = []
    for i in range(len(rc)):
        fname = Path(rc[i].source_file)
        fname = fname.parent.parent / "processed" / fname.name
        if not fname.is_file() or overwrite:
            module_logger.debug(f"Processing {rc[i].source_file}")
            r = rc.load(i)
            try:
                nwbfile, _ = add_lfp_info(r, config)
                export_nwbfile(fname, r, nwbfile, r._nwb_io, debug=True)
                if r.attrs["has_video"]:
                    source_file = str(fname[:-3]) + "avi"
                    dest_file = (
                        str(fname.parent.parent / "processed" / fname.name)[:-3] + "avi"
                    )
                    if not Path(dest_file).is_file() or overwrite:
                        shutil.copyfile(source_file, dest_file)
            except Exception as e:
                module_logger.error(f"Failed to process {rc[i].source_file}")
                module_logger.error(e)
                could_not_process.append(i)
        else:
            module_logger.debug(f"Already processed {rc[i].source_file}")
        row_idx = datatable.index[i]
        out_df.at[row_idx, "nwb_file"] = str(fname)
    out_df = out_df.drop(out_df.index[could_not_process])
    df_to_file(out_df, output_path)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    module_logger.setLevel(logging.DEBUG)
    main(
        snakemake.input[0],
        snakemake.config["simuran_config"],
        snakemake.output[0],
        snakemake.threads,
    )
