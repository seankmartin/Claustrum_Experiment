import logging
from pathlib import Path
import datetime

from skm_pyutils.table import df_from_file, df_to_file
import simuran as smr
from simuran.loaders.nwb_loader import NWBLoader
import pandas as pd
from pynwb.file import Subject
from pynwb import NWBFile

from convert_to_nwb import export_nwbfile

module_logger = logging.getLogger("simuran.custom.process_lfp")


def fix_nwbfile(recording, axona_table, config):
    nwbfile = recording.data
    print(nwbfile)
    name = recording.get_name_for_save(rel_dir=Path("results") / "processed")
    id_ = nwbfile.identifier
    parts = id_.split("--")
    part = parts[-1] + ".set"
    matching_row = axona_table[axona_table["filename"] == part]
    date = matching_row["date"].values[0]
    time = matching_row["time"].values[0]
    session_start_time = datetime.datetime.strptime(
        f"{date} {time}", "%A, %d %b %Y %H:%M:%S"
    ).replace(tzinfo=datetime.timezone.utc)
    new_nwbfile = NWBFile(
        session_description=f"Recording {name}",
        identifier=f"CLA--{name}",
        session_start_time=session_start_time,
        experiment_description="CLA, RSC, and related areas during Operant box task",
        experimenter="Gao Xiang Ham",
        lab="O'Mara lab",
        institution="TCD",
    )
    new_subject = Subject(
        species="Rattus norvegicus domestica",
        strain="Lister Hooded",
        age="P10D/P20D",
        sex="M",
        subject_id=recording.attrs["rat_id"],
        weight=0.330,
    )
    new_nwbfile.subject = new_subject
    for device in nwbfile.devices:
        new_nwbfile.add_device(device)
    for electrode in nwbfile.electrodes:
        new_nwbfile.add_electrode(electrode)
    
    return new_nwbfile


def main(table_path, axona_path, config_path, output_path, num_cpus=1, overwrite=False):
    config = smr.ParamHandler(source_file=config_path)
    config["num_cpus"] = num_cpus

    if isinstance(table_path, pd.DataFrame):
        datatable = table_path
    else:
        datatable = df_from_file(table_path)
    axona_df = df_from_file(axona_path)
    loader = NWBLoader(mode="a") if overwrite else NWBLoader(mode="r")
    rc = smr.RecordingContainer.from_table(datatable, loader)
    out_df = datatable.copy()

    could_not_process = []
    for i in range(len(rc)):
        fname = Path(rc[i].source_file)
        fname = fname.parent.parent / "final" / fname.name
        if not fname.is_file() or overwrite:
            module_logger.debug(f"Processing {rc[i].source_file}")
            r = rc.load(i)
            nwbfile = fix_nwbfile(r, axona_df, config)
            print(nwbfile)
            exit(-1)
            try:
                nwbfile = fix_nwbfile(r, config)

                export_nwbfile(fname, r, nwbfile, r._nwb_io, debug=True)
            except Exception as e:
                module_logger.error(f"Failed to process {rc[i].source_file}")
                module_logger.error(e)
                could_not_process.append(i)
            break
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
        snakemake.input[1],
        snakemake.config["simuran_config"],
        snakemake.output[0],
        snakemake.threads,
    )
